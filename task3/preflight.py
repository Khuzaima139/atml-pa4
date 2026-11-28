"""Real-model preflight of the cached forward path used by the Task 3 samplers (run on the GPU before timing).

For the target and the original student, on the first 2 GSM8K and the first 2 short-form training prompts, each
extended by a sampled continuation of at most --cap tokens (that model, shared D3 law, seed 6304 + prompt index):
(a) split prefill (prompt[:-1] in one call, then one token per call) versus one full forward;
(b) prefill then chunks of gamma + 1 tokens (verification windows) versus the full forward;
(c) full sequence into the cache, crop to several lengths and re-forward the rest, versus the full forward;
(d) one exact speculative generation per prompt (target with the student, gamma --gamma): after every round, the
    next-token logits from each cache versus a fresh full forward of the emitted sequence, and the cache lengths
    (target = context - 1, draft <= context - 1) after every round (protocol AM1).
Each check reports the max abs logit difference, the max total-variation distance of the fp32 softmaxes and the
argmax agreement; it passes when max abs diff <= LOGIT_TOL and max TV <= TV_TOL.

    python -m task3.preflight --out results/task3/preflight_<name>.json
"""
import argparse
from pathlib import Path

import torch

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import data as D
from .lm import HFCachedLM, hf_full_logits
from .sampling import EOS_IDS, GENERATION_SETTINGS, SAMPLER_ID, VOCAB_SIZE, make_generator, probs_from_logits
from .spec import cache_next_logits, speculative_generate, target_only_generate

PREFLIGHT_ID = "t3-preflight-v1"
LOGIT_TOL = 0.25
TV_TOL = 0.01


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    tv = 0.5 * (probs_from_logits(a) - probs_from_logits(b)).abs().sum(-1)
    return {"max_abs_diff": float((a - b).abs().max().item()), "max_tv": float(tv.max().item()),
            "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean().item()), "rows": int(a.shape[0])}


def merge(stats) -> dict:
    stats = [s for s in stats if s]
    if not stats:
        return {}
    rows = sum(s["rows"] for s in stats)
    out = {"max_abs_diff": max(s["max_abs_diff"] for s in stats), "max_tv": max(s["max_tv"] for s in stats),
           "argmax_agreement": sum(s["argmax_agreement"] * s["rows"] for s in stats) / rows, "rows": rows}
    out["pass"] = bool(out["max_abs_diff"] <= LOGIT_TOL and out["max_tv"] <= TV_TOL)
    return out


def check_model(model, seq, n_prompt, gamma):
    lm = HFCachedLM(model, expect_vocab=VOCAB_SIZE)
    with torch.inference_mode():
        full = hf_full_logits(model, seq)
        L = len(seq)
        # (a) split prefill, then one token per call
        lm.reset()
        rows = [lm.forward(seq[: n_prompt - 1])]
        for t in range(n_prompt - 1, L):
            rows.append(lm.forward([seq[t]]))
        split = compare(torch.cat(rows, 0), full)
        # (b) verification windows of gamma + 1 tokens
        lm.reset()
        rows = [lm.forward(seq[: n_prompt - 1])]
        t = n_prompt - 1
        while t < L:
            rows.append(lm.forward(seq[t: t + gamma + 1]))
            t += gamma + 1
        windows = compare(torch.cat(rows, 0), full)
        # (c) crop and re-forward
        lm.reset()
        lm.forward(seq)
        crops = []
        for cut in sorted({n_prompt, max(n_prompt, L - gamma - 1), L - 2, L - 1}):
            if not 0 < cut < L:
                continue
            lm.crop(cut)
            crops.append(compare(lm.forward(seq[cut:]), full[cut:]))
    return split, windows, merge(crops)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-path", default="model_target")
    ap.add_argument("--draft-path", default="model_draft")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--gamma", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    seed_everything(6304)
    prompts, prov = D.training_prompts_head()
    tok = D.load_tokenizer(a.target_path)
    config = {"preflight_id": PREFLIGHT_ID, "sampler_id": SAMPLER_ID, "cap": a.cap, "gamma": a.gamma,
              "device": a.device, "dtype": a.dtype, "target_path": a.target_path, "draft_path": a.draft_path,
              "prompt_ids": [p["prompt_id"] for p in prompts], "data_provenance": prov,
              "logit_tol": LOGIT_TOL, "tv_tol": TV_TOL, "continuation_seed_rule": "6304 + prompt index",
              "generation_settings": GENERATION_SETTINGS}
    rec = RunRecorder(out, config)
    models = {"target": D.load_model(a.target_path, a.device, a.dtype),
              "draft": D.load_model(a.draft_path, a.device, a.dtype)}
    rec.update(model_revisions={k: pv for k, (_, pv) in models.items()})

    n_rounds, length_violations = [0], []
    per_prompt, agg = [], {name: {"split_prefill": [], "verify_windows": [], "crop_reforward": [],
                                  "speculative_cache": []} for name in models}
    for i, pr in enumerate(prompts):
        ids = D.chat_ids(tok, pr["prompt"])
        row = {"prompt_id": pr["prompt_id"], "prompt_tokens": len(ids)}
        for name, (model, _) in models.items():
            gen = make_generator(6304 + i, a.device)
            cont = target_only_generate(HFCachedLM(model, VOCAB_SIZE), ids, a.cap, EOS_IDS, gen)["output_ids"]
            split, windows, crops = check_model(model, ids + cont, len(ids), a.gamma)
            row[name] = {"continuation_tokens": len(cont), "split_prefill": split, "verify_windows": windows,
                         "crop_reforward": crops}
            agg[name]["split_prefill"].append(split)
            agg[name]["verify_windows"].append(windows)
            agg[name]["crop_reforward"].append(crops)
        t_lm = HFCachedLM(models["target"][0], VOCAB_SIZE)
        d_lm = HFCachedLM(models["draft"][0], VOCAB_SIZE)
        spec_stats = {"target": [], "draft": []}

        def probe(_rec, ctx, _out):
            n_rounds[0] += 1
            if t_lm.length != len(ctx) - 1 or d_lm.length > len(ctx) - 1:
                length_violations.append({"prompt_id": pr["prompt_id"], "round": _rec["round"], "ctx": len(ctx),
                                          "target": t_lm.length, "draft": d_lm.length})
            for name, lm in (("target", t_lm), ("draft", d_lm)):
                with torch.inference_mode():
                    fresh = hf_full_logits(models[name][0], ctx)[-1:]
                spec_stats[name].append(compare(cache_next_logits(lm, ctx)[None, :], fresh))

        res = speculative_generate(t_lm, d_lm, ids, a.gamma, a.cap, EOS_IDS, make_generator(7304 + i, a.device),
                                   round_callback=probe)
        row["speculative"] = {"rounds": res["n_rounds"], "output_tokens": res["output_tokens"],
                              "accepted": res["accepted"], "examined": res["examined"],
                              "target": merge(spec_stats["target"]), "draft": merge(spec_stats["draft"])}
        for name in models:
            agg[name]["speculative_cache"].extend(spec_stats[name])
        per_prompt.append(row)
        rec.update(per_prompt=per_prompt)
    summary = {name: {check: merge(v) for check, v in checks.items()} for name, checks in agg.items()}
    passes = {f"{name}_{check}": s["pass"] for name, checks in summary.items() for check, s in checks.items()}
    cache_lengths = {"rounds_checked": n_rounds[0], "violations": length_violations,
                     "pass": bool(n_rounds[0] > 0 and not length_violations)}
    passes["speculative_cache_lengths"] = cache_lengths["pass"]
    all_pass = bool(passes) and all(passes.values())
    rec.finish("ok", summary=summary, cache_lengths=cache_lengths, passes=passes, all_pass=all_pass)
    for name, checks in summary.items():
        for check, s in checks.items():
            print(f"{name:6s} {check:18s} max|d|={s['max_abs_diff']:.3g} maxTV={s['max_tv']:.3g} "
                  f"argmax={s['argmax_agreement']:.4f} pass={s['pass']}")
    print(f"all_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
