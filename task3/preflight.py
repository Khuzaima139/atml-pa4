"""Real-model preflight of the cached forward path used by the Task 3 samplers (run on the GPU before timing).

Fixed inputs: the first 2 GSM8K and the first 2 short-form training prompts. A trajectory step (--trajectory-dtype,
both models loaded) extends each prompt by a sampled continuation of at most --cap tokens per model (that model,
shared D3 law, seed 6304 + prompt index) and runs one exact speculative generation per prompt (target with the
student, gamma --gamma, seed 7304 + prompt index), logging every cache operation (reset, forward, crop) of each
model and the context at the end of every round. Then, for each precision in --dtypes, each model is loaded alone,
checked and freed:
(a) split prefill (prompt[:-1] in one call, then one token per call) versus one full forward;
(b) prefill then chunks of gamma + 1 tokens (verification windows) versus the full forward;
(c) full sequence into the cache, crop to several lengths and re-forward the rest, versus the full forward;
(d) the logged speculative cache operations replayed on the model: after every round, the next-token logits from
    the cache versus a fresh full forward of the emitted sequence, and the cache length (target = context - 1,
    draft <= context - 1).
Each comparison records max and mean abs logit difference, max and mean total-variation distance of the fp32
softmaxes and argmax agreement. Pass rules (protocol AM3): FP32 max abs diff <= 0.05 and max TV <= 0.001 on every
comparison; FP16 max TV <= 0.2 and argmax agreement >= 0.9 on every comparison; both with the cache-length check.
all_pass is true only when both the FP32 and the FP16 rule pass.

    python -m task3.preflight --out results/task3/preflight/<name>.json
"""
import argparse
import gc
from pathlib import Path

import torch

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import data as D
from .lm import HFCachedLM, hf_full_logits
from .sampling import EOS_IDS, GENERATION_SETTINGS, SAMPLER_ID, VOCAB_SIZE, make_generator, probs_from_logits
from .spec import cache_next_logits, speculative_generate, target_only_generate

PREFLIGHT_ID = "t3-preflight-v3"
MODELS = ("target", "draft")
CHECKS = ("split_prefill", "verify_windows", "crop_reforward", "speculative_cache")
REQUIRED_DTYPES = ("float32", "float16")
RULES = {
    "float32": {"max_abs_diff_le": 0.05, "max_tv_le": 0.001},
    "float16": {"max_tv_le": 0.2, "argmax_agreement_ge": 0.9},
}


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    d = (a - b).abs()
    tv = 0.5 * (probs_from_logits(a) - probs_from_logits(b)).abs().sum(-1)
    return {"max_abs_diff": float(d.max().item()), "mean_abs_diff": float(d.mean().item()),
            "max_tv": float(tv.max().item()), "mean_tv": float(tv.mean().item()),
            "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean().item()), "rows": int(a.shape[0])}


def merge(stats) -> dict:
    """Pool comparisons: maxima of maxima, row-weighted means (every row has the same vocabulary width)."""
    stats = [s for s in stats if s]
    if not stats:
        return {}
    rows = sum(s["rows"] for s in stats)
    return {"max_abs_diff": max(s["max_abs_diff"] for s in stats),
            "mean_abs_diff": sum(s["mean_abs_diff"] * s["rows"] for s in stats) / rows,
            "max_tv": max(s["max_tv"] for s in stats),
            "mean_tv": sum(s["mean_tv"] * s["rows"] for s in stats) / rows,
            "argmax_agreement": sum(s["argmax_agreement"] * s["rows"] for s in stats) / rows, "rows": rows}


def rule_pass(s: dict, dtype: str) -> bool:
    """AM3 rule for one pooled comparison; a precision without a rule never passes."""
    r = RULES.get(dtype)
    if r is None or not s or s.get("rows", 0) <= 0:
        return False
    ok = True
    if "max_abs_diff_le" in r:
        ok &= s["max_abs_diff"] <= r["max_abs_diff_le"]
    if "max_tv_le" in r:
        ok &= s["max_tv"] <= r["max_tv_le"]
    if "argmax_agreement_ge" in r:
        ok &= s["argmax_agreement"] >= r["argmax_agreement_ge"]
    return bool(ok)


def evaluate_record(per_dtype: dict) -> dict:
    """Pass flags per precision and all_pass (true only when every required precision is present and passes)."""
    precision_pass = {}
    for dt, block in per_dtype.items():
        passes = {f"{m}_{c}": rule_pass(block["summary"].get(m, {}).get(c, {}), dt) for m in MODELS for c in CHECKS}
        passes["speculative_cache_lengths"] = bool(block["cache_lengths"]["pass"])
        block["passes"] = passes
        block["pass"] = bool(all(passes.values()))
        precision_pass[dt] = block["pass"]
    all_pass = bool(all(precision_pass.get(dt) is True for dt in REQUIRED_DTYPES))
    return {"precision_pass": precision_pass, "all_pass": all_pass}


class RecordingLM:
    """Wraps a cached LM and logs every cache operation in order; round markers are added by the caller."""

    def __init__(self, lm, log):
        self.lm, self.log, self.device = lm, log, lm.device

    @property
    def length(self):
        return self.lm.length

    def reset(self):
        self.log.append(["reset"])
        self.lm.reset()

    def forward(self, ids):
        self.log.append(["forward", [int(t) for t in ids]])
        return self.lm.forward(ids)

    def crop(self, n):
        self.log.append(["crop", int(n)])
        self.lm.crop(n)


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


def replay(model, name, ops, prompt_id):
    """Replay a logged speculative cache-operation sequence; compare at every round marker."""
    lm = HFCachedLM(model, expect_vocab=VOCAB_SIZE)
    stats, violations, n = [], [], 0
    with torch.inference_mode():
        for op in ops:
            if op[0] == "reset":
                lm.reset()
            elif op[0] == "forward":
                lm.forward(op[1])
            elif op[0] == "crop":
                lm.crop(op[1])
            elif op[0] == "round":
                n += 1
                rnd, ctx = op[1], op[2]
                bad = lm.length != len(ctx) - 1 if name == "target" else lm.length > len(ctx) - 1
                if bad:
                    violations.append({"prompt_id": prompt_id, "model": name, "round": rnd, "ctx": len(ctx),
                                       "cache_length": lm.length})
                if lm.length < len(ctx):
                    fresh = hf_full_logits(model, ctx)[-1:]
                    stats.append(compare(cache_next_logits(lm, ctx)[None, :], fresh))
            else:
                raise ValueError(f"unknown cache operation {op[0]!r}")
    return stats, violations, n


def _free(model, device):
    del model
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _peak_reset(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def trajectories(a, prompts, tok):
    """Continuations and logged speculative runs, both models loaded in --trajectory-dtype."""
    _peak_reset(a.device)
    models = {"target": D.load_model(a.target_path, a.device, a.trajectory_dtype),
              "draft": D.load_model(a.draft_path, a.device, a.trajectory_dtype)}
    revisions = {k: pv for k, (_, pv) in models.items()}
    rows = []
    for i, pr in enumerate(prompts):
        ids = D.chat_ids(tok, pr["prompt"])
        row = {"prompt_id": pr["prompt_id"], "prompt_token_ids": ids, "continuations": {}, "ops": {}}
        for name, (model, _) in models.items():
            gen = make_generator(6304 + i, a.device)
            row["continuations"][name] = target_only_generate(HFCachedLM(model, VOCAB_SIZE), ids, a.cap, EOS_IDS,
                                                              gen)["output_ids"]
        logs = {"target": [], "draft": []}
        t_lm = RecordingLM(HFCachedLM(models["target"][0], VOCAB_SIZE), logs["target"])
        d_lm = RecordingLM(HFCachedLM(models["draft"][0], VOCAB_SIZE), logs["draft"])

        def mark(rec, ctx, _out):
            for log in logs.values():
                log.append(["round", int(rec["round"]), [int(t) for t in ctx]])

        res = speculative_generate(t_lm, d_lm, ids, a.gamma, a.cap, EOS_IDS, make_generator(7304 + i, a.device),
                                   round_callback=mark)
        row["ops"] = logs
        row["speculative"] = {"rounds": res["n_rounds"], "output_tokens": res["output_tokens"],
                              "accepted": res["accepted"], "examined": res["examined"]}
        rows.append(row)
    peak = _peak(a.device)
    for name in list(models):
        model = models.pop(name)[0]
        _free(model, a.device)
    return rows, revisions, peak


def run_precision(a, dtype, traj):
    agg = {m: {c: [] for c in CHECKS} for m in MODELS}
    per_prompt = [{"prompt_id": r["prompt_id"], "prompt_tokens": len(r["prompt_token_ids"])} for r in traj]
    violations, rounds, peaks, revisions = [], {m: 0 for m in MODELS}, {}, {}
    for name, path in (("target", a.target_path), ("draft", a.draft_path)):
        _peak_reset(a.device)
        model, prov = D.load_model(path, a.device, dtype)
        revisions[name] = prov
        for row, out in zip(traj, per_prompt):
            ids, cont = row["prompt_token_ids"], row["continuations"][name]
            split, windows, crops = check_model(model, ids + cont, len(ids), a.gamma)
            spec_stats, viol, n = replay(model, name, row["ops"][name], row["prompt_id"])
            violations += viol
            rounds[name] += n
            out[name] = {"continuation_tokens": len(cont), "split_prefill": split, "verify_windows": windows,
                         "crop_reforward": crops, "speculative_cache": merge(spec_stats), "rounds_checked": n}
            for c, s in (("split_prefill", [split]), ("verify_windows", [windows]), ("crop_reforward", [crops]),
                         ("speculative_cache", spec_stats)):
                agg[name][c].extend(s)
        peaks[name] = _peak(a.device)
        _free(model, a.device)
        model = None
    summary = {m: {c: merge(v) for c, v in checks.items()} for m, checks in agg.items()}
    cache_lengths = {"rounds_checked": rounds, "violations": violations,
                     "pass": bool(all(n > 0 for n in rounds.values()) and not violations)}
    return {"dtype": dtype, "rule": RULES.get(dtype), "summary": summary, "cache_lengths": cache_lengths,
            "per_prompt": per_prompt, "peak_vram_allocated_bytes": peaks, "model_revisions": revisions}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-path", default="model_target")
    ap.add_argument("--draft-path", default="model_draft")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtypes", default=",".join(REQUIRED_DTYPES), help="comma-separated precisions to check")
    ap.add_argument("--trajectory-dtype", default="float16", help="precision of the logged trajectories")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--gamma", type=int, default=4)
    a = ap.parse_args(argv)
    dtypes = [x.strip() for x in a.dtypes.split(",") if x.strip()]
    if not dtypes or len(set(dtypes)) != len(dtypes):
        raise SystemExit(f"--dtypes needs distinct precisions, got {a.dtypes!r}")
    for dt in dtypes + [a.trajectory_dtype]:
        if not isinstance(getattr(torch, dt, None), torch.dtype):
            raise SystemExit(f"unknown dtype {dt!r}")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    seed_everything(6304)
    prompts, prov = D.training_prompts_head()
    tok = D.load_tokenizer(a.target_path)
    config = {"preflight_id": PREFLIGHT_ID, "sampler_id": SAMPLER_ID, "cap": a.cap, "gamma": a.gamma,
              "device": a.device, "dtypes": dtypes, "trajectory_dtype": a.trajectory_dtype,
              "required_dtypes": list(REQUIRED_DTYPES), "rules": RULES, "checks": list(CHECKS),
              "target_path": a.target_path, "draft_path": a.draft_path,
              "prompt_ids": [p["prompt_id"] for p in prompts], "data_provenance": prov,
              "continuation_seed_rule": "6304 + prompt index", "speculative_seed_rule": "7304 + prompt index",
              "model_loading": "trajectory step loads both models in trajectory_dtype; each check precision loads "
                               "one model at a time and frees it", "generation_settings": GENERATION_SETTINGS}
    rec = RunRecorder(out, config)
    traj, revisions, traj_peak = trajectories(a, prompts, tok)
    rec.update(model_revisions=revisions, trajectory_peak_vram_allocated_bytes=traj_peak,
               trajectories=[{"prompt_id": r["prompt_id"], "prompt_token_ids": r["prompt_token_ids"],
                              "continuations": r["continuations"], "speculative": r["speculative"],
                              "ops": r["ops"]} for r in traj])
    per_dtype = {}
    for dt in dtypes:
        per_dtype[dt] = run_precision(a, dt, traj)
        rec.update(precisions=per_dtype)
    verdict = evaluate_record(per_dtype)
    rec.finish("ok", precisions=per_dtype, **verdict)
    for dt, block in per_dtype.items():
        for m, checks in block["summary"].items():
            for c, s in checks.items():
                print(f"{dt:8s} {m:6s} {c:18s} max|d|={s['max_abs_diff']:.3g} mean|d|={s['mean_abs_diff']:.3g} "
                      f"maxTV={s['max_tv']:.3g} meanTV={s['mean_tv']:.3g} argmax={s['argmax_agreement']:.4f} "
                      f"pass={block['passes'][f'{m}_{c}']}")
        print(f"{dt:8s} cache_lengths rounds={block['cache_lengths']['rounds_checked']} "
              f"violations={len(block['cache_lengths']['violations'])} pass={block['cache_lengths']['pass']}")
        print(f"{dt:8s} pass={block['pass']}")
    print(f"precision_pass={verdict['precision_pass']} all_pass={verdict['all_pass']}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
