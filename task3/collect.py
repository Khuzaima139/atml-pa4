"""Off-policy teacher collection (protocol D8): one frozen collection shared by both training seeds.

One response per mathematics training prompt (K=1) and 16 per short-form training prompt (K=16), sampled from the
teacher with the shared D3 law at caps 192 / 32; response k of prompt i uses the rollout seed derived from
(6304, prompt ID, k). Output: gzip JSONL through common.io, one record per response (source ID, category, k, rollout
seed, prompt token count, response token IDs, termination reason, length, generation seconds), plus a RunRecorder
JSON next to it (same stem, .json) with the teacher generation time and the held-out guard record.

    python -m task3.collect --out results/task3/collection/teacher_collection.jsonl.gz
"""
import argparse
import time
from collections import Counter
from pathlib import Path

from common.io import sha256_gz_content, write_jsonl_gz
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import data as D
from . import distill as T
from .heldout import HeldoutGuard
from .lm import HFCachedLM
from .sampling import EOS_IDS, GENERATION_SETTINGS, SAMPLER_ID, VOCAB_SIZE, make_generator
from .spec import target_only_generate

DONE = "collect done: "


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output .jsonl.gz (must not exist)")
    ap.add_argument("--teacher-path", default="model_target")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--limit-prompts-math", type=int, default=None, help="tests only; refused under results/")
    ap.add_argument("--limit-prompts-short", type=int, default=None, help="tests only; refused under results/")
    ap.add_argument("--limit-k-short", type=int, default=None, help="tests only; refused under results/")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if not out.name.endswith(".jsonl.gz"):
        raise SystemExit("--out must end with .jsonl.gz")
    run_json = out.with_name(out.name[: -len(".jsonl.gz")] + ".json")
    for p in (out, run_json):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    limits = {"limit_prompts_math": a.limit_prompts_math, "limit_prompts_short": a.limit_prompts_short,
              "limit_k_short": a.limit_k_short}
    limited = any(v is not None for v in limits.values())
    if limited and T.under_results(out):
        raise SystemExit("--limit-* options are refused under results/")

    seed_everything(T.COLLECTION_SEED)
    guard = HeldoutGuard()
    items, prov = D.training_items(a.limit_prompts_math, a.limit_prompts_short)
    guard.check(items, "training prompt")
    k_per = {"math": T.K["math"], "short": a.limit_k_short or T.K["short"]}
    config = {
        "collection_id": T.COLLECTION_ID, "distill_id": T.DISTILL_ID, "sampler_id": SAMPLER_ID,
        "collection_seed": T.COLLECTION_SEED, "k": k_per, "caps": T.CAPS, "eos_ids": list(EOS_IDS),
        "seed_rule": "sha256('t3-collect|6304|<prompt_id>|<k>') first 15 hex digits",
        "n_prompts": {c: sum(1 for x in items if x["category"] == c) for c in T.CATEGORIES},
        "prompt_ids": [x["prompt_id"] for x in items], "limits": limits, "limited": limited,
        "teacher_path": a.teacher_path, "device": a.device, "dtype": a.dtype, "data_provenance": prov,
        "generation_settings": GENERATION_SETTINGS, "out": str(out),
        "prompt_format": "chat template, single user message, add_generation_prompt=True, no explicit system message",
    }
    rec = RunRecorder(run_json, config)
    t0 = time.perf_counter()
    tok = D.load_tokenizer(a.teacher_path)
    model, prov_m = D.load_model(a.teacher_path, a.device, a.dtype)
    if prov_m["vocab_size"] != VOCAB_SIZE:
        raise SystemExit(f"vocabulary {prov_m['vocab_size']} != {VOCAB_SIZE}")
    lm = HFCachedLM(model, expect_vocab=VOCAB_SIZE)
    load_s = time.perf_counter() - t0
    rec.update(model_revisions={"teacher": prov_m}, timing={"load_s": load_s})

    records, terms, gen_s = [], Counter(), {c: 0.0 for c in T.CATEGORIES}
    tokens = {c: 0 for c in T.CATEGORIES}
    prefill, steps = [], []
    for i, it in enumerate(items):
        cat = it["category"]
        ids = D.chat_ids(tok, it["prompt"])
        for k in range(k_per[cat]):
            seed = T.collection_seed(it["prompt_id"], k)
            res = target_only_generate(lm, ids, T.CAPS[cat], EOS_IDS, make_generator(seed, a.device))
            seq = res["output_ids"]
            records.append({"source_id": it["prompt_id"], "category": cat, "k": k, "rollout_seed": seed,
                            "prompt_tokens": len(ids), "response_token_ids": seq, "termination": res["termination"],
                            "length": len(seq), "generation_s": res["total_s"], "prefill_s": res["prefill_s"],
                            "collection_id": T.COLLECTION_ID, "sampler_id": SAMPLER_ID})
            terms[f"{cat}_{res['termination']}"] += 1
            gen_s[cat] += res["total_s"]
            tokens[cat] += len(seq)
            prefill.append(res["prefill_s"])
            steps.extend(res["step_forward_s"])
        rec.update(progress={"prompts_done": i + 1, "responses": len(records), "terminations": dict(terms)},
                   timing={"load_s": load_s, "teacher_generation_s": gen_s, "teacher_generation_s_total":
                           sum(gen_s.values())})
    write_jsonl_gz(out, records)
    steps_sorted = sorted(steps)
    timing = {"load_s": load_s, "teacher_generation_s": gen_s, "teacher_generation_s_total": sum(gen_s.values()),
              "mean_prefill_s": (sum(prefill) / len(prefill)) if prefill else None,
              "median_step_s": steps_sorted[len(steps_sorted) // 2] if steps_sorted else None,
              "n_decode_steps": len(steps)}
    rec.finish("ok", n_responses=len(records), tokens=tokens, terminations=dict(terms), timing=timing,
               heldout_guard=guard.report(), output_content_sha256=sha256_gz_content(out))
    print(f"{DONE}{len(records)} responses, tokens {tokens}, teacher generation {sum(gen_s.values()):.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
