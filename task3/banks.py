"""Prefix-bank generator (protocol D1, D2, D14).

Banks: teacher (target model), original_draft (original student), exact_spec_original_gamma4 (exact speculative
sampler, target with the original student at gamma 4) and own (a trained run's merged final draft). One generation
per held-out prompt with at most 192 new tokens, sampled with the shared D3 law; generation seed
6304 + 10007*j + offset for prompt index j; positions drawn as in the release script; position p stores the prefix
seq[:p-1] and the token seq[p-1] sampled there.

Output (gzip JSONL through common.io): per prompt one record_type "sequence" record (full response once), then
one record_type "position" record per saved position with the release fields. A RunRecorder JSON is written next
to it (same stem, .json).

    python -m task3.banks --kind teacher --out results/task3/banks/teacher.jsonl.gz
    python -m task3.banks --kind own --draft-path <merged dir> --run-name off_policy_6304 --out ...
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common.freeze import require_committed
from common.io import write_jsonl_gz
from common.paths import RESULTS_DIR
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import data as D
from .lm import HFCachedLM
from .sampling import EOS_IDS, GENERATION_SETTINGS, SAMPLER_ID, VOCAB_SIZE, make_generator
from .spec import speculative_generate, target_only_generate

BANK_ID = "t3-banks-v1"
MAX_NEW_TOKENS = 192
SEED_BASE = 6304
SEED_STRIDE = 10007
BINS = ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16))
SPEC_GAMMA = 4
OWN_RUNS = {"off_policy_6304": 1, "off_policy_6305": 2, "on_policy_6304": 3, "on_policy_6305": 4}
KINDS = ("teacher", "original_draft", "exact_spec_original_gamma4", "own")
SAMPLER_CHECK = RESULTS_DIR / "task3" / "sampler_check.json"


def seed_offset(kind, run_name=None) -> int:
    if kind == "teacher":
        return 0
    if kind == "original_draft":
        return 123
    if kind == "exact_spec_original_gamma4":
        return 456
    if kind == "own":
        return 1000 + OWN_RUNS[run_name]
    raise ValueError(kind)


def bank_name(kind, run_name=None) -> str:
    return f"own_{run_name}" if kind == "own" else kind


def generation_seed(j, kind, run_name=None) -> int:
    return SEED_BASE + SEED_STRIDE * j + seed_offset(kind, run_name)


def select_positions(pid, bank, length) -> list:
    """Release scheme: random.Random(f"PA4-banks-v1:6304:{pid}:{bank}"), per bin min(n, available) sorted."""
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    positions = []
    for lo, hi, n in BINS:
        candidates = list(range(lo, min(hi, length) + 1))
        positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))
    return positions


def require_sampler_check(path) -> dict:
    """The speculative bank needs the committed sampler check with every pass flag true."""
    rec = require_committed(path, "sampler check")
    d = json.loads(Path(path).read_text())
    passes = d.get("passes") or {}
    if d.get("status") != "ok" or not passes or not all(passes.values()) or d.get("all_pass") is not True:
        raise SystemExit(f"sampler check {path} does not pass: status={d.get('status')} passes={passes}")
    return {"path": str(path), "sha256": rec["sha256"], "commits": [c["commit"] for c in rec["commits"]],
            "passes": passes}


def require_preflight(path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("status") != "ok" or d.get("all_pass") is not True:
        raise SystemExit(f"preflight {path} does not pass: status={d.get('status')} all_pass={d.get('all_pass')}")
    from common.io import sha256_file

    return {"path": str(path), "sha256": sha256_file(path), "all_pass": True}


def _counts(res):
    keys = ("proposed", "accepted", "examined", "target_verification_passes", "draft_forward_calls", "n_rounds",
            "bonus_tokens", "bonus_skipped_cap", "residual_fallbacks", "discarded")
    return {k: res[k] for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--out", required=True, help="output .jsonl.gz (must not exist)")
    ap.add_argument("--target-path", default="model_target", help="local directory; default the locked hub revision")
    ap.add_argument("--draft-path", default=None, help="local directory; default the locked original student")
    ap.add_argument("--run-name", default=None, choices=sorted(OWN_RUNS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--limit-prompts", type=int, default=None, help="tests only; refused under results/")
    ap.add_argument("--sampler-check", default=str(SAMPLER_CHECK))
    ap.add_argument("--preflight", default=None, help="passing preflight JSON (required for the speculative bank)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if not out.name.endswith(".jsonl.gz"):
        raise SystemExit("--out must end with .jsonl.gz")
    run_json = out.with_name(out.name[: -len(".jsonl.gz")] + ".json")
    for p in (out, run_json):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    if a.limit_prompts is not None and out.resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise SystemExit("--limit-prompts is refused under results/")
    if a.kind == "own":
        if a.draft_path is None or a.run_name is None:
            raise SystemExit("--kind own needs --draft-path (merged final draft) and --run-name")
    elif a.run_name is not None:
        raise SystemExit("--run-name is only for --kind own")
    gate = {}
    if a.kind == "exact_spec_original_gamma4":
        gate["sampler_check"] = require_sampler_check(a.sampler_check)
        if a.preflight is None:
            raise SystemExit("the speculative bank needs --preflight with a passing real-model preflight JSON")
        gate["preflight"] = require_preflight(a.preflight)

    seed_everything(SEED_BASE)
    bank = bank_name(a.kind, a.run_name)
    prompts, prov = D.heldout_prompts()
    if a.limit_prompts is not None:
        prompts = prompts[: a.limit_prompts]
    tok = D.load_tokenizer(a.target_path)
    draft_src = a.draft_path or "model_draft"

    config = {
        "bank_id": BANK_ID, "sampler_id": SAMPLER_ID, "kind": a.kind, "bank": bank, "run_name": a.run_name,
        "max_new_tokens": MAX_NEW_TOKENS, "eos_ids": list(EOS_IDS), "bins": [list(b) for b in BINS],
        "seed_rule": f"{SEED_BASE} + {SEED_STRIDE}*j + {seed_offset(a.kind, a.run_name)} for prompt index j",
        "position_rule": "random.Random(f'PA4-banks-v1:6304:{pid}:{bank}'); per bin min(n, available) sorted; "
                         "position p stores seq[:p-1] and sampled token seq[p-1]",
        "prompt_format": "target tokenizer chat template, single user message, add_generation_prompt=True, "
                         "no explicit system message",
        "gamma": SPEC_GAMMA if a.kind == "exact_spec_original_gamma4" else None,
        "n_prompts": len(prompts), "limit_prompts": a.limit_prompts, "device": a.device, "dtype": a.dtype,
        "target_path": a.target_path, "draft_path": a.draft_path, "data_provenance": prov,
        "generation_settings": GENERATION_SETTINGS, "gate": gate, "out": str(out),
        "deviation": "banks (i) and (ii) generated by this generator instead of the release --prefix-banks "
                     "script (protocol D1)",
    }
    rec = RunRecorder(run_json, config)

    import time

    t_load = time.perf_counter()
    models = {}
    if a.kind in ("teacher", "exact_spec_original_gamma4"):
        models["target"] = D.load_model(a.target_path, a.device, a.dtype)
    if a.kind in ("original_draft", "exact_spec_original_gamma4", "own"):
        models["draft"] = D.load_model(draft_src, a.device, a.dtype)
    for _, (m, pv) in models.items():
        if pv["vocab_size"] != VOCAB_SIZE:
            raise SystemExit(f"vocabulary {pv['vocab_size']} != {VOCAB_SIZE}")
    revisions = {name: pv for name, (_, pv) in models.items()}
    rec.update(model_revisions=revisions)
    lms = {name: HFCachedLM(m, expect_vocab=VOCAB_SIZE) for name, (m, _) in models.items()}
    timing = {"load_s": time.perf_counter() - t_load, "prompt_s": []}

    records, terminations, lengths = [], Counter(), []
    for j, pr in enumerate(prompts):
        t_prompt = time.perf_counter()
        pid = pr["prompt_id"]
        ids = D.chat_ids(tok, pr["prompt"])
        seed = generation_seed(j, a.kind, a.run_name)
        gen = make_generator(seed, a.device)
        if a.kind == "exact_spec_original_gamma4":
            res = speculative_generate(lms["target"], lms["draft"], ids, SPEC_GAMMA, MAX_NEW_TOKENS, EOS_IDS, gen)
            extra = {"speculative_counts": _counts(res)}
        else:
            lm = lms["target"] if a.kind == "teacher" else lms["draft"]
            res = target_only_generate(lm, ids, MAX_NEW_TOKENS, EOS_IDS, gen)
            extra = {}
        seq = res["output_ids"]
        term = res["termination"]
        positions = select_positions(pid, bank, len(seq))
        common = {"prompt_id": pid, "domain": pr["domain"], "group": pr["group"], "bank": bank,
                  "prompt": pr["prompt"], "source_length": len(seq), "source_generation_seed": seed,
                  "termination_reason": term, "model_revision": revisions, "sampler_id": SAMPLER_ID,
                  "bank_id": BANK_ID}
        records.append({"record_type": "sequence", **common, "prompt_index": j, "prompt_token_ids": ids,
                        "response_token_ids": seq, "positions": positions, **extra})
        for p in positions:
            records.append({"record_type": "position", **common, "position": p,
                            "response_prefix_token_ids": seq[: p - 1], "sampled_token_id": seq[p - 1]})
        terminations[term] += 1
        lengths.append(len(seq))
        timing["prompt_s"].append(time.perf_counter() - t_prompt)
        rec.update(timing=timing, progress={"prompts_done": j + 1, "terminations": dict(terminations),
                             "positions": sum(1 for r in records if r["record_type"] == "position")})
    write_jsonl_gz(out, records)
    from common.io import sha256_gz_content

    n_pos = sum(1 for r in records if r["record_type"] == "position")
    rec.finish("ok", n_sequences=len(prompts), n_positions=n_pos, terminations=dict(terminations),
               mean_length=(sum(lengths) / len(lengths)) if lengths else None,
               output_content_sha256=sha256_gz_content(out))
    print(f"{bank}: {len(prompts)} sequences, {n_pos} positions, terminations {dict(terminations)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
