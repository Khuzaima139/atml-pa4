"""Task 2 Kaggle notebook 2: the B3 curriculum runs (protocol AM3) and Part C (C1 validate, C2-C5 eval, C5 bench).

python -m task2.kaggle_nb2 data-check [--out FILE]
python -m task2.kaggle_nb2 model-files --out FILE [--model-path DIR]
python -m task2.kaggle_nb2 prebuild --cache-dir DIR [--workers 4] [--out FILE] [--limit-updates N] [--seeds S ...]
python -m task2.kaggle_nb2 jobs --stage preflight|smoke|main --jobs-dir DIR [--plan FILE] [--results DIR]
       [--status DIR] [--cache-dir DIR] [--seeds S ...]
       [--device DEV --model-path DIR --limit-updates N --limit-eval M --limit-prompts SPEC --limit-reps W,R
       --limit-shapes L:B,.. --limit-configs C,..]   (test options)
python -m task2.kaggle_nb2 job --name NAME --status DIR --out DIR -- CMD ...
python -m task2.kaggle_nb2 preflight-check|smoke-report --jobs-dir DIR
python -m task2.kaggle_nb2 plan --jobs-dir DIR --prebuild FILE [--budget-h 10.5] [--session-h 11.5] [--elapsed-s S]
       [--vram-gib 14] [--out FILE]
python -m task2.kaggle_nb2 idle-check --jobs-dir DIR [--status DIR]
python -m task2.kaggle_nb2 summary|check|zip|final --jobs-dir DIR ...

Inventory. Phase 1 (concurrent queues): GPU 0 runs the three B3 runs python -m task2.train_mqar --mechanism
local_dilated --K 16 --regime mixed --curriculum near_to_mixed at seeds 6304, 6305, 6306 (outputs
results/task2/mqar/local_dilated_curriculum_K16_s<seed>/); GPU 1 runs python -m task2.kv_eval --mode validate on all
180 prompts (results/task2/part_c/validate/). Phase 2 (GPU 0 alone, GPU 1 idle): python -m task2.kv_eval --mode eval on
all 180 prompts (results/task2/part_c/eval/), then python -m task2.kv_bench (results/task2/part_c/bench/). Training
streams: near and mixed K 16 for each seed (6 streams), built and verified by `prebuild` as in notebook 1.

Stages. preflight (outputs under /kaggle/tmp/preflight): GPU 0 runs train_mqar curriculum --limit-updates 20
--limit-eval 64 (seed 6304), kv_eval eval on one 4k-level prompt and kv_bench at batch 8 and L 4096 for the hybrid
configuration (plus the fp16 reference, 1 warmup and 2 timed repetitions); GPU 1 runs kv_eval validate on the same
prompt. The prompt is the 4k-level case with the longest user text (ties by id), the largest prompt of that level.
smoke (under /kaggle/tmp/smoke): GPU 0 runs train_mqar curriculum --limit-updates 100 (seed 6304) with full development
and test evaluation; GPU 1 runs kv_eval eval on the longest prompt of each length level. main: phase 1 queues gpu0 and
gpu1, then the phase 2 queue phase2_gpu0.

Every job runs through `job` (task2.kaggle_nb1.run_job), which records in each run.json the other jobs that ran at the
same time (concurrent_jobs).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

from . import kaggle_nb1 as N1

GiB = N1.GiB
SEEDS = N1.SEEDS
K, MECH, REGIME, CURRICULUM = 16, "local_dilated", "mixed", "near_to_mixed"
STREAM_REGIMES = ("near", "mixed")
FULL_UPDATES = N1.FULL_UPDATES
N_CHECKPOINTS = N1.N_CHECKPOINTS
PREFLIGHT_UPDATES, PREFLIGHT_EVAL = 20, 64
SMOKE_UPDATES = 100
SMOKE_SEED = 6304
PREFLIGHT_BENCH = {"limit_shapes": "4096:8", "limit_configs": "hybrid", "limit_reps": "1,2"}
N_PROMPTS = 180
LEVELS = (2048, 4096)
SAFETY = N1.SAFETY
STAGGER_S = N1.STAGGER_S
BENCH_BLOCKS = 13  # 3 warmup + 10 timed
BENCH_SHAPES = 4
BENCH_CONFIGS = 4  # int4_token, int4_channel_k, hybrid, fp16 reference
CACHE_DIR = N1.CACHE_DIR
DEFAULTS = {
    "preflight": {"results": "/kaggle/tmp/preflight", "status": "notes/kaggle_status_smoke",
                  "logs": "notes/kaggle_logs_smoke", "prefix": "pf_"},
    "smoke": {"results": "/kaggle/tmp/smoke", "status": "notes/kaggle_status_smoke",
              "logs": "notes/kaggle_logs_smoke", "prefix": "sm_"},
    "main": {"results": "results/task2", "status": "notes/kaggle_status", "logs": "notes/kaggle_logs", "prefix": ""},
}
STAGES = tuple(DEFAULTS)
DONE_PATTERN = {"train": "train_mqar done: ", "validate": "validate done: ", "eval": "eval done: ",
                "bench": "bench done: "}
COMPARATOR = "local_dilated_mixed_K16"
LIMIT_FLAGS = {
    "train": (("limit_updates", "--limit-updates"), ("limit_eval", "--limit-eval")),
    "validate": (("limit_prompts", "--limit-prompts"), ("model_path", "--model-path")),
    "eval": (("limit_prompts", "--limit-prompts"), ("model_path", "--model-path")),
    "bench": (("limit_prompts", "--limit-prompts"), ("model_path", "--model-path"), ("limit_reps", "--limit-reps"),
              ("limit_shapes", "--limit-shapes"), ("limit_configs", "--limit-configs")),
}
TEST_KEYS = ("limit_updates", "limit_eval", "limit_prompts", "model_path", "limit_reps", "limit_shapes",
             "limit_configs")

_load_json, _jsonl, _write_new, _gib = N1._load_json, N1._jsonl, N1._write_new, N1._gib


def run_name(seed: int) -> str:
    return f"{MECH}_curriculum_K{K}_s{seed}"


def streams_needed(seeds=SEEDS) -> list:
    return sorted((K, r, s) for r in STREAM_REGIMES for s in seeds)


# ---------------------------------------------------------------------------------------------------------------
# Data check, model files, prompts.

def data_check() -> dict:
    """Notebook 1 checks (generator, development and test files, held-out hash list) plus the retrieval JSONL against
    the data repository manifest (180 distinct cases)."""
    from . import kv_run as KR

    rec = N1.data_check()
    ret = KR.data_record()
    if ret["sha256"] != ret["manifest_sha256"]:
        rec["problems"].append(f"retrieval file sha256 {ret['sha256']} != manifest {ret['manifest_sha256']}")
    else:
        try:
            ret["n_cases"] = len(KR.load_cases())
        except Exception as exc:
            rec["problems"].append(f"retrieval cases: {type(exc).__name__}: {exc}")
    rec["retrieval"] = ret
    return rec


def model_files(model_path=None) -> dict:
    """Download (or read) the locked checkpoint and record the sha256 and size of every file of its snapshot."""
    from common.io import sha256_file

    from . import kv_run as KR

    if model_path is None:
        from huggingface_hub import snapshot_download

        rev = KR.locked_revision()
        root = Path(snapshot_download(KR.MODEL_ID, revision=rev))
        problems = [] if root.name == rev else [f"snapshot directory {root.name} != revision {rev}"]
        source = {"model_id": KR.MODEL_ID, "revision": rev}
    else:
        root, problems, source = Path(model_path), [], {"model_path": str(model_path)}
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            real = p.resolve()
            files.append({"name": p.relative_to(root).as_posix(), "bytes": real.stat().st_size,
                          "sha256": sha256_file(real)})
    if not any(f["name"].endswith(".safetensors") for f in files):
        problems.append("no safetensors file in the snapshot")
    return {**source, "path": str(root), "files": files, "n_files": len(files),
            "total_bytes": sum(f["bytes"] for f in files), "problems": problems}


def longest_case_ids(cases=None) -> dict:
    """Per length level, the id of the case with the longest user text (ties by lower id)."""
    from . import kv_run as KR

    cases = KR.load_cases() if cases is None else cases
    out = {}
    for lv in LEVELS:
        cs = [c for c in cases if int(c["target_tokens"]) == lv]
        out[lv] = min(cs, key=lambda c: (-len(c["user_text"]), c["id"]))["id"]
    return out


def selected_cases(spec):
    from . import kv_run as KR

    return KR.select_cases(KR.load_cases(), spec)


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def _limit_args(kind: str, limits: dict) -> list:
    out = []
    for k, flag in LIMIT_FLAGS[kind]:
        v = limits.get(k)
        if v is not None:
            out += [flag, str(v)]
    return out


def stage_jobs(stage: str, results=None, status=None, cache_dir=CACHE_DIR, test=None, seeds=SEEDS,
               python=sys.executable, prompt_ids=None) -> list:
    """Job specs: name, stage, gpu, queue, phase, kind, out, cmd, wait_for, run, expect."""
    d = DEFAULTS[stage]
    R = Path(results or d["results"])
    S = Path(status or d["status"])
    t = dict(test or {})
    device = t.pop("device", None)
    pre = d["prefix"]
    jobs = []

    def lim(kind, **defaults):
        """Stage limits; test options replace those of the same name and add the model path for Part C jobs."""
        out = {k: v for k, v in defaults.items() if v is not None}
        allowed = {k for k, _ in LIMIT_FLAGS[kind]}
        for k in TEST_KEYS:
            if t.get(k) is not None and k in allowed and (k in defaults or k == "model_path"):
                out[k] = t[k]
        return out

    def add(gpu, queue, phase, kind, name, entry_args, out, limits, run):
        inner = [python, "-m", *entry_args, "--out", str(out)]
        if device:
            inner += ["--device", device]
        inner += _limit_args(kind, limits)
        cmd = [python, "-m", "task2.kaggle_nb2", "job", "--name", pre + name, "--status", str(S), "--out", str(out),
               "--", *inner]
        jobs.append({"name": pre + name, "stage": stage, "gpu": str(gpu), "queue": queue, "phase": phase,
                     "kind": kind, "out": str(out), "cmd": cmd, "wait_for": [], "run": run, "expect": dict(limits)})

    def train(gpu, queue, phase, name, seed, out, limits):
        args = ["task2.train_mqar", "--mechanism", MECH, "--K", str(K), "--regime", REGIME, "--seed", str(seed),
                "--curriculum", CURRICULUM, "--cache-dir", str(cache_dir)]
        add(gpu, queue, phase, "train", name, args, out, limits,
            {"mechanism": MECH, "regime": REGIME, "K": K, "seed": seed, "curriculum": CURRICULUM})

    def kv(gpu, queue, phase, kind, name, out, limits):
        args = ["task2.kv_bench"] if kind == "bench" else ["task2.kv_eval", "--mode", kind]
        add(gpu, queue, phase, kind, name, args, out, limits, {"mode": kind})

    if stage in ("preflight", "smoke") and prompt_ids is None and t.get("limit_prompts") is None:
        prompt_ids = longest_case_ids()
    if stage == "preflight":
        pid = None if prompt_ids is None else prompt_ids[4096]
        train("0", "gpu0", "preflight", "train_curriculum", SMOKE_SEED, R / "train",
              lim("train", limit_updates=PREFLIGHT_UPDATES, limit_eval=PREFLIGHT_EVAL))
        kv("0", "gpu0", "preflight", "eval", "kv_eval", R / "kv_eval", lim("eval", limit_prompts=pid))
        kv("0", "gpu0", "preflight", "bench", "kv_bench", R / "kv_bench", lim("bench", limit_prompts=None, **PREFLIGHT_BENCH))
        kv("1", "gpu1", "preflight", "validate", "kv_validate", R / "kv_validate", lim("validate", limit_prompts=pid))
    elif stage == "smoke":
        ids = None if prompt_ids is None else ",".join(prompt_ids[lv] for lv in LEVELS)
        train("0", "gpu0", "smoke", "train_curriculum", SMOKE_SEED, R / "train",
              lim("train", limit_updates=SMOKE_UPDATES, limit_eval=t.get("limit_eval")))
        kv("1", "gpu1", "smoke", "eval", "kv_eval", R / "kv_eval", lim("eval", limit_prompts=ids))
    else:
        for s in seeds:
            train("0", "gpu0", "phase1", f"train_{run_name(s)}", s, R / "mqar" / run_name(s),
                  lim("train", limit_updates=None, limit_eval=None))
        kv("1", "gpu1", "phase1", "validate", "kv_validate", R / "part_c" / "validate",
           lim("validate", limit_prompts=None))
        kv("0", "phase2_gpu0", "phase2", "eval", "kv_eval", R / "part_c" / "eval", lim("eval", limit_prompts=None))
        kv("0", "phase2_gpu0", "phase2", "bench", "kv_bench", R / "part_c" / "bench",
           lim("bench", limit_prompts=None, limit_reps=None, limit_shapes=None, limit_configs=None))
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)
    return jobs


# ---------------------------------------------------------------------------------------------------------------
# Facts of finished runs.

def _run(j) -> dict:
    return N1._run(j)


def curriculum_facts(run: dict) -> dict:
    """Notebook 1 training facts, with the near stream's load and validation also excluded from the fixed time."""
    f = N1.train_facts(run)
    near = run.get("stream_near") or {}
    extra = ((near.get("validation") or {}).get("seconds") or 0.0) + (near.get("load_seconds") or 0.0)
    f["fixed_s"] = max(0.0, f["fixed_s"] - extra)
    return f


def kv_peak(run: dict) -> int:
    """Largest peak allocated bytes of any device recorded by the run's recorder (0 on CPU)."""
    return max([int(v) for v in (run.get("peak_vram_allocated_bytes") or {}).values()] or [0])


def _rows_gz(path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def preflight_check(meta: list) -> list:
    problems = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            problems.append(f"{j['name']}: run status {run.get('status')!r}")
            continue
        if j["kind"] == "train":
            f = curriculum_facts(run)
            print(f"{j['name']:<24} params {run.get('parameter_count')}, {f['ms_per_update']:.1f} ms/update, peak train "
                  f"{_gib(f['peak_train']) or 0:.2f} GiB, eval {_gib(f['peak_eval']) or 0:.2f} GiB, consumed "
                  f"{json.dumps(run.get('curriculum_consumed'))}")
            if run.get("parameter_count") != N1.PARAMS:
                problems.append(f"{j['name']}: parameter count {run.get('parameter_count')}")
        else:
            tm = run.get("timing") or {}
            print(f"{j['name']:<24} load {tm.get('load_s') or 0:.1f} s, prompts {json.dumps(tm.get('prompt_s'))}, "
                  f"shapes {json.dumps(tm.get('shape_s'))}, peak {_gib(kv_peak(run)):.2f} GiB")
            if j["kind"] == "validate":
                s = run.get("summary") or {}
                print(f"    validate: {s.get('n_tokens_identical')}/{s.get('n_prompts')} identical, max logit diff "
                      f"{json.dumps(s.get('max_abs_logit_diff'))}")
            if j["kind"] == "bench":
                rows = _rows_gz(Path(j["out"]) / "bench_rows.jsonl.gz")
                for r in rows:
                    print(f"    bench L {r['L']} B {r['batch']} {r['config']:<15} {r['status']} peak "
                          f"{_gib(r.get('peak_allocated_bytes')) or 0:.2f} GiB")
                    if r["status"] != "ok":
                        problems.append(f"{j['name']}: {r['config']} status {r['status']}: {r.get('error')}")
    return problems


def smoke_report(meta: list) -> list:
    problems = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            problems.append(f"{j['name']}: run status {run.get('status')!r}")
            continue
        if j["kind"] == "train":
            f = curriculum_facts(run)
            print(f"{j['name']}: {run.get('updates_done')} updates, {f['ms_per_update']:.2f} ms/update, dev checkpoint "
                  f"eval {f['dev_eval_s']:.1f} s, test eval {json.dumps({k: round(v, 1) for k, v in f['test_eval_s'].items()})}"
                  f" s, fixed {f['fixed_s']:.1f} s, rows {json.dumps(run.get('n_rows'))}, peak train "
                  f"{_gib(f['peak_train']) or 0:.2f} GiB, eval {_gib(f['peak_eval']) or 0:.2f} GiB")
        else:
            tm = run.get("timing") or {}
            print(f"{j['name']}: load {tm.get('load_s') or 0:.1f} s, per prompt "
                  f"{json.dumps({k: round(v, 1) for k, v in (tm.get('prompt_s') or {}).items()})} s, rows "
                  f"{(run.get('summary') or {}).get('n_rows')}, peak {_gib(kv_peak(run)):.2f} GiB")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# Projection.

def _one(meta, kind):
    js = [j for j in meta if j["kind"] == kind]
    if len(js) != 1:
        raise SystemExit(f"expected one {kind} job, found {len(js)}")
    return js[0]


def project(jobs_dir, prebuild_rec: dict, budget_h: float = 10.5, session_h: float = 11.5, elapsed_s: float = 0.0,
            vram_gib: float = 14.0, safety: float = SAFETY, seeds=SEEDS, n_prompts: int = N_PROMPTS) -> dict:
    """Projection of both phases from the preflight and smoke records.

    B3 run: smoke ms per update x 4,500 + 6 x the smoke checkpoint evaluation + the smoke test evaluation + the smoke
    fixed time + the prebuild check and validation time of its near and mixed streams. validate: load + n_prompts x the
    preflight per-prompt time (the longest 4k prompt, with its full-forward checks). eval: load + n_prompts / 2 x the
    smoke 2k prompt time + n_prompts / 2 x the smoke 4k prompt time. bench: load + 2 x the preflight 4k cache build +
    4 shapes x the preflight batch-8 L-4096 shape time x (13 blocks x 4 configurations) / (its blocks x configurations).
    Phase 1 = max(GPU 0: the three B3 runs, GPU 1: 60 s offset + validate); phase 2 = eval + bench. Every job time is
    multiplied by the safety factor."""
    pf, sm = N1.load_meta(jobs_dir, "preflight"), N1.load_meta(jobs_dir, "smoke")
    for j in pf + sm:
        if _run(j).get("status") != "ok":
            raise SystemExit(f"{j['name']}: run not finished ok")
    st_s = {}
    for r in prebuild_rec["streams"]:
        if r.get("error"):
            raise SystemExit(f"prebuild failed for K{r['K']} {r['regime']} s{r['seed']}")
        st_s[(r["regime"], r["seed"])] = r["validation_s"] + r["check_s"]
    missing = [(rg, s) for rg in STREAM_REGIMES for s in seeds if (rg, s) not in st_s]
    if missing:
        raise SystemExit(f"prebuild record has no stream for {missing}")
    f = curriculum_facts(_run(_one(sm, "train")))
    rows = []
    for s in seeds:
        parts = {"train": f["ms_per_update"] * FULL_UPDATES / 1000.0, "dev_eval": N_CHECKPOINTS * f["dev_eval_s"],
                 "test_eval": sum(f["test_eval_s"].values()), "fixed": f["fixed_s"],
                 "streams_check_and_validation": st_s[("near", s)] + st_s[("mixed", s)]}
        rows.append({"name": f"train_{run_name(s)}", "phase": "phase1", "gpu": "0", "parts": parts,
                     "projected_s": safety * sum(parts.values())})
    pv, pe, pb = _run(_one(pf, "validate")), _run(_one(pf, "eval")), _run(_one(pf, "bench"))
    se = _run(_one(sm, "eval"))
    tv = pv["timing"]
    v_prompt = max(tv["prompt_s"].values())
    rows.append({"name": "kv_validate", "phase": "phase1", "gpu": "1",
                 "parts": {"load": tv["load_s"], "prompts": n_prompts * v_prompt},
                 "projected_s": safety * (tv["load_s"] + n_prompts * v_prompt)})
    te = se["timing"]
    by_level = {}
    for case in se["config"]["prompt_ids"]:
        lv = int(case.split("-")[1])
        by_level[lv] = max(by_level.get(lv, 0.0), te["prompt_s"][case])
    lv_max = max(by_level.values())
    e_prompts = sum((n_prompts / len(LEVELS)) * by_level.get(lv, lv_max) for lv in LEVELS)
    rows.append({"name": "kv_eval", "phase": "phase2", "gpu": "0",
                 "parts": {"load": te["load_s"], "prompts": e_prompts},
                 "projected_s": safety * (te["load_s"] + e_prompts)})
    tb = pb["timing"]
    blocks = pb["config"]["warmup"] + pb["config"]["repetitions"]
    n_cfg = len(pb["config"]["configs"]) + 1
    shape_s = max(tb["shape_s"].values())
    b_parts = {"load": tb["load_s"], "caches": len(LEVELS) * max(tb["level_caches_s"].values()),
               "shapes": BENCH_SHAPES * shape_s * (BENCH_BLOCKS * BENCH_CONFIGS) / (blocks * n_cfg)}
    rows.append({"name": "kv_bench", "phase": "phase2", "gpu": "0", "parts": b_parts,
                 "projected_s": safety * sum(b_parts.values())})
    secs = {r["name"]: r["projected_s"] for r in rows}
    q0 = sum(r["projected_s"] for r in rows if r["phase"] == "phase1" and r["gpu"] == "0")
    q1 = STAGGER_S + secs["kv_validate"]
    phase1 = max(q0, q1)
    phase2 = secs["kv_eval"] + secs["kv_bench"]
    total = phase1 + phase2
    peaks = {"smoke train": f["peak_train"] or 0, "smoke train eval": f["peak_eval"] or 0}
    ft = curriculum_facts(_run(_one(pf, "train")))
    peaks["preflight train"] = max(ft["peak_train"] or 0, ft["peak_eval"] or 0)
    peaks["preflight validate"] = kv_peak(pv)
    peaks["preflight eval"] = kv_peak(pe)
    peaks["preflight bench"] = kv_peak(pb)
    peaks["smoke eval"] = kv_peak(se)
    reasons = []
    if total > budget_h * 3600:
        reasons.append(f"phase 1 {phase1 / 3600:.2f} h + phase 2 {phase2 / 3600:.2f} h = {total / 3600:.2f} h above "
                       f"{budget_h} h")
    if elapsed_s + total > session_h * 3600:
        reasons.append(f"elapsed {elapsed_s / 3600:.2f} h + projected {total / 3600:.2f} h above {session_h} h")
    reasons += [f"peak {k} {v / GiB:.2f} GiB above {vram_gib} GiB" for k, v in peaks.items() if v > vram_gib * GiB]
    return {"rows": rows, "queue_s": {"gpu0": q0, "gpu1": q1}, "phase1_s": phase1, "phase2_s": phase2,
            "total_s": total, "elapsed_s": elapsed_s, "budget_h": budget_h, "session_h": session_h, "safety": safety,
            "peaks_bytes": peaks, "vram_limit_gib": vram_gib, "smoke_train_facts": f,
            "eval_prompt_s_by_level": {str(k): v for k, v in by_level.items()}, "validate_prompt_s": v_prompt,
            "over": bool(reasons), "reasons": reasons}


def print_plan(p: dict):
    for r in p["rows"]:
        print(f"{r['phase']} GPU {r['gpu']} {r['name']:<38}{r['projected_s'] / 60:>8.1f} min  " +
              ", ".join(f"{k} {v / 60:.1f}" for k, v in r["parts"].items()))
    print(f"phase 1: gpu0 {p['queue_s']['gpu0'] / 3600:.2f} h, gpu1 {p['queue_s']['gpu1'] / 3600:.2f} h (incl. start "
          f"offset) -> {p['phase1_s'] / 3600:.2f} h; phase 2 {p['phase2_s'] / 3600:.2f} h; total "
          f"{p['total_s'] / 3600:.2f} h; elapsed so far {p['elapsed_s'] / 3600:.2f} h; budget {p['budget_h']} h, "
          f"session {p['session_h']} h; safety factor {p['safety']}")
    print("peaks (GiB): " + ", ".join(f"{k} {v / GiB:.2f}" for k, v in p["peaks_bytes"].items()))
    for x in p["reasons"]:
        print("OVER:", x)
    print("plan: within limits" if not p["over"] else "plan: OVER LIMIT, queues not started")


# ---------------------------------------------------------------------------------------------------------------
# Check.

def expected_kv(j) -> dict:
    from . import kv_bench as BN
    from . import kv_run as KR

    ex = j["expect"]
    cases = selected_cases(ex.get("limit_prompts"))
    out = {"n": len(cases), "cells": len(KR.first_per_cell(cases)), "qual": len(KR.qualitative_ids(cases))}
    if j["kind"] == "bench":
        shapes = BN.parse_shapes(ex.get("limit_shapes"))
        levels = {lv for lv in KR.LEVELS if any(int(c["target_tokens"]) == lv for c in cases)}
        shapes = [s for s in shapes if s[0] in levels]
        out["rows"] = len(shapes) * len(BN.parse_configs(ex.get("limit_configs")))
        out["reps"] = tuple(int(x) for x in ex["limit_reps"].split(",")) if ex.get("limit_reps") else \
            (BN.WARMUP, BN.REPS)
    return out


def _check_curriculum(j, run, out: Path, problems: list):
    n = j["name"]
    N1._check_train(j, run, out, problems)
    cfg = run.get("config") or {}
    updates = j["expect"].get("limit_updates") or FULL_UPDATES
    cur = cfg.get("curriculum") or {}
    if cur.get("name") != CURRICULUM or cfg.get("run_name") != run_name(j["run"]["seed"]):
        problems.append(f"{n}: curriculum {cur.get('name')!r} or run_name {cfg.get('run_name')!r} differs")
    from .train_mqar import curriculum_breaks, curriculum_plan
    if cur.get("breaks") != list(curriculum_breaks(updates)):
        problems.append(f"{n}: curriculum breaks {cur.get('breaks')} != {list(curriculum_breaks(updates))}")
    plan = curriculum_plan(updates)
    cons = run.get("curriculum_consumed") or {}
    if (cons.get("n_near"), cons.get("n_mixed")) != (plan[-1]["near"][1], plan[-1]["mixed"][1]):
        problems.append(f"{n}: consumed {cons.get('n_near')}/{cons.get('n_mixed')} != "
                        f"{plan[-1]['near'][1]}/{plan[-1]['mixed'][1]}")
    for key in ("stream", "stream_near"):
        dj = ((run.get(key) or {}).get("disjointness") or {})
        if dj.get("collisions") != 0:
            problems.append(f"{n}: {key} disjointness {dj.get('collisions')!r}")
    p = out / "train_log.jsonl"
    if p.is_file():
        got = [r.get("n_near") for r in _jsonl(p)]
        if got != [r["n_near"] for r in plan]:
            problems.append(f"{n}: train_log n_near differs from the AM3 schedule")


def _check_kv(j, run, out: Path, problems: list):
    n, kind = j["name"], j["kind"]
    ex = expected_kv(j)
    try:
        if kind == "validate":
            rows = _rows_gz(out / "validate_rows.jsonl.gz")
            ff = _rows_gz(out / "full_forward_rows.jsonl.gz")
            s = run.get("summary") or {}
            if len(rows) != ex["n"] or s.get("n_prompts") != ex["n"] or len(ff) != ex["cells"]:
                problems.append(f"{n}: validate rows {len(rows)}, summary {s.get('n_prompts')}, full-forward rows "
                                f"{len(ff)} != {ex['n']}, {ex['n']}, {ex['cells']}")
        elif kind == "eval":
            rows = _rows_gz(out / "eval_rows.jsonl.gz")
            pr = _rows_gz(out / "prompt_rows.jsonl.gz")
            npz = sorted((out / "npz").glob("*.npz"))
            qual = sorted((out / "qualitative").glob("*.txt"))
            if (len(rows), len(pr), len(npz), len(qual)) != (6 * ex["n"], ex["n"], ex["n"], ex["qual"]):
                problems.append(f"{n}: eval rows {len(rows)}, prompt rows {len(pr)}, npz {len(npz)}, qualitative "
                                f"{len(qual)} != {6 * ex['n']}, {ex['n']}, {ex['n']}, {ex['qual']}")
            per = (run.get("summary") or {}).get("per_method") or {}
            if len(per) != 6 or any(v.get("n") != ex["n"] for v in per.values()):
                problems.append(f"{n}: per-method counts {[(k, v.get('n')) for k, v in per.items()]}")
            bad = [r["id"] + "/" + r["method"] for r in rows if not r.get("positions_preserved")]
            if bad:
                problems.append(f"{n}: positions not preserved in {len(bad)} rows, first {bad[:3]}")
        else:
            rows = _rows_gz(out / "bench_rows.jsonl.gz")
            if len(rows) != ex["rows"] or run.get("n_rows") != ex["rows"]:
                problems.append(f"{n}: bench rows {len(rows)} (run {run.get('n_rows')}) != {ex['rows']}")
            for r in rows:
                if r["status"] != "ok":
                    problems.append(f"{n}: L {r['L']} B {r['batch']} {r['config']} status {r['status']}")
                elif (r["n_warmup"], r["n_reps"]) != ex["reps"]:
                    problems.append(f"{n}: L {r['L']} B {r['batch']} {r['config']} reps {(r['n_warmup'], r['n_reps'])}")
    except Exception as exc:
        problems.append(f"{n}: row files unreadable: {type(exc).__name__}: {exc}")


def check(meta: list, status_dir, logs_dir) -> list:
    problems = []
    for j in meta:
        n, out = j["name"], Path(j["out"])
        st = N1._status(status_dir, n)
        if st["exit"] != 0 or not st["done"]:
            problems.append(f"{n}: exit {st['exit']}, done {st['done']}")
        run = _run(j)
        if run.get("status") != "ok" or run.get("step") != "done":
            problems.append(f"{n}: run status {run.get('status')!r}, step {run.get('step')!r}")
            continue
        limited = any(v is not None for v in j["expect"].values())
        if bool((run.get("config") or {}).get("limited")) != limited:
            problems.append(f"{n}: config.limited {(run.get('config') or {}).get('limited')} != {limited}")
        conc = run.get(N1.CONCURRENCY_KEY)
        if not isinstance(conc, dict) or "overlapping" not in conc:
            problems.append(f"{n}: {N1.CONCURRENCY_KEY} missing")
        elif j["phase"] == "phase2" and conc["overlapping"]:
            problems.append(f"{n}: phase 2 job overlapped {[o['name'] for o in conc['overlapping']]}")
        if j["kind"] == "train":
            _check_curriculum(j, run, out, problems)
        else:
            _check_kv(j, run, out, problems)
        log = Path(logs_dir) / f"{n}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        for pat in (DONE_PATTERN[j["kind"]], f"{n}: exit code 0", f"[job] {n}: concurrent jobs"):
            if pat not in text:
                problems.append(f"{n}: log line {pat!r} not found in {log}")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# Summary.

def _a(x):
    return "  -  " if x is None else f"{x:.3f}"


def _cell_acc(run, cell, cond="native"):
    return (((run.get("test_summary") or {}).get(cond) or {}).get(cell) or {}).get("accuracy")


def summary_lines(meta: list, max_lines: int = 60, comparator_root=None) -> list:
    from common.paths import RESULTS_DIR

    comp_root = Path(comparator_root) if comparator_root else RESULTS_DIR / "task2" / "mqar"
    L = []
    train = [j for j in meta if j["kind"] == "train"]
    if train:
        ck = None
        L.append("B3 local_dilated curriculum K16: learned (B6), dev mixed_K16 accuracy per checkpoint, test mixed_K16 / far_K16")
        for j in train:
            run = _run(j)
            s = j["run"]["seed"]
            if run.get("status") != "ok":
                L.append(f"  s{s}: status {run.get('status')!r}")
                continue
            curve = run.get("dev_curve") or []
            ck = [p["update"] for p in curve]
            lf = "Y" if (run.get("learned") or {}).get("learned") else "n"
            dev = " ".join(_a(p["cells"]["mixed_K16"]["accuracy"]) for p in curve)
            L.append(f"  s{s} {lf} dev {dev} | test {_a(_cell_acc(run, 'mixed_K16'))} / {_a(_cell_acc(run, 'far_K16'))}")
        if ck:
            L.append(f"  checkpoints {ck}")
        for j in train:
            s = j["run"]["seed"]
            comp = _load_json(comp_root / f"{COMPARATOR}_s{s}" / "run.json") or {}
            if comp.get("status") == "ok":
                lf = "Y" if (comp.get("learned") or {}).get("learned") else "n"
                L.append(f"  comparator {COMPARATOR}_s{s} {lf} test {_a(_cell_acc(comp, 'mixed_K16'))} / "
                         f"{_a(_cell_acc(comp, 'far_K16'))}")
    for j in meta:
        run = _run(j)
        if j["kind"] == "train":
            continue
        if run.get("status") != "ok":
            L.append(f"{j['name']}: status {run.get('status')!r}")
            continue
        out = Path(j["out"])
        try:
            if j["kind"] == "validate":
                s = run["summary"]
                mx = (s.get("max_abs_logit_diff") or {}).get("max")
                L.append(f"validate: {s['n_tokens_identical']}/{s['n_prompts']} generations identical, max logit diff "
                         f"{mx if mx is None else f'{mx:.4g}'}, generate accuracy {s['generate_accuracy_overall']:.3f}, "
                         f"positions preserved {s['n_positions_preserved']}/{s['n_prompts']}")
            elif j["kind"] == "eval":
                rows = _rows_gz(out / "eval_rows.jsonl.gz")
                per = run["summary"]["per_method"]
                L.append("eval exact match: " + ", ".join(f"{m} {v['accuracy']:.3f}" for m, v in per.items()))
                fp = [r for r in rows if r["method"] == "fp16"]
                cells = sorted({(r["target_tokens"], r["position"]) for r in fp})
                L.append("eval fp16 by length x location: " + ", ".join(
                    f"{lv}-{pos} {sum(r['correct'] for r in fp if (r['target_tokens'], r['position']) == (lv, pos)) / sum(1 for r in fp if (r['target_tokens'], r['position']) == (lv, pos)):.3f}"
                    for lv, pos in cells))
                ev = []
                for m in ("question_aware", "hybrid"):
                    vals = [r["evidence_fraction_retained"] for r in rows if r["method"] == m
                            and r["evidence_fraction_retained"] is not None]
                    ev.append(f"{m} {sum(vals) / len(vals):.3f}" if vals else f"{m} -")
                L.append("eval mean evidence retained: " + ", ".join(ev))
                rf = []
                for m in per:
                    vals = [r["realized_fraction"] for r in rows if r["method"] == m]
                    rf.append(f"{m} {min(vals):.4f}-{max(vals):.4f}")
                L.append("eval realized byte fraction: " + ", ".join(rf))
            else:
                rows = _rows_gz(out / "bench_rows.jsonl.gz")
                sel = {r["config"]: r for r in rows if r["batch"] == 8 and r["L"] == 4096}
                ref = ((sel.get("fp16") or {}).get("stats_ms") or {}).get("decode_step") or {}
                parts = []
                for c, r in sel.items():
                    if c == "fp16":
                        continue
                    sm = r.get("stats_ms") or {}
                    med = {p: (sm.get(p) or {}).get("median") for p in ("pack", "unpack_dequant", "decode_step")}
                    ratio = med["decode_step"] / ref["median"] if med["decode_step"] and ref.get("median") else None
                    parts.append(f"{c} pack {_a(med['pack'])} ms unpack {_a(med['unpack_dequant'])} ms decode ratio "
                                 f"{_a(ratio)}")
                L.append(f"bench B8 L4096 (fp16 decode {_a(ref.get('median'))} ms): " + "; ".join(parts) if sel else
                         "bench: no B8 L4096 rows")
        except Exception as exc:
            L.append(f"{j['name']}: summary unavailable: {type(exc).__name__}: {exc}")
    return L[:max_lines]


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    dc = sub.add_parser("data-check")
    dc.add_argument("--out", default=None)
    mf = sub.add_parser("model-files")
    mf.add_argument("--out", required=True)
    mf.add_argument("--model-path", default=None)
    pb = sub.add_parser("prebuild")
    pb.add_argument("--cache-dir", required=True)
    pb.add_argument("--workers", type=int, default=4)
    pb.add_argument("--out", default=None)
    pb.add_argument("--limit-updates", type=int, default=None, help="tests only")
    pb.add_argument("--seeds", nargs="*", type=int, default=None)
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    j.add_argument("--plan", default=None)
    j.add_argument("--results", default=None)
    j.add_argument("--status", default=None)
    j.add_argument("--cache-dir", default=CACHE_DIR)
    j.add_argument("--seeds", nargs="*", type=int, default=None)
    j.add_argument("--device", default=None)
    j.add_argument("--model-path", default=None)
    j.add_argument("--limit-updates", type=int, default=None)
    j.add_argument("--limit-eval", type=int, default=None)
    for k in ("limit-prompts", "limit-reps", "limit-shapes", "limit-configs"):
        j.add_argument("--" + k, default=None)
    jb = sub.add_parser("job")
    jb.add_argument("--name", required=True)
    jb.add_argument("--status", required=True)
    jb.add_argument("--out", required=True)
    jb.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("preflight-check", "smoke-report"):
        s = sub.add_parser(name)
        s.add_argument("--jobs-dir", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--jobs-dir", required=True)
    p.add_argument("--prebuild", required=True)
    p.add_argument("--budget-h", type=float, default=10.5)
    p.add_argument("--session-h", type=float, default=11.5)
    p.add_argument("--elapsed-s", type=float, default=0.0)
    p.add_argument("--vram-gib", type=float, default=14.0)
    p.add_argument("--out", default=None)
    ic = sub.add_parser("idle-check")
    ic.add_argument("--jobs-dir", required=True)
    ic.add_argument("--status", default=None)
    for name in ("summary", "check", "zip", "final"):
        s = sub.add_parser(name)
        s.add_argument("--jobs-dir", required=True)
        s.add_argument("--status", default=None)
        s.add_argument("--logs", default=None)
        if name == "zip":
            s.add_argument("--out", required=True)
            s.add_argument("--include", nargs="*", default=[])
        if name == "check":
            s.add_argument("--out", default=None)
        if name == "final":
            s.add_argument("--check-report", required=True)
            s.add_argument("--zip", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "job":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            raise SystemExit("job: no command after --")
        return N1.run_job(a.name, a.status, a.out, cmd)
    if a.cmd == "data-check":
        rec = data_check()
        print(json.dumps({"generator": rec["generator"], "files": rec["files"], "retrieval": rec["retrieval"],
                          "heldout_list": {k: rec["heldout_list"][k] for k in
                                           ("path", "sha256", "n_sequences", "files", "tracked", "clean")}}, indent=2))
        for x in rec["problems"]:
            print("PROBLEM:", x)
        if a.out:
            _write_new(a.out, rec)
        print("data-check:", "ok" if not rec["problems"] else f"{len(rec['problems'])} problems")
        return 1 if rec["problems"] else 0
    if a.cmd == "model-files":
        rec = model_files(a.model_path)
        for f in rec["files"]:
            print(f"{f['sha256']}  {f['bytes']:>12}  {f['name']}")
        for x in rec["problems"]:
            print("PROBLEM:", x)
        _write_new(a.out, rec)
        print(f"model-files: {rec['n_files']} files, {rec['total_bytes']} bytes, "
              + ("ok" if not rec["problems"] else f"{len(rec['problems'])} problems"))
        return 1 if rec["problems"] else 0
    if a.cmd == "prebuild":
        streams = streams_needed(tuple(a.seeds) if a.seeds else SEEDS)
        rec = N1.prebuild(a.cache_dir, streams, a.workers, a.limit_updates or FULL_UPDATES)
        N1.print_prebuild(rec)
        if a.out:
            _write_new(a.out, rec)
        return 1 if rec["n_failed"] else 0
    if a.cmd == "jobs":
        test = {"device": a.device, "model_path": a.model_path, "limit_updates": a.limit_updates,
                "limit_eval": a.limit_eval, "limit_prompts": a.limit_prompts, "limit_reps": a.limit_reps,
                "limit_shapes": a.limit_shapes, "limit_configs": a.limit_configs}
        test = {k: v for k, v in test.items() if v is not None}
        if a.plan:
            plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
            if plan.get("over"):
                raise SystemExit(f"plan {a.plan} is over the limits; no main jobs written")
        jobs = stage_jobs(a.stage, a.results, a.status, a.cache_dir, test, tuple(a.seeds) if a.seeds else SEEDS)
        paths = N1.write_jobs(a.stage, a.jobs_dir, jobs)
        for q in N1.queues(jobs):
            print(f"{q}: " + ", ".join(x["name"] for x in jobs if x["queue"] == q))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd in ("preflight-check", "smoke-report"):
        stage = "preflight" if a.cmd == "preflight-check" else "smoke"
        meta = N1.load_meta(a.jobs_dir, stage)
        problems = (preflight_check if stage == "preflight" else smoke_report)(meta)
        for x in problems:
            print("PROBLEM:", x)
        print(f"{a.cmd}: {len(meta)} jobs, {len(problems)} problems")
        return 1 if problems else 0
    if a.cmd == "plan":
        pre = json.loads(Path(a.prebuild).read_text(encoding="utf-8"))
        pl = project(a.jobs_dir, pre, a.budget_h, a.session_h, a.elapsed_s, a.vram_gib)
        print_plan(pl)
        if a.out:
            _write_new(a.out, pl)
        return 1 if pl["over"] else 0
    meta = N1.load_meta(a.jobs_dir, "main")
    status = a.status or DEFAULTS["main"]["status"]
    if a.cmd == "idle-check":
        problems = N1.idle_check([x for x in meta if x["phase"] == "phase1"], status)
        for x in problems:
            print("NOT IDLE:", x)
        print("idle-check:", "no job or queue alive; every phase 1 job final" if not problems else "NOT IDLE")
        return 1 if problems else 0
    if a.cmd == "summary":
        for line in summary_lines(meta):
            print(line)
        return 0
    if a.cmd == "check":
        problems = check(meta, status, a.logs or DEFAULTS["main"]["logs"])
        bad = [j["name"] for j in meta if any(p.startswith(j["name"] + ":") for p in problems)]
        for x in problems:
            print("FAIL:", x)
        if problems:
            print(f"==== CHECK: FAIL: {len(problems)} problems in {len(bad)} of {len(meta)} jobs ====")
            print("FAILED JOBS: " + ", ".join(bad))
        else:
            print(f"==== CHECK: PASS: {len(meta)} jobs, 0 problems ====")
        if a.out:
            _write_new(a.out, {"n_jobs": len(meta), "problems": problems, "failed_jobs": bad})
        return 1 if problems else 0
    if a.cmd == "final":
        reasons = N1.final(meta, status, a.check_report, a.zip)
        for x in reasons:
            print("NOT OK:", x)
        print("notebook complete: every job ok, check clean, zip written" if not reasons else
              f"notebook not complete: {len(reasons)} reasons")
        return 1 if reasons else 0
    res = N1.make_zip(a.out, a.include)
    for x in res["missing"]:
        print("not in the zip (missing):", x)
    print(f"{res['path']}: {res['n_files']} files, {res['bytes']} bytes, sha256 {res['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
