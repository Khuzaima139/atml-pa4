"""Task 2 Kaggle notebook 1: Part A verify and bench, the Part B B1/B2 training runs and the forward-only sweep.

python -m task2.kaggle_nb1 data-check [--out FILE]
python -m task2.kaggle_nb1 prebuild --cache-dir DIR [--workers 4] [--out FILE] [--limit-updates N] [--streams K:REGIME:SEED ...]
python -m task2.kaggle_nb1 jobs --stage preflight|smoke|main --jobs-dir DIR [--plan FILE] [--results DIR]
       [--status DIR] [--cache-dir DIR] [--configs NAME ...] [--seeds S ...]
       [--device DEV --limit-updates N --limit-eval M --verify-lengths L,.. --bench-lengths L,.. --sweep-lengths T,..
       --reps W,R]   (test options)
python -m task2.kaggle_nb1 job --name NAME --status DIR --out DIR -- CMD ...
python -m task2.kaggle_nb1 preflight-check|smoke-report --jobs-dir DIR
python -m task2.kaggle_nb1 plan --jobs-dir DIR --prebuild FILE [--budget-h 10.5] [--session-h 11.5] [--elapsed-s S]
       [--vram-gib 14] [--out FILE]
python -m task2.kaggle_nb1 idle-check --jobs-dir DIR
python -m task2.kaggle_nb1 summary|check|zip|final --jobs-dir DIR ...

Run inventory (protocol B1, seeds 6304, 6305, 6306 each): dense mixed K 4, 8, 16; dense near K 16; window near K 4, 8,
16; window mixed K 16; linear mixed K 4, 8, 16; local_dilated mixed K 4, 16 (13 configurations, 39 runs), plus
python -m task2.part_a verify (evaluation phase) and, in the timing phase on GPU 0 alone, python -m task2.part_a bench
and python -m task2.mqar_sweep. Training streams: the 18 (K, regime, seed) streams of these runs.

Stages. preflight (outputs under /kaggle/tmp/preflight): part_a verify at L 256 and 8192 (the edge cases run at 256)
and train_mqar --limit-updates 20 --limit-eval 64 of every mechanism (K 16 mixed, seed 6304) on GPU 0; the same four
trainings and mqar_sweep at T 8192 (all four mechanisms, 1 warmup and 2 timed repetitions) on GPU 1. smoke (under
/kaggle/tmp/smoke): train_mqar --limit-updates 100 at K 16 mixed, seed 6304, with full development and test evaluation
(dense and window on GPU 0, local_dilated and linear on GPU 1) and part_a verify at L 256 and 2048 (GPU 0). main:
the 39 trainings and part_a verify on GPU 0 / GPU 1 by the plan's assignment (queues gpu0, gpu1; outputs
results/task2/mqar/<mechanism>_<regime>_K<K>_s<seed>/ and results/task2/part_a/verify/), then the timing queue
timing_gpu0 (part_a bench to results/task2/part_a/bench/, mqar_sweep to results/task2/mqar_sweep/).

Every job runs through `job`: the command runs with the Task 1 job environment (PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True), signals are forwarded, its exit code is returned and on a nonzero exit the status fields of
<out>/run.json are printed. The wrapper writes <status>/<name>.interval.json (start, end, GPU) and, when the command has
ended, adds "concurrent_jobs" to <out>/run.json: every other job of the same status directory whose interval overlaps
this one, and those alive at the start.
"""
import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

GiB = 2 ** 30
SEEDS = (6304, 6305, 6306)
CONFIGS = (("dense", "mixed", 4), ("dense", "mixed", 8), ("dense", "mixed", 16), ("dense", "near", 16),
           ("window", "near", 4), ("window", "near", 8), ("window", "near", 16), ("window", "mixed", 16),
           ("linear", "mixed", 4), ("linear", "mixed", 8), ("linear", "mixed", 16),
           ("local_dilated", "mixed", 4), ("local_dilated", "mixed", 16))
MECHANISMS = ("dense", "window", "local_dilated", "linear")
SMOKE_K, SMOKE_REGIME, SMOKE_SEED = 16, "mixed", 6304
PREFLIGHT_UPDATES, PREFLIGHT_EVAL = 20, 64
SMOKE_UPDATES = 100
PREFLIGHT_VERIFY_LENGTHS = (256, 8192)
SMOKE_VERIFY_LENGTHS = (256, 2048)
PREFLIGHT_SWEEP_LENGTHS = (8192,)
PREFLIGHT_SWEEP_REPS = (1, 2)
FULL_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
SWEEP_LENGTHS = (256, 1024, 4096, 8192)
FULL_UPDATES = 4500
N_CHECKPOINTS = 6
DEV_PER_CELL, TEST_PER_CELL = 256, 1024
KS = (4, 8, 16)
EVAL_REGIMES = ("near", "far", "mixed")
PARAMS = 462_336
SAFETY = 1.15
STAGGER_S = 60.0
BENCH_REPS = 13  # 3 warmup + 10 timed
CACHE_DIR = "/kaggle/tmp/mqar_cache"
DEFAULTS = {
    "preflight": {"results": "/kaggle/tmp/preflight", "status": "notes/kaggle_status_smoke",
                  "logs": "notes/kaggle_logs_smoke", "prefix": "pf_"},
    "smoke": {"results": "/kaggle/tmp/smoke", "status": "notes/kaggle_status_smoke",
              "logs": "notes/kaggle_logs_smoke", "prefix": "sm_"},
    "main": {"results": "results/task2", "status": "notes/kaggle_status", "logs": "notes/kaggle_logs", "prefix": ""},
}
STAGES = tuple(DEFAULTS)
DONE_PATTERN = {"train": "train_mqar done: ", "verify": "verify done: ", "bench": "bench done: ",
                "sweep": "mqar_sweep done: "}
CONCURRENCY_KEY = "concurrent_jobs"
CONCURRENCY_METHOD = ("intervals written by the job wrapper into the status directory: start before this job's end "
                      "and end after its start (or no end yet)")


def run_name(mech, regime, K, seed) -> str:
    return f"{mech}_{regime}_K{K}_s{seed}"


def config_name(mech, regime, K) -> str:
    return f"{mech}_{regime}_K{K}"


def streams_needed(configs=CONFIGS, seeds=SEEDS) -> list:
    """Sorted (K, regime, seed) of every training stream the given runs read."""
    return sorted({(K, regime, s) for _, regime, K in configs for s in seeds})


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _jsonl(path) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _gib(b):
    return None if b is None else b / GiB


# ---------------------------------------------------------------------------------------------------------------
# Data check and stream prebuild.

def data_check() -> dict:
    """Generator, development and test files against the data repository manifest; the committed held-out hash list
    (tracked, unmodified, source file sha256 equal to the manifest)."""
    from common.freeze import require_committed

    from . import mqar_data as D

    _, gen = D.load_generator()
    files = {"development": D.file_info(D.DEV_FILE, D.DEV_REL), "test": D.file_info(D.TEST_FILE, D.TEST_REL)}
    git = require_committed(D.HELDOUT_LIST, "held-out hash list")
    _, lst = D.load_heldout_list(D.HELDOUT_LIST)
    by_name = {f["rel"]: f["sha256"] for f in lst["files"]}
    problems = [f"held-out list source {k} sha256 {by_name.get(rel)} != {files[k]['sha256']}"
                for k, rel in (("development", D.DEV_REL), ("test", D.TEST_REL)) if by_name.get(rel) != files[k]["sha256"]]
    return {"generator": gen, "files": files, "heldout_list": {**lst, "tracked": git["tracked"], "clean": git["clean"]},
            "problems": problems}


def build_one(cache_dir, K: int, regime: str, seed: int, n_updates: int = FULL_UPDATES) -> dict:
    """Build (or reuse) one stream in the cache, then verify it: sidecar and file sha256, index range, every row through
    the generator's validate() and the filler check."""
    from . import mqar_data as D

    t0 = time.perf_counter()
    gen, gen_info = D.load_generator()
    start, stop = D.seed_index_range(seed, n_updates)
    arr, info = D.load_or_build_stream(cache_dir, gen, gen_info, K, regime, start, stop)
    t_build = time.perf_counter() - t0
    npy, meta_path, _ = D.cache_paths(cache_dir, K, regime, start, stop)
    t1 = time.perf_counter()
    arr2, sha = D._check_cache(npy, meta_path, gen, K, regime, start, stop, gen_info)
    t_check = time.perf_counter() - t1
    meta = json.loads(meta_path.read_text())
    problems = []
    if sha != info["sha256"] or meta.get("sha256") != sha:
        problems.append(f"sha256 {sha} vs record {info['sha256']} vs sidecar {meta.get('sha256')}")
    if (meta.get("start"), meta.get("stop")) != (start, stop) or arr2.shape[0] != stop - start:
        problems.append(f"index range {meta.get('start')}..{meta.get('stop')} rows {arr2.shape[0]} != {start}..{stop}")
    t2 = time.perf_counter()
    val = D.validate_stream(gen, arr2, K, regime, start)
    t_val = time.perf_counter() - t2
    return {"K": K, "regime": regime, "seed": seed, "start": start, "stop": stop, "path": str(npy), "sha256": sha,
            "built": info["built"], "build_seconds": info["build_seconds"], "load_or_build_s": t_build,
            "check_s": t_check, "validation_s": t_val, "validation": val, "problems": problems, "error": None}


def _build_safe(args):
    try:
        return build_one(*args)
    except Exception as exc:  # reported per stream; the command exits nonzero
        cache_dir, K, regime, seed, _ = args
        return {"K": K, "regime": regime, "seed": seed, "error": f"{type(exc).__name__}: {exc}", "problems": []}


def prebuild(cache_dir, streams, workers: int = 4, n_updates: int = FULL_UPDATES) -> dict:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    args = [(str(cache_dir), K, r, s, n_updates) for K, r, s in streams]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_build_safe, args))
    bad = [r for r in recs if r["error"] or r["problems"]]
    return {"cache_dir": str(cache_dir), "workers": workers, "updates": n_updates, "streams": recs,
            "wall_s": time.perf_counter() - t0, "n_failed": len(bad)}


def print_prebuild(rec: dict):
    for r in rec["streams"]:
        tag = f"K{r['K']:<2} {r['regime']:<5} s{r['seed']}"
        if r["error"]:
            print(f"stream {tag}: FAILED {r['error']}")
            continue
        b = r["build_seconds"]
        print(f"stream {tag} [{r['start']}, {r['stop']}): built {r['built']}, build "
              f"{'-' if b is None else f'{b:.1f}'} s, check {r['check_s']:.1f} s, validation {r['validation_s']:.1f} s"
              f" ({r['validation']['sequences_checked']} sequences), sha256 {r['sha256'][:16]}"
              + (f", PROBLEMS {r['problems']}" if r["problems"] else ""))
    print(f"prebuild: {len(rec['streams'])} streams, {rec['n_failed']} failed, {rec['wall_s']:.1f} s with "
          f"{rec['workers']} workers")


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def _limit_args(entry: str, t: dict) -> list:
    keys = {"task2.train_mqar": (("limit_updates", "--limit-updates"), ("limit_eval", "--limit-eval")),
            "verify": (("verify_lengths", "--limit-lengths"),),
            "bench": (("bench_lengths", "--limit-lengths"), ("reps", "--limit-reps")),
            "task2.mqar_sweep": (("sweep_lengths", "--limit-lengths"), ("reps", "--limit-reps"))}[entry]
    out = []
    for k, flag in keys:
        v = t.get(k)
        if v is not None:
            out += [flag, ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)]
    return out


def static_seconds(name: str) -> float:
    """Weight of a main-stage job without a plan: dense > window > local_dilated > linear, Part A verify largest."""
    if name == "part_a_verify":
        return 5000.0
    mech = name[len("train_"):].split("_mixed_")[0].split("_near_")[0]
    return {"dense": 4.0, "window": 3.0, "local_dilated": 2.5, "linear": 2.0}[mech] * 1000.0


def balance(seconds: dict, order: list) -> dict:
    """Longest-processing-time assignment to GPU 0 / GPU 1 (GPU 1 starts STAGGER_S later); ties by list order."""
    pos = {n: i for i, n in enumerate(order)}
    load = {"0": 0.0, "1": STAGGER_S}
    out = {}
    for n in sorted(seconds, key=lambda n: (-seconds[n], pos.get(n, 0))):
        g = min(load, key=lambda k: (load[k], k))
        out[n] = g
        load[g] += seconds[n]
    return out


def stage_jobs(stage: str, results=None, status=None, cache_dir=CACHE_DIR, plan=None, test=None, configs=CONFIGS,
               seeds=SEEDS, python=sys.executable) -> list:
    """Job specs: name, gpu, queue, phase, kind, entry, run fields, out, cmd, wait_for, expect."""
    d = DEFAULTS[stage]
    R = Path(results or d["results"])
    S = Path(status or d["status"])
    t = dict(test or {})
    device = t.pop("device", None)
    pre = d["prefix"]
    jobs = []

    def add(gpu, queue, phase, kind, name, entry_args, out, limits, run=None):
        inner = [python, "-m", *entry_args, "--out", str(out)]
        if device:
            inner += ["--device", device]
        inner += _limit_args({"train": "task2.train_mqar", "sweep": "task2.mqar_sweep"}.get(kind, kind), limits)
        cmd = [python, "-m", "task2.kaggle_nb1", "job", "--name", pre + name, "--status", str(S), "--out", str(out),
               "--", *inner]
        jobs.append({"name": pre + name, "stage": stage, "gpu": str(gpu), "queue": queue, "phase": phase,
                     "kind": kind, "out": str(out), "cmd": cmd, "wait_for": [], "run": run, "expect": dict(limits)})

    def train(gpu, queue, phase, name, mech, regime, K, seed, out, limits):
        args = ["task2.train_mqar", "--mechanism", mech, "--K", str(K), "--regime", regime, "--seed", str(seed),
                "--cache-dir", str(cache_dir)]
        add(gpu, queue, phase, "train", name, args, out, limits,
            run={"mechanism": mech, "regime": regime, "K": K, "seed": seed})

    def lim(**defaults):
        """Stage limits, replaced by test options of the same name."""
        out = {k: v for k, v in defaults.items() if v is not None}
        for k in ("limit_updates", "limit_eval", "verify_lengths", "bench_lengths", "sweep_lengths", "reps"):
            if t.get(k) is not None and (k in defaults):
                out[k] = t[k]
        return out

    if stage == "preflight":
        add("0", "gpu0", "preflight", "verify", "part_a_verify", ["task2.part_a", "verify"], R / "part_a_verify",
            lim(verify_lengths=list(PREFLIGHT_VERIFY_LENGTHS)))
        for g in ("0", "1"):
            for m in MECHANISMS:
                train(g, f"gpu{g}", "preflight", f"train_{m}_gpu{g}", m, SMOKE_REGIME, SMOKE_K, SMOKE_SEED,
                      R / "train" / f"{m}_gpu{g}", lim(limit_updates=PREFLIGHT_UPDATES, limit_eval=PREFLIGHT_EVAL))
        add("1", "gpu1", "preflight", "sweep", "sweep", ["task2.mqar_sweep"], R / "sweep",
            lim(sweep_lengths=list(PREFLIGHT_SWEEP_LENGTHS), reps=list(PREFLIGHT_SWEEP_REPS)))
    elif stage == "smoke":
        for g, mechs in (("0", ("dense", "window")), ("1", ("local_dilated", "linear"))):
            for m in mechs:
                train(g, f"gpu{g}", "smoke", f"train_{m}", m, SMOKE_REGIME, SMOKE_K, SMOKE_SEED, R / "train" / m,
                      lim(limit_updates=SMOKE_UPDATES, limit_eval=t.get("limit_eval")))
        add("0", "gpu0", "smoke", "verify", "part_a_verify", ["task2.part_a", "verify"], R / "part_a_verify",
            lim(verify_lengths=list(SMOKE_VERIFY_LENGTHS)))
    else:
        specs = {}
        for mech, regime, K in configs:
            for s in seeds:
                specs[f"train_{run_name(mech, regime, K, s)}"] = (mech, regime, K, s)
        names = ["part_a_verify", *specs]
        if plan:
            assign = plan["assignment"]
            secs = {r["name"]: r["projected_s"] for r in plan["rows"]}
        else:
            secs = {n: static_seconds(n) for n in names}
            assign = balance(secs, names)
        missing = [n for n in names if n not in assign]
        if missing:
            raise SystemExit(f"plan has no GPU for {missing}")
        for n in sorted(names, key=lambda n: (-secs.get(n, 0.0), names.index(n))):
            g = assign[n]
            if n == "part_a_verify":
                add(g, f"gpu{g}", "eval", "verify", n, ["task2.part_a", "verify"], R / "part_a" / "verify",
                    lim(verify_lengths=None))
            else:
                mech, regime, K, s = specs[n]
                train(g, f"gpu{g}", "eval", n, mech, regime, K, s,
                      R / "mqar" / run_name(mech, regime, K, s), lim(limit_updates=None, limit_eval=None))
        add("0", "timing_gpu0", "timing", "bench", "part_a_bench", ["task2.part_a", "bench"], R / "part_a" / "bench",
            lim(bench_lengths=None, reps=None))
        add("0", "timing_gpu0", "timing", "sweep", "mqar_sweep", ["task2.mqar_sweep"], R / "mqar_sweep",
            lim(sweep_lengths=None, reps=None))
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)
    return jobs


def queues(jobs: list) -> list:
    out = []
    for j in jobs:
        if j["queue"] not in out:
            out.append(j["queue"])
    return out


def write_jobs(stage: str, jobs_dir, jobs: list) -> dict:
    """<stage>_<queue>.json per queue (queue format) and <stage>_meta.json; refuses existing files."""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for q in queues(jobs):
        rows = [{"name": j["name"], "cmd": j["cmd"], "wait_for": j["wait_for"]} for j in jobs if j["queue"] == q]
        paths[q] = jobs_dir / f"{stage}_{q}.json"
        with open(paths[q], "x", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
            fh.write("\n")
    paths["meta"] = jobs_dir / f"{stage}_meta.json"
    with open(paths["meta"], "x", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=2)
        fh.write("\n")
    return paths


def load_meta(jobs_dir, stage: str) -> list:
    return json.loads((Path(jobs_dir) / f"{stage}_meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------
# Job wrapper with concurrency record.

def _pid_alive(pid) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_intervals(status_dir) -> list:
    out = []
    for p in sorted(Path(status_dir).glob("*.interval.json")):
        rec = _load_json(p)
        if isinstance(rec, dict) and "name" in rec:
            out.append(rec)
    return out


def overlapping(intervals: list, name: str, start: float, end: float) -> list:
    """Intervals of other jobs that started before `end` and ended after `start` (or have no end)."""
    out = []
    for r in intervals:
        if r["name"] == name or r.get("start_unix") is None:
            continue
        e = r.get("end_unix")
        if r["start_unix"] < end and (e is None or e > start):
            out.append({"name": r["name"], "gpu": r.get("gpu"), "start": r.get("start"), "end": r.get("end"),
                        "ended_without_record": e is None and not _pid_alive(r.get("pid"))})
    return sorted(out, key=lambda x: x["name"])


def record_concurrency(out, rec: dict) -> bool:
    """Add rec under CONCURRENCY_KEY to <out>/run.json (atomic rewrite); False if there is no readable run.json."""
    path = Path(out) / "run.json"
    run = _load_json(path)
    if not isinstance(run, dict):
        return False
    run[CONCURRENCY_KEY] = rec
    _write_atomic(path, run)
    return True


def run_job(name: str, status_dir, out, cmd: list) -> int:
    from task1.kaggle_run2 import JOB_ENV, status_lines

    status_dir = Path(status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    iv_path = status_dir / f"{name}.interval.json"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    start = time.time()
    iv = {"name": name, "gpu": gpu, "pid": os.getpid(), "start": _now(), "start_unix": start, "end": None,
          "end_unix": None, "exit": None}
    at_start = sorted(r["name"] for r in read_intervals(status_dir)
                      if r["name"] != name and r.get("end_unix") is None and _pid_alive(r.get("pid")))
    _write_atomic(iv_path, iv)
    child = subprocess.Popen(cmd, env={**os.environ, **JOB_ENV})

    def forward(sig, _frame):
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    code = child.wait()
    end = time.time()
    iv.update(end=_now(), end_unix=end, exit=code)
    _write_atomic(iv_path, iv)
    over = overlapping(read_intervals(status_dir), name, start, end)
    rec = {"method": CONCURRENCY_METHOD, "job": name, "gpu": gpu, "start": iv["start"], "end": iv["end"],
           "alive_at_start": at_start, "overlapping": over,
           "overlapping_same_gpu": [o["name"] for o in over if o["gpu"] == gpu],
           "status_dir": str(status_dir)}
    if not record_concurrency(out, rec):
        print(f"[job] no readable {Path(out) / 'run.json'}; concurrency not recorded", flush=True)
    print(f"[job] {name}: concurrent jobs {[o['name'] for o in over] or 'none'}", flush=True)
    if code != 0:
        for line in status_lines(out, code):
            print(line, flush=True)
    sys.stdout.flush()
    if code < 0:
        signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    return code


# ---------------------------------------------------------------------------------------------------------------
# Reading run records.

def _run(j) -> dict:
    return _load_json(Path(j["out"]) / "run.json") or {}


def _peak_rows(rows) -> int:
    return max([r.get("peak_allocated_bytes") or 0 for r in rows] or [0])


def train_facts(run: dict) -> dict:
    et = run.get("eval_timings_s") or {}
    dev = list((et.get("dev_checkpoints") or {}).values())
    stream = run.get("stream") or {}
    val = (stream.get("validation") or {}).get("seconds") or 0.0
    elapsed = run.get("elapsed_wall_s") or 0.0
    fixed = elapsed - (run.get("train_wall_s") or 0.0) - (run.get("eval_wall_s") or 0.0) - val - \
        (stream.get("load_seconds") or 0.0)
    return {"ms_per_update": run.get("mean_ms_per_update"), "dev_eval_s": (sum(dev) / len(dev)) if dev else None,
            "test_eval_s": dict(et.get("test") or {}), "fixed_s": max(0.0, fixed), "elapsed_s": elapsed,
            "peak_train": run.get("peak_allocated_bytes_train"), "peak_eval": run.get("peak_allocated_bytes_eval")}


def _status(status_dir, name) -> dict:
    p = Path(status_dir) / f"{name}.exit"
    try:
        code = int(p.read_text().strip())
    except (FileNotFoundError, ValueError):
        code = None
    rec = _load_json(Path(status_dir) / f"{name}.json") or {}
    return {"exit": code, "done": (Path(status_dir) / f"{name}.done").exists(), "wall_s": rec.get("wall_s")}


# ---------------------------------------------------------------------------------------------------------------
# Preflight check and smoke report.

def preflight_check(meta: list) -> list:
    problems = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            problems.append(f"{j['name']}: run status {run.get('status')!r}")
            continue
        if j["kind"] == "train":
            f = train_facts(run)
            print(f"{j['name']:<28} params {run.get('parameter_count')}, {f['ms_per_update']:.1f} ms/update, peak "
                  f"train {_gib(f['peak_train']) or 0:.2f} GiB, eval {_gib(f['peak_eval']) or 0:.2f} GiB")
            if run.get("parameter_count") != PARAMS:
                problems.append(f"{j['name']}: parameter count {run.get('parameter_count')}")
        elif j["kind"] == "verify":
            rows = _jsonl(Path(j["out"]) / "verify_rows.jsonl")
            big = max(j["expect"].get("verify_lengths") or FULL_LENGTHS)
            acc = [r for r in rows if r["kind"] == "accuracy" and r["L"] == big]
            for r in acc:
                print(f"  verify L {big} {r['dtype']} {r['label']:<26} {r['status']:<17} peak "
                      f"{_gib(r.get('peak_allocated_bytes')) or 0:.2f} GiB, {r.get('wall_s') or 0:.1f} s")
            d32 = [r for r in acc if r["impl"] == "dense" and r["dtype"] == "fp32"]
            if len(d32) != 1 or d32[0]["status"] != "ok":
                problems.append(f"{j['name']}: dense FP32 at L {big} not ok: {[r['status'] for r in d32]}")
        elif j["kind"] == "sweep":
            rows = _jsonl(Path(j["out"]) / "sweep_rows.jsonl")
            for r in rows:
                print(f"  sweep T {r['T']} {r['mechanism']:<14} {r['status']:<11} "
                      f"{r.get('median_ms') or float('nan'):.2f} ms, peak {_gib(r.get('peak_allocated_bytes')) or 0:.2f} GiB")
            for m in ("local_dilated", "linear"):
                st = [r["status"] for r in rows if r["mechanism"] == m]
                if not st or any(s != "ok" for s in st):
                    problems.append(f"{j['name']}: {m} sweep statuses {st}")
    return problems


def smoke_report(meta: list) -> list:
    problems = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            problems.append(f"{j['name']}: run status {run.get('status')!r}")
            continue
        if j["kind"] == "train":
            f = train_facts(run)
            n = run.get("n_rows") or {}
            tq = {c: s["n_queries"] for c, s in (run.get("test_summary") or {}).get("native", {}).items()}
            print(f"{j['name']}: {run.get('updates_done')} updates, {f['ms_per_update']:.2f} ms/update, dev "
                  f"checkpoint eval {f['dev_eval_s']:.1f} s (x{len(run['eval_timings_s']['dev_checkpoints'])}), test "
                  f"eval {json.dumps({k: round(v, 1) for k, v in f['test_eval_s'].items()})} s, fixed "
                  f"{f['fixed_s']:.1f} s, stream validation {run['stream']['validation']['seconds']:.1f} s for "
                  f"{run['stream']['validation']['sequences_checked']} sequences")
            print(f"    rows {json.dumps(n)}; test queries per cell {json.dumps(tq)}; peak train "
                  f"{_gib(f['peak_train']) or 0:.2f} GiB, eval {_gib(f['peak_eval']) or 0:.2f} GiB")
        elif j["kind"] == "verify":
            rows = _jsonl(Path(j["out"]) / "verify_rows.jsonl")
            print(f"{j['name']}: {len(rows)} rows, status counts {json.dumps(run.get('status_counts'))}, "
                  f"{run.get('elapsed_wall_s', 0):.1f} s, max peak {_gib(_peak_rows(rows)):.2f} GiB")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# Projection.

def _by(meta, kind, mech=None):
    return [j for j in meta if j["kind"] == kind and (mech is None or (j["run"] or {}).get("mechanism") == mech)]


def _acc_seconds(rows, L, labels=None) -> float:
    return sum(r.get("wall_s") or 0.0 for r in rows if r.get("kind") == "accuracy" and r["L"] == L
               and (labels is None or r["label"] in labels))


def project(jobs_dir, prebuild_rec: dict, budget_h: float = 10.5, session_h: float = 11.5, elapsed_s: float = 0.0,
            vram_gib: float = 14.0, safety: float = SAFETY, configs=CONFIGS, seeds=SEEDS) -> dict:
    """Per-job projection of the main stage from the smoke and preflight records (see notes in the result)."""
    pf, sm = load_meta(jobs_dir, "preflight"), load_meta(jobs_dir, "smoke")
    for j in pf + sm:
        if _run(j).get("status") != "ok":
            raise SystemExit(f"{j['name']}: run not finished ok")
    facts = {j["run"]["mechanism"]: train_facts(_run(j)) for j in _by(sm, "train")}
    val_s = {}
    for r in prebuild_rec["streams"]:
        if r.get("error"):
            raise SystemExit(f"prebuild failed for K{r['K']} {r['regime']} s{r['seed']}")
        key = (r["K"], r["regime"])
        val_s[key] = max(val_s.get(key, 0.0), r["validation_s"] + r["check_s"])
    missing = sorted({(K, regime) for _, regime, K in configs} - set(val_s))
    if missing:
        raise SystemExit(f"prebuild record has no stream for (K, regime) {missing}")
    rows = []
    for mech, regime, K in configs:
        f = facts[mech]
        for s in seeds:
            parts = {"train": f["ms_per_update"] * FULL_UPDATES / 1000.0, "dev_eval": N_CHECKPOINTS * f["dev_eval_s"],
                     "test_eval": sum(f["test_eval_s"].values()), "fixed": f["fixed_s"],
                     "stream_check_and_validation": val_s[(K, regime)]}
            rows.append({"name": f"train_{run_name(mech, regime, K, s)}", "phase": "eval", "parts": parts,
                         "projected_s": safety * sum(parts.values())})
    vpf = _by(pf, "verify")[0]
    vsm = _by(sm, "verify")[0]
    e_pf, e_sm = _run(vpf)["elapsed_wall_s"], _run(vsm)["elapsed_wall_s"]
    rows.append({"name": "part_a_verify", "phase": "eval",
                 "parts": {"preflight_256_8192_x2": 2 * e_pf, "smoke_256_2048_x3": 3 * e_sm},
                 "projected_s": safety * (2 * e_pf + 3 * e_sm)})
    # Timing phase: bench per L <= 13 repetitions x (forward + forward-backward <= 2 x the verify forward-backward
    # wall) summed over the bench implementations and both dtypes; 512 and 1024 bounded by 2048, 4096 by 8192.
    from . import part_a as P
    rows_pf, rows_sm = _jsonl(Path(vpf["out"]) / "verify_rows.jsonl"), _jsonl(Path(vsm["out"]) / "verify_rows.jsonl")
    bench_labels = {s["label"] for s in P.bench_specs()}
    big_pf, big_sm = max(vpf["expect"]["verify_lengths"]), max(vsm["expect"]["verify_lengths"])
    small = min(vsm["expect"]["verify_lengths"])
    t256 = _acc_seconds(rows_sm, small, bench_labels)
    t2048 = _acc_seconds(rows_sm, big_sm, bench_labels)
    t8192 = _acc_seconds(rows_pf, big_pf, bench_labels)
    bench_s = BENCH_REPS * 2 * (t256 + 3 * t2048 + 2 * t8192) * 1.1
    sw = _jsonl(Path(_by(pf, "sweep")[0]["out"]) / "sweep_rows.jsonl")
    ms8192 = sum((r.get("max_ms") or 0.0) for r in sw)
    sweep_s = len(SWEEP_LENGTHS) * BENCH_REPS * ms8192 / 1000.0 + 5.0 * len(SWEEP_LENGTHS) * len(MECHANISMS)
    timing = {"part_a_bench": safety * bench_s, "mqar_sweep": safety * sweep_s}
    names = [r["name"] for r in rows]
    secs = {r["name"]: r["projected_s"] for r in rows}
    assign = balance(secs, names)
    q = {"0": 0.0, "1": STAGGER_S}
    for n, g in assign.items():
        q[g] += secs[n]
    eval_end = max(q.values())
    total = eval_end + sum(timing.values())
    peaks = {}
    for mech, f in facts.items():
        peaks[f"smoke train {mech}"] = f["peak_train"] or 0
        peaks[f"smoke eval {mech}"] = f["peak_eval"] or 0
    for j in _by(pf, "train"):
        f = train_facts(_run(j))
        peaks[f"preflight train {j['name']}"] = max(f["peak_train"] or 0, f["peak_eval"] or 0)
    peaks["preflight sweep T 8192"] = _peak_rows(sw)
    part_a_peaks = {"preflight verify": _peak_rows(rows_pf), "smoke verify": _peak_rows(rows_sm)}
    limit = vram_gib * GiB
    reasons = []
    if q["0"] > budget_h * 3600 or q["1"] > budget_h * 3600:
        reasons.append(f"evaluation queue above {budget_h} h: gpu0 {q['0'] / 3600:.2f} h, gpu1 {q['1'] / 3600:.2f} h")
    if total > budget_h * 3600:
        reasons.append(f"evaluation phase plus timing phase {total / 3600:.2f} h above {budget_h} h")
    if elapsed_s + total > session_h * 3600:
        reasons.append(f"elapsed {elapsed_s / 3600:.2f} h + projected {total / 3600:.2f} h above {session_h} h")
    reasons += [f"peak {k} {v / GiB:.2f} GiB above {vram_gib} GiB" for k, v in peaks.items() if v > limit]
    return {"rows": rows, "assignment": assign, "queue_s": {f"gpu{k}": v for k, v in q.items()},
            "timing_s": timing, "eval_phase_s": eval_end, "total_s": total, "elapsed_s": elapsed_s,
            "budget_h": budget_h, "session_h": session_h, "safety": safety, "peaks_bytes": peaks,
            "part_a_peaks_bytes": part_a_peaks, "vram_limit_gib": vram_gib,
            "part_a_peak_note": "Part A rows that do not fit are recorded as oom rows by protocol A3; their peaks are "
                                "printed, not used to stop",
            "smoke_facts": facts, "over": bool(reasons), "reasons": reasons}


def print_plan(p: dict):
    by_gpu = {"0": [], "1": []}
    for r in p["rows"]:
        by_gpu[p["assignment"][r["name"]]].append(r)
    for g, rs in by_gpu.items():
        print(f"GPU {g}: {len(rs)} jobs, {p['queue_s'][f'gpu{g}'] / 3600:.2f} h (incl. start offset)")
        for r in sorted(rs, key=lambda r: -r["projected_s"]):
            print(f"    {r['name']:<34}{r['projected_s'] / 60:>8.1f} min  " +
                  ", ".join(f"{k} {v / 60:.1f}" for k, v in r["parts"].items()))
    for k, v in p["timing_s"].items():
        print(f"timing phase (GPU 0 alone) {k}: {v / 60:.1f} min")
    print(f"evaluation phase {p['eval_phase_s'] / 3600:.2f} h, with timing phase {p['total_s'] / 3600:.2f} h; elapsed "
          f"so far {p['elapsed_s'] / 3600:.2f} h; budget {p['budget_h']} h, session {p['session_h']} h; safety "
          f"factor {p['safety']}")
    print("peaks (GiB): " + ", ".join(f"{k} {v / GiB:.2f}" for k, v in p["peaks_bytes"].items()))
    print("Part A peaks (GiB, not used to stop): " +
          ", ".join(f"{k} {v / GiB:.2f}" for k, v in p["part_a_peaks_bytes"].items()))
    for x in p["reasons"]:
        print("OVER:", x)
    print("plan: within limits" if not p["over"] else "plan: OVER LIMIT, queues not started")


# ---------------------------------------------------------------------------------------------------------------
# Expected counts, check.

def n_eval(limit, full) -> int:
    return full if limit is None else min(int(limit), full)


def expected_part_a(mode: str, lengths, n_backends: int) -> dict:
    from . import part_a as P
    nv = 2 + len(P.VERIFY_TILES) + n_backends  # dense, tiles, sdpa default, forced backends
    nd = len(P.DTYPES)
    if mode == "verify":
        edge_L = P.EDGE_L if P.EDGE_L in lengths else max(lengths)
        n_caus = nd * (2 + len(P.VERIFY_TILES)) * len(P.edge_ts(edge_L))
        return {"rows": len(lengths) * (nd + nd * nv) + n_caus + len(P.EXTREME_TARGETS) * nd * nv + nd * nv}
    nb = 3 + n_backends
    return {"rows": len(lengths) * nb * nd * len(P.PASSES), "backend_table": len(lengths) * nd * len(P.PASSES)}


def _check_train(j, run, out: Path, problems: list):
    from . import query_npz as Q
    n = j["name"]
    ex, r = j["expect"], j["run"]
    updates = ex.get("limit_updates") or FULL_UPDATES
    n_dev, n_test = n_eval(ex.get("limit_eval"), DEV_PER_CELL), n_eval(ex.get("limit_eval"), TEST_PER_CELL)
    cfg = run.get("config", {})
    if (cfg.get("mechanism"), cfg.get("regime"), cfg.get("K"), cfg.get("seed")) != \
            (r["mechanism"], r["regime"], r["K"], r["seed"]):
        problems.append(f"{n}: config run fields differ from the job")
    if run.get("parameter_count") != PARAMS:
        problems.append(f"{n}: parameter count {run.get('parameter_count')} != {PARAMS}")
    if run.get("updates_done") != updates:
        problems.append(f"{n}: updates_done {run.get('updates_done')} != {updates}")
    opens = run.get("test_file_opens")
    if not isinstance(opens, list) or len(opens) != 1 or opens[0].get("after_update") != updates or \
            run.get("test_opened_after_update") != updates:
        problems.append(f"{n}: test_file_opens {opens} not one opening after update {updates}")
    dev_opens = run.get("dev_file_opens")
    if not isinstance(dev_opens, list) or [o.get("after_update") for o in dev_opens] != [0]:
        problems.append(f"{n}: dev_file_opens {dev_opens}")
    conds = ["native", "window64"] if r["mechanism"] == "dense" else ["native"]
    ts = run.get("test_summary") or {}
    if sorted(ts) != sorted(conds):
        problems.append(f"{n}: test conditions {sorted(ts)} != {conds}")
    for cond in conds:
        for K in KS:
            for reg in EVAL_REGIMES:
                c = (ts.get(cond) or {}).get(f"{reg}_K{K}") or {}
                if c.get("n_queries") != n_test * K or c.get("n_sequences") != n_test:
                    problems.append(f"{n}: test {cond} {reg}_K{K} queries {c.get('n_queries')} != {n_test * K}")
    if not isinstance(run.get("learned"), dict) or "learned" not in run["learned"]:
        problems.append(f"{n}: learned flag missing")
    sum_k = len(EVAL_REGIMES) * sum(KS)
    for key, fname, want in (("dev_queries", "dev_queries_final.npz", n_dev * sum_k),
                             ("test_queries", "test_queries.npz", n_test * sum_k * len(conds))):
        p = out / fname
        try:
            arr = Q.read_query_npz(p)
            if arr["split"].size != want or (run.get("n_rows") or {}).get(key) != want:
                problems.append(f"{n}: {fname} rows {arr['split'].size} (run {run.get('n_rows', {}).get(key)}) != {want}")
        except Exception as exc:
            problems.append(f"{n}: {fname} unreadable with allow_pickle=False: {type(exc).__name__}: {exc}")
    ck = out / "model_final.safetensors"
    if not ck.is_file():
        problems.append(f"{n}: checkpoint missing")
    else:
        from common.io import sha256_file
        if sha256_file(ck) != run.get("checkpoint_sha256"):
            problems.append(f"{n}: checkpoint sha256 differs from run.json")
    for fname, want in (("train_log.jsonl", updates), ("dev_curve.jsonl", None)):
        p = out / fname
        if not p.is_file():
            problems.append(f"{n}: {fname} missing")
        elif want is not None and sum(1 for _ in open(p)) != want:
            problems.append(f"{n}: {fname} rows != {want}")


def _check_part_a(j, run, out: Path, problems: list):
    n, mode = j["name"], j["kind"]
    from . import part_a as P
    if mode == "sweep":
        from . import mqar_sweep as SW
        lengths = j["expect"].get("sweep_lengths") or list(SW.LENGTHS)
        rows = _jsonl(out / "sweep_rows.jsonl") if (out / "sweep_rows.jsonl").is_file() else []
        if len(rows) != len(lengths) * len(MECHANISMS):
            problems.append(f"{n}: sweep rows {len(rows)} != {len(lengths) * len(MECHANISMS)}")
        return
    lengths = j["expect"].get(f"{mode}_lengths") or list(P.LENGTHS)
    nb = len((run.get("environment") or {}).get("sdpa_backends_enumerated") or [])
    want = expected_part_a(mode, lengths, nb)
    fname = "verify_rows.jsonl" if mode == "verify" else "bench_rows.jsonl"
    rows = _jsonl(out / fname) if (out / fname).is_file() else []
    if len(rows) != want["rows"] or run.get("n_rows") != want["rows"]:
        problems.append(f"{n}: {fname} rows {len(rows)} (run {run.get('n_rows')}) != {want['rows']} "
                        f"({nb} SDPA backends)")
    if mode == "bench":
        bt = _jsonl(out / "backend_table.jsonl") if (out / "backend_table.jsonl").is_file() else []
        if len(bt) != want["backend_table"]:
            problems.append(f"{n}: backend_table rows {len(bt)} != {want['backend_table']}")
    if run.get("config", {}).get("lengths") != list(lengths):
        problems.append(f"{n}: lengths {run.get('config', {}).get('lengths')} != {list(lengths)}")


def check(meta: list, status_dir, logs_dir) -> list:
    problems = []
    for j in meta:
        n, out = j["name"], Path(j["out"])
        st = _status(status_dir, n)
        if st["exit"] != 0 or not st["done"]:
            problems.append(f"{n}: exit {st['exit']}, done {st['done']}")
        run = _run(j)
        if run.get("status") != "ok" or run.get("step") != "done":
            problems.append(f"{n}: run status {run.get('status')!r}, step {run.get('step')!r}")
            continue
        limited = any(v is not None for v in j["expect"].values())
        if bool(run.get("config", {}).get("limited")) != limited:
            problems.append(f"{n}: config.limited {run.get('config', {}).get('limited')} != {limited}")
        conc = run.get(CONCURRENCY_KEY)
        if not isinstance(conc, dict) or "overlapping" not in conc:
            problems.append(f"{n}: {CONCURRENCY_KEY} missing")
        elif j["phase"] == "timing" and conc["overlapping"]:
            problems.append(f"{n}: timing job overlapped {[o['name'] for o in conc['overlapping']]}")
        if j["kind"] == "train":
            _check_train(j, run, out, problems)
        else:
            _check_part_a(j, run, out, problems)
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


def _dev_lt64(run) -> float:
    return (run.get("learned") or {}).get("accuracy")


def _test_regime(run, cond, regime):
    """Accuracy over every test query of one regime (all K pooled) under one mask condition."""
    ts = (run.get("test_summary") or {}).get(cond) or {}
    n = sum((ts.get(f"{regime}_K{K}") or {}).get("n_queries") or 0 for K in KS)
    if not n:
        return None
    return sum(((ts.get(f"{regime}_K{K}") or {}).get("accuracy") or 0.0) *
               ((ts.get(f"{regime}_K{K}") or {}).get("n_queries") or 0) for K in KS) / n


def summary_lines(meta: list, max_lines: int = 60) -> list:
    L = []
    train = [j for j in meta if j["kind"] == "train"]
    L.append("run                            L dev<64  near   far    mixed  ms/upd | w64 near  far    mixed")
    for j in train:
        run = _run(j)
        nm = j["name"][len("train_"):] if j["name"].startswith("train_") else j["name"]
        if run.get("status") != "ok":
            L.append(f"{nm:<30} status {run.get('status')!r}")
            continue
        lf = "Y" if (run.get("learned") or {}).get("learned") else "n"
        s = (f"{nm:<30} {lf} {_a(_dev_lt64(run))}  " + " ".join(_a(_test_regime(run, 'native', r)) + " "
             for r in EVAL_REGIMES) + f"{run.get('mean_ms_per_update') or 0:6.2f}")
        if j["run"]["mechanism"] == "dense":
            s += " | " + " ".join(_a(_test_regime(run, "window64", r)) for r in EVAL_REGIMES)
        L.append(s)
    tail = []
    bench = [j for j in meta if j["kind"] == "bench"]
    for j in bench:
        p = Path(j["out"]) / "backend_table.jsonl"
        if not p.is_file():
            tail.append("Part A bench: no backend table")
            continue
        bt = _jsonl(p)
        for dt in ("fp16", "fp32"):
            back = sorted({(r.get("default") or {}).get("inferred_backend") for r in bt if r["dtype"] == dt} - {None})
            kern = sorted({k for r in bt if r["dtype"] == dt for k in (r.get("default") or {}).get("cuda_kernels", [])
                           if "attention" in k.lower() or "fmha" in k.lower() or "flash" in k.lower()
                           or "sdp" in k.lower()})
            tail.append(f"Part A default SDPA {dt}: backend {back}; kernels {kern}"[:230])
    groups = {}
    for j in [x for x in meta if x["kind"] in ("verify", "bench")]:
        fname = "verify_rows.jsonl" if j["kind"] == "verify" else "bench_rows.jsonl"
        p = Path(j["out"]) / fname
        if not p.is_file():
            continue
        for r in _jsonl(p):
            if r.get("status") in ("unsupported", "oom", "skipped_projected", "no_reference"):
                g = groups.setdefault((r["label"], r["status"], r["dtype"]), {"kinds": set(), "L": set()})
                g["kinds"].add(f"{j['kind']}:{r.get('pass') or r.get('kind')}")
                g["L"].add(r["L"])
    glines = [f"Part A {k[0]} {k[2]} {k[1]}: L {sorted(v['L'])} in {sorted(v['kinds'])}"
              for k, v in sorted(groups.items())]
    sweep = [j for j in meta if j["kind"] == "sweep"]
    slines = []
    for j in sweep:
        p = Path(j["out"]) / "sweep_rows.jsonl"
        rows = _jsonl(p) if p.is_file() else []
        T = max([r["T"] for r in rows] or [0])
        slines.append(f"sweep T {T}: " + "; ".join(
            f"{r['mechanism']} {r['status']} {r.get('median_ms') or float('nan'):.2f} ms "
            f"{_gib(r.get('peak_allocated_bytes')) or 0:.2f} GiB" for r in rows if r["T"] == T))
    room = max_lines - len(L) - len(tail) - len(slines) - 1
    if len(glines) > room:
        glines = glines[:max(0, room - 1)] + [f"... {len(glines) - max(0, room - 1)} more Part A status groups"]
    out = L + tail + (glines or ["Part A: no unsupported, oom, skipped or no-reference rows"]) + slines
    return out[:max_lines]


# ---------------------------------------------------------------------------------------------------------------
# Idle check, zip, final.

def idle_check(meta_eval: list, status_dir) -> list:
    """Problems if any evaluation job is not final or any queue or job process of the status directory is alive."""
    from common import kaggle_queue as KQ
    problems = []
    status_dir = Path(status_dir)
    for j in meta_eval:
        st, _ = KQ.job_state(status_dir, j["name"])
        if st not in KQ.FINAL_STATES:
            problems.append(f"{j['name']}: state {st}")
    if KQ._queues_alive(status_dir):
        problems.append("a queue process is alive")
    for r in read_intervals(status_dir):
        if r.get("end_unix") is None and _pid_alive(r.get("pid")):
            problems.append(f"job process alive: {r['name']} (pid {r.get('pid')})")
    return problems


ZIP_EXCLUDE = ("smoke", "preflight", "cache")


def make_zip(out, roots) -> dict:
    """Zip every file under the roots except paths (relative to their root) containing an excluded word; a missing
    root is listed under "missing"."""
    import hashlib
    import zipfile
    out = Path(out)
    names, missing = [], []
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as zf:
        for root in map(Path, roots):
            if not root.is_dir():
                missing.append(str(root))
                continue
            for f in sorted(root.rglob("*")):
                rel = f.relative_to(root).as_posix()
                if f.is_file() and not any(w in rel for w in ZIP_EXCLUDE):
                    zf.write(f, f.as_posix())
                    names.append(f.as_posix())
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": h.hexdigest(), "n_files": len(names),
            "missing": missing}


def final(meta: list, status_dir, check_report, zip_path) -> list:
    from task1.kaggle_run1 import final as fin
    return fin(meta, status_dir, check_report, zip_path)


# ---------------------------------------------------------------------------------------------------------------

def _ints(s):
    return None if s is None else [int(x) for x in s.split(",")]


def _configs(names):
    if not names:
        return CONFIGS
    by = {config_name(*c): c for c in CONFIGS}
    bad = [n for n in names if n not in by]
    if bad:
        raise SystemExit(f"unknown configurations {bad}; known {list(by)}")
    return tuple(by[n] for n in names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    dc = sub.add_parser("data-check")
    dc.add_argument("--out", default=None)
    pb = sub.add_parser("prebuild")
    pb.add_argument("--cache-dir", required=True)
    pb.add_argument("--workers", type=int, default=4)
    pb.add_argument("--out", default=None)
    pb.add_argument("--limit-updates", type=int, default=None, help="tests only")
    pb.add_argument("--streams", nargs="*", default=None, help="K:REGIME:SEED (tests only)")
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    j.add_argument("--plan", default=None)
    j.add_argument("--results", default=None)
    j.add_argument("--status", default=None)
    j.add_argument("--cache-dir", default=CACHE_DIR)
    j.add_argument("--configs", nargs="*", default=None)
    j.add_argument("--seeds", nargs="*", type=int, default=None)
    j.add_argument("--device", default=None)
    j.add_argument("--limit-updates", type=int, default=None)
    j.add_argument("--limit-eval", type=int, default=None)
    for k in ("verify-lengths", "bench-lengths", "sweep-lengths", "reps"):
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
        return run_job(a.name, a.status, a.out, cmd)
    if a.cmd == "data-check":
        rec = data_check()
        print(json.dumps({"generator": rec["generator"], "files": rec["files"],
                          "heldout_list": {k: rec["heldout_list"][k] for k in
                                           ("path", "sha256", "n_sequences", "files", "tracked", "clean")}}, indent=2))
        for x in rec["problems"]:
            print("PROBLEM:", x)
        if a.out:
            _write_new(a.out, rec)
        print("data-check:", "ok" if not rec["problems"] else f"{len(rec['problems'])} problems")
        return 1 if rec["problems"] else 0
    if a.cmd == "prebuild":
        if a.streams:
            streams = [(int(k), r, int(s)) for k, r, s in (x.split(":") for x in a.streams)]
        else:
            streams = streams_needed()
        rec = prebuild(a.cache_dir, streams, a.workers, a.limit_updates or FULL_UPDATES)
        print_prebuild(rec)
        if a.out:
            _write_new(a.out, rec)
        return 1 if rec["n_failed"] else 0
    if a.cmd == "jobs":
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
        test = {"device": a.device, "limit_updates": a.limit_updates, "limit_eval": a.limit_eval,
                "verify_lengths": _ints(a.verify_lengths), "bench_lengths": _ints(a.bench_lengths),
                "sweep_lengths": _ints(a.sweep_lengths), "reps": _ints(a.reps)}
        test = {k: v for k, v in test.items() if v is not None}
        jobs = stage_jobs(a.stage, a.results, a.status, a.cache_dir, plan, test, _configs(a.configs),
                          tuple(a.seeds) if a.seeds else SEEDS)
        paths = write_jobs(a.stage, a.jobs_dir, jobs)
        for q in queues(jobs):
            print(f"{q}: " + ", ".join(x["name"] for x in jobs if x["queue"] == q))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd in ("preflight-check", "smoke-report"):
        stage = "preflight" if a.cmd == "preflight-check" else "smoke"
        meta = load_meta(a.jobs_dir, stage)
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
    meta = load_meta(a.jobs_dir, "main")
    status = a.status or DEFAULTS["main"]["status"]
    if a.cmd == "idle-check":
        problems = idle_check([x for x in meta if x["phase"] == "eval"], status)
        for x in problems:
            print("NOT IDLE:", x)
        print("idle-check:", "no job or queue alive; every evaluation job final" if not problems else "NOT IDLE")
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
        reasons = final(meta, status, a.check_report, a.zip)
        for x in reasons:
            print("NOT OK:", x)
        print("notebook complete: every job ok, check clean, zip written" if not reasons else
              f"notebook not complete: {len(reasons)} reasons")
        return 1 if reasons else 0
    res = make_zip(a.out, a.include)
    for x in res["missing"]:
        print("not in the zip (missing):", x)
    print(f"{res['path']}: {res['n_files']} files, {res['bytes']} bytes, sha256 {res['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
