"""Task 2 Kaggle notebook 3: the protocol AM4 diagnostic (python -m task2.kv_diag) on the 18 diagnostic prompts.

python -m task2.kaggle_nb3 jobs --stage preflight|main --jobs-dir DIR [--plan FILE] [--results DIR] [--status DIR]
       [--device DEV --model-path DIR --limit-prompts SPEC]   (test options)
python -m task2.kaggle_nb3 job --name NAME --status DIR --out DIR -- CMD ...
python -m task2.kaggle_nb3 preflight-check --jobs-dir DIR
python -m task2.kaggle_nb3 plan --jobs-dir DIR [--budget-h 10.5] [--session-h 11.5] [--elapsed-s S] [--vram-gib 14]
       [--out FILE]
python -m task2.kaggle_nb3 summary|check|zip|final --jobs-dir DIR ...

Inventory. main: one job on GPU 0, python -m task2.kv_diag on the 18 diagnostic prompts (the lowest-id prompt of each
grid cell) to results/task2/part_c/diag/. preflight (outputs under /kaggle/tmp/preflight): the same entry point on one
prompt, the 4k-level case with the longest user text (ties by id), the largest prompt; it also serves as the smoke run
(timing, peak VRAM, row and file counts of the real code path on the worst-case prompt), and the plan projects the main
job from it. Every job runs through `job` (task2.kaggle_nb1.run_job), which records in run.json the other jobs that ran
at the same time (concurrent_jobs).
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import kaggle_nb1 as N1
from . import kaggle_nb2 as N2

GiB = N1.GiB
SAFETY = N1.SAFETY
N_PROMPTS = 18
DEFAULTS = {
    "preflight": {"results": "/kaggle/tmp/preflight", "status": "notes/kaggle_status_smoke",
                  "logs": "notes/kaggle_logs_smoke", "prefix": "pf_"},
    "main": {"results": "results/task2/part_c", "status": "notes/kaggle_status", "logs": "notes/kaggle_logs",
             "prefix": ""},
}
STAGES = tuple(DEFAULTS)
DONE_PATTERN = "kv_diag done: "
EVAL_ROWS = "results/task2/part_c/eval/eval_rows.jsonl.gz"
REPRODUCED = ("fp16", "int4_token", "int4_channel_k")
TEST_KEYS = ("limit_prompts", "model_path")
ROW_FILES = {"rows": "control_rows.jsonl.gz", "prompts": "prompt_rows.jsonl.gz"}

_load_json, _write_new, _gib = N1._load_json, N1._write_new, N1._gib


def _rows_gz(path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def stage_jobs(stage: str, results=None, status=None, test=None, python=sys.executable, prompt_id=None) -> list:
    """Job specs: name, stage, gpu, queue, phase, kind, out, cmd, wait_for, run, expect."""
    d = DEFAULTS[stage]
    R = Path(results or d["results"])
    S = Path(status or d["status"])
    t = dict(test or {})
    device = t.pop("device", None)
    limits = {}
    if stage == "preflight":
        limits["limit_prompts"] = prompt_id or N2.longest_case_ids()[4096]
    for k in TEST_KEYS:
        if t.get(k) is not None:
            limits[k] = t[k]
    name = d["prefix"] + "kv_diag"
    out = R / "kv_diag" if stage == "preflight" else R / "diag"
    inner = [python, "-m", "task2.kv_diag", "--out", str(out)]
    if device:
        inner += ["--device", device]
    for k, flag in (("limit_prompts", "--limit-prompts"), ("model_path", "--model-path")):
        if limits.get(k) is not None:
            inner += [flag, str(limits[k])]
    cmd = [python, "-m", "task2.kaggle_nb3", "job", "--name", name, "--status", str(S), "--out", str(out), "--",
           *inner]
    return [{"name": name, "stage": stage, "gpu": "0", "queue": "gpu0", "phase": stage, "kind": "diag",
             "out": str(out), "cmd": cmd, "wait_for": [], "run": {"entry": "task2.kv_diag"}, "expect": limits}]


def _run(j) -> dict:
    return N1._run(j)


def _one(meta) -> dict:
    if len(meta) != 1:
        raise SystemExit(f"expected one job, found {len(meta)}")
    return meta[0]


# ---------------------------------------------------------------------------------------------------------------
# Preflight check and plan.

def preflight_check(meta: list) -> list:
    """Timing, peak VRAM and counts of the preflight run (no score is printed or used)."""
    problems = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            problems.append(f"{j['name']}: run status {run.get('status')!r}")
            continue
        tm = run.get("timing") or {}
        print(f"{j['name']}: load {tm.get('load_s') or 0:.1f} s, per prompt "
              f"{json.dumps({k: round(v, 1) for k, v in (tm.get('prompt_s') or {}).items()})} s, peak "
              f"{_gib(N2.kv_peak(run)) or 0:.2f} GiB, kv_shape {json.dumps(run.get('kv_shape'))}")
        problems += [p for p in _check_outputs(j, run, Path(j["out"]))]
        if not problems:
            rows = _rows_gz(Path(j["out"]) / ROW_FILES["rows"])
            for r in rows:
                print(f"    {r['id']} {r['config']:<15} generated {r['n_generated']:>2} ({r['stop']}), tail rebuilds "
                      f"{r['tail_rebuilds']}, peak {_gib(r.get('peak_allocated_bytes')) or 0:.2f} GiB, seconds "
                      f"{json.dumps({k: round(v, 3) for k, v in r['seconds'].items()})}")
    return problems


def project(jobs_dir, budget_h: float = 10.5, session_h: float = 11.5, elapsed_s: float = 0.0,
            vram_gib: float = 14.0, safety: float = SAFETY, n_prompts: int = N_PROMPTS) -> dict:
    """kv_diag = load + n_prompts x the preflight per-prompt time (the longest 4k prompt), times the safety factor."""
    run = _run(_one(N1.load_meta(jobs_dir, "preflight")))
    if run.get("status") != "ok":
        raise SystemExit("preflight run not finished ok")
    tm = run["timing"]
    prompt_s = max(tm["prompt_s"].values())
    parts = {"load": tm["load_s"], "prompts": n_prompts * prompt_s}
    total = safety * sum(parts.values())
    peak = N2.kv_peak(run)
    reasons = []
    if total > budget_h * 3600:
        reasons.append(f"projected {total / 3600:.2f} h above {budget_h} h")
    if elapsed_s + total > session_h * 3600:
        reasons.append(f"elapsed {elapsed_s / 3600:.2f} h + projected {total / 3600:.2f} h above {session_h} h")
    if peak > vram_gib * GiB:
        reasons.append(f"preflight peak {peak / GiB:.2f} GiB above {vram_gib} GiB")
    return {"rows": [{"name": "kv_diag", "gpu": "0", "parts": parts, "projected_s": total}], "total_s": total,
            "prompt_s": prompt_s, "n_prompts": n_prompts, "elapsed_s": elapsed_s, "budget_h": budget_h,
            "session_h": session_h, "safety": safety, "peak_bytes": peak, "vram_limit_gib": vram_gib,
            "over": bool(reasons), "reasons": reasons}


def print_plan(p: dict):
    for r in p["rows"]:
        print(f"GPU {r['gpu']} {r['name']:<10}{r['projected_s'] / 60:>8.1f} min  " +
              ", ".join(f"{k} {v / 60:.1f}" for k, v in r["parts"].items()))
    print(f"total {p['total_s'] / 3600:.2f} h ({p['n_prompts']} prompts x {p['prompt_s']:.1f} s); elapsed so far "
          f"{p['elapsed_s'] / 3600:.2f} h; budget {p['budget_h']} h, session {p['session_h']} h; safety factor "
          f"{p['safety']}; preflight peak {p['peak_bytes'] / GiB:.2f} GiB")
    for x in p["reasons"]:
        print("OVER:", x)
    print("plan: within limits" if not p["over"] else "plan: OVER LIMIT, queue not started")


# ---------------------------------------------------------------------------------------------------------------
# Check.

def expected_ids(j) -> list:
    from . import kv_diag as KD
    from . import kv_run as KR

    return [c["id"] for c in KD.select_prompts(KR.load_cases(), j["expect"].get("limit_prompts"))]


def _check_outputs(j, run: dict, out: Path) -> list:
    """Row, prompt and npz files against the expected prompts, configurations and shapes."""
    from common.io import sha256_file

    from . import kv_diag as KD

    n, problems = j["name"], []
    try:
        ids = expected_ids(j)
        cfg = run.get("config") or {}
        if cfg.get("version") != KD.VERSION or cfg.get("prompt_ids") != ids:
            problems.append(f"{n}: version {cfg.get('version')!r} or prompt ids differ from {len(ids)} expected")
        if not j["expect"] and cfg.get("prompt_ids") != cfg.get("diagnostic_prompt_ids"):
            problems.append(f"{n}: prompt ids are not the 18 diagnostic prompts")
        for key, name in ROW_FILES.items():
            info = (run.get("row_files") or {}).get(key) or {}
            if info.get("path") != name or not (out / name).is_file() or sha256_file(out / name) != info.get("sha256"):
                problems.append(f"{n}: row file {name} missing or sha256 differs from run.json")
        rows = _rows_gz(out / ROW_FILES["rows"])
        prs = _rows_gz(out / ROW_FILES["prompts"])
        pairs = {(r["id"], r["config"]) for r in rows}
        if len(rows) != len(KD.CONFIGS) * len(ids) or pairs != {(i, m) for i in ids for m in KD.CONFIGS}:
            problems.append(f"{n}: {len(rows)} control rows, expected {len(KD.CONFIGS) * len(ids)} "
                            f"(every prompt x configuration once)")
        if [r["id"] for r in prs] != ids:
            problems.append(f"{n}: prompt rows {[r['id'] for r in prs][:3]}... differ from the expected ids")
        bad = [f"{r['id']}/{r['config']}" for r in rows if not r.get("positions_preserved")]
        if bad:
            problems.append(f"{n}: positions not preserved in {len(bad)} rows, first {bad[:3]}")
        nonfinite = [f"{r['id']}/{r['config']}" for r in rows
                     if not all(math.isfinite(r[k]) for k in ("answer_nll_mean", "kl_post_prefill",
                                                              "kl_teacher_forced_mean"))]
        if nonfinite:
            problems.append(f"{n}: non-finite NLL or KL in {len(nonfinite)} rows, first {nonfinite[:3]}")
        ref_bad = [r["id"] for r in rows if r["config"] == KD.REFERENCE
                   and (r["kl_post_prefill"] != 0 or r["kl_teacher_forced_mean"] != 0)]
        if ref_bad:
            problems.append(f"{n}: fp16 reference KL not zero for {ref_bad[:3]}")
        per = (run.get("summary") or {}).get("per_config") or {}
        if set(per) != set(KD.CONFIGS) or any(v.get("n") != len(ids) for v in per.values()):
            problems.append(f"{n}: summary per-config counts {[(k, v.get('n')) for k, v in per.items()]}")
        shp = run.get("kv_shape") or {}
        dims3 = (shp.get("n_layers"), shp.get("n_kv_heads"), shp.get("head_dim"))
        npz = sorted(p.name for p in (out / "npz").glob("*.npz")) if (out / "npz").is_dir() else []
        if npz != sorted(f"{i}.npz" for i in ids):
            problems.append(f"{n}: npz files {len(npz)} != {len(ids)} expected")
        for name in npz:
            with np.load(out / "npz" / name, allow_pickle=False) as z:
                if set(z.files) != set(KD.NPZ_KEYS):
                    problems.append(f"{n}: npz {name} keys differ")
                    continue
                if any(z[k].shape != dims3 for k in KD.CHANNEL_ARRAYS + KD.MASK_ARRAYS) or \
                        any(z[k].shape != dims3[:2] for k in KD.HEAD_ARRAYS):
                    problems.append(f"{n}: npz {name} shapes differ from {dims3}")
                elif not all(np.isfinite(z[k]).all() for k in KD.HEAD_ARRAYS):
                    problems.append(f"{n}: npz {name} non-finite per-head statistics")
    except Exception as exc:
        problems.append(f"{n}: output files unreadable: {type(exc).__name__}: {exc}")
    return problems


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
        elif conc["overlapping"]:
            problems.append(f"{n}: overlapped {[o['name'] for o in conc['overlapping']]}")
        problems += _check_outputs(j, run, out)
        log = Path(logs_dir) / f"{n}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        for pat in (DONE_PATTERN, f"{n}: exit code 0", f"[job] {n}: concurrent jobs"):
            if pat not in text:
                problems.append(f"{n}: log line {pat!r} not found in {log}")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# Summary.

def _f(x, nd=3):
    return "  -  " if x is None else f"{x:.{nd}f}"


def reproduction(rows: list, eval_rows_path=EVAL_ROWS) -> dict:
    """Per evaluated configuration, the prompts whose generated tokens equal the committed eval row of the same
    prompt and method (descriptive)."""
    p = Path(eval_rows_path)
    if not p.is_file():
        return {}
    ref = {(r["id"], r["method"]): r["generated_token_ids"] for r in _rows_gz(p) if r["method"] in REPRODUCED}
    out = {}
    for m in REPRODUCED:
        rs = [r for r in rows if r["config"] == m and (r["id"], m) in ref]
        out[m] = (sum(r["generated_token_ids"] == ref[(r["id"], m)] for r in rs), len(rs))
    return out


def summary_lines(meta: list, max_lines: int = 40, eval_rows_path=EVAL_ROWS) -> list:
    from . import kv_diag as KD

    L = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            L.append(f"{j['name']}: status {run.get('status')!r}, step {run.get('step')!r}")
            continue
        try:
            s = run["summary"]
            per = s["per_config"]
            L.append(f"{j['name']}: {run['config']['n_prompts']} prompts; exact match, mean answer NLL, post-prefill KL, "
                     f"teacher-forced KL, K / V recon")
            for m in KD.CONFIGS:
                v = per.get(m)
                if v is None:
                    continue
                L.append(f"  {KD.LABELS[m]:<34} EM {v['accuracy']:.3f}  NLL {_f(v['mean_answer_nll'], 4)}  post KL "
                         f"{_f(v['mean_kl_post_prefill'], 4)}  TF KL {_f(v['mean_kl_teacher_forced'], 4)}  recon "
                         f"{_f(v['mean_k_recon'])} / {_f(v['mean_v_recon'])}")
            ch = s["channel"]
            pl = ch["per_layer"]
            nl = len(pl["k_ratio"])
            L.append(f"channel max/median ratio per layer, K / V (mean over prompts and KV heads; top group "
                     f"{ch['n_top_channels']} channels):")
            for a in range(0, nl, 7):
                L.append("  " + "  ".join(f"L{i} {pl['k_ratio'][i]:.1f}/{pl['v_ratio'][i]:.1f}"
                                          for i in range(a, min(a + 7, nl))))
            L.append("K relative error per layer, token-wise top/rest | channel-K top/rest:")
            for a in range(0, nl, 4):
                L.append("  " + "  ".join(
                    f"L{i} {pl['k_err_token_top'][i]:.3f}/{pl['k_err_token_rest'][i]:.3f}|"
                    f"{pl['k_err_channel_top'][i]:.3f}/{pl['k_err_channel_rest'][i]:.3f}"
                    for i in range(a, min(a + 4, nl))))
            o = ch["overall"]
            L.append(f"overall: K ratio {o['k_ratio']:.2f}, V ratio {o['v_ratio']:.2f}; K token-wise top/rest "
                     f"{o['k_err_token_top']:.3f}/{o['k_err_token_rest']:.3f}, channel-K top/rest "
                     f"{o['k_err_channel_top']:.3f}/{o['k_err_channel_rest']:.3f}; V token-wise top/rest "
                     f"{o['v_err_token_top']:.3f}/{o['v_err_token_rest']:.3f}")
            ro = s["readout"]
            L.append("AM4.5 read-out (descriptive): " + ", ".join(f"{k} {ro[k]}" for k in KD.PREDICTIONS if k in ro))
            rows = _rows_gz(Path(j["out"]) / ROW_FILES["rows"])
            rep = reproduction(rows, eval_rows_path)
            if rep:
                L.append("tokens equal to the committed eval rows: " + ", ".join(f"{m} {a}/{b}"
                                                                               for m, (a, b) in rep.items()))
        except Exception as exc:
            L.append(f"{j['name']}: summary unavailable: {type(exc).__name__}: {exc}")
    return L[:max_lines]


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    j.add_argument("--plan", default=None)
    j.add_argument("--results", default=None)
    j.add_argument("--status", default=None)
    j.add_argument("--device", default=None)
    j.add_argument("--model-path", default=None)
    j.add_argument("--limit-prompts", default=None)
    jb = sub.add_parser("job")
    jb.add_argument("--name", required=True)
    jb.add_argument("--status", required=True)
    jb.add_argument("--out", required=True)
    jb.add_argument("command", nargs=argparse.REMAINDER)
    pc = sub.add_parser("preflight-check")
    pc.add_argument("--jobs-dir", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--jobs-dir", required=True)
    p.add_argument("--budget-h", type=float, default=10.5)
    p.add_argument("--session-h", type=float, default=11.5)
    p.add_argument("--elapsed-s", type=float, default=0.0)
    p.add_argument("--vram-gib", type=float, default=14.0)
    p.add_argument("--out", default=None)
    for name in ("summary", "check", "zip", "final"):
        s = sub.add_parser(name)
        s.add_argument("--jobs-dir", required=True)
        s.add_argument("--status", default=None)
        s.add_argument("--logs", default=None)
        if name == "summary":
            s.add_argument("--eval-rows", default=EVAL_ROWS)
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
    if a.cmd == "jobs":
        test = {"device": a.device, "model_path": a.model_path, "limit_prompts": a.limit_prompts}
        test = {k: v for k, v in test.items() if v is not None}
        if a.stage == "main":
            if not a.plan:
                raise SystemExit("main jobs need --plan")
            plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
            if plan.get("over"):
                raise SystemExit(f"plan {a.plan} is over the limits; no main jobs written")
        jobs = stage_jobs(a.stage, a.results, a.status, test)
        paths = N1.write_jobs(a.stage, a.jobs_dir, jobs)
        for q in N1.queues(jobs):
            print(f"{q}: " + ", ".join(x["name"] for x in jobs if x["queue"] == q))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd == "preflight-check":
        meta = N1.load_meta(a.jobs_dir, "preflight")
        problems = preflight_check(meta)
        for x in problems:
            print("PROBLEM:", x)
        print(f"preflight-check: {len(meta)} jobs, {len(problems)} problems")
        return 1 if problems else 0
    if a.cmd == "plan":
        pl = project(a.jobs_dir, a.budget_h, a.session_h, a.elapsed_s, a.vram_gib)
        print_plan(pl)
        if a.out:
            _write_new(a.out, pl)
        return 1 if pl["over"] else 0
    meta = N1.load_meta(a.jobs_dir, "main")
    status = a.status or DEFAULTS["main"]["status"]
    if a.cmd == "summary":
        for line in summary_lines(meta, eval_rows_path=a.eval_rows):
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
