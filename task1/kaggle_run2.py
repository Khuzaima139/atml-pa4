"""Run 2 of Task 1 on Kaggle (held-out evaluation): artifact location and checks, job lists, projection, summary,
check, zip.

python -m task1.kaggle_run2 locate --inputs DIR --out FILE [--run1-root DIR] [--conditions ID ...] [--no-hash]
python -m task1.kaggle_run2 jobs --stage preflight|smoke|run2 --jobs-dir DIR --located FILE [--plan FILE]
       [--results DIR] [--status DIR] [--run1-root DIR] [--conditions ID ...] [--smoke-condition ID]
       [--model-path PATH --device DEV --limit-windows N --limit-hellaswag N --limit-dev N --limit-calib N
       --repo-root DIR]   (test options)
python -m task1.kaggle_run2 plan --smoke-dir DIR --located FILE --working-bytes N [--smoke-condition ID]
       [--conditions ID ...] [--budget-h 10.5] [--disk-gb 18] [--out FILE]
       [--min-lmeval-examples 512]   (test option)
python -m task1.kaggle_run2 summary|check|zip|final --jobs-dir DIR [--stage run2] [--status DIR] ...
python -m task1.kaggle_run2 failure-lines --jobs-dir DIR [--stage run2] [--status DIR] [--logs DIR]
python -m task1.kaggle_run2 job --out DIR -- CMD ...

locate finds every artifact directory (<...>/artifacts/<ID>/ with manifest.json and artifact.safetensors) under the
attached notebook outputs and keeps, per condition, the one whose manifest sha256 equals the committed copy
<run1-root>/run1b/<ID>/artifact_manifest.json for q3_int4 and q3_int3 and <run1-root>/run1/<ID>/artifact_manifest.json
for every other condition (and the run-1 record); exactly one match is required. It then checks the artifact file size
and sha256 against that manifest. Nothing is written but the located-artifacts JSON.

Jobs (run2), one background queue per GPU for the evaluation phase and one more on GPU 0 for the resources phase:
  evaluation  fp16 through task1.baseline (full, lm_eval acceptance check) and the 17 compressed conditions through
              task1.evaluate_condition at their configured HellaSwag scope, all with --no-resources; spread over
              GPU 0 and GPU 1 by projected time (longest first onto the less loaded GPU, from the plan file).
  resources   --resources-only for all 18 conditions on GPU 0, fp16 first; every resources job waits for the .exit
              file of every evaluation job (any exit code), so it starts only after the evaluation phase on both
              GPUs has finished. GPU 1 is idle in this phase.
Outputs: results/task1/run2/<ID>/ and results/task1/run2_resources/<ID>/.
Preflight: fp16 and q3_int3 (GPU 0), p3_s50 (GPU 1) with one window per suite and one HellaSwag ID (fp16 also one
calibration sequence), --no-resources, under results_smoke/preflight/. Smoke: fp16 with the first 512 HellaSwag IDs
(so its lm_eval check runs on the full 512 IDs) then the worst-case batches (GPU 0); the smoke condition with the
first 100 HellaSwag IDs then its --resources-only run (GPU 1); under results_smoke/smoke/.
Every job command runs through `job --out DIR -- CMD`: the command's output goes to the job log unchanged, signals are
forwarded to it, its exit code is returned, and on a nonzero exit the status fields of DIR/run.json are printed (lines
starting with STATUS_PREFIX). failure-lines prints, for every job that did not exit 0, the lines of its log that start
with STATUS_PREFIX or the lm_eval check prefix (task1.verify_lmeval.LOG_PREFIX).
summary, check and zip report incomplete jobs and missing files instead of stopping; final returns 1 if any job is
not ok, the check report is missing or lists problems, or the zip is missing.
"""
import argparse
import json
import sys
from pathlib import Path

from . import config as C
from . import kaggle_run1 as K

STAGES = ("preflight", "smoke", "run2")
DEFAULTS = {
    "run2": {"results": "results/task1", "status": "notes/kaggle_status", "logs": "notes/kaggle_logs",
             "prefix": "", "limits": {}},
    "smoke": {"results": "results_smoke/smoke", "status": "notes/kaggle_status_smoke",
              "logs": "notes/kaggle_logs_smoke", "prefix": "sm_", "limits": {"limit_hellaswag": 100},
              "fp16_limits": {"limit_hellaswag": C.HELLASWAG_SUBSET_N}},
    "preflight": {"results": "results_smoke/preflight", "status": "notes/kaggle_status_smoke",
                  "logs": "notes/kaggle_logs_smoke", "prefix": "pf_",
                  "limits": {"limit_windows": 1, "limit_hellaswag": 1, "limit_calib": 1}},
}
RUN1B_CONDITIONS = ("q3_int4", "q3_int3")
CONDITIONS = tuple(C.CONDITIONS)
PREFLIGHT = {"0": ("fp16", "q3_int3"), "1": ("p3_s50",)}
SMOKE_CONDITION = "q3_int4"
RUN1_MANIFEST_COPY = K.RUN1_MANIFEST_COPY
STAGGER_S = 60.0
VRAM_LIMIT_BYTES = 14 * 2**30  # T4: 15 GiB per GPU
STATUS_PREFIX = "[job status]"
TRACEBACK_TAIL = 15
ROOT_GLOBS = {"run1": "**/atml-pa4-t1-run1", "run1b": "**/atml-pa4-t1-run1b"}


def run1_dir(run1_root, cond: str) -> Path:
    return Path(run1_root) / ("run1b" if cond in RUN1B_CONDITIONS else "run1") / cond


# ---------------------------------------------------------------------------------------------------------------
# Locate and verify the run-1 / run-1b artifacts.

def locate(inputs, run1_root="results/task1", conditions=CONDITIONS, hash_files: bool = True) -> dict:
    """Artifact directory of every condition, matched by manifest sha256 to its committed run-1 / run-1b copy."""
    from common.freeze import FreezeError, require_committed
    from common.io import sha256_file

    inputs = Path(inputs)
    found = {}
    for man in sorted(inputs.rglob("manifest.json")):
        d = man.parent
        if d.name in C.CONDITIONS and d.parent.name == "artifacts" and (d / "artifact.safetensors").is_file():
            found.setdefault(d.name, []).append(d)
    rec = {"inputs": str(inputs), "run1_root": str(run1_root), "hash_files": hash_files,
           "roots": {k: sorted(str(p) for p in inputs.glob(g) if p.is_dir()) for k, g in ROOT_GLOBS.items()},
           "conditions": {}, "problems": []}
    probs = rec["problems"]
    for cond in conditions:
        rd = run1_dir(run1_root, cond)
        copy, runp = rd / RUN1_MANIFEST_COPY, rd / "run.json"
        c = {"run_dir": str(rd), "candidates": [str(d) for d in found.get(cond, [])]}
        rec["conditions"][cond] = c
        if not copy.is_file() or not runp.is_file():
            probs.append(f"{cond}: missing {copy} or {runp}")
            continue
        try:
            git = require_committed(copy, f"committed artifact manifest copy of {cond}")
            c["manifest_copy_git"] = {k: git[k] for k in ("tracked", "clean", "rel_path")}
        except FreezeError as exc:
            probs.append(f"{cond}: {exc}")
        want = sha256_file(copy)
        recorded = json.loads(runp.read_text(encoding="utf-8"))["steps"]["artifact"]["manifest_sha256"]
        c["manifest_sha256_expected"] = want
        if want != recorded:
            probs.append(f"{cond}: committed manifest copy sha256 {want} != run record {recorded}")
        hits, other = [], []
        for d in found.get(cond, []):
            (hits if sha256_file(d / "manifest.json") == want else other).append(str(d))
        c["not_used"] = other
        if len(hits) != 1:
            probs.append(f"{cond}: {len(hits)} artifact directories match the committed manifest ({hits}); "
                         f"candidates {c['candidates']}")
            continue
        d = Path(hits[0])
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        st = d / "artifact.safetensors"
        c.update(dir=str(d), manifest_sha256=want, file_bytes=st.stat().st_size, file_sha256_manifest=man["file"]["sha256"],
                 condition_in_manifest=man.get("condition"), design=man.get("design"),
                 allocation_sha256=(man.get("allocation") or {}).get("sha256"))
        if man.get("condition") != cond:
            probs.append(f"{cond}: manifest is for {man.get('condition')}")
        if st.stat().st_size != man["file"]["bytes"]:
            probs.append(f"{cond}: artifact size {st.stat().st_size} != manifest {man['file']['bytes']}")
        if hash_files:
            c["file_sha256"] = sha256_file(st)
            c["file_sha256_verified"] = c["file_sha256"] == man["file"]["sha256"]
            if not c["file_sha256_verified"]:
                probs.append(f"{cond}: artifact sha256 {c['file_sha256']} != manifest {man['file']['sha256']}")
    return rec


def print_located(rec: dict):
    print("notebook output roots:", json.dumps(rec["roots"]))
    print(f"{'condition':<12}{'file bytes':>16}  {'sha256 ok':<10}directory")
    for cond, c in rec["conditions"].items():
        ok = c.get("file_sha256_verified", "-" if not rec["hash_files"] else False)
        print(f"{cond:<12}{c.get('file_bytes', 0):>16,d}  {str(ok):<10}{c.get('dir', '-')}")
        for d in c.get("not_used", []):
            print(f"{'':<12}{'':>16}  {'not used':<10}{d}")
    for x in rec["problems"]:
        print("PROBLEM:", x)
    print("located:", "all artifacts verified" if not rec["problems"] else f"{len(rec['problems'])} problems")


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def _test_args(entry: str, t: dict, resources_only: bool = False) -> list:
    out = []
    if t.get("model_path"):
        out += ["--model-path", str(t["model_path"])]
    if t.get("device"):
        out += ["--device", t["device"]]
    keys = {"task1.baseline": ("limit_windows", "limit_hellaswag", "limit_calib"),
            "task1.evaluate_condition": ("limit_windows", "limit_hellaswag", "limit_dev", "repo_root"),
            "task1.smoke_run2": ("limit_hellaswag",)}[entry]
    if resources_only:  # the run-1 record check needs the same development limit; nothing else is read
        keys = tuple(k for k in keys if k in ("limit_dev", "repo_root"))
    for k in keys:
        if t.get(k) is not None:
            out += ["--" + k.replace("_", "-"), str(t[k])]
    return out


def default_assignment(conditions) -> dict:
    """GPU per evaluation job without a plan file: longest first by a static weight (HellaSwag IDs scored)."""
    w = {c: (C.HELLASWAG_FULL_N + 2 * C.HELLASWAG_SUBSET_N if c == "fp16" else
             C.hellaswag_n(C.CONDITIONS[c]["hellaswag"])) for c in conditions}
    return balance({f"eval_{c}": float(w[c]) for c in conditions}, list(conditions))


def balance(seconds: dict, order: list) -> dict:
    """Longest-processing-time assignment of jobs to GPU 0 / GPU 1 (GPU 1 starts STAGGER_S later); ties by order."""
    pos = {f"eval_{c}": i for i, c in enumerate(order)}
    load = {"0": 0.0, "1": STAGGER_S}
    out = {}
    for name in sorted(seconds, key=lambda n: (-seconds[n], pos.get(n, 0))):
        g = min(load, key=lambda k: (load[k], k))
        out[name] = g
        load[g] += seconds[name]
    return out


def stage_jobs(stage: str, located: dict, results=None, status=None, run1_root=None, plan=None, test=None,
               conditions=CONDITIONS, smoke_condition=SMOKE_CONDITION, python=sys.executable) -> list:
    """Job specs with metadata: name, gpu, queue, phase, entry, condition, out, artifacts, run1, cmd, wait_for, expect."""
    d = DEFAULTS[stage]
    R = Path(results or d["results"])
    S = Path(status or d["status"])
    run1_root = run1_root or located.get("run1_root", "results/task1")
    t_all = {**d["limits"], **(test or {})}
    t_fp16 = {**d["limits"], **d.get("fp16_limits", {}), **(test or {})}
    pre = d["prefix"]
    jobs = []

    def add(gpu, queue, phase, name, entry, cond, out, args, wait_for=(), resources_only=False):
        t = t_fp16 if cond == "fp16" else t_all
        inner = [python, "-m", entry, *args, "--out", str(out), *_test_args(entry, t, resources_only)]
        cmd = [python, "-m", "task1.kaggle_run2", "job", "--out", str(out), "--", *inner]
        loc = located["conditions"].get(cond, {}) if cond else {}
        scope = C.CONDITIONS[cond]["hellaswag"] if cond else None
        expect = {"resources_only": resources_only, "limits": {k: t.get(k) for k in
                  ("limit_windows", "limit_hellaswag", "limit_dev", "limit_calib")},
                  "hellaswag_n": None if (resources_only or not cond) else
                  (t["limit_hellaswag"] if t.get("limit_hellaswag") is not None else C.hellaswag_n(scope))}
        jobs.append({"name": pre + name, "gpu": str(gpu), "queue": queue, "phase": phase, "entry": entry,
                     "kind": phase, "condition": cond, "out": str(out),
                     "artifacts": loc.get("dir") if cond and cond != "fp16" else None,
                     "manifest_sha256": loc.get("manifest_sha256") if cond and cond != "fp16" else None,
                     "run1": str(run1_dir(run1_root, cond)) if cond and cond != "fp16" else None,
                     "cmd": cmd, "wait_for": list(wait_for), "expect": expect})

    def evaluation(gpu, queue, cond, out, mode):
        if cond == "fp16":
            add(gpu, queue, "eval" if mode == "--no-resources" else "resources", f"{_phase(mode)}_{cond}",
                "task1.baseline", cond, out, [mode], resources_only=mode == "--resources-only",
                wait_for=wait if mode == "--resources-only" else ())
            return
        loc = located["conditions"][cond]
        if "dir" not in loc:
            raise SystemExit(f"no verified artifact for {cond}; run locate first")
        add(gpu, queue, "eval" if mode == "--no-resources" else "resources", f"{_phase(mode)}_{cond}",
            "task1.evaluate_condition", cond, out,
            ["--condition", cond, "--artifacts", loc["dir"], "--run1", str(run1_dir(run1_root, cond)), mode],
            resources_only=mode == "--resources-only", wait_for=wait if mode == "--resources-only" else ())

    wait = ()
    if stage == "run2":
        conditions = list(conditions)
        if plan:
            assign = plan["assignment"]
            secs = {r["name"]: r["projected_s"] for r in plan["rows"] if r["phase"] == "eval"}
        else:
            assign = default_assignment(conditions)
            secs = {}
        missing = [c for c in conditions if f"eval_{c}" not in assign]
        if missing:
            raise SystemExit(f"plan has no GPU for {missing}")
        # Each queue runs its jobs longest first (the order in which balance() placed them).
        for c in sorted(conditions, key=lambda c: (-secs.get(f"eval_{c}", 0.0), conditions.index(c))):
            g = assign[f"eval_{c}"]
            evaluation(g, f"gpu{g}", c, R / "run2" / c, "--no-resources")
        wait = [str(S / f"{pre}eval_{c}.exit") for c in conditions]
        for c in ["fp16"] * ("fp16" in conditions) + [c for c in conditions if c != "fp16"]:
            evaluation("0", "res_gpu0", c, R / "run2_resources" / c, "--resources-only")
    elif stage == "preflight":
        for g, conds in PREFLIGHT.items():
            for c in conds:
                if c in conditions:
                    evaluation(g, f"gpu{g}", c, R / "run2" / c, "--no-resources")
    else:
        if "fp16" in conditions:
            evaluation("0", "gpu0", "fp16", R / "eval" / "fp16", "--no-resources")
        add("0", "gpu0", "worst", "worst_batch", "task1.smoke_run2", None, R / "worst_batch", [])
        evaluation("1", "gpu1", smoke_condition, R / "eval" / smoke_condition, "--no-resources")
        evaluation("1", "gpu1", smoke_condition, R / "resources" / smoke_condition, "--resources-only")
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)
    return jobs


def _phase(mode: str) -> str:
    return "eval" if mode == "--no-resources" else "res"


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


# ---------------------------------------------------------------------------------------------------------------
# Projection from the smoke runs.

def _run(path) -> dict:
    r = K._load_json(path)
    if not r or r.get("status") != "ok":
        raise SystemExit(f"smoke run not finished ok: {path}")
    return r


def project(smoke_dir, located: dict, working_bytes: int, budget_h: float = 10.5, disk_gb: float = 18.0,
            safety: float = K.SAFETY, conditions=CONDITIONS, smoke_condition=SMOKE_CONDITION,
            min_lmeval_examples: int = C.HELLASWAG_SUBSET_N) -> dict:
    """Per-job time projection, GPU assignment of the evaluation jobs, queue end times, VRAM and disk. The fp16 smoke's
    lm_eval check must cover min_lmeval_examples IDs (the full 512 on Kaggle); its peak allocated memory (the step
    alone) is checked against VRAM_LIMIT_BYTES like every other smoke peak."""
    sd = Path(smoke_dir)
    base = _run(sd / "eval" / "fp16" / "run.json")
    ev = _run(sd / "eval" / smoke_condition / "run.json")
    res = _run(sd / "resources" / smoke_condition / "run.json")
    worst = _run(sd / "worst_batch" / "run.json")
    tb, te, tr = base["timings_s"], ev["timings_s"], res["timings_s"]
    hs_b = tb["hellaswag_s"] / base["steps"]["hellaswag"]["n"]
    lm_b = tb["lmeval_verify_s"] / base["steps"]["lmeval_verify"]["n_examples"]
    hs_e = te["hellaswag_s"] / ev["steps"]["hellaswag"]["n"]
    ref_bytes = located["conditions"][smoke_condition]["file_bytes"]

    def art_load(c, t):
        return t["artifact_load_s"] * max(1.0, located["conditions"][c]["file_bytes"] / ref_bytes)

    rows = []
    for c in conditions:
        if c == "fp16":
            sec = (tb["load_s"] + tb["enumeration_s"] + tb["ppl_wikitext2_test_s"] + tb["ppl_c4_validation_s"]
                   + hs_b * C.HELLASWAG_FULL_N + tb["calib_nll_c4_calibration_s"] + tb["calib_nll_c4_development_s"]
                   + lm_b * C.HELLASWAG_SUBSET_N)
            how = (f"load + enumeration + both perplexity suites + {C.HELLASWAG_FULL_N} x HellaSwag per ID + "
                   f"calibration and development NLL + {C.HELLASWAG_SUBSET_N} x lm_eval check per ID")
            rsec = tr["load_s"] + tb["enumeration_s"] + tr["resources_s"]
        else:
            n = C.hellaswag_n(C.CONDITIONS[c]["hellaswag"])
            sec = (te["load_s"] + art_load(c, te) + te["calib_nll_c4_development_s"] + te["ppl_wikitext2_test_s"]
                   + te["ppl_c4_validation_s"] + hs_e * n)
            how = (f"load + artifact load (scaled by file bytes) + development NLL + both perplexity suites + "
                   f"{n} x HellaSwag per ID")
            rsec = tr["load_s"] + art_load(c, tr) + tr["resources_s"]
        rows.append({"name": f"eval_{c}", "condition": c, "phase": "eval", "raw_s": sec, "projected_s": sec * safety,
                     "formula": how})
        rows.append({"name": f"res_{c}", "condition": c, "phase": "resources", "raw_s": rsec,
                     "projected_s": rsec * safety, "formula": "load (+ artifact load) + resources"})
    ev_rows = {r["name"]: r["projected_s"] for r in rows if r["phase"] == "eval"}
    assign = balance(ev_rows, list(conditions))
    for r in rows:
        r["gpu"] = assign.get(r["name"], "0")
    end = {"0": sum(s for n, s in ev_rows.items() if assign[n] == "0"),
           "1": STAGGER_S + sum(s for n, s in ev_rows.items() if assign[n] == "1")}
    res_start = max(end.values())
    res_total = sum(r["projected_s"] for r in rows if r["phase"] == "resources")
    q = {"0": res_start + res_total, "1": end["1"]}
    peaks = {"smoke_eval_fp16": max((base.get("peak_vram_allocated_bytes") or {"x": 0}).values()),
             f"smoke_eval_{smoke_condition}": max((ev.get("peak_vram_allocated_bytes") or {"x": 0}).values()),
             f"smoke_res_{smoke_condition}": max((res.get("peak_vram_allocated_bytes") or {"x": 0}).values())}
    for k, u in worst["steps"]["units"].items():
        peaks[f"worst_{k}"] = u["peak_allocated_bytes"] or 0
    lv = base["steps"]["lmeval_verify"]
    peaks["smoke_lmeval_verify_fp16"] = lv.get("peak_allocated_bytes") or 0
    disk = working_bytes + K.RESULTS_BYTES_PER_JOB * len(rows)
    limits = {"queue_hours": budget_h, "disk_bytes": int(disk_gb * 1e9), "vram_bytes": VRAM_LIMIT_BYTES}
    over = [f"GPU {g} queue {s / 3600:.2f} h > {budget_h} h" for g, s in q.items() if s > budget_h * 3600]
    if disk > limits["disk_bytes"]:
        over.append(f"projected /kaggle/working use {disk / 1e9:.2f} GB > {disk_gb} GB")
    over += [f"peak allocated VRAM of {k} {v / 2**30:.2f} GiB > {VRAM_LIMIT_BYTES / 2**30:.0f} GiB"
             for k, v in peaks.items() if v and v > VRAM_LIMIT_BYTES]
    if lv["n_examples"] < min_lmeval_examples:
        over.append(f"fp16 smoke lm_eval check ran on {lv['n_examples']} IDs < {min_lmeval_examples}; its peak "
                    f"memory does not cover the full step")
    if "peak_allocated_bytes" not in lv:
        over.append("fp16 smoke record lacks steps.lmeval_verify.peak_allocated_bytes")
    return {"rows": rows, "assignment": assign, "order": list(conditions), "eval_end_s": end,
            "resources_start_s": res_start, "queue_end_s": q, "queue_end_h": {g: s / 3600 for g, s in q.items()},
            "per_example_s": {"hellaswag_fp16": hs_b, "lmeval_fp16": lm_b, f"hellaswag_{smoke_condition}": hs_e},
            "peak_allocated_bytes": peaks, "lmeval_verify_n_examples": lv["n_examples"], "disk_bytes": disk,
            "working_bytes_now": working_bytes,
            "limits": limits, "safety": safety, "stagger_s": STAGGER_S, "over": over}


def print_plan(p: dict):
    print(f"{'job':<22}{'gpu':>4}{'raw min':>10}{'proj min':>10}")
    for r in p["rows"]:
        print(f"{r['name']:<22}{r['gpu']:>4}{r['raw_s'] / 60:>10.1f}{r['projected_s'] / 60:>10.1f}")
    print("per example (s):", json.dumps({k: round(v, 4) for k, v in p["per_example_s"].items()}))
    print("peak allocated VRAM (GiB):", json.dumps({k: round(v / 2**30, 2) for k, v in p["peak_allocated_bytes"].items()}))
    print(f"fp16 smoke lm_eval check: {p['lmeval_verify_n_examples']} IDs, step peak allocated "
          f"{p['peak_allocated_bytes']['smoke_lmeval_verify_fp16'] / 2**30:.2f} GiB")
    print(f"evaluation phase ends: GPU 0 {p['eval_end_s']['0'] / 3600:.2f} h, GPU 1 {p['eval_end_s']['1'] / 3600:.2f} h;"
          f" resources phase on GPU 0 from {p['resources_start_s'] / 3600:.2f} h")
    for g, h in p["queue_end_h"].items():
        print(f"GPU {g} queue ends after {h:.2f} h (safety factor {p['safety']}, limit {p['limits']['queue_hours']} h)")
    print(f"/kaggle/working projected {p['disk_bytes'] / 1e9:.2f} GB (limit {p['limits']['disk_bytes'] / 1e9:.1f} GB)")
    print("OVER LIMIT: " + "; ".join(p["over"]) if p["over"] else "within limits")


# ---------------------------------------------------------------------------------------------------------------
# Summary.

def _peak(run: dict):
    peaks = run.get("peak_vram_allocated_bytes") or {}
    try:
        return max(peaks.values()) if peaks else None
    except (AttributeError, TypeError, ValueError):
        return None


def summary(meta: list, status_dir) -> dict:
    """Job table and one row per condition from its evaluation and resources records."""
    status_dir = Path(status_dir)
    jobs, incomplete, by = [], [], {}
    for j in meta:
        st = K._status(status_dir, j["name"])
        run = K._load_json(Path(j["out"]) / "run.json") or {}
        jobs.append({"name": j["name"], "gpu": j["gpu"], "queue": j["queue"], "exit": st["exit"], "done": st["done"],
                     "run_status": run.get("status"), "wall_s": st["wall_s"], "peak_vram_allocated_bytes": _peak(run)})
        if j["condition"]:
            by.setdefault(j["condition"], {})[j["phase"]] = run if run.get("status") == "ok" else None
    conds = []
    for c, ph in by.items():
        row = {"condition": c}
        e, r = ph.get("eval"), ph.get("resources")
        try:
            if e:
                s = e["steps"]
                for suite in ("wikitext2_test", "c4_validation"):
                    p = s["perplexity"][suite]
                    row[suite] = {"N": p["N"], "mean_nll": p["mean_nll"], "ppl": p["ppl"]}
                h = s["hellaswag"]
                row["hellaswag"] = {"n": h["n"], "acc": h["acc"], "acc_norm": h["acc_norm"], "scope": h["scope"]}
                row["lmeval_acceptance"] = s["lmeval_verify"]["acceptance"] if c == "fp16" else None
            elif "eval" in ph:
                incomplete.append(f"{c}: evaluation record not ok or missing")
            if r:
                m = r["steps"]["resources"]["measurements"]
                row["prefill_512_ms"] = m["prefill_512"]["median_ms"]
                row["prefill_2048_ms"] = m["prefill_2048"]["median_ms"]
                dk = next(k for k in m if k.startswith("decode_"))
                row["decode_ms"] = m[dk]["median_ms"]
                pk = [x.get("peak_allocated_bytes_max") for x in m.values()]
                row["peak_allocated_bytes"] = max(pk) if all(v is not None for v in pk) else None
            elif "resources" in ph:
                incomplete.append(f"{c}: resources record not ok or missing")
            src = r or e
            if src:
                stg = src["steps"]["storage"]
                if c == "fp16":
                    row["logical_whole_model_bytes"] = stg["fp16_dense"]["whole_model_bytes"]
                    row["file_bytes"] = None
                else:
                    row["logical_whole_model_bytes"] = stg["logical"]["whole_model_bytes"]
                    row["file_bytes"] = stg["artifact_files"]["total_bytes"]
        except (KeyError, TypeError, StopIteration) as exc:
            incomplete.append(f"{c}: record lacks {exc!r}")
        conds.append(row)
    return {"jobs": jobs, "conditions": conds, "incomplete": incomplete}


def _f(x, fmt):
    return format(x, fmt) if x is not None else "-"


def print_summary(sm: dict):
    print(f"{'job':<24}{'gpu':>4}{'exit':>6}{'run':>8}{'wall min':>10}{'peak GiB':>10}")
    for r in sm["jobs"]:
        w = f"{r['wall_s'] / 60:.1f}" if r["wall_s"] is not None else "-"
        v = f"{r['peak_vram_allocated_bytes'] / 2**30:.2f}" if r["peak_vram_allocated_bytes"] else "-"
        print(f"{r['name']:<24}{r['gpu']:>4}{str(r['exit']):>6}{str(r['run_status']):>8}{w:>10}{v:>10}")
    print()
    print(f"{'condition':<11}{'wt2 N':>7}{'wt2 NLL':>9}{'wt2 PPL':>9}{'c4 N':>7}{'c4 NLL':>8}{'c4 PPL':>9}"
          f"{'hs n':>6}{'acc':>7}{'acc_n':>7}{'lm_eval':>8}{'pre512':>8}{'pre2048':>8}{'decode':>8}{'peak MiB':>9}"
          f"{'logical MB':>11}{'file MB':>9}")
    for c in sm["conditions"]:
        w, v, h = c.get("wikitext2_test", {}), c.get("c4_validation", {}), c.get("hellaswag", {})
        pk = c.get("peak_allocated_bytes")
        lb, fb = c.get("logical_whole_model_bytes"), c.get("file_bytes")
        print(f"{c['condition']:<11}{_f(w.get('N'), '7d')}{_f(w.get('mean_nll'), '9.4f')}{_f(w.get('ppl'), '9.2f')}"
              f"{_f(v.get('N'), '7d')}{_f(v.get('mean_nll'), '8.4f')}{_f(v.get('ppl'), '9.2f')}{_f(h.get('n'), '6d')}"
              f"{_f(h.get('acc'), '7.4f')}{_f(h.get('acc_norm'), '7.4f')}{str(c.get('lmeval_acceptance') or '-'):>8}"
              f"{_f(c.get('prefill_512_ms'), '8.1f')}{_f(c.get('prefill_2048_ms'), '8.1f')}"
              f"{_f(c.get('decode_ms'), '8.1f')}{_f(pk / 2**20 if pk else None, '9.0f')}"
              f"{_f(lb / 1e6 if lb else None, '11.1f')}{_f(fb / 1e6 if fb else None, '9.1f')}")
    for x in sm.get("incomplete", []):
        print("incomplete:", x)


# ---------------------------------------------------------------------------------------------------------------
# Check.

BASELINE_FILES = ("run.json", "ppl_wikitext2_test.jsonl.gz", "ppl_c4_validation.jsonl.gz", "hellaswag_full.jsonl.gz",
                  "calib_nll_c4_calibration.jsonl.gz", "calib_nll_c4_development.jsonl.gz", "lmeval_verify.json")
EVAL_KEYS = ("status", "eval_id", "meta.git_commit", "meta.git_dirty", "config.resources_mode", "config.limits",
             "steps.perplexity.wikitext2_test.N", "steps.perplexity.wikitext2_test.mean_nll",
             "steps.perplexity.wikitext2_test.ppl", "steps.perplexity.c4_validation.N",
             "steps.perplexity.c4_validation.mean_nll", "steps.perplexity.c4_validation.ppl", "steps.hellaswag.n",
             "steps.hellaswag.acc", "steps.hellaswag.acc_norm", "steps.hellaswag.scope", "steps.storage")
RES_KEYS = ("status", "eval_id", "meta.git_commit", "config.resources_mode", "steps.storage",
            "steps.resources.measurements.prefill_512.median_ms", "steps.resources.measurements.prefill_2048.median_ms",
            "steps.held_out_guard.attempts")
INPUT_KEYS = ("steps.inputs.artifact.manifest_sha256", "steps.inputs.artifact.dir", "steps.inputs.run1.dir",
              "steps.inputs.freeze.applies")


def expected_files(j: dict) -> tuple:
    if j["phase"] == "worst":
        return ("run.json",)
    if j["expect"]["resources_only"]:
        return ("run.json",)
    if j["condition"] == "fp16":
        return BASELINE_FILES
    from .evaluate_condition import output_files

    return tuple(output_files(C.CONDITIONS[j["condition"]]["hellaswag"], "skip").values())


def check(meta: list, status_dir) -> list:
    """Problems with the outputs of every job (empty when every file and key the summary and report read is present)."""
    status_dir = Path(status_dir)
    problems = []
    for j in meta:
        try:
            _check_job(j, status_dir, problems)
        except Exception as exc:  # a malformed output is reported, the other jobs are still checked
            problems.append(f"{j['name']}: check error {exc!r}")
    return problems


def _need(d, keys, name, what, problems):
    for k in keys:
        try:
            K._get(d, k)
        except KeyError:
            problems.append(f"{name}: {what} lacks {k}")


def _check_job(j: dict, status_dir: Path, problems: list):
    n = j["name"]
    st = K._status(status_dir, n)
    if st["exit"] != 0 or not st["done"]:
        problems.append(f"{n}: exit {st['exit']}, done {st['done']}")
        return
    out = Path(j["out"])
    files = expected_files(j)
    present = sorted(p.name for p in out.iterdir()) if out.is_dir() else []
    for f in files:
        if f not in present:
            problems.append(f"{n}: missing {out / f}")
    extra = sorted(set(present) - set(files))
    if extra:
        problems.append(f"{n}: unexpected files {extra}")
    run = K._load_json(out / "run.json")
    if run is None:
        return
    if run.get("status") != "ok":
        problems.append(f"{n}: run status {run.get('status')}")
    if j["phase"] == "worst":
        _need(run, ["steps.units." + u + ".peak_allocated_bytes" for u in ("load", "worst_batch_hellaswag",
                                                                           "worst_batch_lmeval")], n, "run.json",
              problems)
        return
    ro = j["expect"]["resources_only"]
    _need(run, RES_KEYS if ro else EVAL_KEYS, n, "run.json", problems)
    cfg = run.get("config", {})
    if cfg.get("resources_mode") != ("only" if ro else "skip"):
        problems.append(f"{n}: resources mode {cfg.get('resources_mode')}")
    steps = run.get("steps", {})
    if ro:
        if steps.get("held_out_guard", {}).get("attempts"):
            problems.append(f"{n}: held-out file access attempts recorded")
        if any(k in steps for k in ("perplexity", "hellaswag", "reproduction", "calib_nll", "lmeval_verify")):
            problems.append(f"{n}: resources-only record holds evaluation steps")
    else:
        if "resources" in steps:
            problems.append(f"{n}: evaluation record holds a resources step")
        lim = j["expect"]["limits"]
        if lim.get("limit_windows") is None:
            for suite in ("wikitext2_test", "c4_validation"):
                if steps.get("perplexity", {}).get(suite, {}).get("N") != C.PPL_SCORED_TOKENS:
                    problems.append(f"{n}: {suite} N {steps.get('perplexity', {}).get(suite, {}).get('N')}")
        if steps.get("hellaswag", {}).get("n") != j["expect"]["hellaswag_n"]:
            problems.append(f"{n}: HellaSwag n {steps.get('hellaswag', {}).get('n')} != {j['expect']['hellaswag_n']}")
        if steps.get("hellaswag", {}).get("scope") != C.CONDITIONS[j["condition"]]["hellaswag"]:
            problems.append(f"{n}: HellaSwag scope {steps.get('hellaswag', {}).get('scope')}")
        if j["condition"] == "fp16":
            if steps.get("lmeval_verify", {}).get("acceptance") != "PASS":
                problems.append(f"{n}: lm_eval acceptance {steps.get('lmeval_verify', {}).get('acceptance')}")
        elif not steps.get("reproduction", {}).get("reproduces_run1"):
            problems.append(f"{n}: development NLL does not reproduce run 1")
    if j["condition"] == "fp16":
        _need(run, ("steps.storage.fp16_dense.whole_model_bytes",), n, "run.json", problems)
        return
    _need(run, INPUT_KEYS + ("steps.storage.logical.whole_model_bytes", "steps.storage.artifact_files.total_bytes"),
          n, "run.json", problems)
    inp = steps.get("inputs", {})
    if inp.get("artifact", {}).get("manifest_sha256") != j["manifest_sha256"]:
        problems.append(f"{n}: artifact manifest sha256 differs from the located artifact")
    if inp.get("artifact", {}).get("dir") != j["artifacts"]:
        problems.append(f"{n}: artifact directory {inp.get('artifact', {}).get('dir')} != {j['artifacts']}")
    if inp.get("run1", {}).get("dir") != j["run1"]:
        problems.append(f"{n}: run-1 directory {inp.get('run1', {}).get('dir')} != {j['run1']}")
    fz = inp.get("freeze", {})
    if C.CONDITIONS[j["condition"]]["family"] in ("P3", "Q3"):
        if not fz.get("applies") or fz.get("rel_path") != f"task1/frozen/{j['condition']}.json":
            problems.append(f"{n}: freeze record {fz.get('applies')} {fz.get('rel_path')}")
    elif fz.get("applies"):
        problems.append(f"{n}: freeze guard applied to a non-P3/Q3 condition")


def zip_roots(meta: list) -> list:
    roots = []
    for j in meta:
        r = str(Path(j["out"]).parent)
        if r not in roots:
            roots.append(r)
    return roots


# ---------------------------------------------------------------------------------------------------------------
# Job wrapper and failure lines.

def status_lines(out, code: int) -> list:
    """STATUS_PREFIX lines for a job that exited with code: the status fields of <out>/run.json and the traceback tail."""
    P = STATUS_PREFIX
    path = Path(out) / "run.json"
    lines = [f"{P} exit code {code}; record {path}"]
    run = K._load_json(path)
    if run is None:
        return lines + [f"{P} run.json missing or unreadable"]
    for k in ("status", "acceptance", "error", "end_time", "elapsed_wall_s"):
        if k in run:
            lines.append(f"{P} {k}: {json.dumps(run[k])}")
    lv = (run.get("steps") or {}).get("lmeval_verify")
    if isinstance(lv, dict):
        lines.append(f"{P} steps.lmeval_verify: acceptance {lv.get('acceptance')} run {lv.get('acceptance_run')} "
                     f"n_examples {lv.get('n_examples')} peak_allocated_bytes {lv.get('peak_allocated_bytes')}")
    tb = run.get("traceback")
    if tb:
        lines += [f"{P} traceback: {x}" for x in tb.rstrip().splitlines()[-TRACEBACK_TAIL:]]
    return lines


def run_job(out, cmd: list) -> int:
    """Run cmd (output inherited), forward SIGTERM/SIGINT/SIGHUP to it, print status_lines on a nonzero exit and return
    its exit code (a child ended by signal k ends this process by the same signal)."""
    import os
    import signal
    import subprocess

    child = subprocess.Popen(cmd)

    def forward(sig, _frame):
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    code = child.wait()
    if code != 0:
        for line in status_lines(out, code):
            print(line, flush=True)
    sys.stdout.flush()
    if code < 0:
        signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    return code


def failure_lines(meta: list, status_dir, logs_dir) -> list:
    """For every job without exit code 0: a header and the lines of its log that start with STATUS_PREFIX or the
    lm_eval check prefix."""
    from .verify_lmeval import LOG_PREFIX

    out = []
    for j in meta:
        st = K._status(Path(status_dir), j["name"])
        if st["exit"] == 0:
            continue
        log = Path(logs_dir) / f"{j['name']}.log"
        out.append(f"== {j['name']} (exit {st['exit']}, log {log}) ==")
        if not log.is_file():
            out.append("(no log)")
            continue
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            hits = [x.rstrip("\n") for x in fh if x.startswith((STATUS_PREFIX, LOG_PREFIX))]
        out += hits or ["(no status or lm_eval check lines in the log)"]
    return out


# ---------------------------------------------------------------------------------------------------------------

def _conditions(a):
    conds = a.conditions or list(CONDITIONS)
    bad = [c for c in conds if c not in C.CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}")
    return conds


def _write_new(path, obj):
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("locate")
    lo.add_argument("--inputs", required=True)
    lo.add_argument("--out", required=True)
    lo.add_argument("--run1-root", default="results/task1")
    lo.add_argument("--conditions", nargs="*", default=None)
    lo.add_argument("--no-hash", action="store_true", help="skip the artifact file sha256 (tests only)")
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    j.add_argument("--located", required=True)
    j.add_argument("--plan", default=None)
    for k in ("results", "status", "run1_root", "model_path", "device", "repo_root"):
        j.add_argument("--" + k.replace("_", "-"), default=None)
    for k in ("limit_windows", "limit_hellaswag", "limit_dev", "limit_calib"):
        j.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    j.add_argument("--conditions", nargs="*", default=None)
    j.add_argument("--smoke-condition", default=SMOKE_CONDITION)
    p = sub.add_parser("plan")
    p.add_argument("--smoke-dir", required=True)
    p.add_argument("--located", required=True)
    p.add_argument("--working-bytes", type=int, required=True)
    p.add_argument("--smoke-condition", default=SMOKE_CONDITION)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--budget-h", type=float, default=10.5)
    p.add_argument("--disk-gb", type=float, default=18.0)
    p.add_argument("--out", default=None, help="write the projection JSON here (new file)")
    p.add_argument("--min-lmeval-examples", type=int, default=C.HELLASWAG_SUBSET_N,
                   help="IDs the fp16 smoke lm_eval check must cover (tests only)")
    jb = sub.add_parser("job")
    jb.add_argument("--out", required=True)
    jb.add_argument("command", nargs=argparse.REMAINDER)
    fl = sub.add_parser("failure-lines")
    fl.add_argument("--jobs-dir", required=True)
    fl.add_argument("--stage", choices=STAGES, default="run2")
    fl.add_argument("--status", default=None)
    fl.add_argument("--logs", default=None)
    for name in ("summary", "check", "zip", "final"):
        s = sub.add_parser(name)
        s.add_argument("--jobs-dir", required=True)
        s.add_argument("--stage", choices=STAGES, default="run2")
        s.add_argument("--status", default=None)
        if name == "zip":
            s.add_argument("--out", required=True)
            s.add_argument("--include", nargs="*", default=[])
        if name == "check":
            s.add_argument("--out", default=None, help="write the problems as JSON here (new file)")
        if name == "final":
            s.add_argument("--check-report", required=True)
            s.add_argument("--zip", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "job":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            raise SystemExit("job: no command after --")
        return run_job(a.out, cmd)
    if a.cmd == "locate":
        rec = locate(a.inputs, a.run1_root, _conditions(a), hash_files=not a.no_hash)
        print_located(rec)
        _write_new(a.out, rec)
        return 1 if rec["problems"] else 0
    if a.cmd == "jobs":
        located = json.loads(Path(a.located).read_text(encoding="utf-8"))
        if located.get("problems"):
            raise SystemExit(f"located artifacts have problems: {located['problems']}")
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
        test = {k: getattr(a, k) for k in ("model_path", "device", "repo_root", "limit_windows", "limit_hellaswag",
                                           "limit_dev", "limit_calib") if getattr(a, k) is not None}
        jobs = stage_jobs(a.stage, located, a.results, a.status, a.run1_root, plan, test, _conditions(a),
                          a.smoke_condition)
        paths = write_jobs(a.stage, a.jobs_dir, jobs)
        for q in queues(jobs):
            print(f"{q}: " + ", ".join(x["name"] for x in jobs if x["queue"] == q))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd == "plan":
        located = json.loads(Path(a.located).read_text(encoding="utf-8"))
        pl = project(a.smoke_dir, located, a.working_bytes, a.budget_h, a.disk_gb, conditions=_conditions(a),
                     smoke_condition=a.smoke_condition, min_lmeval_examples=a.min_lmeval_examples)
        print_plan(pl)
        if a.out:
            _write_new(a.out, pl)
        return 1 if pl["over"] else 0
    meta = K.load_meta(a.jobs_dir, a.stage)
    status = a.status or DEFAULTS[a.stage]["status"]
    if a.cmd == "summary":
        print_summary(summary(meta, status))
        return 0
    if a.cmd == "failure-lines":
        for x in failure_lines(meta, status, a.logs or DEFAULTS[a.stage]["logs"]):
            print(x)
        return 0
    if a.cmd == "check":
        problems = check(meta, status)
        for x in problems:
            print("PROBLEM:", x)
        print(f"check: {len(meta)} jobs, {len(problems)} problems")
        if a.out:
            _write_new(a.out, {"n_jobs": len(meta), "problems": problems})
        return 1 if problems else 0
    if a.cmd == "final":
        reasons = K.final(meta, status, a.check_report, a.zip)
        for x in reasons:
            print("NOT OK:", x)
        print("run complete: every job ok, check clean, zip written" if not reasons else
              f"run not complete: {len(reasons)} reasons")
        return 1 if reasons else 0
    res = K.make_zip(a.out, meta, [*zip_roots(meta), *a.include], results_root=None)
    for x in res["missing"]:
        print("not in the zip (missing):", x)
    print(f"{res['path']}: {res['n_files']} files, {res['bytes']} bytes, sha256 {res['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
