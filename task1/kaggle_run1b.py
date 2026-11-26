"""Run 1b of Task 1 on Kaggle (Q3 design v2, calibration and development only): job lists, calib_stats check,
projection, summary, check, zip.

python -m task1.kaggle_run1b jobs --stage preflight|run1b --jobs-dir DIR [--results DIR] [--artifacts DIR]
       [--status DIR] [--calib-stats DIR] [--parts A-B C-D] [--model-path PATH --device DEV --limit-calib N
       --limit-dev N]
python -m task1.kaggle_run1b calib-check --calib-stats DIR [--out FILE]
python -m task1.kaggle_run1b plan --smoke RUN_JSON --working-bytes N [--budget-h 10.5] [--disk-gb 18] [--out FILE]
python -m task1.kaggle_run1b summary --jobs-dir DIR [--stage run1b] [--status DIR]
python -m task1.kaggle_run1b check --jobs-dir DIR [--stage run1b] [--status DIR] [--out FILE]
python -m task1.kaggle_run1b zip --out FILE --jobs-dir DIR [--stage run1b] [--include DIR ...]
python -m task1.kaggle_run1b final --jobs-dir DIR --check-report FILE --zip FILE [--stage run1b] [--status DIR]

Queues (run1b): GPU 0 runs probe quant_v2 on blocks 0-13, then (after both probe parts are done) select q3v2 int4,
then compress q3_int4; GPU 1 runs probe quant_v2 on blocks 14-27, then (after both parts) select q3v2 int3, then
compress q3_int3. The calib_stats outputs are the committed run-1 files (results/task1/calib_stats). Preflight runs
the same graph with one calibration and one development sequence and its own calib_stats run on GPU 0 (the
committed files were made with all calibration sequences). No job reads a held-out evaluation file.
Outputs: results/task1/probe_quant_v2/blocks_AA_BB/, results/task1/select_q3v2_<budget>/, results/task1/run1b/<ID>/;
artifacts under /kaggle/working/artifacts/<ID>/.
"""
import argparse
import json
import sys
from pathlib import Path

from . import config as C
from . import kaggle_run1 as K

STAGES = ("preflight", "run1b")
DEFAULTS = {
    "run1b": {"results": "results/task1", "artifacts": "/kaggle/working/artifacts", "status": "notes/kaggle_status",
              "logs": "notes/kaggle_logs", "prefix": "", "limits": {}, "calib_stats": "results/task1/calib_stats"},
    "preflight": {"results": "results_smoke/preflight", "artifacts": "/kaggle/working/artifacts_smoke",
                  "status": "notes/kaggle_status_smoke", "logs": "notes/kaggle_logs_smoke", "prefix": "pf_",
                  "limits": {"limit_calib": 1, "limit_dev": 1}, "calib_stats": None},
}
PARTS = {"a": (0, 13), "b": (14, 27)}
PART_GPU = {"a": "0", "b": "1"}
BUDGET_GPU = {"int4": "0", "int3": "1"}
Q3V2_CANDIDATES = 1 + len(C.Q3V2_MEASURES)
CALIB_FILES = ("run.json", "dense_stats.safetensors", "dense_stats.manifest.json", "awq_scales.safetensors",
               "awq_scales.manifest.json", "awq_search.json")


def part_dir(results, lo: int, hi: int) -> Path:
    return Path(results) / "probe_quant_v2" / f"blocks_{lo:02d}_{hi:02d}"


def parse_parts(texts) -> dict:
    out = {}
    for k, t in zip("ab", texts):
        lo, hi = (int(x) for x in t.split("-"))
        out[k] = (lo, hi)
    assert len(out) == 2 and out["a"][1] + 1 == out["b"][0] and out["a"][0] == 0, out
    return out


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def stage_jobs(stage: str, results=None, artifacts=None, status=None, calib_stats=None, test=None, parts=None,
               python=sys.executable) -> list:
    """Job specs with metadata: name, gpu, entry, kind, out, condition, artifacts, cmd, wait_for, target, blocks."""
    d = DEFAULTS[stage]
    R = Path(results or d["results"])
    A = Path(artifacts or d["artifacts"])
    S = Path(status or d["status"])
    t = {**d["limits"], **(test or {})}
    pre = d["prefix"]
    parts = parts or PARTS
    jobs = []

    def done(name):
        return str(S / f"{pre}{name}.done")

    def add(gpu, name, entry, out, args, deps=(), kind=None, condition=None, art=None, **extra):
        cmd = [python, "-m", entry, *args, "--out", str(out), *K._test_args(entry, t)]
        jobs.append({"name": pre + name, "gpu": str(gpu), "entry": entry, "kind": kind, "out": str(out),
                     "condition": condition, "artifacts": str(art) if art else None, "cmd": cmd,
                     "wait_for": [done(x) for x in deps], **extra})

    cs_deps = []
    if stage == "preflight":
        cs = Path(calib_stats) if calib_stats else R / "calib_stats"
        add(0, "calib_stats", "task1.calib_stats", cs, [], (), "calib_stats")
        cs_deps = ["calib_stats"]
    else:
        cs = Path(calib_stats or d["calib_stats"])
    part_names = [f"probe_quant_v2_{k}" for k in parts]
    pdirs = {k: part_dir(R, lo, hi) for k, (lo, hi) in parts.items()}

    def probe(k):
        lo, hi = parts[k]
        add(PART_GPU[k], f"probe_quant_v2_{k}", "task1.probe", pdirs[k],
            ["--kind", "quant_v2", "--blocks", f"{lo}-{hi}", "--calib-stats", str(cs)], cs_deps, "probe",
            blocks=[lo, hi])

    def select(target):
        cond = C.Q3_TARGETS[target]["condition"]
        add(BUDGET_GPU[target], f"select_q3v2_{target}", "task1.select", R / f"select_q3v2_{target}",
            ["--kind", "q3v2", "--target", target, "--probe", *[str(pdirs[k]) for k in parts], "--calib-stats",
             str(cs)], part_names, "select", cond, target=target)

    def compress(target):
        cond = C.Q3_TARGETS[target]["condition"]
        alloc = R / f"select_q3v2_{target}" / "selected_allocation.json"
        add(BUDGET_GPU[target], f"compress_{cond}", "task1.compress", R / "run1b" / cond,
            ["--condition", cond, "--artifacts", str(A / cond), "--calib-stats", str(cs), "--allocation", str(alloc)],
            [f"select_q3v2_{target}"] + cs_deps, "compress", cond, A / cond, target=target)

    for k, target in (("a", "int4"), ("b", "int3")):
        probe(k)
        select(target)
        compress(target)
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)
    return jobs


# ---------------------------------------------------------------------------------------------------------------
# calib_stats check (the committed run-1 files).

def calib_check(calib_dir) -> dict:
    """sha256 of the calib_stats files against their run record and manifests; tracked and unmodified in git."""
    from common.io import sha256_file

    from .calib_stats import provenance
    from .compress import git_file_status

    d = Path(calib_dir)
    problems = []
    rec = {"dir": str(d), "files": {}}
    for f in CALIB_FILES:
        p = d / f
        if not p.is_file():
            problems.append(f"missing {p}")
            continue
        g = git_file_status(p)
        rec["files"][f] = {"bytes": p.stat().st_size, "sha256": sha256_file(p), "git": g}
        if not (g.get("tracked") and g.get("clean")):
            problems.append(f"{p} is not tracked and unmodified: {g}")
    if not problems:
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        rec["run_status"] = run.get("status")
        rec["run_limits"] = run["config"]["limits"]
        rec["run_git_commit"] = run["meta"]["git_commit"]
        if run.get("status") != "ok":
            problems.append(f"calib_stats run status {run.get('status')}")
        if run["config"]["limits"].get("calib") is not None or run["config"]["model_path"] is not None:
            problems.append("calib_stats run used test options")
        try:
            rec["provenance"] = provenance(d)
        except SystemExit as exc:
            problems.append(str(exc))
        for k in ("dense_stats", "awq_scales"):
            man = json.loads((d / f"{k}.manifest.json").read_text(encoding="utf-8"))
            if man["file"]["sha256"] != run["steps"]["files"][k]["sha256"]:
                problems.append(f"{k} manifest sha256 differs from the run record")
        search = json.loads((d / "awq_search.json").read_text(encoding="utf-8"))
        if search["scales_file"]["sha256"] != run["steps"]["files"]["awq_scales"]["sha256"]:
            problems.append("awq_search.json names a different scales file")
    rec["problems"] = problems
    return rec


# ---------------------------------------------------------------------------------------------------------------
# Projection from the smoke timings.

def project(smoke: dict, working_bytes: int, budget_h: float = 10.5, disk_gb: float = 18.0, safety: float = K.SAFETY,
            parts=None) -> dict:
    """Per-job time projection (seconds), queue end times and /kaggle/working use for the run1b job lists."""
    parts = parts or PARTS
    u = {k: v["wall_s"] for k, v in smoke["steps"]["units"].items()}
    nU = len(C.Q3V2_UNITS)
    nb = len(C.Q3V2_BITS)
    L, R, Pc, Pd = u["load"], u["restore"], u["calib_pass"], u["dev_pass"]
    art = smoke["steps"]["artifact"]
    art_bytes = art["file_bytes"] + art["manifest_bytes"]

    def probe_s(lo, hi):
        n = hi - lo + 1
        return (L + Pc + u["capture"] + lo * u["block_forward"] + n * u["qs2v2_block"]
                + nb * nU * n * u["isolated_unit"])

    select_s = L + Q3V2_CANDIDATES * (R + u["quant_build"] + Pd + Pc)
    compress_s = L + u["quant_build"] + Pc + Pd + u["artifact_save"] + u["artifact_verify"]
    rows = []
    for j in stage_jobs("run1b", parts=parts):
        if j["kind"] == "probe":
            lo, hi = j["blocks"]
            sec = probe_s(lo, hi)
            how = (f"load + reference pass + capture + {lo} block forwards + {hi - lo + 1} x qs2v2_block + "
                   f"{nb * nU * (hi - lo + 1)} x isolated_unit (restore + largest-unit quantization + calibration "
                   f"pass)")
        elif j["kind"] == "select":
            sec, how = select_s, (f"load + {Q3V2_CANDIDATES} x (restore + v2 build + development + calibration "
                                  f"pass)")
        else:
            sec, how = compress_s, "load + v2 build + calibration and development passes + artifact save and verify"
        rows.append({"name": j["name"], "gpu": j["gpu"], "kind": j["kind"], "raw_s": sec, "projected_s": sec * safety,
                     "formula": how, "artifact_bytes": art_bytes if j["kind"] == "compress" else 0})
    by = {r["name"]: r for r in rows}
    stagger = 60.0
    start = {"0": 0.0, "1": stagger}
    probe_end = {r["gpu"]: start[r["gpu"]] + r["projected_s"] for r in rows if r["kind"] == "probe"}
    both = max(probe_end.values())
    q = {}
    for g in ("0", "1"):
        t = max(probe_end[g], both)
        for r in rows:
            if r["gpu"] == g and r["kind"] != "probe":
                t += r["projected_s"]
        q[g] = t
    disk = working_bytes + sum(r["artifact_bytes"] for r in rows) + K.RESULTS_BYTES_PER_JOB * len(rows)
    limits = {"queue_hours": budget_h, "disk_bytes": int(disk_gb * 1e9)}
    over = [f"GPU {g} queue {s / 3600:.2f} h > {budget_h} h" for g, s in q.items() if s > budget_h * 3600]
    if disk > limits["disk_bytes"]:
        over.append(f"projected /kaggle/working use {disk / 1e9:.2f} GB > {disk_gb} GB")
    return {"rows": rows, "queue_end_s": q, "queue_end_h": {g: s / 3600 for g, s in q.items()},
            "probe_end_s": probe_end, "disk_bytes": disk, "working_bytes_now": working_bytes, "limits": limits,
            "safety": safety, "stagger_s": stagger, "over": over, "jobs_by_name": by}


# ---------------------------------------------------------------------------------------------------------------
# Summary.

def summary(meta: list, status_dir) -> dict:
    """Job table and, per budget, every candidate (winner, collapse flags, bits histogram, bytes, NLL)."""
    from common.io import read_jsonl

    base = K.summary(meta, status_dir)
    budgets, incomplete = [], list(base["incomplete"])
    for j in meta:
        if j["kind"] != "select":
            continue
        out = Path(j["out"])
        sel = K._load_json(out / "selected_allocation.json")
        try:
            cands = read_jsonl(out / "candidates.jsonl.gz")
        except (FileNotFoundError, OSError, ValueError):
            cands = None
        if sel is None or cands is None:
            incomplete.append(f"{j['name']}: selection outputs missing")
            continue
        comp = next((x for x in meta if x["kind"] == "compress" and x.get("target") == j["target"]), None)
        crun = K._load_json(Path(comp["out"]) / "run.json") if comp else None
        try:
            budgets.append({
                "target": j["target"], "condition": j["condition"], "winner": sel["candidate_id"],
                "winner_collapsed_to_uniform": sel["collapsed_to_uniform"],
                "all_candidates_collapsed": sel["all_candidates_collapsed"], "budget_bytes": sel["totals"]["budget_bytes"],
                "candidates": [{"candidate_id": c["candidate_id"], "reference": c["reference"],
                                "collapsed_to_uniform": c["collapsed_to_uniform"],
                                "eligible_packed_bytes": c["totals"]["eligible_packed_bytes"],
                                "budget_bytes": c["totals"]["budget_bytes"], "bits_histogram": c["bits_histogram"],
                                "dev_mean_nll": c["development"]["mean_nll"],
                                "calib_mean_nll": c["calibration"]["mean_nll"], "selected": c["selected"]}
                               for c in cands],
                "compress": None if not crun or crun.get("status") != "ok" else {
                    "dev_mean_nll": crun["steps"]["calib_nll"]["c4_development"]["mean_nll"],
                    "calib_mean_nll": crun["steps"]["calib_nll"]["c4_calibration"]["mean_nll"],
                    "logical_eligible_bytes": crun["steps"]["artifact"]["logical_bytes"]["eligible"]}})
        except (KeyError, TypeError) as exc:
            incomplete.append(f"{j['name']}: selection outputs lack {exc}")
    return {"jobs": base["jobs"], "budgets": budgets, "incomplete": incomplete}


def _hist(h: dict) -> str:
    return " ".join(f"{u}:" + "/".join(str(h[u][str(b)]) for b in C.Q3V2_BITS) for u in h)


def print_summary(sm: dict):
    print(f"{'job':<28}{'gpu':>4}{'exit':>6}{'run':>8}{'wall min':>10}{'peak GiB':>10}")
    for r in sm["jobs"]:
        w = f"{r['wall_s'] / 60:.1f}" if r["wall_s"] is not None else "-"
        v = f"{r['peak_vram_allocated_bytes'] / 2**30:.2f}" if r["peak_vram_allocated_bytes"] else "-"
        print(f"{r['name']:<28}{r['gpu']:>4}{str(r['exit']):>6}{str(r['run_status']):>8}{w:>10}{v:>10}")
    for b in sm["budgets"]:
        print()
        print(f"{b['target']} ({b['condition']}): winner {b['winner']}, winner collapsed "
              f"{b['winner_collapsed_to_uniform']}, all candidates collapsed {b['all_candidates_collapsed']}, "
              f"budget {b['budget_bytes']:,d} B")
        print(f"  bits histogram per unit type, counts at {'/'.join(str(x) for x in C.Q3V2_BITS)} bits")
        print(f"  {'candidate':<24}{'ref':>5}{'coll':>6}{'sel':>5}{'bytes':>14}{'dev NLL':>10}{'calib NLL':>11}  histogram")
        for c in b["candidates"]:
            print(f"  {c['candidate_id']:<24}{str(c['reference'])[0]:>5}{str(c['collapsed_to_uniform'])[0]:>6}"
                  f"{str(c['selected'])[0]:>5}{c['eligible_packed_bytes']:>14,d}{c['dev_mean_nll']:>10.5f}"
                  f"{c['calib_mean_nll']:>11.5f}  {_hist(c['bits_histogram'])}")
        if b["compress"]:
            x = b["compress"]
            print(f"  compress {b['condition']}: dev NLL {x['dev_mean_nll']:.5f}, calib NLL {x['calib_mean_nll']:.5f}, "
                  f"logical eligible {x['logical_eligible_bytes']:,d} B")
    for x in sm.get("incomplete", []):
        print("incomplete:", x)


# ---------------------------------------------------------------------------------------------------------------
# Check.

def check(meta: list, status_dir, hash_artifacts: bool = True) -> list:
    """Problems with the outputs of every job: the run-1 checks plus the design-v2 files and keys."""
    from common.io import sha256_file

    status_dir = Path(status_dir)
    problems = []
    for j in meta:
        before = len(problems)
        try:
            K._check_job(j, status_dir, hash_artifacts, problems, sha256_file)
            if len(problems) == before:
                _check_v2(j, meta, problems)
        except Exception as exc:  # a malformed output is reported, the other jobs are still checked
            problems.append(f"{j['name']}: check error {exc!r}")
    return problems


def _check_v2(j: dict, meta: list, problems: list):
    out = Path(j["out"])
    if j["kind"] == "probe":
        sens = K._load_json(out / "sensitivity.json") or {}
        if sens.get("design") != C.Q3V2_DESIGN or sens.get("block_range") != j["blocks"]:
            problems.append(f"{j['name']}: sensitivity design {sens.get('design')} / blocks {sens.get('block_range')}")
        n = j["blocks"][1] - j["blocks"][0] + 1
        if len(sens.get("units", [])) != n * len(C.Q3V2_UNITS):
            problems.append(f"{j['name']}: {len(sens.get('units', []))} units, expected {n * len(C.Q3V2_UNITS)}")
    elif j["kind"] == "select":
        sel = K._load_json(out / "selected_allocation.json") or {}
        for k in ("design", "per_unit.bits", "per_matrix_bits", "bits_histogram", "chunking.chunk_groups",
                  "probe_union.n_units", "totals.eligible_packed_bytes", "totals.budget_bytes",
                  "collapsed_to_uniform", "all_candidates_collapsed"):
            try:
                K._get(sel, k)
            except KeyError:
                problems.append(f"{j['name']}: selected_allocation.json lacks {k}")
        if sel.get("design") != C.Q3V2_DESIGN:
            problems.append(f"{j['name']}: selected design {sel.get('design')}")
        t = sel.get("totals", {})
        if t.get("eligible_packed_bytes", 0) > t.get("budget_bytes", -1):
            problems.append(f"{j['name']}: packed bytes above the budget")
        n_parts = sum(1 for x in meta if x["kind"] == "probe")
        if sel.get("probe_union", {}).get("n_parts") != n_parts:
            problems.append(f"{j['name']}: selection joined {sel.get('probe_union', {}).get('n_parts')} probe parts")
    elif j["kind"] == "compress":
        man = K._load_json(Path(j["artifacts"]) / "manifest.json") or {}
        if man.get("design") != C.Q3V2_DESIGN or man.get("allocation", {}).get("design") != C.Q3V2_DESIGN:
            problems.append(f"{j['name']}: artifact manifest does not record design {C.Q3V2_DESIGN}")
        run = K._load_json(out / "run.json") or {}
        if run.get("steps", {}).get("allocation", {}).get("design") != C.Q3V2_DESIGN:
            problems.append(f"{j['name']}: run.json allocation design differs")


def zip_roots(meta: list) -> list:
    """Output roots of the run-1b jobs (probe parts' parent, selections, run1b condition runs), deduplicated."""
    roots = []
    for j in meta:
        if j["kind"] == "calib_stats":
            continue
        out = Path(j["out"])
        r = out.parent if j["kind"] in ("probe", "compress") else out
        if str(r) not in roots:
            roots.append(str(r))
    return roots


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    for k in ("results", "artifacts", "status", "calib_stats", "model_path", "device"):
        j.add_argument("--" + k.replace("_", "-"), default=None)
    j.add_argument("--limit-calib", type=int, default=None)
    j.add_argument("--limit-dev", type=int, default=None)
    j.add_argument("--parts", nargs=2, default=None, metavar="A-B", help="block ranges of the two probe parts")
    c = sub.add_parser("calib-check")
    c.add_argument("--calib-stats", required=True)
    c.add_argument("--out", default=None, help="write the check record as JSON here (new file)")
    p = sub.add_parser("plan")
    p.add_argument("--smoke", required=True)
    p.add_argument("--working-bytes", type=int, required=True)
    p.add_argument("--budget-h", type=float, default=10.5)
    p.add_argument("--disk-gb", type=float, default=18.0)
    p.add_argument("--out", default=None, help="write the projection JSON here (new file)")
    for name in ("summary", "check", "zip", "final"):
        s = sub.add_parser(name)
        s.add_argument("--jobs-dir", required=True)
        s.add_argument("--stage", choices=STAGES, default="run1b")
        s.add_argument("--status", default=None)
        if name == "zip":
            s.add_argument("--out", required=True)
            s.add_argument("--include", nargs="*", default=[])
        if name == "check":
            s.add_argument("--no-artifact-hash", action="store_true")
            s.add_argument("--out", default=None, help="write the problems as JSON here (new file)")
        if name == "final":
            s.add_argument("--check-report", required=True)
            s.add_argument("--zip", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "jobs":
        test = {k: getattr(a, k) for k in ("model_path", "device", "limit_calib", "limit_dev")
                if getattr(a, k) is not None}
        parts = parse_parts(a.parts) if a.parts else None
        jobs = stage_jobs(a.stage, a.results, a.artifacts, a.status, a.calib_stats, test, parts)
        paths = K.write_jobs(a.stage, a.jobs_dir, jobs)
        for g in ("0", "1"):
            print(f"GPU {g}: " + ", ".join(x["name"] for x in jobs if x["gpu"] == g))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd == "calib-check":
        rec = calib_check(a.calib_stats)
        for f, r in rec["files"].items():
            print(f"{r['bytes']:>12,d}  {r['sha256']}  {f}")
        for x in rec["problems"]:
            print("PROBLEM:", x)
        print("calib_stats check:", "ok" if not rec["problems"] else f"{len(rec['problems'])} problems")
        if a.out:
            with open(a.out, "x", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, sort_keys=True)
                fh.write("\n")
        return 1 if rec["problems"] else 0
    if a.cmd == "plan":
        smoke = json.loads(Path(a.smoke).read_text(encoding="utf-8"))
        if smoke.get("status") != "ok":
            raise SystemExit(f"smoke run is not finished ok: {smoke.get('status')}")
        pl = project(smoke, a.working_bytes, a.budget_h, a.disk_gb)
        K.print_plan(pl)
        if a.out:
            with open(a.out, "x", encoding="utf-8") as fh:
                json.dump({k: v for k, v in pl.items() if k != "jobs_by_name"}, fh, indent=2, sort_keys=True)
        return 1 if pl["over"] else 0
    meta = K.load_meta(a.jobs_dir, a.stage)
    status = a.status or DEFAULTS[a.stage]["status"]
    if a.cmd == "summary":
        print_summary(summary(meta, status))
        return 0
    if a.cmd == "check":
        problems = check(meta, status, hash_artifacts=not a.no_artifact_hash)
        for x in problems:
            print("PROBLEM:", x)
        print(f"check: {len(meta)} jobs, {len(problems)} problems")
        if a.out:
            with open(a.out, "x", encoding="utf-8") as fh:
                json.dump({"n_jobs": len(meta), "hash_artifacts": not a.no_artifact_hash, "problems": problems}, fh,
                          indent=2)
                fh.write("\n")
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
