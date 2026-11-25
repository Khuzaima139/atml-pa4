"""Run 1 of Task 1 on Kaggle (calibration and development only): job lists, projection, summary, check, zip.

python -m task1.kaggle_run1 jobs --stage preflight|run1 --jobs-dir DIR [--results DIR] [--artifacts DIR]
       [--status DIR] [--model-path PATH --device DEV --limit-calib N --limit-dev N]
python -m task1.kaggle_run1 plan --smoke RUN_JSON --working-bytes N [--budget-h 10.5] [--disk-gb 18]
python -m task1.kaggle_run1 summary --jobs-dir DIR [--stage run1] [--status DIR]
python -m task1.kaggle_run1 check --jobs-dir DIR [--stage run1] [--status DIR] [--out FILE]
python -m task1.kaggle_run1 zip --out FILE --jobs-dir DIR [--stage run1] [--include DIR ...]
python -m task1.kaggle_run1 final --jobs-dir DIR --check-report FILE --zip FILE [--stage run1] [--status DIR]

summary, check and zip report incomplete jobs and missing files instead of stopping, so that all three run after a
failed job; final returns 1 if any job is not ok, the check report is missing or lists problems, or the zip is
missing.

Queues (run1): GPU 0 runs calib_stats, probe quant, the two Q3 selections and the Q1/Q2/Q3 compress jobs; GPU 1
runs compress for fp16, P1, P2 and 2:4, then (after calib_stats is done) probe prune, the two P3 selections and
the P3 compress jobs. Preflight runs one condition of each kind with one calibration and one development sequence.
No job reads a held-out evaluation file.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from . import config as C
from . import storage

STAGES = ("preflight", "run1")
DEFAULTS = {
    "run1": {"results": "results/task1", "artifacts": "/kaggle/working/artifacts", "status": "notes/kaggle_status",
             "logs": "notes/kaggle_logs", "prefix": "", "limits": {}},
    "preflight": {"results": "results_smoke/preflight", "artifacts": "/kaggle/working/artifacts_smoke",
                  "status": "notes/kaggle_status_smoke", "logs": "notes/kaggle_logs_smoke", "prefix": "pf_",
                  "limits": {"limit_calib": 1, "limit_dev": 1}},
}
RUN1_GPU0_COMPRESS = ("q1_b8", "q1_b4", "q1_b3", "q2_b8", "q2_b4", "q2_b3")
RUN1_GPU1_COMPRESS = ("fp16", "p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p2_24_s50")
PREFLIGHT_GPU0_COMPRESS = ("q1_b3", "q2_b3")
PREFLIGHT_GPU1_COMPRESS = ("p1_s50", "p2_s50", "p2_24_s50")
RUN1_MANIFEST_COPY = "artifact_manifest.json"  # task1.evaluate_condition.RUN1_MANIFEST_COPY
SAFETY = 1.15
P3_CANDIDATES = 1 + len(C.P3_MEASURES) * len(C.P3_LAMBDAS)
Q3_CANDIDATES = 1 + len(C.Q3_MEASURES)
RESULTS_BYTES_PER_JOB = 50_000_000  # generous allowance for run JSONs and per-example files


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def _test_args(entry: str, t: dict) -> list:
    out = []
    if t.get("model_path"):
        out += ["--model-path", str(t["model_path"])]
    if t.get("device"):
        out += ["--device", t["device"]]
    if t.get("limit_calib") is not None:
        out += ["--limit-calib", str(t["limit_calib"])]
    if t.get("limit_dev") is not None and entry in ("task1.select", "task1.compress"):
        out += ["--limit-dev", str(t["limit_dev"])]
    return out


def stage_jobs(stage: str, results=None, artifacts=None, status=None, test=None, python=sys.executable) -> list:
    """Job specs with metadata: name, gpu, entry, kind, out, condition, artifacts, cmd, wait_for."""
    d = DEFAULTS[stage]
    R = Path(results or d["results"])
    A = Path(artifacts or d["artifacts"])
    S = Path(status or d["status"])
    t = {**d["limits"], **(test or {})}
    pre = d["prefix"]
    jobs = []

    def done(name):
        return str(S / f"{pre}{name}.done")

    def add(gpu, name, entry, out, args, deps=(), kind=None, condition=None, art=None):
        cmd = [python, "-m", entry, *args, "--out", str(out), *_test_args(entry, t)]
        jobs.append({"name": pre + name, "gpu": str(gpu), "entry": entry, "kind": kind or entry.split(".")[1],
                     "out": str(out), "condition": condition, "artifacts": str(art) if art else None, "cmd": cmd,
                     "wait_for": [done(x) for x in deps]})

    cs = R / "calib_stats"

    def compress(gpu, cond, deps=(), allocation=None):
        fam = C.CONDITIONS[cond]["family"]
        args = ["--condition", cond, "--artifacts", str(A / cond)]
        if fam in ("Q2", "Q3"):
            args += ["--calib-stats", str(cs)]
        if allocation:
            args += ["--allocation", str(R / allocation / "selected_allocation.json")]
        dep = list(deps) + (["calib_stats"] if fam in ("Q2", "Q3") else [])
        add(gpu, f"compress_{cond}", "task1.compress", R / "run1" / cond, args, dep, "compress", cond, A / cond)

    def select(gpu, kind, target):
        probe = "probe_prune" if kind == "p3" else "probe_quant"
        add(gpu, f"select_{kind}_{target}", "task1.select", R / f"select_{kind}_{target}",
            ["--kind", kind, "--target", target, "--probe", str(R / probe), "--calib-stats", str(cs)],
            [probe], "select", (C.P3_TARGETS if kind == "p3" else C.Q3_TARGETS)[target]["condition"])

    def probe(gpu, kind):
        add(gpu, f"probe_{kind}", "task1.probe", R / f"probe_{kind}", ["--kind", kind, "--calib-stats", str(cs)],
            ["calib_stats"], "probe")

    if stage == "run1":
        add(0, "calib_stats", "task1.calib_stats", cs, [], (), "calib_stats")
        probe(0, "quant")
        select(0, "q3", "int4")
        select(0, "q3", "int3")
        for c in RUN1_GPU0_COMPRESS:
            compress(0, c)
        for target in ("int4", "int3"):
            compress(0, C.Q3_TARGETS[target]["condition"], [f"select_q3_{target}"], f"select_q3_{target}")
        for c in RUN1_GPU1_COMPRESS:
            compress(1, c)
        probe(1, "prune")
        select(1, "p3", "s50")
        select(1, "p3", "s75")
        for target in ("s50", "s75"):
            compress(1, C.P3_TARGETS[target]["condition"], [f"select_p3_{target}"], f"select_p3_{target}")
    else:
        add(0, "calib_stats", "task1.calib_stats", cs, [], (), "calib_stats")
        probe(0, "quant")
        select(0, "q3", "int3")
        for c in PREFLIGHT_GPU0_COMPRESS:
            compress(0, c)
        compress(0, "q3_int3", ["select_q3_int3"], "select_q3_int3")
        for c in PREFLIGHT_GPU1_COMPRESS:
            compress(1, c)
        probe(1, "prune")
        select(1, "p3", "s50")
        compress(1, "p3_s50", ["select_p3_s50"], "select_p3_s50")
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)
    return jobs


def write_jobs(stage: str, jobs_dir, jobs: list) -> dict:
    """<stage>_gpu0.json / <stage>_gpu1.json (queue format) and <stage>_meta.json; refuses existing files."""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for gpu in ("0", "1"):
        q = [{"name": j["name"], "cmd": j["cmd"], "wait_for": j["wait_for"]} for j in jobs if j["gpu"] == gpu]
        paths[gpu] = jobs_dir / f"{stage}_gpu{gpu}.json"
        with open(paths[gpu], "x", encoding="utf-8") as fh:
            json.dump(q, fh, indent=2)
            fh.write("\n")
    paths["meta"] = jobs_dir / f"{stage}_meta.json"
    with open(paths["meta"], "x", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=2)
        fh.write("\n")
    return paths


def load_meta(jobs_dir, stage: str) -> list:
    return json.loads((Path(jobs_dir) / f"{stage}_meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------
# Projection from the smoke timings.

def artifact_bytes(cond: str, shapes, n_layers: int) -> int:
    """Stored tensor bytes of a condition's artifact (Q3: the Q2 budget, an upper bound)."""
    spec = C.CONDITIONS[cond]
    fam = spec["family"]
    if fam == "baseline":
        return 0
    if fam.startswith("P"):
        return sum(-(-(a * b) // 8) for a, b in shapes)
    bits = spec.get("bits") or C.Q3_TARGETS[{"q2_b4": "int4", "q2_b3": "int3"}[spec["budget"]]]["bits"]
    total = storage.packed_quant_bytes(shapes, bits)
    if fam in ("Q2", "Q3"):
        hidden, kv = shapes[0][1], shapes[1][0]
        total += n_layers * 2 * (2 * hidden + kv)  # folded norms and v_proj bias (FP16)
    return total


def project(smoke: dict, working_bytes: int, budget_h: float = 10.5, disk_gb: float = 18.0,
            safety: float = SAFETY) -> dict:
    """Per-job time projection (seconds), queue end times and /kaggle/working use for the run1 job lists."""
    u = {k: v["wall_s"] for k, v in smoke["steps"]["units"].items()}
    cnt = smoke["steps"]["counts"]
    shapes = [tuple(s) for s in cnt["eligible_shapes"]]
    nL = cnt["n_layers"]
    art = smoke["steps"]["artifact"]
    L, R, Pc, Pd = u["load"], u["restore"], u["calib_pass"], u["dev_pass"]
    qblock, pblock = u["quant_build"] / nL, u["p1_build"] / nL
    b8 = art["stored_tensor_bytes"]
    header = art["file_bytes"] - art["stored_tensor_bytes"] + art["manifest_bytes"]

    def save_verify(cond):
        f = artifact_bytes(cond, shapes, nL) / b8
        return u["artifact_save"] * f + R + max(u["artifact_verify"] - R, 0.0) * f

    def compress_s(cond):
        fam = C.CONDITIONS[cond]["family"]
        build = {"baseline": u["p1_build"], "P1": u["p1_build"], "P2": u["wanda_build"], "P2_2of4": u["wanda_build"],
                 "P3": u["wanda_build"]}.get(fam, u["quant_build"])
        return L + build + Pc + Pd + save_verify(cond)

    formulas = {
        "calib_stats": (L + u["capture"] + nL * u["awq_block"], "load + capture + n_layers x awq_block"),
        "probe_quant": (L + Pc + u["capture"] + nL * u["qs2_block"] + 4 * nL * (R + qblock + Pc),
                        "load + reference pass + QS2 (capture + n_layers x qs2_block) + QS1 (4 n_layers x "
                        "(restore + one-block quantization + calibration pass))"),
        "probe_prune": (L + Pc + 2 * nL * (R + pblock + Pc),
                        "load + reference pass + S2 (2 n_layers x (restore + one-block pruning + calibration pass))"),
        "select_p3": (L + P3_CANDIDATES * (R + u["wanda_build"] + Pd + Pc),
                      f"load + {P3_CANDIDATES} x (restore + sequential Wanda build + development + calibration pass)"),
        "select_q3": (L + Q3_CANDIDATES * (R + u["quant_build"] + Pd + Pc),
                      f"load + {Q3_CANDIDATES} x (restore + quantized build + development + calibration pass)"),
    }
    jobs = stage_jobs("run1")
    rows = []
    for j in jobs:
        key = j["name"].rsplit("_", 1)[0] if j["kind"] == "select" else j["name"]
        if j["kind"] == "compress":
            sec, how = compress_s(j["condition"]), "load + build + calibration and development passes + artifact"
        else:
            sec, how = formulas[key]
        rows.append({"name": j["name"], "gpu": j["gpu"], "raw_s": sec, "projected_s": sec * safety, "formula": how,
                     "artifact_bytes": artifact_bytes(j["condition"], shapes, nL) + header if j["kind"] == "compress"
                     else 0})
    by = {r["name"]: r for r in rows}
    end0, t = {}, 0.0
    for r in rows:
        if r["gpu"] == "0":
            t += r["projected_s"]
            end0[r["name"]] = t
    stagger = 60.0
    t1 = stagger
    for r in rows:
        if r["gpu"] != "1":
            continue
        if r["name"] == "probe_prune":
            t1 = max(t1, end0["calib_stats"])
        t1 += r["projected_s"]
    q = {"0": max(end0.values()), "1": t1}
    disk = working_bytes + sum(r["artifact_bytes"] for r in rows) + RESULTS_BYTES_PER_JOB * len(rows)
    limits = {"queue_hours": budget_h, "disk_bytes": int(disk_gb * 1e9)}
    over = [f"GPU {g} queue {s / 3600:.2f} h > {budget_h} h" for g, s in q.items() if s > budget_h * 3600]
    if disk > limits["disk_bytes"]:
        over.append(f"projected /kaggle/working use {disk / 1e9:.2f} GB > {disk_gb} GB")
    return {"rows": rows, "queue_end_s": q, "queue_end_h": {g: s / 3600 for g, s in q.items()},
            "disk_bytes": disk, "working_bytes_now": working_bytes, "limits": limits, "safety": safety,
            "stagger_s": stagger, "over": over, "jobs_by_name": by}


def print_plan(p: dict):
    print(f"{'job':<24}{'gpu':>4}{'raw min':>10}{'proj min':>10}{'artifact GB':>13}")
    for r in p["rows"]:
        print(f"{r['name']:<24}{r['gpu']:>4}{r['raw_s'] / 60:>10.1f}{r['projected_s'] / 60:>10.1f}"
              f"{r['artifact_bytes'] / 1e9:>13.3f}")
    for g, h in p["queue_end_h"].items():
        print(f"GPU {g} queue ends after {h:.2f} h (safety factor {p['safety']}, limit {p['limits']['queue_hours']} h)")
    print(f"/kaggle/working projected {p['disk_bytes'] / 1e9:.2f} GB (now {p['working_bytes_now'] / 1e9:.2f} GB, "
          f"limit {p['limits']['disk_bytes'] / 1e9:.1f} GB)")
    print("OVER LIMIT: " + "; ".join(p["over"]) if p["over"] else "within limits")


# ---------------------------------------------------------------------------------------------------------------
# Summary, check and zip.

def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            raise KeyError(dotted)
        d = d[k]
    return d


def _status(status_dir: Path, name: str) -> dict:
    code = None
    p = status_dir / f"{name}.exit"
    if p.is_file():
        try:
            code = int(p.read_text().strip())
        except ValueError:
            code = None
    rec = _load_json(status_dir / f"{name}.json") or {}
    return {"exit": code, "done": (status_dir / f"{name}.done").exists(), "wall_s": rec.get("wall_s")}


def summary(meta: list, status_dir) -> dict:
    status_dir = Path(status_dir)
    jobs, conds, incomplete = [], [], []
    for j in meta:
        st = _status(status_dir, j["name"])
        run = _load_json(Path(j["out"]) / "run.json") or {}
        peaks = run.get("peak_vram_allocated_bytes") or {}
        try:
            peak = max(peaks.values()) if peaks else None
        except (AttributeError, TypeError, ValueError):
            peak = None
        jobs.append({"name": j["name"], "gpu": j["gpu"], "exit": st["exit"], "done": st["done"],
                     "run_status": run.get("status"), "wall_s": st["wall_s"], "peak_vram_allocated_bytes": peak})
        if j["kind"] != "compress":
            continue
        if run.get("status") != "ok":
            incomplete.append(f"{j['condition']}: run status {run.get('status')}")
            continue
        try:
            s = run["steps"]
            fam = C.CONDITIONS[j["condition"]]["family"]
            conds.append({"condition": j["condition"], "family": fam,
                          "zero_fraction": 1 - s["build"]["nonzeros"]["nonzero_fraction"],
                          "nominal_bits": s["storage"]["nominal_bits_per_eligible_weight"],
                          "logical_eligible_bytes": s["artifact"]["logical_bytes"]["eligible"],
                          "logical_whole_model_bytes": s["artifact"]["logical_bytes"]["whole_model"],
                          "file_bytes": s["artifact"]["file"]["bytes"],
                          "calib_mean_nll": s["calib_nll"]["c4_calibration"]["mean_nll"],
                          "dev_mean_nll": s["calib_nll"]["c4_development"]["mean_nll"]})
        except (KeyError, TypeError) as exc:
            incomplete.append(f"{j['condition']}: run.json lacks {exc}")
    return {"jobs": jobs, "conditions": conds, "incomplete": incomplete}


def print_summary(sm: dict):
    print(f"{'job':<24}{'gpu':>4}{'exit':>6}{'run':>8}{'wall min':>10}{'peak GiB':>10}")
    for r in sm["jobs"]:
        w = f"{r['wall_s'] / 60:.1f}" if r["wall_s"] is not None else "-"
        v = f"{r['peak_vram_allocated_bytes'] / 2**30:.2f}" if r["peak_vram_allocated_bytes"] else "-"
        print(f"{r['name']:<24}{r['gpu']:>4}{str(r['exit']):>6}{str(r['run_status']):>8}{w:>10}{v:>10}")
    print()
    print(f"{'condition':<12}{'zeros':>8}{'bits':>7}{'logical MB':>12}{'whole MB':>10}{'file MB':>9}"
          f"{'calib NLL':>11}{'dev NLL':>9}")
    for c in sm["conditions"]:
        print(f"{c['condition']:<12}{c['zero_fraction']:>8.4f}{c['nominal_bits']:>7.2f}"
              f"{c['logical_eligible_bytes'] / 1e6:>12.1f}{c['logical_whole_model_bytes'] / 1e6:>10.1f}"
              f"{c['file_bytes'] / 1e6:>9.1f}{c['calib_mean_nll']:>11.4f}{c['dev_mean_nll']:>9.4f}")
    for x in sm.get("incomplete", []):
        print("not in the condition table:", x)


# Files and keys that later steps read (task1.select, task1.compress, task1.evaluate_condition, the import).
REQUIRED = {
    "calib_stats": {"files": ("run.json", "dense_stats.safetensors", "dense_stats.manifest.json",
                              "awq_scales.safetensors", "awq_scales.manifest.json", "awq_search.json"),
                    "keys": ("status", "config.model_path", "config.limits.calib", "meta.git_commit",
                             "steps.files.dense_stats.sha256", "steps.files.awq_scales.sha256")},
    "probe": {"files": ("run.json", "sensitivity.json", "probe_rows.jsonl.gz", "probe_calib_nll.jsonl.gz"),
              "keys": ("status", "config.kind", "config.model_path", "config.limits.calib", "meta.git_commit",
                       "steps.sensitivity.sha256")},
    "select": {"files": ("run.json", "candidates.jsonl.gz", "candidate_nll.jsonl.gz", "selected_allocation.json"),
               "keys": ("status", "steps.selection.winner", "steps.selection.sha256")},
    "compress": {"files": ("run.json", "projections.jsonl.gz", "calib_nll_c4_calibration.jsonl.gz",
                           "calib_nll_c4_development.jsonl.gz"),
                 "keys": ("status", "config.entry_point", "config.condition", "config.model_path", "config.limits.dev",
                          "meta.git_commit", "meta.git_dirty", "steps.artifact.manifest_sha256",
                          "steps.artifact.file.bytes", "steps.artifact.file.sha256",
                          "steps.artifact.logical_bytes.eligible", "steps.artifact.logical_bytes.whole_model",
                          "steps.calib_nll.c4_development.mean_nll", "steps.calib_nll.c4_calibration.mean_nll",
                          "steps.build.nonzeros.nonzero_fraction", "steps.storage.nominal_bits_per_eligible_weight",
                          "steps.held_out_guard.attempts")},
}
SELECTED_KEYS = ("condition", "candidate_id", "totals", "development.mean_nll", "code_commit", "model_path")
MANIFEST_KEYS = ("eval_id", "condition", "family", "kind", "model", "matrices", "extra_tensors", "logical_bytes",
                 "tensors", "stored_tensor_bytes", "file.sha256", "file.bytes")


def check(meta: list, status_dir, hash_artifacts: bool = True) -> list:
    """Problems with the outputs of every job (empty when all files and keys later steps read are present)."""
    from common.io import sha256_file

    status_dir = Path(status_dir)
    problems = []
    for j in meta:
        try:
            _check_job(j, status_dir, hash_artifacts, problems, sha256_file)
        except Exception as exc:  # a malformed output is reported, the other jobs are still checked
            problems.append(f"{j['name']}: check error {exc!r}")
    return problems


def _check_job(j: dict, status_dir: Path, hash_artifacts: bool, problems: list, sha256_file):
    st = _status(status_dir, j["name"])
    if st["exit"] != 0 or not st["done"]:
        problems.append(f"{j['name']}: exit {st['exit']}, done {st['done']}")
        return
    out = Path(j["out"])
    req = REQUIRED[j["kind"]]
    for f in req["files"]:
        if not (out / f).is_file():
            problems.append(f"{j['name']}: missing {out / f}")
    run = _load_json(out / "run.json")
    if run is None:
        return
    for k in req["keys"]:
        try:
            _get(run, k)
        except KeyError:
            problems.append(f"{j['name']}: run.json lacks {k}")
    if run.get("status") != "ok":
        problems.append(f"{j['name']}: run status {run.get('status')}")
    if j["kind"] == "probe":
        if (out / "sensitivity.json").is_file() and sha256_file(out / "sensitivity.json") != \
                run.get("steps", {}).get("sensitivity", {}).get("sha256"):
            problems.append(f"{j['name']}: sensitivity.json sha256 differs from run.json")
    if j["kind"] == "select":
        sel = _load_json(out / "selected_allocation.json") or {}
        for k in SELECTED_KEYS:
            try:
                _get(sel, k)
            except KeyError:
                problems.append(f"{j['name']}: selected_allocation.json lacks {k}")
        if sel.get("condition") != j["condition"]:
            problems.append(f"{j['name']}: selected allocation is for {sel.get('condition')}")
    if j["kind"] == "compress":
        if run.get("steps", {}).get("held_out_guard", {}).get("attempts"):
            problems.append(f"{j['name']}: held-out file access attempts recorded")
        art = Path(j["artifacts"])
        man_path, st_path = art / "manifest.json", art / "artifact.safetensors"
        if not man_path.is_file() or not st_path.is_file():
            problems.append(f"{j['name']}: artifact files missing in {art}")
            return
        man = _load_json(man_path) or {}
        for k in MANIFEST_KEYS:
            try:
                _get(man, k)
            except KeyError:
                problems.append(f"{j['name']}: manifest lacks {k}")
        if sha256_file(man_path) != run.get("steps", {}).get("artifact", {}).get("manifest_sha256"):
            problems.append(f"{j['name']}: manifest sha256 differs from run.json")
        if man.get("condition") != j["condition"]:
            problems.append(f"{j['name']}: manifest is for {man.get('condition')}")
        if st_path.stat().st_size != man.get("file", {}).get("bytes"):
            problems.append(f"{j['name']}: artifact size differs from the manifest")
        if hash_artifacts and sha256_file(st_path) != man.get("file", {}).get("sha256"):
            problems.append(f"{j['name']}: artifact sha256 differs from the manifest")
        if C.CONDITIONS[j["condition"]]["family"] in ("P3", "Q3"):
            alloc = next((a for a in j["cmd"] if str(a).endswith("selected_allocation.json")), None)
            if alloc is None or not Path(alloc).is_file():
                problems.append(f"{j['name']}: selected allocation file missing: {alloc}")
            elif man.get("allocation", {}).get("sha256") != sha256_file(alloc):
                problems.append(f"{j['name']}: manifest allocation sha256 differs from the selected allocation")


def make_zip(out, meta: list, include=(), results_root=None) -> dict:
    """Zip the results root, the include directories and every artifact manifest (as run-1 copies).

    Missing roots and missing manifests (of failed jobs) are skipped and listed under "missing".
    """
    out = Path(out)
    roots = [Path(results_root)] if results_root else []
    roots += [Path(d) for d in include]
    names, missing = [], []
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if not root.is_dir():
                missing.append(str(root))
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and "smoke" not in p.relative_to(root).as_posix():
                    zf.write(p, p.as_posix())
                    names.append(p.as_posix())
        for j in meta:
            if j["kind"] != "compress":
                continue
            arc = (Path(j["out"]) / RUN1_MANIFEST_COPY).as_posix()
            assert arc not in names, arc
            man = Path(j["artifacts"]) / "manifest.json"
            if not man.is_file():
                missing.append(str(man))
                continue
            zf.write(man, arc)
            names.append(arc)
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": h.hexdigest(), "n_files": len(names),
            "names": names, "missing": missing}


def final(meta: list, status_dir, check_report, zip_path) -> list:
    """Reasons the run is not complete (empty when every job is ok, the check found nothing and the zip exists)."""
    status_dir = Path(status_dir)
    reasons = []
    if not Path(zip_path).is_file():
        reasons.append(f"zip missing: {zip_path}")
    for j in meta:
        st = _status(status_dir, j["name"])
        if st["exit"] != 0 or not st["done"]:
            reasons.append(f"job {j['name']}: exit {st['exit']}, done {st['done']}")
    rep = _load_json(check_report)
    if not isinstance(rep, dict) or not isinstance(rep.get("problems"), list):
        reasons.append(f"check report missing or unreadable: {check_report}")
    elif rep["problems"]:
        reasons.append(f"check found {len(rep['problems'])} problems")
    return reasons


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    for k in ("results", "artifacts", "status", "model_path", "device"):
        j.add_argument("--" + k.replace("_", "-"), default=None)
    j.add_argument("--limit-calib", type=int, default=None)
    j.add_argument("--limit-dev", type=int, default=None)
    p = sub.add_parser("plan")
    p.add_argument("--smoke", required=True)
    p.add_argument("--working-bytes", type=int, required=True)
    p.add_argument("--budget-h", type=float, default=10.5)
    p.add_argument("--disk-gb", type=float, default=18.0)
    p.add_argument("--out", default=None, help="write the projection JSON here (new file)")
    for name in ("summary", "check", "zip", "final"):
        s = sub.add_parser(name)
        s.add_argument("--jobs-dir", required=True)
        s.add_argument("--stage", choices=STAGES, default="run1")
        s.add_argument("--status", default=None)
        if name == "zip":
            s.add_argument("--out", required=True)
            s.add_argument("--results", default=None)
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
        jobs = stage_jobs(a.stage, a.results, a.artifacts, a.status, test)
        paths = write_jobs(a.stage, a.jobs_dir, jobs)
        for g in ("0", "1"):
            print(f"GPU {g}: " + ", ".join(x["name"] for x in jobs if x["gpu"] == g))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd == "plan":
        smoke = json.loads(Path(a.smoke).read_text(encoding="utf-8"))
        if smoke.get("status") != "ok":
            raise SystemExit(f"smoke run is not finished ok: {smoke.get('status')}")
        pl = project(smoke, a.working_bytes, a.budget_h, a.disk_gb)
        print_plan(pl)
        if a.out:
            with open(a.out, "x", encoding="utf-8") as fh:
                json.dump({k: v for k, v in pl.items() if k != "jobs_by_name"}, fh, indent=2, sort_keys=True)
        return 1 if pl["over"] else 0
    meta = load_meta(a.jobs_dir, a.stage)
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
        reasons = final(meta, status, a.check_report, a.zip)
        for x in reasons:
            print("NOT OK:", x)
        print("run complete: every job ok, check clean, zip written" if not reasons else
              f"run not complete: {len(reasons)} reasons")
        return 1 if reasons else 0
    res = make_zip(a.out, meta, a.include, a.results or DEFAULTS[a.stage]["results"])
    for x in res["missing"]:
        print("not in the zip (missing):", x)
    print(f"{res['path']}: {res['n_files']} files, {res['bytes']} bytes, sha256 {res['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
