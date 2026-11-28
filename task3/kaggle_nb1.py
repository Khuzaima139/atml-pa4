"""Task 3 Kaggle notebook 1: prefix banks, teacher collection, the four distillation runs, merges and own-student banks.

python -m task3.kaggle_nb1 data-check --out FILE
python -m task3.kaggle_nb1 model-files --out FILE
python -m task3.kaggle_nb1 jobs --stage smoke|main --jobs-dir DIR [--plan FILE] [--results DIR] [--status DIR]
       [--artifacts DIR] [--preflight FILE] [test options: --device --dtype --target-path --draft-path]
python -m task3.kaggle_nb1 smoke-check --jobs-dir DIR
python -m task3.kaggle_nb1 plan --jobs-dir DIR [--budget-h 10.5] [--session-h 11.5] [--elapsed-s S] [--vram-gib 14]
       [--out FILE]
python -m task3.kaggle_nb1 summary|check|manifest|zip|final --stage smoke|main --jobs-dir DIR ...

Inventory (main). Units: collect (teacher collection, D8); off_policy_6304 and off_policy_6305 (each: train after the
collection, merge, own-student bank; protocol AM1 puts the own bank right after its merge); on_policy_6304 and
on_policy_6305 (train, merge, own bank); bank_teacher, bank_original_draft, bank_exact_spec_original_gamma4 (needs the
passing preflight JSON and the committed sampler check). Two queues, one per GPU: the collection is first on GPU 0;
the other units are assigned by the plan, which projects every job from the smoke JSONs and picks the assignment with
the smallest makespan (off-policy units wait for the collection's .done). `jobs` refuses to write any stage unless the
preflight JSON passes both the FP32 and the FP16 rule (protocol AM2). Smoke (outputs under /kaggle/tmp/t3_smoke):
the collection on 2 + 2 prompts with K_short 2, 2 updates of each method, the merge of the on-policy smoke adapter, and
2 prompts of every bank kind (own bank from the smoke merge).
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path

from common.kaggle_queue import read_int

GiB = 1 << 30
SAFETY = 1.2
JOB_OVERHEAD_S = 60.0
STAGGER_S = 60.0
BANK_KINDS = ("teacher", "original_draft", "exact_spec_original_gamma4")
RUNS = ("off_policy_6304", "off_policy_6305", "on_policy_6304", "on_policy_6305")
UNITS = ("collect",) + RUNS + tuple(f"bank_{k}" for k in BANK_KINDS)
N_HELDOUT = 120
N_TRAIN = {"math": 500, "short": 300}
SMOKE_LIMITS = {"collect": {"limit_prompts_math": 2, "limit_prompts_short": 2, "limit_k_short": 2},
                "train_off": {"limit_prompts_math": 2, "limit_prompts_short": 2, "limit_updates": 2},
                "train_on": {"limit_updates": 2}, "bank": {"limit_prompts": 2}}
DEFAULTS = {
    "smoke": {"results": "/kaggle/tmp/t3_smoke", "status": "notes/kaggle_status_smoke",
              "logs": "notes/kaggle_logs_smoke", "artifacts": "/kaggle/tmp/t3_smoke/artifacts", "prefix": "smoke_"},
    "main": {"results": "results/task3", "status": "notes/kaggle_status", "logs": "notes/kaggle_logs",
             "artifacts": "/kaggle/working/t3_artifacts", "prefix": ""},
}
STAGES = tuple(DEFAULTS)
PREFLIGHT = "results/task3/preflight/nb1_preflight.json"
ZIP_EXCLUDE = ("smoke",)
TEST_KEYS = ("device", "dtype", "target_path", "draft_path")


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, IsADirectoryError):
        return None


def _write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gib(b):
    return None if b is None else b / GiB


# ---------------------------------------------------------------------------------------------------------------
# Setup checks.

def data_check() -> dict:
    """Held-out hash list committed and equal to a rebuild; training sets load with their hashes; the committed
    sampler check passes; the fixed-64 IDs file is committed."""
    from common.freeze import require_committed

    from task3 import data as D
    from task3 import heldout as H
    from task3.banks import SAMPLER_CHECK, require_sampler_check

    rec = require_committed(H.HELDOUT_LIST, "held-out hash list")
    same = H.read_heldout_list() == H.build_heldout_list()
    if not same:
        raise SystemExit("held-out hash list differs from a rebuild from the release files")
    items, prov = D.training_items()
    g = H.HeldoutGuard()
    g.check(items, "training prompt")
    sc = require_sampler_check(SAMPLER_CHECK)
    f64 = require_committed(Path(H.REPO_ROOT) / "task3" / "fixed64_ids.json", "fixed-64 IDs")
    return {"heldout_hash_list_sha256": rec["sha256"], "heldout_list_equals_rebuild": same,
            "training_items": {c: sum(1 for x in items if x["category"] == c) for c in ("math", "short")},
            "training_provenance": prov, "sampler_check": sc, "fixed64_sha256": f64["sha256"]}


def model_files() -> dict:
    """Download both models at the locked revisions and record the sha256 of every snapshot file."""
    from huggingface_hub import snapshot_download

    from task3 import data as D

    out = {}
    for key, rv in D.revisions().items():
        path = Path(snapshot_download(rv["repo"], revision=rv["revision"]))
        out[key] = {**rv, "path": str(path),
                    "files": {p.relative_to(path).as_posix(): _sha256(p) for p in sorted(path.rglob("*"))
                              if p.is_file()}}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Job lists.

def _flags(d: dict) -> list:
    out = []
    for k, v in d.items():
        if v is not None:
            out += ["--" + k.replace("_", "-"), str(v)]
    return out


def _unit_jobs(unit, R: Path, A: Path, prefix, t, python, preflight, limits_for) -> list:
    """Jobs of one unit in order: dict(name, kind, unit, out, run_json, cmd, deps (names), expect)."""
    dev = _flags({"device": t.get("device"), "dtype": t.get("dtype")})
    tp, dp = t.get("target_path"), t.get("draft_path")
    jobs = []
    if unit == "collect":
        out = R / "collection" / "teacher_collection.jsonl.gz"
        lim = limits_for("collect")
        cmd = [python, "-m", "task3.collect", "--out", str(out), *dev, *_flags({"teacher_path": tp}), *_flags(lim)]
        jobs.append({"name": prefix + "collect", "kind": "collect", "out": str(out),
                     "run_json": str(out.with_name("teacher_collection.json")), "cmd": cmd, "deps": [],
                     "expect": lim})
    elif unit.startswith("bank_"):
        kind = unit[len("bank_"):]
        out = R / "banks" / f"{kind}.jsonl.gz"
        lim = limits_for("bank")
        cmd = [python, "-m", "task3.banks", "--kind", kind, "--out", str(out), *dev,
               *_flags({"target_path": tp}), *_flags(lim)]
        if kind in ("original_draft", "exact_spec_original_gamma4") and dp:
            cmd += ["--draft-path", dp]
        if kind == "exact_spec_original_gamma4":
            cmd += ["--preflight", str(preflight)]
        jobs.append({"name": prefix + unit, "kind": "bank", "bank": kind, "out": str(out),
                     "run_json": str(out.with_name(f"{kind}.json")), "cmd": cmd, "deps": [], "expect": lim})
    else:
        method, seed = unit.rsplit("_", 1)
        out = R / "train" / unit
        adapter = A / "adapters" / unit
        lim = limits_for("train_off" if method == "off_policy" else "train_on")
        cmd = [python, "-m", "task3.train", "--method", method, "--seed", seed, "--out", str(out),
               "--adapter-dir", str(adapter), *dev, *_flags({"teacher_path": tp, "student_path": dp}), *_flags(lim)]
        deps = []
        if method == "off_policy":
            coll = R / "collection" / "teacher_collection.jsonl.gz"
            cmd += ["--collection", str(coll)]
            deps = [prefix + "collect"]
        jobs.append({"name": prefix + "train_" + unit, "kind": "train", "run": unit, "out": str(out),
                     "run_json": str(out / "run.json"), "adapter_dir": str(adapter), "cmd": cmd, "deps": deps,
                     "expect": lim})
        merged = A / "merged" / unit
        mjson = R / "merge" / f"{unit}.json"
        cmd = [python, "-m", "task3.merge", "--train-json", str(out / "run.json"), "--adapter-dir", str(adapter),
               "--merged-dir", str(merged), "--out", str(mjson), *dev,
               *_flags({"backbone_path": dp, "tokenizer_path": tp})]
        jobs.append({"name": prefix + "merge_" + unit, "kind": "merge", "run": unit, "out": str(mjson),
                     "run_json": str(mjson), "merged_dir": str(merged), "cmd": cmd,
                     "deps": [prefix + "train_" + unit], "expect": {}})
        bout = R / "banks" / f"own_{unit}.jsonl.gz"
        blim = limits_for("bank")
        cmd = [python, "-m", "task3.banks", "--kind", "own", "--run-name", unit, "--draft-path", str(merged),
               "--out", str(bout), *dev, *_flags({"target_path": tp}), *_flags(blim)]
        jobs.append({"name": prefix + "bank_own_" + unit, "kind": "bank", "bank": f"own_{unit}", "run": unit,
                     "out": str(bout), "run_json": str(bout.with_name(f"own_{unit}.json")), "cmd": cmd,
                     "deps": [prefix + "merge_" + unit], "expect": blim})
    for j in jobs:
        j["unit"] = unit
    return jobs


def queue_orders(assignment: dict) -> dict:
    """Order of units per queue: GPU 0 starts with the collection, then off-policy, on-policy and bank units; GPU 1
    runs on-policy units, then banks, then off-policy units (which wait for the collection)."""
    def rank(u, q):
        group = 0 if u == "collect" else 1 if u.startswith("off_") else 2 if u.startswith("on_") else 3
        if q == "gpu1":
            group = {2: 0, 3: 1, 1: 2}.get(group, group)
        return group, UNITS.index(u)

    return {q: sorted([u for u, qq in assignment.items() if qq == q], key=lambda u: rank(u, q))
            for q in ("gpu0", "gpu1")}


SMOKE_ASSIGNMENT = {"collect": "gpu0", "off_policy_6304": "gpu0", "on_policy_6304": "gpu1",
                    "bank_teacher": "gpu0", "bank_original_draft": "gpu1", "bank_exact_spec_original_gamma4": "gpu0"}


def stage_jobs(stage, results=None, status=None, artifacts=None, preflight=PREFLIGHT, plan=None, test=None,
               python=sys.executable) -> list:
    """Job specs with name, stage, gpu, queue, kind, unit, out, run_json, cmd, wait_for, expect."""
    d = DEFAULTS[stage]
    R, S, A = Path(results or d["results"]), Path(status or d["status"]), Path(artifacts or d["artifacts"])
    t = dict(test or {})
    if stage == "smoke":
        assignment = dict(SMOKE_ASSIGNMENT)

        def limits_for(k):
            return dict(SMOKE_LIMITS[k])
    else:
        if plan is None:
            raise SystemExit("main jobs need a plan")
        assignment = dict(plan["assignment"])

        def limits_for(k):
            return {}
    if assignment.get("collect") != "gpu0":
        raise SystemExit("the collection must be on GPU 0")
    jobs = []
    for q, units in queue_orders(assignment).items():
        for u in units:
            for j in _unit_jobs(u, R, A, d["prefix"], t, python, preflight, limits_for):
                if stage == "smoke" and j["kind"] == "merge" and u.startswith("off_"):
                    continue
                if stage == "smoke" and j["kind"] == "bank" and j["bank"].startswith("own_off"):
                    continue
                j.update(stage=stage, queue=q, gpu=q[-1],
                         wait_for=[str(S / f"{n}.done") for n in j.pop("deps")])
                jobs.append(j)
    return jobs


def queues(jobs) -> list:
    out = []
    for j in jobs:
        if j["queue"] not in out:
            out.append(j["queue"])
    return out


def write_jobs(stage, jobs_dir, jobs) -> dict:
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
    _write_new(paths["meta"], jobs)
    return paths


def load_meta(jobs_dir, stage) -> list:
    return json.loads((Path(jobs_dir) / f"{stage}_meta.json").read_text(encoding="utf-8"))


def _status(status_dir, name) -> dict:
    code = read_int(Path(status_dir) / f"{name}.exit")
    rec = _load_json(Path(status_dir) / f"{name}.json") or {}
    return {"exit": code, "done": (Path(status_dir) / f"{name}.done").exists(), "wall_s": rec.get("wall_s")}


def _run(j) -> dict:
    return _load_json(j["run_json"]) or {}


def _peak(run) -> int:
    vals = list((run.get("peak_vram_allocated_bytes") or {}).values())
    if run.get("peak_gpu_memory_bytes"):
        vals.append(run["peak_gpu_memory_bytes"])
    return max(vals or [0])


# ---------------------------------------------------------------------------------------------------------------
# Expected outputs and the per-job check (shared by smoke-check and check).

def expected(j) -> dict:
    from task3 import distill as T

    e = j["expect"]
    if j["kind"] == "collect":
        n_m = e.get("limit_prompts_math") or N_TRAIN["math"]
        n_s = e.get("limit_prompts_short") or N_TRAIN["short"]
        return {"n_responses": n_m * T.K["math"] + n_s * (e.get("limit_k_short") or T.K["short"])}
    if j["kind"] == "train":
        quota = {"math": e.get("limit_quota_math") or T.QUOTA["math"],
                 "short": e.get("limit_quota_short") or T.QUOTA["short"]}
        sizes = T.update_sizes(quota["math"] + quota["short"])
        if e.get("limit_updates") is not None:
            sizes = sizes[: e["limit_updates"]]
        full = e.get("limit_updates") is None
        return {"sizes": sizes, "scored_total": sum(sizes), "by_category": quota if full else None}
    if j["kind"] == "bank":
        return {"n_sequences": e.get("limit_prompts") or N_HELDOUT}
    return {"d20_prompts": 2 * T.D20_N}


def check_job(j, status_dir=None, logs_dir=None) -> list:
    """Problems of one job: exit status (if status_dir), run JSON, output files and counts, log patterns (if
    logs_dir). Never raises."""
    n, problems = j["name"], []
    try:
        if status_dir is not None:
            st = _status(status_dir, n)
            if st["exit"] != 0 or not st["done"]:
                problems.append(f"{n}: exit {st['exit']}, done {st['done']}")
        run = _run(j)
        if run.get("status") != "ok":
            problems.append(f"{n}: run status {run.get('status')!r} in {j['run_json']}")
            return problems
        cfg = run.get("config") or {}
        limited = bool(j["expect"])
        e = expected(j)
        guard = run.get("heldout_guard")
        if j["kind"] in ("collect", "train", "merge"):
            if not isinstance(guard, dict) or guard.get("heldout_file_opens") != []:
                problems.append(f"{n}: held-out guard record missing or held-out files opened: "
                                f"{(guard or {}).get('heldout_file_opens')}")
        if j["kind"] in ("collect", "bank"):
            from common.io import sha256_gz_content

            if not Path(j["out"]).is_file() or sha256_gz_content(j["out"]) != run.get("output_content_sha256"):
                problems.append(f"{n}: output {j['out']} missing or content sha256 differs from the run JSON")
        if j["kind"] == "collect":
            if bool(cfg.get("limited")) != limited:
                problems.append(f"{n}: config.limited {cfg.get('limited')} != {limited}")
            if run.get("n_responses") != e["n_responses"]:
                problems.append(f"{n}: {run.get('n_responses')} responses, expected {e['n_responses']}")
        elif j["kind"] == "bank":
            if (cfg.get("limit_prompts") is not None) != limited:
                problems.append(f"{n}: config.limit_prompts {cfg.get('limit_prompts')} (limited {limited})")
            if run.get("n_sequences") != e["n_sequences"] or cfg.get("bank") != j["bank"]:
                problems.append(f"{n}: {run.get('n_sequences')} sequences of bank {cfg.get('bank')!r}, expected "
                                f"{e['n_sequences']} of {j['bank']!r}")
        elif j["kind"] == "train":
            from common.io import sha256_gz_content

            from task3 import distill as T

            s = run.get("summary") or {}
            ups = run.get("updates") or []
            if bool(cfg.get("limited")) != limited:
                problems.append(f"{n}: config.limited {cfg.get('limited')} != {limited}")
            if [u["positions"] for u in ups] != e["sizes"] or s.get("n_updates") != len(e["sizes"]):
                problems.append(f"{n}: {len(ups)} updates with positions != expected {len(e['sizes'])} updates")
            if s.get("scored_total") != e["scored_total"]:
                problems.append(f"{n}: scored {s.get('scored_total')} != {e['scored_total']}")
            if e["by_category"] is not None and s.get("scored_by_category") != e["by_category"]:
                problems.append(f"{n}: scored by category {s.get('scored_by_category')} != {e['by_category']}")
            if e["by_category"] is not None and s.get("unconsumed_presentations"):
                problems.append(f"{n}: unconsumed presentations {s['unconsumed_presentations'][:5]}")
            pf = Path(j["out"]) / "presentations.jsonl.gz"
            if not pf.is_file() or sha256_gz_content(pf) != (run.get("presentations_file") or {}).get(
                    "content_sha256"):
                problems.append(f"{n}: presentations file missing or sha256 differs")
            ad = Path(j["adapter_dir"])
            if not ad.is_dir() or T.dir_sha256(ad) != run.get("adapter_sha256"):
                problems.append(f"{n}: adapter {ad} missing or sha256 differs from the run JSON")
            bad = [u["update"] for u in ups if not all(math.isfinite(x) for x in (u["loss"], u["lr"]))]
            if bad:
                problems.append(f"{n}: non-finite loss or LR at updates {bad[:5]}")
        elif j["kind"] == "merge":
            from task3 import distill as T

            d20 = run.get("d20") or {}
            if len(d20.get("per_prompt") or []) != e["d20_prompts"] or \
                    not math.isfinite(d20.get("max_abs_logit_diff", float("nan"))):
                problems.append(f"{n}: D20 record incomplete or non-finite")
            md = Path(j["merged_dir"])
            if not md.is_dir() or T.dir_sha256(md) != run.get("merged_files_sha256"):
                problems.append(f"{n}: merged draft {md} missing or sha256 differs from the run JSON")
        if logs_dir is not None:
            log = Path(logs_dir) / f"{n}.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            pats = [f"{n}: exit code 0"] + [done_pattern(j)]
            for pat in pats:
                if pat not in text:
                    problems.append(f"{n}: log line {pat!r} not found in {log}")
    except Exception as exc:
        problems.append(f"{n}: outputs unreadable: {type(exc).__name__}: {exc}")
    return problems


def done_pattern(j) -> str:
    if j["kind"] == "bank":
        return f"{j['bank']}: {expected(j)['n_sequences']} sequences, "
    return {"collect": "collect done: ", "train": "train done: ", "merge": "merge done: "}[j["kind"]]


def check(meta, status_dir, logs_dir) -> list:
    out = []
    for j in meta:
        out += check_job(j, status_dir, logs_dir)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Smoke report and plan.

def smoke_lines(meta) -> list:
    L = []
    for j in meta:
        run = _run(j)
        if run.get("status") != "ok":
            L.append(f"{j['name']}: status {run.get('status')!r}")
            continue
        tm = run.get("timing") or {}
        peak = _gib(_peak(run)) or 0.0
        if j["kind"] == "train":
            ups = run.get("updates") or []
            L.append(f"{j['name']}: load {tm.get('load_s', 0):.1f} s, {len(ups)} updates, mean update "
                     f"{tm.get('mean_update_wall_s') or 0:.2f} s (teacher {sum(u['teacher_s'] for u in ups):.2f}, "
                     f"rollout {sum(u['rollout_s'] for u in ups):.2f}, fwd/bwd {sum(u['fwd_bwd_s'] for u in ups):.2f},"
                     f" step {sum(u['step_s'] for u in ups):.2f} s), presentations "
                     f"{run['summary']['n_presentations']}, peak {peak:.2f} GiB")
        elif j["kind"] == "collect":
            L.append(f"{j['name']}: load {tm.get('load_s', 0):.1f} s, {run.get('n_responses')} responses, tokens "
                     f"{run.get('tokens')}, prefill {tm.get('mean_prefill_s') or 0:.3f} s, step "
                     f"{tm.get('median_step_s') or 0:.4f} s, peak {peak:.2f} GiB")
        elif j["kind"] == "bank":
            ps = tm.get("prompt_s") or []
            L.append(f"{j['name']}: load {tm.get('load_s', 0):.1f} s, per prompt "
                     f"{[round(x, 2) for x in ps]} s, terminations {run.get('terminations')}, peak {peak:.2f} GiB")
        else:
            L.append(f"{j['name']}: load {tm.get('load_s', 0):.1f} s, merge {tm.get('merge_s', 0):.2f} s, save "
                     f"{tm.get('save_s', 0):.1f} s, D20 max |diff| {(run.get('d20') or {}).get('max_abs_logit_diff')}")
    return L


def _smoke_by(meta, kind, key=None):
    for j in meta:
        if j["kind"] == kind and (key is None or key in (j.get("bank"), j.get("run"), j.get("unit"))):
            return _run(j)
    raise SystemExit(f"no smoke job of kind {kind} {key or ''}")


def project_seconds(smoke_meta) -> dict:
    """Projected seconds per main job from the smoke JSONs (before the safety factor).

    collect: load + 500 x (prefill + 192 x step) + 4800 x (prefill + 32 x step) (every response at its cap);
    train: load + 157 x mean update wall time of the smoke run of that method + adapter save;
    merge: smoke merge elapsed time; bank: load + 120 x the slowest smoke prompt of that kind (own banks from the
    smoke own bank); every job + JOB_OVERHEAD_S for process start."""
    from task3 import distill as T

    for j in smoke_meta:
        if _run(j).get("status") != "ok":
            raise SystemExit(f"smoke job {j['name']} not finished ok")
    c = _smoke_by(smoke_meta, "collect")["timing"]
    pre, step = c["mean_prefill_s"], c["median_step_s"]
    secs = {"collect": c["load_s"] + N_TRAIN["math"] * T.K["math"] * (pre + T.CAPS["math"] * step)
            + N_TRAIN["short"] * T.K["short"] * (pre + T.CAPS["short"] * step)}
    n_full = len(T.update_sizes(sum(T.QUOTA.values())))
    merge_s = _smoke_by(smoke_meta, "merge")["elapsed_wall_s"]
    own = _smoke_by(smoke_meta, "bank", "own_on_policy_6304")["timing"]
    for r in RUNS:
        tr = _smoke_by(smoke_meta, "train", r.rsplit("_", 1)[0] + "_6304")["timing"]
        secs[f"train_{r}"] = tr["load_s"] + n_full * tr["mean_update_wall_s"] + tr.get("adapter_save_s", 0.0)
        secs[f"merge_{r}"] = merge_s
        secs[f"bank_own_{r}"] = own["load_s"] + N_HELDOUT * max(own["prompt_s"])
    for k in BANK_KINDS:
        b = _smoke_by(smoke_meta, "bank", k)["timing"]
        secs[f"bank_{k}"] = b["load_s"] + N_HELDOUT * max(b["prompt_s"])
    return {k: v + JOB_OVERHEAD_S for k, v in secs.items()}


def unit_seconds(job_s: dict, safety=SAFETY) -> dict:
    out = {}
    for u in UNITS:
        if u in RUNS:
            out[u] = safety * sum(job_s[f"{p}_{u}"] for p in ("train", "merge", "bank_own"))
        else:
            out[u] = safety * job_s[u]
    return out


def simulate(assignment: dict, useconds: dict) -> dict:
    """Finish time per queue (s): units run in queue order; off-policy units start after the collection; GPU 1
    starts STAGGER_S later."""
    orders = queue_orders(assignment)
    collect_end = None
    t = {"gpu0": 0.0, "gpu1": STAGGER_S}
    ends = {}
    for u in orders["gpu0"]:
        if u == "collect":
            t["gpu0"] += useconds[u]
            collect_end = t["gpu0"]
            ends[u] = collect_end
            break
    pending = {q: [u for u in orders[q] if u != "collect"] for q in orders}
    for q in ("gpu0", "gpu1"):
        for u in pending[q]:
            if u.startswith("off_"):
                t[q] = max(t[q], collect_end)
            t[q] += useconds[u]
            ends[u] = t[q]
    return {"queue_s": t, "unit_end_s": ends}


def balance(useconds: dict) -> dict:
    best = None
    rest = [u for u in UNITS if u != "collect"]
    for combo in itertools.product(("gpu0", "gpu1"), repeat=len(rest)):
        a = {"collect": "gpu0", **dict(zip(rest, combo))}
        sim = simulate(a, useconds)
        span = max(sim["queue_s"].values())
        if best is None or span < best[0] - 1e-9:
            best = (span, a, sim)
    return {"assignment": best[1], "makespan_s": best[0], **best[2]}


def project(jobs_dir, budget_h=10.5, session_h=11.5, elapsed_s=0.0, vram_gib=14.0, safety=SAFETY) -> dict:
    meta = load_meta(jobs_dir, "smoke")
    job_s = project_seconds(meta)
    us = unit_seconds(job_s, safety)
    bal = balance(us)
    peak = max(_peak(_run(j)) for j in meta)
    reasons = []
    for q, s in bal["queue_s"].items():
        if s > budget_h * 3600:
            reasons.append(f"queue {q} projected {s / 3600:.2f} h above {budget_h} h")
    if elapsed_s + bal["makespan_s"] > session_h * 3600:
        reasons.append(f"elapsed {elapsed_s / 3600:.2f} h + projected {bal['makespan_s'] / 3600:.2f} h above "
                       f"{session_h} h")
    if peak > vram_gib * GiB:
        reasons.append(f"smoke peak {peak / GiB:.2f} GiB above {vram_gib} GiB")
    return {"job_seconds": job_s, "unit_seconds": us, "safety": safety, "job_overhead_s": JOB_OVERHEAD_S,
            "assignment": bal["assignment"], "orders": queue_orders(bal["assignment"]),
            "queue_s": bal["queue_s"], "queue_h": {q: s / 3600 for q, s in bal["queue_s"].items()},
            "unit_end_s": bal["unit_end_s"], "makespan_s": bal["makespan_s"], "elapsed_s": elapsed_s,
            "budget_h": budget_h, "session_h": session_h, "peak_bytes": peak, "vram_limit_gib": vram_gib,
            "over": bool(reasons), "reasons": reasons}


def print_plan(p):
    for q, units in p["orders"].items():
        print(f"{q}: projected {p['queue_h'][q]:.2f} h: " +
              ", ".join(f"{u} {p['unit_seconds'][u] / 60:.0f} min" for u in units))
    print(f"makespan {p['makespan_s'] / 3600:.2f} h; elapsed so far {p['elapsed_s'] / 3600:.2f} h; budget "
          f"{p['budget_h']} h per queue, session {p['session_h']} h; safety {p['safety']}; smoke peak "
          f"{p['peak_bytes'] / GiB:.2f} GiB")
    for x in p["reasons"]:
        print("OVER:", x)
    print("plan: within limits" if not p["over"] else "plan: OVER LIMIT, queues not started")


# ---------------------------------------------------------------------------------------------------------------
# Summary, manifest, zip, final.

def summary_lines(meta, max_lines=40) -> list:
    L = []
    for j in meta:
        try:
            run = _run(j)
            if run.get("status") != "ok":
                L.append(f"{j['name']}: status {run.get('status')!r}")
                continue
            tm = run.get("timing") or {}
            if j["kind"] == "collect":
                L.append(f"{j['name']}: {run['n_responses']} responses, tokens {run['tokens']}, terminations "
                         f"{run['terminations']}, teacher generation {tm['teacher_generation_s_total'] / 60:.1f} min")
            elif j["kind"] == "train":
                s, ups = run["summary"], run["updates"]
                last = ups[-10:]
                L.append(f"{j['name']}: {s['n_updates']} updates, skipped {s['skipped_updates']}, scored "
                         f"{s['scored_by_category']}, {s['n_presentations']} presentations, truncated "
                         f"{len(s['truncated_presentations'])}, wrap-arounds {s['wrap_arounds']}, mean loss last 10 "
                         f"{sum(u['loss'] for u in last) / len(last):.4f}")
                L.append(f"    min: teacher {tm['teacher_scoring_s'] / 60:.1f}, rollout "
                         f"{tm['student_rollout_s'] / 60:.1f}, fwd/bwd {tm['forward_backward_s'] / 60:.1f}, step "
                         f"{tm['optimizer_step_s'] / 60:.2f}, loop {tm['train_loop_s'] / 60:.1f}; peak "
                         f"{(_gib(run.get('peak_gpu_memory_bytes')) or 0):.2f} GiB; final scale "
                         f"{s['final_scale']:.0f}")
            elif j["kind"] == "merge":
                L.append(f"{j['name']}: merge {tm['merge_s']:.2f} s, D20 max |logit diff| "
                         f"{run['d20']['max_abs_logit_diff']:.3g}")
            else:
                L.append(f"{j['name']}: {run['n_sequences']} sequences, {run['n_positions']} positions, terminations "
                         f"{run['terminations']}, mean length {run['mean_length']:.1f}")
        except Exception as exc:
            L.append(f"{j['name']}: summary unavailable: {type(exc).__name__}: {exc}")
    return L[:max_lines]


def timing_lines(meta, status_dir, plan=None) -> list:
    """Wall time per job from the queue status records, next to the plan's projection (with the safety factor)."""
    L, per_q = [], {}
    job_s = (plan or {}).get("job_seconds") or {}
    safety = (plan or {}).get("safety") or 1.0
    for j in meta:
        st = _status(status_dir, j["name"])
        w = st["wall_s"]
        proj = job_s.get(j["name"])
        per_q[j["queue"]] = per_q.get(j["queue"], 0.0) + (w or 0.0)
        L.append(f"{j['queue']} {j['name']:<36} exit {str(st['exit']):>4}  wall "
                 f"{'-' if w is None else f'{w / 60:.1f}'} min  projected "
                 f"{'-' if proj is None else f'{safety * proj / 60:.1f}'} min")
    for q, s in per_q.items():
        L.append(f"{q}: {s / 3600:.2f} h of job wall time; projected "
                 f"{((plan or {}).get('queue_h') or {}).get(q, float('nan')):.2f} h")
    return L


def manifest(artifacts, meta) -> dict:
    """sha256 of every file under the artifacts root, and agreement with the run JSONs."""
    root = Path(artifacts)
    files = {p.relative_to(root).as_posix(): {"sha256": _sha256(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file()} if root.is_dir() else {}
    agree = {}
    for j in meta:
        run = _run(j)
        if j["kind"] == "train":
            d, want = Path(j["adapter_dir"]), run.get("adapter_sha256")
        elif j["kind"] == "merge":
            d, want = Path(j["merged_dir"]), run.get("merged_files_sha256")
        else:
            continue
        try:
            rel = d.relative_to(root).as_posix()
        except ValueError:
            rel = str(d)
        got = {k[len(rel) + 1:]: v["sha256"] for k, v in files.items()
               if k.startswith(rel + "/") and "/" not in k[len(rel) + 1:]}
        agree[j["name"]] = bool(want) and got == want
    return {"root": str(root), "n_files": len(files), "files": files, "agrees_with_run_json": agree}


def make_zip(out, roots) -> dict:
    out = Path(out)
    names, missing = [], []
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as zf:
        for root in map(Path, roots):
            if not root.exists():
                missing.append(str(root))
                continue
            for f in ([root] if root.is_file() else sorted(root.rglob("*"))):
                rel = f.relative_to(root).as_posix() if root.is_dir() else f.name
                if f.is_file() and not any(w in rel for w in ZIP_EXCLUDE):
                    zf.write(f, f.as_posix())
                    names.append(f.as_posix())
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": _sha256(out), "n_files": len(names),
            "missing": missing}


def final(meta, status_dir, check_report, zip_path) -> list:
    reasons = []
    for j in meta:
        st = _status(status_dir, j["name"])
        if st["exit"] != 0 or not st["done"]:
            reasons.append(f"{j['name']}: exit {st['exit']}, done {st['done']}")
    rep = _load_json(check_report)
    if rep is None:
        reasons.append(f"no check report at {check_report}")
    elif rep.get("problems"):
        reasons.append(f"check found {len(rep['problems'])} problems")
    if not Path(zip_path).is_file():
        reasons.append(f"no zip at {zip_path}")
    return reasons


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("data-check", "model-files"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=STAGES, required=True)
    j.add_argument("--jobs-dir", required=True)
    j.add_argument("--plan", default=None)
    j.add_argument("--results", default=None)
    j.add_argument("--status", default=None)
    j.add_argument("--artifacts", default=None)
    j.add_argument("--preflight", default=PREFLIGHT)
    for k in TEST_KEYS:
        j.add_argument("--" + k.replace("_", "-"), default=None)
    sc = sub.add_parser("smoke-check")
    sc.add_argument("--jobs-dir", required=True)
    sc.add_argument("--status", default=DEFAULTS["smoke"]["status"])
    sc.add_argument("--logs", default=DEFAULTS["smoke"]["logs"])
    p = sub.add_parser("plan")
    p.add_argument("--jobs-dir", required=True)
    p.add_argument("--budget-h", type=float, default=10.5)
    p.add_argument("--session-h", type=float, default=11.5)
    p.add_argument("--elapsed-s", type=float, default=0.0)
    p.add_argument("--vram-gib", type=float, default=14.0)
    p.add_argument("--out", default=None)
    for name in ("summary", "timing", "check", "manifest", "zip", "final"):
        s = sub.add_parser(name)
        s.add_argument("--stage", choices=STAGES, default="main")
        s.add_argument("--jobs-dir", required=True)
        s.add_argument("--status", default=None)
        s.add_argument("--logs", default=None)
        if name in ("zip", "check", "manifest"):
            s.add_argument("--out", required=(name != "check"), default=None)
        if name == "zip":
            s.add_argument("--include", nargs="*", default=[])
        if name == "timing":
            s.add_argument("--plan", default=None)
        if name == "manifest":
            s.add_argument("--artifacts", default=None)
        if name == "final":
            s.add_argument("--check-report", required=True)
            s.add_argument("--zip", required=True)
    a = ap.parse_args(argv)

    if a.cmd in ("data-check", "model-files"):
        rec = data_check() if a.cmd == "data-check" else model_files()
        _write_new(a.out, rec)
        print(json.dumps(rec if a.cmd == "data-check" else {k: {"repo": v["repo"], "revision": v["revision"],
                                                               "n_files": len(v["files"])} for k, v in rec.items()},
                         indent=1, default=str)[:4000])
        return 0
    if a.cmd == "jobs":
        from .banks import require_preflight

        require_preflight(a.preflight)  # every stage needs the FP32 and FP16 preflight rules to pass (AM2)
        plan = None
        if a.stage == "main":
            if not a.plan:
                raise SystemExit("main jobs need --plan")
            plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
            if plan.get("over"):
                raise SystemExit(f"plan {a.plan} is over the limits; no main jobs written")
        test = {k: getattr(a, k) for k in TEST_KEYS if getattr(a, k) is not None}
        jobs = stage_jobs(a.stage, a.results, a.status, a.artifacts, a.preflight, plan, test)
        paths = write_jobs(a.stage, a.jobs_dir, jobs)
        for q in queues(jobs):
            print(f"{q}: " + ", ".join(x["name"] for x in jobs if x["queue"] == q))
        print({k: str(v) for k, v in paths.items()})
        return 0
    if a.cmd == "smoke-check":
        meta = load_meta(a.jobs_dir, "smoke")
        for line in smoke_lines(meta):
            print(line)
        problems = check(meta, a.status, a.logs)
        for x in problems:
            print("PROBLEM:", x)
        print(f"smoke-check: {len(meta)} jobs, {len(problems)} problems")
        return 1 if problems else 0
    if a.cmd == "plan":
        pl = project(a.jobs_dir, a.budget_h, a.session_h, a.elapsed_s, a.vram_gib)
        print_plan(pl)
        if a.out:
            _write_new(a.out, pl)
        return 1 if pl["over"] else 0
    meta = load_meta(a.jobs_dir, a.stage)
    d = DEFAULTS[a.stage]
    status, logs = a.status or d["status"], a.logs or d["logs"]
    if a.cmd == "summary":
        for line in summary_lines(meta):
            print(line)
        return 0
    if a.cmd == "timing":
        for line in timing_lines(meta, status, _load_json(a.plan) if a.plan else None):
            print(line)
        return 0
    if a.cmd == "check":
        problems = check(meta, status, logs)
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
    if a.cmd == "manifest":
        rec = manifest(a.artifacts or d["artifacts"], meta)
        _write_new(a.out, rec)
        print(f"manifest: {rec['n_files']} files under {rec['root']}; agrees with run JSONs: "
              f"{rec['agrees_with_run_json']}")
        return 0 if all(rec["agrees_with_run_json"].values()) else 1
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
