"""Held-out evaluation of one compressed condition rebuilt from restore() and its run-1 artifact.

Usage: python -m task1.evaluate_condition --condition ID --artifacts DIR --run1 DIR --out DIR [--device cuda:0]
       [--model-path PATH --limit-windows N --limit-hellaswag N --limit-dev N --repo-root DIR]
       (test options; refused for outputs under results/task1)
--artifacts is the condition's artifact directory (artifact.safetensors, manifest.json); --run1 is its task1.compress
output directory, holding the committed copy of the artifact manifest (artifact_manifest.json). Every condition
except fp16 (task1.baseline) is evaluated here.

Checks before any output is written: every tensor sha256 and the file sha256 against the manifest; the manifest
sha256 against the committed run-1 copy and the run-1 record; for P3/Q3 the freeze guard on task1/frozen/<ID>.json
(tracked, unmodified, touched by exactly one commit, sha256 equal to the allocation sha256 in the manifest).
Then: development NLL (reproduction of run 1), both perplexity suites, HellaSwag at the condition's scope,
dense-execution resources and storage.
--no-resources skips the resources step; --resources-only runs only the executed-model rebuild, resources and storage
(no development NLL, perplexity or HellaSwag) inside task1.data.forbid_held_out, so no held-out test file can be
opened. The default runs every step.
"""
import contextlib
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from common.io import sha256_file
from common.paths import REPO_ROOT
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs

RUN1_MANIFEST_COPY = "artifact_manifest.json"
FROZEN_DIR = Path("task1") / "frozen"
FROZEN_FAMILIES = ("P3", "Q3")
PRUNE_FAMILIES = ("P1", "P2", "P2_2of4", "P3")
QUANT_FAMILIES = ("Q1", "Q2", "Q3")
# Largest accepted |development mean NLL - run-1 value| for the rebuilt model (same code, data and batch size).
REPRO_TOL = 1e-4
BASE_OUTPUT_FILES = {
    "run": "run.json",
    "ppl_wikitext2_test": "ppl_wikitext2_test.jsonl.gz",
    "ppl_c4_validation": "ppl_c4_validation.jsonl.gz",
    "calib_nll_c4_development": "calib_nll_c4_development.jsonl.gz",
}


def output_files(scope: str, mode: str = "default") -> dict:
    if mode == "only":
        return {"run": BASE_OUTPUT_FILES["run"]}
    return {**BASE_OUTPUT_FILES, "hellaswag": f"hellaswag_{scope}.jsonl.gz"}


def resources_mode(args) -> str:
    """'default', 'skip' (--no-resources) or 'only' (--resources-only)."""
    return "only" if args.resources_only else "skip" if args.no_resources else "default"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--run1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", default=None, help="local model directory (tests only)")
    ap.add_argument("--limit-windows", type=int, default=None, help="first N windows per suite (tests only)")
    ap.add_argument("--limit-hellaswag", type=int, default=None, help="first N HellaSwag IDs (tests only)")
    ap.add_argument("--limit-dev", type=int, default=None, help="first N development sequences (tests only)")
    ap.add_argument("--repo-root", default=None, help="repository holding task1/frozen (tests only)")
    res = ap.add_mutually_exclusive_group()
    res.add_argument("--no-resources", action="store_true", help="skip the resources step")
    res.add_argument("--resources-only", action="store_true",
                     help="only the executed-model rebuild, resources and storage; no held-out file is read")
    return ap.parse_args(argv)


def check_args(args) -> dict:
    if args.condition not in C.CONDITIONS:
        raise SystemExit(f"unknown condition {args.condition}; registry: {sorted(C.CONDITIONS)}")
    if C.CONDITIONS[args.condition]["family"] == "baseline":
        raise SystemExit("fp16 is evaluated by task1.baseline, not task1.evaluate_condition")
    limits = {"windows": args.limit_windows, "hellaswag": args.limit_hellaswag, "dev": args.limit_dev}
    for v in limits.values():
        if v is not None and v < 1:
            raise SystemExit("limits must be positive")
    limited = any(v is not None for v in limits.values())
    test_only = limited or args.model_path is not None or args.repo_root is not None
    if test_only and runargs.under(args.out, runargs.REAL_ROOT):
        raise SystemExit(f"refusing test options with an output path under {runargs.REAL_ROOT}: {args.out}")
    if Path(args.out).exists():
        raise SystemExit(f"output directory exists: {args.out}")
    return {"limits": limits, "limited": limited, "test_only": test_only}


def model_source(args) -> dict:
    return {"model_path": str(args.model_path)} if args.model_path else {"model_id": C.MODEL_ID,
                                                                        "revision": C.MODEL_REVISION}


def check_run1(run1_dir, art_dir, cond: str, args) -> dict:
    """Run-1 record of this condition, and the artifact manifest against its committed copy and the record."""
    from common.freeze import FreezeError, require_committed

    from .artifacts import MANIFEST_FILE

    d = Path(run1_dir)
    run_path = d / "run.json"
    if not run_path.is_file():
        raise SystemExit(f"run-1 record not found: {run_path}")
    r1 = json.loads(run_path.read_text(encoding="utf-8"))
    cfg = r1.get("config", {})
    if r1.get("status") != "ok":
        raise SystemExit(f"run-1 record is not finished ok: {r1.get('status')}")
    if cfg.get("entry_point") != "task1.compress" or cfg.get("condition") != cond:
        raise SystemExit(f"run-1 record is for {cfg.get('entry_point')} {cfg.get('condition')}, not task1.compress {cond}")
    if cfg.get("model_path") != args.model_path:
        raise SystemExit("run-1 record used a different model source")
    if cfg.get("limits", {}).get("dev") != args.limit_dev:
        raise SystemExit("run-1 record used a different development limit")
    man_path = Path(art_dir) / MANIFEST_FILE
    if not man_path.is_file():
        raise SystemExit(f"artifact manifest not found: {man_path}")
    try:
        copy = require_committed(d / RUN1_MANIFEST_COPY, "run-1 artifact manifest copy")
    except FreezeError as exc:
        raise SystemExit(f"refusing: {exc}")
    sha = sha256_file(man_path)
    recorded = r1["steps"]["artifact"]["manifest_sha256"]
    if sha != copy["sha256"]:
        raise SystemExit(f"refusing: artifact manifest sha256 {sha} != committed run-1 copy {copy['sha256']}")
    if sha != recorded:
        raise SystemExit(f"refusing: artifact manifest sha256 {sha} != run-1 record {recorded}")
    return {"dir": str(d), "run_sha256": sha256_file(run_path), "run_git_commit": r1["meta"]["git_commit"],
            "run_git_dirty": r1["meta"]["git_dirty"], "manifest_sha256": sha, "manifest_copy": copy,
            "development_mean_nll": r1["steps"]["calib_nll"]["c4_development"]["mean_nll"],
            "calibration_mean_nll": r1["steps"]["calib_nll"]["c4_calibration"]["mean_nll"]}


def verify_artifact(art_dir, cond: str, source: dict) -> tuple:
    """File sha256 and every tensor sha256/shape against the manifest; condition, evaluation ID and model source."""
    from .artifacts import ARTIFACT_FILE, MANIFEST_FILE
    from .tensorfile import ArtifactIntegrityError, load_tensors

    d = Path(art_dir)
    try:
        tensors, man = load_tensors(d / ARTIFACT_FILE, d / MANIFEST_FILE)
    except ArtifactIntegrityError as exc:
        raise SystemExit(f"refusing: artifact integrity check failed: {exc}")
    n = len(tensors)
    del tensors
    if man.get("condition") != cond or man.get("eval_id") != C.EVAL_ID:
        raise SystemExit(f"refusing: artifact is for {man.get('condition')} / {man.get('eval_id')}, not {cond} / {C.EVAL_ID}")
    if man.get("model") != source:
        raise SystemExit(f"refusing: artifact model source {man.get('model')} != {source}")
    rec = {"dir": str(d), "file": man["file"], "n_tensors_verified": n, "stored_tensor_bytes": man["stored_tensor_bytes"],
           "manifest_bytes": (d / MANIFEST_FILE).stat().st_size, "manifest_sha256": sha256_file(d / MANIFEST_FILE),
           "file_sha256_verified": True, "tensor_sha256_verified": True}
    return man, rec


def check_freeze(cond: str, man: dict, repo_root) -> dict:
    """Freeze guard on task1/frozen/<ID>.json for P3/Q3; returns the record (or a not-applicable record)."""
    from common.freeze import FreezeError, check_frozen

    fam = C.CONDITIONS[cond]["family"]
    if fam not in FROZEN_FAMILIES:
        return {"applies": False}
    path = Path(repo_root) / FROZEN_DIR / f"{cond}.json"
    alloc = man.get("allocation") or {}
    want = alloc.get("sha256")
    if not want:
        raise SystemExit("refusing: artifact manifest records no allocation sha256")
    try:
        rec = check_frozen(path, want, f"frozen allocation of {cond}")
    except FreezeError as exc:
        raise SystemExit(f"refusing: {exc}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("condition") != cond:
        raise SystemExit(f"refusing: frozen allocation is for condition {frozen.get('condition')}")
    return {"applies": True, "file": str(path), **rec, "allocation_sha256_in_manifest": want,
            "candidate_id": frozen.get("candidate_id"), "totals": frozen.get("totals")}


def storage_report(model, man: dict, fam: str, frozen: dict) -> dict:
    """Effective nonzeros, nominal bits, logical bytes, sparse16 bytes (pruning), artifact bytes, whole-model bytes."""
    import torch

    from . import storage
    from .model import count_params, enumerate_eligible

    elig = enumerate_eligible(model)
    counts = count_params(model)
    shapes = [tuple(e.module.weight.shape) for e in elig]
    nnz = [int(torch.count_nonzero(e.module.weight)) for e in elig]
    mats = man["matrices"]
    assert [m["qualname"] for m in mats] == [e.qualname for e in elig]
    assert nnz == [m["nonzeros"] for m in mats]
    n_w = [a * b for a, b in shapes]
    n_elig = sum(n_w)
    assert n_elig == counts["eligible"]
    st = storage.report(shapes, counts["non_eligible"], nnz if fam in PRUNE_FAMILIES else None)
    st["nonzeros"] = {"global": sum(nnz), "eligible_weights": n_elig, "nonzero_fraction": sum(nnz) / n_elig,
                      "zero_fraction": 1 - sum(nnz) / n_elig,
                      "per_matrix": {m["qualname"]: n for m, n in zip(mats, nnz)}}
    elig_bytes = man["logical_bytes"]["eligible"]
    if fam in QUANT_FAMILIES:
        bits = [m["bits"] for m in mats]
        pq = storage.packed_quant_bytes(shapes, bits, detail=True)
        assert pq["total"] == elig_bytes, (pq["total"], elig_bytes)
        st["nominal_bits_per_eligible_weight"] = sum(b * n for b, n in zip(bits, n_w)) / n_elig
        st["packed_quant"] = {**pq, "bits_per_matrix": {m["qualname"]: m["bits"] for m in mats},
                              "extra_fp16_scale_bytes": 0}
        if fam == "Q3":
            budget = frozen["totals"]["budget_bytes"]
            assert pq["total"] <= budget, (pq["total"], budget)
            st["packed_quant"]["budget_bytes"] = budget
            st["packed_quant"]["budget_whole_model_bytes"] = storage.whole_model_bytes(budget, counts["non_eligible"])
    else:
        kept = [m["mask_kept"] for m in mats]
        mask_zeros = sum(n - k for n, k in zip(n_w, kept))
        assert storage.sparse16_bytes(shapes, nnz) == elig_bytes, elig_bytes
        st["nominal_bits_per_eligible_weight"] = 16
        st["mask"] = {"kept": sum(kept), "zeros": mask_zeros, "zero_fraction": mask_zeros / n_elig}
        if fam == "P3":
            assert mask_zeros == frozen["totals"]["p2_zeros"], (mask_zeros, frozen["totals"])
            st["mask"]["p2_zeros"] = frozen["totals"]["p2_zeros"]
    st["representation"] = "packed_quant" if fam in QUANT_FAMILIES else "sparse16"  # as in task1.compress
    st["logical"] = {"format": man["logical_bytes"]["format"], "eligible_bytes": elig_bytes,
                     "bits_per_eligible_weight": storage.bits_per_eligible_weight(elig_bytes, shapes),
                     "non_eligible_params": counts["non_eligible"],
                     "whole_model_bytes": storage.whole_model_bytes(elig_bytes, counts["non_eligible"]),
                     "tied_lm_head_counted_once": counts["lm_head_tied"]}
    assert st["logical"]["whole_model_bytes"] == man["logical_bytes"]["whole_model"]
    st["param_counts"] = {k: counts[k] for k in ("eligible", "embedding", "norms", "biases", "other", "total",
                                                 "non_eligible", "lm_head_tied")}
    return st


def run(args):
    lim = check_args(args)
    cond = args.condition
    spec = C.CONDITIONS[cond]
    fam = spec["family"]
    scope = spec["hellaswag"]
    source = model_source(args)
    repo_root = Path(args.repo_root) if args.repo_root else REPO_ROOT

    # Refusals before any output exists.
    run1 = check_run1(args.run1, args.artifacts, cond, args)
    man, art = verify_artifact(args.artifacts, cond, source)
    frozen = check_freeze(cond, man, repo_root)

    import torch

    from .artifacts import load_artifact
    from .data import (HELLASWAG_FILE, SEQUENCE_FILES, WINDOW_FILES, file_info, forbid_held_out, read_hellaswag,
                       read_sequences, read_windows)
    from .evaluate import calib_nll, hellaswag, perplexity, write_rows
    from .model import Pristine, load_model, load_tokenizer
    from .resources import latency, prompt_ids

    mode = resources_mode(args)
    files = output_files(scope, mode)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    cfg = {"entry_point": "task1.evaluate_condition", "condition": cond, "condition_spec": spec,
           "settings": C.settings_dict(), "device": args.device, "model_path": args.model_path, **lim,
           "hellaswag_scope": scope, "output_files": files, "artifacts_dir": str(args.artifacts),
           "run1_dir": str(args.run1), "repo_root": str(repo_root), "repro_tol": REPRO_TOL,
           "resources_mode": mode,
           "data_files": [file_info(f) for f in (*WINDOW_FILES.values(), SEQUENCE_FILES["c4_calibration"],
                                                 SEQUENCE_FILES["c4_development"], HELLASWAG_FILE)]}
    if mode == "only":
        cfg["data_files"] = [file_info(SEQUENCE_FILES["c4_calibration"])]
    rec = RunRecorder(out / files["run"], cfg)
    steps = {"inputs": {"artifact": art, "run1": run1, "freeze": frozen}}
    timings = {}
    cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    guard_cm = forbid_held_out() if mode == "only" else contextlib.nullcontext(None)
    try:
        with guard_cm as guard:
            t = time.perf_counter()
            model, info = load_model(args.device, args.model_path)
            tok = load_tokenizer(args.model_path)
            pristine = Pristine(model)
            timings["load_s"] = time.perf_counter() - t
            rec.update(model=info, eval_id=C.EVAL_ID, steps=steps, timings_s=timings,
                       tokenizer={"class": type(tok).__name__, "name_or_path": tok.name_or_path, "len": len(tok)})

            # Executed model: restore() then the artifact (sha256 values checked again by the loader).
            t = time.perf_counter()
            man2 = load_artifact(model, pristine, args.artifacts)
            assert man2["file"]["sha256"] == art["file"]["sha256"]
            timings["artifact_load_s"] = time.perf_counter() - t
            steps["executed_model"] = {"restored_tensors_verified": len(pristine.names), "artifact_applied": True,
                                       "extra_tensors": man2["extra_tensors"]}
            rec.update(steps=steps, timings_s=timings)

            if mode != "only":
                # Development NLL against the run-1 value of the same executed model.
                t = time.perf_counter()
                devel = read_sequences("c4_development", limit=args.limit_dev)
                if args.limit_dev is None:
                    assert len(devel) == C.N_DEVELOPMENT
                rows, summ = calib_nll(model, devel, split="c4_development")
                write_rows(out / files["calib_nll_c4_development"], rows)
                diff = summ["mean_nll"] - run1["development_mean_nll"]
                summ.update(file=files["calib_nll_c4_development"], run1_mean_nll=run1["development_mean_nll"],
                            diff_vs_run1=diff, reproduces_run1=abs(diff) <= REPRO_TOL)
                steps["reproduction"] = summ
                timings["calib_nll_c4_development_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)
                if abs(diff) > REPRO_TOL:
                    raise RuntimeError(f"development mean NLL {summ['mean_nll']} differs from run 1 by {diff}")

                # Perplexity suites.
                steps["perplexity"] = {}
                for suite in ("wikitext2_test", "c4_validation"):
                    t = time.perf_counter()
                    windows = read_windows(suite, limit=args.limit_windows)
                    exp_n = None if args.limit_windows is not None else C.PPL_SCORED_TOKENS
                    rows, summ = perplexity(model, windows, suite=suite, expected_n=exp_n)
                    write_rows(out / files[f"ppl_{suite}"], rows)
                    summ["file"] = files[f"ppl_{suite}"]
                    steps["perplexity"][suite] = summ
                    timings[f"ppl_{suite}_s"] = time.perf_counter() - t
                    rec.update(steps=steps, timings_s=timings)

                # HellaSwag at the condition's scope.
                t = time.perf_counter()
                hs = read_hellaswag(limit=args.limit_hellaswag)
                if args.limit_hellaswag is None:
                    hs = hs[:C.hellaswag_n(scope)]
                exp = None if args.limit_hellaswag is not None else C.hellaswag_n(scope)
                rows, summ = hellaswag(model, tok, hs, scope=scope, expected_n=exp)
                write_rows(out / files["hellaswag"], rows)
                summ["file"] = files["hellaswag"]
                steps["hellaswag"] = summ
                timings["hellaswag_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)

            if mode != "skip":
                # Resources: the baseline protocol on the dense FP16 execution of the stored weights.
                t = time.perf_counter()
                steps["resources"] = latency(model, prompt_ids(read_sequences("c4_calibration")))
                timings["resources_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)

            # Storage.
            steps["storage"] = storage_report(model, man, fam, frozen)
            steps["storage"]["artifact_files"] = {"safetensors_bytes": art["file"]["bytes"],
                                                  "manifest_bytes": art["manifest_bytes"],
                                                  "total_bytes": art["file"]["bytes"] + art["manifest_bytes"],
                                                  "stored_tensor_bytes": art["stored_tensor_bytes"]}
            if cuda:
                steps["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(next(model.parameters()).device)
        if guard is not None:
            steps["held_out_guard"] = guard
            assert not guard["attempts"], guard
        rec.finish("ok", steps=steps, timings_s=timings)
    except BaseException as exc:
        rec.finish("failed", steps=steps, timings_s=timings, error=repr(exc), traceback=traceback.format_exc())
        raise
    return out


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
