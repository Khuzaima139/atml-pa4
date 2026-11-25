"""Build one compression condition from the restored FP16 weights, measure it on calibration data, save its artifact.

Usage: python -m task1.compress --condition ID --out DIR --artifacts DIR [--calib-stats DIR] [--allocation FILE]
       [--device cuda:0] [--model-path PATH --limit-calib N --limit-dev N]   (tests only; refused for outputs
       under results/task1)
Conditions: fp16, p1_*, p2_*, p2_24_s50, q1_*, q2_* (q2 reads the AWQ scales of a calib_stats run), and
p3_s50, p3_s75, q3_int4, q3_int3 only with --allocation (a selected_allocation.json of task1.select; q3 also
reads the AWQ scales). Reads only the C4 calibration (statistics, masks, scales, calibration NLL) and development
(NLL) files; no held-out file can be opened while it runs (task1.data.forbid_held_out).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from common.io import sha256_file
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs

OUTPUT_FILES = {
    "run": "run.json",
    "projections": "projections.jsonl.gz",
    "calib_nll_c4_calibration": "calib_nll_c4_calibration.jsonl.gz",
    "calib_nll_c4_development": "calib_nll_c4_development.jsonl.gz",
}
NEEDS_ALLOCATION = ("P3", "Q3")
NEEDS_CALIB_STATS = ("Q2", "Q3")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--calib-stats", default=None)
    ap.add_argument("--allocation", default=None, help="selected_allocation.json (P3 and Q3 only)")
    runargs.add_test_args(ap)
    return ap.parse_args(argv)


def check_condition(args):
    if args.condition not in C.CONDITIONS:
        raise SystemExit(f"unknown condition {args.condition}; registry: {sorted(C.CONDITIONS)}")
    fam = C.CONDITIONS[args.condition]["family"]
    if fam in NEEDS_ALLOCATION and args.allocation is None:
        raise SystemExit(f"condition {args.condition} (family {fam}) requires an allocation file (--allocation)")
    if fam not in NEEDS_ALLOCATION and args.allocation is not None:
        raise SystemExit(f"--allocation is only used by P3 and Q3 conditions, not {args.condition}")
    if fam in NEEDS_CALIB_STATS and args.calib_stats is None:
        raise SystemExit(f"condition {args.condition} requires --calib-stats")
    if fam not in NEEDS_CALIB_STATS and args.calib_stats is not None:
        raise SystemExit(f"--calib-stats is only used by Q2 and Q3 conditions, not {args.condition}")
    return fam


def git_file_status(path) -> dict:
    """Whether a file is tracked in this repository and unmodified against HEAD (recorded, not enforced here)."""
    import subprocess

    from common.paths import REPO_ROOT

    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        return {"in_repo": False, "tracked": False, "clean": None}
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True)
    tracked = git("ls-files", "--error-unmatch", "--", str(rel)).returncode == 0
    clean = tracked and git("diff", "--quiet", "HEAD", "--", str(rel)).returncode == 0
    last = git("log", "-1", "--format=%H", "--", str(rel)).stdout.strip() or None
    return {"in_repo": True, "tracked": tracked, "clean": clean, "last_commit": last}


def load_allocation(path, cond: str, args, matrices, calib_prov=None) -> tuple:
    """Validate a selected allocation file for this condition and model; returns (allocation, record)."""
    from . import allocate as A

    path = Path(path)
    sha = sha256_file(path)
    al = json.loads(path.read_text(encoding="utf-8"))
    spec = C.CONDITIONS[cond]
    if al.get("condition") != cond:
        raise SystemExit(f"allocation file is for condition {al.get('condition')}, not {cond}")
    if al.get("model_path") != args.model_path:
        raise SystemExit("allocation file was selected on a different model source")
    if al.get("unit") != C.ALLOC_UNIT or al.get("eval_id") != C.EVAL_ID:
        raise SystemExit("allocation file has a different allocation unit or evaluation identifier")
    quals = [m["qualname"] for m in matrices]
    key = "per_matrix_zeros" if spec["family"] == "P3" else "per_matrix_bits"
    if sorted(al.get(key, {})) != sorted(quals):
        raise SystemExit(f"allocation file {key} does not name exactly the eligible matrices of this model")
    try:
        if spec["family"] == "P3":
            zeros = {q: int(al[key][q]) for q in quals}
            tot = A.check_p3_zeros(matrices, zeros, spec["sparsity"])
            assert tot["total_zeros"] == al["totals"]["zeros"] == al["totals"]["p2_zeros"], (tot, al["totals"])
            value = zeros
        else:
            ref_bits = C.Q3_TARGETS[al["target"]]["bits"]
            bits = {q: int(al[key][q]) for q in quals}
            tot = A.check_q3_bits(matrices, bits, ref_bits)
            assert tot["eligible_packed_bytes"] == al["totals"]["eligible_packed_bytes"], (tot, al["totals"])
            assert tot["budget_bytes"] == al["totals"]["budget_bytes"], (tot, al["totals"])
            if calib_prov is not None:
                assert al["inputs"]["calib_stats"]["awq_scales_sha256"] == calib_prov["awq_scales_sha256"], \
                    "allocation was selected with different AWQ scales"
            value = [bits[m["qualname"]] for m in matrices if m["name"] == C.ELIGIBLE[0]]
    except (AssertionError, KeyError) as exc:
        raise SystemExit(f"allocation file failed validation: {exc!r}")
    rec = {"file": str(path), "sha256": sha, "candidate_id": al["candidate_id"], "measure": al["measure"],
           "lambda": al["lambda"], "selection_code_commit": al["code_commit"], "totals": tot,
           "inputs": al["inputs"], "git": git_file_status(path)}
    return value, rec


def load_awq_scales(calib_dir, bits: int, args, device) -> tuple:
    """AWQ scales at one bit width from a finished calib_stats run, with sha256 and provenance checks."""
    from .awq import GROUPS
    from .calib_stats import OUTPUT_FILES as CF, scale_name
    from .tensorfile import load_tensors

    from .calib_stats import load_run

    calib_dir = Path(calib_dir)
    crun = load_run(calib_dir, args)
    tensors, man = load_tensors(calib_dir / CF["awq_scales"], calib_dir / CF["awq_scales_manifest"], device=device)
    recorded = crun["steps"]["files"]["awq_scales"]["sha256"]
    if man["file"]["sha256"] != recorded:
        raise SystemExit("AWQ scales sha256 differs from the calib_stats run record")
    search = json.loads((calib_dir / CF["awq_search"]).read_text(encoding="utf-8"))
    assert search["scales_file"]["sha256"] == recorded
    n_layers = 1 + max(e["layer"] for e in search["entries"])
    scales = {(li, g): tensors[scale_name(li, g, bits)] for li in range(n_layers) for g in GROUPS}
    alphas = {f"{e['layer']}.{e['group']}": e["alpha"] for e in search["entries"] if e["bits"] == bits}
    prov = {"dir": str(calib_dir), "run_sha256": sha256_file(calib_dir / CF["run"]),
            "awq_scales_sha256": recorded, "awq_search_sha256": sha256_file(calib_dir / CF["awq_search"]),
            "calib_git_commit": crun["meta"]["git_commit"], "bits": bits, "alphas": alphas}
    return scales, prov


def run(args):
    fam = check_condition(args)
    lim = runargs.check(args, [args.out, args.artifacts])

    import torch

    from . import storage
    from .artifacts import executed_hashes, load_artifact, save_artifact
    from .build import build_condition
    from .calib_stats import provenance
    from .data import SEQUENCE_FILES, file_info, forbid_held_out, read_sequences
    from .evaluate import calib_nll, write_rows
    from .model import Pristine, count_params, enumerate_eligible, load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    spec = C.CONDITIONS[args.condition]
    cfg = {"entry_point": "task1.compress", "condition": args.condition, "condition_spec": spec,
           "settings": C.settings_dict(), "device": args.device, "model_path": args.model_path, **lim,
           "output_files": OUTPUT_FILES, "artifacts_dir": str(args.artifacts), "calib_stats_dir": args.calib_stats,
           "allocation_file": args.allocation,
           "data_files": [file_info(SEQUENCE_FILES[s]) for s in ("c4_calibration", "c4_development")],
           "batch_size": C.CALIB_BATCH_SIZE}
    rec = RunRecorder(out / OUTPUT_FILES["run"], cfg)
    steps, timings = {}, {}
    cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    try:
        with forbid_held_out() as guard:
            t = time.perf_counter()
            model, info = load_model(args.device, args.model_path)
            pristine = Pristine(model)
            steps["restore"] = {"tensors_verified": pristine.restore(model)}
            timings["load_s"] = time.perf_counter() - t
            rec.update(model=info, eval_id=C.EVAL_ID, steps=steps, timings_s=timings)
            dev = next(model.parameters()).device

            calib = read_sequences("c4_calibration", limit=args.limit_calib)
            devel = read_sequences("c4_development", limit=args.limit_dev)
            if args.limit_calib is None:
                assert len(calib) == C.N_CALIBRATION
            if args.limit_dev is None:
                assert len(devel) == C.N_DEVELOPMENT

            awq_scales = zeros_pm = bits_pb = None
            if fam == "Q2":
                awq_scales, steps["awq_source"] = load_awq_scales(args.calib_stats, spec["bits"], args, dev)
            if fam in NEEDS_ALLOCATION:
                from .select import matrices_of

                cprov = provenance(args.calib_stats) if fam == "Q3" else None
                value, steps["allocation"] = load_allocation(args.allocation, args.condition, args,
                                                             matrices_of(model), cprov)
                if fam == "P3":
                    zeros_pm = value
                else:
                    bits_pb = value
                    awq_scales, steps["awq_source"] = {}, {}
                    for b in sorted(set(bits_pb)):
                        sc, steps["awq_source"][str(b)] = load_awq_scales(args.calib_stats, b, args, dev)
                        awq_scales.update({k: v for k, v in sc.items() if bits_pb[k[0]] == b})
                rec.update(steps=steps, timings_s=timings)

            def log(li, tm):
                steps["build_progress"] = {"last_block": li, "timings_s": tm}
                rec.update(steps=steps, timings_s=timings)

            if cuda:
                torch.cuda.reset_peak_memory_stats(dev)
            t = time.perf_counter()
            res = build_condition(model, calib, args.condition, C.CALIB_BATCH_SIZE, awq_scales=awq_scales,
                                  zeros_per_matrix=zeros_pm, bits_per_block=bits_pb, pristine=pristine, log=log)
            timings["build_total_s"] = time.perf_counter() - t
            timings.update({f"build_{k}": v for k, v in res.timings.items()})
            steps.pop("build_progress", None)
            peak_build = torch.cuda.max_memory_allocated(dev) if cuda else None

            # Per-projection records and global summaries.
            write_rows(out / OUTPUT_FILES["projections"], res.records)
            elig = enumerate_eligible(model)
            counts = count_params(model)
            shapes = [tuple(e.module.weight.shape) for e in elig]
            nnz = [r["nonzeros"] for r in res.records]
            assert [r["qualname"] for r in res.records] == [e.qualname for e in elig]
            assert nnz == [int(torch.count_nonzero(e.module.weight)) for e in elig]
            n_elig = counts["eligible"]
            err_num = sum(r["err_num"] for r in res.records)
            err_den = sum(r["err_den"] for r in res.records)
            steps["build"] = {
                "family": fam, "calibration_sequences": len(calib),
                "nonzeros": {"global": sum(nnz), "eligible_weights": n_elig, "nonzero_fraction": sum(nnz) / n_elig,
                             "per_projection": {r["qualname"]: r["nonzeros"] for r in res.records}},
                "output_error": {"what": "||(W_eff - W) X||_F^2 / ||W X||_F^2 per projection on dense calibration "
                                         "inputs; W_eff in the original basis", "pooled_rel": err_num / err_den,
                                 "mean_rel": sum(r["rel_output_error"] for r in res.records) / len(res.records),
                                 "max_rel": max(r["rel_output_error"] for r in res.records),
                                 "file": OUTPUT_FILES["projections"]},
                "score_distributions": res.score_edges or None,
                "peak_gpu_allocated_bytes": peak_build}
            if fam in ("P1", "P2", "P2_2of4", "P3"):
                steps["build"]["mask_zeros_total"] = sum(r["mask_zeros"] for r in res.records)
                steps["build"]["target_zeros_total"] = sum(r["target_zeros"] for r in res.records)
                assert steps["build"]["mask_zeros_total"] == steps["build"]["target_zeros_total"]
            if fam == "P3":
                assert steps["build"]["mask_zeros_total"] == steps["allocation"]["totals"]["p2_total_zeros"]
                assert {r["qualname"]: r["mask_zeros"] for r in res.records} == zeros_pm

            st = storage.report(shapes, counts["non_eligible"], nnz)
            if fam in ("Q1", "Q2", "Q3"):
                bits_pm = [r["bits"] for r in res.records]
                pq = storage.packed_quant_bytes(shapes, bits_pm, detail=True)
                st["packed_quant"] = {**pq, "bits": spec["bits"] if fam != "Q3" else bits_pb,
                                      "whole_model_bytes": storage.whole_model_bytes(pq["total"], counts["non_eligible"]),
                                      "bits_per_eligible_weight": storage.bits_per_eligible_weight(pq["total"], shapes),
                                      "extra_fp16_scale_bytes": 0}
                n_w = [a * b for a, b in shapes]
                st["nominal_bits_per_eligible_weight"] = sum(b * n for b, n in zip(bits_pm, n_w)) / sum(n_w)
                if fam == "Q3":
                    budget = steps["allocation"]["totals"]["budget_bytes"]
                    assert pq["total"] <= budget, (pq["total"], budget)
                    st["packed_quant"]["budget_bytes"] = budget
                    st["packed_quant"]["budget_whole_model_bytes"] = storage.whole_model_bytes(budget,
                                                                                               counts["non_eligible"])
                st["representation"] = "packed_quant"
            else:
                st["nominal_bits_per_eligible_weight"] = 16
                st["representation"] = "sparse16" if fam != "baseline" else "fp16_dense"
            steps["storage"] = st
            rec.update(steps=steps, timings_s=timings)

            # Calibration and development NLL of the executed model.
            steps["calib_nll"] = {}
            for split, seqs in (("c4_calibration", calib), ("c4_development", devel)):
                t = time.perf_counter()
                rows, summ = calib_nll(model, seqs, split=split)
                write_rows(out / OUTPUT_FILES[f"calib_nll_{split}"], rows)
                summ["file"] = OUTPUT_FILES[f"calib_nll_{split}"]
                steps["calib_nll"][split] = summ
                timings[f"calib_nll_{split}_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)

            # Artifact: save, then rebuild from restore() + artifact and compare every executed tensor.
            t = time.perf_counter()
            before = executed_hashes(model, sorted(res.extra_tensors))
            source = ({"model_path": str(args.model_path)} if args.model_path
                      else {"model_id": C.MODEL_ID, "revision": C.MODEL_REVISION})
            alloc_meta = ({"allocation": {k: steps["allocation"][k] for k in ("file", "sha256", "candidate_id")}}
                          if fam in NEEDS_ALLOCATION else None)
            man = save_artifact(args.artifacts, args.condition, source, res.payload, res.extra_tensors, model,
                                extra_meta=alloc_meta)
            timings["artifact_save_s"] = time.perf_counter() - t
            t = time.perf_counter()
            load_artifact(model, pristine, args.artifacts)
            after = executed_hashes(model, sorted(res.extra_tensors))
            assert after == before, "artifact reload does not reproduce the executed weights"
            timings["artifact_verify_s"] = time.perf_counter() - t
            art = Path(args.artifacts)
            steps["artifact"] = {"dir": str(art), "file": man["file"], "manifest_bytes": (art / "manifest.json").stat().st_size,
                                 "manifest_sha256": sha256_file(art / "manifest.json"),
                                 "logical_bytes": man["logical_bytes"], "stored_tensor_bytes": man["stored_tensor_bytes"],
                                 "n_tensors": len(man["tensors"]), "extra_tensors": man["extra_tensors"],
                                 "reload_matches_executed": True, "executed_tensors_checked": len(after)}
            steps["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(dev) if cuda else None
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
