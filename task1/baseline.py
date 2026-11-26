"""Unchanged FP16 baseline: enumeration, perplexity, HellaSwag, calibration NLL, lm_eval check, resources, storage.

The lm_eval check (first 512 HellaSwag IDs) prints one line per lm_eval run and the acceptance verdict to stdout
(on FAIL also disagreements and the largest per-candidate differences); see task1.verify_lmeval.

Usage: python -m task1.baseline --out results/task1/fp16 [--device cuda:0] [--model-path PATH]
       [--limit-windows N --limit-hellaswag N --limit-calib N]   (limits: tests and smoke runs only)
       [--no-resources | --resources-only]
--no-resources skips the latency and peak-memory step; --resources-only runs only enumeration, resources and storage
(no perplexity, HellaSwag, calibration NLL or lm_eval check) inside task1.data.forbid_held_out, so no held-out test
file can be opened. The default runs every step.
"""
import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

from common.io import write_json_new
from common.paths import RESULTS_DIR
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C

CONDITION = "fp16"
OUTPUT_FILES = {
    "run": "run.json",
    "ppl_wikitext2_test": "ppl_wikitext2_test.jsonl.gz",
    "ppl_c4_validation": "ppl_c4_validation.jsonl.gz",
    "hellaswag": "hellaswag_full.jsonl.gz",
    "calib_nll_c4_calibration": "calib_nll_c4_calibration.jsonl.gz",
    "calib_nll_c4_development": "calib_nll_c4_development.jsonl.gz",
    "lmeval_verify": "lmeval_verify.json",
}
STEPS = ("enumeration", "perplexity", "hellaswag", "calib_nll", "lmeval_verify", "resources", "storage")
RECORDED_ENV = ("PYTORCH_CUDA_ALLOC_CONF",)
RESOURCE_MODES = {"default": STEPS, "skip": tuple(x for x in STEPS if x != "resources"),
                  "only": ("enumeration", "resources", "storage")}


def resources_mode(args) -> str:
    """'default', 'skip' (--no-resources) or 'only' (--resources-only)."""
    return "only" if args.resources_only else "skip" if args.no_resources else "default"


def output_files(mode: str) -> dict:
    return {"run": OUTPUT_FILES["run"]} if mode == "only" else dict(OUTPUT_FILES)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--limit-hellaswag", type=int, default=None)
    ap.add_argument("--limit-calib", type=int, default=None)
    res = ap.add_mutually_exclusive_group()
    res.add_argument("--no-resources", action="store_true", help="skip the latency and peak-memory step")
    res.add_argument("--resources-only", action="store_true",
                     help="only enumeration, resources and storage; no held-out file is read")
    return ap.parse_args(argv)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_args(args) -> dict:
    """Refuse an existing output directory and limits combined with a results/task1 output path."""
    out = Path(args.out)
    limits = {"windows": args.limit_windows, "hellaswag": args.limit_hellaswag, "calib": args.limit_calib}
    limited = any(v is not None for v in limits.values())
    if limited and _under(out, RESULTS_DIR / "task1"):
        raise SystemExit(f"refusing limits with an output path under {RESULTS_DIR / 'task1'}: {out}")
    for v in limits.values():
        if v is not None and v < 1:
            raise SystemExit("limits must be positive")
    if out.exists():
        raise SystemExit(f"output directory exists: {out}")
    return {"limits": limits, "limited": limited}


def run(args):
    lim = check_args(args)
    mode = resources_mode(args)
    files = output_files(mode)

    from . import storage
    from .data import file_info, read_hellaswag, read_sequences, read_windows, HELLASWAG_FILE, SEQUENCE_FILES, WINDOW_FILES
    from .evaluate import calib_nll, hellaswag, perplexity, write_rows
    from .model import Pristine, count_params, enumerate_eligible, load_model, load_tokenizer, nonzeros_per_matrix
    from .resources import latency, prompt_ids
    from .data import forbid_held_out
    from .verify_lmeval import verify
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    cfg = {"condition": CONDITION, "condition_spec": C.CONDITIONS[CONDITION], "settings": C.settings_dict(),
           "device": args.device, "model_path": args.model_path, **lim,
           "hellaswag_scope": C.CONDITIONS[CONDITION]["hellaswag"], "output_files": files,
           "resources_mode": mode, "steps_run": list(RESOURCE_MODES[mode]),
           "env": {k: os.environ.get(k) for k in RECORDED_ENV},
           "data_files": [file_info(f) for f in (*WINDOW_FILES.values(), *SEQUENCE_FILES.values(), HELLASWAG_FILE)]}
    if mode == "only":
        cfg["data_files"] = [file_info(SEQUENCE_FILES["c4_calibration"])]
    rec = RunRecorder(out / files["run"], cfg)
    cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    dev = torch.device(args.device)
    steps = {}
    timings = {}
    rep = {"acceptance": None}
    guard_cm = forbid_held_out() if mode == "only" else contextlib.nullcontext(None)
    try:
        with guard_cm as guard:
            t = time.perf_counter()
            model, info = load_model(args.device, args.model_path)
            tok = load_tokenizer(args.model_path)
            pristine = Pristine(model)
            timings["load_s"] = time.perf_counter() - t
            rec.update(model=info, tokenizer={"class": type(tok).__name__, "name_or_path": tok.name_or_path,
                                              "len": len(tok), "bos_token_id": tok.bos_token_id,
                                              "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id},
                       eval_id=C.EVAL_ID, timings_s=timings)

            # Enumeration and parameter counts.
            t = time.perf_counter()
            elig = enumerate_eligible(model)
            counts = count_params(model)
            steps["enumeration"] = {"n_eligible_matrices": len(elig), "first": elig[0].qualname,
                                    "last": elig[-1].qualname, "per_block": len(C.ELIGIBLE), "param_counts": counts,
                                    "pristine_tensors_verified": pristine.verify(model)}
            if args.model_path is None:
                assert len(elig) == C.EXPECTED_ELIGIBLE_MATRICES and counts["eligible"] == C.EXPECTED_ELIGIBLE_PARAMS
                assert counts["total"] == C.EXPECTED_TOTAL_PARAMS
            timings["enumeration_s"] = time.perf_counter() - t
            rec.update(steps=steps, timings_s=timings)

            if mode != "only":
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

                # HellaSwag, full 2,000 IDs.
                t = time.perf_counter()
                scope = C.CONDITIONS[CONDITION]["hellaswag"]
                hs_records = read_hellaswag(limit=args.limit_hellaswag)
                if args.limit_hellaswag is None:
                    hs_records = hs_records[:C.hellaswag_n(scope)]
                exp = None if args.limit_hellaswag is not None else C.hellaswag_n(scope)
                hs_rows, hs_summ = hellaswag(model, tok, hs_records, scope=scope, expected_n=exp)
                write_rows(out / files["hellaswag"], hs_rows)
                hs_summ["file"] = files["hellaswag"]
                steps["hellaswag"] = hs_summ
                timings["hellaswag_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)

                # Calibration and development NLL.
                steps["calib_nll"] = {}
                calib_all = read_sequences("c4_calibration")
                for split in ("c4_calibration", "c4_development"):
                    t = time.perf_counter()
                    seqs = (calib_all if split == "c4_calibration" else read_sequences(split))[:args.limit_calib]
                    if args.limit_calib is None:
                        n_exp = {"c4_calibration": C.N_CALIBRATION, "c4_development": C.N_DEVELOPMENT}[split]
                        assert len(seqs) == n_exp
                    rows, summ = calib_nll(model, seqs, split=split)
                    write_rows(out / files[f"calib_nll_{split}"], rows)
                    summ["file"] = files[f"calib_nll_{split}"]
                    steps["calib_nll"][split] = summ
                    timings[f"calib_nll_{split}_s"] = time.perf_counter() - t
                    rec.update(steps=steps, timings_s=timings)

                # lm_eval check on the first 512 IDs: acceptance on a float32 copy of the model (the FP16 model waits
                # on CPU meanwhile), two informational runs on the FP16 model. Peak memory of this step alone: the CUDA
                # peak statistics are reset at its start (the peak before the step is kept as
                # peak_allocated_bytes_before_step). Parameter dtypes, devices and fingerprints are checked afterwards.
                t = time.perf_counter()
                n_v = min(C.HELLASWAG_SUBSET_N, len(hs_rows))
                pre_peak = None
                if cuda:
                    torch.cuda.synchronize(dev)
                    pre_peak = torch.cuda.max_memory_allocated(dev)
                    torch.cuda.reset_peak_memory_stats(dev)
                dtypes_before = sorted({str(p.dtype) for p in model.parameters()})
                devices_before = sorted({str(p.device) for p in model.parameters()})
                rep = verify(model, tok, hs_records[:n_v], hs_rows[:n_v])
                if cuda:
                    torch.cuda.synchronize(dev)
                peak_a = torch.cuda.max_memory_allocated(dev) if cuda else None
                peak_r = torch.cuda.max_memory_reserved(dev) if cuda else None
                print_lmeval(rep)
                write_json_new(out / files["lmeval_verify"], rep)
                dtypes_after = sorted({str(p.dtype) for p in model.parameters()})
                assert dtypes_after == dtypes_before, (dtypes_before, dtypes_after)
                devices_after = sorted({str(p.device) for p in model.parameters()})
                assert devices_after == devices_before, (devices_before, devices_after)
                n_fp = pristine.verify(model)  # parameter fingerprints unchanged since load
                steps["lmeval_verify"] = {"file": files["lmeval_verify"], "n_examples": n_v,
                                          "acceptance_run": rep["acceptance_run"], "acceptance": rep["acceptance"],
                                          "acceptance_detail": rep["acceptance_detail"],
                                          "runs": {k: {kk: v[kk] for kk in v if kk != "per_example"}
                                                   for k, v in rep["runs"].items()},
                                          "peak_allocated_bytes": peak_a, "peak_reserved_bytes": peak_r,
                                          "peak_allocated_bytes_before_step": pre_peak,
                                          "batch_sizes": {k: v["batch_size"] for k, v in rep["runs"].items()},
                                          "fp16_model_after": {"parameter_dtypes": dtypes_after,
                                                               "parameter_devices": devices_after,
                                                               "fingerprints_verified": n_fp}}
                timings["lmeval_verify_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)

            if mode != "skip":
                # Resources.
                t = time.perf_counter()
                prompts = prompt_ids(calib_all if mode != "only" else read_sequences("c4_calibration"))
                steps["resources"] = latency(model, prompts)
                timings["resources_s"] = time.perf_counter() - t
                rec.update(steps=steps, timings_s=timings)

            # Storage: FP16 dense and the 16-bit sparse representation of the same weights.
            shapes = [tuple(e.module.weight.shape) for e in elig]
            steps["storage"] = storage.report(shapes, counts["non_eligible"], nonzeros_per_matrix(model))
            if args.model_path is None:
                assert shapes == C.full_model_matrix_shapes()
            steps["pristine_after"] = {"tensors_verified": pristine.verify(model)}
        if guard is not None:
            steps["held_out_guard"] = guard
            assert not guard["attempts"], guard
        rec.finish(run_status(rep["acceptance"]), steps=steps, timings_s=timings, acceptance=rep["acceptance"])
    except BaseException as exc:  # record the failure, then propagate
        rec.finish("failed", steps=steps, timings_s=timings, error=repr(exc), traceback=traceback.format_exc())
        raise
    return out, rep["acceptance"]


def print_lmeval(rep: dict):
    """The lm_eval check lines (one per run, the verdict; on FAIL disagreements and largest differences) to stdout."""
    from .verify_lmeval import report_lines

    for line in report_lines(rep):
        print(line, flush=True)


def run_status(acceptance) -> str:
    """Run status after all outputs are written: ok on PASS, acceptance_failed on FAIL; ok without the lm_eval check
    (--resources-only, acceptance None)."""
    return {"PASS": "ok", "FAIL": "acceptance_failed", None: "ok"}[acceptance]


def exit_code(acceptance) -> int:
    return 0 if acceptance in ("PASS", None) else 1


def main(argv=None):
    args = parse_args(argv)
    _, acc = run(args)
    return exit_code(acc)


if __name__ == "__main__":
    sys.exit(main())
