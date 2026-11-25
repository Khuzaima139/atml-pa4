"""Smoke timing of full-size work units on one GPU, the input of the run-1 time and disk projection.

Usage: python -m task1.smoke --out DIR --artifacts DIR [--device cuda:0]
       [--model-path PATH --limit-calib N --limit-dev N]   (tests only; refused for outputs under results/task1)
Units, each with wall time, peak allocated/reserved GPU memory and host peak RSS:
  load               model load and the pristine CPU snapshot
  restore            one restore() of every parameter with fingerprint verification
  calib_pass         calibration NLL over all calibration sequences (one probe or selection pass)
  dev_pass           development NLL over all development sequences
  worst_batch_calib  one calibration batch of CALIB_BATCH_SIZE sequences (the largest fixed batch here)
  capture            block-0 inputs of all calibration sequences
  awq_block          AWQ scale search of block 0 at all four bit widths with its preservation check
  qs2_block          block 0 AWQ-folded and quantized at all four bit widths, measured on the dense stream
  p1_build           P1 at 0.50 (all blocks, score distributions included)
  wanda_build        P2 at 0.50, sequential Wanda (all blocks, score distributions included)
  quant_build        Q1 at 8 bits (all blocks)
  artifact_save      the Q1 8-bit artifact written to --artifacts
  artifact_verify    restore() plus loading that artifact with sha256 checks
Reads only the calibration and development files (task1.data.forbid_held_out).
"""
import argparse
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs

UNITS = ("load", "restore", "calib_pass", "dev_pass", "worst_batch_calib", "capture", "awq_block", "qs2_block",
         "p1_build", "wanda_build", "quant_build", "artifact_save", "artifact_verify")
OUTPUT_FILES = {"run": "run.json"}


def host_maxrss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifacts", required=True, help="new directory for the Q1 8-bit artifact")
    runargs.add_test_args(ap)
    return ap.parse_args(argv)


def files_written(*dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and ".tmp" not in p.name:
                out[str(p)] = p.stat().st_size
    return out


def run(args):
    lim = runargs.check(args, [args.out, args.artifacts])

    import torch

    from . import awq
    from .artifacts import load_artifact, save_artifact
    from .blocks import capture_block_inputs, layers, projections
    from .build import SCALE_GROUPS, build_condition, error_pass, quantize_block
    from .data import SEQUENCE_FILES, file_info, forbid_held_out, read_sequences
    from .evaluate import calib_nll, score_spans
    from .model import Pristine, count_params, enumerate_eligible, load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    cfg = {"entry_point": "task1.smoke", "units": list(UNITS), "settings": C.settings_dict(), "device": args.device,
           "model_path": args.model_path, **lim, "output_files": OUTPUT_FILES, "artifacts_dir": str(args.artifacts),
           "data_files": [file_info(SEQUENCE_FILES[s]) for s in ("c4_calibration", "c4_development")],
           "batch_size": C.CALIB_BATCH_SIZE}
    rec = RunRecorder(out / OUTPUT_FILES["run"], cfg)
    steps = {"units": {}}
    cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    dev = torch.device(args.device)

    def unit(name, fn, **info):
        if cuda:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        t = time.perf_counter()
        res = fn()
        if cuda:
            torch.cuda.synchronize(dev)
        steps["units"][name] = {
            "wall_s": time.perf_counter() - t,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev) if cuda else None,
            "host_maxrss_bytes": host_maxrss_bytes(), **info}
        rec.update(steps=steps)
        return res

    try:
        with forbid_held_out() as guard:
            def load():
                m, i = load_model(args.device, args.model_path)
                return m, i, Pristine(m)

            model, info, pristine = unit("load", load, includes_pristine_snapshot=True)
            unit("restore", lambda: pristine.restore(model))
            rec.update(model=info, eval_id=C.EVAL_ID)
            calib = read_sequences("c4_calibration", limit=args.limit_calib)
            devel = read_sequences("c4_development", limit=args.limit_dev)
            if args.limit_calib is None:
                assert len(calib) == C.N_CALIBRATION
            if args.limit_dev is None:
                assert len(devel) == C.N_DEVELOPMENT
            elig = enumerate_eligible(model)
            counts = count_params(model)
            L = len(layers(model))
            steps["counts"] = {"n_calibration": len(calib), "n_development": len(devel), "n_layers": L,
                               "eligible_shapes": [list(e.module.weight.shape) for e in elig],
                               "non_eligible_params": counts["non_eligible"],
                               "seq_len": len(calib[0]["token_ids"])}

            unit("calib_pass", lambda: calib_nll(model, calib, split="c4_calibration"), n_sequences=len(calib))
            unit("dev_pass", lambda: calib_nll(model, devel, split="c4_development"), n_sequences=len(devel))
            wb = calib[:C.CALIB_BATCH_SIZE]
            unit("worst_batch_calib", lambda: score_spans(model, [s["token_ids"] for s in wb],
                                                          [(1, len(s["token_ids"])) for s in wb], C.CALIB_BATCH_SIZE),
                 batch=[len(wb), len(wb[0]["token_ids"])])

            stream = unit("capture", lambda: capture_block_inputs(model, calib, C.CALIB_BATCH_SIZE))
            shim = SimpleNamespace(config=model.config, model=SimpleNamespace(layers=layers(model)[:1]),
                                   parameters=model.parameters)
            res = unit("awq_block", lambda: awq.search_model(shim, stream), bits=list(awq.SEARCH_BITS),
                       n_alphas=len(awq.ALPHA_GRID))
            steps["units"]["awq_block"]["block_timings"] = res["block_timings"]
            scales = res["scales"]
            block0 = layers(model)[0]
            geom = awq.geometry(model.config)

            def qs2():
                W = {n: m.weight.detach().float() for n, m in projections(block0).items()}
                for b in awq.SEARCH_BITS:
                    _, E, _, _ = quantize_block(block0, W, b, {g: scales[(0, g, b)] for g in SCALE_GROUPS}, geom)
                    error_pass(block0, stream, W, E)

            unit("qs2_block", qs2, bits=list(awq.SEARCH_BITS))
            del stream, res
            pristine.verify(model)

            for name, cond in (("p1_build", "p1_s50"), ("wanda_build", "p2_s50"), ("quant_build", "q1_b8")):
                pristine.restore(model)
                built = unit(name, lambda c=cond: build_condition(model, calib, c, C.CALIB_BATCH_SIZE,
                                                                  pristine=pristine), condition=cond)
                steps["units"][name]["build_timings"] = built.timings
            source = ({"model_path": str(args.model_path)} if args.model_path
                      else {"model_id": C.MODEL_ID, "revision": C.MODEL_REVISION})
            man = unit("artifact_save", lambda: save_artifact(args.artifacts, "q1_b8", source, built.payload,
                                                              built.extra_tensors, model))
            unit("artifact_verify", lambda: load_artifact(model, pristine, args.artifacts))
            art = Path(args.artifacts)
            steps["artifact"] = {"condition": "q1_b8", "dir": str(art), "file_bytes": man["file"]["bytes"],
                                 "stored_tensor_bytes": man["stored_tensor_bytes"],
                                 "manifest_bytes": (art / "manifest.json").stat().st_size,
                                 "n_tensors": len(man["tensors"]), "logical_bytes": man["logical_bytes"]}
            del built
            pristine.restore(model)
            steps["pristine_after"] = {"tensors_verified": pristine.verify(model)}
        steps["held_out_guard"] = guard
        assert not guard["attempts"], guard
        assert set(steps["units"]) == set(UNITS), sorted(steps["units"])
        steps["files_written"] = files_written(out, args.artifacts)
        steps["files_written"].pop(str(out / OUTPUT_FILES["run"]), None)
        rec.finish("ok", steps=steps)
    except BaseException as exc:
        rec.finish("failed", steps=steps, error=repr(exc), traceback=traceback.format_exc())
        raise
    return out


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
