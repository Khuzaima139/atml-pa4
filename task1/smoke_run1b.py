"""Smoke timing of the run-1b work units (Q3 design v2) on one GPU, the input of the run-1b time projection.

Usage: python -m task1.smoke_run1b --out DIR --calib-stats DIR --artifacts DIR [--device cuda:0]
       [--model-path PATH --limit-calib N --limit-dev N]   (tests only; refused for outputs under results/task1)
Units, each with wall time, peak allocated/reserved GPU memory and host peak RSS:
  load               model load and the pristine CPU snapshot
  restore            one restore() of every parameter with fingerprint verification
  calib_pass         calibration NLL over all calibration sequences (the FP16 reference of a probe part)
  dev_pass           development NLL over all development sequences
  worst_batch_calib  one calibration batch of CALIB_BATCH_SIZE sequences
  capture            block-0 inputs of all calibration sequences
  block_forward      one dense forward of block 0 on that stream (reaching a probe part's first block)
  qs2v2_block        QS2v2 of block 0: own-group folded quantization at all four bit widths, each measured on the
                     dense stream
  isolated_unit      one QS1v2 pass: restore(), fold and quantize the largest unit (block 0 mlp_in) at 2 bits,
                     calibration NLL over all calibration sequences
  quant_build        a full Q3 design-v2 build (per-matrix bits: uniform 4, AWQ scales at 4 bits)
  artifact_save      that build's artifact written to --artifacts
  artifact_verify    restore() plus loading that artifact with sha256 checks
Reads only the calibration and development files (task1.data.forbid_held_out); AWQ scales from --calib-stats.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs
from .smoke import files_written, host_maxrss_bytes

UNITS = ("load", "restore", "calib_pass", "dev_pass", "worst_batch_calib", "capture", "block_forward", "qs2v2_block",
         "isolated_unit", "quant_build", "artifact_save", "artifact_verify")
OUTPUT_FILES = {"run": "run.json"}
SMOKE_CONDITION = "q3_int4"
SMOKE_UNIT = "mlp_in"
SMOKE_UNIT_BITS = 2
SMOKE_BUILD_BITS = 4


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-stats", required=True)
    ap.add_argument("--artifacts", required=True, help="new directory for the smoke artifact")
    runargs.add_test_args(ap)
    return ap.parse_args(argv)


def run(args):
    lim = runargs.check(args, [args.out, args.artifacts])

    import torch

    from . import q3v2
    from .artifacts import load_artifact, save_artifact
    from .awq import geometry
    from .blocks import capture_block_inputs, layers, projections, run_block
    from .build import SCALE_GROUPS, build_condition, error_pass
    from .calib_stats import provenance
    from .compress import load_awq_scales
    from .data import SEQUENCE_FILES, file_info, forbid_held_out, read_sequences
    from .evaluate import calib_nll, score_spans
    from .model import Pristine, count_params, enumerate_eligible, load_model
    from .select import matrices_of

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    cfg = {"entry_point": "task1.smoke_run1b", "design": C.Q3V2_DESIGN, "units": list(UNITS),
           "settings": C.settings_dict(), "q3v2_settings": C.q3v2_settings(), "device": args.device,
           "model_path": args.model_path, **lim, "output_files": OUTPUT_FILES, "artifacts_dir": str(args.artifacts),
           "calib_stats_dir": args.calib_stats,
           "smoke_choices": {"condition": SMOKE_CONDITION, "isolated_unit": SMOKE_UNIT,
                             "isolated_unit_bits": SMOKE_UNIT_BITS, "build_bits": SMOKE_BUILD_BITS},
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
            steps["calib_stats_source"] = provenance(args.calib_stats)
            scales = {}
            for b in C.Q3V2_BITS:
                sc, _ = load_awq_scales(args.calib_stats, b, args, dev)
                scales.update({(li, g, b): v for (li, g), v in sc.items()})
            calib = read_sequences("c4_calibration", limit=args.limit_calib)
            devel = read_sequences("c4_development", limit=args.limit_dev)
            if args.limit_calib is None:
                assert len(calib) == C.N_CALIBRATION
            if args.limit_dev is None:
                assert len(devel) == C.N_DEVELOPMENT
            mats = matrices_of(model)
            units = q3v2.units_of(mats)
            ch = q3v2.chunking(units)
            elig = enumerate_eligible(model)
            counts = count_params(model)
            L = len(layers(model))
            steps["counts"] = {"n_calibration": len(calib), "n_development": len(devel), "n_layers": L,
                               "n_units": len(units), "chunk_groups": ch["chunk_groups"],
                               "unit_chunks_block0": ch["unit_chunks"][:len(C.Q3V2_UNITS)],
                               "eligible_shapes": [list(e.module.weight.shape) for e in elig],
                               "non_eligible_params": counts["non_eligible"], "seq_len": len(calib[0]["token_ids"])}

            unit("calib_pass", lambda: calib_nll(model, calib, split="c4_calibration"), n_sequences=len(calib))
            unit("dev_pass", lambda: calib_nll(model, devel, split="c4_development"), n_sequences=len(devel))
            wb = calib[:C.CALIB_BATCH_SIZE]
            unit("worst_batch_calib", lambda: score_spans(model, [s["token_ids"] for s in wb],
                                                          [(1, len(s["token_ids"])) for s in wb], C.CALIB_BATCH_SIZE),
                 batch=[len(wb), len(wb[0]["token_ids"])])
            stream = unit("capture", lambda: capture_block_inputs(model, calib, C.CALIB_BATCH_SIZE))
            block0 = layers(model)[0]
            unit("block_forward", lambda: run_block(block0, stream))
            geom = geometry(model.config)

            def qs2v2():
                W = {n: m.weight.detach().float() for n, m in projections(block0).items()}
                for b in C.Q3V2_BITS:
                    E = q3v2.own_fold_effective(block0, W, b, {g: scales[(0, g, b)] for g in SCALE_GROUPS}, geom)
                    error_pass(block0, stream, W, E)

            unit("qs2v2_block", qs2v2, bits=list(C.Q3V2_BITS))
            del stream

            def isolated():
                pristine.restore(model)
                g = q3v2.UNIT_GROUP[SMOKE_UNIT]
                q3v2.fold_quant_unit(block0, SMOKE_UNIT, SMOKE_UNIT_BITS, scales[(0, g, SMOKE_UNIT_BITS)], geom)
                return calib_nll(model, calib, split="c4_calibration")

            unit("isolated_unit", isolated, unit=q3v2.unit_id(0, SMOKE_UNIT), bits=SMOKE_UNIT_BITS)
            pristine.restore(model)

            bpm = {m["qualname"]: SMOKE_BUILD_BITS for m in mats}
            aw = {(u["layer"], u["scale_group"]): scales[(u["layer"], u["scale_group"], SMOKE_BUILD_BITS)]
                  for u in units}
            built = unit("quant_build", lambda: build_condition(model, calib, SMOKE_CONDITION, C.CALIB_BATCH_SIZE,
                                                                awq_scales=aw, bits_per_matrix=bpm,
                                                                pristine=pristine), condition=SMOKE_CONDITION)
            steps["units"]["quant_build"]["build_timings"] = built.timings
            source = ({"model_path": str(args.model_path)} if args.model_path
                      else {"model_id": C.MODEL_ID, "revision": C.MODEL_REVISION})
            man = unit("artifact_save", lambda: save_artifact(args.artifacts, SMOKE_CONDITION, source, built.payload,
                                                              built.extra_tensors, model,
                                                              extra_meta={"design": C.Q3V2_DESIGN}))
            unit("artifact_verify", lambda: load_artifact(model, pristine, args.artifacts))
            art = Path(args.artifacts)
            steps["artifact"] = {"condition": SMOKE_CONDITION, "dir": str(art), "file_bytes": man["file"]["bytes"],
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
