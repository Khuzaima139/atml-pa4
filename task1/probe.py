"""Block sensitivity probes for the P3 and Q3 allocations, on the C4 calibration file only.

Usage: python -m task1.probe --kind prune|quant|quant_v2 --calib-stats DIR --out DIR [--blocks A-B] [--device cuda:0]
       [--model-path PATH --limit-calib N]   (tests only; refused for outputs under results/task1)
prune: S1 = OWL outlier ratio of each block (dense calibration norms, no forward);
       S2 = isolated loss increase with only that block Wanda-pruned at S in PROBE_SPARSITIES.
quant: QS1 = isolated loss increase with only that block AWQ-folded and quantized at b in PROBE_BITS;
       QS2 = sum over the block's projections of the relative output error at b on dense calibration inputs.
quant_v2 (Q3 design v2, fold-group units attn_in/attn_out/mlp_in/mlp_out of blocks A..B, --blocks A-B, default all):
       QS1v2 = isolated loss increase with only that unit folded (its group's scales searched at b) and only its
               projections quantized at b;
       QS2v2 = sum over the unit's projections of the relative output error at b (own-group fold) on dense inputs.
       Each part records its own FP16 reference; task1.select --kind q3v2 joins the parts.
Every isolated probe starts from restore(); the FP16 calibration NLL is measured once and subtracted.
No held-out file can be opened while the probe runs (task1.data.forbid_held_out).
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

from common.io import sha256_file, write_json_new
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs

OUTPUT_FILES = {
    "run": "run.json",
    "sensitivity": "sensitivity.json",
    "probe_rows": "probe_rows.jsonl.gz",
    "probe_calib_nll": "probe_calib_nll.jsonl.gz",
}
KINDS = {"prune": ("S1", "S2"), "quant": ("QS1", "QS2"), "quant_v2": C.Q3V2_MEASURES}
DEFINITIONS = {
    "S1": "fraction of the block's Wanda scores |W_ij|*||X_j|| (dense calibration norms) over its seven projections "
          "greater than OWL_M times their mean (OWL outlier ratio); no forward pass",
    "S2": "calibration mean NLL with only this block Wanda-pruned at S (dense calibration norms, per-row masks, other "
          "blocks dense) minus the FP16 calibration mean NLL",
    "QS1": "calibration mean NLL with only this block AWQ-folded (scales searched at b) and quantized at b minus the "
           "FP16 calibration mean NLL",
    "QS2": "sum over the block's seven projections of ||(W_eff - W) X||_F^2 / ||W X||_F^2 at b on dense calibration "
           "inputs, W_eff the AWQ-folded quantized weight in the original basis",
    "QS1v2": "calibration mean NLL after restore() with only this fold-group unit folded (its group's AWQ scales "
             "searched at b; partner tensors rewritten, no other fold) and only its projections quantized at b, minus "
             "the FP16 calibration mean NLL of the same probe run",
    "QS2v2": "sum over the unit's projections of ||(W_eff - W) X||_F^2 / ||W X||_F^2 at b on dense calibration inputs, "
             "W_eff = Q(W diag(s)) diag(1/s) in the original basis with s the unit's own group scales at b",
}


def level_key(level) -> str:
    """Key of a probe level: sparsity '0.50' / '0.75', bits '2' / '3' / '4' / '8'."""
    return str(level) if isinstance(level, int) else f"{float(level):.2f}"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--calib-stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default=None, help="block range A-B, inclusive (quant_v2 only; default all blocks)")
    runargs.add_test_args(ap)
    return ap.parse_args(argv)


def parse_blocks(text) -> tuple:
    """'A-B' -> (A, B) with 0 <= A <= B."""
    try:
        a, b = (int(x) for x in str(text).split("-"))
    except ValueError:
        raise SystemExit(f"--blocks must be A-B with integers, got {text!r}")
    if not 0 <= a <= b:
        raise SystemExit(f"--blocks needs 0 <= A <= B, got {text!r}")
    return a, b


def owl_ratio(weights, norms, m: float = C.OWL_M) -> dict:
    """OWL outlier ratio of one block: share of Wanda scores above m times the mean over all its projections."""
    import torch

    with torch.no_grad():
        scores = [w.detach().float().abs() * x.float().to(w.device)[None, :] for w, x in zip(weights, norms)]
        n = sum(s.numel() for s in scores)
        mean = sum(float(s.sum(dtype=torch.float64)) for s in scores) / n
        n_out = sum(int((s > m * mean).sum()) for s in scores)
    return {"value": n_out / n, "n_scores": n, "n_outliers": n_out, "mean_score": mean, "threshold": m * mean}


def prune_block(model, li: int, S: float, x_l2: dict) -> dict:
    """Wanda-prune the seven projections of block li at S with the given input norms; returns zeros per matrix."""
    import torch

    from . import prune as P
    from .blocks import layers, projections
    from .build import _qual

    out = {}
    with torch.no_grad():
        for n, mod in projections(layers(model)[li]).items():
            qual = _qual(li, n)
            w = mod.weight
            z = P.target_zeros(tuple(w.shape), S)
            keep = P.row_mask(P.wanda_score(w, x_l2[qual]), z)
            w.masked_fill_(~keep, 0)
            out[qual] = z
    return out


def quant_block(model, li: int, bits: int, block_scales: dict) -> None:
    """AWQ-fold block li with the scales searched at bits and quantize its seven projections at bits."""
    from .awq import geometry
    from .blocks import layers, projections
    from .build import apply_block, quantize_block

    block = layers(model)[li]
    W = {n: m.weight.detach().float() for n, m in projections(block).items()}
    execw, _, extra, _ = quantize_block(block, W, bits, block_scales, geometry(model.config))
    apply_block(block, execw, extra)


def qs2_errors(model, calib, scales: dict, bits_list, batch_size: int = C.CALIB_BATCH_SIZE) -> dict:
    """{(layer, bits): {name: (err_num, err_den)}} on the dense stream; the model is not modified."""
    from .awq import geometry
    from .blocks import capture_block_inputs, layers, projections
    from .build import SCALE_GROUPS, error_pass, quantize_block

    geom = geometry(model.config)
    stream = capture_block_inputs(model, calib, batch_size)
    out = {}
    for li, block in enumerate(layers(model)):
        W = {n: m.weight.detach().float() for n, m in projections(block).items()}
        nxt = None
        for b in bits_list:
            _, E, _, _ = quantize_block(block, W, b, {g: scales[(li, g, b)] for g in SCALE_GROUPS}, geom)
            acc, _, nxt = error_pass(block, stream, W, E)
            out[(li, b)] = {n: tuple(acc[n]) for n in C.ELIGIBLE}
        stream = nxt
    return out


def run_quant_v2(model, pristine, calib, scales, lo, hi, values, probe_rows, effort, ref, nll_pass, steps, rec,
                 timings) -> list:
    """QS2v2 and QS1v2 of every fold-group unit of blocks lo..hi; fills values, probe_rows and effort."""
    from . import q3v2
    from .awq import geometry
    from .blocks import capture_block_inputs, layers, projections, run_block
    from .build import SCALE_GROUPS, error_pass
    from .select import matrices_of

    geom = geometry(model.config)
    units = [u for u in q3v2.units_of(matrices_of(model)) if lo <= u["layer"] <= hi]
    n_seq = len(calib)
    bits = C.Q3V2_BITS

    t0 = time.perf_counter()
    stream = capture_block_inputs(model, calib, C.CALIB_BATCH_SIZE)
    for li in range(lo):
        stream = run_block(layers(model)[li], stream)
    values["QS2v2"] = {level_key(b): [] for b in bits}
    for li in range(lo, hi + 1):
        block = layers(model)[li]
        W = {n: m.weight.detach().float() for n, m in projections(block).items()}
        nxt = None
        for b in bits:
            E = q3v2.own_fold_effective(block, W, b, {g: scales[(li, g, b)] for g in SCALE_GROUPS}, geom)
            acc, _, nxt = error_pass(block, stream, W, E)
            for u in units:
                if u["layer"] != li:
                    continue
                v = sum(acc[n][0] / acc[n][1] for n in u["projections"])
                values["QS2v2"][level_key(b)].append(v)
                probe_rows.append({"eval_id": C.EVAL_ID, "kind": "quant_v2", "design": C.Q3V2_DESIGN,
                                   "measure": "QS2v2", "level": level_key(b), "layer": li, "unit": u["unit"],
                                   "unit_id": u["id"], "unit_index": u["index"], "value": v,
                                   "projections": {n: {"err_num": acc[n][0], "err_den": acc[n][1],
                                                       "rel": acc[n][0] / acc[n][1]} for n in u["projections"]}})
            del E
        stream = nxt
        del W
    del stream
    n_blocks = hi - lo + 1
    effort["QS2v2"] = {"wall_s": time.perf_counter() - t0, "full_model_calibration_passes": 0,
                       "block_forward_passes": lo + n_blocks * len(bits), "calibration_sequences_per_pass": n_seq,
                       "restores": 0, "blocks_run_to_reach_range": lo}

    t0 = time.perf_counter()
    values["QS1v2"] = {}
    passes = 0
    for b in bits:
        key = level_key(b)
        values["QS1v2"][key] = []
        for u in units:
            t1 = time.perf_counter()
            pristine.restore(model)
            li = u["layer"]
            detail = q3v2.fold_quant_unit(layers(model)[li], u["unit"], b, scales[(li, u["scale_group"], b)], geom)
            summ = nll_pass("QS1v2", key, li, unit=u["unit"], unit_id=u["id"])
            passes += 1
            delta = summ["mean_nll"] - ref["mean_nll"]
            values["QS1v2"][key].append(delta)
            probe_rows.append({"eval_id": C.EVAL_ID, "kind": "quant_v2", "design": C.Q3V2_DESIGN, "measure": "QS1v2",
                               "level": key, "layer": li, "unit": u["unit"], "unit_id": u["id"],
                               "unit_index": u["index"], "value": delta, "calib_mean_nll": summ["mean_nll"],
                               "calib_nll_sum": summ["nll_sum"], "n_scored": summ["N"],
                               "fp16_calib_mean_nll": ref["mean_nll"], "wall_s": time.perf_counter() - t1,
                               **{k: v for k, v in detail.items() if k not in ("unit",)}})
            steps["progress"] = {"measure": "QS1v2", "level": key, "unit": u["id"], "done": passes,
                                 "of": len(units) * len(bits)}
            rec.update(steps=steps, timings_s=timings)
    pristine.restore(model)
    effort["QS1v2"] = {"wall_s": time.perf_counter() - t0, "full_model_calibration_passes": passes,
                       "block_forward_passes": 0, "calibration_sequences_per_pass": n_seq, "restores": passes}
    return units


def run(args):
    if args.limit_dev is not None:
        raise SystemExit("--limit-dev is not used by the probes")
    if args.blocks is not None and args.kind != "quant_v2":
        raise SystemExit("--blocks is only used by --kind quant_v2")
    block_arg = parse_blocks(args.blocks) if args.blocks is not None else None
    lim = runargs.check(args, [args.out])

    import torch

    from .blocks import layers, projections
    from .build import SCALE_GROUPS, _qual
    from .calib_stats import load_dense_l2, provenance
    from .compress import load_awq_scales
    from .data import SEQUENCE_FILES, file_info, forbid_held_out, read_sequences
    from .evaluate import calib_nll, write_rows
    from .model import Pristine, load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    measures = KINDS[args.kind]
    cfg = {"entry_point": "task1.probe", "kind": args.kind, "measures": list(measures),
           "definitions": {m: DEFINITIONS[m] for m in measures}, "settings": C.settings_dict(),
           "alloc_settings": C.alloc_settings(), "device": args.device, "model_path": args.model_path, **lim,
           "output_files": OUTPUT_FILES, "calib_stats_dir": args.calib_stats,
           "data_files": [file_info(SEQUENCE_FILES["c4_calibration"])], "batch_size": C.CALIB_BATCH_SIZE}
    if args.kind == "quant_v2":
        cfg.update(design=C.Q3V2_DESIGN, q3v2_settings=C.q3v2_settings(), blocks=args.blocks)
    rec = RunRecorder(out / OUTPUT_FILES["run"], cfg)
    steps, timings = {}, {}
    try:
        with forbid_held_out() as guard:
            t = time.perf_counter()
            model, info = load_model(args.device, args.model_path)
            pristine = Pristine(model)
            steps["restore"] = {"tensors_verified": pristine.restore(model)}
            timings["load_s"] = time.perf_counter() - t
            rec.update(model=info, eval_id=C.EVAL_ID, steps=steps, timings_s=timings)
            dev = next(model.parameters()).device
            L = len(layers(model))
            if args.kind == "quant_v2":
                lo, hi = block_arg if block_arg is not None else (0, L - 1)
                if hi >= L:
                    raise SystemExit(f"--blocks {args.blocks} exceeds the {L} blocks of this model")
                steps["block_range"] = [lo, hi]

            steps["calib_stats_source"] = provenance(args.calib_stats)
            if args.kind == "prune":
                x_l2, _ = load_dense_l2(args.calib_stats, args, dev)
            else:
                scales, alphas = {}, {}
                for b in C.PROBE_BITS:
                    sc, prov = load_awq_scales(args.calib_stats, b, args, dev)
                    scales.update({(li, g, b): v for (li, g), v in sc.items()})
                    alphas[str(b)] = prov["alphas"]
                steps["awq_alphas"] = alphas
            calib = read_sequences("c4_calibration", limit=args.limit_calib)
            if args.limit_calib is None:
                assert len(calib) == C.N_CALIBRATION
            n_seq = len(calib)

            nll_rows, probe_rows = [], []

            def nll_pass(measure, level, layer, **extra):
                rows, summ = calib_nll(model, calib, split="c4_calibration")
                for r in rows:
                    nll_rows.append({**r, "measure": measure, "level": level, "layer": layer, **extra})
                return summ

            t = time.perf_counter()
            ref = nll_pass("fp16", None, None, **({"unit": None, "unit_id": None} if args.kind == "quant_v2" else {}))
            timings["fp16_calib_nll_s"] = time.perf_counter() - t
            steps["fp16_calib_nll"] = ref
            effort = {"fp16_reference": {"wall_s": timings["fp16_calib_nll_s"], "full_model_calibration_passes": 1,
                                         "block_forward_passes": 0, "calibration_sequences_per_pass": n_seq}}
            values = {}
            rec.update(steps=steps, timings_s=timings)

            def isolated(measure, levels, mutate):
                t0 = time.perf_counter()
                vals, passes = {}, 0
                for lv in levels:
                    key = level_key(lv)
                    vals[key] = []
                    for li in range(L):
                        t1 = time.perf_counter()
                        pristine.restore(model)
                        detail = mutate(li, lv)
                        summ = nll_pass(measure, key, li)
                        passes += 1
                        delta = summ["mean_nll"] - ref["mean_nll"]
                        vals[key].append(delta)
                        probe_rows.append({"eval_id": C.EVAL_ID, "kind": args.kind, "measure": measure, "level": key,
                                           "layer": li, "value": delta, "calib_mean_nll": summ["mean_nll"],
                                           "calib_nll_sum": summ["nll_sum"], "n_scored": summ["N"],
                                           "fp16_calib_mean_nll": ref["mean_nll"], "wall_s": time.perf_counter() - t1,
                                           **detail})
                        steps["progress"] = {"measure": measure, "level": key, "layer": li}
                        rec.update(steps=steps, timings_s=timings)
                pristine.restore(model)
                effort[measure] = {"wall_s": time.perf_counter() - t0, "full_model_calibration_passes": passes,
                                   "block_forward_passes": 0, "calibration_sequences_per_pass": n_seq,
                                   "restores": passes}
                return vals

            if args.kind == "prune":
                t0 = time.perf_counter()
                s1 = []
                for li, block in enumerate(layers(model)):
                    projs = projections(block)
                    r = owl_ratio([projs[n].weight for n in C.ELIGIBLE], [x_l2[_qual(li, n)] for n in C.ELIGIBLE])
                    s1.append(r["value"])
                    probe_rows.append({"eval_id": C.EVAL_ID, "kind": "prune", "measure": "S1", "level": "all",
                                       "layer": li, **r, "owl_m": C.OWL_M})
                values["S1"] = s1
                effort["S1"] = {"wall_s": time.perf_counter() - t0, "full_model_calibration_passes": 0,
                                "block_forward_passes": 0, "calibration_sequences_per_pass": n_seq, "restores": 0}
                values["S2"] = isolated("S2", C.PROBE_SPARSITIES,
                                        lambda li, S: {"zeros": sum(prune_block(model, li, S, x_l2).values())})
            elif args.kind == "quant_v2":
                units = run_quant_v2(model, pristine, calib, scales, lo, hi, values, probe_rows, effort, ref,
                                     nll_pass, steps, rec, timings)
            else:
                t0 = time.perf_counter()
                errs = qs2_errors(model, calib, scales, C.PROBE_BITS)
                values["QS2"] = {}
                for b in C.PROBE_BITS:
                    values["QS2"][level_key(b)] = []
                    for li in range(L):
                        e = errs[(li, b)]
                        v = sum(num / den for num, den in e.values())
                        values["QS2"][level_key(b)].append(v)
                        probe_rows.append({"eval_id": C.EVAL_ID, "kind": "quant", "measure": "QS2",
                                           "level": level_key(b), "layer": li, "value": v,
                                           "projections": {n: {"err_num": num, "err_den": den, "rel": num / den}
                                                           for n, (num, den) in e.items()}})
                effort["QS2"] = {"wall_s": time.perf_counter() - t0, "full_model_calibration_passes": 0,
                                 "block_forward_passes": L * len(C.PROBE_BITS), "calibration_sequences_per_pass": n_seq,
                                 "restores": 0}
                def quant_one(li, b):
                    quant_block(model, li, b, {g: scales[(li, g, b)] for g in SCALE_GROUPS})
                    return {"bits": b}

                values["QS1"] = isolated("QS1", C.PROBE_BITS, quant_one)
            steps.pop("progress", None)
            steps["pristine_after"] = {"tensors_verified": pristine.verify(model)}

            write_rows(out / OUTPUT_FILES["probe_rows"], probe_rows)
            write_rows(out / OUTPUT_FILES["probe_calib_nll"], nll_rows)
            sens = {"eval_id": C.EVAL_ID, "kind": args.kind, "unit": C.ALLOC_UNIT, "n_blocks": L,
                    "measures": list(measures), "definitions": {m: DEFINITIONS[m] for m in measures},
                    "values": {m: values[m] for m in measures}, "fp16_calib_nll": ref, "effort": effort,
                    "calib_stats": steps["calib_stats_source"], "model": info, "limits": lim["limits"],
                    "alloc_settings": C.alloc_settings()}
            if args.kind == "quant_v2":
                sens.update(unit=C.Q3V2_UNIT, design=C.Q3V2_DESIGN, block_range=[lo, hi],
                            n_units_total=L * len(C.Q3V2_UNITS),
                            units=[{k: u[k] for k in ("index", "id", "layer", "unit", "scale_group", "projections",
                                                      "groups")} for u in units],
                            q3v2_settings=C.q3v2_settings())
            write_json_new(out / OUTPUT_FILES["sensitivity"], sens)
            steps["sensitivity"] = {"file": OUTPUT_FILES["sensitivity"],
                                    "sha256": sha256_file(out / OUTPUT_FILES["sensitivity"]), "values": sens["values"]}
            steps["effort"] = effort
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                steps["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(dev)
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
