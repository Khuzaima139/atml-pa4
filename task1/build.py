"""Block-by-block construction of one compression condition on restored FP16 weights.

For every block, in order:
  1. (P2, P3, 2:4 only) one forward of the not-yet-pruned block on the pruned-model stream accumulates the
     per-channel sums of squares of all seven projection inputs (sequential Wanda statistics);
  2. the executed weights of the seven projections are built (mask, quantize, or fold then quantize);
  3. one forward of the unchanged block on the dense-model stream measures, for each projection,
     ||(W_eff - W) X||_F^2 and ||W X||_F^2 on the dense calibration inputs (W_eff = executed weight mapped
     back to the original basis) and the dense per-channel statistics; its output is the next dense input;
  4. the executed weights (and, for Q2/Q3, the folded norms and v_proj bias) are written into the block;
  5. (P2, P3, 2:4 only) the modified block is re-run on the pruned stream to produce the next block's inputs.
"""
import time
from dataclasses import dataclass, field

import torch

from . import config as C
from . import prune as P
from .awq import NON_ELIGIBLE_FOLDED, col_row_scales, folded_params, geometry, unfold_weight
from .blocks import ChannelStats, capture_block_inputs, collect_stats, input_hooks, layers, projections, run_block
from .quant import dequantize, pack_mask, pack_quantized, quantize

PRUNE_FAMILIES = ("P1", "P2", "P2_2of4", "P3")
QUANT_FAMILIES = ("Q1", "Q2", "Q3")
SEQUENTIAL = ("P2", "P2_2of4", "P3")
AWQ_FAMILIES = ("Q2", "Q3")
SCALE_GROUPS = ("qkv", "o", "gateup", "down")
BUILDABLE = ("baseline",) + PRUNE_FAMILIES + QUANT_FAMILIES


@dataclass
class BuildResult:
    records: list = field(default_factory=list)        # one dict per eligible projection
    payload: list = field(default_factory=list)        # artifact entries per matrix
    extra_tensors: dict = field(default_factory=dict)  # modified non-eligible tensors (full parameter names)
    timings: dict = field(default_factory=dict)
    score_edges: dict = field(default_factory=dict)


def _qual(li, n):
    return f"model.layers.{li}.{'self_attn' if n in C.ATTN_PROJ else 'mlp'}.{n}"


@torch.inference_mode()
def error_pass(block, stream, W: dict, E: dict):
    """Dense forward of the unchanged block measuring output errors of E against W on its inputs."""
    projs = projections(block)
    D = {n: E[n] - W[n] for n in W}
    acc = {n: [0.0, 0.0] for n in W}
    stats = {n: ChannelStats(projs[n].in_features, projs[n].weight.device) for n in W}

    def fn(n, x):
        stats[n].update(x)
        xf = x.float()
        acc[n][0] += float((xf @ D[n].T).square().sum(dtype=torch.float64))
        acc[n][1] += float((xf @ W[n].T).square().sum(dtype=torch.float64))

    with input_hooks(block, fn):
        out = run_block(block, stream)
    return acc, stats, out


@torch.no_grad()
def quantize_block(block, W: dict, bits: int, block_scales=None, geom=None) -> tuple:
    """Quantize the seven projections of one block at one bit width, plain (block_scales None) or AWQ-folded.

    Returns (executed weights in the model dtype, FP32 effective weights in the original basis, FP32 folded
    non-eligible tensors keyed by block-relative name, {name: task1.quant.Quantized}).
    """
    projs = projections(block)
    col = row = fold = None
    extra = {}
    if block_scales is not None:
        bs = {g: block_scales[g].to(W["q_proj"].device) for g in SCALE_GROUPS}
        col, row = col_row_scales(bs, geom)
        fold = folded_params(block, bs, geom)
        extra = {pn: fold[pn] for pn in NON_ELIGIBLE_FOLDED}
    execw, E, qres = {}, {}, {}
    for n in C.ELIGIBLE:
        src = W[n] if fold is None else fold[("self_attn." if n in C.ATTN_PROJ else "mlp.") + n + ".weight"]
        r = quantize(src, bits)
        execw[n] = dequantize(r.q, r.z, r.scale).to(projs[n].weight.dtype)
        E[n] = execw[n].float() if fold is None else unfold_weight(execw[n], col[n], row.get(n))
        qres[n] = r
    return execw, E, extra, qres


@torch.no_grad()
def apply_block(block, execw: dict, extra: dict):
    """Write executed projection weights and folded non-eligible tensors into a block."""
    projs = projections(block)
    for n, w in execw.items():
        projs[n].weight.copy_(w)
    params = dict(block.named_parameters())
    for pn, v in extra.items():
        params[pn].copy_(v.to(params[pn].dtype))


def _zeros_for(qual, shape, spec, zeros_per_matrix):
    if zeros_per_matrix is not None:
        return int(zeros_per_matrix[qual])
    return P.target_zeros(shape, spec["sparsity"])


@torch.inference_mode()
def build_condition(model, calib_seqs, cond_id: str, batch_size: int = C.CALIB_BATCH_SIZE, awq_scales=None,
                    zeros_per_matrix=None, bits_per_block=None, pristine=None, log=None,
                    with_distributions: bool = True) -> BuildResult:
    """Apply condition cond_id in place to a model holding the restored FP16 weights.

    awq_scales: {(layer, group): FP32 distinct scales} at each block's bit width (Q2, Q3).
    zeros_per_matrix: {qualname: zeros} replacing sparsity * numel (unstructured pruning; required for P3).
    bits_per_block: bit width of every block (required for Q3).
    pristine: task1.model.Pristine, required for pruning score distributions.
    with_distributions: compute pruning score distributions (skipped only for development candidates).
    """
    spec = C.CONDITIONS[cond_id]
    fam = spec["family"]
    if fam not in BUILDABLE:
        raise ValueError(f"condition {cond_id} (family {fam}) is not built here")
    if fam in AWQ_FAMILIES:
        assert awq_scales is not None, f"{fam} requires AWQ scales"
    if fam == "P3":
        assert zeros_per_matrix is not None, "P3 requires per-matrix zero counts"
    if fam == "Q3":
        assert bits_per_block is not None and len(bits_per_block) == len(layers(model)), "Q3 requires bits per block"
        assert set(bits_per_block) <= set(C.Q3_BITS), bits_per_block
    else:
        assert bits_per_block is None, "bits_per_block is only used by Q3"
    if fam in PRUNE_FAMILIES:
        assert pristine is not None, "pruning requires the pristine snapshot for score distributions"
    geom = geometry(model.config)
    res = BuildResult()
    tm = {k: 0.0 for k in ("capture_s", "calibration_stats_s", "construct_s", "error_pass_s", "apply_s", "rerun_s")}
    t = time.perf_counter()
    dense = capture_block_inputs(model, calib_seqs, batch_size)
    tm["capture_s"] = time.perf_counter() - t
    seq = [(h.clone(), kw) for h, kw in dense] if fam in SEQUENTIAL else None
    norms = {"dense": {}, "sequential": {}}

    for li, block in enumerate(layers(model)):
        projs = projections(block)
        W = {n: projs[n].weight.detach().float() for n in C.ELIGIBLE}
        recs = {n: {"eval_id": C.EVAL_ID, "layer": li, "name": n, "qualname": _qual(li, n),
                    "shape": list(projs[n].weight.shape)} for n in C.ELIGIBLE}
        seq_stats = None
        if seq is not None:
            t = time.perf_counter()
            seq_stats, _ = collect_stats(block, seq)
            tm["calibration_stats_s"] += time.perf_counter() - t

        t = time.perf_counter()
        execw, E, extra = {}, {}, {}
        if fam == "baseline":
            E = W
        elif fam in PRUNE_FAMILIES:
            for n in C.ELIGIBLE:
                qual, shape, w = recs[n]["qualname"], tuple(W[n].shape), W[n]
                mag = P.magnitude_score(w)
                if fam == "P1":
                    score = mag
                else:
                    norms["sequential"][qual] = seq_stats[n].l2().cpu()
                    score = P.wanda_score(w, seq_stats[n].l2())
                if fam == "P2_2of4":
                    keep = P.mask_2of4(score)
                    z = shape[0] * shape[1] // 2
                    recs[n]["disagreement_with_magnitude"] = P.disagreement(keep, P.row_mask(mag, z))
                    recs[n]["disagreement_with_unstructured_wanda"] = P.disagreement(keep, P.row_mask(score, z))
                else:
                    z = _zeros_for(qual, shape, spec, zeros_per_matrix)
                    keep = P.row_mask(score, z)
                    if fam in ("P2", "P3"):
                        recs[n]["disagreement_with_magnitude"] = P.disagreement(keep, P.row_mask(mag, z))
                pruned = int((~keep).sum())
                assert pruned == z, (qual, pruned, z)
                execw[n] = projs[n].weight.detach().masked_fill(~keep, 0)
                E[n] = execw[n].float()
                recs[n].update(target_zeros=z, mask_zeros=pruned)
                res.payload.append({"qualname": qual, "kind": "mask", "shape": list(shape),
                                    "tensors": {"mask": pack_mask(keep)}, "mask_kept": int(keep.sum())})
        else:
            b = int(bits_per_block[li]) if fam == "Q3" else spec["bits"]
            bs = {g: awq_scales[(li, g)] for g in SCALE_GROUPS} if fam in AWQ_FAMILIES else None
            execw, E, extra, qres = quantize_block(block, W, b, bs, geom)
            for n in C.ELIGIBLE:
                r = qres[n]
                recs[n].update(bits=b, groups=int(r.scale.numel()), n_guarded_groups=r.n_guarded,
                               n_clipped_codes=r.n_clipped, n_one_sided_groups=r.n_one_sided)
                res.payload.append({"qualname": recs[n]["qualname"], "kind": "quant", "shape": list(W[n].shape),
                                    "bits": b, "tensors": pack_quantized(r)})
        tm["construct_s"] += time.perf_counter() - t

        t = time.perf_counter()
        acc, dstats, dense = error_pass(block, dense, W, E)
        tm["error_pass_s"] += time.perf_counter() - t
        for n in C.ELIGIBLE:
            num, den = acc[n]
            recs[n].update(err_num=num, err_den=den, rel_output_error=num / den)
            if fam in PRUNE_FAMILIES:
                norms["dense"][recs[n]["qualname"]] = dstats[n].l2().cpu()
                recs[n]["x_norm_cv_dense"] = P.coef_variation(dstats[n].l2())
                if seq_stats is not None:
                    recs[n]["x_norm_cv_sequential"] = P.coef_variation(seq_stats[n].l2())

        t = time.perf_counter()
        apply_block(block, execw, extra)
        params = dict(block.named_parameters())
        for pn in extra:
            res.extra_tensors[f"model.layers.{li}.{pn}"] = params[pn].detach().cpu().clone()
        for n in C.ELIGIBLE:
            recs[n]["nonzeros"] = int(torch.count_nonzero(projs[n].weight))
        tm["apply_s"] += time.perf_counter() - t

        if seq is not None:
            t = time.perf_counter()
            seq = run_block(block, seq)
            tm["rerun_s"] += time.perf_counter() - t
        res.records.extend(recs[n] for n in C.ELIGIBLE)
        if log:
            log(li, {k: round(v, 3) for k, v in tm.items()})
        del W, E, execw

    if fam in PRUNE_FAMILIES and with_distributions:
        t = time.perf_counter()
        wanda_norms = norms["dense"] if fam == "P1" else norms["sequential"]
        res.score_edges = score_distributions(res.records, pristine, wanda_norms,
                                              "dense" if fam == "P1" else "sequential",
                                              next(model.parameters()).device)
        tm["distributions_s"] = time.perf_counter() - t
    tm["compression_s"] = tm["construct_s"] + tm["apply_s"] + (
        tm["capture_s"] + tm["calibration_stats_s"] + tm["rerun_s"] if fam in SEQUENTIAL else 0.0)
    tm["measurement_s"] = tm["error_pass_s"] + (0.0 if fam in SEQUENTIAL else tm["capture_s"])
    res.timings = tm
    return res


@torch.no_grad()
def score_distributions(records, pristine, wanda_norms: dict, wanda_source: str, device) -> dict:
    """Magnitude and Wanda score distributions of the original weights, on log10 edges spanning all projections."""
    def scores(rec):
        w = pristine.cpu[rec["qualname"] + ".weight"].to(device)
        mag = P.magnitude_score(w)
        return mag, P.wanda_score(w, wanda_norms[rec["qualname"]].to(device))

    ranges = {"magnitude": [], "wanda": []}
    for rec in records:
        mag, wan = scores(rec)
        ranges["magnitude"].append(P.log10_range(mag))
        ranges["wanda"].append(P.log10_range(wan))
    edges = {k: P.log10_edges(v) for k, v in ranges.items()}
    for rec in records:
        mag, wan = scores(rec)
        rec["dist_magnitude"] = P.distribution(mag, edges["magnitude"])
        rec["dist_wanda"] = P.distribution(wan, edges["wanda"])
    return {"magnitude_log10_edges": edges["magnitude"], "wanda_log10_edges": edges["wanda"],
            "wanda_norm_source": wanda_source, "quantile_levels": list(P.QUANTILES), "n_bins": P.N_BINS}
