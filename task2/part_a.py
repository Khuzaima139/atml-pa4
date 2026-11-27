"""Task 2 Part A: accuracy verification and benchmark of dense, tiled and SDPA causal attention.

Usage: python -m task2.part_a verify --out DIR [--device cuda:0]
       python -m task2.part_a bench  --out DIR [--device cuda:0]
Test-only options (refused for any output under results/task2):
       --limit-lengths L1,L2,...    replace the sequence lengths
       --limit-reps WARMUP,TIMED    replace the 3 warmup and 10 timed repetitions (bench)
       --limit-projection-gib X     replace the 14 GiB dense projection limit

verify: for every L, dtype and implementation, output and gradient (dq, dk, dv) errors against a dense FP32 reference,
plus edge cases at L = 256 (causality under perturbation of future keys and values, extreme scores, a fully masked
row). Reference per row dtype: FP32 rows use dense attention in FP32 on the FP32 inputs; FP16 rows use dense attention
in FP32 on the FP16 inputs cast back to FP32 (and the FP16 upstream gradient cast to FP32), so FP16 input rounding is
not counted as attention error. FP16 rows also carry the comparison against the FP32-input reference in columns
prefixed vs_fp32_inputs_. bench: latency and peak memory for forward-only and forward-plus-backward passes, and a
backend table from profiler operator names and forced-backend support.

Outputs under --out: run.json, verify_rows.jsonl (verify) or bench_rows.jsonl and backend_table.jsonl (bench), and
score_bytes.json (theoretical score-matrix bytes per (L, dtype)).
"""
import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from common.io import write_json_new
from common.paths import RESULTS_DIR
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import attention_ops as ops
from .rows import RowWriter

VERSION = "task2-part-a-v2"
REAL_ROOT = RESULTS_DIR / "task2"

B, H, D = 2, 4, 64
SEED = 6304
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
ELEM_BYTES = {"fp16": 2, "fp32": 4}
VERIFY_TILES = ((32, 32), (64, 64), (128, 128), (64, 128))
BENCH_TILE = (128, 128)
WARMUP, REPS = 3, 10
PROJECTION_LIMIT_GIB = 14.0
EDGE_L = 256
EXTREME_TARGETS = (1e3, 1e4)
PASSES = ("forward", "forward_backward")
STATUSES = ("ok", "skipped_projected", "oom", "unsupported", "no_reference")
REFERENCE_INPUTS = {"fp32": "fp32_inputs", "fp16": "fp16_inputs_cast_to_fp32"}
SECONDARY_PREFIX = "vs_fp32_inputs_"
CAST_NAMES = ("q", "k", "v", "g")
RECORDED_ENV = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")

OUTPUT_FILES = {
    "verify": {"run": "run.json", "rows": "verify_rows.jsonl", "score_bytes": "score_bytes.json"},
    "bench": {"run": "run.json", "rows": "bench_rows.jsonl", "backend_table": "backend_table.jsonl",
              "score_bytes": "score_bytes.json"},
}


# ---------------------------------------------------------------------------------------------------------------
# Implementations

def spec_dense():
    return {"impl": "dense", "label": "dense", "tile": None, "backend": None}


def spec_tiled(tile):
    return {"impl": "tiled", "label": f"tiled_{tile[0]}x{tile[1]}", "tile": list(tile), "backend": None}


def spec_sdpa(backend=None):
    name = None if backend is None else backend.name
    return {"impl": "sdpa", "label": "sdpa_default" if backend is None else f"sdpa_{name}", "tile": None,
            "backend": name}


def verify_specs():
    return [spec_dense()] + [spec_tiled(t) for t in VERIFY_TILES] + [spec_sdpa()] + \
        [spec_sdpa(b) for b in ops.sdpa_backends()]


def bench_specs():
    return [spec_dense(), spec_tiled(BENCH_TILE), spec_sdpa()] + [spec_sdpa(b) for b in ops.sdpa_backends()]


def run_spec(spec, q, k, v, mask=None):
    if spec["impl"] == "dense":
        return ops.dense_causal(q, k, v, mask=mask)
    if spec["impl"] == "tiled":
        return ops.tiled_causal(q, k, v, spec["tile"][0], spec["tile"][1], mask=mask)
    backend = None
    if spec["backend"] is not None:
        from torch.nn.attention import SDPBackend
        backend = SDPBackend.__members__[spec["backend"]]
    return ops.sdpa_causal(q, k, v, backend=backend, mask=mask)


def spec_fields(spec):
    return {"impl": spec["impl"], "label": spec["label"], "tile": spec["tile"], "backend_forced": spec["backend"]}


# ---------------------------------------------------------------------------------------------------------------
# Inputs, memory and errors

def make_inputs(L: int) -> dict:
    """FP32 q, k, v and the upstream gradient g, drawn in that order on the CPU from one generator seeded 6304;
    perturbation tensors pk, pv (used only by the causality check) are drawn after them from the same generator."""
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    shape = (B, H, L, D)
    names = ("q", "k", "v", "g", "pk", "pv")
    return {n: torch.randn(shape, generator=gen, dtype=torch.float32) for n in names}


def reference_inputs(inp: dict, dtype_name: str) -> dict:
    """Inputs of the dense FP32 reference for rows of one dtype: the FP32 inputs themselves for FP32 rows; for FP16 rows,
    q, k, v and the upstream gradient g rounded to FP16 and cast back to FP32 (pk, pv unchanged)."""
    if dtype_name == "fp32":
        return inp
    dtype = DTYPES[dtype_name]
    return {n: (t.to(dtype).to(torch.float32) if n in CAST_NAMES else t) for n, t in inp.items()}


def cast_round(t, dtype_name: str):
    """A tensor rounded to the row dtype and cast back to FP32 (identity for FP32)."""
    return t if dtype_name == "fp32" else t.to(DTYPES[dtype_name]).to(torch.float32)


def is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def mem_reset(device):
    """Free cached blocks and reset peak statistics; return currently allocated bytes (None on CPU)."""
    gc.collect()
    if not is_cuda(device):
        return None
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def mem_peak(device) -> dict:
    if not is_cuda(device):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    torch.cuda.synchronize(device)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def cleanup(device):
    gc.collect()
    if is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def leaf(t, device, dtype, grad: bool):
    """A new leaf tensor (always a copy, so the shared CPU inputs are never modified)."""
    return t.detach().to(device=device, dtype=dtype, copy=True).requires_grad_(grad)


def fwd_bwd(spec, inp, dtype, device, mask=None, q_override=None):
    """Forward and autograd backward with the fixed upstream gradient; returns (output, (dq, dk, dv)), detached."""
    q0 = inp["q"] if q_override is None else q_override
    q, k, v = (leaf(t, device, dtype, True) for t in (q0, inp["k"], inp["v"]))
    out = run_spec(spec, q, k, v, mask=mask)
    grads = torch.autograd.grad(out, (q, k, v), inp["g"].to(device=device, dtype=dtype))
    return out.detach(), tuple(gr.detach() for gr in grads)


def error_fields(out, grads, ref) -> dict:
    """Max and mean absolute error (FP32) of the output and each gradient against the reference."""
    r_out, r_grads = ref
    row = {}
    for name, a, b in (("out", out, r_out),) + tuple(zip(("dq", "dk", "dv"), grads, r_grads)):
        diff = (a.float() - b.float()).abs()
        row[f"{name}_max_abs_err"] = float(diff.max())
        row[f"{name}_mean_abs_err"] = float(diff.mean())
        row[f"{name}_finite"] = bool(torch.isfinite(a).all())
    return row


def secondary_error_fields(out, grads, ref) -> dict:
    """Max and mean absolute errors against the FP32-input reference, prefixed vs_fp32_inputs_ (FP16 rows only)."""
    full = error_fields(out, grads, ref)
    return {SECONDARY_PREFIX + k: v for k, v in full.items() if not k.endswith("_finite")}


def all_error_fields(out, grads, refs: dict, dtype_name: str) -> dict:
    """Primary errors against the reference of the row dtype; FP16 rows add the FP32-input comparison."""
    row = {"reference_inputs": REFERENCE_INPUTS[dtype_name],
           **error_fields(out, grads, refs[dtype_name])}
    if dtype_name != "fp32":
        row.update(secondary_error_fields(out, grads, refs["fp32"]))
    return row


def projected(L, dtype_name):
    return ops.dense_projected_peak_bytes(B, H, L, D, ELEM_BYTES[dtype_name])


def score_bytes_table(lengths, limit_gib) -> list:
    rows = []
    for L in lengths:
        for name in DTYPES:
            p = projected(L, name)
            rows.append({
                "L": L, "dtype": name, "B": B, "H": H, "d": D,
                "score_matrix_bytes": ops.score_matrix_bytes(B, H, L, ELEM_BYTES[name]),
                "causal_mask_bytes": L * L,
                "dense_projected_peak_bytes": p, "dense_projected_peak_gib": ops.gib(p),
                "dense_attempted": ops.gib(p) < limit_gib,
                "tiled_block_bytes_fp32_128x128": B * H * BENCH_TILE[0] * BENCH_TILE[1] * 4,
                "projection_formula": "4 * B*H*L*L*e + L*L + 9 * B*H*L*d*e (e = element bytes)",
            })
    return rows


# ---------------------------------------------------------------------------------------------------------------
# verify

def _attempt(spec, dtype_name, L, limit_gib, fn, device):
    """Run fn() under the projection, out-of-memory and forced-backend rules; return (status, result, extra)."""
    if spec["impl"] == "dense":
        p = projected(L, dtype_name)
        if ops.gib(p) >= limit_gib:
            return "skipped_projected", None, {"error": f"skipped: projected {ops.gib(p):.2f} GiB"}
    try:
        return "ok", fn(), {"error": None}
    except torch.cuda.OutOfMemoryError as exc:
        failure = ("oom", str(exc))
    except RuntimeError as exc:
        if not (spec["impl"] == "sdpa" and spec["backend"] is not None):
            raise
        failure = ("unsupported", str(exc))
    cleanup(device)  # after the except block, so the traceback no longer holds the failed call's tensors
    return failure[0], None, {"error": failure[1]}


def verify_accuracy(writer, lengths, device, limit_gib, counts):
    for L in lengths:
        inp = make_inputs(L)
        refs, ref_status = {}, {}
        for ref_dtype in ("fp32", "fp16"):
            r_inp = reference_inputs(inp, ref_dtype)
            mem_reset(device)
            status, ref, extra = _attempt(spec_dense(), "fp32", L, limit_gib,
                                          lambda: fwd_bwd(spec_dense(), r_inp, torch.float32, device), device)
            writer.write({"version": VERSION, "kind": "reference", "L": L, "dtype": "fp32",
                          "for_rows_dtype": ref_dtype, "reference_inputs": REFERENCE_INPUTS[ref_dtype],
                          "status": status, "dense_projected_peak_bytes": projected(L, "fp32"), **mem_peak(device),
                          **extra})
            counts[status] = counts.get(status, 0) + 1
            refs[ref_dtype], ref_status[ref_dtype] = ref, status
            del r_inp, ref
        for dtype_name, dtype in DTYPES.items():
            needed = ("fp32",) if dtype_name == "fp32" else (dtype_name, "fp32")
            missing = [n for n in needed if refs[n] is None]
            for spec in verify_specs():
                row = {"version": VERSION, "kind": "accuracy", "L": L, "dtype": dtype_name, **spec_fields(spec)}
                if missing:
                    row.update(status="no_reference", reference_inputs=REFERENCE_INPUTS[dtype_name],
                               error="; ".join(f"dense FP32 reference on {REFERENCE_INPUTS[n]} {ref_status[n]}"
                                               for n in missing))
                else:
                    mem_reset(device)
                    t0 = time.perf_counter()
                    st, res, ex = _attempt(spec, dtype_name, L, limit_gib,
                                           lambda: fwd_bwd(spec, inp, dtype, device), device)
                    row.update(status=st, wall_s=time.perf_counter() - t0, **mem_peak(device), **ex)
                    row["reference_inputs"] = REFERENCE_INPUTS[dtype_name]
                    if res is not None:
                        row.update(all_error_fields(res[0], res[1], refs, dtype_name))
                    del res
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                writer.write(row)
        del refs, inp
        cleanup(device)


def edge_ts(L):
    return sorted({t for t in (0, 37, L // 2 - 1, L - 2) if 0 <= t < L - 1})


def verify_causality(writer, L, device, counts):
    """Perturb keys and values after position t; outputs at positions <= t must be bitwise unchanged."""
    inp = make_inputs(L)
    specs = [spec_dense()] + [spec_tiled(t) for t in VERIFY_TILES] + [spec_sdpa()]
    for dtype_name, dtype in DTYPES.items():
        q, k, v = (inp[n].to(device=device, dtype=dtype) for n in ("q", "k", "v"))
        pk, pv = (inp[n].to(device=device, dtype=dtype) for n in ("pk", "pv"))
        for spec in specs:
            with torch.no_grad():
                base = run_spec(spec, q, k, v)
                for t in edge_ts(L):
                    k2, v2 = k.clone(), v.clone()
                    k2[:, :, t + 1:] = pk[:, :, t + 1:]
                    v2[:, :, t + 1:] = pv[:, :, t + 1:]
                    pert = run_spec(spec, q, k2, v2)
                    past = (pert[:, :, :t + 1].float() - base[:, :, :t + 1].float()).abs()
                    future = (pert[:, :, t + 1:].float() - base[:, :, t + 1:].float()).abs()
                    row = {"version": VERSION, "kind": "edge_causality", "L": L, "dtype": dtype_name,
                           **spec_fields(spec), "t": t, "status": "ok",
                           "bitwise_equal_upto_t": bool(torch.equal(pert[:, :, :t + 1], base[:, :, :t + 1])),
                           "max_abs_change_upto_t": float(past.max()),
                           "max_abs_change_after_t": float(future.max()),
                           "required_bitwise": spec["impl"] in ("dense", "tiled")}
                    counts["ok"] = counts.get("ok", 0) + 1
                    writer.write(row)


def verify_extreme(writer, L, device, counts):
    """Scale q so the largest causal score magnitude reaches each target; record finiteness and errors."""
    inp = make_inputs(L)
    with torch.no_grad():
        s = torch.matmul(inp["q"] * (D ** -0.5), inp["k"].transpose(-2, -1))
        s = s.masked_fill(~ops.causal_mask(L, L, "cpu"), 0.0)
        s_max = float(s.abs().max())
    for target in EXTREME_TARGETS:
        factor = target / s_max
        q_ext = inp["q"] * factor
        refs = {d: fwd_bwd(spec_dense(), reference_inputs(inp, d), torch.float32, device,
                           q_override=cast_round(q_ext, d)) for d in DTYPES}
        for dtype_name, dtype in DTYPES.items():
            q_cast = q_ext.to(dtype)
            with torch.no_grad():
                s_dt = torch.matmul((q_cast.float() * (D ** -0.5)), inp["k"].to(dtype).float().transpose(-2, -1))
                s_dt = s_dt.masked_fill(~ops.causal_mask(L, L, "cpu"), 0.0)
            for spec in verify_specs():
                row = {"version": VERSION, "kind": "edge_extreme", "L": L, "dtype": dtype_name, **spec_fields(spec),
                       "target_max_abs_score": target, "q_scale_factor": factor,
                       "achieved_max_abs_score": float(s_dt.abs().max()),
                       "q_cast_finite": bool(torch.isfinite(q_cast).all())}
                st, res, ex = _attempt(spec, dtype_name, L, float("inf"),
                                       lambda: fwd_bwd(spec, inp, dtype, device, q_override=q_ext), device)
                row.update(status=st, reference_inputs=REFERENCE_INPUTS[dtype_name], **ex)
                if res is not None:
                    row.update(all_error_fields(res[0], res[1], refs, dtype_name))
                counts[st] = counts.get(st, 0) + 1
                writer.write(row)


def verify_masked_row(writer, L, device, counts):
    """Extra mask removing every key of one query row. Dense and tiled return output 0 and gradient 0 for that row;
    SDPA's behaviour is recorded as observed (including an error, if raised)."""
    inp = make_inputs(L)
    r = L // 2
    mask = torch.ones(L, L, dtype=torch.bool)
    mask[r] = False
    mask = mask.to(device)
    refs = {d: fwd_bwd(spec_dense(), reference_inputs(inp, d), torch.float32, device, mask=mask) for d in DTYPES}
    keep = torch.ones(L, dtype=torch.bool, device=device)
    keep[r] = False
    for dtype_name, dtype in DTYPES.items():
        for spec in verify_specs():
            row = {"version": VERSION, "kind": "edge_masked_row", "L": L, "dtype": dtype_name, **spec_fields(spec),
                   "masked_row": r, "reference_inputs": REFERENCE_INPUTS[dtype_name]}
            try:
                out, grads = fwd_bwd(spec, inp, dtype, device, mask=mask)
                dq = grads[0]
                row_out = out[:, :, r].float()
                others = (out[:, :, keep].float() - refs[dtype_name][0][:, :, keep]).abs()
                documented = bool((row_out == 0).all() and (dq[:, :, r] == 0).all()
                                  and all(torch.isfinite(t).all() for t in (out, *grads)))
                row.update(status="ok", error=None,
                           row_out_max_abs=float(row_out.abs().nan_to_num(float("inf")).max()),
                           row_out_nan_count=int(torch.isnan(row_out).sum()),
                           row_dq_max_abs=float(dq[:, :, r].float().abs().nan_to_num(float("inf")).max()),
                           row_dq_nan_count=int(torch.isnan(dq[:, :, r]).sum()),
                           out_all_finite=bool(torch.isfinite(out).all()),
                           grads_all_finite=bool(all(torch.isfinite(gr).all() for gr in grads)),
                           other_rows_max_abs_err=float(others.max()),
                           matches_documented_zero_row=documented)
                if dtype_name != "fp32":
                    others32 = (out[:, :, keep].float() - refs["fp32"][0][:, :, keep]).abs()
                    row[SECONDARY_PREFIX + "other_rows_max_abs_err"] = float(others32.max())
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                if spec["impl"] != "sdpa":
                    raise
                row.update(status="unsupported", error=str(exc))
            if row["status"] != "ok":
                cleanup(device)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            writer.write(row)


# ---------------------------------------------------------------------------------------------------------------
# bench

def time_fn(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    cuda = is_cuda(device)
    for _ in range(reps):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def time_stats(times):
    s = sorted(times)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"times_ms": times, "median_ms": med, "mean_ms": sum(s) / n, "min_ms": s[0], "max_ms": s[-1]}


def bench_condition(spec, inp, dtype_name, pass_name, L, device, warmup, reps, limit_gib):
    dtype = DTYPES[dtype_name]
    grad = pass_name == "forward_backward"
    q, k, v = (leaf(inp[n], device, dtype, grad) for n in ("q", "k", "v"))
    g = inp["g"].to(device=device, dtype=dtype)

    if grad:
        def fn():
            out = run_spec(spec, q, k, v)
            torch.autograd.grad(out, (q, k, v), g)
    else:
        def fn():
            run_spec(spec, q, k, v)

    base = mem_reset(device)
    st, times, ex = _attempt(spec, dtype_name, L, limit_gib, lambda: time_fn(fn, device, warmup, reps), device)
    row = {"status": st, "baseline_allocated_bytes": base, **mem_peak(device), **ex,
           "timer": "cuda_event" if is_cuda(device) else "perf_counter", "warmup": warmup, "repetitions": reps,
           "unit": "ms"}
    if times is not None:
        row.update(time_stats(times))
    del q, k, v, g, fn
    cleanup(device)
    return row


def backend_rows(inp, L, device):
    rows = []
    for dtype_name, dtype in DTYPES.items():
        q, k, v, g = (inp[n].to(device=device, dtype=dtype) for n in ("q", "k", "v", "g"))
        for pass_name in PASSES:
            back = pass_name == "forward_backward"
            row = {"version": VERSION, "L": L, "dtype": dtype_name, "pass": pass_name}
            try:
                row.update(default=ops.profile_sdpa(q, k, v, backward=back, grad=g), default_error=None)
            except torch.cuda.OutOfMemoryError as exc:
                row.update(default=None, default_error=str(exc))
            cleanup(device)
            try:
                row.update(forced=ops.forced_backend_support(q, k, v, backward=back), forced_error=None)
            except torch.cuda.OutOfMemoryError as exc:
                row.update(forced=None, forced_error=str(exc))
            cleanup(device)
            rows.append(row)
        del q, k, v, g
        cleanup(device)
    return rows


def run_bench(writer, table_writer, lengths, device, warmup, reps, limit_gib, counts, orders):
    base = bench_specs()
    for li, L in enumerate(lengths):
        rot = li % len(base)
        order = base[rot:] + base[:rot]
        orders[str(L)] = [s["label"] for s in order]
        inp = make_inputs(L)
        for pos, spec in enumerate(order):
            for dtype_name in DTYPES:
                for pass_name in PASSES:
                    row = {"version": VERSION, "kind": "bench", "L": L, "dtype": dtype_name, "pass": pass_name,
                           **spec_fields(spec), "order_position": pos,
                           **bench_condition(spec, inp, dtype_name, pass_name, L, device, warmup, reps, limit_gib)}
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                    writer.write(row)
        for row in backend_rows(inp, L, device):
            table_writer.write(row)
        del inp
        cleanup(device)


# ---------------------------------------------------------------------------------------------------------------
# Entry point

def under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("verify", "bench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-lengths", default=None, help="comma-separated lengths (tests only)")
    ap.add_argument("--limit-reps", default=None, help="WARMUP,TIMED (tests only)")
    ap.add_argument("--limit-projection-gib", type=float, default=None, help="dense projection limit (tests only)")
    return ap.parse_args(argv)


def check_args(args) -> dict:
    """Refuse an existing output directory, and any --limit-* option with an output under results/task2."""
    limits = {"lengths": args.limit_lengths, "reps": args.limit_reps, "projection_gib": args.limit_projection_gib}
    limited = any(v is not None for v in limits.values())
    if limited and under(args.out, REAL_ROOT):
        raise SystemExit(f"refusing --limit-* with an output path under {REAL_ROOT}: {args.out}")
    if Path(args.out).exists():
        raise SystemExit(f"output directory exists: {args.out}")
    lengths = LENGTHS
    if args.limit_lengths is not None:
        lengths = tuple(int(x) for x in args.limit_lengths.split(","))
        if not lengths or min(lengths) < 2:
            raise SystemExit("lengths must be at least 2")
    warmup, reps = WARMUP, REPS
    if args.limit_reps is not None:
        warmup, reps = (int(x) for x in args.limit_reps.split(","))
        if warmup < 0 or reps < 1:
            raise SystemExit("--limit-reps needs WARMUP >= 0 and TIMED >= 1")
    limit_gib = PROJECTION_LIMIT_GIB if args.limit_projection_gib is None else args.limit_projection_gib
    return {"limits": limits, "limited": limited, "lengths": lengths, "warmup": warmup, "reps": reps,
            "projection_limit_gib": limit_gib}


def environment(device) -> dict:
    env = {"device": str(device), "env": {k: os.environ.get(k) for k in RECORDED_ENV},
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "matmul_allow_fp16_reduced_precision_reduction":
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           "sdpa_backends_enumerated": [b.name for b in ops.sdpa_backends()],
           "cpu_threads": torch.get_num_threads()}
    for flag in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(torch.backends.cuda, flag, None)
        env[flag] = fn() if fn is not None else None
    if is_cuda(device):
        dev = torch.device(device)
        env["gpu_name"] = torch.cuda.get_device_name(dev)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(dev))
        env["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(dev).total_memory
    else:
        env["gpu_name"] = None
        env["gpu_capability"] = None
        env["gpu_total_memory_bytes"] = None
    return env


def main(argv=None):
    args = parse_args(argv)
    chk = check_args(args)
    out = Path(args.out)
    device = torch.device(args.device)
    if is_cuda(device):
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(SEED)
    files = OUTPUT_FILES[args.mode]
    lengths = chk["lengths"]
    edge_L = EDGE_L if EDGE_L in lengths else max(lengths)
    config = {
        "version": VERSION, "mode": args.mode, "device": str(device), "B": B, "H": H, "d": D, "seed": SEED,
        "lengths": list(lengths), "dtypes": list(DTYPES), "limits": chk["limits"], "limited": chk["limited"],
        "projection_limit_gib": chk["projection_limit_gib"], "output_files": files,
        "reference": "dense_causal in FP32 with the fixed upstream gradient; FP32 rows: on the FP32 inputs; FP16 rows: "
                     "on q, k, v and the upstream gradient rounded to FP16 and cast to FP32. FP16 rows also carry "
                     "errors against the FP32-input reference in columns prefixed vs_fp32_inputs_",
        "reference_inputs": REFERENCE_INPUTS, "secondary_error_prefix": SECONDARY_PREFIX,
        "score_scaling": ops.SCORE_SCALE_NOTE,
        "input_generation": "torch.randn on CPU from one generator seeded 6304 per L, order q, k, v, g, pk, pv; "
                            "FP16 inputs cast from the FP32 draws",
    }
    if args.mode == "verify":
        config.update(impls=[s["label"] for s in verify_specs()], verify_tiles=[list(t) for t in VERIFY_TILES],
                      edge_L=edge_L, edge_causality_t=edge_ts(edge_L), extreme_targets=list(EXTREME_TARGETS),
                      masked_row=edge_L // 2)
    else:
        config.update(impls=[s["label"] for s in bench_specs()], bench_tile=list(BENCH_TILE), passes=list(PASSES),
                      warmup=chk["warmup"], repetitions=chk["reps"],
                      forward_pass_note="forward-only inputs do not require grad, so no autograd graph is built",
                      order_rule="implementation order rotated by one position per length index")
    out.mkdir(parents=True)
    rec = RunRecorder(out / files["run"], config)
    write_json_new(out / files["score_bytes"], {"version": VERSION,
                                                "rows": score_bytes_table(lengths, chk["projection_limit_gib"])})
    counts, orders = {}, {}
    rec.update(environment=environment(device), status_counts=counts, step="started")
    try:
        with RowWriter(out / files["rows"]) as writer:
            if args.mode == "verify":
                verify_accuracy(writer, lengths, device, chk["projection_limit_gib"], counts)
                rec.update(status_counts=counts, step="accuracy_done")
                verify_causality(writer, edge_L, device, counts)
                rec.update(status_counts=counts, step="causality_done")
                verify_extreme(writer, edge_L, device, counts)
                rec.update(status_counts=counts, step="extreme_done")
                verify_masked_row(writer, edge_L, device, counts)
                n_rows = writer.n_rows
            else:
                with RowWriter(out / files["backend_table"]) as table:
                    run_bench(writer, table, lengths, device, chk["warmup"], chk["reps"],
                              chk["projection_limit_gib"], counts, orders)
                n_rows = writer.n_rows
        rec.finish("ok", status_counts=counts, implementation_orders=orders, n_rows=n_rows, step="done")
    except BaseException as exc:
        rec.finish("failed", status_counts=counts, implementation_orders=orders,
                   error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    print(f"{args.mode} done: {out} status counts {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
