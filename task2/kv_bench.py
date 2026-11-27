"""Task 2 Part C runner: C5 packed-cache encode/decode benchmark (protocol AM2.8).

Usage: python -m task2.kv_bench --out DIR [--device cuda:0]
Test-only options (refused for any output under results/task2):
       --limit-prompts N | ID,ID,...   the first N prompt ids, or the listed ids
       --model-path DIR                a local model directory instead of the locked checkpoint
       --limit-reps WARMUP,TIMED       replace the 3 warmup and 10 timed repetitions

Caches. For each context level (2048 and 4096 nominal tokens) the first 8 prompt ids (sorted) are prefilled in FP16
and the context cache is truncated to L = 2048 or L = 4096 tokens. Batch 1 uses the first of them, batch 8 all 8
(stacked on the batch axis). The hybrid's FP16 positions come from each prompt's question-aware scores restricted to
[0, L) with the global ranking of task2.kv_select (8 sink and the last 64 positions of the truncated cache first) and
the group count of task2.kv_budget.hybrid_split at L. Gold fields of the prompts are never read.

Configurations: int4_token, int4_channel_k, hybrid, plus the fp16 reference for the decode step. Each (L, batch)
runs 3 warmup and 10 timed repetition blocks; in every block each configuration runs once, in an order rotated by one
position per block (recorded). Per block and configuration (times in ms, CUDA events around synchronized work):
  select          hybrid only: ranking and FP16/INT2 split of every batch row from the scores
  pack            quantize and bit-pack every layer from the FP16 context cache (hybrid: gather, INT2 groups, FP16
                  positions, int32 index list)
  unpack_dequant  unpack and dequantize every layer to FP16 tensors in position order (hybrid: scatter by position)
  decode_step     one forward of one new token per sequence through all layers: unpack_dequant of every layer
                  written into the session cache, then the model step; fp16: the model step on the FP16 cache
Serialized bytes are the bytes of the stored tensors (payload, metadata, FP16 values, indices) and must equal the
task2.kv_budget formulas times the batch size. Peak allocated and reserved memory are reset before each
configuration run and the maximum over timed blocks is kept. A configuration that runs out of memory (status oom) or
raises NotImplementedError (status unsupported) is recorded with its error text and skipped for the remaining blocks.

Outputs under --out: run.json, bench_rows.jsonl.gz (one row per L, batch and configuration).
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import kv_budget as KB
from . import kv_cache_ops as KO
from . import kv_metrics as KM
from . import kv_run as KR
from . import kv_select as KS
from .kv_quant import GROUP, QuantBlock, relative_error
from .rows import RowWriter

VERSION = KR.VERSION
CONFIGS = ("int4_token", "int4_channel_k", "hybrid")
REFERENCE = "fp16"
ALL_CONFIGS = CONFIGS + (REFERENCE,)
BATCHES = (1, 8)
N_PER_LEVEL = 8
TRUNCATE = {2048: 2048, 4096: 4096}
WARMUP, REPS = 3, 10
PHASES = {"int4_token": ("pack", "unpack_dequant", "decode_step"),
          "int4_channel_k": ("pack", "unpack_dequant", "decode_step"),
          "hybrid": ("select", "pack", "unpack_dequant", "decode_step"),
          "fp16": ("decode_step",)}
STATUSES = ("ok", "oom", "unsupported")
OUTPUT_FILES = {"run": "run.json", "rows": "bench_rows.jsonl"}
ORDER_RULE = "configurations (int4_token, int4_channel_k, hybrid, fp16) rotated left by the block index; warmup " \
             "blocks are numbered first"


# ---------------------------------------------------------------------------------------------------------------------
# Batched packed caches


@dataclass
class BenchPacked:
    config: str
    L: int
    layers: list  # per layer dict with k_q, v_q (QuantBlock), and FP16 tensors k_fp, v_fp or k_tail, v_tail
    indices: torch.Tensor | None  # int32 [B, F] FP16 positions (hybrid)


def complement(fp_idx: torch.Tensor, L: int) -> torch.Tensor:
    """Ascending positions of each row not in fp_idx ([B, F] -> [B, L - F])."""
    B = fp_idx.shape[0]
    mask = torch.ones(B, L, dtype=torch.bool, device=fp_idx.device)
    mask.scatter_(1, fp_idx.long(), False)
    return mask.nonzero(as_tuple=True)[1].view(B, -1)


def _expand(idx, shape):
    B, H, _, d = shape
    return idx[:, None, :, None].expand(B, H, idx.shape[1], d)


def select_hybrid(scores: list, L: int, f: int, device) -> torch.Tensor:
    """FP16 positions [B, f] (ascending per row) from each row's scores over [0, L)."""
    rows = [KS.hybrid_positions(KS.global_ranking(s[:L]), f)[0] for s in scores]
    return torch.stack(rows).to(device)


def pack(config: str, keys: list, values: list, fp_idx: torch.Tensor | None = None) -> BenchPacked:
    L = keys[0].shape[2]
    layers = []
    if config in ("int4_token", "int4_channel_k"):
        layout = "token" if config == "int4_token" else "channel"
        n_q = (L // GROUP) * GROUP
        for k, v in zip(keys, values):
            layers.append({"k_q": QuantBlock.encode(k[:, :, :n_q], layout, 4),
                           "v_q": QuantBlock.encode(v[:, :, :n_q], "token", 4),
                           "k_tail": k[:, :, n_q:].contiguous(), "v_tail": v[:, :, n_q:].contiguous()})
        return BenchPacked(config, L, layers, None)
    if config != "hybrid":
        raise ValueError(f"unknown configuration {config!r}")
    low = complement(fp_idx, L)
    if low.shape[1] % GROUP:
        raise ValueError(f"{low.shape[1]} INT2 positions are not whole {GROUP}-token groups")
    shape = keys[0].shape
    g_fp, g_low = _expand(fp_idx.long(), shape), _expand(low, shape)
    for k, v in zip(keys, values):
        layers.append({"k_q": QuantBlock.encode(k.gather(2, g_low), "channel", 2),
                       "v_q": QuantBlock.encode(v.gather(2, g_low), "token", 2),
                       "k_fp": k.gather(2, g_fp), "v_fp": v.gather(2, g_fp)})
    return BenchPacked(config, L, layers, fp_idx.to(torch.int32))


def unpack(p: BenchPacked) -> list:
    """Per layer (K, V) FP16 [B, H, L, d] in position order."""
    out = []
    if p.indices is None:
        for lay in p.layers:
            out.append((torch.cat([lay["k_q"].decode(), lay["k_tail"]], dim=2),
                        torch.cat([lay["v_q"].decode(), lay["v_tail"]], dim=2)))
        return out
    fp = p.indices.long()
    low = complement(fp, p.L)
    for lay in p.layers:
        shape = (*lay["k_fp"].shape[:2], p.L, lay["k_fp"].shape[3])
        g_fp, g_low = _expand(fp, shape), _expand(low, shape)
        kv = []
        for q, f in ((lay["k_q"], lay["k_fp"]), (lay["v_q"], lay["v_fp"])):
            t = torch.empty(shape, dtype=f.dtype, device=f.device)
            t.scatter_(2, g_low, q.decode(f.dtype))
            t.scatter_(2, g_fp, f)
            kv.append(t)
        out.append(tuple(kv))
    return out


def _tb(t) -> int:
    return 0 if t is None else t.numel() * t.element_size()


def nbytes(p: BenchPacked) -> dict:
    payload = metadata = retained = tail = 0
    for lay in p.layers:
        for q in (lay["k_q"], lay["v_q"]):
            payload += q.payload_bytes()
            metadata += q.metadata_bytes()
        retained += _tb(lay.get("k_fp")) + _tb(lay.get("v_fp"))
        tail += _tb(lay.get("k_tail")) + _tb(lay.get("v_tail"))
    idx = _tb(p.indices)
    return {"payload": payload, "metadata": metadata, "fp16_retained": retained, "fp16_tail": tail, "indices": idx,
            "total": payload + metadata + retained + tail + idx}


def formula_bytes(config: str, shape: KB.KVShape, L: int) -> dict:
    if config == "hybrid":
        m, f = KB.hybrid_split(shape, L)
        return {"bytes": KB.hybrid_bytes(shape, L, m), "m": m, "f": f}
    if config == REFERENCE:
        return {"bytes": KB.m16(shape, L), "m": None, "f": None}
    return {"bytes": KB.m4(shape, L), "m": None, "f": None}


# ---------------------------------------------------------------------------------------------------------------------
# Timing


def timed(fn, device):
    """Run fn once; returns (result, milliseconds). CUDA: events around the call after a synchronize."""
    if KR.is_cuda(device):
        torch.cuda.synchronize(device)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        r = fn()
        e.record()
        torch.cuda.synchronize(device)
        return r, float(s.elapsed_time(e))
    t0 = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t0) * 1e3


def guarded(fn):
    """(status, error text, result): ok, oom (CUDA out of memory) or unsupported (NotImplementedError)."""
    try:
        return "ok", None, fn()
    except torch.cuda.OutOfMemoryError as exc:
        return "oom", f"{type(exc).__name__}: {exc}", None
    except NotImplementedError as exc:
        return "unsupported", f"{type(exc).__name__}: {exc}", None


def stats(vals: list) -> dict | None:
    if not vals:
        return None
    return {"median": statistics.median(vals), "mean": statistics.fmean(vals), "min": min(vals), "max": max(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}


def rotate(seq, k):
    k %= len(seq)
    return list(seq[k:] + seq[:k])


def _cleanup(device):
    gc.collect()
    if KR.is_cuda(device):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------------
# Caches from real prompts


@torch.no_grad()
def level_caches(model, tokenizer, cases: list, L: int, device) -> list:
    """Context caches truncated to L, with each prompt's question-aware scores and the token at position L."""
    out = []
    for case in cases:
        inputs, _ = KR.prepare_prompt(tokenizer, case)  # gold fields discarded
        if inputs.context_len < L:
            raise RuntimeError(f"{inputs.id}: context {inputs.context_len} shorter than L={L}")
        ids = inputs.ids_tensor(device)
        prefill = KO.full_prefill(model, ids, inputs.context_len)
        KR.assert_fp16_cache(prefill.keys, prefill.values)
        scores, _ = KO.score_prefill(model, prefill)
        out.append({"id": inputs.id, "context_len": inputs.context_len,
                    "keys": [k[:, :, :L].contiguous() for k in prefill.keys],
                    "values": [v[:, :, :L].contiguous() for v in prefill.values],
                    "scores": scores[:L].float().cpu(), "next_token": int(inputs.input_ids[L])})
        del prefill
    return out


def _fresh_cache(model, keys, values):
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    for i, (k, v) in enumerate(zip(keys, values)):
        cache.update(k.clone(), v.clone(), i)
    return cache


@torch.no_grad()
def model_step(model, cache, tok, L):
    B = tok.shape[0]
    pos = torch.full((B, 1), L, dtype=torch.long, device=tok.device)
    return model(input_ids=tok, past_key_values=cache, position_ids=pos,
                 cache_position=torch.tensor([L], device=tok.device), use_cache=True, logits_to_keep=1).logits


@torch.no_grad()
def bench_shape(model, shape, sel: list, L: int, device, warmup: int, reps: int) -> tuple[list, list]:
    """Every configuration at one (L, batch). Returns (rows, block orders)."""
    B = len(sel)
    keys = [torch.cat([c["keys"][i] for c in sel], dim=0) for i in range(shape.n_layers)]
    values = [torch.cat([c["values"][i] for c in sel], dim=0) for i in range(shape.n_layers)]
    KR.assert_fp16_cache(keys, values)
    tok = torch.tensor([[c["next_token"]] for c in sel], dtype=torch.long, device=device)
    scores = [c["scores"] for c in sel]
    state = {c: {"status": "ok", "error": None, "times": {p: [] for p in PHASES[c]}, "peak_allocated": None,
                 "peak_reserved": None, "bytes": None, "recon": None} for c in ALL_CONFIGS}
    hyb = formula_bytes("hybrid", shape, L)
    caches = {}
    orders = []
    for block in range(warmup + reps):
        order = rotate(ALL_CONFIGS, block)
        is_timed = block >= warmup
        orders.append({"block": block, "kind": "timed" if is_timed else "warmup", "order": order})
        for cfg in order:
            st = state[cfg]
            if st["status"] != "ok":
                continue

            def run(cfg=cfg, st=st):
                KR.reset_peak(device)
                t = {}
                if cfg == REFERENCE:
                    if "fp16" not in caches:
                        caches["fp16"] = _fresh_cache(model, keys, values)
                    _, t["decode_step"] = timed(lambda: model_step(model, caches["fp16"], tok, L), device)
                    caches["fp16"].crop(L)
                    if st["bytes"] is None:
                        b = sum(_tb(k) + _tb(v) for k, v in zip(keys, values))
                        st["bytes"] = {"payload": 0, "metadata": 0, "fp16_retained": b, "fp16_tail": 0,
                                       "indices": 0, "total": b}
                else:
                    fp_idx = None
                    if cfg == "hybrid":
                        fp_idx, t["select"] = timed(lambda: select_hybrid(scores, L, hyb["f"], device), device)
                    packed, t["pack"] = timed(lambda: pack(cfg, keys, values, fp_idx), device)
                    kv, t["unpack_dequant"] = timed(lambda: unpack(packed), device)
                    if st["bytes"] is None:
                        st["bytes"] = nbytes(packed)
                        ek = [relative_error(k, kh) for k, (kh, _) in zip(keys, kv)]
                        ev = [relative_error(v, vh) for v, (_, vh) in zip(values, kv)]
                        st["recon"] = {"k_mean": statistics.fmean(ek), "v_mean": statistics.fmean(ev)}
                    if "quant" not in caches:
                        caches["quant"] = _fresh_cache(model, [k for k, _ in kv], [v for _, v in kv])
                    del kv
                    cache = caches["quant"]

                    def step():
                        for layer, (k, v) in zip(cache.layers, unpack(packed)):
                            layer.keys, layer.values = k, v
                        return model_step(model, cache, tok, L)

                    _, t["decode_step"] = timed(step, device)
                    del packed
                pa, pr = KR.peak_allocated(device), KR.peak_reserved(device)
                return t, pa, pr

            status, err, res = guarded(run)
            if status != "ok":
                st.update(status=status, error=err)
                caches.pop("quant" if cfg != REFERENCE else "fp16", None)
                _cleanup(device)
                continue
            t, pa, pr = res
            if is_timed:
                for p, ms in t.items():
                    st["times"][p].append(ms)
                if pa is not None:
                    st["peak_allocated"] = max(pa, st["peak_allocated"] or 0)
                    st["peak_reserved"] = max(pr, st["peak_reserved"] or 0)
    rows = []
    ids = [c["id"] for c in sel]
    m16 = KB.m16(shape, L)
    for cfg in ALL_CONFIGS:
        st = state[cfg]
        fb = formula_bytes(cfg, shape, L)
        b = st["bytes"]
        if b is not None and b["total"] != B * fb["bytes"]:
            raise RuntimeError(f"{cfg} L={L} B={B}: counted bytes {b['total']} != {B} x formula {fb['bytes']}")
        rows.append({
            "version": VERSION, "L": L, "batch": B, "config": cfg, "status": st["status"], "error": st["error"],
            "cache_ids": ids, "repeated_caches": len(set(ids)) < len(ids), "n_warmup": warmup, "n_reps": reps,
            "times_ms": st["times"], "stats_ms": {p: stats(v) for p, v in st["times"].items()},
            "bytes": b, "bytes_per_sequence": None if b is None else {k: v // B for k, v in b.items()},
            "formula_bytes_per_sequence": fb["bytes"], "fp16_bytes_per_sequence": m16,
            "realized_fraction": fb["bytes"] / m16, "hybrid_m": fb["m"], "hybrid_f": fb["f"],
            "recon_rel_error": st["recon"], "peak_allocated_bytes": st["peak_allocated"],
            "peak_reserved_bytes": st["peak_reserved"],
        })
    del caches, keys, values
    _cleanup(device)
    return rows, orders


# ---------------------------------------------------------------------------------------------------------------------
# Entry point


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-prompts", default=None, help="N or ID,ID,... (tests only)")
    ap.add_argument("--model-path", default=None, help="local model directory (tests only)")
    ap.add_argument("--limit-reps", default=None, help="WARMUP,TIMED (tests only)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    limits = {"prompts": args.limit_prompts, "model_path": args.model_path, "reps": args.limit_reps}
    limited = KR.check_out(args.out, limits)
    warmup, reps = WARMUP, REPS
    if args.limit_reps is not None:
        warmup, reps = (int(x) for x in args.limit_reps.split(","))
        if warmup < 0 or reps < 1:
            raise SystemExit("--limit-reps needs WARMUP >= 0 and TIMED >= 1")
    out = Path(args.out)
    device = torch.device(args.device)
    if KR.is_cuda(device):
        torch.cuda.set_device(device)
    seed_everything(KR.SEED)
    cases = KR.select_cases(KR.load_cases(), args.limit_prompts)
    per_level = {lv: [c for c in cases if int(c["target_tokens"]) == lv][:N_PER_LEVEL] for lv in KR.LEVELS}
    config = {
        "version": VERSION, "scorer": KM.SCORER, "device": str(device),
        "model_id": KR.MODEL_ID if args.model_path is None else None,
        "model_revision": KR.locked_revision() if args.model_path is None else None, "model_path": args.model_path,
        "dtype": "float16", "attn_implementation": KR.ATTN_IMPLEMENTATION, "limits": limits, "limited": limited,
        "output_files": OUTPUT_FILES, "data": KR.data_record(), "configs": list(CONFIGS), "reference": REFERENCE,
        "batches": list(BATCHES), "truncate": {str(k): v for k, v in TRUNCATE.items()}, "warmup": warmup,
        "repetitions": reps, "phases": {k: list(v) for k, v in PHASES.items()}, "order_rule": ORDER_RULE,
        "prompt_ids_per_level": {str(k): [c["id"] for c in v] for k, v in per_level.items()},
        "statuses": list(STATUSES), "seed": KR.SEED,
        "timing": "CUDA events around each synchronized call (perf_counter on CPU), milliseconds",
        "generation": {"settings": None, "note": "one model step per repetition on a fixed input token; no token "
                       "is generated or sampled"},
    }
    out.mkdir(parents=True)
    rec = RunRecorder(out / OUTPUT_FILES["run"], config)
    orders, n_rows = {}, 0
    try:
        rec.update(environment=KR.environment(device), step="loading")
        tokenizer = KR.load_tokenizer(args.model_path)
        model, info = KR.load_model(device, args.model_path)
        model_facts = KR.check_model(model)
        shape = KB.KVShape.from_config(model.config)
        rec.update(model=info, model_facts=model_facts, kv_shape=vars(shape) | {"channels": shape.channels},
                   tokenizer=KR.tokenizer_meta(tokenizer, args.model_path), step="loaded")
        plain = out / OUTPUT_FILES["rows"]
        with RowWriter(plain) as writer:
            for level in KR.LEVELS:
                lv_cases = per_level[level]
                if not lv_cases:
                    continue
                L = TRUNCATE[level]
                caches = level_caches(model, tokenizer, lv_cases, L, device)
                for B in BATCHES:
                    sel = [caches[i % len(caches)] for i in range(B)]
                    rows, blk = bench_shape(model, shape, sel, L, device, warmup, reps)
                    orders[f"L{L}_B{B}"] = blk
                    for r in rows:
                        writer.write(r)
                    n_rows += len(rows)
                    rec.update(step=f"L{L}_B{B}", block_orders=orders, n_rows=n_rows)
                del caches
                _cleanup(device)
        row_files = {"rows": KR.finalize_rows(plain)}
        rec.finish("ok", block_orders=orders, n_rows=n_rows, row_files=row_files, step="done")
    except BaseException as exc:
        rec.finish("failed", block_orders=orders, n_rows=n_rows, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())
        raise
    print(f"bench done: {out} rows {n_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
