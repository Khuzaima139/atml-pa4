"""Latency and peak-memory measurement of dense execution (batch 1).

CUDA: CUDA events with torch.cuda.synchronize around every timed region and peak-memory statistics reset
before every repetition. CPU: the same code timed with time.perf_counter (used by tests).
"""
import statistics
import time

import torch

from . import config as C

EXECUTION = {
    "mode": "dense_fp16",
    "description": "dense PyTorch nn.Linear matmuls on the stored FP16 weights (zeros and reconstructed "
                   "quantized values included); no sparse or low-bit kernels",
}


def prompt_ids(calib_sequences, lengths=C.PREFILL_LENGTHS) -> dict:
    """Fixed prompts: the first calibration sequences concatenated, truncated to each length."""
    out = {}
    for L in lengths:
        ids, rows = [], []
        for s in calib_sequences:
            if len(ids) >= L:
                break
            ids.extend(s["token_ids"])
            rows.append(s["source_row"])
        assert len(ids) >= L, (L, len(ids))
        out[L] = {"token_ids": ids[:L], "source_rows": rows, "split": calib_sequences[0]["split"]}
    return out


class _Timer:
    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.device = device

    def reset_peak(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def peaks(self):
        if not self.cuda:
            return None, None
        return torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)

    def time(self, fn):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            start.record()
            res = fn()
            end.record()
            torch.cuda.synchronize(self.device)
            return start.elapsed_time(end), res
        t0 = time.perf_counter()
        res = fn()
        return (time.perf_counter() - t0) * 1000.0, res


def _prefill(model, ids):
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    return out


def _decode(model, past, first_token, steps):
    tok = first_token
    toks = []
    for _ in range(steps):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)  # greedy, EOS ignored
        toks.append(int(tok))
    return toks


def _measure(timer, fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times, alloc, reserved = [], [], []
    for _ in range(reps):
        timer.reset_peak()
        ms, res = timer.time(fn)
        a, r = timer.peaks()
        del res  # outputs (KV cache, logits) are not kept across repetitions
        times.append(ms)
        alloc.append(a)
        reserved.append(r)
    entry = {"unit": "ms", "repetitions": reps, "warmup": warmup, "times_ms": times,
             "median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times),
             "peak_allocated_bytes": alloc, "peak_reserved_bytes": reserved,
             "peak_allocated_bytes_max": (max(alloc) if alloc[0] is not None else None),
             "peak_reserved_bytes_max": (max(reserved) if reserved[0] is not None else None)}
    return entry


@torch.inference_mode()
def latency(model, prompts: dict, warmup: int = C.LATENCY_WARMUP, reps: int = C.LATENCY_REPS,
            decode_steps: int = C.DECODE_STEPS) -> dict:
    """Prefill latency for each prompt length and a greedy decode after a 512-token prefill."""
    assert reps >= 5
    dev = next(model.parameters()).device
    timer = _Timer(dev)
    out = {"execution": EXECUTION, "device": str(dev), "timer": "cuda_events" if timer.cuda else "perf_counter",
           "batch_size": 1, "eval_id": C.EVAL_ID,
           "device_name": torch.cuda.get_device_name(dev) if timer.cuda else "cpu",
           "attn_implementation": model.config._attn_implementation,
           "prompts": {str(L): {"source_rows": p["source_rows"], "split": p["split"], "length": L}
                       for L, p in prompts.items()},
           "measurements": {}}
    for L, p in prompts.items():
        ids = torch.tensor([p["token_ids"]], dtype=torch.long, device=dev)
        entry = _measure(timer, lambda: _prefill(model, ids), warmup, reps)
        entry["what"] = f"prefill of {L} tokens with KV cache creation, logits for the last position only"
        out["measurements"][f"prefill_{L}"] = entry

    L0 = C.PREFILL_LENGTHS[0]
    ids = torch.tensor([prompts[L0]["token_ids"]], dtype=torch.long, device=dev)

    def decode_once():
        pre = _prefill(model, ids)
        first = pre.logits[:, -1].argmax(dim=-1, keepdim=True)
        return pre.past_key_values, first

    def run():
        past, first = state["past"], state["first"]
        return _decode(model, past, first, decode_steps)

    # Each repetition decodes from a fresh, untimed prefill of the same prompt.
    times, alloc, reserved, tokens = [], [], [], []
    for i in range(warmup + reps):
        state = {}
        state["past"], state["first"] = decode_once()
        timer.reset_peak()
        ms, toks = timer.time(run)
        a, r = timer.peaks()
        del state
        if i >= warmup:
            times.append(ms)
            alloc.append(a)
            reserved.append(r)
            tokens.append(toks)
    assert all(t == tokens[0] for t in tokens), "greedy decode not repeatable"
    out["measurements"][f"decode_{decode_steps}_after_prefill_{L0}"] = {
        "what": f"{decode_steps} greedy decode steps with the KV cache after an untimed {L0}-token prefill, "
                "own loop, EOS ignored",
        "unit": "ms", "repetitions": reps, "warmup": warmup, "times_ms": times,
        "median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times),
        "median_ms_per_token": statistics.median(times) / decode_steps,
        "peak_allocated_bytes": alloc, "peak_reserved_bytes": reserved,
        "peak_allocated_bytes_max": (max(alloc) if alloc[0] is not None else None),
        "peak_reserved_bytes_max": (max(reserved) if reserved[0] is not None else None),
        "decoded_token_ids": tokens[0]}
    return out
