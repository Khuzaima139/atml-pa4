"""Task 2 Part C post hoc diagnostic of the int4_token result (protocol AM4).

Usage: python -m task2.kv_diag --out DIR [--device cuda:0]
Test-only options (refused for any output under results/task2):
       --limit-prompts N | ID,ID,...   the first N of the 18 diagnostic prompts, or the listed ids (any case id)
       --model-path DIR                a local model directory instead of the locked checkpoint

Prompts (AM4.2): the lowest-id prompt of each of the 18 grid cells (task2.kv_run.first_per_cell).

Per prompt: one full FP16 prefill (task2.kv_cache_ops.full_prefill); channel statistics of the FP16 pre-question
context cache (AM4.3); then six cache configurations in an order rotated by one position per prompt, each through the
eval pipeline (compression, suffix re-run with absolute position_ids and physical cache_position, greedy decoding,
teacher-forced pass on the gold code from the post-suffix cache state). Configurations (AM4.4):
  fp16            reference (task2.kv_cache_ops.run_method, unchanged)
  k_int4_token    K INT4 token-wise, V FP16 (new configuration, DiagPackedKVCache)
  v_int4_token    K FP16, V INT4 token-wise (new configuration)
  kv_int8_token   K and V INT8 token-wise (new configuration)
  int4_token      as evaluated (task2.kv_cache_ops.run_method, unchanged)
  int4_channel_k  as evaluated (task2.kv_cache_ops.run_method, unchanged)
The new configurations use the AM2.1 quantizer (task2.kv_quant.quantize_groups) on the AM2.2 groups with the AM2.2
trailing rule; INT8 codes are stored one per byte without packing. The gold code is used only after every
configuration of the prompt has finished decoding.

Outputs under --out: run.json, control_rows.jsonl.gz (one row per prompt and configuration), prompt_rows.jsonl.gz
(one row per prompt), npz/<id>.npz (per layer, KV head and channel statistics, numeric arrays only). Row files are
written as plain JSONL while running and compressed when the run finishes.
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import kv_budget as KB
from . import kv_cache_ops as KO
from . import kv_eval as EV
from . import kv_metrics as KM
from . import kv_run as KR
from .kv_quant import GROUP, PackedKVCache, PackedLayer, QuantBlock, quantize_groups
from .rows import RowWriter

VERSION = "task2-kv-diag-v1"
NPZ_FORMAT_VERSION = 1
TF_DEVICE = EV.TF_DEVICE
TOP_FRACTION = 0.01
REFERENCE = "fp16"
CONTROLS = ("k_int4_token", "v_int4_token", "kv_int8_token", "int4_token", "int4_channel_k")
CONFIGS = (REFERENCE,) + CONTROLS
EVALUATED = ("fp16", "int4_token", "int4_channel_k")  # run through task2.kv_cache_ops.run_method unchanged
# (layout, bits) per tensor for the new configurations; None keeps the tensor FP16 at every position
SPECS = {"k_int4_token": {"k_spec": ("token", 4), "v_spec": None},
         "v_int4_token": {"k_spec": None, "v_spec": ("token", 4)},
         "kv_int8_token": {"k_spec": ("token", 8), "v_spec": ("token", 8)}}
QUANT = {"fp16": ("fp16", "fp16"), "k_int4_token": ("int4_token", "fp16"), "v_int4_token": ("fp16", "int4_token"),
         "kv_int8_token": ("int8_token", "int8_token"), "int4_token": ("int4_token", "int4_token"),
         "int4_channel_k": ("int4_channel", "int4_token")}
LABELS = {"fp16": "fp16 reference", "k_int4_token": "(a) K INT4 token-wise, V FP16",
          "v_int4_token": "(b) K FP16, V INT4 token-wise", "kv_int8_token": "(c) K and V INT8 token-wise",
          "int4_token": "(d) int4_token as evaluated", "int4_channel_k": "(e) int4_channel_k as evaluated"}
OUTPUT_FILES = {"run": "run.json", "rows": "control_rows.jsonl", "prompts": "prompt_rows.jsonl", "npz_dir": "npz"}
CHANNEL_ARRAYS = ("k_abs_max", "k_abs_median", "v_abs_max", "v_abs_median", "k_norm", "v_norm", "k_err_token",
                  "k_err_channel", "v_err_token")
HEAD_ARRAYS = ("k_ratio", "v_ratio", "k_err_token_top", "k_err_token_rest", "k_err_channel_top", "k_err_channel_rest",
               "v_err_token_top", "v_err_token_rest")
MASK_ARRAYS = ("k_top", "v_top")
SCALAR_ARRAYS = ("context_len", "n_top", "format_version", "version_ascii")
NPZ_KEYS = CHANNEL_ARRAYS + HEAD_ARRAYS + MASK_ARRAYS + SCALAR_ARRAYS
GROUP_KEYS = HEAD_ARRAYS[2:]
ORDER_RULE = "task2.kv_diag.CONFIGS rotated left by (prompt_index mod 6); prompt_index = rank of the id"
PREDICTIONS = {
    "c_int8": "kv_int8_token exact match within 5 points of fp16 and mean teacher-forced KL below 0.05",
    "a_k_only": "k_int4_token exact match at most 0.1",
    "b_v_only": "v_int4_token exact match within 5 points of fp16",
    "k_ratio": "mean K channel max/median ratio at least 3 times the mean V ratio",
    "k_rest_error": "token-wise K error on the non-outlier channels at least 3 times channel-K's",
    "rival_token_path_defect": "kv_int8_token prediction does not hold",
}


# ---------------------------------------------------------------------------------------------------------------------
# Blocks and the control cache


def group_view(x: torch.Tensor, layout: str) -> torch.Tensor:
    """The AM2.2 groups of x [B, H, T, d] with the group on the last axis: token layout [B, H, T, d/64, 64] (64
    consecutive head-dimension elements per token), channel layout [B, H, T/64, d, 64] (64 consecutive tokens per
    channel). Same arrangement as task2.kv_quant.QuantBlock.encode."""
    B, H, T, d = x.shape
    if layout == "token":
        if d % GROUP:
            raise ValueError(f"head_dim {d} is not a multiple of {GROUP}")
        return x.reshape(B, H, T, d // GROUP, GROUP)
    if layout == "channel":
        if T % GROUP:
            raise ValueError(f"channel layout needs whole 64-token groups, got T={T}")
        return x.reshape(B, H, T // GROUP, GROUP, d).transpose(-1, -2)
    raise ValueError(f"unknown layout {layout!r}")


class Int8Block(QuantBlock):
    """QuantBlock with 8-bit codes stored one per byte (payload = codes, no packing). Decoding, extension and byte
    counts are those of QuantBlock (unpacking 8-bit codes is the identity)."""

    @classmethod
    def encode(cls, x: torch.Tensor, layout: str, bits: int = 8) -> "Int8Block":
        if bits != 8:
            raise ValueError(f"Int8Block stores 8-bit codes, got bits={bits}")
        codes, scale, offset = quantize_groups(group_view(x, layout), 8)
        return cls(layout, 8, codes, scale, offset)


@dataclass
class Fp16Block:
    """The FP16 values of the tensor a control leaves unquantized, held where the quantized block would be."""

    values: torch.Tensor  # [B, H, T, d]
    layout: str = "fp16"
    bits: int = 16

    @classmethod
    def encode(cls, x: torch.Tensor) -> "Fp16Block":
        return cls(x.contiguous())

    @property
    def n_tokens(self) -> int:
        return self.values.shape[2]

    def decode(self, dtype=torch.float16) -> torch.Tensor:
        return self.values.to(dtype)

    def extend(self, other: "Fp16Block") -> None:
        self.values = torch.cat([self.values, other.values], dim=2)

    def payload_bytes(self) -> int:
        return 0

    def metadata_bytes(self) -> int:
        return 0

    def fp16_bytes(self) -> int:
        return self.values.numel() * self.values.element_size()


def encode_block(x: torch.Tensor, spec):
    """spec None: Fp16Block; (layout, 8): Int8Block; (layout, 2 or 4): QuantBlock."""
    if spec is None:
        return Fp16Block.encode(x)
    layout, bits = spec
    if bits == 8:
        return Int8Block.encode(x, layout, 8)
    return QuantBlock.encode(x, layout, bits)


def spec_name(spec) -> str:
    return "fp16" if spec is None else f"int{spec[1]}_{spec[0]}"


@dataclass
class DiagPackedKVCache(PackedKVCache):
    """Packed cache of the new configurations: every complete 64-token prefix group of K and V held in the block of
    its spec (Fp16Block, Int8Block or QuantBlock), the trailing r context tokens and appended entries in the FP16
    tail, and a complete 64-token tail group moved into the blocks after the forward pass that completed it."""

    k_spec: tuple | None = None
    v_spec: tuple | None = None

    @classmethod
    def control(cls, method: str, keys: list, values: list, k_spec, v_spec) -> "DiagPackedKVCache":
        L = keys[0].shape[2]
        n_q = (L // GROUP) * GROUP
        layers = [PackedLayer(k_q=encode_block(k[:, :, :n_q], k_spec), v_q=encode_block(v[:, :, :n_q], v_spec),
                              k_tail=k[:, :, n_q:].contiguous(), v_tail=v[:, :, n_q:].contiguous())
                  for k, v in zip(keys, values)]
        ar = torch.arange(L, device=keys[0].device)
        obj = cls(method=method, context_len=L, layers=layers, q_positions=ar[:n_q], fp_positions=ar[:0],
                  tail_positions=ar[n_q:], indices=None, k_layout=None, bits=None, quantize_tail=True,
                  k_spec=k_spec, v_spec=v_spec)
        return cls._finish(obj)

    def nbytes(self) -> dict:
        payload = metadata = retained = tail = 0
        for lay in self.layers:
            for qb in (lay.k_q, lay.v_q):
                if isinstance(qb, Fp16Block):
                    retained += qb.fp16_bytes()
                elif qb is not None:
                    payload += qb.payload_bytes()
                    metadata += qb.metadata_bytes()
            tail += lay.k_tail.numel() * lay.k_tail.element_size() + lay.v_tail.numel() * lay.v_tail.element_size()
        return {"payload": payload, "metadata": metadata, "fp16_retained": retained, "fp16_tail": tail, "indices": 0,
                "total": payload + metadata + retained + tail}

    def append(self, new_k: list, new_v: list, new_positions: torch.Tensor) -> int:
        new_positions = new_positions.to(self.tail_positions.device)
        self.tail_positions = torch.cat([self.tail_positions, new_positions])
        n_groups = self.tail_positions.numel() // GROUP
        for lay, k, v in zip(self.layers, new_k, new_v):
            lay.k_tail = torch.cat([lay.k_tail, k.to(lay.k_tail.dtype)], dim=2)
            lay.v_tail = torch.cat([lay.v_tail, v.to(lay.v_tail.dtype)], dim=2)
            if n_groups:
                n = n_groups * GROUP
                lay.k_q.extend(encode_block(lay.k_tail[:, :, :n], self.k_spec))
                lay.v_q.extend(encode_block(lay.v_tail[:, :, :n], self.v_spec))
                lay.k_tail = lay.k_tail[:, :, n:].contiguous()
                lay.v_tail = lay.v_tail[:, :, n:].contiguous()
        if n_groups:
            n = n_groups * GROUP
            self.q_positions = torch.cat([self.q_positions, self.tail_positions[:n]])
            self.tail_positions = self.tail_positions[n:]
        return n_groups

    def clone(self, device=None) -> "DiagPackedKVCache":
        if device is None:
            return copy.deepcopy(self)

        def mv(t):
            return None if t is None else t.to(device, copy=True)

        def mvb(b):
            if b is None:
                return None
            if isinstance(b, Fp16Block):
                return Fp16Block(mv(b.values))
            return type(b)(b.layout, b.bits, mv(b.payload), mv(b.scale), mv(b.offset))

        layers = [PackedLayer(mvb(lay.k_q), mvb(lay.v_q), mv(lay.k_fp), mv(lay.v_fp), mv(lay.k_tail), mv(lay.v_tail))
                  for lay in self.layers]
        return DiagPackedKVCache(
            method=self.method, context_len=self.context_len, layers=layers, q_positions=mv(self.q_positions),
            fp_positions=mv(self.fp_positions), tail_positions=mv(self.tail_positions), indices=mv(self.indices),
            k_layout=self.k_layout, bits=self.bits, quantize_tail=self.quantize_tail,
            context_bytes=dict(self.context_bytes), k_spec=self.k_spec, v_spec=self.v_spec)


# ---------------------------------------------------------------------------------------------------------------------
# One configuration on one prompt


@torch.no_grad()
def run_control(model, prefill: KO.Prefill, input_ids: torch.Tensor, method: str,
                max_new_tokens: int = KO.MAX_NEW_TOKENS, eos_ids=KO.EOS_IDS, tf_device=None):
    """A new configuration through the steps of task2.kv_cache_ops.run_method: compression of the context, suffix
    re-run, a post-suffix session copy (on tf_device) for the teacher-forced pass, greedy decoding.
    Returns (MethodRun, session copy)."""
    if method not in SPECS:
        raise ValueError(f"unknown control {method!r}")
    dev = input_ids.device
    L = prefill.context_len
    KO._reset_peak(dev)
    start_alloc = int(torch.cuda.memory_allocated(dev)) if torch.device(dev).type == "cuda" else None
    keys = [k[:, :, :L] for k in prefill.keys]
    values = [v[:, :, :L] for v in prefill.values]
    with KO._Timer(dev) as t_c:
        packed = DiagPackedKVCache.control(method, keys, values, **SPECS[method])
    recon = packed.reconstruction_errors(keys, values)
    context_bytes = dict(packed.context_bytes)
    with KO._Timer(dev) as t_s:
        session = KO.CacheSession.start(model, packed, keys[0].dtype)
        first = session.forward(input_ids[:, L:], logits_to_keep=1)[-1]
    tf_session = session.clone(tf_device)
    with KO._Timer(dev) as t_d:
        gen = KO.greedy_decode(session, first, max_new_tokens, eos_ids)
    info = {"method": method, "L": L, "k_spec": spec_name(SPECS[method]["k_spec"]),
            "v_spec": spec_name(SPECS[method]["v_spec"]), "fp16_positions": None}
    run = KO.MethodRun(
        method=method, info=info, first_logits=first, tokens=gen["tokens"], stop=gen["stop"],
        context_bytes=context_bytes, live_bytes=packed.nbytes(), n_appended=packed.n_appended, recon=recon,
        seconds={"compression": t_c.seconds, "suffix": t_s.seconds, "decode": t_d.seconds},
        peak_bytes=KO._peak(dev), tail_rebuilds=session.rebuilds, start_allocated_bytes=start_alloc,
        final_positions=packed.positions().cpu(), physical_len=session.physical_len, next_pos=session.next_pos)
    return run, tf_session


def run_config(model, shape, prefill, input_ids, method: str, tf_device=TF_DEVICE):
    """Evaluated configurations through task2.kv_cache_ops.run_method unchanged; new ones through run_control."""
    if method in EVALUATED:
        run, tf_session, _ = KO.run_method(model, shape, prefill, input_ids, method, None, tf_device=tf_device)
        return run, tf_session
    return run_control(model, prefill, input_ids, method, tf_device=tf_device)


def config_order(prompt_index: int) -> list:
    k = prompt_index % len(CONFIGS)
    return list(CONFIGS[k:] + CONFIGS[:k])


# ---------------------------------------------------------------------------------------------------------------------
# Channel statistics (AM4.3)


def n_top(head_dim: int) -> int:
    return math.ceil(TOP_FRACTION * head_dim)


def _rel(err: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-channel ||err||_2 / ||ref||_2 over tokens (dim 1 of [H, L, d]); 0 for a zero channel reconstructed
    exactly, inf for a zero channel reconstructed with an error."""
    num, den = err.norm(dim=1), ref.norm(dim=1)
    out = num / torch.where(den > 0, den, torch.ones_like(den))
    return torch.where(den > 0, out, torch.where(num > 0, torch.full_like(num, math.inf), torch.zeros_like(num)))


def _top_mask(chmax: torch.Tensor, k: int) -> torch.Tensor:
    """[H, d] mask of the k largest channels per head; ties go to the lower channel index."""
    idx = torch.argsort(chmax, dim=-1, descending=True, stable=True)[:, :k]
    mask = torch.zeros_like(chmax, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _group_means(err: torch.Tensor, mask: torch.Tensor):
    top = torch.stack([err[h][mask[h]].mean() for h in range(err.shape[0])])
    rest = torch.stack([err[h][~mask[h]].mean() for h in range(err.shape[0])])
    return top, rest


@torch.no_grad()
def channel_stats(keys: list, values: list) -> dict:
    """AM4.3 statistics of the context cache (per layer [1, H_kv, L, d] FP16) as numpy arrays: per channel
    [n_layers, H_kv, d], per head [n_layers, H_kv], masks uint8 [n_layers, H_kv, d]. The token-wise and channel-K
    reconstructions are the evaluated packed caches (task2.kv_quant.PackedKVCache.uniform)."""
    tok = PackedKVCache.uniform("int4_token", keys, values, "token", 4)
    chan = PackedKVCache.uniform("int4_channel_k", keys, values, "channel", 4)
    k_top_n = n_top(keys[0].shape[-1])
    out = {k: [] for k in CHANNEL_ARRAYS + HEAD_ARRAYS + MASK_ARRAYS}
    for i, (k16, v16) in enumerate(zip(keys, values)):
        K, V = k16[0].float(), v16[0].float()  # [H, L, d]
        kt, vt = (x[0].float() for x in tok.layer_tensors(i, torch.float32))
        kc = chan.layer_tensors(i, torch.float32)[0][0].float()
        ak, av = K.abs(), V.abs()
        kmax, vmax = ak.amax(1), av.amax(1)
        kmed = torch.quantile(ak, 0.5, dim=1)
        vmed = torch.quantile(av, 0.5, dim=1)
        e_kt, e_kc, e_vt = _rel(kt - K, K), _rel(kc - K, K), _rel(vt - V, V)
        km, vm = _top_mask(kmax, k_top_n), _top_mask(vmax, k_top_n)
        vals = {"k_abs_max": kmax, "k_abs_median": kmed, "v_abs_max": vmax, "v_abs_median": vmed,
                "k_norm": K.norm(dim=1), "v_norm": V.norm(dim=1), "k_err_token": e_kt, "k_err_channel": e_kc,
                "v_err_token": e_vt,
                "k_ratio": kmax.amax(-1) / torch.quantile(kmax, 0.5, dim=-1),
                "v_ratio": vmax.amax(-1) / torch.quantile(vmax, 0.5, dim=-1)}
        vals["k_err_token_top"], vals["k_err_token_rest"] = _group_means(e_kt, km)
        vals["k_err_channel_top"], vals["k_err_channel_rest"] = _group_means(e_kc, km)
        vals["v_err_token_top"], vals["v_err_token_rest"] = _group_means(e_vt, vm)
        for name, t in vals.items():
            out[name].append(t.cpu().numpy().astype(np.float32))
        out["k_top"].append(km.cpu().numpy().astype(np.uint8))
        out["v_top"].append(vm.cpu().numpy().astype(np.uint8))
    res = {k: np.stack(v) for k, v in out.items()}
    res["context_len"] = np.array(keys[0].shape[2], dtype=np.int32)
    res["n_top"] = np.array(k_top_n, dtype=np.int32)
    res["format_version"] = np.array(NPZ_FORMAT_VERSION, dtype=np.int32)
    res["version_ascii"] = np.frombuffer(VERSION.encode("ascii"), dtype=np.uint8).copy()
    return res


def prompt_channel_summary(st: dict) -> dict:
    """Means over layers and KV heads of one prompt's per-head arrays, and the count of zero-norm channels."""
    out = {f"{k}_mean": float(np.mean(st[k])) for k in HEAD_ARRAYS}
    out["n_zero_norm_channels"] = int((st["k_norm"] == 0).sum() + (st["v_norm"] == 0).sum())
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Stages


def generation_stage(model, shape, inputs: KR.PromptInputs, device, order) -> dict:
    """Prefill, channel statistics and every configuration's compression, suffix re-run and decoding. Receives no
    gold field."""
    ids = inputs.ids_tensor(device)
    prefill = KO.full_prefill(model, ids, inputs.context_len)
    KR.assert_fp16_cache(prefill.keys, prefill.values)
    L = inputs.context_len
    with KO._Timer(device) as t_st:
        stats = channel_stats([k[:, :, :L] for k in prefill.keys], [v[:, :, :L] for v in prefill.values])
    runs, sessions = {}, {}
    for m in order:
        runs[m], sessions[m] = run_config(model, shape, prefill, ids, m)
    return {"prefill": prefill, "stats": stats, "stats_seconds": t_st.seconds, "runs": runs, "sessions": sessions}


def control_row(inputs, gold, idx, order, m, run, tf, ref_run, ref_tf, tokenizer) -> dict:
    text = EV.decode_text(tokenizer, run.tokens)
    em = KM.exact_match(text, gold.code)
    nll = KM.nll_rows(tf["logits"], gold.code_ids)
    kl_tf = KM.kl_rows(ref_tf["logits"], tf["logits"])
    kl_post = KM.kl_rows(ref_run.first_logits[None], run.first_logits[None])[0]
    recon = run.recon
    return {
        "version": VERSION, "scorer": KM.SCORER, "id": inputs.id, "prompt_index": idx, "config": m,
        "label": LABELS[m], "k_quant": QUANT[m][0], "v_quant": QUANT[m][1], "evaluated_code_path": m in EVALUATED,
        "config_order": order, "config_slot": order.index(m), **inputs.cell(), "context_len": inputs.context_len,
        "suffix_len": inputs.suffix_len, "generated_token_ids": run.tokens, "generated_text": text,
        "n_generated": len(run.tokens), "stop": run.stop, "hit_max_new_tokens": run.stop == "max_new_tokens",
        "prediction": em["prediction"], "correct": em["correct"], "no_match": em["no_match"],
        "answer_nll_mean": float(nll.mean()), "answer_nll_per_token": [float(x) for x in nll],
        "kl_post_prefill": float(kl_post), "kl_teacher_forced_mean": float(kl_tf.mean()),
        "kl_teacher_forced_per_position": [float(x) for x in kl_tf],
        "k_recon_mean": None if recon is None else recon["k_mean"],
        "v_recon_mean": None if recon is None else recon["v_mean"],
        "k_recon_per_layer": None if recon is None else recon["k_per_layer"],
        "v_recon_per_layer": None if recon is None else recon["v_per_layer"],
        "context_bytes": dict(run.context_bytes), "n_appended": run.n_appended,
        "live_bytes_end": dict(run.live_bytes), "positions_preserved": EV.positions_preserved(run, inputs.context_len),
        "tail_rebuilds": run.tail_rebuilds, "seconds": {**run.seconds, "teacher_forced": tf["seconds"]},
        "peak_allocated_bytes": run.peak_bytes, "start_allocated_bytes": run.start_allocated_bytes,
    }


def select_prompts(cases: list, spec) -> list:
    """The 18 diagnostic prompts (AM4.2); test subsets: the first N of them, or listed ids (any case id)."""
    diag = set(KR.first_per_cell(cases))
    diag_cases = [c for c in cases if c["id"] in diag]
    if spec is not None and not spec.strip().isdigit():
        return KR.select_cases(cases, spec)
    return KR.select_cases(diag_cases, spec)


def run_diag(model, tokenizer, shape, cases, device, out, files, writers, rec, timing) -> dict:
    rows_all, prompt_summ = [], []
    for idx, case in enumerate(cases):
        t_prompt = time.perf_counter()
        inputs, gold = KR.prepare_prompt(tokenizer, case)
        order = config_order(idx)
        gen = generation_stage(model, shape, inputs, device, order)
        # gold fields enter here, after every configuration has finished decoding
        tf = EV.teacher_forced_stage(gen, gold, device, order)
        runs = gen["runs"]
        rows = {m: control_row(inputs, gold, idx, order, m, runs[m], tf[m], runs[REFERENCE], tf[REFERENCE],
                               tokenizer) for m in order}
        for m in order:
            writers["rows"].write(rows[m])
            rows_all.append(rows[m])
        npz_name = f"{files['npz_dir']}/{inputs.id}.npz"
        KR.write_npz_new(out / npz_name, gen["stats"])
        cs = prompt_channel_summary(gen["stats"])
        prompt_summ.append(gen["stats"])
        pre = gen["prefill"]
        writers["prompts"].write({
            "version": VERSION, "scorer": KM.SCORER, "id": inputs.id, "prompt_index": idx, **inputs.cell(),
            "code": gold.code, "context_len": inputs.context_len, "suffix_len": inputs.suffix_len,
            "total_len": inputs.total_len, "config_order": order, "prefill_seconds": pre.seconds,
            "prefill_peak_allocated_bytes": pre.peak_bytes, "channel_stats_seconds": gen["stats_seconds"],
            "fp16_correct": rows[REFERENCE]["correct"], "npz": npz_name, "channel_summary": cs,
        })
        del gen, tf, runs, pre
        timing["prompt_s"][inputs.id] = time.perf_counter() - t_prompt
        rec.update(step="diag", prompts_done=idx + 1, timing=timing)
    return summarize(rows_all, prompt_summ)


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def summarize(rows: list, stats: list) -> dict:
    """Per configuration the eval aggregates; per layer and overall the channel statistics (means over prompts and KV
    heads, then over layers); the AM4.5 read-out (descriptive)."""
    per = {}
    for m in CONFIGS:
        rs = [r for r in rows if r["config"] == m]
        if not rs:
            continue
        per[m] = {"label": LABELS[m], "n": len(rs), "n_correct": sum(r["correct"] for r in rs),
                  "accuracy": sum(r["correct"] for r in rs) / len(rs),
                  "n_no_match": sum(r["no_match"] for r in rs),
                  "n_hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in rs),
                  "mean_answer_nll": _mean(r["answer_nll_mean"] for r in rs),
                  "mean_kl_post_prefill": _mean(r["kl_post_prefill"] for r in rs),
                  "mean_kl_teacher_forced": _mean(r["kl_teacher_forced_mean"] for r in rs),
                  "mean_k_recon": _mean(r["k_recon_mean"] for r in rs if r["k_recon_mean"] is not None),
                  "mean_v_recon": _mean(r["v_recon_mean"] for r in rs if r["v_recon_mean"] is not None),
                  "n_positions_preserved": sum(r["positions_preserved"] for r in rs)}
    channel = None
    if stats:
        per_layer = {k: np.mean(np.stack([s[k] for s in stats]), axis=(0, 2)).tolist() for k in HEAD_ARRAYS}
        overall = {k: float(np.mean(v)) for k, v in per_layer.items()}
        channel = {"per_layer": per_layer, "overall": overall, "n_prompts": len(stats),
                   "n_top_channels": int(stats[0]["n_top"]),
                   "n_zero_norm_channels": int(sum((s["k_norm"] == 0).sum() + (s["v_norm"] == 0).sum()
                                                   for s in stats))}
    return {"n_rows": len(rows), "per_config": per, "channel": channel, "readout": readout(per, channel)}


def readout(per: dict, channel) -> dict:
    """The AM4.5 predictions evaluated on this run (descriptive; nothing is selected from it)."""
    def acc(m):
        return (per.get(m) or {}).get("accuracy")

    fp = acc(REFERENCE)
    out = {"definitions": PREDICTIONS}
    if fp is None:
        return out
    c, a, b = acc("kv_int8_token"), acc("k_int4_token"), acc("v_int4_token")
    c_kl = (per.get("kv_int8_token") or {}).get("mean_kl_teacher_forced")
    eps = 1e-12
    if c is not None and c_kl is not None:
        out["c_int8"] = bool(abs(c - fp) <= 0.05 + eps and c_kl < 0.05)
        out["rival_token_path_defect"] = not out["c_int8"]
    if a is not None:
        out["a_k_only"] = bool(a <= 0.1 + eps)
    if b is not None:
        out["b_v_only"] = bool(abs(b - fp) <= 0.05 + eps)
    if channel is not None:
        o = channel["overall"]
        out["k_ratio"] = bool(o["k_ratio"] >= 3 * o["v_ratio"])
        out["k_rest_error"] = bool(o["k_err_token_rest"] >= 3 * o["k_err_channel_rest"])
        out["values"] = {"k_ratio": o["k_ratio"], "v_ratio": o["v_ratio"],
                         "k_err_token_rest": o["k_err_token_rest"], "k_err_channel_rest": o["k_err_channel_rest"]}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Entry point


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-prompts", default=None, help="N or ID,ID,... (tests only)")
    ap.add_argument("--model-path", default=None, help="local model directory (tests only)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    limits = {"prompts": args.limit_prompts, "model_path": args.model_path}
    limited = KR.check_out(args.out, limits)
    out = Path(args.out)
    device = torch.device(args.device)
    if KR.is_cuda(device):
        torch.cuda.set_device(device)
    seed_everything(KR.SEED)
    files = OUTPUT_FILES
    data = KR.data_record()
    all_cases = KR.load_cases()
    cases = select_prompts(all_cases, args.limit_prompts)
    config = {
        "version": VERSION, "scorer": KM.SCORER, "protocol": "task2/PROTOCOL.md AM4", "pipeline_version": KR.VERSION,
        "device": str(device), "model_id": KR.MODEL_ID if args.model_path is None else None,
        "model_revision": KR.locked_revision() if args.model_path is None else None,
        "model_path": args.model_path, "dtype": "float16", "attn_implementation": KR.ATTN_IMPLEMENTATION,
        "limits": limits, "limited": limited, "output_files": files, "data": data, "n_prompts": len(cases),
        "prompt_ids": [c["id"] for c in cases], "diagnostic_prompt_ids": KR.first_per_cell(all_cases),
        "configs": list(CONFIGS), "labels": LABELS, "quant": {m: list(q) for m, q in QUANT.items()},
        "evaluated_code_path": list(EVALUATED),
        "control_specs": {m: {"k": spec_name(s["k_spec"]), "v": spec_name(s["v_spec"])} for m, s in SPECS.items()},
        "group_size": GROUP, "int8_storage": "one uint8 code per element, no packing; FP16 scale and offset per group",
        "top_fraction": TOP_FRACTION, "top_channels_rule": "ceil(top_fraction * head_dim) per layer and KV head by "
        "channel max |x|; ties to the lower channel index",
        "median_rule": "torch.quantile(q=0.5, linear): mean of the two middle values for an even count",
        "order_rule": ORDER_RULE, "seed": KR.SEED, "predictions": PREDICTIONS,
        "generation": {"settings": KR.GREEDY, "decoder": "argmax loop task2.kv_cache_ops.greedy_decode; the last "
                       "generated token is not fed back", "stop": "EOS ids or max_new_tokens"},
        "teacher_forced_session_device": TF_DEVICE,
        "npz_arrays": {"per_channel": list(CHANNEL_ARRAYS), "per_head": list(HEAD_ARRAYS), "masks": list(MASK_ARRAYS),
                       "scalars": list(SCALAR_ARRAYS)},
        "npz_format_version": NPZ_FORMAT_VERSION,
    }
    out.mkdir(parents=True)
    rec = RunRecorder(out / files["run"], config)
    timing = {"load_s": None, "prompt_s": {}}
    try:
        rec.update(environment=KR.environment(device), step="loading")
        t_load = time.perf_counter()
        tokenizer = KR.load_tokenizer(args.model_path)
        model, info = KR.load_model(device, args.model_path)
        model_facts = KR.check_model(model)
        shape = KB.KVShape.from_config(model.config)
        timing["load_s"] = time.perf_counter() - t_load
        rec.update(model=info, model_facts=model_facts, kv_shape=vars(shape) | {"channels": shape.channels},
                   tokenizer=KR.tokenizer_meta(tokenizer, args.model_path), timing=timing, step="loaded")
        plain = {k: out / files[k] for k in ("rows", "prompts")}
        with RowWriter(plain["rows"]) as w_rows, RowWriter(plain["prompts"]) as w_prompts:
            writers = {"rows": w_rows, "prompts": w_prompts}
            summary = run_diag(model, tokenizer, shape, cases, device, out, files, writers, rec, timing)
        row_files = {k: KR.finalize_rows(p) for k, p in plain.items()}
        rec.finish("ok", summary=summary, row_files=row_files, timing=timing, step="done")
    except BaseException as exc:
        rec.finish("failed", timing=timing, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    print(f"kv_diag done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
