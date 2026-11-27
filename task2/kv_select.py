"""Question-aware importance scores and retention sets for Part C (protocol C2, C3, AM2.5).

Scores: during the full FP16 prefill, forward hooks on each layer's q_proj keep the query projections at the
question-suffix positions; the attention implementation of the model is unchanged. Each layer's queries are rotated
with the model's own rotary embedding at their absolute positions (model dtype), cast to FP32, and scored against the
cached, already rotated keys (FP32, expanded to the query heads) with a causally masked softmax over every cached key.
Probabilities on context positions are averaged over layers, all query heads and suffix positions, giving one global
ranking shared by all layers and heads.

Inputs are token positions and model tensors only: nothing in this module takes the answer code or evidence offsets.
"""
from __future__ import annotations

import torch

from task2.kv_budget import N_RECENT, N_SINK


class SuffixQueryCapture:
    """Context manager keeping q_proj outputs at positions >= start for every decoder layer."""

    def __init__(self, model, start: int):
        self.model, self.start = model, int(start)
        self.queries: list[torch.Tensor | None] = [None] * len(model.model.layers)
        self._handles = []

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            self._handles.append(layer.self_attn.q_proj.register_forward_hook(self._hook(i)))
        return self

    def _hook(self, i):
        def fn(module, inputs, output):
            self.queries[i] = output[:, self.start:, :].detach().clone()
        return fn

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def suffix_attention(model, queries: list, keys: list, suffix_start: int, per_layer: bool = False):
    """FP32 causal attention probabilities of the suffix queries over all cached keys.

    queries[i]: q_proj output of layer i at the suffix, [1, S, H_q * d]. keys[i]: cached rotated keys [1, H_kv, T, d]
    with T >= suffix_start + S. Returns the per-layer list of probabilities [1, H_q, S, T] if per_layer, else the
    layer, head and suffix mean over keys [T] in FP32.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    S = queries[0].shape[1]
    T = keys[0].shape[2]
    dev = keys[0].device
    pos = torch.arange(suffix_start, suffix_start + S, device=dev)
    allowed = torch.arange(T, device=dev)[None, :] <= pos[:, None]  # [S, T]
    total = torch.zeros(T, dtype=torch.float32, device=dev)
    out = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        q = queries[i].view(1, S, -1, attn.head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(q, pos[None])
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = repeat_kv(keys[i][:, :, :T].float(), attn.num_key_value_groups)
        logits = torch.matmul(q.float(), k.transpose(-1, -2)) * attn.scaling
        logits = logits.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        if per_layer:
            out.append(probs)
        else:
            total += probs.mean(dim=(0, 1, 2))
    return out if per_layer else total / len(model.model.layers)


def question_scores(model, queries: list, keys: list, context_len: int, suffix_start: int) -> torch.Tensor:
    """Scores [context_len] in FP32: mean suffix attention probability on each context position."""
    return suffix_attention(model, queries, keys, suffix_start)[:context_len]


def safeguard_positions(L: int, n_sink: int = N_SINK, n_recent: int = N_RECENT) -> torch.Tensor:
    """The first n_sink and last n_recent context positions (union, ascending)."""
    idx = torch.cat([torch.arange(min(n_sink, L)), torch.arange(max(0, L - n_recent), L)])
    return torch.unique(idx)


def global_ranking(scores: torch.Tensor, n_sink: int = N_SINK, n_recent: int = N_RECENT) -> torch.Tensor:
    """All context positions: safeguards first (ascending), then the rest by descending score, ties to the lower
    position."""
    scores = scores.detach().float().cpu()
    L = scores.numel()
    safe = safeguard_positions(L, n_sink, n_recent)
    rest_mask = torch.ones(L, dtype=torch.bool)
    rest_mask[safe] = False
    rest = torch.nonzero(rest_mask).flatten()
    order = torch.argsort(-scores[rest], stable=True)
    return torch.cat([safe, rest[order]])


def sink_recent_positions(L: int, n: int, n_sink: int = N_SINK, n_recent: int = N_RECENT) -> torch.Tensor:
    """First n_sink positions plus the n - n_sink most recent positions (ascending)."""
    if n >= L:
        return torch.arange(L)
    if n - n_sink < n_recent:
        raise ValueError(f"n={n} leaves fewer than {n_recent} recent positions")
    return torch.cat([torch.arange(n_sink), torch.arange(L - (n - n_sink), L)])


def question_aware_positions(ranking: torch.Tensor, n: int) -> torch.Tensor:
    """The first n positions of the global ranking, ascending."""
    return torch.sort(ranking[:n]).values


def hybrid_positions(ranking: torch.Tensor, f: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(FP16 positions, low-bit positions): the first f of the ranking and the ordered complement."""
    fp = torch.sort(ranking[:f]).values
    low = torch.sort(ranking[f:]).values
    return fp, low
