"""Part C cache pipeline (protocol C1, C6, C7, C8, AM2.2-AM2.6).

Order: full FP16 prefill of the whole prompt (logits_to_keep 1) with suffix query capture; question-aware scoring;
compression of the pre-question context [0, L) into the method's packed object; a DynamicCache built from the
dequantized tensors (retained positions in original order); re-run of the question suffix with absolute position_ids
and physical cache_position; greedy decoding. Appended entries are added to the packed object after each forward pass,
and when the quantized layouts complete a 64-token tail group the affected layer tensors are rebuilt from the packed
codes, so attention always uses the stored values.

Gold answer tokens are used only by teacher_forced_logits, after generation, on a copy of the post-suffix session.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from task2 import kv_budget as KB
from task2 import kv_select as KS
from task2.kv_quant import PackedKVCache

METHODS = ("fp16", "sink_recent", "question_aware", "int4_token", "int4_channel_k", "hybrid")
SCORED_METHODS = ("question_aware", "hybrid")
EOS_IDS = (151645, 151643)
MAX_NEW_TOKENS = 32


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class _Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        _sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync(self.device)
        self.seconds = time.perf_counter() - self.t0
        return False


def _peak(device):
    if torch.device(device).type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _reset_peak(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------------------------------------------------
# Prefill and scoring


@dataclass
class Prefill:
    context_len: int
    total_len: int
    keys: list  # per layer [1, H_kv, T, d], rotated
    values: list
    last_logits: torch.Tensor  # [V] FP32, full-forward logits at the first answer position
    queries: list  # per layer q_proj output at the suffix [1, S, H_q * d]
    seconds: float = 0.0
    peak_bytes: int | None = None


@torch.no_grad()
def full_prefill(model, input_ids: torch.Tensor, context_len: int) -> Prefill:
    """Full FP16 prefill of the whole prompt with logits_to_keep 1 and suffix query capture."""
    from transformers import DynamicCache

    if input_ids.shape[0] != 1:
        raise ValueError("the Part C pipeline runs one prompt at a time")
    dev = input_ids.device
    _reset_peak(dev)
    cache = DynamicCache(config=model.config)
    with KS.SuffixQueryCapture(model, context_len) as cap, _Timer(dev) as t:
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    keys = [layer.keys for layer in cache.layers]
    values = [layer.values for layer in cache.layers]
    return Prefill(int(context_len), int(input_ids.shape[1]), keys, values, out.logits[0, -1].float(),
                   cap.queries, t.seconds, _peak(dev))


@torch.no_grad()
def score_prefill(model, prefill: Prefill) -> tuple[torch.Tensor, float]:
    """Question-aware scores over the context and the scoring time in seconds."""
    with _Timer(prefill.keys[0].device) as t:
        s = KS.question_scores(model, prefill.queries, prefill.keys, prefill.context_len, prefill.context_len)
    return s, t.seconds


# ---------------------------------------------------------------------------------------------------------------------
# Compression


def compress(method: str, shape: KB.KVShape, keys: list, values: list, scores: torch.Tensor | None = None):
    """Compress context tensors (per layer [1, H_kv, L, d]) into the method's packed object.

    Returns (packed, info) where info holds the budget, the counts chosen and the FP16 position set.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method in SCORED_METHODS and scores is None:
        raise ValueError(f"{method} needs question-aware scores")
    L = keys[0].shape[2]
    budget = KB.m4(shape, L)
    info = {"method": method, "L": L, "M16": KB.m16(shape, L), "M4": budget}
    if method == "fp16":
        packed = PackedKVCache.fp16(keys, values)
    elif method == "sink_recent":
        n = KB.sink_recent_n(shape, L, budget)
        pos = KS.sink_recent_positions(L, n)
        packed = PackedKVCache.evict(method, keys, values, pos,
                                     torch.tensor([KB.N_SINK, n - KB.N_SINK], dtype=torch.int32))
        info.update(n=n, formula_bytes=KB.sink_recent_bytes(shape, n))
    elif method == "question_aware":
        n = KB.question_aware_n(shape, L, budget)
        pos = KS.question_aware_positions(KS.global_ranking(scores), n)
        packed = PackedKVCache.evict(method, keys, values, pos, pos.to(torch.int32))
        info.update(n=n, formula_bytes=KB.question_aware_bytes(shape, n))
    elif method in ("int4_token", "int4_channel_k"):
        layout = "token" if method == "int4_token" else "channel"
        packed = PackedKVCache.uniform(method, keys, values, layout, 4)
        info.update(formula_bytes=budget)
    else:
        m, f = KB.hybrid_split(shape, L, budget)
        fp, _ = KS.hybrid_positions(KS.global_ranking(scores), f)
        packed = PackedKVCache.hybrid(keys, values, fp, 2)
        info.update(m=m, f=f, formula_bytes=KB.hybrid_bytes(shape, L, m))
    if method == "fp16":
        info["formula_bytes"] = info["M16"]
    info["fp16_positions"] = packed.fp_positions.cpu() if method in ("sink_recent", "question_aware", "hybrid") \
        else None
    return packed, info


# ---------------------------------------------------------------------------------------------------------------------
# Session on a compressed cache


@dataclass
class CacheSession:
    """A packed cache, its dequantized DynamicCache and the next absolute position."""

    model: object
    packed: PackedKVCache
    cache: object
    next_pos: int
    rebuilds: int = 0

    @classmethod
    def start(cls, model, packed: PackedKVCache, dtype=None) -> "CacheSession":
        return cls(model, packed, packed.to_dynamic_cache(model.config, dtype), packed.context_len)

    @property
    def physical_len(self) -> int:
        return self.cache.get_seq_length()

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, logits_to_keep: int = 1) -> torch.Tensor:
        """Run ids [1, q] at the next absolute positions; returns FP32 logits [k, V] of the last k positions
        (all q positions when logits_to_keep is 0)."""
        q = ids.shape[1]
        dev = ids.device
        phys = self.physical_len
        position_ids = torch.arange(self.next_pos, self.next_pos + q, device=dev)[None]
        cache_position = torch.arange(phys, phys + q, device=dev)
        out = self.model(input_ids=ids, past_key_values=self.cache, position_ids=position_ids,
                         cache_position=cache_position, use_cache=True, logits_to_keep=logits_to_keep)
        new_k = [layer.keys[:, :, -q:] for layer in self.cache.layers]
        new_v = [layer.values[:, :, -q:] for layer in self.cache.layers]
        groups = self.packed.append(new_k, new_v, position_ids[0])
        if groups:
            for i, layer in enumerate(self.cache.layers):
                layer.keys, layer.values = self.packed.layer_tensors(i, layer.keys.dtype)
            self.rebuilds += groups
        self.next_pos += q
        return out.logits[0].float()

    def clone(self) -> "CacheSession":
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config)
        for i, layer in enumerate(self.cache.layers):
            cache.update(layer.keys.clone(), layer.values.clone(), i)
        return CacheSession(self.model, self.packed.clone(), cache, self.next_pos, self.rebuilds)


@torch.no_grad()
def greedy_decode(session: CacheSession, first_logits: torch.Tensor, max_new_tokens: int = MAX_NEW_TOKENS,
                  eos_ids=EOS_IDS) -> dict:
    """Greedy decoding (argmax, no repetition penalty) until an EOS id or max_new_tokens; the last generated token is
    not fed back."""
    dev = session.packed.layers[0].k_tail.device
    eos = set(int(e) for e in eos_ids)
    tokens, logits, stop = [], first_logits, "max_new_tokens"
    while True:
        tok = int(torch.argmax(logits))
        tokens.append(tok)
        if tok in eos:
            stop = "eos"
            break
        if len(tokens) >= max_new_tokens:
            break
        logits = session.forward(torch.tensor([[tok]], device=dev))[-1]
    return {"tokens": tokens, "stop": stop}


@torch.no_grad()
def teacher_forced_logits(session: CacheSession, first_logits: torch.Tensor, gold_ids) -> torch.Tensor:
    """FP32 logits [n, V] predicting each gold token: row 0 is first_logits, then one forward per gold token
    (the last gold token is not fed). Consumes the session; pass a clone."""
    dev = session.packed.layers[0].k_tail.device
    rows = [first_logits.float()]
    for tok in list(gold_ids)[:-1]:
        rows.append(session.forward(torch.tensor([[int(tok)]], device=dev))[-1])
    return torch.stack(rows)


# ---------------------------------------------------------------------------------------------------------------------
# One method on one prompt


@dataclass
class MethodRun:
    method: str
    info: dict
    first_logits: torch.Tensor
    tokens: list
    stop: str
    context_bytes: dict
    live_bytes: dict
    n_appended: int
    recon: dict | None
    seconds: dict = field(default_factory=dict)
    peak_bytes: int | None = None
    tail_rebuilds: int = 0


@torch.no_grad()
def run_method(model, shape: KB.KVShape, prefill: Prefill, input_ids: torch.Tensor, method: str,
               scores: torch.Tensor | None = None, max_new_tokens: int = MAX_NEW_TOKENS, eos_ids=EOS_IDS,
               suffix_logits_all: bool = False):
    """Compress, re-run the suffix and decode one method. Returns (MethodRun, post-suffix session copy for the
    teacher-forced pass, suffix logits [S, V] if suffix_logits_all else None)."""
    dev = input_ids.device
    L = prefill.context_len
    _reset_peak(dev)
    keys = [k[:, :, :L] for k in prefill.keys]
    values = [v[:, :, :L] for v in prefill.values]
    with _Timer(dev) as t_c:
        packed, info = compress(method, shape, keys, values, scores)
    recon = packed.reconstruction_errors(keys, values)
    context_bytes = dict(packed.context_bytes)
    with _Timer(dev) as t_s:
        session = CacheSession.start(model, packed, keys[0].dtype)
        suffix_logits = session.forward(input_ids[:, L:], logits_to_keep=0 if suffix_logits_all else 1)
    first = suffix_logits[-1]
    tf_session = session.clone()
    with _Timer(dev) as t_d:
        gen = greedy_decode(session, first, max_new_tokens, eos_ids)
    run = MethodRun(
        method=method, info=info, first_logits=first, tokens=gen["tokens"], stop=gen["stop"],
        context_bytes=context_bytes, live_bytes=packed.nbytes(), n_appended=packed.n_appended, recon=recon,
        seconds={"compression": t_c.seconds, "suffix": t_s.seconds, "decode": t_d.seconds},
        peak_bytes=_peak(dev), tail_rebuilds=session.rebuilds,
    )
    return run, tf_session, (suffix_logits if suffix_logits_all else None)
