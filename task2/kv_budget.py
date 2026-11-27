"""Integer byte arithmetic for the Part C KV-cache budget (protocol AM2.3).

All shape constants come from the model config: n_l layers, H_kv KV heads, d_h head dimension. C = 2 n_l H_kv d_h
is the number of stored KV channels per token; one FP16 position costs 2C bytes.
"""
from __future__ import annotations

from dataclasses import dataclass

GROUP = 64
N_SINK = 8
N_RECENT = 64
N_SAFEGUARD = N_SINK + N_RECENT
META_BYTES = 4  # FP16 scale + FP16 offset per group
INDEX_BYTES = 4  # int32 per stored position
SINK_RECENT_INDEX_BYTES = 8  # two int32 counts


class BudgetError(ValueError):
    """Raised when the safeguard positions cannot fit inside the matched budget."""


@dataclass(frozen=True)
class KVShape:
    n_layers: int
    n_kv_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config) -> "KVShape":
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(int(config.num_hidden_layers), int(config.num_key_value_heads), int(head_dim))

    @property
    def channels(self) -> int:
        """C = 2 n_l H_kv d_h (K and V)."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim

    @property
    def fp16_bytes_per_position(self) -> int:
        return 2 * self.channels


def group_bytes(bits: int) -> int:
    """Bytes of one 64-element group: packed payload plus FP16 scale and offset."""
    if (GROUP * bits) % 8:
        raise ValueError(f"bits={bits} does not pack a 64-element group into whole bytes")
    return GROUP * bits // 8 + META_BYTES


def m16(shape: KVShape, L: int) -> int:
    """M16(L) = 4 n_l H_kv d_h L."""
    return shape.fp16_bytes_per_position * int(L)


def mb(shape: KVShape, L: int, bits: int) -> int:
    """M_b(L) = C [floor(L/64) (64b/8 + 4) + 2 (L mod 64)] (manual line 437-439)."""
    L = int(L)
    return shape.channels * ((L // GROUP) * group_bytes(bits) + 2 * (L % GROUP))


def m4(shape: KVShape, L: int) -> int:
    return mb(shape, L, 4)


def sink_recent_bytes(shape: KVShape, n: int) -> int:
    return shape.fp16_bytes_per_position * n + SINK_RECENT_INDEX_BYTES


def question_aware_bytes(shape: KVShape, n: int) -> int:
    return (shape.fp16_bytes_per_position + INDEX_BYTES) * n


def hybrid_bytes(shape: KVShape, L: int, m: int, low_bits: int = 2) -> int:
    """INT2 groups (channel-wise K, token-wise V) for 64m positions plus FP16 and index bytes for F = L - 64m."""
    f = int(L) - GROUP * m
    return shape.channels * m * group_bytes(low_bits) + (shape.fp16_bytes_per_position + INDEX_BYTES) * f


def _check_safeguards(n: int, L: int, what: str) -> None:
    if n < min(N_SAFEGUARD, L):
        raise BudgetError(f"{what}: {n} positions fit, fewer than the {N_SAFEGUARD} safeguards (L={L})")


def sink_recent_n(shape: KVShape, L: int, budget: int | None = None) -> int:
    """Largest N with 2C N + 8 <= M4, capped at L."""
    budget = m4(shape, L) if budget is None else int(budget)
    n = min((budget - SINK_RECENT_INDEX_BYTES) // shape.fp16_bytes_per_position, int(L))
    _check_safeguards(n, L, "sink-plus-recent")
    return n


def question_aware_n(shape: KVShape, L: int, budget: int | None = None) -> int:
    """Largest N with (2C + 4) N <= M4, capped at L."""
    budget = m4(shape, L) if budget is None else int(budget)
    n = min(budget // (shape.fp16_bytes_per_position + INDEX_BYTES), int(L))
    _check_safeguards(n, L, "question-aware")
    return n


def hybrid_split(shape: KVShape, L: int, budget: int | None = None, low_bits: int = 2) -> tuple[int, int]:
    """Smallest group count m whose hybrid bytes fit the budget; returns (m, F = L - 64m)."""
    L = int(L)
    budget = m4(shape, L) if budget is None else int(budget)
    for m in range(L // GROUP + 1):
        if hybrid_bytes(shape, L, m, low_bits) <= budget:
            f = L - GROUP * m
            _check_safeguards(f, L, "hybrid FP16 set")
            return m, f
    raise BudgetError(f"hybrid: no group count fits the budget {budget} (L={L})")


def decode_time_fp16_bytes(shape: KVShape, n_appended: int) -> int:
    """FP16 bytes of n appended entries (question suffix and fed-back generated tokens)."""
    return shape.fp16_bytes_per_position * int(n_appended)


def realized_fraction(nbytes: int, shape: KVShape, L: int) -> float:
    return int(nbytes) / m16(shape, L)


def budget_table(shape: KVShape, L: int) -> dict:
    """Every budget quantity for one prompt, as plain integers (plus realized fractions)."""
    L = int(L)
    M16, M4 = m16(shape, L), m4(shape, L)
    n_sr, n_qa = sink_recent_n(shape, L, M4), question_aware_n(shape, L, M4)
    m, f = hybrid_split(shape, L, M4)
    b_sr, b_qa, b_h = sink_recent_bytes(shape, n_sr), question_aware_bytes(shape, n_qa), hybrid_bytes(shape, L, m)
    return {
        "L": L, "r": L % GROUP, "channels": shape.channels, "M16": M16, "M4": M4,
        "sink_recent_n": n_sr, "sink_recent_bytes": b_sr,
        "question_aware_n": n_qa, "question_aware_bytes": b_qa,
        "hybrid_m": m, "hybrid_int2_positions": GROUP * m, "hybrid_fp16_positions": f, "hybrid_bytes": b_h,
        "realized_fraction": {
            "fp16": 1.0, "int4_token": M4 / M16, "int4_channel_k": M4 / M16,
            "sink_recent": b_sr / M16, "question_aware": b_qa / M16, "hybrid": b_h / M16,
        },
    }
