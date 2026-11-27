"""Task 2 Part B model: a two-block pre-norm decoder in which only the attention operator varies.

Architecture (identical parameter shapes for every mechanism): learned token embeddings (vocabulary 256, width 128)
plus fixed sinusoidal positional embeddings; 2 pre-norm blocks, each x + attn(LN(x)) then x + MLP(LN(x)), with
LayerNorm (weight and bias), q, k, v, o linear layers with bias (4 heads of dimension 32) and an MLP
128 -> 512 -> 128 with exact GELU and biases; a final LayerNorm; an untied linear head 128 -> 256 without bias.
Trainable parameters: 462,336 (the protocol B3 count, which excludes an output-head bias). Initialization is the
PyTorch default of each module (Embedding N(0, 1), Linear Kaiming-uniform weight and uniform bias, LayerNorm weight 1
and bias 0), drawn in construction order, so one seed gives identical weights for every mechanism.

Attention operators (scores q k^T / sqrt(32) for the softmax variants):
  dense          SDPA with is_causal=True; no mask tensor is built.
  window         SDPA with an explicit (T, T) boolean mask: key j allowed iff i - 63 <= j <= i.
  local_dilated  SDPA with an explicit (4, T, T) boolean mask: heads 0-1 use the window above; heads 2-3 allow
                 j = i - 4n for n = 0..63 with j >= 0.
  linear         phi(x) = ELU(x) + 1; a_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + 1e-6) with S_t = sum_{s<=t} phi(k_s) v_s^T
                 and z_t = sum_{s<=t} phi(k_s), computed with cumulative sums over the sequence. This materializes a
                 B x H x T x 32 x 32 prefix tensor per layer. linear_attention_recurrent is the step-by-step version,
                 used only for verification.
The explicit masks are correctness implementations; their storage is not that of an efficient sparse kernel.
Eval-time option window64: a dense-trained model evaluated with the window mask (model.mask_override("window64")).
"""
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256
D_MODEL = 128
N_HEADS = 4
HEAD_DIM = 32
D_FF = 512
N_LAYERS = 2
WINDOW = 64
DILATION = 4
DILATED_SLOTS = 64
LINEAR_EPS = 1e-6
POS_BASE = 10000.0
MECHANISMS = ("dense", "window", "local_dilated", "linear")
MASK_CONDITIONS = ("native", "window64")
DENSE_PARAMS = 462_336
HEAD_TYPES = {"dense": ["dense"] * 4, "window": ["window"] * 4,
              "local_dilated": ["window", "window", "dilated", "dilated"], "linear": ["linear"] * 4}


def sinusoidal(T: int, d: int = D_MODEL, device=None) -> torch.Tensor:
    """(T, d) FP32 table: PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(p / 10000^(2i/d))."""
    pos = torch.arange(T, dtype=torch.float32, device=device)[:, None]
    inv = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(POS_BASE) / d))
    pe = torch.zeros(T, d, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


def window_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) bool, True iff i - 63 <= j <= i."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j >= i - (WINDOW - 1))


def dilated_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) bool, True iff j = i - 4n for some n in 0..63 and j >= 0."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    off = i - j
    return (off >= 0) & (off % DILATION == 0) & (off <= DILATION * (DILATED_SLOTS - 1))


def local_dilated_mask(T: int, device=None) -> torch.Tensor:
    """(4, T, T) bool: heads 0-1 window, heads 2-3 dilated."""
    w, d = window_mask(T, device), dilated_mask(T, device)
    return torch.stack([w, w, d, d])


def phi(x):
    return F.elu(x) + 1.0


def linear_attention_cumsum(q, k, v, eps: float = LINEAR_EPS):
    """Causal ELU+1 linear attention with cumulative sums; q, k, v (B, H, T, d). Builds the (B, H, T, d, d) prefix S."""
    fq, fk = phi(q), phi(k)
    S = torch.cumsum(fk.unsqueeze(-1) * v.unsqueeze(-2), dim=2)  # (B, H, T, d_k, d_v)
    z = torch.cumsum(fk, dim=2)  # (B, H, T, d_k)
    num = torch.einsum("bhtk,bhtkv->bhtv", fq, S)
    den = (fq * z).sum(-1, keepdim=True) + eps
    return num / den


def linear_attention_recurrent(q, k, v, eps: float = LINEAR_EPS):
    """The same output by the recurrence S_t = S_{t-1} + phi(k_t) v_t^T, z_t = z_{t-1} + phi(k_t) (verification only)."""
    B, H, T, d = q.shape
    fq, fk = phi(q), phi(k)
    S = q.new_zeros(B, H, d, v.shape[-1])
    z = q.new_zeros(B, H, d)
    outs = []
    for t in range(T):
        S = S + fk[:, :, t, :, None] * v[:, :, t, None, :]
        z = z + fk[:, :, t]
        num = torch.einsum("bhk,bhkv->bhv", fq[:, :, t], S)
        den = (fq[:, :, t] * z).sum(-1, keepdim=True) + eps
        outs.append(num / den)
    return torch.stack(outs, dim=2)


def linear_prefix_bytes(B: int, T: int, elem: int = 4) -> int:
    return B * N_HEADS * T * HEAD_DIM * HEAD_DIM * elem


def mask_bytes(mechanism: str, T: int) -> int:
    """Bytes of the explicit boolean mask a mechanism builds (0 for dense SDPA with is_causal and for linear)."""
    return {"dense": 0, "window": T * T, "local_dilated": N_HEADS * T * T, "linear": 0}[mechanism]


def materialization(mechanism: str, B: int, T: int) -> dict:
    """What a forward pass at (B, T) materializes beyond the activations, per layer."""
    return {"dense_mask": mechanism in ("window", "local_dilated"), "dense_mask_bytes": mask_bytes(mechanism, T),
            "dense_mask_shape": {"window": [T, T], "local_dilated": [N_HEADS, T, T]}.get(mechanism),
            "linear_prefix": mechanism == "linear",
            "linear_prefix_bytes_per_layer": linear_prefix_bytes(B, T) if mechanism == "linear" else 0,
            "linear_prefix_shape": [B, N_HEADS, T, HEAD_DIM, HEAD_DIM] if mechanism == "linear" else None,
            "note": {"dense": "SDPA is_causal=True; no mask tensor",
                     "window": "explicit dense boolean mask through SDPA (correctness implementation)",
                     "local_dilated": "explicit dense per-head boolean mask through SDPA (correctness implementation)",
                     "linear": "cumulative-sum prefix tensor of phi(k) v^T, FP32, one per layer"}[mechanism]}


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x, op: str, masks: "MaskCache"):
        B, T, _ = x.shape

        def heads(t):
            return t.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        if op == "dense":
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif op == "window":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=masks.get("window", T, x.device))
        elif op == "local_dilated":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=masks.get("local_dilated", T, x.device)[None])
        elif op == "linear":
            a = linear_attention_cumsum(q, k, v)
        else:
            raise ValueError(f"unknown attention operator {op}")
        return self.o(a.transpose(1, 2).reshape(B, T, D_MODEL))


class MaskCache:
    """Boolean masks built once per (kind, T, device); not part of the state dict."""

    def __init__(self):
        self._m = {}

    def get(self, kind, T, device):
        key = (kind, T, str(device))
        if key not in self._m:
            fn = {"window": window_mask, "local_dilated": local_dilated_mask}[kind]
            self._m[key] = fn(T, device)
        return self._m[key]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x, op, masks):
        x = x + self.attn(self.ln1(x), op, masks)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class MQARModel(nn.Module):
    def __init__(self, mechanism: str):
        super().__init__()
        if mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}")
        self.mechanism = mechanism
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self._op = mechanism
        self._masks = MaskCache()
        self._pe = {}

    @property
    def operator(self) -> str:
        return self._op

    @contextmanager
    def mask_override(self, condition: str):
        """Evaluate with another mask condition: 'native' (the trained operator) or 'window64' (dense models only)."""
        if condition not in MASK_CONDITIONS:
            raise ValueError(f"mask condition must be one of {MASK_CONDITIONS}")
        if condition == "window64" and self.mechanism != "dense":
            raise ValueError("window64 applies to dense-trained models only")
        prev = self._op
        self._op = "window" if condition == "window64" else self.mechanism
        try:
            yield self
        finally:
            self._op = prev

    def positional(self, T, device):
        key = (T, str(device))
        if key not in self._pe:
            self._pe[key] = sinusoidal(T, D_MODEL, device)
        return self._pe[key]

    def forward_embeddings(self, x):
        """Blocks, final norm and head on given (B, T, 128) input embeddings; returns (B, T, 256) logits."""
        for blk in self.blocks:
            x = blk(x, self._op, self._masks)
        return self.head(self.ln_f(x))

    def forward(self, tokens):
        """tokens (B, T) int64 -> logits (B, T, 256)."""
        T = tokens.shape[1]
        x = self.tok(tokens) + self.positional(T, tokens.device)
        return self.forward_embeddings(x)


def build_model(mechanism: str, seed: int) -> MQARModel:
    """Seeded construction on the CPU in FP32 (the same seed gives the same weights for every mechanism)."""
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        model = MQARModel(mechanism).to(torch.float32)
    finally:
        torch.random.set_rng_state(gen_state)
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------------------------
# Theoretical costs

def candidates(head_type: str, i: int) -> int:
    """Key positions a query at position i can attend to directly in one head."""
    if head_type in ("dense", "linear"):
        return i + 1
    if head_type == "window":
        return min(i + 1, WINDOW)
    return min(i // DILATION + 1, DILATED_SLOTS)


def candidate_set(head_type: str, i: int) -> set:
    if head_type in ("dense", "linear"):
        return set(range(i + 1))
    if head_type == "window":
        return set(range(max(0, i - WINDOW + 1), i + 1))
    return {i - DILATION * n for n in range(DILATED_SLOTS) if i - DILATION * n >= 0}


def kv_positions(head_type: str, T: int) -> int:
    """Cached positions per head needed to continue decoding after T tokens."""
    if head_type == "dense":
        return T
    if head_type == "window":
        return min(T, WINDOW)
    return min(T, DILATION * (DILATED_SLOTS - 1) + 1)


def theoretical_costs(mechanism: str, T: int = 256) -> dict:
    types = HEAD_TYPES[mechanism]
    distinct = sorted(set(types))
    per_type = {}
    for ht in distinct:
        counts = [candidates(ht, i) for i in range(T)]
        per_type[ht] = {"attended_pairs_per_head_per_layer": None if ht == "linear" else sum(counts),
                        "candidates_at_last_position": counts[-1], "candidates_max": max(counts),
                        "candidates_mean": sum(counts) / T}
    union_last = len(set().union(*(candidate_set(ht, T - 1) for ht in types)))
    out = {"T": T, "mechanism": mechanism, "head_types": types, "per_head_type": per_type,
           "union_candidates_at_last_position": union_last}
    if mechanism == "linear":
        elems = N_HEADS * (HEAD_DIM * HEAD_DIM + HEAD_DIM)
        out.update(attended_pairs_per_layer=None,
                   candidates_note="every earlier position contributes through a fixed-size state; none is "
                                   "retrieved individually",
                   persistent_state={"elements_per_layer": elems, "bytes_per_layer_fp32": 4 * elems,
                                     "bytes_per_layer_fp16": 2 * elems,
                                     "definition": "S (4 x 32 x 32) and z (4 x 32) per layer"},
                   kv_cache=None)
    else:
        pos = [kv_positions(ht, T) for ht in types]
        elems = 2 * sum(pos) * HEAD_DIM
        out.update(attended_pairs_per_layer=sum(per_type[ht]["attended_pairs_per_head_per_layer"] for ht in types),
                   persistent_state=None,
                   kv_cache={"positions_per_head": pos, "elements_per_layer": elems, "bytes_per_layer_fp32": 4 * elems,
                             "bytes_per_layer_fp16": 2 * elems,
                             "definition": "keys and values of the positions each head can still attend to when "
                                           "decoding continues after T tokens"})
    return out


def all_theoretical_costs(T: int = 256) -> dict:
    return {m: theoretical_costs(m, T) for m in MECHANISMS}
