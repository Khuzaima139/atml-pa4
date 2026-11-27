"""Exact causal attention operators: explicit dense, tiled online softmax, and PyTorch SDPA.

All operators take q, k, v of shape (B, H, L, d) and return (B, H, L, d) in the input dtype.

Boolean masks follow the torch scaled_dot_product_attention convention: True means the key takes part in attention.
An optional extra mask must broadcast to (B, H, L, L); it is combined with the causal mask by logical and.

Score scaling. Scores are s_ij = q_i k_j^T / sqrt(d), computed as (q * d**-0.5) @ k^T: the query is scaled before the
matmul so that large FP16 scores do not overflow before the division. For d = 64 the factor 1/8 is a power of two,
so the scaling is exact in FP16 and FP32 (outside the subnormal range).

Row maxima. The running or row maximum is computed without autograd tracking. Softmax is invariant to a constant
shift of each row, so the output does not depend on the maximum and the gradient obtained with a detached maximum is
the exact gradient (the omitted path contributes zero analytically). This avoids storing max indices for backward.
"""
import re

import torch
import torch.nn.functional as F

SCORE_SCALE_NOTE = "q is multiplied by d**-0.5 before the q k^T matmul"


def _check(q, k, v):
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q, k, v must share shape (B, H, L, d); got {tuple(q.shape)}, {tuple(k.shape)}, "
                         f"{tuple(v.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must share one dtype")


def causal_mask(n_q: int, n_k: int, device, q_offset: int = 0, k_offset: int = 0) -> torch.Tensor:
    """Boolean (n_q, n_k) mask, True where key index <= query index (absolute indices with the given offsets)."""
    qi = torch.arange(q_offset, q_offset + n_q, device=device)
    kj = torch.arange(k_offset, k_offset + n_k, device=device)
    return qi[:, None] >= kj[None, :]


def dense_causal(q, k, v, mask=None):
    """Explicit dense causal attention; every operation runs in the input dtype.

    Steps: s = (q / sqrt(d)) k^T (an L x L score matrix per head), positions outside the causal mask (and outside the
    optional extra boolean mask) set to -inf, row maximum subtracted, exp, division by the row sum, multiplication by v.

    Fully masked rows (possible only with an extra mask) return output 0 and receive gradient 0: their row maximum is
    replaced by 0, every exp is exactly 0, and the row sum 0 is replaced by 1 before the division.
    """
    _check(q, k, v)
    L, d = q.shape[-2], q.shape[-1]
    allowed = causal_mask(L, L, q.device)
    if mask is not None:
        allowed = allowed & mask.to(device=q.device, dtype=torch.bool)
    s = torch.matmul(q * (d ** -0.5), k.transpose(-2, -1))
    s = s.masked_fill(~allowed, float("-inf"))
    with torch.no_grad():
        m = s.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(s - m)
    denom = e.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    p = e / denom
    return torch.matmul(p, v)


def tiled_causal(q, k, v, tile_q: int, tile_k: int, mask=None):
    """Exact causal attention with the online-softmax update over key tiles.

    For each query tile, key tiles are visited in increasing order and the running maximum m, normalizer l and
    unnormalized accumulator o are updated as
        m' = max(m, max s),  l' = exp(m - m') l + sum_j exp(s_j - m'),  o' = exp(m - m') o + sum_j exp(s_j - m') V_j,
    and the output is o / l after the last tile. Matmuls (q k^T and p v) run in the input dtype (p is cast to it for
    the second matmul); m, l and o are kept in FP32; the output is cast to the input dtype.

    Key tiles that lie entirely after the last query of the query tile are skipped; tiles that cross the diagonal are
    masked element-wise. The full L x L score matrix is never formed: at most one tile_q x tile_k score block per
    (batch, head) exists at a time in the forward computation. L need not be a multiple of the tile sizes.

    Backward runs through autograd. Autograd stores the per-tile intermediates (score blocks, exponentials, the cast
    p blocks), so the stored total grows like the number of visited tile pairs, roughly L^2 / 2 elements per head.
    This is a Python reference implementation of the algorithm, not an optimized or fused FlashAttention kernel, and
    its speed does not measure the speed of the tiled algorithm.

    Fully masked rows (possible only with the extra mask) return output 0 and receive gradient 0: the running maximum
    stays -inf and is replaced by 0 inside each exponential, every exp is exactly 0, and l = 0 is replaced by 1 at the end.
    """
    _check(q, k, v)
    if tile_q < 1 or tile_k < 1:
        raise ValueError("tile sizes must be positive")
    B, H, L, d = q.shape
    dtype = q.dtype
    qs = q * (d ** -0.5)
    extra = None
    if mask is not None:
        extra = mask.to(device=q.device, dtype=torch.bool).expand(B, H, L, L)
    outs = []
    for i0 in range(0, L, tile_q):
        i1 = min(i0 + tile_q, L)
        q_t = qs[:, :, i0:i1]
        m = torch.full((B, H, i1 - i0, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, i1 - i0, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, i1 - i0, d), device=q.device, dtype=torch.float32)
        for j0 in range(0, i1, tile_k):  # key tiles starting after the last query (j0 > i1 - 1) are skipped
            j1 = min(j0 + tile_k, L)
            s = torch.matmul(q_t, k[:, :, j0:j1].transpose(-2, -1)).float()
            allowed = None
            if j1 - 1 > i0:  # some key in the tile follows some query in the tile
                allowed = causal_mask(i1 - i0, j1 - j0, q.device, i0, j0)
            if extra is not None:
                blk = extra[:, :, i0:i1, j0:j1]
                allowed = blk if allowed is None else (allowed & blk)
            if allowed is not None:
                s = s.masked_fill(~allowed, float("-inf"))
            with torch.no_grad():
                m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
                m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
                alpha = torch.exp(m - m_safe)
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p.to(dtype), v[:, :, j0:j1]).float()
            m = m_new
        l = torch.where(l == 0, torch.ones_like(l), l)
        outs.append((o / l).to(dtype))
    return torch.cat(outs, dim=2)


def sdpa_causal(q, k, v, backend=None, mask=None):
    """torch scaled_dot_product_attention with is_causal=True.

    backend: None for the default dispatch, or one torch.nn.attention.SDPBackend member forced on its own through
    torch.nn.attention.sdpa_kernel. A forced backend that cannot run raises (RuntimeError); it is never replaced by
    another backend.

    mask: optional extra boolean mask. SDPA does not accept is_causal together with attn_mask, so with an extra mask
    the call passes attn_mask = causal & mask and is_causal=False.
    """
    _check(q, k, v)

    def call():
        if mask is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        L = q.shape[-2]
        allowed = causal_mask(L, L, q.device) & mask.to(device=q.device, dtype=torch.bool)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, is_causal=False)

    if backend is None:
        return call()
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel([backend]):
        return call()


# ---------------------------------------------------------------------------------------------------------------
# Backend detection

def sdpa_backends():
    """Every SDPBackend member except ERROR, in enum order."""
    from torch.nn.attention import SDPBackend
    return [b for name, b in SDPBackend.__members__.items() if name != "ERROR"]


# aten operator name -> backend label, checked in order (the CPU fused kernel has its own label)
_ATEN_BACKEND = [
    (re.compile(r"_scaled_dot_product_flash_attention_for_cpu"), "FLASH_ATTENTION_CPU"),
    (re.compile(r"_scaled_dot_product_flash_attention(?!_for_cpu)|_flash_attention_forward|_flash_attention_backward"),
     "FLASH_ATTENTION"),
    (re.compile(r"_scaled_dot_product_efficient_attention|_efficient_attention_forward|_efficient_attention_backward"),
     "EFFICIENT_ATTENTION"),
    (re.compile(r"_scaled_dot_product_cudnn_attention|_cudnn_attention"), "CUDNN_ATTENTION"),
    (re.compile(r"_scaled_dot_product_fused_attention_overrideable"), "OVERRIDEABLE"),
    (re.compile(r"_scaled_dot_product_attention_math"), "MATH"),
]


def infer_backend(op_names) -> str:
    """Backend label from profiled aten operator names; 'unknown' if no SDPA operator is recognised."""
    found = []
    for pattern, label in _ATEN_BACKEND:
        if any(pattern.search(n) for n in op_names) and label not in found:
            found.append(label)
    if not found:
        return "unknown"
    return found[0] if len(found) == 1 else "+".join(found)


def profile_sdpa(q, k, v, backward: bool = False, grad=None) -> dict:
    """Run the default SDPA call (and optionally its backward) under torch.profiler.

    Returns the sorted attention-related aten operator names, the sorted CUDA kernel names (empty on CPU) and the
    inferred backend label.
    """
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if q.is_cuda:
        acts.append(ProfilerActivity.CUDA)
    if backward:
        q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    with profile(activities=acts) as prof:
        out = sdpa_causal(q, k, v)
        if backward:
            torch.autograd.grad(out, (q, k, v), grad if grad is not None else torch.ones_like(out))
        if q.is_cuda:
            torch.cuda.synchronize(q.device)
    aten, kernels = set(), set()
    for e in prof.events():
        dev = str(getattr(e, "device_type", ""))
        if "CUDA" in dev:
            kernels.add(e.name)
        elif e.name.startswith("aten::") and ("attention" in e.name or "sdp" in e.name):
            aten.add(e.name)
    return {"aten_ops": sorted(aten), "cuda_kernels": sorted(kernels), "inferred_backend": infer_backend(aten)}


def forced_backend_support(q, k, v, backward: bool = False) -> dict:
    """For each SDPBackend forced on its own: {'supported': bool, 'error': str | None}."""
    out = {}
    for b in sdpa_backends():
        qq, kk, vv = (t.detach().requires_grad_(backward) for t in (q, k, v))
        try:
            o = sdpa_causal(qq, kk, vv, backend=b)
            if backward:
                torch.autograd.grad(o, (qq, kk, vv), torch.ones_like(o))
            out[b.name] = {"supported": True, "error": None}
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            out[b.name] = {"supported": False, "error": str(exc)}
    return out


def score_matrix_bytes(B: int, H: int, L: int, elem_bytes: int) -> int:
    """Bytes of one dense B x H x L x L score tensor."""
    return B * H * L * L * elem_bytes


def dense_projected_peak_bytes(B: int, H: int, L: int, d: int, elem_bytes: int) -> int:
    """Projected peak of dense forward plus autograd backward: 4 score-sized tensors (upper end of the 3-4x range),
    the boolean L x L causal mask, and 9 B x H x L x d tensors (q, k, v, o, their gradients, the upstream gradient)."""
    return 4 * score_matrix_bytes(B, H, L, elem_bytes) + L * L + 9 * B * H * L * d * elem_bytes


GIB = float(1 << 30)


def gib(n_bytes) -> float:
    return n_bytes / GIB

