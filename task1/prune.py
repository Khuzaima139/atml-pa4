"""Per-output-row pruning masks (magnitude, Wanda, 2:4) and score-distribution summaries.

Masks are boolean "keep" tensors. A matrix receives a total number of zeros Z; with `out` rows,
k = Z // out and the first Z % out rows (lowest index) receive k + 1 zeros. Each row zeroes exactly
its count of lowest scores; ties are broken by column index (the lower column is pruned first) through a
stable ascending sort. 2:4: in each group of 4 contiguous input columns the 2 highest scores are kept
(same tie rule). Scores are FP32: magnitude |W|, Wanda |W_ij| * ||X_j||_2.
"""
from fractions import Fraction

import torch

PRUNE_CONVENTION = (
    "per output row; row zeros k=Z//out, rows with index < Z%out get k+1; lowest scores pruned, ties pruned "
    "in increasing column order (stable sort); 2:4 keeps the 2 highest of each 4 contiguous input columns; "
    "scores FP32: magnitude |W|, Wanda |W|*||X_j||_2 with ||X_j|| over all calibration tokens of the input"
)
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
N_BINS = 64
_ROW_CHUNK = 2048


def target_zeros(shape, sparsity: float) -> int:
    """Exact zeros for a sparsity; the product must be an integer."""
    out, inp = shape
    z = Fraction(str(sparsity)) * out * inp
    assert z.denominator == 1, (shape, sparsity)
    return int(z)


def zeros_per_row(total: int, out: int) -> torch.Tensor:
    k, r = divmod(int(total), int(out))
    counts = torch.full((out,), k, dtype=torch.long)
    counts[:r] += 1
    return counts


def magnitude_score(w: torch.Tensor) -> torch.Tensor:
    return w.detach().float().abs()


def wanda_score(w: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    assert x_norm.shape == (w.shape[1],)
    return w.detach().float().abs() * x_norm.float().to(w.device)[None, :]


def _ranks(score: torch.Tensor) -> torch.Tensor:
    """Rank of every entry in its last dimension under a stable ascending sort."""
    order = torch.sort(score, dim=-1, stable=True).indices
    ar = torch.arange(score.shape[-1], device=score.device).expand_as(order)
    return torch.empty_like(order).scatter_(-1, order, ar)


@torch.no_grad()
def row_mask(score: torch.Tensor, total_zeros: int) -> torch.Tensor:
    """Keep mask with exactly total_zeros pruned entries, split across rows as k / k+1."""
    assert score.dim() == 2 and bool(torch.isfinite(score).all())
    out, inp = score.shape
    counts = zeros_per_row(total_zeros, out).to(score.device)
    assert int(counts.max()) <= inp
    keep = torch.empty(score.shape, dtype=torch.bool, device=score.device)
    for a in range(0, out, _ROW_CHUNK):
        keep[a:a + _ROW_CHUNK] = _ranks(score[a:a + _ROW_CHUNK]) >= counts[a:a + _ROW_CHUNK, None]
    assert int((~keep).sum()) == total_zeros
    return keep


@torch.no_grad()
def mask_2of4(score: torch.Tensor) -> torch.Tensor:
    assert score.dim() == 2 and score.shape[1] % 4 == 0 and bool(torch.isfinite(score).all())
    out, inp = score.shape
    keep = torch.empty(score.shape, dtype=torch.bool, device=score.device)
    for a in range(0, out, _ROW_CHUNK):
        s = score[a:a + _ROW_CHUNK].reshape(-1, inp // 4, 4)
        keep[a:a + _ROW_CHUNK] = (_ranks(s) >= 2).reshape(-1, inp)
    assert int((~keep).sum()) == out * inp // 2
    return keep


def disagreement(keep_a: torch.Tensor, keep_b: torch.Tensor) -> float:
    """Fraction of weights pruned by exactly one of the two masks."""
    return float((keep_a ^ keep_b).sum()) / keep_a.numel()


def coef_variation(x: torch.Tensor) -> float:
    x = x.double()
    return float(x.std(unbiased=False) / x.mean())


def quantiles(t: torch.Tensor, qs=QUANTILES) -> list:
    """Linear-interpolation quantiles (numpy 'linear') of all entries."""
    v = torch.sort(t.detach().reshape(-1).double()).values
    n = v.numel()
    out = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(float(v[lo] + (v[hi] - v[lo]) * frac))
    return out


def log10_range(t: torch.Tensor) -> tuple:
    pos = t[t > 0]
    if pos.numel() == 0:
        return None
    lv = torch.log10(pos.double())
    return float(lv.min()), float(lv.max())


def log10_edges(ranges) -> list:
    """N_BINS + 1 equally spaced log10 edges spanning the union of (min, max) ranges."""
    ranges = [r for r in ranges if r is not None]
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    if hi == lo:
        hi = lo + 1.0
    return torch.linspace(lo, hi, N_BINS + 1, dtype=torch.float64).tolist()


def distribution(t: torch.Tensor, edges) -> dict:
    """Quantiles of all entries, count of zero scores and a histogram of log10 of positive scores."""
    flat = t.detach().reshape(-1)
    pos = flat[flat > 0]
    hist = [0] * N_BINS
    if pos.numel():
        lv = torch.log10(pos.double())
        e = torch.tensor(edges, dtype=torch.float64, device=lv.device)
        assert float(lv.min()) >= edges[0] and float(lv.max()) <= edges[-1]
        idx = torch.searchsorted(e[1:-1].contiguous(), lv, right=True)
        hist = torch.bincount(idx, minlength=N_BINS).tolist()
    return {"n": int(flat.numel()), "n_zero": int((flat == 0).sum()), "quantiles": quantiles(flat),
            "mean": float(flat.double().mean()), "log10_hist": hist}
