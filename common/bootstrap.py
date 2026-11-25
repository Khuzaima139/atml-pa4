"""Paired percentile bootstrap for pooled perplexity and accuracy differences (B - A)."""
import math

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 6304
PERCENTILES = (2.5, 97.5)
_CHUNK = 1_000  # draws generated per call to the generator; fixed so results are reproducible
METHOD = "paired percentile bootstrap; one index draw per resample applied to both conditions; numpy linear percentiles"


def _draws(n_units: int, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_resamples:
        k = min(_CHUNK, n_resamples - done)
        yield rng.integers(0, n_units, size=(k, n_units))
        done += k


def _interval(values):
    lo, hi = np.percentile(values, PERCENTILES)
    return [float(lo), float(hi)]


def paired_ppl_diff(nll_a, n_a, nll_b, n_b, n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                    unit: str = "window") -> dict:
    """Pooled PPL difference over resampled whole windows: exp(sum nll_B / sum n) - exp(sum nll_A / sum n).

    nll_* are per-window NLL sums and n_* per-window scored-token counts, aligned by window.
    """
    nll_a, nll_b = np.asarray(nll_a, dtype=np.float64), np.asarray(nll_b, dtype=np.float64)
    n_a, n_b = np.asarray(n_a, dtype=np.int64), np.asarray(n_b, dtype=np.int64)
    assert nll_a.shape == nll_b.shape == n_a.shape == n_b.shape and nll_a.ndim == 1
    assert np.array_equal(n_a, n_b), "conditions must score the same tokens"
    W = nll_a.size

    def ppl(nll, n):
        return math.exp(nll.sum() / n.sum())

    point = ppl(nll_b, n_b) - ppl(nll_a, n_a)
    point_nll = nll_b.sum() / n_b.sum() - nll_a.sum() / n_a.sum()
    d_ppl, d_nll = [], []
    for idx in _draws(W, n_resamples, seed):
        na = n_a[idx].sum(axis=1)
        ma = nll_a[idx].sum(axis=1) / na
        mb = nll_b[idx].sum(axis=1) / n_b[idx].sum(axis=1)
        d_ppl.append(np.exp(mb) - np.exp(ma))
        d_nll.append(mb - ma)
    d_ppl, d_nll = np.concatenate(d_ppl), np.concatenate(d_nll)
    return {"statistic": "pooled perplexity difference (B - A)", "point": float(point), "ci": _interval(d_ppl),
            "nll_point": float(point_nll), "nll_ci": _interval(d_nll), "percentiles": list(PERCENTILES),
            "n_resamples": n_resamples, "seed": seed, "unit": unit, "n_units": W, "method": METHOD}


def paired_acc_diff(correct_a, correct_b, n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                    unit: str = "example") -> dict:
    """Accuracy difference (B - A) over resampled examples."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1
    n = a.size
    point = b.mean() - a.mean()
    diffs = [b[idx].mean(axis=1) - a[idx].mean(axis=1) for idx in _draws(n, n_resamples, seed)]
    diffs = np.concatenate(diffs)
    return {"statistic": "accuracy difference (B - A)", "point": float(point), "ci": _interval(diffs),
            "percentiles": list(PERCENTILES), "n_resamples": n_resamples, "seed": seed, "unit": unit,
            "n_units": n, "method": METHOD}
