import math

import numpy as np
import pytest

from common.bootstrap import paired_acc_diff, paired_ppl_diff

KEYS = {"point", "ci", "percentiles", "n_resamples", "seed", "unit", "n_units", "method", "statistic"}


def _data(seed=0, W=40):
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 50, size=W)
    nll_a = n * rng.uniform(2.0, 3.0, size=W)
    nll_b = nll_a + n * rng.normal(0.05, 0.05, size=W)
    return nll_a, n, nll_b


def test_identical_inputs_give_zero_interval():
    nll, n, _ = _data()
    r = paired_ppl_diff(nll, n, nll, n, n_resamples=2000)
    assert r["point"] == 0.0 and r["ci"] == [0.0, 0.0] and r["nll_ci"] == [0.0, 0.0]
    c = np.array([1, 0, 1, 1, 0], dtype=bool)
    a = paired_acc_diff(c, c, n_resamples=2000)
    assert a["point"] == 0.0 and a["ci"] == [0.0, 0.0]


def test_defaults_and_keys():
    nll_a, n, nll_b = _data()
    r = paired_ppl_diff(nll_a, n, nll_b, n)
    assert KEYS | {"nll_point", "nll_ci"} <= set(r)
    assert r["n_resamples"] == 10_000 and r["seed"] == 6304 and r["unit"] == "window" and r["n_units"] == 40
    assert r["ci"][0] <= r["point"] <= r["ci"][1]
    a = paired_acc_diff([1, 0, 1], [1, 1, 1])
    assert KEYS <= set(a) and a["unit"] == "example" and a["n_resamples"] == 10_000 and a["seed"] == 6304


def test_fixed_seed_reproducible_and_seed_matters():
    nll_a, n, nll_b = _data()
    r1 = paired_ppl_diff(nll_a, n, nll_b, n, n_resamples=3000)
    r2 = paired_ppl_diff(nll_a, n, nll_b, n, n_resamples=3000)
    r3 = paired_ppl_diff(nll_a, n, nll_b, n, n_resamples=3000, seed=1)
    assert r1 == r2 and r1["ci"] != r3["ci"]


def test_pooled_formula_by_hand_three_windows():
    nll_a = np.array([10.0, 30.0, 5.0])
    nll_b = np.array([12.0, 29.0, 7.0])
    n = np.array([4, 10, 2])
    r = paired_ppl_diff(nll_a, n, nll_b, n, n_resamples=50, seed=7)
    # Point estimate: pooled PPL of each condition over all windows.
    assert r["point"] == pytest.approx(math.exp(48 / 16) - math.exp(45 / 16), abs=1e-12)
    assert r["nll_point"] == pytest.approx(48 / 16 - 45 / 16, abs=1e-12)
    # Resamples: the same window indices for both conditions, pooled sums before exponentiating.
    idx = np.random.default_rng(7).integers(0, 3, size=(50, 3))
    diffs = []
    for row in idx:
        sa = sum(nll_a[i] for i in row)
        sb = sum(nll_b[i] for i in row)
        sn = sum(n[i] for i in row)
        diffs.append(math.exp(sb / sn) - math.exp(sa / sn))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    assert r["ci"] == pytest.approx([lo, hi], abs=1e-12)


def test_acc_by_hand():
    a = np.array([1, 0, 0, 1])
    b = np.array([1, 1, 0, 1])
    r = paired_acc_diff(a, b, n_resamples=20, seed=3)
    assert r["point"] == pytest.approx(0.25)
    idx = np.random.default_rng(3).integers(0, 4, size=(20, 4))
    d = [b[i].mean() - a[i].mean() for i in idx]
    assert r["ci"] == pytest.approx(list(np.percentile(d, [2.5, 97.5])), abs=1e-12)


def test_mismatched_tokens_refused():
    with pytest.raises(AssertionError):
        paired_ppl_diff([1.0, 2.0], [1, 2], [1.0, 2.0], [1, 3])
