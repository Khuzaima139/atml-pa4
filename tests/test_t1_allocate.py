"""P3 and Q3 allocation rules: exact totals, bounds, reference equality, monotonicity, ties and the exact knapsack."""
import random

import pytest

from task1 import allocate as A
from task1 import config as C
from task1 import storage

SMALL_SHAPES = {"q_proj": (256, 256), "k_proj": (128, 256), "v_proj": (128, 256), "o_proj": (256, 256),
                "gate_proj": (512, 256), "up_proj": (512, 256), "down_proj": (256, 512)}


def _mats(n_blocks, shapes):
    return [{"layer": li, "name": n, "qualname": f"model.layers.{li}.{n}", "shape": list(shapes[n])}
            for li in range(n_blocks) for n in C.ELIGIBLE]


FULL = _mats(C.N_LAYERS, C.FULL_MODEL_SHAPES)
SMALL4 = _mats(4, SMALL_SHAPES)


def _sens(n, seed):
    rnd = random.Random(seed)
    return [rnd.random() ** 3 for _ in range(n)]


def test_constants():
    assert C.P3_LAMBDAS == (0.04, 0.08, 0.12, 0.16, 0.20) and C.P3_REFERENCE_LAMBDA == 0.0
    assert (C.P3_S_MIN, C.P3_S_MAX, C.OWL_M) == (0.10, 0.85, 5.0)
    assert C.PROBE_SPARSITIES == (0.50, 0.75) and C.PROBE_BITS == (2, 3, 4, 8)
    for t, b in (("int4", 4), ("int3", 3)):
        assert A.uniform_budget(FULL, b) == C.Q3_FULL_MODEL_BUDGET_BYTES[t]
        assert C.Q3_TARGETS[t]["bits"] == b
    assert C.Q3_FULL_MODEL_BUDGET_BYTES == {"int4": 680_687_616, "int3": 515_633_664}


@pytest.mark.parametrize("mats", [FULL, SMALL4], ids=["full", "small4"])
@pytest.mark.parametrize("S", C.PROBE_SPARSITIES)
def test_p3_exact_total_bounds_and_reference(mats, S):
    L = A.n_blocks(mats)
    p2 = A.p2_zeros(mats, S)
    for seed in range(5):
        sens = _sens(L, seed)
        for lam in C.P3_LAMBDAS + (0.6,):  # 0.6 forces clipping at both bounds
            r = A.p3_allocation(sens, S, lam, mats)
            assert r["total_zeros"] == sum(p2.values()) == r["p2_total_zeros"]
            assert sum(r["zeros"].values()) == sum(p2.values())
            for li, s in enumerate(r["block_sparsity_achieved"]):
                n = sum(m["shape"][0] * m["shape"][1] for m in mats if m["layer"] == li)
                assert C.P3_S_MIN - 7 / n <= s <= C.P3_S_MAX + 7 / n
            assert all(C.P3_S_MIN <= s <= C.P3_S_MAX for s in r["block_sparsity_real"])
            assert abs(r["bisect_residual"]) <= C.P3_BISECT_TOL
        ref = A.p3_allocation(sens, S, 0.0, mats)
        assert ref["zeros"] == p2 and ref["shift"] == 0.0


def test_p3_monotone():
    for seed in range(10):
        sens = _sens(C.N_LAYERS, 100 + seed)
        for S in C.PROBE_SPARSITIES:
            for lam in C.P3_LAMBDAS:
                r = A.p3_allocation(sens, S, lam, FULL)
                z = r["zeros"]
                for a in range(C.N_LAYERS):
                    for b in range(C.N_LAYERS):
                        if sens[a] > sens[b]:
                            sa, sb = r["block_sparsity_real"][a], r["block_sparsity_real"][b]
                            assert sa <= sb
                            for n in C.ELIGIBLE:  # same shape: never more zeros for the more sensitive block
                                za, zb = z[f"model.layers.{a}.{n}"], z[f"model.layers.{b}.{n}"]
                                if sa < sb:
                                    assert za <= zb
                                else:  # both clipped to the same bound: the fixed tie rule (lower block first)
                                    assert sa == sb and abs(za - zb) <= 1


def test_p3_bisection():
    raw = [0.0, 0.3, 0.9, 1.4]
    for S in (0.2, 0.5, 0.75):
        c, it, resid = A.find_shift(raw, S)
        mean = sum(min(max(x + c, C.P3_S_MIN), C.P3_S_MAX) for x in raw) / len(raw)
        assert abs(mean - S) <= C.P3_BISECT_TOL and 0 < it <= C.P3_BISECT_MAX_ITERS
    assert A.find_shift([0.5] * 4, 0.5) == (0.0, 0, 0.0)
    assert A.normalise([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]
    assert A.normalise([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]


def test_p3_largest_remainder_ties():
    # 4 blocks of 4x4 matrices at S = 0.5: s = 0.5 -+ 1/32 gives remainders of exactly 0.5 in every matrix;
    # 14 extra weights go to blocks 0 and 1 (lower block first), all their projections.
    mats = _mats(4, {n: (4, 4) for n in C.ELIGIBLE})
    for _ in range(2):
        r = A.p3_allocation([1.0, 0.0, 0.0, 1.0], 0.5, 1 / 32, mats)
        assert r["block_sparsity_real"] == [0.46875, 0.53125, 0.53125, 0.46875]
        assert r["n_largest_remainder_added"] == 14
        want = {0: 8, 1: 9, 2: 8, 3: 7}
        assert r["zeros"] == {m["qualname"]: want[m["layer"]] for m in mats}
    # Partial assignment: equal remainders go to the lower enumeration index (block, then projection order).
    assert A.largest_remainder([1.5, 2.5, 3.5], 8) == [2, 3, 3]
    assert A.largest_remainder([1.5, 2.5, 3.5, 0.9], 8) == [2, 2, 3, 1]
    assert A.largest_remainder([2.0, 3.0], 5) == [2, 3]
    with pytest.raises(AssertionError):
        A.p3_allocation([0.0, 1.0], 0.5, 0.1, _mats(1, {n: (4, 4) for n in C.ELIGIBLE}) + [
            {"layer": 1, "name": n, "qualname": f"b1.{n}", "shape": [8, 8]} for n in C.ELIGIBLE])


def test_q3_dp_equals_brute_force():
    costs = A.block_costs(SMALL4)
    for ref_bits in (4, 3):
        budget = A.uniform_budget(SMALL4, ref_bits)
        for seed in range(60):
            rnd = random.Random(seed)
            if seed % 3 == 0:   # small integers: many exact ties
                sens = [{b: float(rnd.randint(0, 3)) for b in C.Q3_BITS} for _ in range(4)]
            else:
                sens = [{b: rnd.random() * (9 - b) ** 2 - 0.2 for b in C.Q3_BITS} for _ in range(4)]
            dp = A.q3_knapsack(sens, costs, budget)
            bf = A.q3_brute_force(sens, costs, budget)
            assert (dp["bits"], dp["bytes"]) == (bf["bits"], bf["bytes"]), seed
            assert dp["bytes"] <= budget
            al = A.q3_allocation(sens, SMALL4, ref_bits)
            assert al["bits"] == dp["bits"] and al["bytes"] <= al["budget"] == budget
            shapes = [tuple(m["shape"]) for m in SMALL4]
            assert storage.packed_quant_bytes(shapes, list(al["per_matrix_bits"].values())) == al["bytes"]


def test_q3_full_model_budget_and_ties():
    rnd = random.Random(7)
    sens = [{b: rnd.random() * (9 - b) for b in C.Q3_BITS} for _ in range(C.N_LAYERS)]
    for t, tb in C.Q3_TARGETS.items():
        al = A.q3_allocation(sens, FULL, tb["bits"])
        assert al["bytes"] <= C.Q3_FULL_MODEL_BUDGET_BYTES[t] == al["budget"]
        A.check_q3_bits(FULL, al["per_matrix_bits"], tb["bits"])
    # Equal sensitivity at every width: the fewest bytes win (all 2-bit).
    flat = [{b: 1.0 for b in C.Q3_BITS} for _ in range(4)]
    assert A.q3_allocation(flat, SMALL4, 4)["bits"] == [2, 2, 2, 2]
    # Equal objective, equal bytes: the lexicographically smaller bit vector wins.
    sym = [{2: 0.0, 3: 0.0, 4: 0.0, 8: 0.0} for _ in range(4)]
    sym[0] = {2: 5.0, 3: 0.0, 4: 0.0, 8: 0.0}
    sym[1] = {2: 5.0, 3: 0.0, 4: 0.0, 8: 0.0}
    r = A.q3_allocation(sym, SMALL4, 3)
    assert r["bits"] == [3, 3, 2, 2]


def test_q3_collapsed_to_uniform():
    sens = [{2: 100.0, 3: 10.0, 4: 1.0, 8: 0.99} for _ in range(4)]
    for ref_bits in (4, 3):
        al = A.q3_allocation(sens, SMALL4, ref_bits)
        assert al["bits"] == [ref_bits] * 4 and al["collapsed_to_uniform"] is True
    al = A.q3_allocation([{2: 100.0, 3: 10.0, 4: 1.0, 8: 0.0}, *sens[1:]], SMALL4, 4)
    assert al["collapsed_to_uniform"] is True or al["bits"] != [4] * 4
    u = A.uniform_q3(SMALL4, 3)
    assert u["bits"] == [3] * 4 and u["bytes"] == A.uniform_budget(SMALL4, 3)


def test_validators_refuse():
    S = 0.5
    z = dict(A.p2_zeros(SMALL4, S))
    first = next(iter(z))
    z[first] += 1
    with pytest.raises(AssertionError):
        A.check_p3_zeros(SMALL4, z, S)
    bits = {m["qualname"]: 4 for m in SMALL4}
    bits["model.layers.0.q_proj"] = 8
    with pytest.raises(AssertionError):  # two widths in one block
        A.check_q3_bits(SMALL4, bits, 4)
    with pytest.raises(AssertionError):  # over budget
        A.check_q3_bits(SMALL4, {m["qualname"]: 8 if m["layer"] == 0 else 4 for m in SMALL4}, 4)
