"""Block-level allocation rules for nonuniform pruning (P3) and mixed-precision quantization (Q3).

Matrices are described by dicts {"layer", "name", "qualname", "shape"} in enumeration order (layer, then
q, k, v, o, gate, up, down). Every projection of a block receives the block's sparsity (P3) or bit width (Q3).

P3: u = min-max normalised sensitivity over blocks (u = 0.5 for every block when all values are equal);
raw s_l = S + lambda - 2 * lambda * u_l (more sensitive blocks are pruned less); one shift c is found by bisection
so that the mean over blocks of clip(s_l + c, P3_S_MIN, P3_S_MAX) equals S; each matrix gets the real target
s_l * numel, floored, and single weights are added by largest remainder (ties: lower block sensitivity, then lower
block, then projection order) until the total equals the P2 total at S exactly. A candidate whose per-matrix counts
equal P2's is flagged collapsed_to_uniform.

Q3: exact multiple-choice knapsack over blocks with bits in Q3_BITS: minimise sum_l sensitivity(l, b_l) (exact
rational arithmetic on the float sensitivities) subject to total eligible packed bytes (task1.storage) <= budget;
dynamic programming over integer byte totals; ties: fewer bytes, then the lexicographically smaller bit vector.
"""
import itertools
import math
from fractions import Fraction

from . import config as C
from . import prune as P
from . import storage

P3_RULE = (
    "u = min-max normalised block sensitivity (all equal -> 0.5); s_l = S + lambda - 2*lambda*u_l; shift c by "
    "bisection so that mean_l clip(s_l + c, 0.10, 0.85) = S; per matrix zeros = floor(s_l * numel), then +1 by "
    "largest remainder (ties: lower block sensitivity, then lower block, then projection order) to the exact P2 "
    "total at S; ranking rule: "
    "sequential Wanda per output row (unchanged)"
)
Q3_RULE = (
    "exact multiple-choice knapsack over blocks, bits in {2,3,4,8}: minimise sum_l sensitivity(l, b_l) (exact "
    "rational sums) subject to eligible packed bytes (task1.storage) <= the Q2 uniform budget; dynamic programming "
    "over integer byte totals; ties: fewer bytes, then lexicographically smaller bit vector by block index"
)


def _numel(shape) -> int:
    return int(shape[0]) * int(shape[1])


def n_blocks(matrices) -> int:
    layers = sorted({m["layer"] for m in matrices})
    assert layers == list(range(len(layers))), layers
    for li in layers:
        assert [m["name"] for m in matrices if m["layer"] == li] == list(C.ELIGIBLE), li
    return len(layers)


def normalise(sens) -> list:
    """Min-max normalisation to [0, 1]; every value 0.5 when all are equal."""
    v = [float(x) for x in sens]
    assert v and all(math.isfinite(x) for x in v), v
    lo, hi = min(v), max(v)
    if hi == lo:
        return [0.5] * len(v)
    return [(x - lo) / (hi - lo) for x in v]


def _clip(x: float) -> float:
    return min(max(x, C.P3_S_MIN), C.P3_S_MAX)


def _mean_clipped(raw, c: float) -> float:
    return math.fsum(_clip(s + c) for s in raw) / len(raw)


def find_shift(raw, S: float) -> tuple:
    """Shift c with mean_l clip(raw_l + c) = S (bisection); returns (c, iterations, residual)."""
    assert C.P3_S_MIN <= S <= C.P3_S_MAX, S
    if _mean_clipped(raw, 0.0) == S:
        return 0.0, 0, 0.0
    lo = C.P3_S_MIN - max(raw)   # every clipped value at the lower bound
    hi = C.P3_S_MAX - min(raw)   # every clipped value at the upper bound
    assert _mean_clipped(raw, lo) <= S <= _mean_clipped(raw, hi)
    it = 0
    for it in range(1, C.P3_BISECT_MAX_ITERS + 1):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:        # bracket at adjacent floats
            break
        if _mean_clipped(raw, mid) < S:
            lo = mid
        else:
            hi = mid
    c = hi if abs(_mean_clipped(raw, hi) - S) <= abs(_mean_clipped(raw, lo) - S) else lo
    resid = _mean_clipped(raw, c) - S
    assert abs(resid) <= C.P3_BISECT_TOL, (resid, it)
    return c, it, resid


def p2_zeros(matrices, S: float) -> dict:
    """Uniform per-matrix zeros of P2 at sparsity S."""
    return {m["qualname"]: P.target_zeros(tuple(m["shape"]), S) for m in matrices}


def block_sparsity(matrices, zeros: dict, L: int) -> list:
    num = [0] * L
    den = [0] * L
    for m in matrices:
        num[m["layer"]] += int(zeros[m["qualname"]])
        den[m["layer"]] += _numel(m["shape"])
    return [a / b for a, b in zip(num, den)]


def check_p3_zeros(matrices, zeros: dict, S: float) -> dict:
    """Assert the exact P2 total and the per-block bounds (within one weight per matrix); returns totals."""
    assert list(zeros) == [m["qualname"] for m in matrices], "matrix set or order differs"
    total = sum(int(z) for z in zeros.values())
    target = sum(p2_zeros(matrices, S).values())
    assert total == target, (total, target)
    L = n_blocks(matrices)
    for li in range(L):
        ms = [m for m in matrices if m["layer"] == li]
        z = sum(int(zeros[m["qualname"]]) for m in ms)
        n = sum(_numel(m["shape"]) for m in ms)
        slack = len(ms)
        assert C.P3_S_MIN * n - slack <= z <= C.P3_S_MAX * n + slack, (li, z / n)
    for m in matrices:
        z, n = int(zeros[m["qualname"]]), _numel(m["shape"])
        assert 0 <= z <= n and C.P3_S_MIN * n - 1 <= z <= C.P3_S_MAX * n + 1, (m["qualname"], z / n)
    return {"total_zeros": total, "p2_total_zeros": target,
            "eligible_weights": sum(_numel(m["shape"]) for m in matrices)}


def largest_remainder(targets, total: int, tie_keys=None) -> list:
    """Floor every real target, then add one to the largest remainders until the sum is total.

    Equal remainders go to the lower tie key (P3: the block's sensitivity), then the lower index (enumeration
    order = block, then projection order).
    """
    zeros = [int(math.floor(t)) for t in targets]
    deficit = int(total) - sum(zeros)
    assert 0 <= deficit <= len(targets), (deficit, len(targets))
    keys = [0.0] * len(targets) if tie_keys is None else [float(k) for k in tie_keys]
    assert len(keys) == len(targets)
    order = sorted(range(len(targets)), key=lambda i: (-(targets[i] - zeros[i]), keys[i], i))
    for i in order[:deficit]:
        zeros[i] += 1
    assert sum(zeros) == total
    return zeros


def p3_allocation(sens, S: float, lam: float, matrices) -> dict:
    """P3 per-block sparsities and exact per-matrix zero counts for one sensitivity vector and lambda.

    Blocks must hold equal numbers of eligible weights, so the mean over blocks is the weight-weighted mean.
    """
    L = n_blocks(matrices)
    assert len(sens) == L, (len(sens), L)
    sizes = {sum(_numel(m["shape"]) for m in matrices if m["layer"] == li) for li in range(L)}
    assert len(sizes) == 1, f"blocks of unequal size: {sorted(sizes)}"
    assert lam >= 0
    u = normalise(sens)
    raw = [S + lam - 2.0 * lam * x for x in u]
    c, iters, resid = find_shift(raw, S)
    s_block = [_clip(r + c) for r in raw]
    targets = [s_block[m["layer"]] * _numel(m["shape"]) for m in matrices]
    total = sum(p2_zeros(matrices, S).values())
    zeros = largest_remainder(targets, total, tie_keys=[float(sens[m["layer"]]) for m in matrices])
    deficit = total - sum(int(math.floor(t)) for t in targets)
    zmap = {m["qualname"]: z for m, z in zip(matrices, zeros)}
    totals = check_p3_zeros(matrices, zmap, S)
    collapsed = zmap == p2_zeros(matrices, S)
    return {"rule": P3_RULE, "sparsity": S, "lambda": lam, "sensitivity": [float(x) for x in sens], "u": u,
            "raw": raw, "shift": c, "bisect_iterations": iters, "bisect_residual": resid,
            "block_sparsity_real": s_block, "block_sparsity_achieved": block_sparsity(matrices, zmap, L),
            "n_largest_remainder_added": deficit, "zeros": zmap, "collapsed_to_uniform": collapsed, **totals}


# ---------------------------------------------------------------------------------------------------------------
def block_costs(matrices, bits_options=C.Q3_BITS) -> list:
    """Eligible packed bytes of each block at each bit width: [{b: bytes}]."""
    L = n_blocks(matrices)
    return [{b: storage.packed_quant_bytes([tuple(m["shape"]) for m in matrices if m["layer"] == li], b)
             for b in bits_options} for li in range(L)]


def uniform_budget(matrices, bits: int) -> int:
    return storage.packed_quant_bytes([tuple(m["shape"]) for m in matrices], bits)


def _frac(x) -> Fraction:
    x = float(x)
    assert math.isfinite(x), x
    return Fraction(x)


def q3_knapsack(sens, costs, budget: int, bits_options=C.Q3_BITS) -> dict:
    """Exact DP. sens, costs: per block {b: value}; returns bits, bytes and objective (min sum of sensitivities)."""
    L = len(costs)
    assert len(sens) == L and L > 0
    opts = sorted(int(b) for b in bits_options)
    S = [{b: _frac(sens[li][b]) for b in opts} for li in range(L)]
    K = [{b: int(costs[li][b]) for b in opts} for li in range(L)]
    min_rest = [0] * (L + 1)
    for li in range(L - 1, -1, -1):
        min_rest[li] = min_rest[li + 1] + min(K[li].values())
    assert min_rest[0] <= budget, "budget below the all-minimum-bits cost"
    states = {0: (Fraction(0), ())}       # prefix bytes -> (objective, bit vector)
    n_states = []
    for li in range(L):
        nxt = {}
        for used, (obj, vec) in states.items():
            for b in opts:
                nb = used + K[li][b]
                if nb + min_rest[li + 1] > budget:
                    continue
                cand = (obj + S[li][b], vec + (b,))
                cur = nxt.get(nb)
                if cur is None or cand < cur:
                    nxt[nb] = cand
        states = nxt
        n_states.append(len(states))
    best_bytes, (best_obj, best_vec) = min(states.items(), key=lambda kv: (kv[1][0], kv[0], kv[1][1]))
    return {"rule": Q3_RULE, "bits": list(best_vec), "bytes": best_bytes, "budget": int(budget),
            "objective": float(best_obj), "objective_exact": f"{best_obj.numerator}/{best_obj.denominator}",
            "dp_states_per_block": n_states}


def q3_brute_force(sens, costs, budget: int, bits_options=C.Q3_BITS) -> dict:
    """Exhaustive search with the same objective and tie rule (tests)."""
    L = len(costs)
    opts = sorted(int(b) for b in bits_options)
    best = None
    for vec in itertools.product(opts, repeat=L):
        nb = sum(int(costs[li][b]) for li, b in enumerate(vec))
        if nb > budget:
            continue
        obj = sum((_frac(sens[li][b]) for li, b in enumerate(vec)), Fraction(0))
        key = (obj, nb, vec)
        if best is None or key < best:
            best = key
    obj, nb, vec = best
    return {"bits": list(vec), "bytes": nb, "objective": float(obj)}


def q3_allocation(sens, matrices, ref_bits: int) -> dict:
    """Q3 bits per block under the uniform-ref_bits budget, with per-matrix bits and the collapse flag."""
    costs = block_costs(matrices)
    budget = uniform_budget(matrices, ref_bits)
    res = q3_knapsack(sens, costs, budget)
    assert res["bytes"] <= budget
    bits = res["bits"]
    per_matrix = {m["qualname"]: bits[m["layer"]] for m in matrices}
    assert storage.packed_quant_bytes([tuple(m["shape"]) for m in matrices], list(per_matrix.values())) == res["bytes"]
    return {**res, "ref_bits": ref_bits, "collapsed_to_uniform": all(b == ref_bits for b in bits),
            "block_bytes": [costs[li][b] for li, b in enumerate(bits)], "per_matrix_bits": per_matrix,
            "sensitivity": [{str(b): float(sens[li][b]) for b in C.Q3_BITS} for li in range(len(bits))]}


def uniform_q3(matrices, ref_bits: int) -> dict:
    """The uniform allocation (= Q2 at ref_bits), the reference of every Q3 candidate set."""
    L = n_blocks(matrices)
    costs = block_costs(matrices)
    bits = [ref_bits] * L
    nb = sum(costs[li][ref_bits] for li in range(L))
    assert nb == uniform_budget(matrices, ref_bits)
    return {"rule": "uniform (Q2 reference)", "bits": bits, "bytes": nb, "budget": nb, "ref_bits": ref_bits,
            "collapsed_to_uniform": True, "block_bytes": [costs[li][ref_bits] for li in range(L)],
            "per_matrix_bits": {m["qualname"]: ref_bits for m in matrices}}


def check_q3_bits(matrices, per_matrix_bits: dict, ref_bits: int) -> dict:
    """Assert allowed bits, one width per block, and bytes within the uniform budget; returns totals."""
    assert list(per_matrix_bits) == [m["qualname"] for m in matrices], "matrix set or order differs"
    L = n_blocks(matrices)
    for li in range(L):
        bs = {int(per_matrix_bits[m["qualname"]]) for m in matrices if m["layer"] == li}
        assert len(bs) == 1 and bs <= set(C.Q3_BITS), (li, bs)
    shapes = [tuple(m["shape"]) for m in matrices]
    nb = storage.packed_quant_bytes(shapes, [int(b) for b in per_matrix_bits.values()])
    budget = uniform_budget(matrices, ref_bits)
    assert nb <= budget, (nb, budget)
    return {"eligible_packed_bytes": nb, "budget_bytes": budget, "ref_bits": ref_bits}
