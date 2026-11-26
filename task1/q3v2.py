"""Q3 design v2: fold-group allocation units, exact chunked packed costs, exact knapsack, unit folds and measures.

Units (config.Q3V2_UNITS): per block attn_in (q, k, v), attn_out (o), mlp_in (gate, up), mlp_out (down), each one
AWQ scale group (qkv, o, gateup, down). Unit order: block, then attn_in, attn_out, mlp_in, mlp_out.

Costs: every matrix of a unit holds G groups of GROUP_SIZE input columns; its packed bytes (task1.storage) are
G * (GROUP_SIZE*b + 16 + b) / 8 when G*b is a multiple of 8 (codes, FP16 scale, zero-point per group). With
g = gcd of the units' group counts, a unit with k*g groups costs k * (GROUP_SIZE*b + 16 + b) units of g/8 bytes;
chunking() computes g and asserts this against storage.packed_quant_bytes for every unit and bit width.

Knapsack: minimise the exact sum of sensitivities (every float is a dyadic rational; all are scaled to integers over
one common power-of-two denominator) subject to cost <= floor(8 * budget_bytes / g). Dynamic programming from the
last unit to the first over integer suffix costs, keeping per cost the minimum (objective, bit of this unit); the
result is the minimum of (objective, bytes, bit vector) over all feasible allocations, i.e. ties go to fewer bytes,
then to the lexicographically smaller bit vector by unit order.
"""
import itertools
import math
from fractions import Fraction

from . import config as C
from . import storage

UNIT_NAMES = tuple(u for u, _, _ in C.Q3V2_UNITS)
UNIT_GROUP = {u: g for u, g, _ in C.Q3V2_UNITS}
UNIT_PROJS = {u: tuple(p) for u, _, p in C.Q3V2_UNITS}
PROJ_UNIT = {p: u for u, _, ps in C.Q3V2_UNITS for p in ps}
GROUP_UNIT = {g: u for u, g, _ in C.Q3V2_UNITS}
# Non-eligible or other-unit tensors (block-relative names) that a unit's fold rewrites besides its projections.
FOLD_PARTNERS = {
    "attn_in": ("input_layernorm.weight",),
    "attn_out": ("self_attn.v_proj.weight", "self_attn.v_proj.bias"),
    "mlp_in": ("post_attention_layernorm.weight",),
    "mlp_out": ("mlp.up_proj.weight",),
}
assert set(PROJ_UNIT) == set(C.ELIGIBLE) and len(PROJ_UNIT) == len(C.ELIGIBLE)


def unit_id(layer: int, unit: str) -> str:
    return f"{layer}.{unit}"


def units_of(matrices) -> list:
    """Units in order with their matrices and quantization group counts."""
    from .allocate import n_blocks

    L = n_blocks(matrices)
    by = {(m["layer"], m["name"]): m for m in matrices}
    out = []
    for li in range(L):
        for u, g, projs in C.Q3V2_UNITS:
            ms = [by[(li, n)] for n in projs]
            out.append({"index": len(out), "id": unit_id(li, u), "layer": li, "unit": u, "scale_group": g,
                        "projections": list(projs), "qualnames": [m["qualname"] for m in ms],
                        "shapes": [list(m["shape"]) for m in ms],
                        "groups": sum(storage.groups(tuple(m["shape"])) for m in ms)})
    return out


def eighth_bytes_per_group(bits: int, group_size: int = C.GROUP_SIZE) -> int:
    """Codes (group_size * b bits), FP16 scale (16 bits) and zero-point (b bits) of one group, in bits = 1/8 bytes."""
    return group_size * int(bits) + 16 + int(bits)


def chunking(units, bits_options=C.Q3V2_BITS, group_size: int = C.GROUP_SIZE) -> dict:
    """Chunk size g (gcd of unit group counts), chunks per unit and integer unit costs in g/8-byte units."""
    g = 0
    for u in units:
        g = math.gcd(g, int(u["groups"]))
    assert g > 0
    chunks = [int(u["groups"]) // g for u in units]
    costs = []
    for u, k in zip(units, chunks):
        row = {}
        for b in bits_options:
            c = k * eighth_bytes_per_group(b, group_size)
            exact = storage.packed_quant_bytes([tuple(s) for s in u["shapes"]], int(b), group_size)
            assert c * g == 8 * exact, (u["id"], b, c, g, exact)
            row[int(b)] = c
        costs.append(row)
    return {"chunk_groups": g, "unit_chunks": chunks, "costs": costs,
            "eighth_bytes_per_group": {str(b): eighth_bytes_per_group(b, group_size) for b in bits_options},
            "cost_unit": f"{g}/8 bytes"}


def budget_units(budget_bytes: int, chunk_groups: int) -> int:
    """Largest integer cost c with c * g / 8 <= budget_bytes (exact: unit costs are integers)."""
    return (8 * int(budget_bytes)) // int(chunk_groups)


def exact_ints(values) -> tuple:
    """Floats as integers over one common power-of-two denominator: (ints, denominator)."""
    fr = []
    for v in values:
        x = float(v)
        assert math.isfinite(x), v
        fr.append(Fraction(x))
    D = max(f.denominator for f in fr)
    assert all(D % f.denominator == 0 for f in fr)
    return [f.numerator * (D // f.denominator) for f in fr], D


def knapsack(sens, costs, budget: int, bits_options=C.Q3V2_BITS) -> dict:
    """Exact DP. sens, costs: per unit {b: value}; budget in the same integer cost units."""
    N = len(costs)
    assert len(sens) == N and N > 0
    opts = sorted(int(b) for b in bits_options)
    flat, D = exact_ints([sens[i][b] for i in range(N) for b in opts])
    V = [dict(zip(opts, flat[i * len(opts):(i + 1) * len(opts)])) for i in range(N)]
    K = [{b: int(costs[i][b]) for b in opts} for i in range(N)]
    min_prefix = [0] * (N + 1)
    for i in range(N):
        min_prefix[i + 1] = min_prefix[i] + min(K[i].values())
    assert min_prefix[N] <= budget, "budget below the all-minimum-bits cost"
    levels = [None] * (N + 1)
    levels[N] = {0: (0, None, None)}          # suffix cost -> (objective, bit of this unit, next suffix cost)
    n_states = [0] * N
    for i in range(N - 1, -1, -1):
        nxt = {}
        cap = budget - min_prefix[i]
        for used, (obj, _, _) in levels[i + 1].items():
            for b in opts:
                nb = used + K[i][b]
                if nb > cap:
                    continue
                cand = (obj + V[i][b], b)
                cur = nxt.get(nb)
                if cur is None or cand < cur[:2]:
                    nxt[nb] = (cand[0], b, used)
        levels[i] = nxt
        n_states[i] = len(nxt)
    best = min(levels[0].items(), key=lambda kv: (kv[1][0], kv[0], kv[1][1]))
    cost, (obj, _, _) = best
    bits, c = [], cost
    for i in range(N):
        _, b, nxt_c = levels[i][c]
        bits.append(b)
        c = nxt_c
    assert c == 0
    exact = Fraction(obj, D)
    assert sum(K[i][b] for i, b in enumerate(bits)) == cost
    return {"bits": bits, "cost": cost, "budget": int(budget), "objective": float(exact),
            "objective_exact": f"{exact.numerator}/{exact.denominator}", "dp_states_per_unit": n_states}


def brute_force(sens, costs, budget: int, bits_options=C.Q3V2_BITS) -> dict:
    """Exhaustive search with the same objective and tie rule (tests)."""
    N = len(costs)
    opts = sorted(int(b) for b in bits_options)
    best = None
    for vec in itertools.product(opts, repeat=N):
        nb = sum(int(costs[i][b]) for i, b in enumerate(vec))
        if nb > budget:
            continue
        obj = sum((Fraction(float(sens[i][b])) for i, b in enumerate(vec)), Fraction(0))
        key = (obj, nb, vec)
        if best is None or key < best:
            best = key
    obj, nb, vec = best
    return {"bits": list(vec), "cost": nb, "objective": float(obj)}


def uniform_budget(matrices, bits: int) -> int:
    return storage.packed_quant_bytes([tuple(m["shape"]) for m in matrices], bits)


def per_matrix_bits(matrices, units, bits) -> dict:
    by_qual = {q: b for u, b in zip(units, bits) for q in u["qualnames"]}
    return {m["qualname"]: int(by_qual[m["qualname"]]) for m in matrices}


def bits_histogram(units, bits) -> dict:
    """{unit type: {bits: count}} over all blocks."""
    out = {u: {str(b): 0 for b in C.Q3V2_BITS} for u in UNIT_NAMES}
    for u, b in zip(units, bits):
        out[u["unit"]][str(int(b))] += 1
    return out


def _expect_full_model(units, ch):
    if len(units) == C.Q3V2_FULL_MODEL_N_UNITS and ch["chunk_groups"] == C.Q3V2_FULL_MODEL_CHUNK_GROUPS:
        per = {}
        for u, k in zip(units, ch["unit_chunks"]):
            per.setdefault(u["unit"], set()).add(k)
        return per == {k: {v} for k, v in C.Q3V2_FULL_MODEL_UNIT_CHUNKS.items()}
    return False


def _alloc_record(matrices, units, ch, bits, budget, ref_bits) -> dict:
    pm = per_matrix_bits(matrices, units, bits)
    nb = storage.packed_quant_bytes([tuple(m["shape"]) for m in matrices], list(pm.values()))
    cost = sum(ch["costs"][i][b] for i, b in enumerate(bits))
    assert cost * ch["chunk_groups"] == 8 * nb, (cost, nb)
    assert nb <= budget, (nb, budget)
    return {"design": C.Q3V2_DESIGN, "unit": C.Q3V2_UNIT, "bits": [int(b) for b in bits], "bytes": nb,
            "budget": int(budget), "ref_bits": ref_bits, "cost_units": cost,
            "budget_units": budget_units(budget, ch["chunk_groups"]),
            "collapsed_to_uniform": all(int(b) == ref_bits for b in bits),
            "unit_ids": [u["id"] for u in units],
            "unit_bytes": [storage.packed_quant_bytes([tuple(s) for s in u["shapes"]], int(b))
                           for u, b in zip(units, bits)],
            "bits_histogram": bits_histogram(units, bits), "per_matrix_bits": pm,
            "chunking": {"chunk_groups": ch["chunk_groups"], "unit_chunks": ch["unit_chunks"],
                         "eighth_bytes_per_group": ch["eighth_bytes_per_group"], "cost_unit": ch["cost_unit"],
                         "full_model_expected": _expect_full_model(units, ch)}}


def allocation(sens, matrices, ref_bits: int) -> dict:
    """Knapsack bits per unit under the uniform-ref_bits budget. sens: per unit {b: value} in unit order."""
    units = units_of(matrices)
    assert len(sens) == len(units), (len(sens), len(units))
    ch = chunking(units)
    budget = uniform_budget(matrices, ref_bits)
    res = knapsack(sens, ch["costs"], budget_units(budget, ch["chunk_groups"]))
    rec = _alloc_record(matrices, units, ch, res["bits"], budget, ref_bits)
    assert rec["cost_units"] == res["cost"]
    return {**rec, "rule": C.Q3V2_RULE, "objective": res["objective"], "objective_exact": res["objective_exact"],
            "dp_states_per_unit": res["dp_states_per_unit"],
            "sensitivity": [{str(b): float(sens[i][b]) for b in C.Q3V2_BITS} for i in range(len(units))]}


def uniform(matrices, ref_bits: int) -> dict:
    """The uniform allocation (= Q2 at ref_bits), the reference of every candidate set."""
    units = units_of(matrices)
    ch = chunking(units)
    budget = uniform_budget(matrices, ref_bits)
    rec = _alloc_record(matrices, units, ch, [ref_bits] * len(units), budget, ref_bits)
    assert rec["bytes"] == budget
    return {**rec, "rule": "uniform (Q2 reference)"}


def check_bits(matrices, per_matrix: dict, ref_bits: int) -> dict:
    """Assert matrix set and order, allowed bits, one width per unit and bytes within the budget; returns totals."""
    assert list(per_matrix) == [m["qualname"] for m in matrices], "matrix set or order differs"
    units = units_of(matrices)
    bits = []
    for u in units:
        bs = {int(per_matrix[q]) for q in u["qualnames"]}
        assert len(bs) == 1 and bs <= set(C.Q3V2_BITS), (u["id"], bs)
        bits.append(bs.pop())
    nb = storage.packed_quant_bytes([tuple(m["shape"]) for m in matrices], [int(b) for b in per_matrix.values()])
    budget = uniform_budget(matrices, ref_bits)
    assert nb <= budget, (nb, budget)
    return {"eligible_packed_bytes": nb, "budget_bytes": budget, "ref_bits": ref_bits, "unit_bits": bits}


# ---------------------------------------------------------------------------------------------------------------
# Model operations (torch).

def _distinct_sizes(block, geom) -> dict:
    return {"qkv": block.input_layernorm.weight.numel(), "o": geom["n_kv"] * geom["head_dim"],
            "gateup": block.post_attention_layernorm.weight.numel(), "down": block.mlp.down_proj.in_features}


def fold_quant_unit(block, unit: str, bits: int, s, geom) -> dict:
    """In place: apply only this unit's fold (scales s of its group) and quantize only its projections at bits.

    The other three groups get unit scales, so their tensors are written back unchanged; the fold also rewrites the
    unit's partner tensors (FOLD_PARTNERS) in the model dtype.
    """
    import torch

    from .awq import _prefix, assign, folded_params
    from .quant import dequantize, quantize

    group = UNIT_GROUP[unit]
    dev = block.input_layernorm.weight.device
    with torch.no_grad():
        bs = {g: torch.ones(n, dtype=torch.float32, device=dev) for g, n in _distinct_sizes(block, geom).items()}
        assert s.numel() == bs[group].numel(), (unit, s.numel(), bs[group].numel())
        bs[group] = s.float().to(dev)
        fold = folded_params(block, bs, geom)
        info = {"unit": unit, "scale_group": group, "bits": int(bits), "n_guarded_groups": 0, "n_clipped_codes": 0}
        for n in UNIT_PROJS[unit]:
            key = _prefix(n) + ".weight"
            r = quantize(fold[key], bits)
            fold[key] = dequantize(r.q, r.z, r.scale)
            info["n_guarded_groups"] += r.n_guarded
            info["n_clipped_codes"] += r.n_clipped
        assign(block, fold)
    return info


def own_fold_effective(block, W: dict, bits: int, layer_scales: dict, geom) -> dict:
    """Original-basis effective weights of all seven projections, each folded with its own group's scales only."""
    import torch

    from .awq import col_row_scales
    from .blocks import projections
    from .quant import dequantize, quantize

    projs = projections(block)
    dev = W["q_proj"].device
    with torch.no_grad():
        col, _ = col_row_scales({g: layer_scales[g].to(dev) for g in GROUP_UNIT}, geom)
        E = {}
        for n in C.ELIGIBLE:
            c = col[n].to(dev)[None, :]
            r = quantize(W[n].float() * c, bits)
            E[n] = dequantize(r.q, r.z, r.scale).to(projs[n].weight.dtype).float() / c
    return E
