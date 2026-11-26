"""Q3 design v2: units, exact chunked costs, knapsack against brute force, unit folds, probe-part join."""
import math
import random

import pytest
import torch

from task1 import config as C
from task1 import q3v2 as Q
from task1 import storage
from task1.select import join_probe_parts, matrices_of

TINY = {"q_proj": (256, 256), "k_proj": (128, 256), "v_proj": (128, 256), "o_proj": (256, 256),
        "gate_proj": (512, 256), "up_proj": (512, 256), "down_proj": (256, 512)}


def _mats(n_blocks, shapes):
    return [{"layer": li, "name": n, "qualname": f"model.layers.{li}.{'self_attn' if n in C.ATTN_PROJ else 'mlp'}.{n}",
             "shape": list(shapes[n])} for li in range(n_blocks) for n in C.ELIGIBLE]


FULL = _mats(C.N_LAYERS, C.FULL_MODEL_SHAPES)


def test_constants_and_units():
    assert C.Q3V2_DESIGN == "fold_group_v2" and C.Q3V2_UNIT == "fold_group"
    assert [u for u, _, _ in C.Q3V2_UNITS] == ["attn_in", "attn_out", "mlp_in", "mlp_out"]
    assert [g for _, g, _ in C.Q3V2_UNITS] == ["qkv", "o", "gateup", "down"]
    assert sorted(p for _, _, ps in C.Q3V2_UNITS for p in ps) == sorted(C.ELIGIBLE)
    units = Q.units_of(FULL)
    assert len(units) == C.Q3V2_FULL_MODEL_N_UNITS == 112
    assert [u["id"] for u in units[:5]] == ["0.attn_in", "0.attn_out", "0.mlp_in", "0.mlp_out", "1.attn_in"]
    assert units[0]["qualnames"] == ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
                                     "model.layers.0.self_attn.v_proj"]
    assert sum(u["groups"] for u in units) == storage.packed_quant_bytes([tuple(m["shape"]) for m in FULL], 4,
                                                                         detail=True)["groups"]


def test_full_model_chunking():
    units = Q.units_of(FULL)
    ch = Q.chunking(units)
    assert ch["chunk_groups"] == C.Q3V2_FULL_MODEL_CHUNK_GROUPS == 3072
    assert ch["chunk_groups"] == math.gcd(*[u["groups"] for u in units])
    assert ch["unit_chunks"][:4] == [8, 6, 70, 35]
    assert {u["unit"]: k for u, k in zip(units, ch["unit_chunks"])} == C.Q3V2_FULL_MODEL_UNIT_CHUNKS
    assert ch["eighth_bytes_per_group"] == {"2": 274, "3": 403, "4": 532, "8": 1048}
    for t, b in (("int4", 4), ("int3", 3)):
        budget = Q.uniform_budget(FULL, b)
        assert budget == C.Q3_FULL_MODEL_BUDGET_BYTES[t]
        assert (8 * budget) % ch["chunk_groups"] == 0
        assert Q.budget_units(budget, ch["chunk_groups"]) * ch["chunk_groups"] == 8 * budget
        uni = Q.uniform(FULL, b)
        assert uni["bytes"] == budget and uni["collapsed_to_uniform"] and uni["chunking"]["full_model_expected"]


@pytest.mark.parametrize("mats", [FULL, _mats(4, TINY)], ids=["full", "tiny"])
def test_chunk_costs_reproduce_storage_for_random_allocations(mats):
    units = Q.units_of(mats)
    ch = Q.chunking(units)
    rng = random.Random(6304)
    shapes = [tuple(m["shape"]) for m in mats]
    for _ in range(50):
        bits = [rng.choice(C.Q3V2_BITS) for _ in units]
        cost = sum(ch["costs"][i][b] for i, b in enumerate(bits))
        pm = Q.per_matrix_bits(mats, units, bits)
        nb = storage.packed_quant_bytes(shapes, list(pm.values()))
        assert cost * ch["chunk_groups"] == 8 * nb
        per_unit = [storage.packed_quant_bytes([tuple(s) for s in u["shapes"]], b) for u, b in zip(units, bits)]
        assert sum(per_unit) == nb


def test_chunking_refuses_inexact_costs():
    units = [{"id": "x", "groups": 3, "shapes": [[3, 128]]}]   # 3 groups at 3 bits: ceil(9/8) zero-point bytes
    with pytest.raises(AssertionError):
        Q.chunking(units)


def _instance(seed, n, int_values=False):
    rng = random.Random(seed)
    chunks = [rng.choice([1, 2, 3, 5, 8]) for _ in range(n)]
    costs = [{b: k * Q.eighth_bytes_per_group(b) for b in C.Q3V2_BITS} for k in chunks]
    sens = []
    for _ in range(n):
        if int_values:
            sens.append({b: float(rng.randint(0, 3)) for b in C.Q3V2_BITS})
        else:
            base = rng.uniform(0.001, 1.0)
            sens.append({2: base * rng.uniform(5, 30), 3: base * rng.uniform(1.5, 5), 4: base,
                         8: rng.uniform(-1e-3, 1e-3)})
    return sens, costs


@pytest.mark.parametrize("seed", range(12))
def test_dp_equals_brute_force_mixed_sizes(seed):
    n = 6
    sens, costs = _instance(seed, n, int_values=seed % 3 == 0)
    rng = random.Random(seed + 100)
    lo = sum(min(c.values()) for c in costs)
    hi = sum(max(c.values()) for c in costs)
    ref_bits = rng.choice([3, 4])
    for budget in (sum(c[ref_bits] for c in costs), rng.randint(lo, hi), lo, hi):
        dp = Q.knapsack(sens, costs, budget)
        bf = Q.brute_force(sens, costs, budget)
        assert dp["bits"] == bf["bits"] and dp["cost"] == bf["cost"] and dp["objective"] == bf["objective"]
        assert dp["cost"] <= budget and len(dp["dp_states_per_unit"]) == n


def test_dp_tie_rules():
    costs = [{b: k * Q.eighth_bytes_per_group(b) for b in C.Q3V2_BITS} for k in (1, 1)]
    flat = [{b: 0.0 for b in C.Q3V2_BITS}] * 2
    big = 10 ** 9
    # All objectives equal: fewest bytes wins (both at 2 bits).
    assert Q.knapsack(flat, costs, big)["bits"] == [2, 2]
    # Equal objective and equal bytes: lexicographically smaller vector by unit order.
    sens = [{2: 1.0, 3: 0.0, 4: 1.0, 8: 1.0}, {2: 0.0, 3: 1.0, 4: 1.0, 8: 1.0}]
    sym = [{2: 0.0, 3: 1.0, 4: 1.0, 8: 1.0}, {2: 1.0, 3: 0.0, 4: 1.0, 8: 1.0}]
    budget = costs[0][2] + costs[1][3]
    assert Q.knapsack(sens, costs, budget)["bits"] == [3, 2]
    assert Q.knapsack(sym, costs, budget)["bits"] == [2, 3]
    both = [{2: 0.0, 3: 0.0, 4: 1.0, 8: 1.0}] * 2
    assert Q.knapsack(both, costs, budget)["bits"] == [2, 2]


def test_exact_objective_beyond_float_sums():
    costs = [{b: Q.eighth_bytes_per_group(b) for b in C.Q3V2_BITS} for _ in range(3)]
    # 1e16 + 1 - 1e16 is 0 in float but 1 exactly.
    sens = [{2: 1e16, 3: 1e16, 4: 1e16, 8: 1e16}, {2: 1.0, 3: 1.0, 4: 1.0, 8: 1.0},
            {2: -1e16, 3: -1e16, 4: -1e16, 8: -1e16}]
    r = Q.knapsack(sens, costs, 10 ** 9)
    assert r["objective_exact"] == "1/1"


def test_allocation_records_and_check():
    mats = _mats(4, TINY)
    sens = [{2: 1.0 + i, 3: 0.5, 4: 0.1, 8: 0.0} for i in range(16)]
    al = Q.allocation(sens, mats, 3)
    assert al["design"] == C.Q3V2_DESIGN and al["bytes"] <= al["budget"] == Q.uniform_budget(mats, 3)
    assert len(al["bits"]) == len(al["unit_ids"]) == 16 and len(al["per_matrix_bits"]) == 28
    assert sum(sum(h.values()) for h in al["bits_histogram"].values()) == 16
    tot = Q.check_bits(mats, al["per_matrix_bits"], 3)
    assert tot["unit_bits"] == al["bits"] and tot["eligible_packed_bytes"] == al["bytes"]
    bad = dict(al["per_matrix_bits"])
    bad["model.layers.0.self_attn.k_proj"] = 8 if bad["model.layers.0.self_attn.q_proj"] != 8 else 2
    with pytest.raises(AssertionError):
        Q.check_bits(mats, bad, 3)
    over = {q: 8 for q in al["per_matrix_bits"]}
    with pytest.raises(AssertionError):
        Q.check_bits(mats, over, 3)


# ---------------------------------------------------------------------------------------------------------------
def _scales(model, seed):
    from task1.awq import geometry

    g = torch.Generator().manual_seed(seed)
    geom = geometry(model.config)
    out = {}
    for li, block in enumerate(model.model.layers):
        sizes = Q._distinct_sizes(block, geom)
        for grp, n in sizes.items():
            out[(li, grp)] = 0.5 + torch.rand(n, generator=g)
    return out, geom


@pytest.mark.parametrize("unit", Q.UNIT_NAMES)
def test_isolated_unit_touches_only_named_unit(compress4_fp32, unit):
    from task1.model import Pristine

    model = compress4_fp32
    pristine = Pristine(model)
    scales, geom = _scales(model, 7)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    li = 2
    info = Q.fold_quant_unit(model.model.layers[li], unit, 3, scales[(li, Q.UNIT_GROUP[unit])], geom)
    assert info["unit"] == unit and info["bits"] == 3
    prefix = f"model.layers.{li}."
    own = {prefix + ("self_attn." if n in C.ATTN_PROJ else "mlp.") + n + ".weight" for n in Q.UNIT_PROJS[unit]}
    partners = {prefix + t for t in Q.FOLD_PARTNERS[unit]}
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert own <= changed and changed <= own | partners
    for n in partners:
        assert n in changed
    # The executed unit weights are quantized folded weights: at most 2^3 distinct values per group of 128.
    for n in own:
        w = dict(model.named_parameters())[n].detach()
        for row in w[:4]:
            for grp in row.split(C.GROUP_SIZE):
                assert grp.unique().numel() <= 8
    pristine.restore(model)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_unit_fold_preserves_function_before_rounding(compress4_fp32):
    """Each unit's fold alone, without quantization, keeps the block output (FP32 model)."""
    from task1.awq import assign, folded_params

    model = compress4_fp32
    scales, geom = _scales(model, 9)
    block = model.model.layers[1]
    h = torch.randn(2, 5, model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    pe = model.model.rotary_emb(h, torch.arange(5)[None].expand(2, 5))

    def fwd():
        out = block(h, position_embeddings=pe)
        return out[0] if isinstance(out, tuple) else out

    with torch.no_grad():
        y0 = fwd()
        for unit in Q.UNIT_NAMES:
            bs = {g: torch.ones(n) for g, n in Q._distinct_sizes(block, geom).items()}
            bs[Q.UNIT_GROUP[unit]] = scales[(1, Q.UNIT_GROUP[unit])]
            saved = {n: p.detach().clone() for n, p in block.named_parameters()}
            assign(block, folded_params(block, bs, geom))
            assert torch.allclose(fwd(), y0, rtol=1e-4, atol=1e-5), unit
            assign(block, saved)


def test_own_fold_effective_matches_definition(compress4_fp32):
    from task1.quant import fake_quant

    model = compress4_fp32
    scales, geom = _scales(model, 11)
    block = model.model.layers[0]
    W = {n: m.weight.detach().float() for n, m in
         {n: getattr(block.self_attn if n in C.ATTN_PROJ else block.mlp, n) for n in C.ELIGIBLE}.items()}
    E = Q.own_fold_effective(block, W, 4, {g: scales[(0, g)] for g in Q.GROUP_UNIT}, geom)
    s_o = scales[(0, "o")].view(geom["n_kv"], geom["head_dim"]).repeat_interleave(geom["n_rep"], 0).reshape(-1)
    col = {"q_proj": scales[(0, "qkv")], "k_proj": scales[(0, "qkv")], "v_proj": scales[(0, "qkv")],
           "o_proj": s_o, "gate_proj": scales[(0, "gateup")], "up_proj": scales[(0, "gateup")],
           "down_proj": scales[(0, "down")]}
    for n in C.ELIGIBLE:
        want = fake_quant(W[n] * col[n][None, :], 4) / col[n][None, :]
        assert torch.allclose(E[n], want, rtol=0, atol=1e-6), n


def test_build_uniform_bits_per_matrix_equals_bits_per_block(compress4_fp32, tiny_compress4_dir):
    from task1.build import SCALE_GROUPS, build_condition
    from task1.model import Pristine, load_model

    seqs = [{"token_ids": list(range(100 + i, 132 + i))} for i in range(2)]
    scales, _ = _scales(compress4_fp32, 13)
    aw = {(li, g): scales[(li, g)] for li in range(4) for g in SCALE_GROUPS}
    mats = matrices_of(compress4_fp32)
    a = build_condition(compress4_fp32, seqs, "q3_int3", 2, awq_scales=aw, bits_per_block=[3, 3, 3, 3],
                        pristine=Pristine(compress4_fp32))
    m2, _ = load_model("cpu", tiny_compress4_dir, dtype="float32")
    b = build_condition(m2, seqs, "q3_int3", 2, awq_scales=aw, bits_per_matrix={m["qualname"]: 3 for m in mats},
                        pristine=Pristine(m2))
    assert [r["bits"] for r in a.records] == [r["bits"] for r in b.records]
    for x, y in zip(a.payload, b.payload):
        for k in ("qcodes", "qzeros", "qscales"):
            assert torch.equal(x["tensors"][k], y["tensors"][k])
    with pytest.raises(AssertionError):
        build_condition(m2, seqs, "q3_int3", 2, awq_scales=aw, bits_per_block=[3] * 4,
                        bits_per_matrix={m["qualname"]: 3 for m in mats}, pristine=Pristine(m2))


# ---------------------------------------------------------------------------------------------------------------
def _part(mats, lo, hi, ref, offset=0.0):
    units = [u for u in Q.units_of(mats) if lo <= u["layer"] <= hi]
    vals = {m: {str(b): [offset + 10 * u["index"] + b for u in units] for b in C.Q3V2_BITS}
            for m in C.Q3V2_MEASURES}
    return ({"design": C.Q3V2_DESIGN, "unit": C.Q3V2_UNIT, "n_units_total": 4 * len(C.Q3V2_UNITS),
             "fp16_calib_nll": {"mean_nll": ref}, "block_range": [lo, hi],
             "units": [{"index": u["index"], "id": u["id"]} for u in units], "values": vals}, {"dir": f"{lo}-{hi}"})


def test_probe_union_check():
    mats = _mats(4, TINY)
    a, b = _part(mats, 0, 1, 2.5), _part(mats, 2, 3, 2.5 + 5e-7)
    values, info = join_probe_parts([b, a], mats)
    assert info["n_units"] == 16 and info["n_parts"] == 2 and info["part_of_unit"] == [1] * 8 + [0] * 8
    assert values["QS1v2"][0] == {2: 2.0, 3: 3.0, 4: 4.0, 8: 8.0}
    assert values["QS2v2"][15] == {b: 150.0 + b for b in C.Q3V2_BITS}
    with pytest.raises(SystemExit, match="exactly once"):
        join_probe_parts([a], mats)
    with pytest.raises(SystemExit, match="covered by probe parts"):
        join_probe_parts([a, _part(mats, 1, 3, 2.5)], mats)
    with pytest.raises(SystemExit, match="different FP16"):
        join_probe_parts([a, _part(mats, 2, 3, 2.5 + 2e-6)], mats)
    bad = _part(mats, 2, 3, 2.5)
    bad[0]["design"] = "block"
    with pytest.raises(SystemExit, match="is not a fold_group_v2 probe"):
        join_probe_parts([a, bad], mats)
