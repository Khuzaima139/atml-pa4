"""Part C budget arithmetic, quantizer, packing, packed-cache byte accounting and selection sets."""
import inspect

import pytest
import torch

from common.seeding import seed_everything
from task2 import kv_budget as KB
from task2 import kv_cache_ops as KO
from task2 import kv_select as KS
from task2.kv_quant import (GROUP, PackedKVCache, QuantBlock, dequantize_groups, pack_codes, quantize_groups,
                            unpack_codes)

TINY = KB.KVShape(n_layers=2, n_kv_heads=2, head_dim=128)
QWEN = KB.KVShape(n_layers=28, n_kv_heads=2, head_dim=128)
LENGTHS = (64, 100, 2048, 2630, 4100)


def _cache(L, shape=TINY, dtype=torch.float16, seed=6304):
    g = torch.Generator().manual_seed(seed + L)
    ks = [torch.randn(1, shape.n_kv_heads, L, shape.head_dim, generator=g).to(dtype) for _ in range(shape.n_layers)]
    vs = [torch.randn(1, shape.n_kv_heads, L, shape.head_dim, generator=g).to(dtype) for _ in range(shape.n_layers)]
    return ks, vs


# ---------------------------------------------------------------------------------------------------------------- budget

def test_shape_from_config():
    from transformers import Qwen2Config

    cfg = Qwen2Config(hidden_size=1536, num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2)
    assert KB.KVShape.from_config(cfg) == QWEN
    assert QWEN.channels == 14336 and QWEN.fp16_bytes_per_position == 28672


def test_budget_formulas_match_manual_examples():
    assert KB.m16(QWEN, 2633) == 75_493_376
    assert KB.m4(QWEN, 2633) == 21_417_984
    assert KB.group_bytes(4) == 36 and KB.group_bytes(2) == 20
    assert KB.mb(TINY, 100, 4) == TINY.channels * (36 + 2 * 36)


@pytest.mark.parametrize("L", [2048, 2630, 4100, 2633, 5243])
def test_eviction_n_is_maximal(L):
    for shape in (TINY, QWEN):
        M4 = KB.m4(shape, L)
        n = KB.sink_recent_n(shape, L)
        assert KB.sink_recent_bytes(shape, n) <= M4 < KB.sink_recent_bytes(shape, n + 1)
        n = KB.question_aware_n(shape, L)
        assert KB.question_aware_bytes(shape, n) <= M4 < KB.question_aware_bytes(shape, n + 1)


@pytest.mark.parametrize("L", [2048, 2630, 4100, 2633, 5243])
def test_hybrid_m_is_minimal(L):
    for shape in (TINY, QWEN):
        M4 = KB.m4(shape, L)
        m, f = KB.hybrid_split(shape, L)
        assert f == L - 64 * m and f >= KB.N_SAFEGUARD
        assert KB.hybrid_bytes(shape, L, m) <= M4
        assert m == 0 or KB.hybrid_bytes(shape, L, m - 1) > M4
        assert f <= KB.question_aware_n(shape, L)


def test_safeguards_that_do_not_fit_raise():
    with pytest.raises(KB.BudgetError):
        KB.sink_recent_n(TINY, 64)
    with pytest.raises(KB.BudgetError):
        KB.question_aware_n(TINY, 100)


def test_budget_table_for_qwen_lengths():
    t = KB.budget_table(QWEN, 2633)
    assert (t["question_aware_n"], t["hybrid_m"], t["hybrid_fp16_positions"]) == (746, 35, 393)
    assert t["sink_recent_n"] == (t["M4"] - 8) // 28672
    assert 0 < t["realized_fraction"]["hybrid"] <= t["realized_fraction"]["int4_channel_k"]


# ------------------------------------------------------------------------------------------------------------- quantizer

@pytest.mark.parametrize("bits", [2, 4])
def test_pack_unpack_round_trip(bits):
    g = torch.Generator().manual_seed(1)
    codes = torch.randint(0, 1 << bits, (3, 5, 64), generator=g, dtype=torch.uint8)
    packed = pack_codes(codes, bits)
    assert packed.dtype == torch.uint8 and packed.shape == (3, 5, 64 * bits // 8)
    assert torch.equal(unpack_codes(packed, bits), codes)


def test_packing_is_low_bits_first():
    c = torch.zeros(64, dtype=torch.uint8)
    c[0], c[1] = 0x3, 0xA
    assert int(pack_codes(c, 4)[0]) == 0xA3
    c = torch.zeros(64, dtype=torch.uint8)
    c[:4] = torch.tensor([1, 2, 3, 0], dtype=torch.uint8)
    assert int(pack_codes(c, 2)[0]) == 1 | (2 << 2) | (3 << 4)


@pytest.mark.parametrize("bits", [2, 4])
def test_dequantization_error_within_half_scale(bits):
    g = torch.Generator().manual_seed(2)
    x = (3 * torch.randn(40, 64, generator=g)).to(torch.float16)
    codes, scale, offset = quantize_groups(x, bits)
    assert scale.dtype == torch.float16 and offset.dtype == torch.float16
    assert int(codes.max()) <= (1 << bits) - 1
    err = (dequantize_groups(codes, scale, offset, torch.float32) - x.float()).abs()
    bound = scale.float()[:, None] / 2 * (1 + 1e-3) + 1e-6
    assert bool((err <= bound).all())


def test_constant_group_exact():
    x = torch.full((2, 64), 1.234, dtype=torch.float16)
    codes, scale, offset = quantize_groups(x, 4)
    assert bool((scale == 0).all()) and bool((codes == 0).all())
    assert torch.equal(dequantize_groups(codes, scale, offset, torch.float16), x)


def test_round_half_even():
    x = torch.zeros(64)
    x[1] = 15.0  # scale 1, offset 0
    x[2], x[3], x[4] = 2.5, 3.5, 0.5
    codes, scale, offset = quantize_groups(x, 4)
    assert float(scale) == 1.0 and float(offset) == 0.0
    assert codes[2:5].tolist() == [2, 4, 0]


@pytest.mark.parametrize("layout", ["token", "channel"])
def test_quant_block_layouts(layout):
    x = torch.randn(1, 2, 128, 128).to(torch.float16)
    qb = QuantBlock.encode(x, layout, 4)
    assert qb.n_tokens == 128
    if layout == "token":
        assert qb.scale.shape == (1, 2, 128, 2)
        assert torch.equal(qb.scale[0, 1, 5, 1], ((x[0, 1, 5, 64:].float().max() - x[0, 1, 5, 64:].float().min())
                                                  / 15).half())
    else:
        assert qb.scale.shape == (1, 2, 2, 128)
        col = x[0, 1, 64:128, 7].float()
        assert torch.equal(qb.scale[0, 1, 1, 7], ((col.max() - col.min()) / 15).half())
    rec = qb.decode(torch.float32)
    assert rec.shape == x.shape
    assert float((rec - x.float()).abs().max()) < 0.5


# --------------------------------------------------------------------------------------------------- packed-cache bytes

def _scores(L, seed=3):
    return torch.rand(L, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("L", LENGTHS)
def test_int4_bytes_equal_formula(L):
    ks, vs = _cache(L)
    b = {}
    for method in ("int4_token", "int4_channel_k"):
        packed, info = KO.compress(method, TINY, ks, vs)
        nb = packed.nbytes()
        assert nb["total"] == KB.m4(TINY, L) == info["formula_bytes"]
        assert nb["fp16_tail"] == TINY.fp16_bytes_per_position * (L % GROUP)
        assert nb["indices"] == 0
        b[method] = nb
    assert b["int4_token"] == b["int4_channel_k"]


@pytest.mark.parametrize("L", LENGTHS)
def test_fp16_bytes_equal_m16(L):
    ks, vs = _cache(L)
    packed, _ = KO.compress("fp16", TINY, ks, vs)
    assert packed.nbytes()["total"] == KB.m16(TINY, L)


@pytest.mark.parametrize("L", [2048, 2630, 4100])
def test_eviction_and_hybrid_bytes_equal_formula(L):
    ks, vs = _cache(L)
    s = _scores(L)
    M4 = KB.m4(TINY, L)
    for method in ("sink_recent", "question_aware", "hybrid"):
        packed, info = KO.compress(method, TINY, ks, vs, s)
        total = packed.nbytes()["total"]
        assert total == info["formula_bytes"] <= M4
    packed, info = KO.compress("hybrid", TINY, ks, vs, s)
    nb = packed.nbytes()
    assert nb["indices"] == 4 * info["f"]
    assert nb["payload"] + nb["metadata"] == TINY.channels * info["m"] * 20


@pytest.mark.parametrize("L", [64, 100])
def test_short_contexts_raise_for_eviction(L):
    ks, vs = _cache(L)
    with pytest.raises(KB.BudgetError):
        KO.compress("question_aware", TINY, ks, vs, _scores(L))


@pytest.mark.parametrize("method", ["int4_token", "int4_channel_k", "hybrid"])
def test_dequantized_values_in_position_order(method):
    L = 2048
    ks, vs = _cache(L)
    packed, info = KO.compress(method, TINY, ks, vs, _scores(L))
    k, v = packed.layer_tensors(1, torch.float32)
    assert k.shape == ks[1].shape
    assert float((k - ks[1].float()).abs().max()) < (0.5 if method != "hybrid" else 2.0)
    if method == "hybrid":
        fp = info["fp16_positions"]
        assert torch.equal(k[:, :, fp], ks[1][:, :, fp].float())
    else:
        assert torch.equal(k[:, :, 2048 - L % 64:], ks[1][:, :, 2048 - L % 64:].float())
    rec = packed.reconstruction_errors(ks, vs)
    assert 0 < rec["k_mean"] < 1 and len(rec["v_per_layer"]) == 2


def test_tail_group_quantized_at_64_tokens():
    L = 100  # r = 36
    ks, vs = _cache(L)
    packed, _ = KO.compress("int4_channel_k", TINY, ks, vs)
    per_pos = TINY.fp16_bytes_per_position
    g = torch.Generator().manual_seed(9)
    pos = L
    for step in range(1, 30):
        nk = [torch.randn(1, 2, 1, 128, generator=g).half() for _ in range(2)]
        nv = [torch.randn(1, 2, 1, 128, generator=g).half() for _ in range(2)]
        groups = packed.append(nk, nv, torch.tensor([pos]))
        pos += 1
        tail = 36 + step
        if tail < 64:
            assert groups == 0
            assert packed.nbytes()["total"] == KB.m4(TINY, L) + per_pos * step
        else:
            break
    assert groups == 1 and step == 28
    assert packed.nbytes()["total"] == KB.m4(TINY, L + step)
    assert packed.layers[0].k_q.n_tokens == 128 and packed.layers[0].k_tail.shape[2] == 0
    assert packed.n_appended == 28
    assert torch.equal(packed.positions(), torch.arange(L + 28))


def test_hybrid_and_eviction_keep_appended_fp16():
    L = 2048
    ks, vs = _cache(L)
    for method in ("hybrid", "question_aware"):
        packed, _ = KO.compress(method, TINY, ks, vs, _scores(L))
        before = packed.nbytes()["total"]
        nk = [torch.randn(1, 2, 70, 128).half() for _ in range(2)]
        assert packed.append(nk, nk, torch.arange(L, L + 70)) == 0
        assert packed.nbytes()["total"] == before + 70 * TINY.fp16_bytes_per_position


# ------------------------------------------------------------------------------------------------------------ selection

def test_selections_include_safeguards_and_global_ranking():
    L = 2048
    s = _scores(L)
    ranking = KS.global_ranking(s)
    assert torch.equal(torch.sort(ranking).values, torch.arange(L))
    safe = set(range(8)) | set(range(L - 64, L))
    assert set(ranking[:72].tolist()) == safe
    rest = ranking[72:]
    assert bool((s[rest][1:] <= s[rest][:-1]).all())
    n = KB.question_aware_n(TINY, L)
    qa = KS.question_aware_positions(ranking, n)
    assert safe <= set(qa.tolist()) and qa.numel() == n
    sr = KS.sink_recent_positions(L, KB.sink_recent_n(TINY, L))
    assert safe <= set(sr.tolist())
    m, f = KB.hybrid_split(TINY, L)
    fp, low = KS.hybrid_positions(ranking, f)
    assert set(fp.tolist()) == set(ranking[:f].tolist()) and set(fp.tolist()) <= set(qa.tolist())
    assert low.numel() == 64 * m and not (set(fp.tolist()) & set(low.tolist()))


def test_ranking_ties_go_to_lower_position():
    s = torch.zeros(300)
    s[100] = s[200] = 1.0
    r = KS.global_ranking(s)
    assert r[72:76].tolist() == [100, 200, 8, 9]


def test_selection_and_compression_apis_take_no_gold():
    banned = {"code", "gold", "gold_ids", "answer", "evidence", "evidence_char_start", "evidence_char_end",
              "evidence_positions", "target", "label"}
    fns = [KS.suffix_attention, KS.question_scores, KS.global_ranking, KS.sink_recent_positions,
           KS.question_aware_positions, KS.hybrid_positions, KS.safeguard_positions, KS.SuffixQueryCapture.__init__,
           KO.full_prefill, KO.score_prefill, KO.compress, KO.run_method, KO.greedy_decode, KO.CacheSession.forward,
           PackedKVCache.evict, PackedKVCache.uniform, PackedKVCache.hybrid, PackedKVCache.fp16]
    for fn in fns:
        params = set(inspect.signature(fn).parameters)
        assert not (params & banned), (fn.__qualname__, params & banned)
    assert "gold_ids" in inspect.signature(KO.teacher_forced_logits).parameters


def test_seeded_cache_is_reproducible():
    seed_everything(6304)
    a = _cache(128)
    b = _cache(128)
    assert all(torch.equal(x, y) for x, y in zip(a[0], b[0]))
