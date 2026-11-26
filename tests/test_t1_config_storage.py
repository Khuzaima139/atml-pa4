import json

import pytest

from common.paths import SOURCE_REVISIONS_LOCK
from task1 import config as C
from task1 import storage as S

FULL_IDS = {"fp16", "p1_s50", "p2_s50", "p3_s50", "q1_b4", "q1_b3", "q2_b4", "q2_b3", "q3_int4", "q3_int3"}
ALL_IDS = ["fp16", "p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p3_s50", "p3_s75", "p2_24_s50",
           "q1_b8", "q1_b4", "q1_b3", "q2_b8", "q2_b4", "q2_b3", "q3_int4", "q3_int3"]


def test_revision_from_lock():
    lock = json.loads(SOURCE_REVISIONS_LOCK.read_text())
    assert C.MODEL_ID == "Qwen/Qwen2.5-1.5B"
    assert C.MODEL_REVISION == lock["revisions"]["model_base"]


def test_condition_registry_and_hellaswag_scope():
    assert list(C.CONDITIONS) == ALL_IDS
    assert {k for k, v in C.CONDITIONS.items() if v["hellaswag"] == "full"} == FULL_IDS
    assert {k for k, v in C.CONDITIONS.items() if v["hellaswag"] == "subset512"} == set(ALL_IDS) - FULL_IDS
    assert C.hellaswag_n("full") == 2000 and C.hellaswag_n("subset512") == 512


def test_fixed_settings():
    assert C.EVAL_ID == "t1-eval-v1"
    assert C.ELIGIBLE == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert (C.SEED, C.GROUP_SIZE, C.DTYPE) == (6304, 128, "float16")
    assert C.SPARSITIES == (0.25, 0.50, 0.75) and C.BITS == (8, 4, 3) and C.Q3_BITS == (2, 3, 4, 8)
    for b in (C.PPL_BATCH_SIZE, C.HELLASWAG_BATCH_SIZE, C.CALIB_BATCH_SIZE, C.LMEVAL_BATCH_SIZE):
        assert isinstance(b, int) and b >= 1
    assert C.LATENCY_REPS >= 5
    assert C.LMEVAL_ACCEPTANCE_RUN == "fp32_model"
    assert C.settings_dict()["lmeval_acceptance_run"] == C.LMEVAL_ACCEPTANCE_RUN


SHAPES = C.full_model_matrix_shapes()
NON_ELIG = C.FULL_MODEL_NON_ELIGIBLE_PARAMS


def test_full_model_shapes():
    assert len(SHAPES) == 196
    assert sum(o * i for o, i in SHAPES) == 1_310_195_712
    assert NON_ELIG == 233_518_592


def test_fp16_bytes():
    assert S.fp16_bytes(SHAPES) == 2_620_391_424
    assert S.whole_model_bytes(S.fp16_bytes(SHAPES), NON_ELIG) == 3_087_428_608


@pytest.mark.parametrize("bits,eligible,whole", [
    (8, 1_340_903_424, 1_807_940_608), (4, 680_687_616, 1_147_724_800),
    (3, 515_633_664, 982_670_848), (2, 350_579_712, 817_616_896)])
def test_packed_quant_bytes(bits, eligible, whole):
    assert S.packed_quant_bytes(SHAPES, bits) == eligible
    assert S.whole_model_bytes(eligible, NON_ELIG) == whole


def test_packed_quant_parts_and_groups():
    d = S.packed_quant_bytes(SHAPES, 4, detail=True)
    assert d["groups"] == 10_235_904
    assert d["codes"] == 655_097_856 and d["scales"] == 20_471_808 and d["zero_points"] == 5_117_952
    # Mixed bits per matrix: the sum of per-matrix totals.
    bits = [8 if i % 2 else 3 for i in range(len(SHAPES))]
    assert S.packed_quant_bytes(SHAPES, bits) == sum(S.packed_quant_matrix(s, b)["total"] for s, b in zip(SHAPES, bits))


def test_packed_quant_byte_padding_per_matrix():
    # 3 x 130 at 3 bits: 390 codes -> ceil(1170/8) = 147 bytes; groups 3 * 2 = 6 -> 12 B scales,
    # ceil(18/8) = 3 B zero-points.
    m = S.packed_quant_matrix((3, 130), 3)
    assert (m["codes"], m["scales"], m["zero_points"], m["groups"]) == (147, 12, 3, 6)


@pytest.mark.parametrize("frac,eligible", [(0.25, 2_129_068_032), (0.50, 1_473_970_176), (0.75, 818_872_320)])
def test_sparse16_bytes(frac, eligible):
    nnz = [round(o * i * (1 - frac)) for o, i in SHAPES]
    assert S.sparse16_bytes(SHAPES, nnz) == eligible


def test_sparse16_mask_padding():
    # 3 x 3 = 9 weights -> 2 mask bytes; 4 nonzeros -> 8 bytes.
    assert S.sparse16_bytes([(3, 3)], [4]) == 10
    with pytest.raises(AssertionError):
        S.sparse16_bytes([(3, 3)], [10])


def test_storage_report_dense():
    nnz = [o * i for o, i in SHAPES]
    r = S.report(SHAPES, NON_ELIG, nnz)
    assert r["fp16_dense"]["eligible_bytes"] == 2_620_391_424
    assert r["fp16_dense"]["bits_per_eligible_weight"] == 16.0
    assert r["sparse16"]["eligible_bytes"] == 163_774_464 + 2_620_391_424
    assert r["sparse16"]["bits_per_eligible_weight"] == 17.0
    assert r["eval_id"] == C.EVAL_ID
