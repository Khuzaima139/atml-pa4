"""Sensitivity probe pieces: OWL ratio, isolated block changes, QS2 errors, and the P3/Q3 build paths."""
import pytest
import torch

from task1 import config as C
from task1 import prune as P
from task1.awq import geometry
from task1.blocks import layers, projections
from task1.build import SCALE_GROUPS, build_condition
from task1.data import read_sequences
from task1.model import Pristine
from task1.probe import level_key, owl_ratio, prune_block, qs2_errors, quant_block


def test_owl_ratio_hand_built():
    w = torch.ones(10, 10)
    w[3, 7] = 1000.0
    x = torch.ones(10)
    r = owl_ratio([w], [x])  # mean = (99 + 1000) / 100 = 10.99, threshold 54.95
    assert r["value"] == pytest.approx(0.01) and r["n_outliers"] == 1 and r["n_scores"] == 100
    assert r["mean_score"] == pytest.approx(10.99) and r["threshold"] == pytest.approx(54.95)
    # Pooled over projections, scaled by input norms: scores [1,2,1,20] and [1, 1].
    w1 = torch.tensor([[1.0, -1.0], [1.0, -10.0]])
    w2 = torch.tensor([[1.0, 1.0]])
    r = owl_ratio([w1, w2], [torch.tensor([1.0, 2.0]), torch.ones(2)], m=2.0)
    assert r["mean_score"] == pytest.approx(26 / 6) and r["n_outliers"] == 1 and r["value"] == pytest.approx(1 / 6)
    assert owl_ratio([w1, w2], [torch.tensor([1.0, 2.0]), torch.ones(2)], m=5.0)["n_outliers"] == 0


def test_level_keys():
    assert [level_key(s) for s in C.PROBE_SPARSITIES] == ["0.50", "0.75"]
    assert [level_key(b) for b in C.PROBE_BITS] == ["2", "3", "4", "8"]


def _unchanged_outside(model, pristine, allowed: set):
    changed = set()
    for n, p in model.named_parameters():
        if not torch.equal(p.detach().cpu(), pristine.cpu[n]):
            changed.add(n)
    assert changed <= allowed, sorted(changed - allowed)
    return changed


def test_isolated_probes_touch_only_the_named_block(compress4_fp32):
    model = compress4_fp32
    pristine = Pristine(model)
    g = torch.Generator().manual_seed(3)
    x_l2 = {f"model.layers.{li}.{'self_attn' if n in C.ATTN_PROJ else 'mlp'}.{n}":
            torch.rand(m.in_features, generator=g) + 0.1
            for li, b in enumerate(layers(model)) for n, m in projections(b).items()}
    li = 2
    zeros = prune_block(model, li, 0.5, x_l2)
    elig = {f"{q}.weight" for q in zeros}
    assert _unchanged_outside(model, pristine, elig) == elig
    for q, z in zeros.items():
        w = dict(model.named_parameters())[q + ".weight"]
        assert int((w == 0).sum()) == z == w.numel() // 2
        keep = P.row_mask(P.wanda_score(pristine.cpu[q + ".weight"], x_l2[q]), z)
        assert torch.equal(w != 0, keep)

    pristine.restore(model)
    geom = geometry(model.config)
    hd = geom["n_kv"] * geom["head_dim"]
    cfg = model.config
    sc = {"qkv": torch.rand(cfg.hidden_size, generator=g) + 0.5, "o": torch.rand(hd, generator=g) + 0.5,
          "gateup": torch.rand(cfg.hidden_size, generator=g) + 0.5,
          "down": torch.rand(cfg.intermediate_size, generator=g) + 0.5}
    li = 1
    quant_block(model, li, 3, sc)
    allowed = {n for n, _ in model.named_parameters() if n.startswith(f"model.layers.{li}.")}
    changed = _unchanged_outside(model, pristine, allowed)
    want = {f"model.layers.{li}.{p}" for p in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                                "self_attn.v_proj.bias")}
    assert want <= changed and len(changed) == 7 + 3


def test_qs2_errors_leave_model_unchanged(compress4_fp32, tmp_path):
    from task1 import awq
    from task1.blocks import capture_block_inputs

    model = compress4_fp32
    pristine = Pristine(model)
    calib = read_sequences("c4_calibration", limit=1)
    res = awq.search_model(model, capture_block_inputs(model, calib, 1), bits_list=C.PROBE_BITS, alphas=(0.0, 0.5))
    errs = qs2_errors(model, calib, res["scales"], C.PROBE_BITS, batch_size=1)
    pristine.verify(model)
    _unchanged_outside(model, pristine, set())
    for li in range(4):
        tot = [sum(a / b for a, b in errs[(li, bb)].values()) for bb in C.PROBE_BITS]
        assert tot[0] > tot[1] > tot[2] > tot[3] > 0, tot


def test_p3_build_with_p2_counts_equals_p2(compress4_fp32):
    model = compress4_fp32
    pristine = Pristine(model)
    calib = read_sequences("c4_calibration", limit=1)
    p2 = build_condition(model, calib, "p2_s50", 1, pristine=pristine)
    w_p2 = {n: p.detach().clone() for n, p in model.named_parameters()}
    pristine.restore(model)
    zeros = {r["qualname"]: r["target_zeros"] for r in p2.records}
    p3 = build_condition(model, calib, "p3_s50", 1, zeros_per_matrix=zeros, pristine=pristine, with_distributions=False)
    for n, p in model.named_parameters():
        assert torch.equal(p, w_p2[n]), n
    assert [r["mask_zeros"] for r in p3.records] == [r["mask_zeros"] for r in p2.records]
    assert all("disagreement_with_magnitude" in r for r in p3.records) and p3.score_edges == {}
    pristine.restore(model)
    with pytest.raises(AssertionError, match="per-matrix zero counts"):
        build_condition(model, calib, "p3_s50", 1, pristine=pristine)


def test_q3_build_uniform_equals_q2_and_mixed_bits(compress4_fp32):
    from task1 import awq
    from task1.blocks import capture_block_inputs

    model = compress4_fp32
    pristine = Pristine(model)
    calib = read_sequences("c4_calibration", limit=1)
    res = awq.search_model(model, capture_block_inputs(model, calib, 1), bits_list=(2, 4), alphas=(0.0, 0.5))
    pristine.restore(model)
    sc4 = {(li, g): res["scales"][(li, g, 4)] for li in range(4) for g in SCALE_GROUPS}
    build_condition(model, calib, "q2_b4", 1, awq_scales=sc4)
    w_q2 = {n: p.detach().clone() for n, p in model.named_parameters()}
    pristine.restore(model)
    build_condition(model, calib, "q3_int4", 1, awq_scales=sc4, bits_per_block=[4, 4, 4, 4])
    for n, p in model.named_parameters():
        assert torch.equal(p, w_q2[n]), n
    pristine.restore(model)
    bits = [2, 4, 4, 2]
    mixed = {(li, g): res["scales"][(li, g, b)] for li, b in enumerate(bits) for g in SCALE_GROUPS}
    out = build_condition(model, calib, "q3_int4", 1, awq_scales=mixed, bits_per_block=bits)
    assert [r["bits"] for r in out.records] == [b for b in bits for _ in C.ELIGIBLE]
    pristine.restore(model)
    with pytest.raises(AssertionError, match="bits per block"):
        build_condition(model, calib, "q3_int4", 1, awq_scales=sc4)
    with pytest.raises(AssertionError, match="only used by Q3"):
        build_condition(model, calib, "q2_b4", 1, awq_scales=sc4, bits_per_block=[4] * 4)
