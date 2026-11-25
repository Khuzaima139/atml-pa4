"""AWQ scale construction, folding, GQA tying, effective weights and the scale search."""
import pytest
import torch

from task1 import awq
from task1 import config as C
from task1.blocks import capture_block_inputs, layers, projections, run_block
from task1.build import error_pass
from task1.data import read_sequences
from task1.quant import fake_quant


def _random_scales(geom, hidden, inter, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda n: torch.exp(torch.randn(n, generator=g))  # noqa: E731
    return {"qkv": r(hidden), "o": r(geom["n_kv"] * geom["head_dim"]), "gateup": r(hidden), "down": r(inter)}


def test_alpha_zero_gives_unit_scales():
    imp = torch.rand(64) * 10 + 1e-6
    imp[3] = 0.0
    assert torch.equal(awq.scales_for(imp, 0.0), torch.ones(64))
    s = awq.scales_for(imp, 0.5)
    assert float(s.max() * s.min()) == pytest.approx(1.0, rel=1e-6)
    assert float(s.min()) > 0  # floor at 1e-4 before normalization


def test_alpha_grid():
    assert awq.ALPHA_GRID[0] == 0.0 and awq.ALPHA_GRID[-1] == 0.95 and len(awq.ALPHA_GRID) == 20
    assert awq.SEARCH_BITS == (2, 3, 4, 8)


def test_gqa_tie_structure(compress_fp32):
    geom = awq.geometry(compress_fp32.config)
    assert geom == {"n_heads": 4, "n_kv": 2, "head_dim": 64, "n_rep": 2}
    s = torch.arange(1, 129, dtype=torch.float32)  # one value per (KV head, dim)
    e = awq.expand("o", s, geom)
    assert e.shape == (256,)
    for h in range(4):  # query head h reads KV head h // 2
        assert torch.equal(e[64 * h:64 * (h + 1)], s[64 * (h // 2):64 * (h // 2 + 1)])
    absmean = torch.rand(256)
    imp = awq.importance("o", absmean, geom)
    want = torch.stack([(absmean[0:64] + absmean[64:128]) / 2, (absmean[128:192] + absmean[192:256]) / 2])
    assert imp.shape == (128,) and torch.allclose(imp, want.reshape(-1))
    assert torch.equal(awq.expand("qkv", absmean, geom), absmean)


@torch.inference_mode()
def test_folding_preserves_block_outputs(compress_fp32):
    model = compress_fp32
    cfg = model.config
    geom = awq.geometry(cfg)
    stream = capture_block_inputs(model, read_sequences("c4_calibration", limit=2), batch_size=2)
    for li, block in enumerate(layers(model)):
        ref = run_block(block, stream)
        before = {n: p.clone() for n, p in block.named_parameters()}
        sc = _random_scales(geom, cfg.hidden_size, cfg.intermediate_size, seed=li)
        err = awq.check_preservation(block, stream[0], ref[0][0], sc, geom)
        assert err["max_abs_err_over_max_abs_out"] < 1e-5 and err["rel_fro_update"] < 1e-5, err
        for n, p in block.named_parameters():  # the check leaves the block unchanged
            assert torch.equal(p, before[n]), n
        # The fold changes the tensors it should change.
        fold = awq.folded_params(block, sc, geom)
        assert set(awq.NON_ELIGIBLE_FOLDED) <= set(fold)
        assert not torch.allclose(fold["self_attn.v_proj.bias"], block.self_attn.v_proj.bias)
        stream = ref


@torch.inference_mode()
def test_effective_weights_without_quantization_have_zero_error(compress_fp32):
    model = compress_fp32
    cfg = model.config
    geom = awq.geometry(cfg)
    stream = capture_block_inputs(model, read_sequences("c4_calibration", limit=2), batch_size=2)
    block = layers(model)[0]
    sc = _random_scales(geom, cfg.hidden_size, cfg.intermediate_size, seed=7)
    col, row = awq.col_row_scales(sc, geom)
    W = {n: m.weight.float() for n, m in projections(block).items()}
    E = {n: awq.unfold_weight(awq.fold_weight(W[n], col[n], row.get(n)), col[n], row.get(n)) for n in W}
    acc, _, _ = error_pass(block, stream, W, E)
    for n, (num, den) in acc.items():
        assert num / den < 1e-12, (n, num / den)
    # v_proj and up_proj carry both an input-column and an output-row scale.
    assert set(row) == {"v_proj", "up_proj"}
    assert torch.equal(col["o_proj"], awq.expand("o", sc["o"], geom))


@torch.inference_mode()
def test_search_group_losses_and_choice(compress_fp32):
    model = compress_fp32
    geom = awq.geometry(model.config)
    stream = capture_block_inputs(model, read_sequences("c4_calibration", limit=1), batch_size=1)
    block = layers(model)[0]
    X, st, _ = awq.capture_group_inputs(block, stream)
    assert X["qkv"].shape == (512, 256) and X["down"].shape == (512, 512)
    assert torch.allclose(st["q_proj"].sumsq, (X["qkv"].float() ** 2).sum(0))
    projs = projections(block)
    ws = [projs[n].weight for n in ("gate_proj", "up_proj")]
    imp = awq.importance("gateup", st["gate_proj"].absmean(), geom)
    res = awq.search_group(ws, X["gateup"], imp, "gateup", geom, bits_list=(3, 4), alphas=(0.0, 0.5, 0.9))
    for b, r in res["by_bits"].items():
        assert len(r["losses"]) == 3 and r["losses"][r["index"]] == min(r["losses"])
        # alpha = 0 is plain group quantization.
        d0 = torch.cat([fake_quant(w.float(), b) - w.float() for w in ws])
        assert r["losses"][0] == pytest.approx(float(((X["gateup"].float() @ d0.T) ** 2).sum()), rel=1e-5)
        assert torch.allclose(r["scales"], awq.scales_for(imp, r["alpha"]))
    assert res["by_bits"][3]["losses"][0] > res["by_bits"][4]["losses"][0]
    assert res["loss_ref"] == pytest.approx(float(((X["gateup"].float() @ torch.cat(ws).T) ** 2).sum()), rel=1e-5)


def test_search_model_outputs(compress_fp32):
    model = compress_fp32
    stream = capture_block_inputs(model, read_sequences("c4_calibration", limit=1), batch_size=1)
    res = awq.search_model(model, stream, bits_list=(4,), alphas=(0.0, 0.5))
    n_layers = len(layers(model))
    assert len(res["entries"]) == n_layers * 4 and set(res["scales"]) == {
        (li, g, 4) for li in range(n_layers) for g in awq.GROUPS}
    assert res["scales"][(0, "o", 4)].shape == (128,)
    assert all(p["max_abs_err_over_max_abs_out"] < 1e-5 for p in res["preservation"])
    assert set(res["stats"]) == {(li, n) for li in range(n_layers) for n in C.ELIGIBLE}
