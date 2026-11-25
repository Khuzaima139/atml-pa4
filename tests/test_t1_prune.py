"""Per-row pruning masks, 2:4 masks, sequential Wanda statistics and score distributions."""
import pytest
import torch

from task1 import config as C
from task1 import prune as P
from task1.blocks import layers, projections
from task1.build import build_condition
from task1.data import read_sequences
from task1.model import Pristine, enumerate_eligible
from task1.quant import unpack_mask


def test_zeros_per_row_split():
    assert P.zeros_per_row(10, 4).tolist() == [3, 3, 2, 2]
    assert P.zeros_per_row(8, 4).tolist() == [2, 2, 2, 2]
    assert P.zeros_per_row(3, 5).tolist() == [1, 1, 1, 0, 0]


@pytest.mark.parametrize("total", [0, 1, 37, 128, 129, 255, 640])
def test_row_mask_exact_counts(total):
    torch.manual_seed(total)
    score = torch.rand(5, 128)
    keep = P.row_mask(score, total)
    per_row = (~keep).sum(dim=1)
    assert per_row.tolist() == P.zeros_per_row(total, 5).tolist() and int(per_row.sum()) == total
    for r in range(5):
        k = int(per_row[r])
        if 0 < k < 128:  # pruned entries are exactly the k lowest scores of the row
            assert float(score[r][~keep[r]].max()) <= float(score[r][keep[r]].min())


def test_target_zeros_integer_for_task_shapes():
    for s in C.SPARSITIES:
        for shape in set(C.FULL_MODEL_SHAPES.values()) | {(256, 256), (512, 256), (256, 512), (128, 256)}:
            z = P.target_zeros(shape, s)
            assert z % shape[0] == 0 and z == int(s * shape[0] * shape[1])
    with pytest.raises(AssertionError):
        P.target_zeros((3, 3), 0.5)


def test_ties_pruned_in_column_order():
    score = torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 0.5]])
    keep = P.row_mask(score, 3)
    assert keep.tolist() == [[False, False, True, True, True, False]]
    keep24 = P.mask_2of4(torch.tensor([[1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0, 1.0]]))
    assert keep24.tolist() == [[False, False, True, True, True, False, True, False]]


def test_mask_2of4_pattern():
    torch.manual_seed(3)
    score = torch.rand(6, 256)
    keep = P.mask_2of4(score)
    g = keep.reshape(6, 64, 4)
    assert bool((g.sum(-1) == 2).all())
    sg = score.reshape(6, 64, 4)
    kept_min = torch.where(g, sg, torch.inf).amin(-1)
    pruned_max = torch.where(~g, sg, -torch.inf).amax(-1)
    assert bool((kept_min >= pruned_max).all())


def test_magnitude_and_wanda_differ_on_loud_input():
    # Column 0 holds the smallest weights but its input is loud: magnitude prunes it, Wanda keeps it.
    w = torch.tensor([[0.1, 0.5, 0.6, 0.7], [0.2, 0.4, 0.8, 0.9]])
    x_norm = torch.tensor([100.0, 1.0, 1.0, 1.0])
    mag = P.row_mask(P.magnitude_score(w), 2)
    wan = P.row_mask(P.wanda_score(w, x_norm), 2)
    assert mag[:, 0].tolist() == [False, False] and wan[:, 0].tolist() == [True, True]
    assert P.disagreement(mag, wan) == 0.5


def test_distribution_quantiles_and_histogram():
    t = torch.tensor([0.0, 1e-3, 1e-2, 1e-1, 1.0])
    edges = P.log10_edges([P.log10_range(t)])
    assert len(edges) == P.N_BINS + 1 and edges[0] == pytest.approx(-3) and edges[-1] == pytest.approx(0)
    d = P.distribution(t, edges)
    assert d["n"] == 5 and d["n_zero"] == 1 and sum(d["log10_hist"]) == 4
    assert d["log10_hist"][0] == 1 and d["log10_hist"][-1] == 1
    assert d["quantiles"][0] == 0.0 and d["quantiles"][-1] == 1.0 and d["quantiles"][5] == pytest.approx(1e-2)
    import numpy as np
    assert d["quantiles"] == pytest.approx(np.quantile(t.double().numpy(), P.QUANTILES).tolist())


def _slow_p2_masks(model, seqs, sparsity, batch_size):
    """Reference: per block, a full model forward (earlier blocks already pruned) collects input norms."""
    masks = {}
    for li, block in enumerate(layers(model)):
        projs = projections(block)
        sums = {n: torch.zeros(m.in_features, dtype=torch.float32) for n, m in projs.items()}
        def hook(_m, a, n):
            sums[n].add_((a[0].reshape(-1, a[0].shape[-1]).float() ** 2).sum(0))

        hs = [m.register_forward_pre_hook(lambda _m, a, n=n: hook(_m, a, n)) for n, m in projs.items()]
        with torch.inference_mode():
            for a in range(0, len(seqs), batch_size):
                ids = torch.tensor([s["token_ids"] for s in seqs[a:a + batch_size]])
                model.model(input_ids=ids, use_cache=False)
        for h in hs:
            h.remove()
        for n, m in projs.items():
            w = m.weight.detach()
            score = w.float().abs() * sums[n].sqrt()[None, :]
            keep = P.row_mask(score, P.target_zeros(tuple(w.shape), sparsity))
            with torch.no_grad():
                w.masked_fill_(~keep, 0)
            masks[f"model.layers.{li}.{'self_attn' if n in C.ATTN_PROJ else 'mlp'}.{n}"] = keep
    return masks


def test_p2_sequential_equals_slow_reference(compress_fp32):
    model = compress_fp32
    seqs = read_sequences("c4_calibration", limit=3)
    pristine = Pristine(model)
    ref = _slow_p2_masks(model, seqs, 0.5, batch_size=2)
    ref_w = {e.qualname: e.module.weight.detach().clone() for e in enumerate_eligible(model)}
    pristine.restore(model)
    res = build_condition(model, seqs, "p2_s50", batch_size=2, pristine=pristine)
    for ent in res.payload:
        assert torch.equal(unpack_mask(ent["tensors"]["mask"], ent["shape"]), ref[ent["qualname"]]), ent["qualname"]
    for e in enumerate_eligible(model):
        assert torch.equal(e.module.weight, ref_w[e.qualname])
    # Sequential statistics differ from dense ones after block 0, so later masks are not the dense-Wanda masks.
    assert all(r["mask_zeros"] == r["target_zeros"] == r["shape"][0] * r["shape"][1] // 2 for r in res.records)


def test_build_pruning_records_and_exact_zero_totals(compress_fp32):
    model = compress_fp32
    seqs = read_sequences("c4_calibration", limit=2)
    pristine = Pristine(model)
    zeros = {e.qualname: e.module.weight.numel() // 3 + 1 for e in enumerate_eligible(model)}
    res = build_condition(model, seqs, "p2_s25", batch_size=2, pristine=pristine, zeros_per_matrix=zeros)
    for r, e in zip(res.records, enumerate_eligible(model)):
        assert r["mask_zeros"] == zeros[e.qualname]
        per_row = (e.module.weight == 0).sum(1)
        assert per_row.tolist() == P.zeros_per_row(zeros[e.qualname], r["shape"][0]).tolist()
        assert 0 <= r["disagreement_with_magnitude"] <= 1 and r["x_norm_cv_sequential"] > 0
        assert sum(r["dist_wanda"]["log10_hist"]) == r["dist_wanda"]["n"] - r["dist_wanda"]["n_zero"]
    assert res.score_edges["wanda_norm_source"] == "sequential"
    assert len(res.score_edges["magnitude_log10_edges"]) == P.N_BINS + 1
