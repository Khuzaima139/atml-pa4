"""Task 2 Part B units: model size and initialization, masks, linear attention, loss, schedule, data streams, cache."""
import json

import numpy as np
import pytest
import torch

from task2 import mqar_data as D
from task2 import mqar_model as M
from task2 import train_mqar as TR


@pytest.fixture(scope="module")
def gen():
    return D.load_generator()


# ---------------------------------------------------------------------------------------------------------------
# Model

@pytest.mark.parametrize("mech", M.MECHANISMS)
def test_parameter_count(mech):
    assert M.count_params(M.build_model(mech, 6304)) == M.DENSE_PARAMS == 462_336


def test_identical_initial_weights_across_mechanisms():
    states = {m: M.build_model(m, 6305).state_dict() for m in M.MECHANISMS}
    ref = states["dense"]
    for m, st in states.items():
        assert list(st) == list(ref)
        for k in ref:
            assert st[k].shape == ref[k].shape and torch.equal(st[k], ref[k]), (m, k)
    other = M.build_model("dense", 6306).state_dict()
    assert not torch.equal(other["tok.weight"], ref["tok.weight"])


def test_build_model_independent_of_global_rng():
    a = M.build_model("window", 6304).state_dict()
    torch.rand(1000)
    b = M.build_model("window", 6304).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_head_has_no_bias_and_projections_have_bias():
    m = M.build_model("dense", 6304)
    assert m.head.bias is None
    blk = m.blocks[0]
    for lin in (blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o, blk.fc1, blk.fc2):
        assert lin.bias is not None
    assert blk.ln1.bias is not None and m.ln_f.bias is not None


@pytest.mark.parametrize("T", [10, 70, 300])
def test_window_mask_contents(T):
    m = M.window_mask(T)
    for i in range(T):
        for j in range(T):
            assert bool(m[i, j]) == (i - 63 <= j <= i)
    assert int(m.sum(1)[-1]) == min(T, 64)
    assert not torch.triu(m, diagonal=1).any()


@pytest.mark.parametrize("T", [10, 70, 300])
def test_dilated_mask_contents(T):
    m = M.dilated_mask(T)
    for i in range(T):
        allowed = {i - 4 * n for n in range(64) if i - 4 * n >= 0}
        assert set(torch.nonzero(m[i]).flatten().tolist()) == allowed
    assert not torch.triu(m, diagonal=1).any()
    if T == 300:
        assert bool(m[299, 299 - 252]) and not bool(m[299, 299 - 256])
        assert int(m[299].sum()) == 64


def test_local_dilated_mask_per_head():
    T = 120
    m = M.local_dilated_mask(T)
    assert m.shape == (4, T, T) and m.dtype == torch.bool
    assert torch.equal(m[0], M.window_mask(T)) and torch.equal(m[1], M.window_mask(T))
    assert torch.equal(m[2], M.dilated_mask(T)) and torch.equal(m[3], M.dilated_mask(T))


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_linear_cumsum_equals_recurrent(dtype, tol):
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 4, 37, 32, generator=g, dtype=dtype) for _ in range(3))
    a = M.linear_attention_cumsum(q, k, v)
    b = M.linear_attention_recurrent(q, k, v)
    assert (a - b).abs().max() < tol


def test_linear_attention_is_causal():
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(1, 4, 20, 32, generator=g) for _ in range(3))
    base = M.linear_attention_cumsum(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 11:] += 5.0
    v2[:, :, 11:] -= 3.0
    pert = M.linear_attention_cumsum(q, k2, v2)
    assert torch.equal(base[:, :, :11], pert[:, :, :11])
    assert not torch.allclose(base[:, :, 11:], pert[:, :, 11:])


@pytest.mark.parametrize("mech", M.MECHANISMS)
def test_model_is_causal(mech):
    model = M.build_model(mech, 6304).eval()
    g = torch.Generator().manual_seed(3)
    tok = torch.randint(0, 256, (2, 90), generator=g)
    tok2 = tok.clone()
    tok2[:, 50:] = torch.randint(0, 256, (2, 40), generator=g)
    with torch.no_grad():
        a, b = model(tok), model(tok2)
    assert torch.allclose(a[:, :50], b[:, :50], atol=1e-5)


def test_window64_differs_only_beyond_63():
    model = M.build_model("dense", 6304).eval()
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        short = torch.randint(0, 256, (2, 64), generator=g)
        a = model(short)
        with model.mask_override("window64"):
            b = model(short)
        assert torch.allclose(a, b, atol=1e-5)
        long = torch.randint(0, 256, (2, 140), generator=g)
        a = model(long)
        with model.mask_override("window64"):
            assert model.operator == "window"
            b = model(long)
        assert model.operator == "dense"
    assert torch.allclose(a[:, :64], b[:, :64], atol=1e-5)
    assert (a[:, 64:] - b[:, 64:]).abs().amax(dim=-1).min() > 1e-4


def test_window64_refused_for_other_mechanisms():
    with pytest.raises(ValueError, match="dense-trained"):
        with M.build_model("linear", 6304).mask_override("window64"):
            pass


def test_theoretical_costs_values():
    c = M.all_theoretical_costs(256)
    assert c["dense"]["attended_pairs_per_layer"] == 4 * 32_896
    assert c["window"]["per_head_type"]["window"]["attended_pairs_per_head_per_layer"] == 14_368
    assert c["local_dilated"]["per_head_type"]["dilated"]["attended_pairs_per_head_per_layer"] == 8_320
    assert c["local_dilated"]["attended_pairs_per_layer"] == 45_376
    assert c["local_dilated"]["union_candidates_at_last_position"] == 112
    assert c["linear"]["persistent_state"]["bytes_per_layer_fp32"] == 16_896
    assert c["dense"]["kv_cache"]["bytes_per_layer_fp32"] == 262_144
    assert c["window"]["kv_cache"]["bytes_per_layer_fp32"] == 65_536


def test_materialization_bytes():
    assert M.materialization("dense", 2, 256)["dense_mask_bytes"] == 0
    assert M.materialization("window", 2, 256)["dense_mask_bytes"] == 256 * 256
    assert M.materialization("local_dilated", 2, 256)["dense_mask_bytes"] == 4 * 256 * 256
    lin = M.materialization("linear", 2, 256)
    assert lin["linear_prefix"] and lin["linear_prefix_bytes_per_layer"] == 2 * 4 * 256 * 32 * 32 * 4


# ---------------------------------------------------------------------------------------------------------------
# Loss and schedule

def test_loss_uses_only_query_positions_and_value_slice():
    g = torch.Generator().manual_seed(5)
    B, T, K = 3, 40, 4
    logits = torch.randn(B, T, 256, generator=g, requires_grad=True)
    qpos = torch.stack([torch.randperm(T - 10, generator=g)[:K] + 10 for _ in range(B)])
    ans = torch.randint(128, 192, (B, K), generator=g)
    loss, acc = TR.query_loss(logits, qpos, ans)
    loss.backward()
    grad = logits.grad
    allowed = torch.zeros(B, T, 256, dtype=torch.bool)
    for b in range(B):
        allowed[b, qpos[b], 128:192] = True
    assert (grad[~allowed] == 0).all()
    assert (grad[allowed] != 0).all()
    pert = logits.detach().clone()
    pert[~allowed] += 100 * torch.randn(int((~allowed).sum()), generator=g)
    assert torch.equal(TR.query_loss(pert, qpos, ans)[0], loss.detach())
    sub = torch.stack([logits.detach()[b, qpos[b], 128:192] for b in range(B)])
    ref = torch.nn.functional.cross_entropy(sub.reshape(-1, 64), (ans - 128).reshape(-1))
    assert torch.allclose(loss.detach(), ref)
    assert float(acc) == float((sub.argmax(-1) == ans - 128).float().mean())


def test_lr_schedule_values():
    assert TR.lr_at(0) == 0.0
    assert TR.lr_at(200) == pytest.approx(8e-4, rel=0, abs=1e-15)
    assert TR.lr_at(4500) == pytest.approx(0.15 * 8e-4, rel=1e-12)
    assert TR.lr_at(1) == pytest.approx(8e-4 / 200)
    vals = [TR.lr_at(u) for u in range(4501)]
    assert all(a < b for a, b in zip(vals[:200], vals[1:201]))
    assert all(a > b for a, b in zip(vals[200:-1], vals[201:]))
    with pytest.raises(ValueError):
        TR.lr_at(4501)


def test_scaled_checkpoints():
    assert TR.scaled_checkpoints(4500) == [500, 1000, 1500, 2000, 3000, 4500]
    assert TR.scaled_checkpoints(9) == [1, 2, 3, 4, 6, 9]
    assert TR.scaled_warmup(4500) == 200 and TR.scaled_warmup(9) == 1


# ---------------------------------------------------------------------------------------------------------------
# Data

def test_seed_index_ranges():
    assert D.seed_index_range(6304) == (0, 144_000)
    assert D.seed_index_range(6305) == (144_000, 288_000)
    assert D.seed_index_range(6306) == (288_000, 432_000)
    assert D.seed_index_range(6305, n_updates=3) == (144_000, 144_096)
    for bad in (6303, 6307):
        with pytest.raises(ValueError):
            D.seed_index_range(bad)


def test_generator_sha_checked(gen):
    mod, info = gen
    assert info["sha256"] == info["manifest_sha256"]
    assert mod.make_example("train", 4, "near", 0)["id"] == "train-near-K4-00000"


def test_stream_rows_match_generator_in_index_order(gen):
    mod, _ = gen
    arr = D.build_stream(mod, 8, "mixed", 144_000, 144_064)
    assert arr.dtype == np.uint8 and arr.shape == (64, 256 + 24)
    for n in (0, 31, 63):
        ex = mod.make_example("train", 8, "mixed", 144_000 + n)
        back = D.decode_row(arr[n], 8, "mixed", 144_000 + n)
        for key in ("id", "tokens", "query_positions", "source_positions", "answer_ids", "distances"):
            assert back[key] == ex[key]
    info = D.validate_stream(mod, arr, 8, "mixed", 144_000)
    assert info["batches_checked"] == 2 and info["sequences_checked"] == 64


def test_validate_stream_catches_corruption(gen):
    mod, _ = gen
    arr = D.build_stream(mod, 4, "near", 0, 32)
    bad = arr.copy()
    bad[5, 256 + 8] = 130 if bad[5, 256 + 8] != 130 else 131  # an answer id
    with pytest.raises(AssertionError):
        D.validate_stream(mod, bad, 4, "near", 0)
    bad = arr.copy()
    filler = [p for p in range(8, 256) if p not in set(arr[3, 256:260].tolist())][0]
    bad[3, filler] = 150  # a value id in a filler slot
    with pytest.raises(AssertionError, match="non-filler"):
        D.validate_stream(mod, bad, 4, "near", 0)


def test_disjointness_check_and_forced_collision(gen):
    mod, _ = gen
    arr = D.build_stream(mod, 4, "near", 0, 32)
    ref = D.heldout_hashes([D.DEV_FILE, D.TEST_FILE])
    assert len(ref) == 2304 + 9216
    info = D.check_disjoint(arr, ref, 0, 4, "near")
    assert info["collisions"] == 0 and info["n_train_hashed"] == 32
    dev_tokens = D.read_eval_file(D.DEV_FILE, "development", limit_per_cell=1)["near_K4"]["tokens"][0]
    forced = arr.copy()
    forced[7, :256] = dev_tokens
    with pytest.raises(D.DisjointnessError, match="development-near-K4-00000"):
        D.check_disjoint(forced, ref, 0, 4, "near")


def _meta(meta_path):
    return json.loads(meta_path.read_text())


def test_cache_reuse_and_rebuild(gen, tmp_path):
    mod, info = gen
    cache = tmp_path / "cache"
    arr, rec = D.load_or_build_stream(cache, mod, info, 4, "near", 0, 64)
    assert rec["built"] and rec["rebuild_reason"] is None
    npy, meta_path, _ = D.cache_paths(cache, 4, "near", 0, 64)
    sha = rec["sha256"]
    assert _meta(meta_path)["sha256"] == sha

    arr2, rec2 = D.load_or_build_stream(cache, mod, info, 4, "near", 0, 64)
    assert not rec2["built"] and rec2["sha256"] == sha and np.array_equal(arr, arr2)

    raw = bytearray(npy.read_bytes())  # flip one byte of the payload: sha mismatch
    raw[-1] ^= 1
    npy.write_bytes(bytes(raw))
    arr3, rec3 = D.load_or_build_stream(cache, mod, info, 4, "near", 0, 64)
    assert rec3["built"] and "sha256" in rec3["rebuild_reason"]
    assert rec3["sha256"] == sha and np.array_equal(arr3, arr)

    m = _meta(meta_path)  # sidecar with a different index range
    m["stop"] = 96
    meta_path.write_text(json.dumps(m))
    _, rec4 = D.load_or_build_stream(cache, mod, info, 4, "near", 0, 64)
    assert rec4["built"] and "stop" in rec4["rebuild_reason"]

    bad = arr.copy()  # tampered rows with a consistent sidecar sha: caught by the spot check
    bad[0, 100] = 2 if bad[0, 100] != 2 else 3
    np.save(npy, bad, allow_pickle=False)
    m = _meta(meta_path)
    m["sha256"] = D.sha256_file(npy)
    meta_path.write_text(json.dumps(m))
    arr5, rec5 = D.load_or_build_stream(cache, mod, info, 4, "near", 0, 64)
    assert rec5["built"] and "differs from the generator" in rec5["rebuild_reason"]
    assert np.array_equal(arr5, arr)


def test_read_eval_file_and_query_records(gen):
    mod, _ = gen
    cells = D.read_eval_file(D.DEV_FILE, "development", gen=mod, limit_per_cell=3)
    assert list(cells) == D.cell_order() and len(cells) == 9
    for name, c in cells.items():
        K = c["K"]
        assert name == f"{c['regime']}_K{K}"
        assert c["tokens"].shape == (3, 256) and c["query_position"].shape == (3, K)
        assert list(c["sequence_index"]) == [0, 1, 2]
        assert np.array_equal(c["distance"], c["query_position"] - c["source_position"])
        rows = np.arange(3)[:, None]
        assert np.array_equal(c["tokens"][rows, c["source_position"] + 1], c["answer_id"])
        assert np.array_equal(c["tokens"][rows, c["query_position"]], c["tokens"][rows, c["source_position"]])
    recs = D.query_records(cells)
    assert len(recs) == 3 * 3 * (4 + 8 + 16)
    assert set(recs[0]) == {"cell", "id", "sequence_index", "K", "regime", "query_position", "source_position",
                            "distance", "answer_id"}
    full = D.read_eval_file(D.TEST_FILE, "test")
    assert all(len(c["ids"]) == 1024 for c in full.values())
