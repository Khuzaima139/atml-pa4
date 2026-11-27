"""CPU tests of the Task 2 exact attention operators."""
import pytest
import torch

from task2 import attention_ops as ops

B, H, D = 2, 3, 16


def rand_qkv(L, dtype=torch.float32, seed=6304, d=D):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, L, d, generator=g).to(dtype) for _ in range(4))


def reference(q, k, v, mask=None):
    """Straightforward float64 causal attention with torch.softmax."""
    q, k, v = (t.double() for t in (q, k, v))
    L, d = q.shape[-2], q.shape[-1]
    allowed = torch.ones(L, L, dtype=torch.bool).tril()
    if mask is not None:
        allowed = allowed & mask
    s = (q @ k.transpose(-2, -1)) / d ** 0.5
    p = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
    return p @ v


def grads_of(fn, q, k, v, g, **kw):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fn(q, k, v, **kw)
    return (out.detach(),) + torch.autograd.grad(out, (q, k, v), g.to(out.dtype))


def tiled(tq, tk):
    return lambda q, k, v, **kw: ops.tiled_causal(q, k, v, tq, tk, **kw)


@pytest.mark.parametrize("L", [1, 7, 64, 100])
def test_dense_matches_reference_outputs_and_grads(L):
    q, k, v, g = rand_qkv(L)
    got = grads_of(ops.dense_causal, q, k, v, g)
    want = grads_of(reference, q.double(), k.double(), v.double(), g.double())
    for a, b in zip(got, want):
        torch.testing.assert_close(a.double(), b, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("L,tile", [(100, (32, 32)), (100, (3, 5)), (100, (64, 128)), (100, (128, 128)),
                                    (64, (64, 64)), (37, (1, 1)), (65, (16, 64))])
def test_tiled_matches_dense_fp32(L, tile):
    q, k, v, g = rand_qkv(L)
    got = grads_of(tiled(*tile), q, k, v, g)
    want = grads_of(ops.dense_causal, q, k, v, g)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("tile", [(32, 32), (64, 128), (7, 3)])
def test_tiled_fp16_close_to_fp32_reference(tile):
    q, k, v, g = rand_qkv(100)
    ref = reference(q, k, v)
    out16 = ops.tiled_causal(q.half(), k.half(), v.half(), *tile)
    dense16 = ops.dense_causal(q.half(), k.half(), v.half())
    assert out16.dtype == torch.float16
    assert (out16.double() - ref).abs().max() < 1e-2
    assert (out16.float() - dense16.float()).abs().max() < 1e-2
    got = grads_of(tiled(*tile), q.half(), k.half(), v.half(), g)
    assert all(torch.isfinite(t).all() for t in got)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fn", [ops.dense_causal, tiled(32, 32), tiled(64, 128), tiled(5, 7)],
                         ids=["dense", "tiled32", "tiled64x128", "tiled5x7"])
def test_causality_bitwise(dtype, fn):
    L = 100
    q, k, v, _ = rand_qkv(L, dtype)
    pk, pv, _, _ = rand_qkv(L, dtype, seed=6305)
    base = fn(q, k, v)
    for t in (0, 31, 37, 63, 98):
        k2, v2 = k.clone(), v.clone()
        k2[:, :, t + 1:] = pk[:, :, t + 1:]
        v2[:, :, t + 1:] = pv[:, :, t + 1:]
        pert = fn(q, k2, v2)
        assert torch.equal(pert[:, :, :t + 1], base[:, :, :t + 1])
        assert not torch.equal(pert[:, :, t + 1:], base[:, :, t + 1:])


@pytest.mark.parametrize("fn", [ops.dense_causal, tiled(32, 32), tiled(64, 128)], ids=["dense", "tiled32", "tiled64x128"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fully_masked_row_returns_zero_with_zero_gradient(fn, dtype):
    L, r = 100, 50
    q, k, v, g = rand_qkv(L, dtype)
    mask = torch.ones(L, L, dtype=torch.bool)
    mask[r] = False
    out, dq, dk, dv = grads_of(fn, q, k, v, g, mask=mask)
    assert torch.equal(out[:, :, r], torch.zeros_like(out[:, :, r]))
    assert torch.equal(dq[:, :, r], torch.zeros_like(dq[:, :, r]))
    assert all(torch.isfinite(t).all() for t in (out, dq, dk, dv))
    if dtype == torch.float32:
        keep = torch.arange(L) != r
        ref = reference(q, k, v, mask=mask)
        torch.testing.assert_close(out[:, :, keep].double(), ref[:, :, keep], atol=2e-6, rtol=1e-5)


def test_extra_mask_tiled_matches_dense():
    L = 100
    q, k, v, g = rand_qkv(L)
    mask = torch.rand(L, L, generator=torch.Generator().manual_seed(1)) > 0.3
    mask.fill_diagonal_(True)
    got = grads_of(tiled(16, 32), q, k, v, g, mask=mask)
    want = grads_of(ops.dense_causal, q, k, v, g, mask=mask)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("target", [1e3, 1e4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_extreme_scores_finite(target, dtype):
    L = 64
    q, k, v, g = rand_qkv(L, d=64)
    s = (q @ k.transpose(-2, -1)).tril() / 8.0
    q = q * (target / s.abs().max())
    q, k, v = (t.to(dtype) for t in (q, k, v))
    for fn in (ops.dense_causal, tiled(16, 16), tiled(64, 64)):
        out = grads_of(fn, q, k, v, g)
        assert all(torch.isfinite(t).all() for t in out)
    if dtype == torch.float32:
        a = grads_of(tiled(16, 16), q, k, v, g)
        b = grads_of(ops.dense_causal, q, k, v, g)
        torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=1e-5)


def test_tiled_skips_future_tiles_and_never_forms_full_scores(monkeypatch):
    L, tq, tk = 100, 32, 32
    q, k, v, _ = rand_qkv(L)
    shapes = []
    real = torch.matmul

    def spy(a, b):
        out = real(a, b)
        shapes.append(tuple(out.shape))
        return out

    monkeypatch.setattr(torch, "matmul", spy)
    ops.tiled_causal(q, k, v, tq, tk)
    # query tiles end at 32, 64, 96, 100 -> 1 + 2 + 3 + 4 key tiles; one q k^T and one p v matmul each
    assert len(shapes) == 2 * (1 + 2 + 3 + 4)
    for shp in shapes:
        assert shp[-2] <= tq and shp[-1] <= max(tk, D)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sdpa_matches_dense(dtype):
    q, k, v, g = rand_qkv(100, dtype)
    a = grads_of(ops.sdpa_causal, q, k, v, g)
    b = grads_of(ops.dense_causal, q.float(), k.float(), v.float(), g)
    tol = 2e-5 if dtype == torch.float32 else 2e-2
    for x, y in zip(a, b):
        torch.testing.assert_close(x.float(), y, atol=tol, rtol=tol)
    from torch.nn.attention import SDPBackend
    m = ops.sdpa_causal(q, k, v, backend=SDPBackend.MATH)
    torch.testing.assert_close(m.float(), b[0], atol=tol, rtol=tol)


def test_sdpa_with_extra_mask_matches_dense():
    L = 64
    q, k, v, _ = rand_qkv(L)
    mask = torch.rand(L, L, generator=torch.Generator().manual_seed(2)) > 0.5
    mask.fill_diagonal_(True)
    torch.testing.assert_close(ops.sdpa_causal(q, k, v, mask=mask), ops.dense_causal(q, k, v, mask=mask),
                               atol=2e-6, rtol=1e-5)


def test_forced_unsupported_backend_raises_and_is_recorded():
    from torch.nn.attention import SDPBackend
    q, k, v, _ = rand_qkv(32)
    with pytest.raises(RuntimeError):
        ops.sdpa_causal(q, k, v, backend=SDPBackend.EFFICIENT_ATTENTION)
    sup = ops.forced_backend_support(q, k, v)
    assert set(sup) == {b.name for b in ops.sdpa_backends()}
    assert "ERROR" not in sup
    assert sup["MATH"] == {"supported": True, "error": None}
    assert sup["EFFICIENT_ATTENTION"]["supported"] is False and sup["EFFICIENT_ATTENTION"]["error"]


def test_profile_and_infer_backend():
    q, k, v, g = rand_qkv(32)
    for back in (False, True):
        p = ops.profile_sdpa(q, k, v, backward=back, grad=g)
        assert "aten::scaled_dot_product_attention" in p["aten_ops"]
        assert p["cuda_kernels"] == []
        assert p["inferred_backend"] != "unknown"
    assert ops.infer_backend(["aten::_scaled_dot_product_flash_attention_for_cpu"]) == "FLASH_ATTENTION_CPU"
    assert ops.infer_backend(["aten::_scaled_dot_product_flash_attention"]) == "FLASH_ATTENTION"
    assert ops.infer_backend(["aten::_scaled_dot_product_efficient_attention"]) == "EFFICIENT_ATTENTION"
    assert ops.infer_backend(["aten::_scaled_dot_product_cudnn_attention"]) == "CUDNN_ATTENTION"
    assert ops.infer_backend(["aten::_scaled_dot_product_attention_math", "aten::bmm"]) == "MATH"
    assert ops.infer_backend(["aten::bmm"]) == "unknown"


def test_projection_formula():
    # B 2, H 4, d 64, L 8192, FP32: 4 * 2 GiB + 64 MiB mask + 9 * 16 MiB
    p = ops.dense_projected_peak_bytes(2, 4, 8192, 64, 4)
    assert p == 4 * (2 << 30) + (64 << 20) + (144 << 20)
    assert ops.gib(p) < 14
    assert ops.score_matrix_bytes(2, 4, 256, 2) == 1 << 20


def test_input_validation():
    q = torch.zeros(2, 2, 8, 4)
    with pytest.raises(ValueError):
        ops.dense_causal(q, q[:, :, :4], q)
    with pytest.raises(ValueError):
        ops.tiled_causal(q, q.half(), q, 4, 4)
    with pytest.raises(ValueError):
        ops.tiled_causal(q, q, q, 0, 4)
