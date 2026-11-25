"""Group quantizer, rounding, bit packing and packed byte counts."""
import numpy as np
import pytest
import torch

from task1 import storage
from task1.quant import (dequantize, pack_bits, pack_mask, pack_quantized, quantize, unpack_bits, unpack_mask,
                         unpack_quantized)


def _group(values, gs=8):
    return torch.tensor([values], dtype=torch.float32), gs


def test_hand_checked_4bit_group():
    # min -1.5, max 6.0 -> s = 7.5/15 = 0.5 (exact in FP16); z = round(3) = 3.
    w, gs = _group([-1.5, 0.0, 0.26, 1.0, 2.2, 3.74, 5.0, 6.0])
    r = quantize(w, 4, gs)
    assert r.scale.dtype == torch.float16 and float(r.scale) == 0.5 and int(r.z) == 3
    # w/s = -3, 0, 0.52, 2, 4.4, 7.48, 10, 12 -> round -3,0,1,2,4,7,10,12 -> +3
    assert r.q.tolist() == [[0, 3, 4, 5, 7, 10, 13, 15]]
    assert dequantize(r.q, r.z, r.scale, gs).tolist() == [[-1.5, 0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 6.0]]
    assert r.n_clipped == 0 and r.n_guarded == 0 and r.n_one_sided == 0


def test_hand_checked_3bit_group_with_fp16_scale():
    # min -0.3, max 0.4 -> s32 = 0.1 -> FP16 0.0999755859375; z = round(3.0007) = 3.
    w, gs = _group([-0.3, -0.12, 0.0, 0.05, 0.149, 0.26, 0.33, 0.4])
    r = quantize(w, 3, gs)
    s = float(torch.tensor(0.7 / 7, dtype=torch.float32).to(torch.float16))
    assert float(r.scale) == s == 0.0999755859375 and int(r.z) == 3
    # w/s = -3.0007,-1.2003,0,0.5001,1.4904,2.6006,3.3008,4.001 -> -3,-1,0,1,1,3,3,4 -> +3
    assert r.q.tolist() == [[0, 2, 3, 4, 4, 6, 6, 7]]
    deq = dequantize(r.q, r.z, r.scale, gs)
    assert torch.equal(deq, torch.tensor([[-3, -1, 0, 1, 1, 3, 3, 4]], dtype=torch.float32) * s)


def test_round_half_to_even():
    # s = 1 (range 7 at 3 bits), z = 0: w = 2.5 -> 2 and 4.5 -> 4 (half to even), 3.5 -> 4.
    w, gs = _group([0.0, 2.5, 3.5, 4.5, 1.0, 5.0, 6.0, 7.0])
    r = quantize(w, 3, gs)
    assert float(r.scale) == 1.0 and int(r.z) == 0
    assert r.q.tolist() == [[0, 2, 4, 4, 1, 5, 6, 7]]


def test_constant_group_guard_and_one_sided_count():
    w = torch.zeros(1, 8)
    r = quantize(w, 4, 8)
    assert r.n_guarded == 1 and float(r.scale) == 1.0 and torch.equal(dequantize(r.q, r.z, r.scale, 8), w)
    r = quantize(torch.linspace(1, 2, 8)[None], 4, 8)
    assert r.n_one_sided == 1 and int(r.z) == 0 and r.n_clipped > 0


def test_group_layout_along_input_columns():
    torch.manual_seed(0)
    w = torch.randn(3, 256)
    r = quantize(w, 4)
    assert tuple(r.scale.shape) == (3, 2) and tuple(r.z.shape) == (3, 2)
    for g in range(2):
        part = quantize(w[:, 128 * g:128 * (g + 1)].contiguous(), 4)
        assert torch.equal(part.scale[:, 0], r.scale[:, g]) and torch.equal(part.q, r.q[:, 128 * g:128 * (g + 1)])
    err = (dequantize(r.q, r.z, r.scale) - w).abs()
    assert bool((err <= r.scale.float().repeat_interleave(128, 1) / 2 + 1e-6).all())


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_unpack_round_trip(bits):
    rng = np.random.default_rng(bits)
    for n in (1, 7, 8, 13, 1000):
        v = rng.integers(0, 1 << bits, size=n).astype(np.uint8)
        p = pack_bits(v, bits)
        assert p.dtype == np.uint8 and p.size == -(-n * bits // 8)
        assert np.array_equal(unpack_bits(p, bits, n), v)


def test_pack_little_endian_layout():
    # 3-bit values 5 (101), 3 (011), 7 (111): stream bits LSB first 1,0,1, 1,1,0, 1,1,1 -> byte0 = 0b11011101.
    p = pack_bits(np.array([5, 3, 7], dtype=np.uint8), 3)
    assert p.tolist() == [0b11011101, 0b1]
    assert pack_bits(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8), 1).tolist() == [1, 1]
    with pytest.raises(AssertionError):
        pack_bits(np.array([8], dtype=np.uint8), 3)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("shape", [(5, 256), (3, 512), (1536, 128)])
def test_packed_bytes_equal_storage(bits, shape):
    torch.manual_seed(1)
    r = quantize(torch.randn(shape), bits)
    pk = pack_quantized(r)
    want = storage.packed_quant_matrix(shape, bits)
    assert pk["qcodes"].numel() == want["codes"] and pk["qzeros"].numel() == want["zero_points"]
    assert pk["qscales"].numel() * pk["qscales"].element_size() == want["scales"]
    got = pk["qcodes"].numel() + pk["qzeros"].numel() + 2 * pk["qscales"].numel()
    assert got == want["total"] == storage.packed_quant_bytes([shape], bits)
    q, z, s = unpack_quantized(pk["qcodes"], pk["qzeros"], pk["qscales"], bits, shape)
    assert torch.equal(q, r.q.cpu()) and torch.equal(z, r.z.cpu()) and torch.equal(s, r.scale)


def test_mask_pack_round_trip_and_bytes():
    torch.manual_seed(2)
    keep = torch.rand(7, 256) > 0.5
    p = pack_mask(keep)
    assert p.numel() == -(-keep.numel() // 8) == storage.sparse16_bytes([(7, 256)], [0])
    assert torch.equal(unpack_mask(p, (7, 256)), keep)
