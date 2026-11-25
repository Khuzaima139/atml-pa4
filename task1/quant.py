"""Group-wise asymmetric uniform weight quantizer shared by Q1, Q2 and Q3, and little-endian bit packing.

Quantizer (fixed for every quantized condition):
- groups of GROUP_SIZE contiguous input columns in each output row; the input width must be a multiple
  of the group size (no padding is needed for any Task 1 shape);
- s = (max - min) / (2^b - 1) computed in FP32, cast to FP16, and the FP16 value used everywhere;
  a group whose FP16 scale is 0 uses s = 1 (counted as guarded);
- z = clamp(round(-min / s), 0, 2^b - 1); q = clamp(round(w / s) + z, 0, 2^b - 1);
- w_hat = s * (q - z) in FP32, then cast to the model dtype; torch.round rounds half to even;
- no clipping search: min and max are the group extremes.
Packing: values flattened row-major (codes [out, in], zero-points [out, groups]), b bits each; value i
occupies bits [i*b, (i+1)*b) of a little-endian bit stream (stream bit j is bit j % 8 of byte j // 8),
padded with zero bits to a whole byte at the end of each matrix. Scales are stored as FP16 [out, groups].
Masks use the same packing with b = 1 (bit 1 = weight kept).
"""
from dataclasses import dataclass

import numpy as np
import torch

from . import config as C

QUANT_CONVENTION = (
    "asymmetric uniform, group 128 contiguous input columns per row; s=(max-min)/(2^b-1) in FP32 cast to FP16; "
    "FP16 s==0 -> s=1; z=clamp(round(-min/s),0,2^b-1); q=clamp(round(w/s)+z,0,2^b-1); w_hat=s*(q-z) in FP32 "
    "then model dtype; round half to even; no clipping search; no padding (width multiple of 128)"
)
PACKING_CONVENTION = (
    "row-major values, b bits each, little-endian bit stream (bit j -> bit j%8 of byte j//8), byte padding at the "
    "end of each matrix; codes [out,in] at b bits, zero-points [out,groups] at b bits, scales FP16 [out,groups]; "
    "masks [out,in] at 1 bit (1 = kept)"
)


@dataclass
class Quantized:
    q: torch.Tensor        # uint8 [out, in]
    z: torch.Tensor        # uint8 [out, groups]
    scale: torch.Tensor    # float16 [out, groups]
    bits: int
    n_guarded: int         # groups whose FP16 scale was 0
    n_clipped: int         # codes changed by the clamp
    n_one_sided: int       # groups with min > 0 or max < 0 (the clamp of z can clip there)


@torch.no_grad()
def quantize(w: torch.Tensor, bits: int, group_size: int = C.GROUP_SIZE) -> Quantized:
    assert w.dim() == 2 and 1 <= bits <= 8, (tuple(w.shape), bits)
    out, inp = w.shape
    assert inp % group_size == 0, (inp, group_size)
    G = inp // group_size
    maxq = (1 << bits) - 1
    wg = w.detach().float().reshape(out, G, group_size)
    wmin = wg.amin(dim=-1)
    wmax = wg.amax(dim=-1)
    s16 = ((wmax - wmin) / maxq).to(torch.float16)
    assert bool(torch.isfinite(s16).all()), "non-finite group scale"
    guard = s16 == 0
    s16 = torch.where(guard, torch.ones_like(s16), s16)
    s = s16.float()
    z = torch.clamp(torch.round(-wmin / s), 0, maxq)
    raw = torch.round(wg / s[..., None]) + z[..., None]
    q = torch.clamp(raw, 0, maxq)
    return Quantized(q=q.to(torch.uint8).reshape(out, inp), z=z.to(torch.uint8), scale=s16, bits=bits,
                     n_guarded=int(guard.sum()), n_clipped=int((raw != q).sum()),
                     n_one_sided=int(((wmin > 0) | (wmax < 0)).sum()))


@torch.no_grad()
def dequantize(q: torch.Tensor, z: torch.Tensor, scale: torch.Tensor, group_size: int = C.GROUP_SIZE) -> torch.Tensor:
    """FP32 reconstruction s * (q - z)."""
    out, inp = q.shape
    G = inp // group_size
    assert tuple(z.shape) == (out, G) and tuple(scale.shape) == (out, G) and scale.dtype == torch.float16
    qg = q.reshape(out, G, group_size).float()
    return (scale.float()[..., None] * (qg - z.float()[..., None])).reshape(out, inp)


def fake_quant(w: torch.Tensor, bits: int, group_size: int = C.GROUP_SIZE) -> torch.Tensor:
    r = quantize(w, bits, group_size)
    return dequantize(r.q, r.z, r.scale, group_size)


def pack_bits(values, bits: int) -> np.ndarray:
    """Pack unsigned values (< 2^bits) into a little-endian bit stream; returns uint8 bytes."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    v = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    assert 1 <= bits <= 8
    if bits == 8:
        return v.copy()
    assert not np.any(v >> bits), "value does not fit in the bit width"
    planes = (v[:, None] >> np.arange(bits, dtype=np.uint8)) & np.uint8(1)
    return np.packbits(planes.reshape(-1), bitorder="little")


def unpack_bits(packed, bits: int, n: int) -> np.ndarray:
    """Inverse of pack_bits for n values."""
    if isinstance(packed, torch.Tensor):
        packed = packed.detach().cpu().numpy()
    p = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    assert p.size == -(-n * bits // 8), (p.size, n, bits)
    if bits == 8:
        return p.copy()
    flat = np.unpackbits(p, count=n * bits, bitorder="little").reshape(n, bits)
    return (flat.astype(np.uint16) << np.arange(bits, dtype=np.uint16)).sum(axis=1).astype(np.uint8)


def pack_quantized(r: Quantized) -> dict:
    """Packed codes, packed zero-points and FP16 scales of one matrix (CPU)."""
    return {"qcodes": torch.from_numpy(pack_bits(r.q, r.bits)),
            "qzeros": torch.from_numpy(pack_bits(r.z, r.bits)),
            "qscales": r.scale.detach().cpu().contiguous()}


def unpack_quantized(codes, zeros, scales, bits: int, shape) -> tuple:
    out, inp = shape
    G = scales.shape[1]
    q = torch.from_numpy(unpack_bits(codes, bits, out * inp)).reshape(out, inp)
    z = torch.from_numpy(unpack_bits(zeros, bits, out * G)).reshape(out, G)
    return q, z, scales


def pack_mask(keep: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(pack_bits(keep.to(torch.uint8), 1))


def unpack_mask(packed, shape) -> torch.Tensor:
    out, inp = shape
    return torch.from_numpy(unpack_bits(packed, 1, out * inp)).reshape(out, inp).bool()
