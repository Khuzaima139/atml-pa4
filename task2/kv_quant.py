"""Group quantization, bit packing and packed KV-cache objects for Part C (protocol AM2.1, AM2.2, AM2.4).

Quantizer: asymmetric min/max per 64-element group, FP16 scale = (max - min) / (2^b - 1) and FP16 offset = min,
codes = clamp(round_half_even((x - offset) / scale), 0, 2^b - 1) computed in FP32 with the FP16-rounded scale and
offset. A group whose FP16 scale is 0 stores codes 0 and reconstructs to its offset.

Packing: codes go into uint8 low bits first. INT4: element 2i in bits 0-3 and element 2i+1 in bits 4-7 of byte i.
INT2: element 4i+j in bits 2j..2j+1 of byte i. A 64-element group packs into 64 b / 8 bytes without padding.

Layer caches have shape [B, H_kv, T, d_h]. Token-wise layout: groups of 64 along d_h for each token. Channel-wise
layout (keys): groups of 64 consecutive tokens for each channel.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

GROUP = 64
LAYOUTS = ("token", "channel")


# ---------------------------------------------------------------------------------------------------------------------
# Group quantizer and packing


def quantize_groups(x: torch.Tensor, bits: int):
    """Quantize the last dimension (size 64) of x; returns (codes uint8, scale fp16, offset fp16)."""
    if x.shape[-1] != GROUP:
        raise ValueError(f"last dimension must be {GROUP}, got {x.shape[-1]}")
    qmax = (1 << bits) - 1
    xf = x.float()
    mn, mx = xf.amin(-1), xf.amax(-1)
    scale = ((mx - mn) / qmax).to(torch.float16)
    offset = mn.to(torch.float16)
    s, o = scale.float().unsqueeze(-1), offset.float().unsqueeze(-1)
    nonzero = s > 0
    q = torch.round((xf - o) / torch.where(nonzero, s, torch.ones_like(s)))  # torch.round: half to even
    codes = torch.where(nonzero, q, torch.zeros_like(q)).clamp_(0, qmax).to(torch.uint8)
    return codes, scale, offset


def dequantize_groups(codes: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, dtype=torch.float16):
    """offset + scale * code, computed in FP32 and cast to dtype."""
    return (offset.float().unsqueeze(-1) + scale.float().unsqueeze(-1) * codes.float()).to(dtype)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes along the last dimension, low bits first."""
    if bits not in (2, 4):
        raise ValueError(f"bits must be 2 or 4, got {bits}")
    per = 8 // bits
    c = codes.to(torch.uint8).reshape(*codes.shape[:-1], codes.shape[-1] // per, per)
    out = c[..., 0].clone()
    for j in range(1, per):
        out |= c[..., j] << (bits * j)
    return out


def unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of pack_codes."""
    per = 8 // bits
    shifts = torch.arange(0, 8, bits, dtype=torch.uint8, device=packed.device)
    c = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return c.reshape(*packed.shape[:-1], packed.shape[-1] * per)


# ---------------------------------------------------------------------------------------------------------------------
# Quantized block of tokens for one layer and one of K or V


@dataclass
class QuantBlock:
    """Packed tokens of one layer tensor. Token layout: payload [B,H,T,d/64,8b], scale [B,H,T,d/64].
    Channel layout: payload [B,H,T/64,d,8b], scale [B,H,T/64,d]."""

    layout: str
    bits: int
    payload: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor

    @classmethod
    def encode(cls, x: torch.Tensor, layout: str, bits: int) -> "QuantBlock":
        B, H, T, d = x.shape
        if layout == "token":
            if d % GROUP:
                raise ValueError(f"head_dim {d} is not a multiple of {GROUP}")
            g = x.reshape(B, H, T, d // GROUP, GROUP)
        elif layout == "channel":
            if T % GROUP:
                raise ValueError(f"channel layout needs whole 64-token groups, got T={T}")
            g = x.reshape(B, H, T // GROUP, GROUP, d).transpose(-1, -2)
        else:
            raise ValueError(f"unknown layout {layout!r}")
        codes, scale, offset = quantize_groups(g, bits)
        return cls(layout, bits, pack_codes(codes, bits), scale, offset)

    @property
    def n_tokens(self) -> int:
        return self.payload.shape[2] * (GROUP if self.layout == "channel" else 1)

    def codes(self) -> torch.Tensor:
        return unpack_codes(self.payload, self.bits)

    def decode(self, dtype=torch.float16) -> torch.Tensor:
        """Dequantized tensor [B, H, T, d] in token order."""
        g = dequantize_groups(self.codes(), self.scale, self.offset, dtype)
        B, H = g.shape[:2]
        if self.layout == "token":
            return g.reshape(B, H, g.shape[2], -1)
        return g.transpose(-1, -2).reshape(B, H, g.shape[2] * GROUP, g.shape[3])

    def extend(self, other: "QuantBlock") -> None:
        assert (other.layout, other.bits) == (self.layout, self.bits)
        self.payload = torch.cat([self.payload, other.payload], dim=2)
        self.scale = torch.cat([self.scale, other.scale], dim=2)
        self.offset = torch.cat([self.offset, other.offset], dim=2)

    def payload_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size()

    def metadata_bytes(self) -> int:
        return self.scale.numel() * self.scale.element_size() + self.offset.numel() * self.offset.element_size()


def _tbytes(t) -> int:
    return 0 if t is None else t.numel() * t.element_size()


def relative_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """||x - x_hat||_F / ||x||_F in FP32."""
    x, x_hat = x.float(), x_hat.float()
    return float(torch.linalg.vector_norm(x - x_hat) / torch.linalg.vector_norm(x))


# ---------------------------------------------------------------------------------------------------------------------
# Packed cache


@dataclass
class PackedLayer:
    k_q: QuantBlock | None = None
    v_q: QuantBlock | None = None
    k_fp: torch.Tensor | None = None  # FP16 retained positions [B, H, F, d]
    v_fp: torch.Tensor | None = None
    k_tail: torch.Tensor | None = None  # FP16 tail [B, H, r, d]: trailing context and appended entries
    v_tail: torch.Tensor | None = None


@dataclass
class PackedKVCache:
    """Stored form of one method's cache. Positions are absolute token positions shared by every layer and head.

    q_positions: positions held in the quantized blocks (in block order, ascending).
    fp_positions: positions held as FP16 retained entries (ascending).
    tail_positions: positions held in the FP16 tail (ascending, after every other position).
    indices: the stored position list counted in bytes (int32), or None when the positions are implied.
    """

    method: str
    context_len: int
    layers: list[PackedLayer]
    q_positions: torch.Tensor
    fp_positions: torch.Tensor
    tail_positions: torch.Tensor
    indices: torch.Tensor | None
    k_layout: str | None = None
    bits: int | None = None
    quantize_tail: bool = False
    context_bytes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ builders
    @staticmethod
    def _empty(x, n=0):
        return x.new_empty(x.shape[0], x.shape[1], n, x.shape[3])

    @classmethod
    def _finish(cls, obj: "PackedKVCache") -> "PackedKVCache":
        obj.context_bytes = obj.nbytes()
        return obj

    @classmethod
    def fp16(cls, keys: list, values: list) -> "PackedKVCache":
        """Identity compression: every context position kept as stored, no position list."""
        L = keys[0].shape[2]
        dev = keys[0].device
        layers = [PackedLayer(k_fp=k, v_fp=v, k_tail=cls._empty(k), v_tail=cls._empty(v)) for k, v in zip(keys, values)]
        ar = torch.arange(L, device=dev)
        return cls._finish(cls("fp16", L, layers, ar[:0], ar, ar[:0], None))

    @classmethod
    def evict(cls, method: str, keys: list, values: list, positions: torch.Tensor, indices: torch.Tensor):
        """FP16 eviction: keep `positions` (ascending); `indices` is the stored int32 form."""
        L = keys[0].shape[2]
        pos = positions.to(keys[0].device)
        if not bool((pos[1:] > pos[:-1]).all()):
            raise ValueError("positions must be strictly ascending")
        layers = [
            PackedLayer(k_fp=k[:, :, pos].contiguous(), v_fp=v[:, :, pos].contiguous(),
                        k_tail=cls._empty(k), v_tail=cls._empty(v))
            for k, v in zip(keys, values)
        ]
        return cls._finish(cls(method, L, layers, pos[:0], pos, pos[:0], indices.to(torch.int32)))

    @classmethod
    def uniform(cls, method: str, keys: list, values: list, k_layout: str, bits: int = 4):
        """Every complete 64-token prefix group quantized (K in k_layout, V token-wise); trailing r stays FP16."""
        L = keys[0].shape[2]
        n_q = (L // GROUP) * GROUP
        layers = []
        for k, v in zip(keys, values):
            layers.append(PackedLayer(
                k_q=QuantBlock.encode(k[:, :, :n_q], k_layout, bits),
                v_q=QuantBlock.encode(v[:, :, :n_q], "token", bits),
                k_tail=k[:, :, n_q:].contiguous(), v_tail=v[:, :, n_q:].contiguous(),
            ))
        ar = torch.arange(L, device=keys[0].device)
        return cls._finish(cls(method, L, layers, ar[:n_q], ar[:0], ar[n_q:], None, k_layout, bits, True))

    @classmethod
    def hybrid(cls, keys: list, values: list, fp_positions: torch.Tensor, low_bits: int = 2):
        """FP16 at fp_positions (stored as an int32 list); the ordered complement in low-bit groups
        (channel-wise K, token-wise V). The complement must hold whole 64-token groups."""
        L = keys[0].shape[2]
        dev = keys[0].device
        fp = fp_positions.to(dev)
        mask = torch.ones(L, dtype=torch.bool, device=dev)
        mask[fp] = False
        low = torch.nonzero(mask).flatten()
        if low.numel() % GROUP:
            raise ValueError(f"{low.numel()} low-bit positions do not form whole {GROUP}-token groups")
        layers = []
        for k, v in zip(keys, values):
            layers.append(PackedLayer(
                k_q=QuantBlock.encode(k[:, :, low], "channel", low_bits) if low.numel() else None,
                v_q=QuantBlock.encode(v[:, :, low], "token", low_bits) if low.numel() else None,
                k_fp=k[:, :, fp].contiguous(), v_fp=v[:, :, fp].contiguous(),
                k_tail=cls._empty(k), v_tail=cls._empty(v),
            ))
        return cls._finish(cls("hybrid", L, layers, low, fp, fp[:0], fp.to(torch.int32), "channel", low_bits, False))

    # ------------------------------------------------------------------ accounting
    def nbytes(self) -> dict:
        """Live bytes by component, counted from the stored tensors (per batch as stored)."""
        payload = metadata = retained = tail = 0
        for lay in self.layers:
            for qb in (lay.k_q, lay.v_q):
                if qb is not None:
                    payload += qb.payload_bytes()
                    metadata += qb.metadata_bytes()
            retained += _tbytes(lay.k_fp) + _tbytes(lay.v_fp)
            tail += _tbytes(lay.k_tail) + _tbytes(lay.v_tail)
        idx = _tbytes(self.indices)
        return {"payload": payload, "metadata": metadata, "fp16_retained": retained, "fp16_tail": tail,
                "indices": idx, "total": payload + metadata + retained + tail + idx}

    @property
    def n_appended(self) -> int:
        """Entries at positions >= context_len (question suffix and fed-back tokens)."""
        return int((self.positions() >= self.context_len).sum())

    def context_positions_kept(self) -> int:
        return self.n_positions - self.n_appended

    @property
    def n_positions(self) -> int:
        return self.q_positions.numel() + self.fp_positions.numel() + self.tail_positions.numel()

    # ------------------------------------------------------------------ reconstruction
    def positions(self) -> torch.Tensor:
        """Absolute positions of the physical cache entries, in physical (ascending) order."""
        return torch.cat([torch.sort(torch.cat([self.q_positions, self.fp_positions])).values, self.tail_positions])

    def layer_tensors(self, i: int, dtype=None):
        """Dequantized (K, V) of layer i in physical order: retained context positions ascending, then the tail."""
        lay = self.layers[i]
        dtype = dtype or lay.k_tail.dtype
        ks, vs = [], []
        if lay.k_q is not None:
            ks.append(lay.k_q.decode(dtype))
            vs.append(lay.v_q.decode(dtype))
        if lay.k_fp is not None:
            ks.append(lay.k_fp.to(dtype))
            vs.append(lay.v_fp.to(dtype))
        k, v = torch.cat(ks, dim=2), torch.cat(vs, dim=2)
        if self.q_positions.numel() and self.fp_positions.numel():
            order = torch.argsort(torch.cat([self.q_positions, self.fp_positions]))
            k, v = k[:, :, order], v[:, :, order]
        return torch.cat([k, lay.k_tail.to(dtype)], dim=2), torch.cat([v, lay.v_tail.to(dtype)], dim=2)

    def to_dynamic_cache(self, config, dtype=None):
        from transformers import DynamicCache

        cache = DynamicCache(config=config)
        for i in range(len(self.layers)):
            k, v = self.layer_tensors(i, dtype)
            cache.update(k, v, i)
        return cache

    def reconstruction_errors(self, keys: list, values: list) -> dict | None:
        """Relative Frobenius errors of K and V over all context positions, per layer and mean over layers.
        Defined only while every context position is held (FP16, uniform and hybrid caches) and before appends."""
        if self.n_positions != self.context_len or self.context_positions_kept() != self.context_len:
            return None
        ek, ev = [], []
        for i, (k, v) in enumerate(zip(keys, values)):
            kh, vh = self.layer_tensors(i, k.dtype)
            ek.append(relative_error(k, kh))
            ev.append(relative_error(v, vh))
        return {"k_per_layer": ek, "v_per_layer": ev, "k_mean": sum(ek) / len(ek), "v_mean": sum(ev) / len(ev)}

    # ------------------------------------------------------------------ appends
    def append(self, new_k: list, new_v: list, new_positions: torch.Tensor) -> int:
        """Add appended entries (one tensor [B, H, q, d] per layer) to the FP16 tail. In the quantized layouts every
        complete 64-token tail group is quantized. Returns the number of groups quantized per layer."""
        new_positions = new_positions.to(self.tail_positions.device)
        self.tail_positions = torch.cat([self.tail_positions, new_positions])
        n_groups = (self.tail_positions.numel() // GROUP) if self.quantize_tail else 0
        for lay, k, v in zip(self.layers, new_k, new_v):
            lay.k_tail = torch.cat([lay.k_tail, k.to(lay.k_tail.dtype)], dim=2)
            lay.v_tail = torch.cat([lay.v_tail, v.to(lay.v_tail.dtype)], dim=2)
            if n_groups:
                n = n_groups * GROUP
                kq = QuantBlock.encode(lay.k_tail[:, :, :n], self.k_layout, self.bits)
                vq = QuantBlock.encode(lay.v_tail[:, :, :n], "token", self.bits)
                lay.k_q.extend(kq)
                lay.v_q.extend(vq)
                lay.k_tail = lay.k_tail[:, :, n:].contiguous()
                lay.v_tail = lay.v_tail[:, :, n:].contiguous()
        if n_groups:
            n = n_groups * GROUP
            self.q_positions = torch.cat([self.q_positions, self.tail_positions[:n]])
            self.tail_positions = self.tail_positions[n:]
        return n_groups

    def clone(self, device=None) -> "PackedKVCache":
        """Deep copy; with a device, every tensor is copied to that device (no copy is made on the source device)."""
        if device is None:
            return copy.deepcopy(self)

        def mv(t):
            return None if t is None else t.to(device, copy=True)

        def mvq(qb):
            return None if qb is None else QuantBlock(qb.layout, qb.bits, mv(qb.payload), mv(qb.scale), mv(qb.offset))

        layers = [PackedLayer(mvq(lay.k_q), mvq(lay.v_q), mv(lay.k_fp), mv(lay.v_fp), mv(lay.k_tail), mv(lay.v_tail))
                  for lay in self.layers]
        return PackedKVCache(self.method, self.context_len, layers, mv(self.q_positions), mv(self.fp_positions),
                             mv(self.tail_positions), mv(self.indices), self.k_layout, self.bits, self.quantize_tail,
                             dict(self.context_bytes))
