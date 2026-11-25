"""Compressed-model artifacts: safetensors + JSON manifest, and a loader that rebuilds the executed model.

Pruning: per matrix a bit-packed keep mask ("<qualname>.mask", 1 bit per weight, little bit order along
input columns, row-major) and its nonzero count; the kept values are the restored FP16 weights.
Quantization: per matrix packed codes ("<qualname>.qcodes"), packed zero-points ("<qualname>.qzeros"),
FP16 scales ("<qualname>.qscales") and the bit width; folded non-eligible tensors (input_layernorm,
post_attention_layernorm, v_proj bias) under their parameter names when present.
Logical bytes follow task1.storage (sparse16 for pruning, packed group quantization for quantization).
"""
from pathlib import Path

import torch

from . import config as C
from . import storage
from .model import enumerate_eligible
from .prune import PRUNE_CONVENTION
from .quant import PACKING_CONVENTION, QUANT_CONVENTION, dequantize, unpack_mask, unpack_quantized
from .tensorfile import load_tensors, save_tensors, tensor_sha256

ARTIFACT_FILE = "artifact.safetensors"
MANIFEST_FILE = "manifest.json"


def save_artifact(art_dir, condition: str, model_source: dict, payload: list, extra_tensors: dict, model) -> dict:
    """Write the artifact of a built condition; returns the manifest."""
    art_dir = Path(art_dir)
    elig = enumerate_eligible(model)
    params = dict(model.named_parameters())
    n_non_elig = sum(p.numel() for p in model.parameters()) - sum(e.module.weight.numel() for e in elig)
    shapes = [tuple(e.module.weight.shape) for e in elig]
    by_qual = {m["qualname"]: m for m in payload}
    assert not payload or [m["qualname"] for m in payload] == [e.qualname for e in elig]
    tensors, matrices = {}, []
    kinds = {m["kind"] for m in payload} or {"dense"}
    assert len(kinds) == 1, kinds
    kind = kinds.pop()
    for e in elig:
        w = e.module.weight
        nnz = int(torch.count_nonzero(w))
        ent = {"qualname": e.qualname, "kind": kind, "shape": list(w.shape), "nonzeros": nnz}
        m = by_qual.get(e.qualname)
        if kind == "mask":
            tensors[e.qualname + ".mask"] = m["tensors"]["mask"]
            ent["mask_kept"] = m["mask_kept"]
            ent["logical_bytes"] = storage.sparse16_bytes([tuple(w.shape)], [nnz])
            assert m["tensors"]["mask"].numel() == -(-w.numel() // 8)
        elif kind == "quant":
            for k in ("qcodes", "qzeros", "qscales"):
                tensors[f"{e.qualname}.{k}"] = m["tensors"][k]
            pq = storage.packed_quant_matrix(tuple(w.shape), m["bits"])
            assert m["tensors"]["qcodes"].numel() == pq["codes"], (e.qualname, pq)
            assert m["tensors"]["qzeros"].numel() == pq["zero_points"], (e.qualname, pq)
            assert m["tensors"]["qscales"].numel() * 2 == pq["scales"], (e.qualname, pq)
            ent.update(bits=m["bits"], groups=pq["groups"], logical_bytes=pq["total"])
        else:
            ent["logical_bytes"] = storage.fp16_bytes([tuple(w.shape)])
        matrices.append(ent)
    for name, t in extra_tensors.items():
        assert name in params and params[name].dtype == t.dtype, name
        tensors[name] = t
    eligible_bytes = sum(m["logical_bytes"] for m in matrices)
    if kind == "quant":
        assert eligible_bytes == storage.packed_quant_bytes(shapes, [m["bits"] for m in matrices])
    elif kind == "mask":
        assert eligible_bytes == storage.sparse16_bytes(shapes, [m["nonzeros"] for m in matrices])
    meta = {"eval_id": C.EVAL_ID, "condition": condition, "family": C.CONDITIONS[condition]["family"], "kind": kind,
            "model": model_source, "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "conventions": {"quantizer": QUANT_CONVENTION, "packing": PACKING_CONVENTION,
                            "pruning": PRUNE_CONVENTION, "storage": storage.STORAGE_CONVENTION},
            "matrices": matrices, "extra_tensors": sorted(extra_tensors),
            "logical_bytes": {"eligible": eligible_bytes, "non_eligible_params": n_non_elig,
                              "whole_model": storage.whole_model_bytes(eligible_bytes, n_non_elig),
                              "format": {"mask": "sparse16", "quant": "packed_group_quant",
                                         "dense": "fp16_dense"}[kind]}}
    return save_tensors(art_dir / ARTIFACT_FILE, tensors, meta, art_dir / MANIFEST_FILE)


@torch.no_grad()
def load_artifact(model, pristine, art_dir) -> dict:
    """Restore the FP16 weights, then apply the artifact (sha256 values checked first); returns the manifest."""
    art_dir = Path(art_dir)
    tensors, manifest = load_tensors(art_dir / ARTIFACT_FILE, art_dir / MANIFEST_FILE)
    pristine.restore(model)
    elig = {e.qualname: e.module for e in enumerate_eligible(model)}
    assert [m["qualname"] for m in manifest["matrices"]] == list(elig)
    for m in manifest["matrices"]:
        w = elig[m["qualname"]].weight
        assert list(w.shape) == m["shape"], m["qualname"]
        if m["kind"] == "mask":
            keep = unpack_mask(tensors[m["qualname"] + ".mask"], m["shape"]).to(w.device)
            w.masked_fill_(~keep, 0)
        elif m["kind"] == "quant":
            q, z, s = unpack_quantized(tensors[m["qualname"] + ".qcodes"], tensors[m["qualname"] + ".qzeros"],
                                       tensors[m["qualname"] + ".qscales"], m["bits"], m["shape"])
            w.copy_(dequantize(q.to(w.device), z.to(w.device), s.to(w.device)).to(w.dtype))
        assert int(torch.count_nonzero(w)) == m["nonzeros"], m["qualname"]
    params = dict(model.named_parameters())
    for name in manifest["extra_tensors"]:
        t = tensors[name]
        assert params[name].dtype == t.dtype and params[name].shape == t.shape, name
        params[name].copy_(t)
    return manifest


def executed_hashes(model, extra_names=()) -> dict:
    """sha256 of every eligible weight and of the named extra parameters."""
    params = dict(model.named_parameters())
    out = {e.qualname + ".weight": tensor_sha256(e.module.weight) for e in enumerate_eligible(model)}
    for n in extra_names:
        out[n] = tensor_sha256(params[n])
    return out
