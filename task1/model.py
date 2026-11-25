"""Model loading, eligible-projection enumeration, pristine weight snapshot and parameter counts."""
from dataclasses import dataclass

import torch
from torch import nn

from . import config as C

_FP_CHUNK = 1 << 24  # elements per reduction chunk for fingerprints


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(model_path=None):
    """Tokenizer from a local directory, or the locked revision of the checkpoint."""
    from transformers import AutoTokenizer

    if model_path is not None:
        return AutoTokenizer.from_pretrained(model_path)
    return AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.MODEL_REVISION)


def load_model(device, model_path=None, dtype: str = C.DTYPE):
    """Load the causal LM at the locked revision (or a local path) in FP16 with SDPA, eval mode.

    Returns (model, info) where info records the source, dtype and effective attention backend.
    """
    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": _dtype(dtype), "attn_implementation": C.ATTN_IMPLEMENTATION}
    if model_path is None:
        model = AutoModelForCausalLM.from_pretrained(C.MODEL_ID, revision=C.MODEL_REVISION, **kwargs)
        source = {"model_id": C.MODEL_ID, "revision": C.MODEL_REVISION}
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        source = {"model_path": str(model_path)}
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    info = {
        **source,
        "requested_dtype": dtype,
        "parameter_dtypes": dtypes,
        "attn_implementation_requested": C.ATTN_IMPLEMENTATION,
        "attn_implementation_effective": model.config._attn_implementation,
        "model_class": type(model).__name__,
        "device": str(next(model.parameters()).device),
    }
    assert dtypes == [str(_dtype(dtype))], dtypes
    return model, info


@dataclass(frozen=True)
class Eligible:
    layer: int
    name: str
    qualname: str
    module: nn.Linear


def _blocks(model):
    return model.model.layers


def enumerate_eligible(model) -> list:
    """Eligible projections in fixed order (layer, then q,k,v,o,gate,up,down) with checks."""
    shapes = C.expected_shapes(model.config)
    out = []
    for li, block in enumerate(_blocks(model)):
        for name in C.ELIGIBLE:
            parent = block.self_attn if name in C.ATTN_PROJ else block.mlp
            sub = "self_attn" if name in C.ATTN_PROJ else "mlp"
            mod = getattr(parent, name)
            assert isinstance(mod, nn.Linear), (li, name, type(mod))
            assert tuple(mod.weight.shape) == shapes[name], (li, name, tuple(mod.weight.shape))
            out.append(Eligible(li, name, f"model.layers.{li}.{sub}.{name}", mod))
    n_layers = len(_blocks(model))
    assert len(out) == n_layers * len(C.ELIGIBLE), len(out)
    per_block = {}
    for e in out:
        per_block[e.layer] = per_block.get(e.layer, 0) + 1
    assert set(per_block.values()) == {len(C.ELIGIBLE)}, per_block

    # Exclusions: embeddings, norms, biases and the LM head are not eligible weights.
    elig_ptrs = {e.module.weight.data_ptr() for e in out}
    assert len(elig_ptrs) == len(out)
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    assert emb.data_ptr() not in elig_ptrs and head.data_ptr() not in elig_ptrs
    for pname, p in model.named_parameters():
        if pname.endswith(".bias") or "norm" in pname.split(".")[-2] or "embed" in pname:
            assert p.data_ptr() not in elig_ptrs, pname
    # Every nn.Linear in the model is either eligible or the LM head.
    linears = {m.weight.data_ptr() for m in model.modules() if isinstance(m, nn.Linear)}
    assert linears == elig_ptrs | {head.data_ptr()}, "unexpected linear layer"
    check_tied(model)
    return out


def check_tied(model) -> bool:
    """Assert lm_head.weight and embed_tokens.weight share storage."""
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    assert head is emb or (head.data_ptr() == emb.data_ptr() and head.shape == emb.shape), "lm_head not tied"
    return True


def _is_norm(pname: str) -> bool:
    return pname.split(".")[-2].endswith("norm")


def count_params(model) -> dict:
    """Exact parameter counts; tied parameters counted once."""
    elig = enumerate_eligible(model)
    elig_ptrs = {e.module.weight.data_ptr() for e in elig}
    emb_ptr = model.get_input_embeddings().weight.data_ptr()
    counts = {"eligible": 0, "embedding": 0, "norms": 0, "biases": 0, "other": 0}
    names = {"norms": [], "other": []}
    for pname, p in model.named_parameters():  # deduplicates tied parameters
        if p.data_ptr() in elig_ptrs:
            counts["eligible"] += p.numel()
        elif p.data_ptr() == emb_ptr:
            counts["embedding"] += p.numel()
        elif pname.endswith(".bias"):
            counts["biases"] += p.numel()
        elif _is_norm(pname):
            counts["norms"] += p.numel()
        else:
            counts["other"] += p.numel()
            names["other"].append(pname)
    counts["total"] = sum(p.numel() for p in model.parameters())
    assert counts["total"] == sum(counts[k] for k in ("eligible", "embedding", "norms", "biases", "other"))
    counts["non_eligible"] = counts["total"] - counts["eligible"]
    counts["lm_head_tied"] = check_tied(model)
    counts["eligible_matrices"] = len(elig)
    counts["eligible_exact_zeros"] = int(sum(int((e.module.weight == 0).sum()) for e in elig))
    counts["other_names"] = names["other"]
    return counts


def nonzeros_per_matrix(model) -> list:
    """Nonzero count of every eligible matrix in enumeration order."""
    return [int(torch.count_nonzero(e.module.weight)) for e in enumerate_eligible(model)]


@torch.no_grad()
def fingerprint(t: torch.Tensor) -> tuple:
    """(float64 sum, float64 sum of squares, exact-zero count), reduced in fixed chunks."""
    flat = t.detach().reshape(-1)
    s = 0.0
    sq = 0.0
    zeros = 0
    for i in range(0, flat.numel(), _FP_CHUNK):
        c = flat[i:i + _FP_CHUNK].to(torch.float64)
        s += float(c.sum())
        sq += float((c * c).sum())
        zeros += int((c == 0).sum())
    return (s, sq, zeros)


class Pristine:
    """CPU snapshot of every parameter taken once after loading, with load-time fingerprints."""

    def __init__(self, model):
        self.names = []
        self.cpu = {}
        self.fp = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                self.names.append(name)
                self.cpu[name] = p.detach().to("cpu", copy=True)
                self.fp[name] = fingerprint(p)

    def verify(self, model) -> int:
        """Assert every parameter matches its load-time fingerprint; return tensors checked."""
        params = dict(model.named_parameters())
        assert list(params) == self.names, "parameter set changed"
        bad = [n for n in self.names if fingerprint(params[n]) != self.fp[n]]
        assert not bad, f"fingerprint mismatch: {bad[:5]}"
        return len(self.names)

    def restore(self, model) -> int:
        """Copy the snapshot back into the model and verify fingerprints."""
        params = dict(model.named_parameters())
        assert list(params) == self.names, "parameter set changed"
        with torch.no_grad():
            for n in self.names:
                params[n].copy_(self.cpu[n], non_blocking=False)
        check_tied(model)
        return self.verify(model)
