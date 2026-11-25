"""Exact logical storage of eligible weights under FP16, 16-bit sparse and packed group-quantized formats.

Conventions (fixed for every Task 1 condition):
- FP16 dense: 2 bytes per weight.
- Sparse16: per matrix, one mask bit per eligible weight (byte-padded per matrix) plus 2 bytes per nonzero.
- Packed group quantization (group size g along input columns, b bits): per matrix with shape [out, in],
  groups = out * ceil(in / g); codes = out * in values packed at b bits, ceil(out*in*b / 8) bytes;
  one FP16 scale per group (2 bytes); one zero-point per group packed at b bits, ceil(groups*b / 8) bytes.
  Byte padding is per matrix; codes of a trailing partial group are not padded to g.
- Whole model: eligible bytes + 2 bytes per non-eligible parameter (embeddings, norms, biases; tied LM head
  counted once).
"""
from . import config as C

STORAGE_CONVENTION = (
    "fp16: 2 B/weight; sparse16: ceil(n/8) mask bytes + 2 B per nonzero per matrix; "
    "packed: ceil(n*b/8) code bytes + 2 B FP16 scale per group + ceil(G*b/8) zero-point bytes per matrix, "
    "group size 128 along input columns; whole model adds non-eligible parameters at FP16, tied once"
)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _n(shape) -> int:
    out, inp = shape
    return int(out) * int(inp)


def fp16_bytes(shapes) -> int:
    return sum(2 * _n(s) for s in shapes)


def sparse16_bytes(shapes, nonzeros_per_matrix) -> int:
    assert len(shapes) == len(nonzeros_per_matrix)
    total = 0
    for s, nnz in zip(shapes, nonzeros_per_matrix):
        assert 0 <= nnz <= _n(s)
        total += _ceil_div(_n(s), 8) + 2 * int(nnz)
    return total


def groups(shape, group_size: int = C.GROUP_SIZE) -> int:
    out, inp = shape
    return int(out) * _ceil_div(int(inp), group_size)


def packed_quant_matrix(shape, bits: int, group_size: int = C.GROUP_SIZE) -> dict:
    G = groups(shape, group_size)
    codes = _ceil_div(_n(shape) * bits, 8)
    scales = 2 * G
    zeros = _ceil_div(G * bits, 8)
    return {"codes": codes, "scales": scales, "zero_points": zeros, "total": codes + scales + zeros, "groups": G}


def packed_quant_bytes(shapes, bits_per_matrix, group_size: int = C.GROUP_SIZE, detail: bool = False):
    if isinstance(bits_per_matrix, int):
        bits_per_matrix = [bits_per_matrix] * len(shapes)
    assert len(shapes) == len(bits_per_matrix)
    parts = {"codes": 0, "scales": 0, "zero_points": 0, "total": 0, "groups": 0}
    for s, b in zip(shapes, bits_per_matrix):
        m = packed_quant_matrix(s, int(b), group_size)
        for k in parts:
            parts[k] += m[k]
    return parts if detail else parts["total"]


def whole_model_bytes(eligible_bytes: int, non_eligible_params: int) -> int:
    return int(eligible_bytes) + 2 * int(non_eligible_params)


def bits_per_eligible_weight(eligible_bytes: int, shapes) -> float:
    return 8.0 * eligible_bytes / sum(_n(s) for s in shapes)


def report(shapes, non_eligible_params: int, nonzeros_per_matrix=None) -> dict:
    """FP16 dense reference and, when nonzeros are given, the sparse16 representation of the same model."""
    n_elig = sum(_n(s) for s in shapes)
    fp16 = fp16_bytes(shapes)
    out = {"convention": STORAGE_CONVENTION, "eval_id": C.EVAL_ID, "eligible_weights": n_elig,
           "eligible_matrices": len(shapes), "non_eligible_params": int(non_eligible_params),
           "fp16_dense": {"eligible_bytes": fp16, "whole_model_bytes": whole_model_bytes(fp16, non_eligible_params),
                          "bits_per_eligible_weight": bits_per_eligible_weight(fp16, shapes)}}
    if nonzeros_per_matrix is not None:
        nnz = int(sum(nonzeros_per_matrix))
        sp = sparse16_bytes(shapes, nonzeros_per_matrix)
        out["sparse16"] = {"nonzeros": nnz, "nonzero_fraction": nnz / n_elig, "eligible_bytes": sp,
                           "whole_model_bytes": whole_model_bytes(sp, non_eligible_params),
                           "bits_per_eligible_weight": bits_per_eligible_weight(sp, shapes)}
    return out
