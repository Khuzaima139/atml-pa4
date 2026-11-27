"""Per-query evaluation rows of Task 2 Part B as compressed npz files of numeric arrays.

One file holds one split (development or test) of one run; row r of every column describes one supervised query.
Columns and dtypes are fixed by FIELDS; names are stored as small integer codes (CODES, also written into the run's
run.json) and the result version as its ASCII bytes. The record id is not stored as text: it is
f"{split}-{regime}-K{K}-{sequence_index:05d}", exactly the supplied id, and record_ids() rebuilds it. Files are
written with np.savez_compressed to a new path and read with np.load(allow_pickle=False).
"""
import os
from pathlib import Path

import numpy as np

from . import mqar_data as D

FORMAT_VERSION = 1
SPLIT_CODES = {"development": 0, "test": 1}
CELL_CODES = {c: i for i, c in enumerate(D.cell_order())}
REGIME_CODES = {r: i for i, r in enumerate(D.EVAL_REGIMES)}
MASK_CODES = {"native": 0, "window64": 1}
FIELDS = {
    "split": np.uint8, "cell": np.uint8, "K": np.uint8, "regime": np.uint8, "sequence_index": np.int32,
    "mask_condition": np.uint8, "query_position": np.int16, "source_position": np.int16, "distance": np.int16,
    "answer_id": np.int16, "predicted_id": np.int16, "correct": np.uint8, "answer_logit": np.float32,
    "query_loss": np.float32,
}
META_FIELDS = {"format_version": np.int32, "version_ascii": np.uint8}
ID_RULE = "f'{split}-{regime}-K{K}-{sequence_index:05d}' (the supplied record id)"


def codes() -> dict:
    return {"format_version": FORMAT_VERSION, "split": SPLIT_CODES, "cell": CELL_CODES, "regime": REGIME_CODES,
            "mask_condition": MASK_CODES, "fields": {k: np.dtype(v).name for k, v in FIELDS.items()},
            "meta_fields": {k: np.dtype(v).name for k, v in META_FIELDS.items()}, "id_rule": ID_RULE}


def cell_columns(cell: dict, res: dict, condition: str) -> dict:
    """Columns of one evaluated cell (read_eval_file cell, eval results with (n, K) arrays) under one condition."""
    n, K = cell["query_position"].shape
    size = n * K

    def full(value):
        return np.full(size, value)

    return {"split": full(SPLIT_CODES[cell["split"]]), "cell": full(CELL_CODES[cell["cell"]]), "K": full(K),
            "regime": full(REGIME_CODES[cell["regime"]]),
            "sequence_index": np.repeat(cell["sequence_index"], K), "mask_condition": full(MASK_CODES[condition]),
            "query_position": cell["query_position"].ravel(), "source_position": cell["source_position"].ravel(),
            "distance": cell["distance"].ravel(), "answer_id": cell["answer_id"].ravel(),
            "predicted_id": res["predicted_id"].ravel(), "correct": res["correct"].ravel(),
            "answer_logit": res["answer_logit"].ravel(), "query_loss": res["query_loss"].ravel()}


def concat(parts: list) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def _cast(name, arr, dtype):
    arr = np.asarray(arr)
    out = arr.astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.integer) and not np.array_equal(out, arr):
        raise ValueError(f"{name}: values do not fit {np.dtype(dtype).name}")
    return out


def write_query_npz(path, columns: dict, version: str) -> Path:
    """Write the columns (every FIELDS name, equal lengths) to a new .npz path."""
    path = Path(path)
    if set(columns) != set(FIELDS):
        raise ValueError(f"columns {sorted(columns)} != {sorted(FIELDS)}")
    arrays = {k: _cast(k, columns[k], FIELDS[k]) for k in FIELDS}
    lengths = {a.shape for a in arrays.values()}
    if len(lengths) != 1 or len(next(iter(lengths))) != 1:
        raise ValueError(f"columns must be 1-d of one length, got {lengths}")
    arrays["format_version"] = np.array(FORMAT_VERSION, dtype=np.int32)
    arrays["version_ascii"] = np.frombuffer(version.encode("ascii"), dtype=np.uint8).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_query_npz(path) -> dict:
    """Load a per-query file with allow_pickle=False; checks the columns, dtypes, lengths and format version.
    Returns the FIELDS arrays plus 'format_version' (int) and 'version' (str)."""
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        if names != set(FIELDS) | set(META_FIELDS):
            raise ValueError(f"{path}: arrays {sorted(names)} differ from the format")
        out = {k: z[k] for k in FIELDS}
        fmt, ver = int(z["format_version"]), z["version_ascii"].tobytes().decode("ascii")
    for k, dtype in FIELDS.items():
        if out[k].dtype != np.dtype(dtype) or out[k].ndim != 1 or out[k].shape != out["split"].shape:
            raise ValueError(f"{path}: {k} has dtype {out[k].dtype} and shape {out[k].shape}")
    if fmt != FORMAT_VERSION:
        raise ValueError(f"{path}: format version {fmt} != {FORMAT_VERSION}")
    out["format_version"], out["version"] = fmt, ver
    return out


def _inverse(mapping):
    return {v: k for k, v in mapping.items()}


def record_ids(arrays: dict) -> list:
    split, regime = _inverse(SPLIT_CODES), _inverse(REGIME_CODES)
    return [f"{split[int(s)]}-{regime[int(r)]}-K{int(k)}-{int(i):05d}"
            for s, r, k, i in zip(arrays["split"], arrays["regime"], arrays["K"], arrays["sequence_index"])]


def to_rows(arrays: dict) -> list:
    """Decoded per-query dicts (names instead of codes, the id, correct as bool, Python numbers)."""
    inv = {"split": _inverse(SPLIT_CODES), "cell": _inverse(CELL_CODES), "regime": _inverse(REGIME_CODES),
           "mask_condition": _inverse(MASK_CODES)}
    ids = record_ids(arrays)
    rows = []
    for r in range(len(ids)):
        row = {"version": arrays["version"], "id": ids[r]}
        for k in FIELDS:
            v = arrays[k][r].item()
            row[k] = inv[k][v] if k in inv else (bool(v) if k == "correct" else v)
        rows.append(row)
    return rows
