"""Hashing and deterministic JSON / JSONL(.gz) helpers."""
import gzip
import hashlib
import io as _io
import json
import os
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path) -> str:
    """SHA-256 of the raw bytes of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_gz_content(path) -> str:
    """SHA-256 of the decompressed bytes of a gzip file."""
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _jsonl_bytes(records) -> bytes:
    buf = _io.StringIO()
    for rec in records:
        buf.write(json.dumps(rec, sort_keys=True, ensure_ascii=False))
        buf.write("\n")
    return buf.getvalue().encode("utf-8")


def write_jsonl_gz(path, records, overwrite: bool = False) -> Path:
    """Write records as gzip JSONL with sorted keys, mtime 0 and no embedded filename."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    data = _jsonl_bytes(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(data)
    os.replace(tmp, path)
    return path


def read_jsonl(path) -> list:
    """Read a JSONL file, plain or gzip-compressed (detected by magic bytes)."""
    with open(path, "rb") as fh:
        is_gz = fh.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gz else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json_new(path, obj, indent: int = 2) -> Path:
    """Write JSON to a path that must not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return path
