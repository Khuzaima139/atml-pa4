"""Task 2 Part B data: training streams from the course MQAR generator, and readers for the supplied evaluation files.

Generator. The course module external/atml-pa4-datasets/scripts/mqar_generator.py is loaded from its file after its
sha256 is checked against the data repository's SHA256SUMS.json. Its functions make_example(split, k, regime, index),
validate(example) and stable_seed(split, k, regime, index) are used unchanged.

Training stream. For (K, regime, seed) the stream is make_example("train", K, regime, i) for i in
[(seed - 6304) * 144000, (seed - 6303) * 144000), in index order, cut into 4,500 consecutive batches of 32. Every
batch is checked with the generator's validate() plus the filler check below before training starts.

Cache. Each stream is stored as one uint8 array of shape (N, T + 3K) in a .npy file, with a JSON sidecar recording the
index range, layout, generator sha256 and the file's sha256. Row layout: tokens[0:T], query positions[T:T+K], source
positions[T+K:T+2K], answer ids[T+2K:T+3K], each sorted by query position as the generator returns them (all values
are below 256). A cache file is used only if its sidecar matches the requested stream, its sha256 matches the sidecar,
and 18 spot-checked rows equal freshly generated ones; otherwise it is rebuilt. Builds are serialized by a lock file.

Disjointness. Every training sequence is hashed (sha256 of its 256 token bytes) and compared with the hashes of every
sequence in the supplied development and test files; any match raises DisjointnessError. Only the token arrays of
those files are read for this check.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from common.io import read_jsonl, sha256_file
from common.paths import DATA_REPO_DIR

GENERATOR_REL = "scripts/mqar_generator.py"
DEV_REL = "data/mqar/mqar_development.jsonl.gz"
TEST_REL = "data/mqar/mqar_test.jsonl.gz"
SHA_MANIFEST = "SHA256SUMS.json"
DEV_FILE = DATA_REPO_DIR / DEV_REL
TEST_FILE = DATA_REPO_DIR / TEST_REL

T = 256
BATCH = 32
UPDATES = 4500
SEQS_PER_SEED = BATCH * UPDATES  # 144,000
BASE_SEED = 6304
SEEDS = (6304, 6305, 6306)
KS = (4, 8, 16)
TRAIN_REGIMES = ("near", "mixed")
EVAL_REGIMES = ("near", "far", "mixed")
FILLER_RANGE = (2, 64)  # filler ids 2..63
CACHE_VERSION = "task2-mqar-stream-v1"
N_SPOT = 16  # evenly spaced spot checks, plus the first and last index
HASH_NAME = "sha256 of the 256 token values as uint8 bytes"
ID_RE = re.compile(r"^(train|development|test)-(near|far|mixed)-K(4|8|16)-(\d{5})$")


class DisjointnessError(RuntimeError):
    pass


class CacheError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# Generator

def manifest_sha(rel: str, data_root=DATA_REPO_DIR):
    path = Path(data_root) / SHA_MANIFEST
    return json.loads(path.read_text())[rel]


def load_generator(data_root=DATA_REPO_DIR):
    """The course generator module, loaded from its file after a sha256 check; returns (module, info)."""
    path = Path(data_root) / GENERATOR_REL
    actual = sha256_file(path)
    expected = manifest_sha(GENERATOR_REL, data_root)
    if actual != expected:
        raise RuntimeError(f"generator sha256 {actual} differs from {SHA_MANIFEST} value {expected}")
    spec = importlib.util.spec_from_file_location("pa4_mqar_generator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("make_example", "validate", "stable_seed"):
        if not callable(getattr(mod, name, None)):
            raise RuntimeError(f"generator lacks {name}")
    if mod.T != T:
        raise RuntimeError(f"generator T {mod.T} != {T}")
    return mod, {"path": str(path), "sha256": actual, "manifest_sha256": expected, "T": mod.T,
                 "master_seed": mod.MASTER_SEED}


# ---------------------------------------------------------------------------------------------------------------
# Index ranges and encoding

def seed_index_range(seed: int, n_updates: int = UPDATES, batch: int = BATCH):
    """[start, stop) of the training indices drawn by one seed: start = (seed - 6304) * 144000."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if not 1 <= n_updates <= UPDATES or batch != BATCH:
        raise ValueError("n_updates must be in [1, 4500] and batch 32")
    start = (seed - BASE_SEED) * SEQS_PER_SEED
    return start, start + n_updates * batch


def row_width(K: int) -> int:
    return T + 3 * K


def encode_example(ex: dict) -> np.ndarray:
    K = ex["K"]
    row = np.empty(row_width(K), dtype=np.uint8)
    row[:T] = ex["tokens"]
    row[T:T + K] = ex["query_positions"]
    row[T + K:T + 2 * K] = ex["source_positions"]
    row[T + 2 * K:] = ex["answer_ids"]
    return row


def decode_row(row: np.ndarray, K: int, regime: str, index: int, split: str = "train") -> dict:
    """The generator's record format rebuilt from one cache row (distances = query - source)."""
    q = [int(x) for x in row[T:T + K]]
    s = [int(x) for x in row[T + K:T + 2 * K]]
    return {"id": f"{split}-{regime}-K{K}-{index:05d}", "split": split, "regime": regime, "K": K,
            "tokens": [int(x) for x in row[:T]], "query_positions": q, "source_positions": s,
            "answer_ids": [int(x) for x in row[T + 2 * K:]], "distances": [a - b for a, b in zip(q, s)]}


def check_filler(ex: dict):
    """Every position other than the 2K key/value slots and the K query slots holds a filler id in 2..63."""
    K = ex["K"]
    special = set(range(2 * K)) | set(ex["query_positions"])
    lo, hi = FILLER_RANGE
    bad = [p for p, t in enumerate(ex["tokens"]) if p not in special and not lo <= t < hi]
    if bad:
        raise AssertionError(f"{ex['id']}: non-filler ids at positions {bad[:5]}")


def build_stream(gen, K: int, regime: str, start: int, stop: int) -> np.ndarray:
    """Generate indices [start, stop) in order (make_example runs validate() on each example)."""
    arr = np.empty((stop - start, row_width(K)), dtype=np.uint8)
    for n, i in enumerate(range(start, stop)):
        arr[n] = encode_example(gen.make_example("train", K, regime, i))
    return arr


def validate_stream(gen, arr: np.ndarray, K: int, regime: str, start: int, batch: int = BATCH) -> dict:
    """Run the generator's validate() and the filler check on every example of every batch."""
    if arr.shape[0] % batch:
        raise CacheError(f"stream length {arr.shape[0]} is not a multiple of {batch}")
    n_batches = arr.shape[0] // batch
    for b in range(n_batches):
        for n in range(b * batch, (b + 1) * batch):
            ex = decode_row(arr[n], K, regime, start + n)
            gen.validate(ex)
            check_filler(ex)
    return {"batches_checked": n_batches, "sequences_checked": int(arr.shape[0]),
            "checks": "generator validate() and filler ids 2..63 outside key, value and query slots"}


# ---------------------------------------------------------------------------------------------------------------
# Cache

def cache_name(K: int, regime: str, start: int, stop: int) -> str:
    return f"mqar_train_K{K}_{regime}_{start}_{stop}"


def cache_paths(cache_dir, K, regime, start, stop):
    base = Path(cache_dir) / cache_name(K, regime, start, stop)
    return base.with_suffix(".npy"), base.with_suffix(".json"), base.with_suffix(".lock")


def spot_indices(start: int, stop: int):
    n = stop - start
    return sorted({start, stop - 1} | {start + int(round(x)) for x in np.linspace(0, n - 1, N_SPOT)})


def _expected_meta(K, regime, start, stop, gen_info):
    return {"cache_version": CACHE_VERSION, "K": K, "regime": regime, "split": "train", "start": start,
            "stop": stop, "T": T, "shape": [stop - start, row_width(K)], "dtype": "uint8",
            "layout": "tokens[0:T], query_positions[T:T+K], source_positions[T+K:T+2K], answer_ids[T+2K:T+3K]",
            "generator_sha256": gen_info["sha256"]}


def _check_cache(npy, meta_path, gen, K, regime, start, stop, gen_info):
    """Return (array, sha256) of a trusted cache, or raise CacheError naming the first failed check."""
    if not npy.is_file() or not meta_path.is_file():
        raise CacheError("cache file or sidecar missing")
    try:
        meta = json.loads(meta_path.read_text())
    except ValueError as exc:
        raise CacheError(f"sidecar unreadable: {exc}")
    expected = _expected_meta(K, regime, start, stop, gen_info)
    for key, val in expected.items():
        if meta.get(key) != val:
            raise CacheError(f"sidecar {key} {meta.get(key)!r} != {val!r}")
    sha = sha256_file(npy)
    if sha != meta.get("sha256"):
        raise CacheError(f"file sha256 {sha} != sidecar {meta.get('sha256')}")
    try:
        arr = np.load(npy, allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise CacheError(f"array unreadable: {exc}")
    if arr.dtype != np.uint8 or list(arr.shape) != expected["shape"]:
        raise CacheError(f"array {arr.dtype} {arr.shape} != uint8 {expected['shape']}")
    for i in spot_indices(start, stop):
        if not np.array_equal(arr[i - start], encode_example(gen.make_example("train", K, regime, i))):
            raise CacheError(f"row of index {i} differs from the generator")
    return arr, sha


@contextmanager
def _locked(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_or_build_stream(cache_dir, gen, gen_info, K: int, regime: str, start: int, stop: int):
    """The stream array for indices [start, stop) and a record of where it came from."""
    if K not in KS or regime not in TRAIN_REGIMES:
        raise ValueError(f"K must be in {KS} and regime in {TRAIN_REGIMES}")
    npy, meta_path, lock = cache_paths(cache_dir, K, regime, start, stop)
    with _locked(lock):
        t0 = time.perf_counter()
        try:
            arr, sha = _check_cache(npy, meta_path, gen, K, regime, start, stop, gen_info)
            info = {"built": False, "rebuild_reason": None, "build_seconds": None}
        except CacheError as exc:
            reason = None if not (npy.exists() or meta_path.exists()) else str(exc)
            arr = build_stream(gen, K, regime, start, stop)
            build_s = time.perf_counter() - t0
            tmp = npy.with_name(npy.name + f".tmp{os.getpid()}.npy")
            np.save(tmp, arr, allow_pickle=False)
            sha = sha256_file(tmp)
            os.replace(tmp, npy)
            meta = {**_expected_meta(K, regime, start, stop, gen_info), "sha256": sha}
            tmp_meta = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
            tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            os.replace(tmp_meta, meta_path)
            info = {"built": True, "rebuild_reason": reason, "build_seconds": build_s}
        info.update(path=str(npy), sha256=sha, K=K, regime=regime, start=start, stop=stop,
                    n_sequences=int(arr.shape[0]), shape=list(arr.shape), cache_version=CACHE_VERSION,
                    spot_checked_indices=spot_indices(start, stop), load_seconds=time.perf_counter() - t0)
    return arr, info


# ---------------------------------------------------------------------------------------------------------------
# Hashes and disjointness

def sequence_hash(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype=np.uint8).tobytes()).hexdigest()


def file_info(path, rel=None, data_root=DATA_REPO_DIR) -> dict:
    sha = sha256_file(path)
    info = {"path": str(path), "sha256": sha}
    if rel is not None:
        expected = manifest_sha(rel, data_root)
        if sha != expected:
            raise RuntimeError(f"{path}: sha256 {sha} differs from {SHA_MANIFEST} value {expected}")
        info["manifest_sha256"] = expected
    return info


def heldout_hashes(paths) -> dict:
    """Map sequence hash -> record id for every record of the given JSONL files (token arrays only)."""
    out = {}
    for p in paths:
        for rec in read_jsonl(p):
            out[sequence_hash(rec["tokens"])] = rec["id"]
    return out


def check_disjoint(arr: np.ndarray, ref_hashes: dict, start: int, K: int, regime: str) -> dict:
    """Hash every training sequence; raise DisjointnessError if any equals a development or test sequence."""
    hashes = [hashlib.sha256(arr[n, :T].tobytes()).hexdigest() for n in range(arr.shape[0])]
    hits = [(f"train-{regime}-K{K}-{start + n:05d}", ref_hashes[h]) for n, h in enumerate(hashes) if h in ref_hashes]
    if hits:
        raise DisjointnessError(f"{len(hits)} training sequences equal held-out sequences, first {hits[:3]}")
    return {"hash": HASH_NAME, "n_train_hashed": len(hashes), "n_train_unique": len(set(hashes)),
            "n_reference_hashes": len(ref_hashes), "collisions": 0}


# ---------------------------------------------------------------------------------------------------------------
# Evaluation files

def cell_name(regime: str, K: int) -> str:
    return f"{regime}_K{K}"


def cell_order():
    return [cell_name(r, K) for K in KS for r in EVAL_REGIMES]


def read_eval_file(path, split: str, gen=None, limit_per_cell=None) -> dict:
    """Cells of a supplied development or test file: cell -> arrays of one (K, regime).

    Each cell holds ids, sequence_index (from the id), tokens (n, T) uint8, and (n, K) int arrays query_position,
    source_position, distance and answer_id, plus K and regime. Records are validated with the generator when given.
    limit_per_cell keeps the first records of each cell in file order (tests only).
    """
    groups = {}
    for rec in read_jsonl(path):
        m = ID_RE.match(rec["id"])
        if m is None or m.group(1) != split or m.group(2) != rec["regime"] or int(m.group(3)) != rec["K"]:
            raise ValueError(f"unexpected record id {rec['id']} in {path}")
        if gen is not None:
            gen.validate(rec)
        cell = cell_name(rec["regime"], rec["K"])
        g = groups.setdefault(cell, [])
        if limit_per_cell is None or len(g) < limit_per_cell:
            g.append((int(m.group(4)), rec))
    cells = {}
    for cell in cell_order():
        if cell not in groups:
            raise ValueError(f"{path}: no records for cell {cell}")
        recs = groups[cell]
        K = recs[0][1]["K"]
        cells[cell] = {
            "cell": cell, "split": split, "K": K, "regime": recs[0][1]["regime"],
            "ids": [r["id"] for _, r in recs],
            "sequence_index": np.array([i for i, _ in recs], dtype=np.int64),
            "tokens": np.array([r["tokens"] for _, r in recs], dtype=np.uint8),
            "query_position": np.array([r["query_positions"] for _, r in recs], dtype=np.int64),
            "source_position": np.array([r["source_positions"] for _, r in recs], dtype=np.int64),
            "distance": np.array([r["distances"] for _, r in recs], dtype=np.int64),
            "answer_id": np.array([r["answer_ids"] for _, r in recs], dtype=np.int64),
        }
    return cells


def query_records(cells: dict) -> list:
    """Flat per-query records of read_eval_file cells."""
    out = []
    for cell, c in cells.items():
        for n in range(len(c["ids"])):
            for j in range(c["K"]):
                out.append({"cell": cell, "id": c["ids"][n], "sequence_index": int(c["sequence_index"][n]),
                            "K": c["K"], "regime": c["regime"], "query_position": int(c["query_position"][n, j]),
                            "source_position": int(c["source_position"][n, j]),
                            "distance": int(c["distance"][n, j]), "answer_id": int(c["answer_id"][n, j])})
    return out
