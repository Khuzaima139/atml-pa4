"""Held-out guard (protocol AM1): a committed hash list of the 120 held-out prompts, a refusal check for every item a
training or collection process loads, and a record of every held-out file the process opens.

Hash definition: sha256 of the UTF-8 prompt text exactly as released, and sha256 of its normalised form (Unicode
NFKC, lower case, runs of whitespace collapsed to one space, stripped). An item is refused when either hash is listed.
The list file is one gzip JSON object (mtime 0): version, hash definition, source files with their sha256 (a gzip
file by its decompressed content, content_sha256, since data preparation regenerates it with a new header; any other
file by its raw bytes, sha256), and per prompt its ID, domain, group and both hashes, in release order.
"""
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

from common.freeze import require_committed
from common.io import sha256_file, sha256_gz_content
from common.paths import REPO_ROOT, RESULTS_DIR

HELDOUT_ID = "t3-heldout-hash-v1"
HELDOUT_LIST = REPO_ROOT / "task3" / "heldout_hashes.json.gz"
HASH_DEFINITION = ("sha256 of the UTF-8 prompt text as released; sha256 of NFKC, lower-cased, whitespace-collapsed, "
                   "stripped text; an item is held out when either hash is listed")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def text_hashes(text: str) -> dict:
    return {"sha256_exact": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "sha256_norm": hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()}


def heldout_files() -> list:
    """Held-out data files: GSM8K test (locked file and manifest) and the short-form test file."""
    from . import data as D

    return [D.GSM8K_TEST_LOCKED, D.GSM8K_TEST_IDS, D.SHORT_TEST]


def heldout_dirs() -> list:
    """Directories holding evaluation prefixes (prefix banks)."""
    return [RESULTS_DIR / "task3" / "banks"]


def file_record(path, root) -> dict:
    """Name relative to root and hash: decompressed content for a .gz file, raw bytes otherwise."""
    name = Path(path).relative_to(root).as_posix()
    if name.endswith(".gz"):
        return {"name": name, "content_sha256": sha256_gz_content(path)}
    return {"name": name, "sha256": sha256_file(path)}


def build_heldout_list() -> dict:
    from . import data as D

    prompts, prov = D.heldout_prompts()
    return {"heldout_id": HELDOUT_ID, "hash_definition": HASH_DEFINITION,
            "files": [file_record(p, D.DATA_REPO_DIR) for p in heldout_files()],
            "data_provenance": prov, "n": len(prompts),
            "prompts": [{"prompt_id": p["prompt_id"], "domain": p["domain"], "group": p["group"],
                         **text_hashes(p["prompt"])} for p in prompts]}


def write_heldout_list(path, obj) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(path, "xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(data)
    return path


def read_heldout_list(path=HELDOUT_LIST) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


# Opens of held-out files seen by this process (an audit hook cannot be removed, so it is installed once).
_OPENS = []
_HOOK = {"installed": False, "files": set(), "names": set(), "dirs": ()}


def _audit(event, args):
    if event != "open" or not args:
        return
    target = args[0]
    if isinstance(target, int):
        return
    try:
        p = os.fspath(target)
    except TypeError:
        return
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if os.path.basename(p) not in _HOOK["names"] and not any(d in p for d in _HOOK["dirs"]):
        return
    real = os.path.realpath(p)
    if real in _HOOK["files"] or any(real.startswith(d + os.sep) for d in _HOOK["dirs"]):
        _OPENS.append(real)


def _install_hook():
    if _HOOK["installed"]:
        return
    files = [os.path.realpath(p) for p in heldout_files()]
    _HOOK["files"] = set(files)
    _HOOK["names"] = {os.path.basename(p) for p in files}
    _HOOK["dirs"] = tuple(os.path.realpath(d) for d in heldout_dirs())
    sys.addaudithook(_audit)
    _HOOK["installed"] = True


class HeldoutGuard:
    """Loads the committed hash list, refuses held-out items and records held-out file opens since creation."""

    def __init__(self, path=HELDOUT_LIST):
        rec = require_committed(path, "held-out hash list")
        obj = read_heldout_list(path)
        if obj.get("heldout_id") != HELDOUT_ID or obj.get("n") != len(obj.get("prompts", [])):
            raise SystemExit(f"held-out hash list {path}: unexpected version or count")
        self.path = Path(path)
        self.sha256 = rec["sha256"]
        self.n = obj["n"]
        self.hashes = {h for p in obj["prompts"] for h in (p["sha256_exact"], p["sha256_norm"])}
        self.items_checked = 0
        _install_hook()
        self._start = len(_OPENS)

    def check(self, items, what="item"):
        """Refuse (SystemExit) if any item's prompt text is listed; items are dicts with prompt_id and prompt."""
        bad = []
        for it in items:
            h = text_hashes(it["prompt"])
            self.items_checked += 1
            if h["sha256_exact"] in self.hashes or h["sha256_norm"] in self.hashes:
                bad.append(it["prompt_id"])
        if bad:
            raise SystemExit(f"held-out guard: refusing {len(bad)} listed {what}(s), first {bad[:5]}")

    def opens(self) -> list:
        return list(_OPENS[self._start:])

    def report(self) -> dict:
        return {"heldout_id": HELDOUT_ID, "hash_list": str(self.path), "hash_list_sha256": self.sha256,
                "n_listed_prompts": self.n, "items_checked": self.items_checked, "heldout_file_opens": self.opens(),
                "watched_files": sorted(_HOOK["files"]), "watched_dirs": list(_HOOK["dirs"])}
