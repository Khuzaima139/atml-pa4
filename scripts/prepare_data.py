"""Prepare and verify the fixed public dataset selections.

Clones the course data repository at a pinned commit, verifies its checksum
manifest, runs its public preparation with the locked source revisions, and
compares content hashes of the generated files against data_lock/.

Usage (from the repository root):
    python scripts/prepare_data.py                # prepare, then verify against data_lock/
    python scripts/prepare_data.py --skip-setup   # verify existing files only
    python scripts/prepare_data.py --write-lock   # first run only: create data_lock/public_assets.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io import read_jsonl, sha256_file, sha256_gz_content, write_json_new  # noqa: E402
from common.paths import (  # noqa: E402
    DATA_LOCK_DIR, DATA_REPO_DIR, PUBLIC_ASSETS_LOCK, SOURCE_REVISIONS_LOCK,
)

DATA_REPO_URL = "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"
DATA_REPO_COMMIT = "6a0a9ec9da7ef3d313ee3036cfec493886723644"
EXPECTED_MISSING = {"scripts/check_release_STAFF_LEGACY.py"}
PUBLIC_LOCKED_REL = Path("data") / "public_locked"
REVISIONS_REL = Path("data") / "local_source_revisions.json"
PUBLIC_FILES = [
    "c4_calibration_locked.jsonl.gz",
    "c4_development_locked.jsonl.gz",
    "c4_validation_windows_locked.jsonl.gz",
    "gsm8k_test_locked.jsonl.gz",
    "gsm8k_train_locked.jsonl.gz",
    "hellaswag_val_locked.jsonl.gz",
    "wikitext2_test_windows_locked.jsonl.gz",
]
LOCK_FORMAT = "public_assets_v1"


class CheckError(RuntimeError):
    pass


def run(cmd, cwd=None, env=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def ensure_clone(repo: Path):
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", DATA_REPO_URL, repo])
    if git_head(repo) != DATA_REPO_COMMIT:
        run(["git", "-C", repo, "-c", "advice.detachedHead=false", "checkout", "--quiet", DATA_REPO_COMMIT])
    head = git_head(repo)
    if head != DATA_REPO_COMMIT:
        raise CheckError(f"data repository HEAD {head} != pinned {DATA_REPO_COMMIT}")


def verify_repo_checksums(repo: Path):
    manifest = json.loads((repo / "SHA256SUMS.json").read_text(encoding="utf-8"))
    problems, missing_ok = [], []
    for rel, expected in sorted(manifest.items()):
        path = repo / rel
        if not path.is_file():
            if rel in EXPECTED_MISSING:
                missing_ok.append(rel)
            else:
                problems.append(f"missing: {rel}")
        elif sha256_file(path) != expected:
            problems.append(f"sha256 mismatch: {rel}")
    if problems:
        raise CheckError("data repository checksum failures:\n  " + "\n  ".join(problems))
    print(f"SHA256SUMS.json: {len(manifest) - len(missing_ok)} files matched; "
          f"expected missing: {missing_ok or 'none'}")


def summarize_locked_dir(locked_dir: Path, expected_files=PUBLIC_FILES) -> dict:
    """Content hash, record count and scored-token total for each locked file."""
    locked_dir = Path(locked_dir)
    present = sorted(p.name for p in locked_dir.glob("*.jsonl.gz")) if locked_dir.is_dir() else []
    if sorted(expected_files) != present:
        raise CheckError(f"{locked_dir}: expected files {sorted(expected_files)}, found {present}")
    files = {}
    for name in sorted(expected_files):
        path = locked_dir / name
        records = read_jsonl(path)
        has_scored = any("scored_tokens" in r for r in records)
        files[name] = {
            "content_sha256": sha256_gz_content(path),
            "records": len(records),
            "scored_tokens": (sum(int(r.get("scored_tokens", 0)) for r in records) if has_scored else None),
        }
    return files


def compare_summaries(expected: dict, actual: dict) -> list:
    """Return a list of human-readable differences (empty when identical)."""
    diffs = []
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            diffs.append(f"{name}: missing")
        elif name not in expected:
            diffs.append(f"{name}: not in lock")
        else:
            for key in sorted(set(expected[name]) | set(actual[name])):
                if expected[name].get(key) != actual[name].get(key):
                    diffs.append(f"{name}.{key}: lock={expected[name].get(key)!r} actual={actual[name].get(key)!r}")
    return diffs


def build_lock(files: dict) -> dict:
    return {
        "format": LOCK_FORMAT,
        "data_repo_url": DATA_REPO_URL,
        "data_repo_commit": DATA_REPO_COMMIT,
        "hash": "sha256 of decompressed content",
        "files": files,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-lock", action="store_true", help="create data_lock/public_assets.json (first run only)")
    ap.add_argument("--skip-setup", action="store_true", help="only verify existing files")
    args = ap.parse_args(argv)

    if args.write_lock and PUBLIC_ASSETS_LOCK.exists():
        raise CheckError(f"{PUBLIC_ASSETS_LOCK} already exists; refusing --write-lock")
    if not args.write_lock and not PUBLIC_ASSETS_LOCK.exists():
        raise CheckError(f"{PUBLIC_ASSETS_LOCK} not found; run with --write-lock on the first preparation")

    repo = DATA_REPO_DIR
    if args.skip_setup:
        if not repo.exists():
            raise CheckError(f"{repo} not found; cannot use --skip-setup")
        head = git_head(repo)
        if head != DATA_REPO_COMMIT:
            raise CheckError(f"data repository HEAD {head} != pinned {DATA_REPO_COMMIT}")
    else:
        ensure_clone(repo)
    verify_repo_checksums(repo)

    clone_revisions = repo / REVISIONS_REL
    if not args.skip_setup:
        if SOURCE_REVISIONS_LOCK.exists():
            shutil.copyfile(SOURCE_REVISIONS_LOCK, clone_revisions)
            print(f"copied {SOURCE_REVISIONS_LOCK} -> {clone_revisions}")
        env = dict(os.environ, HF_HUB_DISABLE_XET="1")
        run([sys.executable, "scripts/setup_data.py", "--public"], cwd=repo, env=env)

    if not clone_revisions.is_file():
        raise CheckError(f"{clone_revisions} not found after preparation")
    files = summarize_locked_dir(repo / PUBLIC_LOCKED_REL)
    for name, info in files.items():
        print(f"  {name}: records={info['records']} scored_tokens={info['scored_tokens']} "
              f"content_sha256={info['content_sha256']}")

    if args.write_lock:
        DATA_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        if SOURCE_REVISIONS_LOCK.exists():
            if SOURCE_REVISIONS_LOCK.read_bytes() != clone_revisions.read_bytes():
                raise CheckError(f"{SOURCE_REVISIONS_LOCK} differs from {clone_revisions}")
        else:
            shutil.copyfile(clone_revisions, SOURCE_REVISIONS_LOCK)
            print(f"wrote {SOURCE_REVISIONS_LOCK}")
        write_json_new(PUBLIC_ASSETS_LOCK, build_lock(files))
        print(f"wrote {PUBLIC_ASSETS_LOCK}")
        return 0

    problems = []
    lock = json.loads(PUBLIC_ASSETS_LOCK.read_text(encoding="utf-8"))
    for key in ("format", "data_repo_url", "data_repo_commit"):
        if lock.get(key) != build_lock({})[key]:
            problems.append(f"lock {key}: {lock.get(key)!r} != {build_lock({})[key]!r}")
    problems += compare_summaries(lock.get("files", {}), files)
    if not SOURCE_REVISIONS_LOCK.exists():
        problems.append(f"{SOURCE_REVISIONS_LOCK} missing")
    elif json.loads(SOURCE_REVISIONS_LOCK.read_text(encoding="utf-8")) != \
            json.loads(clone_revisions.read_text(encoding="utf-8")):
        problems.append(f"source revisions differ: {SOURCE_REVISIONS_LOCK} vs {clone_revisions}")
    if problems:
        raise CheckError("public data does not match data_lock/:\n  " + "\n  ".join(problems))
    print(f"OK: {len(files)} public files match {PUBLIC_ASSETS_LOCK.name}; source revisions match")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
