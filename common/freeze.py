"""Git state of single files, and the freeze guard for settings chosen on development data.

A frozen file must sit in a git work tree, be tracked, have no staged or unstaged change, and be touched by
exactly one commit reachable from HEAD (the freeze commit). Its sha256 must equal the value recorded by the
step that consumed it.
"""
import subprocess
from pathlib import Path

from .io import sha256_file


class FreezeError(RuntimeError):
    pass


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


def file_record(path) -> dict:
    """Repository, tracking, cleanliness and the commits (HEAD history) touching one file."""
    p = Path(path).resolve()
    rec = {"path": str(path), "exists": p.is_file(), "in_repo": False, "tracked": False, "clean": False,
           "commits": [], "repo_root": None, "rel_path": None, "head": None}
    start = p.parent if p.parent.is_dir() else Path.cwd()
    top = _git(start, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return rec
    root = Path(top.stdout.strip()).resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        return rec
    rec.update(in_repo=True, repo_root=str(root), rel_path=rel)
    head = _git(root, "rev-parse", "HEAD")
    rec["head"] = head.stdout.strip() if head.returncode == 0 else None
    rec["tracked"] = _git(root, "ls-files", "--error-unmatch", "--", rel).returncode == 0
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", rel)
    rec["status_porcelain"] = status.stdout.rstrip("\n")
    rec["clean"] = rec["tracked"] and status.returncode == 0 and not status.stdout.strip()
    if rec["head"]:
        log = _git(root, "log", "--format=%H%x09%cI%x09%aI", "--", rel)
        for line in log.stdout.splitlines():
            h, cdate, adate = line.split("\t")
            rec["commits"].append({"commit": h, "committer_date": cdate, "author_date": adate})
    return rec


def require_committed(path, what: str) -> dict:
    """Refuse a file that is missing, untracked or modified; returns its record with the sha256."""
    rec = file_record(path)
    if not rec["exists"]:
        raise FreezeError(f"{what}: file not found: {path}")
    if not rec["in_repo"]:
        raise FreezeError(f"{what}: not inside a git work tree: {path}")
    if not rec["tracked"]:
        raise FreezeError(f"{what}: not tracked by git: {path}")
    if not rec["clean"]:
        raise FreezeError(f"{what}: modified in the working tree or index: {path} ({rec['status_porcelain']!r})")
    rec["sha256"] = sha256_file(path)
    return rec


def check_frozen(path, expected_sha256: str, what: str = "frozen file") -> dict:
    """Freeze guard: committed, unmodified, touched by exactly one commit, and sha256 equal to the recorded value."""
    rec = require_committed(path, what)
    n = len(rec["commits"])
    if n != 1:
        raise FreezeError(f"{what}: touched by {n} commits in git log (exactly one required): {path}")
    if rec["sha256"] != expected_sha256:
        raise FreezeError(f"{what}: sha256 {rec['sha256']} != recorded {expected_sha256}: {path}")
    rec["freeze_commit"] = rec["commits"][0]["commit"]
    rec["freeze_commit_date"] = rec["commits"][0]["committer_date"]
    rec["freeze_author_date"] = rec["commits"][0]["author_date"]
    return rec
