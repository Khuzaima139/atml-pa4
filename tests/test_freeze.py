"""Git file records and the freeze guard."""
import subprocess

import pytest

from common.freeze import FreezeError, check_frozen, file_record, require_committed
from common.io import sha256_file


def _git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "a.json").write_text('{"x": 1}\n')
    _git(r, "add", "a.json")
    _git(r, "commit", "-q", "-m", "Add a")
    return r


def test_outside_repo(tmp_path):
    f = tmp_path / "loose.json"
    f.write_text("{}")
    rec = file_record(f)
    assert rec["exists"] and not rec["in_repo"] and not rec["tracked"]
    with pytest.raises(FreezeError, match="not inside a git work tree"):
        require_committed(f, "x")


def test_frozen_ok_and_refusals(repo):
    f = repo / "a.json"
    rec = check_frozen(f, sha256_file(f))
    assert rec["freeze_commit"] == _git(repo, "rev-parse", "HEAD") and rec["rel_path"] == "a.json"
    with pytest.raises(FreezeError, match="sha256"):
        check_frozen(f, "0" * 64)
    f.write_text('{"x": 2}\n')
    _git(repo, "add", "a.json")
    with pytest.raises(FreezeError, match="modified"):  # staged change
        check_frozen(f, sha256_file(f))
    _git(repo, "commit", "-q", "-m", "Change a")
    with pytest.raises(FreezeError, match="touched by 2 commits"):
        check_frozen(f, sha256_file(f))
    (repo / "b.json").write_text("{}")
    with pytest.raises(FreezeError, match="not tracked"):
        require_committed(repo / "b.json", "b")
    with pytest.raises(FreezeError, match="not found"):
        require_committed(repo / "c.json", "c")
