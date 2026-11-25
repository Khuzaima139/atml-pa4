import gzip
import json

import pytest

import prepare_data as pd

NAMES = ["a_locked.jsonl.gz", "b_locked.jsonl.gz"]


def _make_dir(root, changed=False):
    root.mkdir()
    rows_a = [{"id": i, "scored_tokens": 10 + i} for i in range(3)]
    rows_b = [{"id": "x", "text": "first"}, {"id": "y", "text": "second"}]
    if changed:
        rows_b[1]["text"] = "changed"
    for name, rows in zip(NAMES, (rows_a, rows_b)):
        with gzip.open(root / name, "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    return root


def test_identical_dirs_pass(tmp_path):
    a = pd.summarize_locked_dir(_make_dir(tmp_path / "a"), NAMES)
    b = pd.summarize_locked_dir(_make_dir(tmp_path / "b"), NAMES)
    assert pd.compare_summaries(a, b) == []
    assert a["a_locked.jsonl.gz"]["records"] == 3
    assert a["a_locked.jsonl.gz"]["scored_tokens"] == 33
    assert a["b_locked.jsonl.gz"]["scored_tokens"] is None


def test_changed_record_fails(tmp_path):
    a = pd.summarize_locked_dir(_make_dir(tmp_path / "a"), NAMES)
    b = pd.summarize_locked_dir(_make_dir(tmp_path / "b", changed=True), NAMES)
    diffs = pd.compare_summaries(a, b)
    assert len(diffs) == 1 and diffs[0].startswith("b_locked.jsonl.gz.content_sha256")


def test_unexpected_file_set_fails(tmp_path):
    d = _make_dir(tmp_path / "a")
    with pytest.raises(pd.CheckError):
        pd.summarize_locked_dir(d, NAMES + ["c_locked.jsonl.gz"])
