import gzip
import hashlib

import pytest

from common.io import read_jsonl, sha256_file, sha256_gz_content, write_json_new, write_jsonl_gz

RECORDS = [{"id": "b", "x": 2, "a": [1, 2]}, {"z": None, "id": "a"}]


def test_write_json_new_refuses_overwrite(tmp_path):
    p = tmp_path / "r.json"
    write_json_new(p, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json_new(p, {"a": 2})


def test_jsonl_gz_is_byte_identical_across_writes(tmp_path):
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_jsonl_gz(a, RECORDS)
    write_jsonl_gz(b, RECORDS)
    assert a.read_bytes() == b.read_bytes()
    assert read_jsonl(a) == RECORDS
    with pytest.raises(FileExistsError):
        write_jsonl_gz(a, RECORDS)


def test_gz_content_hash_matches_plain_bytes(tmp_path):
    gz = tmp_path / "a.jsonl.gz"
    write_jsonl_gz(gz, RECORDS)
    plain = tmp_path / "a.jsonl"
    plain.write_bytes(gzip.decompress(gz.read_bytes()))
    assert sha256_gz_content(gz) == sha256_file(plain) == hashlib.sha256(plain.read_bytes()).hexdigest()
    assert read_jsonl(plain) == RECORDS
