"""Append-only JSON-lines writer: the file is created exclusively and every row is flushed and synced on write."""
import json
import os
from pathlib import Path


class RowWriter:
    """Write one JSON object per line to a new file; a crash keeps every row written so far."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "x", encoding="utf-8")
        self.n_rows = 0

    def write(self, row: dict) -> dict:
        self._fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str))
        self._fh.write("\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.n_rows += 1
        return row

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
