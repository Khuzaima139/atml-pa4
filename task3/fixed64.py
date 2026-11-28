"""Fixed-64 diagnostic prompt IDs (protocol D13).

The 60 GSM8K test IDs (GSM8K-{order:04d}) in manifest order, permuted with numpy default_rng(6304), first 24 kept.

    python -m task3.fixed64 --out task3/fixed64_ids.json
"""
import argparse
from pathlib import Path

import numpy as np

from common.io import read_jsonl, sha256_file, write_json_new

from .data import GSM8K_TEST_IDS

FIXED64_ID = "t3-fixed64-v1"
SEED = 6304
N_KEEP = 24


def fixed64_ids() -> list:
    ids = [f"GSM8K-{m['order']:04d}" for m in read_jsonl(GSM8K_TEST_IDS)]
    perm = np.random.default_rng(SEED).permutation(len(ids))
    return [ids[int(i)] for i in perm[:N_KEEP]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="task3/fixed64_ids.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    ids = fixed64_ids()
    write_json_new(out, {
        "fixed64_id": FIXED64_ID, "rule": "60 GSM8K test IDs in manifest order, permuted with "
        "numpy.random.default_rng(6304).permutation(60), first 24 kept (protocol D13)",
        "source": str(GSM8K_TEST_IDS.relative_to(GSM8K_TEST_IDS.parents[2])),
        "source_sha256": sha256_file(GSM8K_TEST_IDS), "n": len(ids), "ids": ids})
    print(f"{len(ids)} ids -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
