"""Write the held-out hash list used by the Task 2 Part B disjointness check.

Usage: python -m task2.make_heldout_hashes [--out task2/heldout_hashes.json.gz]

Reads the token arrays of the supplied MQAR development and test files once (after checking their sha256 against the
data repository manifest) and writes one gzip JSON object (mtime 0): the version, the hash definition, the source
file names with their sha256 and sequence counts, and the sha256 and id of every sequence in file order. Refuses an
existing output path.
"""
import argparse
import sys

from . import mqar_data as D


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(D.HELDOUT_LIST))
    args = ap.parse_args(argv)
    obj = D.build_heldout_list()
    try:
        D.write_heldout_list(args.out, obj)
    except FileExistsError:
        raise SystemExit(f"output exists: {args.out}")
    print(f"wrote {args.out}: {len(obj['sequences'])} sequences from {[f['name'] for f in obj['files']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
