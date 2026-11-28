"""Write the held-out hash list used by the Task 3 held-out guard (protocol AM1).

Usage: python -m task3.make_heldout_hashes [--out task3/heldout_hashes.json.gz]

Reads the 120 held-out prompts once (60 GSM8K test questions, 60 short-form test inputs, after checking their sha256
against data_lock and the release SHA256SUMS.json) and writes one gzip JSON object (mtime 0) with both hashes of every
prompt in release order. Refuses an existing output path.
"""
import argparse
import sys

from . import heldout as H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(H.HELDOUT_LIST))
    args = ap.parse_args(argv)
    obj = H.build_heldout_list()
    try:
        H.write_heldout_list(args.out, obj)
    except FileExistsError:
        raise SystemExit(f"output exists: {args.out}")
    print(f"wrote {args.out}: {obj['n']} prompts from {[f['name'] for f in obj['files']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
