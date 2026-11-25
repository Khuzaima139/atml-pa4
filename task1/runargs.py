"""Argument checks shared by the Task 1 compression entry points."""
from pathlib import Path

from common.paths import RESULTS_DIR

REAL_ROOT = RESULTS_DIR / "task1"


def under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def add_test_args(ap):
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", default=None, help="local model directory (tests only)")
    ap.add_argument("--limit-calib", type=int, default=None, help="first N calibration sequences (tests only)")
    ap.add_argument("--limit-dev", type=int, default=None, help="first N development sequences (tests only)")


def check(args, out_dirs) -> dict:
    """Refuse existing outputs, and test-only options with any output under results/task1."""
    limits = {"calib": args.limit_calib, "dev": args.limit_dev}
    for v in limits.values():
        if v is not None and v < 1:
            raise SystemExit("limits must be positive")
    test_only = args.model_path is not None or any(v is not None for v in limits.values())
    for d in out_dirs:
        if d is None:
            continue
        if test_only and under(d, REAL_ROOT):
            raise SystemExit(f"refusing --model-path/--limit-* with an output path under {REAL_ROOT}: {d}")
        if Path(d).exists():
            raise SystemExit(f"output directory exists: {d}")
    return {"limits": limits, "limited": any(v is not None for v in limits.values()), "test_only": test_only}
