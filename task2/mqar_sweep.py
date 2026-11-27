"""Task 2 Part B forward-only inference sweep of the four attention mechanisms on fixed random embeddings.

Usage: python -m task2.mqar_sweep --out DIR [--device cuda:0]
Test-only options (refused for any output under results/task2):
       --limit-lengths T1,T2,...   replace the sequence lengths
       --limit-reps WARMUP,TIMED   replace the 3 warmup and 10 timed repetitions

For each T in {256, 1024, 4096, 8192} one fixed FP32 input of shape (2, T, 128) is drawn from torch.randn on the CPU
with a generator seeded 6304 and fed (without token or positional embeddings) through the blocks, final norm and head
of a 2-block model of each mechanism built with seed 6304 (identical weights for every mechanism), under
torch.no_grad. Timing: 3 warmup and 10 timed repetitions, CUDA-synchronized with CUDA events (perf_counter on CPU);
peak allocated and reserved memory are reset per condition. The explicit masks are built during warmup and stay
resident, so they count in the peak. Each row records whether a dense T x T mask or a linear prefix tensor is
materialized and its bytes, and a status (ok, oom, unsupported) with error text.

Outputs under --out: run.json and sweep_rows.jsonl.
"""
import argparse
import sys
import traceback
from pathlib import Path

import torch

from common.paths import RESULTS_DIR
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import mqar_model as M
from .part_a import cleanup, environment, is_cuda, mem_peak, mem_reset, time_fn, time_stats, under
from .rows import RowWriter

VERSION = "task2-part-b-sweep-v1"
REAL_ROOT = RESULTS_DIR / "task2"
B = 2
SEED = 6304
LENGTHS = (256, 1024, 4096, 8192)
WARMUP, REPS = 3, 10
STATUSES = ("ok", "oom", "unsupported")
OUTPUT_FILES = {"run": "run.json", "rows": "sweep_rows.jsonl"}


def make_input(T: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    return torch.randn((B, T, M.D_MODEL), generator=gen, dtype=torch.float32)


def run_condition(model, x, device, warmup, reps) -> dict:
    def fn():
        with torch.no_grad():
            model.forward_embeddings(x)

    base = mem_reset(device)
    try:
        times = time_fn(fn, device, warmup, reps)
        status, error = "ok", None
    except torch.cuda.OutOfMemoryError as exc:
        times, status, error = None, "oom", str(exc)
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        times, status, error = None, ("oom" if oom else "unsupported"), str(exc)
    row = {"status": status, "error": error, "baseline_allocated_bytes": base, **mem_peak(device),
           "timer": "cuda_event" if is_cuda(device) else "perf_counter", "warmup": warmup, "repetitions": reps,
           "unit": "ms"}
    if times is not None:
        row.update(time_stats(times))
    return row


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-lengths", default=None, help="tests only")
    ap.add_argument("--limit-reps", default=None, help="WARMUP,TIMED (tests only)")
    return ap.parse_args(argv)


def check_args(args) -> dict:
    limits = {"lengths": args.limit_lengths, "reps": args.limit_reps}
    limited = any(v is not None for v in limits.values())
    if limited and under(args.out, REAL_ROOT):
        raise SystemExit(f"refusing --limit-* with an output path under {REAL_ROOT}: {args.out}")
    if Path(args.out).exists():
        raise SystemExit(f"output directory exists: {args.out}")
    lengths = LENGTHS if args.limit_lengths is None else tuple(int(x) for x in args.limit_lengths.split(","))
    if not lengths or min(lengths) < 1:
        raise SystemExit("lengths must be positive")
    warmup, reps = WARMUP, REPS
    if args.limit_reps is not None:
        warmup, reps = (int(x) for x in args.limit_reps.split(","))
        if warmup < 0 or reps < 1:
            raise SystemExit("--limit-reps needs WARMUP >= 0 and TIMED >= 1")
    return {"limits": limits, "limited": limited, "lengths": lengths, "warmup": warmup, "reps": reps}


def main(argv=None):
    args = parse_args(argv)
    chk = check_args(args)
    out = Path(args.out)
    device = torch.device(args.device)
    if is_cuda(device):
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(SEED)
    config = {"version": VERSION, "device": str(device), "dtype": "float32", "B": B, "H": M.N_HEADS,
              "head_dim": M.HEAD_DIM, "d_model": M.D_MODEL, "layers": M.N_LAYERS, "seed": SEED,
              "lengths": list(chk["lengths"]), "mechanisms": list(M.MECHANISMS), "warmup": chk["warmup"],
              "repetitions": chk["reps"], "limits": chk["limits"], "limited": chk["limited"],
              "output_files": OUTPUT_FILES,
              "input": "torch.randn (B, T, 128) FP32 on CPU from a generator seeded 6304 per T; fed to "
                       "forward_embeddings (blocks, final norm, head) without token or positional embeddings",
              "model_init": "build_model(mechanism, 6304): identical weights for every mechanism",
              "forward_only": "torch.no_grad; no autograd graph"}
    out.mkdir(parents=True)
    rec = RunRecorder(out / OUTPUT_FILES["run"], config)
    rec.update(environment=environment(device), step="started")
    counts = {}
    try:
        with RowWriter(out / OUTPUT_FILES["rows"]) as writer:
            for T in chk["lengths"]:
                x = make_input(T).to(device)
                for mech in M.MECHANISMS:
                    model = M.build_model(mech, SEED).to(device).eval()
                    row = {"version": VERSION, "mechanism": mech, "T": T, "B": B, "H": M.N_HEADS,
                           "head_dim": M.HEAD_DIM, "d_model": M.D_MODEL, "dtype": "float32",
                           "materialization": M.materialization(mech, B, T),
                           **run_condition(model, x, device, chk["warmup"], chk["reps"])}
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                    writer.write(row)
                    del model
                    cleanup(device)
                del x
                cleanup(device)
                rec.update(status_counts=counts, step=f"T_{T}_done")
            n_rows = writer.n_rows
        rec.finish("ok", status_counts=counts, n_rows=n_rows, step="done")
    except BaseException as exc:
        rec.finish("failed", status_counts=counts, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())
        raise
    print(f"mqar_sweep done: {out} status counts {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
