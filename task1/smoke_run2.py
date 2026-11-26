"""Worst-case batches of the run-2 evaluation on the unchanged FP16 model (memory and time only).

Usage: python -m task1.smoke_run2 --out DIR [--device cuda:0] [--model-path PATH --limit-hellaswag N]
The output directory must lie outside results/task1. Units, each with wall time, peak allocated/reserved GPU memory
and host peak RSS:
  load                   model and tokenizer load
  worst_batch_hellaswag  task1.evaluate.score_spans on the HELLASWAG_BATCH_SIZE longest (context + continuation)
                         sequences of the full HellaSwag ID list (the first batch of the full run, which sorts by
                         length, longest first)
  worst_batch_lmeval     lm_eval HFLM.loglikelihood (the acceptance-run softmax dtype) on the requests of the
                         LMEVAL_BATCH_SIZE / 4 examples of the first HELLASWAG_SUBSET_N IDs with the longest request
No score, log likelihood or accuracy is written: only lengths, the selected example orders, times and memory.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs
from .smoke import host_maxrss_bytes

OUTPUT_FILES = {"run": "run.json"}
UNITS = ("load", "worst_batch_hellaswag", "worst_batch_lmeval")
LMEVAL_SOFTMAX_DTYPE = "float32"  # the acceptance run (softmax_dtype=float32)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", default=None, help="local model directory (tests only)")
    ap.add_argument("--limit-hellaswag", type=int, default=None, help="first N HellaSwag IDs (tests only)")
    return ap.parse_args(argv)


def check_args(args):
    if runargs.under(args.out, runargs.REAL_ROOT):
        raise SystemExit(f"refusing a smoke output path under {runargs.REAL_ROOT}: {args.out}")
    if args.limit_hellaswag is not None and args.limit_hellaswag < 1:
        raise SystemExit("limits must be positive")
    if Path(args.out).exists():
        raise SystemExit(f"output directory exists: {args.out}")


def longest_sequences(reqs, k: int) -> list:
    """(example index, choice index, length) of the k longest context + continuation sequences, longest first."""
    items = [(len(ctx) + len(cont), i, j) for i, ex in enumerate(reqs)
             for j, (ctx, cont) in enumerate(zip(ex["context_tokens"], ex["continuation_tokens"]))]
    items.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(i, j, n) for n, i, j in items[:k]]


def longest_examples(reqs, k: int) -> list:
    """Indices of the k examples with the longest request, longest first."""
    m = [(max(len(c) + len(t) for c, t in zip(ex["context_tokens"], ex["continuation_tokens"])), i)
         for i, ex in enumerate(reqs)]
    m.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in m[:k]]


def run(args):
    check_args(args)
    import torch

    from .data import HELLASWAG_FILE, file_info, read_hellaswag
    from .evaluate import hellaswag_requests, score_spans
    from .model import load_model, load_tokenizer
    from .verify_lmeval import lmeval_loglikelihoods

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    cfg = {"entry_point": "task1.smoke_run2", "units": list(UNITS), "settings": C.settings_dict(),
           "device": args.device, "model_path": args.model_path, "limits": {"hellaswag": args.limit_hellaswag},
           "output_files": OUTPUT_FILES, "data_files": [file_info(HELLASWAG_FILE)],
           "batch_sizes": {"hellaswag": C.HELLASWAG_BATCH_SIZE, "lmeval": C.LMEVAL_BATCH_SIZE},
           "lmeval_softmax_dtype": LMEVAL_SOFTMAX_DTYPE}
    rec = RunRecorder(out / OUTPUT_FILES["run"], cfg)
    steps = {"units": {}}
    cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    dev = torch.device(args.device)

    def unit(name, fn, **info):
        if cuda:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        t = time.perf_counter()
        res = fn()
        if cuda:
            torch.cuda.synchronize(dev)
        steps["units"][name] = {
            "wall_s": time.perf_counter() - t,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev) if cuda else None,
            "host_maxrss_bytes": host_maxrss_bytes(), **info}
        rec.update(steps=steps)
        return res

    try:
        model, info = unit("load", lambda: load_model(args.device, args.model_path))
        tok = load_tokenizer(args.model_path)
        rec.update(model=info, eval_id=C.EVAL_ID)
        records = read_hellaswag(limit=args.limit_hellaswag)
        if args.limit_hellaswag is None:
            records = records[:C.HELLASWAG_FULL_N]
            assert len(records) == C.HELLASWAG_FULL_N
        reqs = hellaswag_requests(tok, records)

        top = longest_sequences(reqs, C.HELLASWAG_BATCH_SIZE)
        seqs = [reqs[i]["context_tokens"][j] + reqs[i]["continuation_tokens"][j] for i, j, _ in top]
        spans = [(len(reqs[i]["context_tokens"][j]), n) for i, j, n in top]
        unit("worst_batch_hellaswag", lambda: score_spans(model, seqs, spans, C.HELLASWAG_BATCH_SIZE),
             batch=[len(seqs), top[0][2]], orders=[records[i]["order"] for i, _, _ in top],
             lengths=[n for _, _, n in top])

        sub = reqs[:C.HELLASWAG_SUBSET_N]
        k = max(1, C.LMEVAL_BATCH_SIZE // 4)
        idx = longest_examples(sub, k)
        sel = [records[i] for i in idx]
        unit("worst_batch_lmeval",
             lambda: lmeval_loglikelihoods(model, tok, sel, C.LMEVAL_BATCH_SIZE, softmax_dtype=LMEVAL_SOFTMAX_DTYPE),
             n_requests=4 * len(sel), orders=[r["order"] for r in sel],
             longest_request=max(len(c) + len(t) for i in idx
                                 for c, t in zip(sub[i]["context_tokens"], sub[i]["continuation_tokens"])))
        steps["counts"] = {"n_hellaswag": len(records), "n_sequences": 4 * len(records),
                           "max_sequence_length": top[0][2]}
        rec.finish("ok", steps=steps)
    except BaseException as exc:
        rec.finish("failed", steps=steps, error=repr(exc), traceback=traceback.format_exc())
        raise
    return out


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
