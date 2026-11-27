"""Task 2 Part B: train one MQAR model (mechanism, K, training regime, seed) under the fixed protocol and evaluate it.

Usage: python -m task2.train_mqar --mechanism {dense,window,local_dilated,linear} --K {4,8,16} --regime {near,mixed}
           --seed {6304,6305,6306} --out DIR [--cache-dir DIR] [--device cuda:0]
       python -m task2.train_mqar --mechanism local_dilated --K 16 --regime mixed --curriculum near_to_mixed
           --seed {6304,6305,6306} --out DIR [...]     (protocol AM3; run name local_dilated_curriculum_K16_s<seed>)
Test-only options (refused for any output under results/task2):
       --limit-updates N   train N updates (checkpoints and warmup scaled by N / 4500)
       --limit-eval M      evaluate the first M sequences of each development and test cell

Protocol: FP32; AdamW (peak lr 8e-4, betas 0.9/0.999, eps 1e-8, weight decay 0.01 on every parameter); batch 32;
4,500 updates; update u (1-based) uses lr(u) = peak * u / 200 for u <= 200, then peak * (1 - 0.85 (u - 200) / 4300),
so lr(200) = peak and lr(4500) = 0.15 peak; global gradient-norm clip 1.0; the seed sets initialization and selects
the training index range [(seed - 6304) * 144000, (seed - 6303) * 144000) of the (K, regime) stream.
Loss: cross-entropy over logits[..., 128:192] at the supervised query positions only, label = answer id - 128.
Accuracy: argmax over those 64 logits equals the label.

Evaluation: at updates 500, 1000, 1500, 2000, 3000 and 4500 on all 9 development cells (accuracy and loss per cell
and per distance bin). The test file is opened only after update 4500; all 9 test cells are evaluated, dense runs
also with the eval-time window64 mask.

Disjointness: the training stream is checked against the committed held-out hash list task2/heldout_hashes.json.gz
(which must be tracked and unmodified); the development and test files are not opened for it. run.json records every
opening of the development file (dev_file_opens) and of the test file (test_file_opens, empty until the final
evaluation), each with the update count after which it happened.

Curriculum (protocol AM3, --curriculum near_to_mixed; only with mechanism local_dilated, K 16 and regime mixed, the
regime of the final batches and of the learned flag): batch u (1-based) holds n_near(u) = round(32 f(u)) sequences of
the seed's near K16 stream followed by 32 - n_near(u) sequences of its mixed K16 stream; each stream is consumed in
index order from the start of the seed's index range. f(u) = 1 for u <= a, (b - u) / (b - a) for a < u < b and 0 for
u >= b, with a = 1000 and b = 2000 (with --limit-updates N: a = round(1000 N / 4500), b = round(2000 N / 4500),
recorded); round is half-to-even on the exact rational value. Both streams are loaded for the range [start, stop) of
the seed (a superset of what is consumed), validated and checked against the held-out hash list. run.json records the
mixed stream under "stream" and the near stream under "stream_near", the schedule under config.curriculum and the
consumed index ranges under curriculum_consumed; train_log rows carry curriculum_f, n_near, near_index_range and
mixed_index_range (in place of index_range); dev_curve rows carry curriculum_f, n_near and the cumulative near and
mixed counts at the checkpoint.

Outputs under --out: run.json, train_log.jsonl (one row per update), dev_curve.jsonl (one row per checkpoint, cell and
mask condition), dev_queries_final.npz and test_queries.npz (one row per query, format in task2.query_npz),
model_final.safetensors.
"""
import argparse
import sys
from fractions import Fraction
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.freeze import require_committed
from common.io import sha256_file
from common.paths import REPO_ROOT, RESULTS_DIR
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import mqar_data as D
from . import mqar_model as M
from . import query_npz as Q
from .part_a import cleanup, environment, is_cuda, under
from .rows import RowWriter

VERSION = "task2-part-b-v1"
REAL_ROOT = RESULTS_DIR / "task2"
DEFAULT_CACHE = REPO_ROOT / "cache" / "mqar"

PEAK_LR = 8e-4
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
WARMUP = 200
FINAL_FRAC = 0.15
CLIP = 1.0
CHECKPOINTS = (500, 1000, 1500, 2000, 3000, 4500)
EVAL_BATCH = 64
VALUE_LO, VALUE_HI = 128, 192
N_CLASSES = VALUE_HI - VALUE_LO
BINS = ((0, 32), (32, 64), (64, 128), (128, 256))
LEARNED_THRESHOLD = 0.5
LEARNED_MAX_DISTANCE = 64
SCORER = "64-way cross-entropy and argmax over logits 128..191 at query positions; label = answer id - 128"

CURRICULA = ("near_to_mixed",)
CURRICULUM_MECHANISM, CURRICULUM_K, CURRICULUM_REGIME = "local_dilated", 16, "mixed"
CURRICULUM_BREAKS = (1000, 2000)
CURRICULUM_RULE = ("batch u (1-based): n_near(u) = round(32 f(u)) near rows first, then 32 - n_near(u) mixed rows; "
                   "f(u) = 1 for u <= a, (b - u) / (b - a) for a < u < b, 0 for u >= b; round half to even on the "
                   "exact rational; each stream consumed in index order from the start of the seed's index range")

OUTPUT_FILES = {"run": "run.json", "train_log": "train_log.jsonl", "dev_curve": "dev_curve.jsonl",
                "dev_queries": "dev_queries_final.npz", "test_queries": "test_queries.npz",
                "checkpoint": "model_final.safetensors"}


# ---------------------------------------------------------------------------------------------------------------
# Schedule, loss, scaling

def lr_at(u: int, total: int = D.UPDATES, warmup: int = WARMUP, peak: float = PEAK_LR) -> float:
    """Learning rate of update u (1-based); lr_at(0) = 0 is the warmup start."""
    if not 0 <= u <= total:
        raise ValueError(f"update {u} outside [0, {total}]")
    if u <= warmup:
        return peak * u / warmup
    return peak * (1.0 - (1.0 - FINAL_FRAC) * (u - warmup) / (total - warmup))


def scaled_checkpoints(n_updates: int):
    if n_updates == D.UPDATES:
        return list(CHECKPOINTS)
    return sorted({max(1, round(c * n_updates / D.UPDATES)) for c in CHECKPOINTS} | {n_updates})


def scaled_warmup(n_updates: int) -> int:
    return WARMUP if n_updates == D.UPDATES else max(1, round(WARMUP * n_updates / D.UPDATES))


def run_name(mechanism: str, regime: str, K: int, seed: int, curriculum=None) -> str:
    if curriculum is not None:
        return f"{mechanism}_curriculum_K{K}_s{seed}"
    return f"{mechanism}_{regime}_K{K}_s{seed}"


def curriculum_breaks(n_updates: int = D.UPDATES):
    """(a, b) of the near fraction schedule: (1000, 2000) at 4,500 updates, scaled by n_updates / 4500 otherwise."""
    if n_updates == D.UPDATES:
        return CURRICULUM_BREAKS
    return tuple(round(Fraction(x * n_updates, D.UPDATES)) for x in CURRICULUM_BREAKS)


def curriculum_f(u: int, n_updates: int = D.UPDATES) -> Fraction:
    """Near fraction f(u) of update u (1-based), exact."""
    if not 1 <= u <= n_updates:
        raise ValueError(f"update {u} outside [1, {n_updates}]")
    a, b = curriculum_breaks(n_updates)
    if u <= a:
        return Fraction(1)
    if u >= b:
        return Fraction(0)
    return Fraction(b - u, b - a)


def n_near_at(u: int, n_updates: int = D.UPDATES, batch: int = D.BATCH) -> int:
    return round(batch * curriculum_f(u, n_updates))


def curriculum_plan(n_updates: int = D.UPDATES, batch: int = D.BATCH) -> list:
    """Per update: f, n_near and the [lo, hi) offsets into the near and mixed streams (offsets from the start of the
    seed's index range), each stream consumed contiguously in index order."""
    rows, pn, pm = [], 0, 0
    for u in range(1, n_updates + 1):
        n = n_near_at(u, n_updates, batch)
        rows.append({"update": u, "f": curriculum_f(u, n_updates), "n_near": n, "near": (pn, pn + n),
                     "mixed": (pm, pm + batch - n)})
        pn, pm = pn + n, pm + batch - n
    return rows


def curriculum_batch(near, mixed, row: dict):
    """Rows of one curriculum batch: the near slice followed by the mixed slice (numpy or torch, same type)."""
    a = near[row["near"][0]:row["near"][1]]
    b = mixed[row["mixed"][0]:row["mixed"][1]]
    if isinstance(a, np.ndarray):
        return np.concatenate([a, b])
    return torch.cat([a, b])


def query_logits(logits, qpos):
    """(B, T, V) logits and (B, K) positions -> (B, K, 64) logits of the value ids 128..191."""
    idx = qpos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return logits.gather(1, idx)[..., VALUE_LO:VALUE_HI]


def query_loss(logits, qpos, answers):
    """Mean 64-way cross-entropy over all query positions of the batch, and the accuracy; no other position counts."""
    sub = query_logits(logits, qpos)
    labels = answers - VALUE_LO
    loss = F.cross_entropy(sub.reshape(-1, N_CLASSES), labels.reshape(-1))
    acc = (sub.argmax(-1) == labels).float().mean()
    return loss, acc


def bin_name(lo, hi):
    return f"[{lo},{hi})"


# ---------------------------------------------------------------------------------------------------------------
# Evaluation

@torch.no_grad()
def eval_cell(model, cell: dict, device, condition: str) -> dict:
    """Per-query predictions of one cell under one mask condition (numpy arrays of shape (n, K))."""
    model.eval()
    preds, logit_ans, losses = [], [], []
    with model.mask_override(condition):
        for i in range(0, len(cell["ids"]), EVAL_BATCH):
            tok = torch.from_numpy(cell["tokens"][i:i + EVAL_BATCH]).to(device=device, dtype=torch.long)
            qpos = torch.from_numpy(cell["query_position"][i:i + EVAL_BATCH]).to(device)
            ans = torch.from_numpy(cell["answer_id"][i:i + EVAL_BATCH]).to(device)
            sub = query_logits(model(tok), qpos)
            labels = ans - VALUE_LO
            preds.append((sub.argmax(-1) + VALUE_LO).cpu())
            logit_ans.append(sub.gather(-1, labels.unsqueeze(-1)).squeeze(-1).cpu())
            losses.append(F.cross_entropy(sub.transpose(1, 2), labels, reduction="none").cpu())
    model.train()
    pred = torch.cat(preds).numpy()
    return {"predicted_id": pred, "correct": pred == cell["answer_id"],
            "answer_logit": torch.cat(logit_ans).numpy().astype(np.float32),
            "query_loss": torch.cat(losses).numpy().astype(np.float32)}


def summarize(cell: dict, res: dict) -> dict:
    """Accuracy and mean loss of a cell overall, per distance bin and per (distance - 1) mod 4."""
    dist, corr, loss = cell["distance"], res["correct"], res["query_loss"]

    def stats(sel):
        n = int(sel.sum())
        return {"n_queries": n, "accuracy": float(corr[sel].mean()) if n else None,
                "loss": float(loss[sel].mean()) if n else None}

    every = np.ones_like(corr, dtype=bool)
    return {**stats(every), "n_sequences": len(cell["ids"]),
            "bins": {bin_name(lo, hi): stats((dist >= lo) & (dist < hi)) for lo, hi in BINS},
            "residue_mod4": {str(r): stats((dist - 1) % 4 == r) for r in range(4)}}


def pooled_bins(cells: dict, results: dict) -> dict:
    """Every query of the given cells pooled by recorded distance into the four bins."""
    dist = np.concatenate([cells[c]["distance"].ravel() for c in results])
    corr = np.concatenate([results[c]["correct"].ravel() for c in results])
    out = {}
    for lo, hi in BINS:
        sel = (dist >= lo) & (dist < hi)
        n = int(sel.sum())
        out[bin_name(lo, hi)] = {"n_queries": n, "accuracy": float(corr[sel].mean()) if n else None}
    return out


def learned_flag(cells: dict, results: dict, K: int, regime: str) -> dict:
    """Protocol B6: final development accuracy >= 0.5 on distance < 64 queries of the run's own (K, regime) cell."""
    cell = D.cell_name(regime, K)
    sel = cells[cell]["distance"] < LEARNED_MAX_DISTANCE
    n = int(sel.sum())
    acc = float(results[cell]["correct"][sel].mean()) if n else None
    return {"cell": cell, "n_queries": n, "accuracy": acc, "threshold": LEARNED_THRESHOLD,
            "max_distance_exclusive": LEARNED_MAX_DISTANCE, "learned": bool(acc is not None and acc >= LEARNED_THRESHOLD),
            "rule": "protocol B6"}


def sync(device):
    if is_cuda(device):
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------------------------------------------
# Entry point

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mechanism", required=True, choices=M.MECHANISMS)
    ap.add_argument("--K", required=True, type=int, choices=D.KS)
    ap.add_argument("--regime", required=True, choices=D.TRAIN_REGIMES)
    ap.add_argument("--seed", required=True, type=int, choices=D.SEEDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--curriculum", default=None, choices=CURRICULA)
    ap.add_argument("--limit-updates", type=int, default=None, help="tests only")
    ap.add_argument("--limit-eval", type=int, default=None, help="tests only")
    return ap.parse_args(argv)


def check_args(args) -> dict:
    limits = {"updates": args.limit_updates, "eval": args.limit_eval}
    limited = any(v is not None for v in limits.values())
    if limited and under(args.out, REAL_ROOT):
        raise SystemExit(f"refusing --limit-* with an output path under {REAL_ROOT}: {args.out}")
    if args.curriculum is not None and (args.mechanism, args.K, args.regime) != \
            (CURRICULUM_MECHANISM, CURRICULUM_K, CURRICULUM_REGIME):
        raise SystemExit(f"--curriculum {args.curriculum} needs --mechanism {CURRICULUM_MECHANISM} --K {CURRICULUM_K} "
                         f"--regime {CURRICULUM_REGIME}; got {args.mechanism}, K {args.K}, {args.regime}")
    if Path(args.out).exists():
        raise SystemExit(f"output directory exists: {args.out}")
    n_updates = D.UPDATES if args.limit_updates is None else args.limit_updates
    if not 1 <= n_updates <= D.UPDATES:
        raise SystemExit("--limit-updates must be in [1, 4500]")
    if args.limit_eval is not None and args.limit_eval < 1:
        raise SystemExit("--limit-eval must be positive")
    return {"limits": limits, "limited": limited, "n_updates": n_updates}


def main(argv=None):
    args = parse_args(argv)
    chk = check_args(args)
    out = Path(args.out)
    device = torch.device(args.device)
    if is_cuda(device):
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_updates = chk["n_updates"]
    warmup = scaled_warmup(n_updates)
    checkpoints = scaled_checkpoints(n_updates)
    start, stop = D.seed_index_range(args.seed, n_updates)

    seed_everything(args.seed)
    model = M.build_model(args.mechanism, args.seed)
    n_params = M.count_params(model)
    if n_params != M.DENSE_PARAMS:
        raise RuntimeError(f"parameter count {n_params} != {M.DENSE_PARAMS}")
    conditions = list(M.MASK_CONDITIONS) if args.mechanism == "dense" else ["native"]

    config = {
        "version": VERSION, "scorer": SCORER, "mechanism": args.mechanism, "K": args.K, "regime": args.regime,
        "seed": args.seed, "device": str(device), "dtype": "float32", "cache_dir": str(Path(args.cache_dir)),
        "limits": chk["limits"], "limited": chk["limited"], "output_files": OUTPUT_FILES,
        "updates": n_updates, "batch": D.BATCH, "train_index_range": [start, stop],
        "optimizer": {"name": "AdamW", "peak_lr": PEAK_LR, "betas": list(BETAS), "eps": ADAM_EPS,
                      "weight_decay": WEIGHT_DECAY, "decay_applies_to": "every parameter"},
        "schedule": {"warmup_updates": warmup, "final_fraction": FINAL_FRAC,
                     "rule": "update u (1-based) uses peak*u/warmup for u <= warmup, then linear decay to "
                             "final_fraction*peak at the last update"},
        "grad_clip_global_norm": CLIP, "checkpoints": checkpoints, "eval_batch": EVAL_BATCH,
        "distance_bins": [list(b) for b in BINS], "mask_conditions": conditions,
        "model": {"vocab": M.VOCAB, "d_model": M.D_MODEL, "heads": M.N_HEADS, "head_dim": M.HEAD_DIM,
                  "d_ff": M.D_FF, "layers": M.N_LAYERS, "norm": "pre-norm LayerNorm with weight and bias",
                  "positional": "fixed sinusoidal, added", "head": "untied linear, no bias",
                  "projection_biases": True, "init": "PyTorch module defaults, seeded construction on CPU",
                  "parameters": n_params, "materialization_train": M.materialization(args.mechanism, D.BATCH, D.T)},
        "theoretical_costs": M.theoretical_costs(args.mechanism, D.T),
        "theoretical_costs_all": M.all_theoretical_costs(D.T),
        "learned_rule": "protocol B6: final development accuracy >= 0.5 on distance < 64 queries of the run's own "
                        "(K, regime) cell",
        "query_file_format": Q.codes(),
        "heldout_list": str(D.HELDOUT_LIST),
    }
    if args.curriculum is not None:
        a, b = curriculum_breaks(n_updates)
        config["run_name"] = run_name(args.mechanism, args.regime, args.K, args.seed, args.curriculum)
        config["curriculum"] = {
            "name": args.curriculum, "protocol": "AM3", "rule": CURRICULUM_RULE, "breaks": [a, b],
            "breaks_full": list(CURRICULUM_BREAKS), "scaled": n_updates != D.UPDATES,
            "scaling": "a = round(1000 N / 4500), b = round(2000 N / 4500) with N updates (half to even)",
            "streams": {"near": D.cell_name("near", args.K), "mixed": D.cell_name("mixed", args.K)},
            "batch_order": "near rows first, then mixed rows",
            "learned_cell": D.cell_name(args.regime, args.K)}
    out.mkdir(parents=True)
    rec = RunRecorder(out / OUTPUT_FILES["run"], config)
    rec.update(environment=environment(device), step="started")
    try:
        result = _run(args, chk, out, rec, model, device, warmup, checkpoints, start, stop, conditions)
        rec.finish("ok", step="done", **result)
    except BaseException as exc:
        rec.finish("failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    print(f"train_mqar done: {out}")
    return 0


def _run(args, chk, out, rec, model, device, warmup, checkpoints, start, stop, conditions):
    n_updates = chk["n_updates"]
    gen, gen_info = D.load_generator()
    dev_opens, test_opens = [], []
    list_git = require_committed(D.HELDOUT_LIST, "held-out hash list")
    ref_hashes, list_info = D.load_heldout_list(D.HELDOUT_LIST)
    list_info.update(tracked=list_git["tracked"], clean=list_git["clean"], commits=list_git["commits"])
    def ready_stream(regime):
        a, st = D.load_or_build_stream(args.cache_dir, gen, gen_info, args.K, regime, start, stop)
        t0 = time.perf_counter()
        st["validation"] = D.validate_stream(gen, a, args.K, regime, start)
        st["validation"]["seconds"] = time.perf_counter() - t0
        st["disjointness"] = {**D.check_disjoint(a, ref_hashes, start, args.K, regime), "heldout_list": list_info,
                              "note": "checked against the committed list only; no development or test file opened"}
        return a, st

    curriculum = args.curriculum is not None
    extra = {}
    if curriculum:
        arr_near, stream_near = ready_stream("near")
        extra["stream_near"] = stream_near
    arr, stream = ready_stream(args.regime)
    del ref_hashes
    rec.update(generator=gen_info, stream=stream, **extra, dev_file_opens=dev_opens, test_file_opens=test_opens,
               step="stream_ready")

    dev = D.read_eval_file(D.DEV_FILE, "development", gen=gen, limit_per_cell=args.limit_eval, audit=dev_opens,
                           after_update=0, manifest_rel=D.DEV_REL)
    rec.update(dev_file_opens=dev_opens, test_file_opens=test_opens, step="dev_loaded")
    data = torch.from_numpy(arr).to(device)
    plan = None
    if curriculum:
        data_near = torch.from_numpy(arr_near).to(device)
        plan = curriculum_plan(n_updates)
        if plan[-1]["near"][1] > len(arr_near) or plan[-1]["mixed"][1] > len(arr):
            raise RuntimeError("curriculum consumes more sequences than the loaded streams hold")
    K, Tn = args.K, D.T
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

    train_s, peak_train, peak_eval, eval_s = 0.0, 0, 0, 0.0
    curve = []
    eval_timings = {"dev_checkpoints": {}, "test": {}}
    if is_cuda(device):
        cleanup(device)
        torch.cuda.reset_peak_memory_stats(device)
    with RowWriter(out / OUTPUT_FILES["train_log"]) as log, RowWriter(out / OUTPUT_FILES["dev_curve"]) as dcw:
        for u in range(1, n_updates + 1):
            lr = lr_at(u, n_updates, warmup)
            sync(device)
            t_a = time.perf_counter()
            if curriculum:
                row = plan[u - 1]
                batch = curriculum_batch(data_near, data, row)
                assert batch.shape[0] == D.BATCH
            else:
                batch = data[(u - 1) * D.BATCH:u * D.BATCH]
            tok = batch[:, :Tn].long()
            qpos = batch[:, Tn:Tn + K].long()
            ans = batch[:, Tn + 2 * K:Tn + 3 * K].long()
            for g in opt.param_groups:
                g["lr"] = lr
            loss, acc = query_loss(model(tok), qpos, ans)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sync(device)
            dt = time.perf_counter() - t_a
            train_s += dt
            entry = {"version": VERSION, "update": u, "loss": float(loss.detach()), "accuracy": float(acc), "lr": lr,
                     "grad_norm": float(gnorm), "clipped": bool(float(gnorm) > CLIP), "step_ms": dt * 1000.0}
            if curriculum:
                entry.update(curriculum_f=float(row["f"]), n_near=row["n_near"],
                             near_index_range=[start + row["near"][0], start + row["near"][1]],
                             mixed_index_range=[start + row["mixed"][0], start + row["mixed"][1]])
            else:
                entry["index_range"] = [start + (u - 1) * D.BATCH, start + u * D.BATCH]
            log.write(entry)
            if u in checkpoints:
                if is_cuda(device):
                    peak_train = max(peak_train, torch.cuda.max_memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                t_e = time.perf_counter()
                results = {c: eval_cell(model, dev[c], device, "native") for c in D.cell_order()}
                point = {"update": u, "cells": {}}
                cur = {}
                if curriculum:
                    cur = {"curriculum_f": float(row["f"]), "n_near": row["n_near"],
                           "near_consumed": row["near"][1], "mixed_consumed": row["mixed"][1]}
                    point.update(cur)
                for c in D.cell_order():
                    s = summarize(dev[c], results[c])
                    dcw.write({"version": VERSION, "update": u, "split": "development", "cell": c,
                               "K": dev[c]["K"], "regime": dev[c]["regime"], "mask_condition": "native", **cur, **s})
                    point["cells"][c] = {"accuracy": s["accuracy"], "loss": s["loss"]}
                curve.append(point)
                dt_e = time.perf_counter() - t_e
                eval_timings["dev_checkpoints"][str(u)] = dt_e
                eval_s += dt_e
                if is_cuda(device):
                    peak_eval = max(peak_eval, torch.cuda.max_memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                rec.update(dev_curve=curve, updates_done=u, train_wall_s=train_s, dev_file_opens=dev_opens,
                           test_file_opens=test_opens, step=f"checkpoint_{u}")
        n_log, n_curve = log.n_rows, dcw.n_rows

    final_dev = results
    learned = learned_flag(dev, final_dev, args.K, args.regime)
    dev_cols = Q.concat([Q.cell_columns(dev[c], final_dev[c], "native") for c in D.cell_order()])
    Q.write_query_npz(out / OUTPUT_FILES["dev_queries"], dev_cols, VERSION)
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(out / OUTPUT_FILES["checkpoint"]), metadata={"version": VERSION, "mechanism": args.mechanism})
    rec.update(learned=learned, train_wall_s=train_s, test_file_opens=test_opens, step="training_done")

    test_opened_after = n_updates
    test = D.read_eval_file(D.TEST_FILE, "test", gen=gen, limit_per_cell=args.limit_eval, audit=test_opens,
                            after_update=n_updates, manifest_rel=D.TEST_REL)
    t_e = time.perf_counter()
    test_parts, test_summary, test_pooled = [], {}, {}
    for cond in conditions:
        t_c = time.perf_counter()
        res = {c: eval_cell(model, test[c], device, cond) for c in D.cell_order()}
        eval_timings["test"][cond] = time.perf_counter() - t_c
        test_summary[cond] = {c: summarize(test[c], res[c]) for c in D.cell_order()}
        test_pooled[cond] = pooled_bins(test, res)
        test_parts += [Q.cell_columns(test[c], res[c], cond) for c in D.cell_order()]
    eval_s += time.perf_counter() - t_e
    if is_cuda(device):
        peak_eval = max(peak_eval, torch.cuda.max_memory_allocated(device))
    test_cols = Q.concat(test_parts)
    Q.write_query_npz(out / OUTPUT_FILES["test_queries"], test_cols, VERSION)

    out_extra = {}
    if curriculum:
        last = plan[-1]
        out_extra["curriculum_consumed"] = {
            "near_index_range": [start, start + last["near"][1]], "mixed_index_range": [start, start + last["mixed"][1]],
            "n_near": last["near"][1], "n_mixed": last["mixed"][1],
            "n_near_by_update_sum": sum(r["n_near"] for r in plan)}
    return {
        **out_extra,
        "learned": learned, "dev_curve": curve, "test_summary": test_summary, "test_pooled_bins": test_pooled,
        "test_opened_after_update": test_opened_after, "updates_done": n_updates,
        "dev_file_opens": dev_opens, "test_file_opens": test_opens,
        "train_wall_s": train_s, "train_wall_note": "sum of CUDA-synchronized update times; evaluation, logging "
                                                    "and checkpoint writing excluded",
        "mean_ms_per_update": 1000.0 * train_s / n_updates, "eval_wall_s": eval_s,
        "eval_timings_s": eval_timings,
        "eval_timings_note": "wall seconds of each development checkpoint evaluation (all 9 cells) and of each test "
                             "mask condition (all 9 cells, summaries included)",
        "peak_allocated_bytes_train": peak_train if is_cuda(device) else None,
        "peak_allocated_bytes_eval": peak_eval if is_cuda(device) else None,
        "parameter_count": M.count_params(model),
        "n_rows": {"train_log": n_log, "dev_curve": n_curve, "dev_queries": int(dev_cols["split"].size),
                   "test_queries": int(test_cols["split"].size)},
        "checkpoint_sha256": sha256_file(out / OUTPUT_FILES["checkpoint"]),
        "dev_eval_sequences_per_cell": {c: len(dev[c]["ids"]) for c in D.cell_order()},
        "test_eval_sequences_per_cell": {c: len(test[c]["ids"]) for c in D.cell_order()},
    }


if __name__ == "__main__":
    sys.exit(main())
