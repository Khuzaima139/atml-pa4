"""Task 2 analysis: tables, recomputation checks, bootstrap, prediction outcomes and figures.

Also writes the Part C qualitative export.

Reads committed result files under --results (default results/task2) and writes a new directory --out.

    python -m task2.analysis --out results/task2/analysis
    python -m task2.analysis --figures-only --out results/task2/figures

Headline numbers are recomputed from the per-example files (query npz files, per-prompt JSONL rows and npz
arrays) and asserted equal to the values stored in each run.json. --figures-only recomputes the figure sources from
the committed results with the same assertions, checks them against the CSV files in --analysis, and writes only the
figures. Stored values are unrounded; ROUND_RULE is the single display rule.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from common import bootstrap as B
from common.io import read_jsonl, sha256_file, write_json_new
from common.paths import REPO_ROOT
from common.runmeta import RunRecorder
from task1.analysis import AnalysisError, check, committed_check, write_csv  # noqa: F401

ANALYSIS_ID = "t2-analysis-v1"
ROUND_RULE = "stored values are unrounded; displayed values use 4 significant digits (Python format '.4g')"
REL_TOL = 1e-9
SEEDS = (6304, 6305, 6306)
KS = (4, 8, 16)
REGIMES = ("near", "far", "mixed")
BINS = ((0, 32), (32, 64), (64, 128), (128, 256))
CHECKPOINTS = (500, 1000, 1500, 2000, 3000, 4500)
UPDATES = 4500
WINDOW = 64
ACCESS_LIMIT = 2 * (WINDOW - 1)
CHANCE_LABEL = 1 / 64
CONFIGS = ("dense_mixed_K4", "dense_mixed_K8", "dense_mixed_K16", "dense_near_K16",
           "window_near_K4", "window_near_K8", "window_near_K16", "window_mixed_K16",
           "local_dilated_mixed_K4", "local_dilated_mixed_K16", "local_dilated_curriculum_K16",
           "linear_mixed_K4", "linear_mixed_K8", "linear_mixed_K16")
MECHANISMS = ("dense", "window", "local_dilated", "linear")
MECH_LABEL = {"dense": "Dense", "window": "Window 64", "local_dilated": "Local + dilated", "linear": "Linear"}
REGIME_LABEL = {"near": "near", "mixed": "mixed", "curriculum": "near-to-mixed curriculum"}
METHODS = ("fp16", "sink_recent", "question_aware", "int4_token", "int4_channel_k", "hybrid")
METHOD_LABEL = {"fp16": "FP16 full cache", "sink_recent": "Sink + recent (FP16)",
                "question_aware": "Question-aware (FP16)", "int4_token": "INT4 token-wise K and V",
                "int4_channel_k": "INT4 channel-wise K, token-wise V", "hybrid": "FP16 + INT2 hybrid"}
EVICTION = ("sink_recent", "question_aware")
QUANTIZED = ("int4_token", "int4_channel_k", "hybrid")
LEVELS = (2048, 4096)
POSITIONS = ("start", "middle", "end")
DISTRACTORS = (0, 2, 8)
# AM2.7 principal differences (B - A).
C_PAIRS = (("fp16", "sink_recent"), ("fp16", "question_aware"), ("fp16", "int4_token"), ("fp16", "int4_channel_k"),
           ("fp16", "hybrid"), ("sink_recent", "question_aware"), ("int4_token", "int4_channel_k"),
           ("question_aware", "hybrid"), ("int4_channel_k", "hybrid"))
BOOT_RESAMPLES = 10_000
BOOT_SEED = 6304
DIAG_CONFIGS = ("fp16", "k_int4_token", "v_int4_token", "kv_int8_token", "int4_token", "int4_channel_k")
CODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")
A_IMPLS = ("dense", "tiled_32x32", "tiled_64x64", "tiled_128x128", "tiled_64x128", "sdpa_default", "sdpa_MATH",
           "sdpa_EFFICIENT_ATTENTION")
A_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
A_DTYPES = ("fp16", "fp32")
A_BENCH_TILE = "tiled_128x128"
QUAL_RULE = ("AM2.9, fixed before Part C results: for each context length, the lowest-ID prompt in each of the start, "
             "middle and end cells with 8 distractors (6 prompts); the files are the committed eval qualitative "
             "exports, concatenated in prompt-ID order")
B3_RULE = ("AM3.3: the intervention and its schedule were chosen from the B1/B2 development learning curves (AM3.2) "
           "and fixed in protocol amendment AM3 before any B3 run; no B3 setting is selected on any data and no B3 "
           "checkpoint is chosen (the update-4500 model is the B3 model)")


def fmt(v):
    """The single display rounding rule (ROUND_RULE)."""
    return "n/a" if v is None else f"{v:.4g}"


def close(a, b, tol=REL_TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def parse_config(c):
    mech, regime, k = c.rsplit("_", 2)
    return mech, regime, int(k[1:])


def own_cell(c):
    mech, regime, K = parse_config(c)
    return f"{'mixed' if regime == 'curriculum' else regime}_K{K}"


def config_label(c):
    mech, regime, K = parse_config(c)
    return f"{MECH_LABEL[mech]}, trained {REGIME_LABEL[regime]} K={K}"


def mmm(vals):
    v = [float(x) for x in vals]
    return {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "per_seed": v}


def overlap(a, b):
    """Protocol B8: seed min-max ranges overlap (no counted difference)."""
    return not (min(a) > max(b) or min(b) > max(a))


# ------------------------------------------------------------------------------------------------ inputs
class Inputs:
    """Reads files under the results root and records the sha256 of each file read."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.read = {}

    def _rec(self, p: Path):
        self.read[str(p)] = sha256_file(p)

    def json(self, rel):
        p = self.root / rel
        self._rec(p)
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def jsonl(self, rel):
        p = self.root / rel
        self._rec(p)
        return read_jsonl(p)

    def npz(self, rel):
        p = self.root / rel
        self._rec(p)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def text(self, rel):
        p = self.root / rel
        self._rec(p)
        return p.read_text(encoding="utf-8")


def load(inp: Inputs) -> dict:
    D = {"runs": {}}
    for c in CONFIGS:
        for s in SEEDS:
            d = f"mqar/{c}_s{s}"
            run = inp.json(f"{d}/run.json")
            check(run["status"] == "ok" and run["step"] == "done", f"status {c}_s{s}")
            check(run["updates_done"] == UPDATES, f"updates {c}_s{s}")
            D["runs"][(c, s)] = {"run": run, "t": inp.npz(f"{d}/test_queries.npz"),
                                 "devq": inp.npz(f"{d}/dev_queries_final.npz"),
                                 "dev": inp.jsonl(f"{d}/dev_curve.jsonl"), "train": inp.jsonl(f"{d}/train_log.jsonl")}
    A = "part_a"
    D["verify_run"], D["verify"] = inp.json(f"{A}/verify/run.json"), inp.jsonl(f"{A}/verify/verify_rows.jsonl")
    D["bench_run"], D["bench"] = inp.json(f"{A}/bench/run.json"), inp.jsonl(f"{A}/bench/bench_rows.jsonl")
    D["backend"] = inp.jsonl(f"{A}/bench/backend_table.jsonl")
    D["score_bytes"] = inp.json(f"{A}/bench/score_bytes.json")["rows"]
    D["sweep_run"], D["sweep"] = inp.json("mqar_sweep/run.json"), inp.jsonl("mqar_sweep/sweep_rows.jsonl")
    C = "part_c"
    D["eval_run"] = inp.json(f"{C}/eval/run.json")
    D["eval"] = inp.jsonl(f"{C}/eval/eval_rows.jsonl.gz")
    D["prompts"] = inp.jsonl(f"{C}/eval/prompt_rows.jsonl.gz")
    D["npz"] = {p["id"]: inp.npz(f"{C}/eval/{p['npz']}") for p in D["prompts"]}
    D["qual"] = {}
    for p in sorted(D["prompts"], key=lambda r: r["id"]):
        if p.get("qualitative"):
            D["qual"][p["id"]] = inp.text(f"{C}/eval/{p['qualitative']}")
    D["validate_run"] = inp.json(f"{C}/validate/run.json")
    D["validate"] = inp.jsonl(f"{C}/validate/validate_rows.jsonl.gz")
    D["cbench_run"], D["cbench"] = inp.json(f"{C}/bench/run.json"), inp.jsonl(f"{C}/bench/bench_rows.jsonl.gz")
    D["diag_run"] = inp.json(f"{C}/diag/run.json")
    D["diag"] = inp.jsonl(f"{C}/diag/control_rows.jsonl.gz")
    D["diag_prompts"] = inp.jsonl(f"{C}/diag/prompt_rows.jsonl.gz")
    D["diag_npz"] = {p["id"]: inp.npz(f"{C}/diag/{p['npz']}") for p in D["diag_prompts"]}
    for key in ("eval_run", "validate_run", "cbench_run", "diag_run", "verify_run", "bench_run", "sweep_run"):
        check(D[key]["status"] == "ok" and D[key]["step"] == "done", f"status {key}")
    return D


# ------------------------------------------------------------------------------------------------ Part B helpers
def cell_codes(run):
    return run["config"]["query_file_format"]["cell"]


def select(t, run, mask=0, K=None, regime=None, dmin=None, dmax=None, residue=None):
    m = t["mask_condition"] == mask
    codes = cell_codes(run)
    if K is not None or regime is not None:
        ok = [code for name, code in codes.items()
              if (K is None or name.endswith(f"_K{K}")) and (regime is None or name.startswith(f"{regime}_"))]
        m &= np.isin(t["cell"], ok)
    if dmin is not None:
        m &= t["distance"] >= dmin
    if dmax is not None:
        m &= t["distance"] < dmax
    if residue is not None:
        m &= ((t["distance"].astype(np.int64) - 1) % 4) == residue
    return m


def seed_acc(D, c, mask=0, **kw):
    out = []
    for s in SEEDS:
        R = D["runs"][(c, s)]
        m = select(R["t"], R["run"], mask, **kw)
        check(m.sum() > 0, f"empty selection {c} {kw}")
        out.append(float(R["t"]["correct"][m].mean()))
    return out


def seed_n(D, c, mask=0, **kw):
    R = D["runs"][(c, SEEDS[0])]
    return int(select(R["t"], R["run"], mask, **kw).sum())


def masks_of(c):
    return (0, 1) if parse_config(c)[0] == "dense" else (0,)


MASK_NAME = {0: "native", 1: "window64"}


def check_part_b(D):
    """Recompute stored summaries from the query npz files and logs; return the checks done."""
    n = {"cells": 0, "bins": 0, "residues": 0, "learned": 0, "dev_final": 0, "train_ms": 0}
    for (c, s), R in D["runs"].items():
        run, t = R["run"], R["t"]
        check(len(t["correct"]) == run["n_rows"]["test_queries"], f"test rows {c}_s{s}")
        check(len(R["train"]) == run["n_rows"]["train_log"] == UPDATES, f"train rows {c}_s{s}")
        check([x["update"] for x in R["train"]] == list(range(1, UPDATES + 1)), f"train updates {c}_s{s}")
        check(run["parameter_count"] == 462_336, f"parameters {c}_s{s}")
        for mask in masks_of(c):
            summ = run["test_summary"][MASK_NAME[mask]]
            for cell, code in cell_codes(run).items():
                m = (t["mask_condition"] == mask) & (t["cell"] == code)
                acc = float(t["correct"][m].mean())
                check(close(acc, summ[cell]["accuracy"]) and int(m.sum()) == summ[cell]["n_queries"],
                      f"test accuracy {c}_s{s} {MASK_NAME[mask]} {cell}")
                n["cells"] += 1
                for r, v in summ[cell]["residue_mod4"].items():
                    mm_ = m & (((t["distance"].astype(np.int64) - 1) % 4) == int(r))
                    if v["n_queries"]:
                        check(close(float(t["correct"][mm_].mean()), v["accuracy"]), f"residue {c}_s{s} {cell} {r}")
                    check(int(mm_.sum()) == v["n_queries"], f"residue n {c}_s{s} {cell} {r}")
                    n["residues"] += 1
            for (a, b) in BINS:
                key = f"[{a},{b})"
                m = (t["mask_condition"] == mask) & (t["distance"] >= a) & (t["distance"] < b)
                v = run["test_pooled_bins"][MASK_NAME[mask]][key]
                check(int(m.sum()) == v["n_queries"] and close(float(t["correct"][m].mean()), v["accuracy"]),
                      f"pooled bin {c}_s{s} {key}")
                n["bins"] += 1
        oc = own_cell(c)
        dq = R["devq"]
        code = cell_codes(run)[oc]
        m = (dq["cell"] == code) & (dq["mask_condition"] == 0)
        final = [x for x in R["dev"] if x["update"] == UPDATES and x["cell"] == oc
                 and x.get("mask_condition", "native") == "native"]
        check(len(final) == 1 and close(float(dq["correct"][m].mean()), final[0]["accuracy"]), f"dev final {c}_s{s}")
        n["dev_final"] += 1
        m64 = m & (dq["distance"] < 64)
        L = run["learned"]
        check(L["cell"] == oc and close(float(dq["correct"][m64].mean()), L["accuracy"])
              and int(m64.sum()) == L["n_queries"] and (L["accuracy"] >= 0.5) == L["learned"], f"learned {c}_s{s}")
        n["learned"] += 1
        ms = [x["step_ms"] for x in R["train"]]
        check(close(float(np.mean(ms)), run["mean_ms_per_update"], 1e-6), f"train ms {c}_s{s}")
        n["train_ms"] += 1
    return n


def learned_count(D, c):
    return sum(bool(D["runs"][(c, s)]["run"]["learned"]["learned"]) for s in SEEDS)


# ------------------------------------------------------------------------------------------------ Part B tables
def b_test_cells(D):
    rows = []
    for c in CONFIGS:
        for mask in masks_of(c):
            for K in KS:
                for g in REGIMES:
                    for s, a in zip(SEEDS, seed_acc(D, c, mask, K=K, regime=g)):
                        rows.append({"configuration": c, "mask": MASK_NAME[mask], "seed": s, "test_regime": g,
                                     "test_K": K, "accuracy": a, "n_queries": seed_n(D, c, mask, K=K, regime=g)})
    return rows


def b1_accuracy(D):
    rows = []
    for c in CONFIGS:
        mech, regime, Kt = parse_config(c)
        for mask in masks_of(c):
            restricted = mech == "window" or mask == 1
            for K in KS:
                for g in REGIMES:
                    v = mmm(seed_acc(D, c, mask, K=K, regime=g))
                    rows.append({"configuration": c, "label": config_label(c), "mechanism": mech,
                                 "training_regime": regime, "training_K": Kt, "mask": MASK_NAME[mask],
                                 "test_regime": g, "test_K": K, "own_cell": f"{g}_K{K}" == own_cell(c) and mask == 0,
                                 "mean": v["mean"], "min": v["min"], "max": v["max"], "per_seed": v["per_seed"],
                                 "n_queries_per_seed": seed_n(D, c, mask, K=K, regime=g),
                                 "chance_label_space": CHANCE_LABEL, "chance_1_over_K": 1 / K,
                                 "learned_seeds": learned_count(D, c),
                                 "access_note": (f"window 64: positions within 2(W-1) = {ACCESS_LIMIT} reachable; far "
                                                 "queries (distance >= 128) unreachable") if restricted else ""})
    return rows


def b_bins(D):
    rows = []
    for c in CONFIGS:
        for mask in masks_of(c):
            for K in KS:
                for a, b in BINS:
                    v = mmm(seed_acc(D, c, mask, K=K, dmin=a, dmax=b))
                    rows.append({"configuration": c, "label": config_label(c), "mask": MASK_NAME[mask],
                                 "training_K": parse_config(c)[2], "training_regime": parse_config(c)[1],
                                 "test_K": K, "bin": f"[{a},{b})", "mean": v["mean"], "min": v["min"],
                                 "max": v["max"], "per_seed": v["per_seed"],
                                 "n_queries_per_seed": seed_n(D, c, mask, K=K, dmin=a, dmax=b)})
    return rows


def _ctrl(name, ca, ma, cb, mb, subset, a, b):
    return {"control": name, "configuration_a": ca, "mask_a": MASK_NAME[ma], "configuration_b": cb,
            "mask_b": MASK_NAME[mb], "test_subset": subset, "a_mean": float(np.mean(a)), "a_min": min(a),
            "a_max": max(a), "b_mean": float(np.mean(b)), "b_min": min(b), "b_max": max(b),
            "b_minus_a": float(np.mean(b) - np.mean(a)), "seed_ranges_overlap": overlap(a, b)}


def b2_mask_controls(D):
    rows = []
    for c in ("dense_mixed_K4", "dense_mixed_K8", "dense_mixed_K16", "dense_near_K16"):
        for K in KS:
            for g in REGIMES:
                rows.append(_ctrl("dense native vs test-time window64 mask", c, 0, c, 1, f"{g}_K{K}",
                                  seed_acc(D, c, 0, K=K, regime=g), seed_acc(D, c, 1, K=K, regime=g)))
            rows.append(_ctrl("dense native vs test-time window64 mask", c, 0, c, 1, f"distance<64 at K{K}",
                              seed_acc(D, c, 0, K=K, dmax=64), seed_acc(D, c, 1, K=K, dmax=64)))
    for K in KS:
        rows.append(_ctrl("window trained near vs mixed", "window_near_K16", 0, "window_mixed_K16", 0,
                          f"distance<64 at K{K}", seed_acc(D, "window_near_K16", K=K, dmax=64),
                          seed_acc(D, "window_mixed_K16", K=K, dmax=64)))
        for g in REGIMES:
            rows.append(_ctrl("window trained near vs mixed", "window_near_K16", 0, "window_mixed_K16", 0,
                              f"{g}_K{K}", seed_acc(D, "window_near_K16", K=K, regime=g),
                              seed_acc(D, "window_mixed_K16", K=K, regime=g)))
    for K in KS:
        for g in REGIMES:
            rows.append(_ctrl("B3 curriculum vs comparator", "local_dilated_mixed_K16", 0,
                              "local_dilated_curriculum_K16", 0, f"{g}_K{K}",
                              seed_acc(D, "local_dilated_mixed_K16", K=K, regime=g),
                              seed_acc(D, "local_dilated_curriculum_K16", K=K, regime=g)))
    return rows


def b2_far_residue(D):
    rows = []
    for c in CONFIGS:
        for K in KS:
            for r in range(4):
                v = mmm(seed_acc(D, c, K=K, regime="far", residue=r))
                rows.append({"configuration": c, "label": config_label(c), "test_K": K, "residue_distance_minus_1_mod_4": r,
                             "mean": v["mean"], "min": v["min"], "max": v["max"], "per_seed": v["per_seed"],
                             "n_queries_per_seed": seed_n(D, c, K=K, regime="far", residue=r),
                             "direct_dilated_hop_reaches_value": r == 0})
    return rows


def b_dev_curves(D):
    rows = []
    for c in CONFIGS:
        oc = own_cell(c)
        for s in SEEDS:
            dev = {x["update"]: x for x in D["runs"][(c, s)]["dev"] if x["cell"] == oc
                   and x.get("mask_condition", "native") == "native"}
            check(sorted(dev) == list(CHECKPOINTS), f"dev checkpoints {c}_s{s}")
            for u in CHECKPOINTS:
                x = dev[u]
                b = x["bins"]
                nq = b["[0,32)"]["n_queries"] + b["[32,64)"]["n_queries"]
                lt = sum((b[k]["accuracy"] or 0.0) * b[k]["n_queries"] for k in ("[0,32)", "[32,64)")) / nq
                rows.append({"configuration": c, "label": config_label(c), "seed": s, "update": u, "cell": oc,
                             "accuracy": x["accuracy"], "accuracy_distance_lt_64": lt, "loss": x["loss"],
                             "n_queries": x["n_queries"], "curriculum_near_fraction": x.get("curriculum_f")})
    return rows


def b3_theoretical(D):
    tc = D["runs"][(CONFIGS[0], SEEDS[0])]["run"]["config"]["theoretical_costs_all"]
    for (c, s), R in D["runs"].items():
        check(R["run"]["config"]["theoretical_costs_all"] == tc, f"theoretical costs differ in {c}_s{s}")
        check(R["run"]["config"]["theoretical_costs"] == tc[parse_config(c)[0]], f"theoretical costs {c}_s{s}")
    layers = D["runs"][(CONFIGS[0], SEEDS[0])]["run"]["config"]["model"]["layers"]
    rows = []
    for m in MECHANISMS:
        x = tc[m]
        per = x["per_head_type"]
        state = x["persistent_state"] or x["kv_cache"]
        rows.append({"mechanism": m, "label": MECH_LABEL[m], "T": x["T"], "head_types": x["head_types"],
                     "attended_pairs_per_head_per_layer": {h: per[h]["attended_pairs_per_head_per_layer"] for h in per},
                     "attended_pairs_per_layer": x["attended_pairs_per_layer"],
                     "accessible_positions_per_query_at_last_position": {h: per[h]["candidates_at_last_position"] for h in per},
                     "accessible_positions_per_query_mean": {h: per[h]["candidates_mean"] for h in per},
                     "union_accessible_positions_at_last_position": x["union_candidates_at_last_position"],
                     "state_kind": "persistent state (S and z)" if x["persistent_state"] else "KV cache positions",
                     "state_bytes_per_layer_fp32": state["bytes_per_layer_fp32"],
                     "state_bytes_per_layer_fp16": state["bytes_per_layer_fp16"],
                     "state_bytes_all_layers_fp32": layers * state["bytes_per_layer_fp32"], "layers": layers,
                     "note": x.get("candidates_note", state["definition"])})
    return rows


def b3_measured_training(D):
    rows = []
    for c in CONFIGS:
        for s in SEEDS:
            run = D["runs"][(c, s)]["run"]
            cj = run["concurrent_jobs"]
            rows.append({"configuration": c, "seed": s, "mean_ms_per_update": run["mean_ms_per_update"],
                         "train_wall_s": run["train_wall_s"], "peak_allocated_bytes_train": run["peak_allocated_bytes_train"],
                         "gpu": cj["gpu"], "measured_under_concurrency": bool(cj["overlapping"]),
                         "overlapping_jobs": [f"{o['name']} (GPU {o['gpu']})" for o in cj["overlapping"]],
                         "same_gpu_overlaps": len(cj["overlapping_same_gpu"]),
                         "timer_note": run["train_wall_note"]})
    return rows


def b3_sweep(D):
    rows = []
    check(len(D["sweep"]) == 16 and all(x["status"] == "ok" for x in D["sweep"]), "sweep rows")
    for x in sorted(D["sweep"], key=lambda r: (MECHANISMS.index(r["mechanism"]), r["T"])):
        check(close(float(np.median(x["times_ms"])), x["median_ms"]), f"sweep median {x['mechanism']} {x['T']}")
        m = x["materialization"]
        rows.append({"mechanism": x["mechanism"], "label": MECH_LABEL[x["mechanism"]], "T": x["T"], "B": x["B"],
                     "dtype": x["dtype"], "median_ms": x["median_ms"], "min_ms": x["min_ms"], "max_ms": x["max_ms"],
                     "repetitions": x["repetitions"], "warmup": x["warmup"],
                     "peak_allocated_bytes": x["peak_allocated_bytes"],
                     "baseline_allocated_bytes": x["baseline_allocated_bytes"],
                     "mask_bytes": m["dense_mask_bytes"], "mask_shape": m["dense_mask_shape"],
                     "linear_prefix_bytes_per_layer": m["linear_prefix_bytes_per_layer"],
                     "linear_prefix_shape": m["linear_prefix_shape"], "note": m["note"]})
    return rows


def b3_items(D, theo):
    c = "local_dilated_curriculum_K16"
    t = {r["mechanism"]: r for r in theo}["local_dilated"]
    runs = [D["runs"][(c, s)]["run"] for s in SEEDS]
    return [{"item": "extra persistent state", "value": "none (AM3.4); the curriculum changes only the batch "
             "composition", "source": "protocol AM3.4"},
            {"item": "attended positions", "value": json.dumps(t["accessible_positions_per_query_at_last_position"]),
             "source": "run.json theoretical_costs (local_dilated, T 256): positions per query at the last position "
                       "by head type; two window heads and two dilated heads per layer"},
            {"item": "attended pairs per layer at T 256", "value": str(t["attended_pairs_per_layer"]),
             "source": "run.json theoretical_costs"},
            {"item": "trainable parameters", "value": str(runs[0]["parameter_count"]),
             "source": "run.json parameter_count (0% change from the dense baseline)"},
            {"item": "sequences consumed (near, mixed)",
             "value": json.dumps([[r["curriculum_consumed"]["n_near"], r["curriculum_consumed"]["n_mixed"]] for r in runs]),
             "source": "run.json curriculum_consumed per seed"},
            {"item": "development-selection rule", "value": B3_RULE, "source": "protocol AM3.3"}]


# ------------------------------------------------------------------------------------------------ Part A tables
def grad_err(x, p=""):
    return max(x[p + "dq_max_abs_err"], x[p + "dk_max_abs_err"], x[p + "dv_max_abs_err"])


def check_part_a(D):
    from collections import Counter
    for key, rows in (("verify", D["verify"]), ("bench", D["bench"])):
        run = D[f"{key}_run"]
        check(len(rows) == run["n_rows"], f"{key} row count")
        got = Counter(x["status"] for x in rows)
        check(dict(got) == {k: v for k, v in run["status_counts"].items() if v}, f"{key} status counts")
    for x in D["bench"]:
        if x["status"] == "ok":
            check(close(float(np.median(x["times_ms"])), x["median_ms"]), f"bench median {x['label']} {x['L']}")
    return {"verify_rows": len(D["verify"]), "bench_rows": len(D["bench"])}


def a_errors(D):
    rows = []
    for x in D["verify"]:
        if x["kind"] != "accuracy":
            continue
        r = {"implementation": x["label"], "impl_family": x["impl"], "dtype": x["dtype"], "L": x["L"],
             "status": x["status"], "reference_inputs": x["reference_inputs"], "error": x.get("error")}
        if x["status"] == "ok":
            r.update(out_max_abs_err=x["out_max_abs_err"], grad_max_abs_err=grad_err(x),
                     dq_max_abs_err=x["dq_max_abs_err"], dk_max_abs_err=x["dk_max_abs_err"],
                     dv_max_abs_err=x["dv_max_abs_err"], out_mean_abs_err=x["out_mean_abs_err"],
                     finite=all(x[k] for k in ("out_finite", "dq_finite", "dk_finite", "dv_finite")),
                     vs_fp32_inputs_out_max_abs_err=x.get("vs_fp32_inputs_out_max_abs_err"),
                     vs_fp32_inputs_grad_max_abs_err=(grad_err(x, "vs_fp32_inputs_") if x["dtype"] == "fp16" else None))
        rows.append(r)
    return rows


def a_bench(D):
    rows = []
    for x in D["bench"]:
        r = {"implementation": x["label"], "dtype": x["dtype"], "L": x["L"], "pass": x["pass"], "status": x["status"],
             "error": x.get("error"), "repetitions": x["repetitions"], "warmup": x["warmup"], "timer": x["timer"]}
        if x["status"] == "ok":
            r.update(median_ms=x["median_ms"], min_ms=x["min_ms"], max_ms=x["max_ms"],
                     peak_allocated_bytes=x["peak_allocated_bytes"],
                     baseline_allocated_bytes=x["baseline_allocated_bytes"],
                     added_peak_bytes=x["peak_allocated_bytes"] - x["baseline_allocated_bytes"])
        rows.append(r)
    return rows


def a_backend(D):
    rows = []
    for x in sorted(D["backend"], key=lambda r: (r["dtype"], r["pass"], r["L"])):
        rows.append({"dtype": x["dtype"], "pass": x["pass"], "L": x["L"],
                     "default_backend": x["default"]["inferred_backend"],
                     "default_cuda_kernels": x["default"]["cuda_kernels"], "default_aten_ops": x["default"]["aten_ops"],
                     **{f"forced_{k}": ("supported" if v["supported"] else f"unsupported: {v['error']}")
                        for k, v in sorted(x["forced"].items())}})
    return rows


def a_backend_compact(D):
    groups = {}
    for x in D["backend"]:
        key = (x["dtype"], x["pass"], x["default"]["inferred_backend"], tuple(x["default"]["cuda_kernels"]),
               tuple(sorted(k for k, v in x["forced"].items() if v["supported"])),
               tuple(sorted(k for k, v in x["forced"].items() if not v["supported"])))
        groups.setdefault(key, []).append(x["L"])
    return [{"dtype": k[0], "pass": k[1], "default_backend": k[2], "default_cuda_kernels": list(k[3]),
             "forced_supported": list(k[4]), "forced_unsupported": list(k[5]), "L": sorted(v)}
            for k, v in sorted(groups.items())]


def a_unsupported(D):
    groups = {}
    for name, rows in (("verify", D["verify"]), ("bench", D["bench"])):
        for x in rows:
            if x["status"] == "ok":
                continue
            key = (name, x["status"], x["label"], x["dtype"], x["kind"] if name == "verify" else x["pass"], x.get("error"))
            groups.setdefault(key, []).append(x["L"])
    rows = [{"source": k[0], "status": k[1], "implementation": k[2], "dtype": k[3], "kind_or_pass": k[4],
             "error": k[5], "L": sorted(v), "n_rows": len(v)} for k, v in sorted(groups.items(), key=str)]
    for st in ("skipped", "oom"):
        if not any(r["status"] == st for r in rows):
            rows.append({"source": "verify and bench", "status": st, "implementation": "", "dtype": "",
                         "kind_or_pass": "", "error": "no row with this status", "L": [], "n_rows": 0})
    return rows


def a_storage(D):
    bk = {(x["label"], x["dtype"], x["L"], x["pass"]): x for x in D["bench"] if x["status"] == "ok"}
    rows = []
    for sb in sorted(D["score_bytes"], key=lambda r: (r["dtype"], r["L"])):
        dt, L = sb["dtype"], sb["L"]
        r = {"dtype": dt, "L": L, "score_matrix_bytes": sb["score_matrix_bytes"],
             "causal_mask_bytes": sb["causal_mask_bytes"], "dense_projected_peak_bytes": sb["dense_projected_peak_bytes"],
             "tiled_block_bytes_fp32_128x128": sb["tiled_block_bytes_fp32_128x128"],
             "dense_attempted": sb["dense_attempted"], "projection_formula": sb["projection_formula"]}
        for impl in ("dense", A_BENCH_TILE, "sdpa_default"):
            for p in ("forward", "forward_backward"):
                x = bk[(impl, dt, L, p)]
                r[f"{impl}_{p}_peak_bytes"] = x["peak_allocated_bytes"]
                r[f"{impl}_{p}_added_bytes"] = x["peak_allocated_bytes"] - x["baseline_allocated_bytes"]
        r["tiled_over_dense_forward_backward_peak"] = (r[f"{A_BENCH_TILE}_forward_backward_peak_bytes"]
                                                      / r["dense_forward_backward_peak_bytes"])
        rows.append(r)
    return rows


def peak_slope(D, impl, dtype, field="peak"):
    bk = {(x["label"], x["dtype"], x["L"], x["pass"]): x for x in D["bench"] if x["status"] == "ok"}
    a, b = bk[(impl, dtype, 2048, "forward")], bk[(impl, dtype, 8192, "forward")]
    if field == "peak":
        return math.log(b["peak_allocated_bytes"] / a["peak_allocated_bytes"]) / math.log(4)
    da = a["peak_allocated_bytes"] - a["baseline_allocated_bytes"]
    db = b["peak_allocated_bytes"] - b["baseline_allocated_bytes"]
    return math.log(db / da) / math.log(4) if da > 0 and db > 0 else None


# ------------------------------------------------------------------------------------------------ Part C
def by_method(D):
    out = {m: sorted([r for r in D["eval"] if r["method"] == m], key=lambda r: r["id"]) for m in METHODS}
    ids = [r["id"] for r in out["fp16"]]
    check(len(ids) == 180 and len(set(ids)) == 180, "fp16 ids")
    check(all([r["id"] for r in out[m]] == ids for m in METHODS), "id lists differ across methods")
    return out, ids


def check_part_c(D, E, ids):
    P = {p["id"]: p for p in D["prompts"]}
    check(sorted(P) == ids, "prompt ids")
    run = D["eval_run"]
    check(run["summary"]["n_rows"] == len(D["eval"]) == 1080, "eval row count")
    n_ev = 0
    for m in METHODS:
        X = E[m]
        for r in X:
            ms = CODE_RE.findall(r["generated_text"])
            check((ms[0] if ms else None) == (r["prediction"] or None), f"extractor {m} {r['id']}")
            check(r["correct"] == (r["prediction"] == P[r["id"]]["code"]), f"correct {m} {r['id']}")
            check(r["no_match"] == (not ms), f"no_match {m} {r['id']}")
            check(r["context_bytes"]["total"] == r["formula_bytes"], f"formula bytes {m} {r['id']}")
            check(close(r["realized_fraction"], r["context_bytes"]["total"] / r["M16"]), f"fraction {m} {r['id']}")
            if m != "fp16":
                check(r["context_bytes"]["total"] <= r["M4"], f"over budget {m} {r['id']}")
            z = D["npz"][r["id"]]
            ev = set(z["evidence_positions"].tolist())
            check(len(ev) == r["evidence_token_count"] == P[r["id"]]["evidence_token_count"], f"evidence count {r['id']}")
            kept = {"sink_recent": "sink_recent_positions", "question_aware": "question_aware_positions",
                    "hybrid": "hybrid_fp16_positions"}.get(m)
            if kept:
                pos = z[kept].tolist()
                check(len(pos) == (r["hybrid_f"] if m == "hybrid" else r["eviction_n"]), f"kept count {m} {r['id']}")
                k = len(ev & set(pos))
                check(k == r["evidence_retained_count"] and close(k / len(ev), r["evidence_fraction_retained"]),
                      f"evidence retained {m} {r['id']}")
                n_ev += 1
            else:
                check(r["evidence_fraction_retained"] is None, f"evidence field {m} {r['id']}")
        s = run["summary"]["per_method"][m]
        got = {"n": len(X), "n_correct": sum(r["correct"] for r in X),
               "accuracy": float(np.mean([r["correct"] for r in X])),
               "mean_answer_nll": float(np.mean([r["answer_nll_mean"] for r in X])),
               "mean_kl_post_prefill": float(np.mean([r["kl_post_prefill"] for r in X])),
               "mean_kl_teacher_forced": float(np.mean([r["kl_teacher_forced_mean"] for r in X])),
               "mean_realized_fraction": float(np.mean([r["realized_fraction"] for r in X])),
               "n_no_match": sum(r["no_match"] for r in X), "n_hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in X),
               "n_positions_preserved": sum(r["positions_preserved"] for r in X)}
        for k, v in got.items():
            check(close(v, s[k]), f"eval summary {m} {k}: {v} vs {s[k]}")
    for cell, v in run["summary"]["fp16_accuracy_by_cell"].items():
        lv, pos, dis = cell.split("-")
        X = [r for r in E["fp16"] if r["target_tokens"] == int(lv) and r["position"] == pos and r["distractors"] == int(dis)]
        check(len(X) == v["n"] and sum(r["correct"] for r in X) == v["n_correct"], f"fp16 cell {cell}")
    V = D["validate"]
    vs = D["validate_run"]["summary"]
    check(len(V) == 180 and not vs["diverged"], "validate rows")
    return {"eval_rows": len(D["eval"]), "evidence_rows_checked": n_ev,
            "validate_tokens_identical": sum(r["tokens_identical"] for r in V)}


def c1_table(D, E):
    fp_ok = {r["id"] for r in E["fp16"] if r["correct"]}
    rows = []
    for m in METHODS:
        X = E[m]
        cond = [r["correct"] for r in X if r["id"] in fp_ok]
        ev = [r["evidence_fraction_retained"] for r in X if r["evidence_fraction_retained"] is not None]
        kr = [r["k_recon_mean"] for r in X if r["k_recon_mean"] is not None]
        vr = [r["v_recon_mean"] for r in X if r["v_recon_mean"] is not None]
        rf = [r["realized_fraction"] for r in X]
        r = {"method": m, "label": METHOD_LABEL[m], "n": len(X), "exact_match": float(np.mean([x["correct"] for x in X])),
             "n_correct": sum(x["correct"] for x in X),
             "exact_match_conditional_fp16_correct": float(np.mean(cond)) if cond else None,
             "n_fp16_correct": len(cond), "mean_reference_answer_nll": float(np.mean([x["answer_nll_mean"] for x in X])),
             "mean_post_prefill_kl": float(np.mean([x["kl_post_prefill"] for x in X])),
             "mean_teacher_forced_kl": float(np.mean([x["kl_teacher_forced_mean"] for x in X])),
             "mean_k_recon_rel_error": float(np.mean(kr)) if kr else None,
             "mean_v_recon_rel_error": float(np.mean(vr)) if vr else None,
             "mean_evidence_retained": float(np.mean(ev)) if ev else None,
             "realized_fraction_min": min(rf), "realized_fraction_median": float(np.median(rf)),
             "realized_fraction_max": max(rf)}
        for k in ("payload", "metadata", "fp16_retained", "fp16_tail", "indices", "total"):
            r[f"mean_context_bytes_{k}"] = float(np.mean([x["context_bytes"][k] for x in X]))
        r["mean_shared_decode_time_bytes"] = float(np.mean([x["decode_time_bytes"] for x in X]))
        r["mean_total_live_bytes_end"] = float(np.mean([x["total_live_bytes_end"] for x in X]))
        r["n_no_match"] = sum(x["no_match"] for x in X)
        r["n_hit_max_new_tokens"] = sum(x["hit_max_new_tokens"] for x in X)
        r["count_range"] = ({"N": [min(x["eviction_n"] for x in X), max(x["eviction_n"] for x in X)]} if m in EVICTION
                            else {"m": [min(x["hybrid_m"] for x in X), max(x["hybrid_m"] for x in X)],
                                  "F": [min(x["hybrid_f"] for x in X), max(x["hybrid_f"] for x in X)]}
                            if m == "hybrid" else None)
        rows.append(r)
    return rows


def c_rows(E):
    rows = []
    for m in METHODS:
        for x in E[m]:
            rows.append({"id": x["id"], "method": m, "target_tokens": x["target_tokens"], "position": x["position"],
                         "distractors": x["distractors"], "context_len": x["context_len"], "correct": x["correct"],
                         "prediction": x["prediction"], "no_match": x["no_match"], "answer_nll_mean": x["answer_nll_mean"],
                         "kl_post_prefill": x["kl_post_prefill"], "kl_teacher_forced_mean": x["kl_teacher_forced_mean"],
                         "k_recon_mean": x["k_recon_mean"], "v_recon_mean": x["v_recon_mean"],
                         "evidence_fraction_retained": x["evidence_fraction_retained"],
                         "realized_fraction": x["realized_fraction"], "context_bytes_total": x["context_bytes"]["total"],
                         "M4": x["M4"], "M16": x["M16"], "decode_time_bytes": x["decode_time_bytes"],
                         "peak_allocated_bytes": x["peak_allocated_bytes"]})
    return rows


def c2_bootstrap(E):
    rows = []
    for a, b in C_PAIRS:
        for metric, key in (("exact_match", "correct"), ("reference_answer_nll", "answer_nll_mean")):
            va = [float(x[key]) for x in E[a]]
            vb = [float(x[key]) for x in E[b]]
            res = B.paired_acc_diff(va, vb, n_resamples=BOOT_RESAMPLES, seed=BOOT_SEED, unit="prompt")
            check(close(res["point"], float(np.mean(vb) - np.mean(va))), f"bootstrap point {a} {b} {metric}")
            rows.append({"metric": metric, "a": a, "b": b, "difference": f"{b} minus {a}", "point": res["point"],
                         "ci_low": res["ci"][0], "ci_high": res["ci"][1], "ci_excludes_0": not (res["ci"][0] <= 0 <= res["ci"][1]),
                         "n_prompts": res["n_units"], "n_resamples": res["n_resamples"], "seed": res["seed"],
                         "percentiles": res["percentiles"], "method": res["method"] + "; statistic: mean difference"})
    return rows


def c_cells(E):
    rows = []
    for m in METHODS:
        for lv in LEVELS:
            for pos in POSITIONS:
                for dis in DISTRACTORS:
                    X = [x for x in E[m] if x["target_tokens"] == lv and x["position"] == pos and x["distractors"] == dis]
                    rows.append({"method": m, "grouping": "cell", "target_tokens": lv, "position": pos, "distractors": dis,
                                 "n": len(X), "exact_match": float(np.mean([x["correct"] for x in X])),
                                 "mean_reference_answer_nll": float(np.mean([x["answer_nll_mean"] for x in X])),
                                 "mean_evidence_retained": (float(np.mean([x["evidence_fraction_retained"] for x in X]))
                                                            if X[0]["evidence_fraction_retained"] is not None else None)})
        for lv in LEVELS:
            for pos in POSITIONS:
                X = [x for x in E[m] if x["target_tokens"] == lv and x["position"] == pos]
                rows.append({"method": m, "grouping": "length x location", "target_tokens": lv, "position": pos,
                             "distractors": "all", "n": len(X), "exact_match": float(np.mean([x["correct"] for x in X])),
                             "mean_reference_answer_nll": float(np.mean([x["answer_nll_mean"] for x in X])),
                             "mean_evidence_retained": (float(np.mean([x["evidence_fraction_retained"] for x in X]))
                                                        if X[0]["evidence_fraction_retained"] is not None else None)})
        for dis in DISTRACTORS:
            X = [x for x in E[m] if x["distractors"] == dis]
            rows.append({"method": m, "grouping": "distractors", "target_tokens": "all", "position": "all",
                         "distractors": dis, "n": len(X), "exact_match": float(np.mean([x["correct"] for x in X])),
                         "mean_reference_answer_nll": float(np.mean([x["answer_nll_mean"] for x in X])),
                         "mean_evidence_retained": (float(np.mean([x["evidence_fraction_retained"] for x in X]))
                                                    if X[0]["evidence_fraction_retained"] is not None else None)})
    for r in rows:
        check(r["n"] == {"cell": 10, "length x location": 30, "distractors": 60}[r["grouping"]], f"cell size {r}")
    return rows


def c3_systems(D, E):
    P = D["prompts"]
    rows = []
    for m in METHODS:
        X = E[m]
        r = {"method": m, "label": METHOD_LABEL[m], "n": len(X),
             "mean_full_prefill_s": float(np.mean([p["prefill_seconds"] for p in P])),
             "mean_scoring_s": float(np.mean([p["scoring_seconds"] for p in P])),
             "scoring_used_by_method": m in ("question_aware", "hybrid"),
             "max_prefill_peak_allocated_bytes": max(p["prefill_peak_allocated_bytes"] for p in P)}
        for k in ("compression", "suffix", "decode", "teacher_forced"):
            r[f"mean_{k}_s"] = float(np.mean([x["seconds"][k] for x in X]))
        r["mean_peak_allocated_bytes"] = float(np.mean([x["peak_allocated_bytes"] for x in X]))
        r["max_peak_allocated_bytes"] = max(x["peak_allocated_bytes"] for x in X)
        r["mean_added_peak_bytes"] = float(np.mean([x["peak_allocated_bytes"] - x["start_allocated_bytes"] for x in X]))
        r["mean_calculated_context_bytes"] = float(np.mean([x["formula_bytes"] for x in X]))
        r["mean_serialized_context_bytes"] = float(np.mean([x["context_bytes"]["total"] for x in X]))
        r["mean_total_live_cache_bytes_end"] = float(np.mean([x["total_live_bytes_end"] for x in X]))
        r["note"] = ("prefill and scoring are measured once per prompt and shared by all methods; serialized bytes are "
                     "the packed tensors counted; allocated memory includes model weights and temporary dequantized "
                     "K and V")
        rows.append(r)
    return rows


def c4_bench(D):
    B_ = D["cbench"]
    check(len(B_) == 16 and all(x["status"] == "ok" for x in B_), "C5 bench rows")
    fp = {(x["L"], x["batch"]): x for x in B_ if x["config"] == "fp16"}
    rows = []
    order = ("fp16", "int4_token", "int4_channel_k", "hybrid")
    for x in sorted(B_, key=lambda r: (r["L"], r["batch"], order.index(r["config"]))):
        st = x["stats_ms"]
        for k, v in x["times_ms"].items():
            check(len(v) == x["n_reps"] == 10 and close(float(np.median(v)), st[k]["median"]), f"bench median {k}")
        check(x["bytes_per_sequence"]["total"] == x["formula_bytes_per_sequence"], "bench bytes")
        base = fp[(x["L"], x["batch"])]["stats_ms"]["decode_step"]["median"]
        rows.append({"config": x["config"], "L": x["L"], "batch": x["batch"], "n_reps": x["n_reps"],
                     "n_warmup": x["n_warmup"],
                     "pack_ms_median": st["pack"]["median"] if st.get("pack") else None,
                     "unpack_dequant_ms_median": st["unpack_dequant"]["median"] if st.get("unpack_dequant") else None,
                     "decode_step_ms_median": st["decode_step"]["median"],
                     "decode_step_ratio_to_fp16": st["decode_step"]["median"] / base,
                     "serialized_bytes_per_sequence": x["bytes_per_sequence"]["total"],
                     "fp16_bytes_per_sequence": x["fp16_bytes_per_sequence"], "realized_fraction": x["realized_fraction"],
                     "bytes_by_component_per_sequence": x["bytes_per_sequence"],
                     "peak_allocated_bytes": x["peak_allocated_bytes"], "peak_reserved_bytes": x["peak_reserved_bytes"],
                     "cache_ids": x["cache_ids"]})
    return rows


def check_diag(D):
    R = D["diag"]
    run = D["diag_run"]
    check(len(R) == run["summary"]["n_rows"] == 108, "diag rows")
    ids = sorted(p["id"] for p in D["diag_prompts"])
    check(ids == sorted(run["config"]["diagnostic_prompt_ids"]) and len(ids) == 18, "diag ids")
    P = {p["id"]: p for p in D["diag_prompts"]}
    for x in R:
        ms = CODE_RE.findall(x["generated_text"])
        check((ms[0] if ms else None) == (x["prediction"] or None), f"diag extractor {x['config']} {x['id']}")
        check(x["correct"] == (x["prediction"] == P[x["id"]]["code"]), f"diag correct {x['id']}")
    per = {}
    for c in DIAG_CONFIGS:
        X = [x for x in R if x["config"] == c]
        check(len(X) == 18, f"diag rows {c}")
        got = {"n": len(X), "n_correct": sum(x["correct"] for x in X), "accuracy": float(np.mean([x["correct"] for x in X])),
               "mean_answer_nll": float(np.mean([x["answer_nll_mean"] for x in X])),
               "mean_kl_post_prefill": float(np.mean([x["kl_post_prefill"] for x in X])),
               "mean_kl_teacher_forced": float(np.mean([x["kl_teacher_forced_mean"] for x in X])),
               "mean_k_recon": float(np.mean([x["k_recon_mean"] for x in X])),
               "mean_v_recon": float(np.mean([x["v_recon_mean"] for x in X])),
               "n_no_match": sum(x["no_match"] for x in X), "n_hit_max_new_tokens": sum(x["hit_max_new_tokens"] for x in X)}
        s = run["summary"]["per_config"][c]
        for k, v in got.items():
            check(close(v, s[k]), f"diag summary {c} {k}")
        per[c] = got | {"label": s["label"]}
    keys = ("k_ratio", "v_ratio", "k_err_token_top", "k_err_token_rest", "k_err_channel_top", "k_err_channel_rest",
            "v_err_token_top", "v_err_token_rest")
    Z = [D["diag_npz"][i] for i in ids]
    per_layer = {k: np.mean(np.stack([z[k] for z in Z]), axis=(0, 2)) for k in keys}
    overall = {k: float(per_layer[k].mean()) for k in keys}
    ch = run["summary"]["channel"]
    for k in keys:
        check(close(overall[k], ch["overall"][k], 1e-6), f"channel overall {k}")
        check(all(close(float(a), b, 1e-6) for a, b in zip(per_layer[k], ch["per_layer"][k])), f"channel per layer {k}")
    return per, per_layer, overall


def am4_readout(per, overall):
    fp = per["fp16"]["accuracy"]
    vals = {
        "c_int8": (abs(per["kv_int8_token"]["accuracy"] - fp) <= 0.05 and per["kv_int8_token"]["mean_kl_teacher_forced"] < 0.05),
        "a_k_only": per["k_int4_token"]["accuracy"] <= 0.1,
        "b_v_only": abs(per["v_int4_token"]["accuracy"] - fp) <= 0.05,
        "k_ratio": overall["k_ratio"] >= 3 * overall["v_ratio"],
        "k_rest_error": overall["k_err_token_rest"] >= 3 * overall["k_err_channel_rest"],
    }
    vals["rival_token_path_defect"] = not vals["c_int8"]
    return vals


def c5_tables(per, per_layer, overall):
    ctrl = [{"config": c, "label": per[c]["label"], **{k: v for k, v in per[c].items() if k != "label"},
             "post_hoc": True, "status": "AM4 post hoc diagnostic; not a matched-budget condition"} for c in DIAG_CONFIGS]
    chan = []
    for i in range(len(per_layer["k_ratio"])):
        chan.append({"layer": i, **{k: float(per_layer[k][i]) for k in per_layer}})
    chan.append({"layer": "mean over layers", **overall})
    return ctrl, chan


def qualitative(D, E):
    P = {p["id"]: p for p in D["prompts"]}
    ids = sorted(D["qual"])
    check(len(ids) == 6, "qualitative prompt count")
    recs, parts = [], [f"Qualitative export (Task 2 Part C). Selection rule: {QUAL_RULE}.",
                       f"Prompt IDs: {', '.join(ids)}", ""]
    for i in ids:
        p = P[i]
        recs.append({"id": i, "target_tokens": p["target_tokens"], "position": p["position"],
                     "distractors": p["distractors"], "code": p["code"], "context_len": p["context_len"],
                     "evidence_positions": D["npz"][i]["evidence_positions"].tolist(),
                     "methods": {m: {"correct": x["correct"], "prediction": x["prediction"],
                                     "generated_text": x["generated_text"],
                                     "evidence_fraction_retained": x["evidence_fraction_retained"]}
                                 for m in METHODS for x in E[m] if x["id"] == i},
                     "source_file": p["qualitative"]})
        parts += [f"===== {i} ({p['qualitative']}) =====", D["qual"][i].rstrip("\n"), ""]
    return recs, "\n".join(parts) + "\n"


# ------------------------------------------------------------------------------------------------ predictions
def _pred(pid, text, metric, threshold, measured, holds, notes=""):
    return {"prediction": pid, "protocol_text": text, "metric": metric, "threshold": threshold,
            "measured": measured, "comparison_holds": holds, "notes": notes}


def predictions(D, E, c1, boot, bench, per, overall, am4):
    P = []
    v = [x for x in D["verify"] if x["kind"] == "accuracy" and x["status"] == "ok"]
    bk = {(x["label"], x["dtype"], x["L"], x["pass"]): x for x in D["bench"] if x["status"] == "ok"}
    # A-H1
    f32 = max(x["out_max_abs_err"] for x in v if x["dtype"] == "fp32")
    P.append(_pred("A-H1a", "FP32: dense, tiled and SDPA within 1e-5 max abs error of the FP32 dense reference",
                   "max output max-abs error over FP32 accuracy rows (dense, 4 tiles, supported SDPA backends, all L)",
                   "<= 1e-5", {"max_out_err": f32,
                               "max_grad_err": max(grad_err(x) for x in v if x["dtype"] == "fp32")}, f32 <= 1e-5,
                   "gradient error reported alongside; the prediction names output error"))
    det = {}
    for L in (2048, 4096, 8192):
        d = [x for x in v if x["dtype"] == "fp16" and x["L"] == L]
        dense = [x["out_max_abs_err"] for x in d if x["label"] == "dense"][0]
        det[L] = {"dense": dense, **{x["label"]: x["out_max_abs_err"] for x in d if x["impl"] == "tiled"}}
    hb = all(det[L]["dense"] > e for L in det for k, e in det[L].items() if k != "dense")
    P.append(_pred("A-H1b", "FP16: dense output error larger than tiled at L >= 2048",
                   "FP16 output max-abs error (AM1 reference), dense vs each tile, L 2048-8192", "dense > tiled at every L, tile",
                   det, hb))
    ratios = {L: {k: det[L]["dense"] / e for k, e in det[L].items() if k != "dense"} for L in det}
    rv = all(0.5 <= r <= 2 for L in ratios for r in ratios[L].values())
    P.append(_pred("A-H1 rival", "FP16 errors equal within a factor of 2", "dense / tiled FP16 output error ratio, L >= 2048",
                   "every ratio in [0.5, 2]", ratios, rv))
    # A-H2
    sl = {f"{i}|{dt}": {"peak": peak_slope(D, i, dt), "added": peak_slope(D, i, dt, "added")}
          for i in ("dense", A_BENCH_TILE, "sdpa_default", "sdpa_MATH", "sdpa_EFFICIENT_ATTENTION") for dt in A_DTYPES}
    ok2 = all(sl[f"dense|{dt}"]["peak"] >= 1.8 and sl[f"{A_BENCH_TILE}|{dt}"]["peak"] <= 1.2
              and sl[f"sdpa_default|{dt}"]["peak"] <= 1.2 for dt in A_DTYPES)
    P.append(_pred("A-H2", "Forward peak allocated memory log-log slope from L 2048 to 8192: dense >= 1.8; tiled and SDPA <= 1.2",
                   "log(peak(8192)/peak(2048))/log 4 of recorded forward peak allocated bytes (bench tile 128x128; SDPA = default dispatch)",
                   "dense >= 1.8; tiled <= 1.2; sdpa_default <= 1.2 (both dtypes)", sl, ok2,
                   "'added' = peak minus allocation before the call; forced MATH and EFFICIENT_ATTENTION shown for reference"))
    rq = any(sl[f"sdpa_default|{dt}"]["peak"] >= 1.8 for dt in A_DTYPES)
    P.append(_pred("A-H2 rival", "SDPA dispatches a quadratic path on T4 for some dtype",
                   "sdpa_default forward peak slope and the default backend in the backend table", "slope >= 1.8 for some dtype",
                   {dt: sl[f"sdpa_default|{dt}"]["peak"] for dt in A_DTYPES}, rq,
                   "default backend by profiler kernels: " + ", ".join(sorted({f"{x['dtype']} {x['pass']} {x['default']['inferred_backend']}" for x in D["backend"]}))))
    # A-H3
    r3 = {f"{dt}|{L}": bk[(A_BENCH_TILE, dt, L, "forward_backward")]["peak_allocated_bytes"]
          / bk[("dense", dt, L, "forward_backward")]["peak_allocated_bytes"] for dt in A_DTYPES for L in (2048, 4096, 8192)}
    P.append(_pred("A-H3", "Tiled forward plus autograd backward peak memory >= 0.8x dense at L >= 2048",
                   "tiled_128x128 / dense forward+backward peak allocated bytes", ">= 0.8 at every L >= 2048, both dtypes",
                   r3, all(x >= 0.8 for x in r3.values())))
    # A-H4
    fastest, slower = {}, {}
    for dt in A_DTYPES:
        for L in A_LENGTHS:
            for p in ("forward", "forward_backward"):
                s = bk[("sdpa_default", dt, L, p)]["median_ms"]
                others = {i: bk[(i, dt, L, p)]["median_ms"] for i in ("dense", A_BENCH_TILE)}
                fastest[f"{dt}|{L}|{p}"] = s < min(others.values())
                if L <= 1024:
                    slower[f"{dt}|{L}|{p}"] = others[A_BENCH_TILE] > others["dense"]
    P.append(_pred("A-H4a", "SDPA fastest in every condition", "median ms: sdpa_default vs dense and tiled_128x128 per (dtype, L, pass)",
                   "sdpa_default lowest in all 24 conditions", {"n_fastest": sum(fastest.values()), "n": len(fastest),
                                                                 "not_fastest": [k for k, x in fastest.items() if not x]},
                   all(fastest.values())))
    P.append(_pred("A-H4b", "Python tiled slower than dense at L <= 1024", "median ms tiled_128x128 vs dense, L 256-1024, both passes and dtypes",
                   "tiled > dense in all 12 conditions", {"n_slower": sum(slower.values()), "n": len(slower)},
                   all(slower.values())))
    faster = {f"{dt}|{L}|{p}": bk[(A_BENCH_TILE, dt, L, p)]["median_ms"] < bk[("dense", dt, L, p)]["median_ms"]
              for dt in A_DTYPES for L in (2048, 4096, 8192) for p in ("forward", "forward_backward")}
    P.append(_pred("A-H4 rival", "tiled faster than dense at some L >= 2048", "median ms tiled_128x128 vs dense, L >= 2048",
                   "tiled < dense in at least one condition", [k for k, x in faster.items() if x], any(faster.values())))
    # B-H1
    lc = {c: learned_count(D, c) for c in CONFIGS}
    P.append(_pred("B-H1a", "Dense mixed learned at K 4 and 8", "B6 learned seeds (dense gate: all three)",
                   "3/3 seeds at K 4 and K 8", {"dense_mixed_K4": lc["dense_mixed_K4"], "dense_mixed_K8": lc["dense_mixed_K8"]},
                   lc["dense_mixed_K4"] == 3 and lc["dense_mixed_K8"] == 3))
    a4 = seed_acc(D, "dense_mixed_K4", K=4, regime="mixed")
    a16 = seed_acc(D, "dense_mixed_K16", K=16, regime="mixed")
    P.append(_pred("B-H1b", "mixed accuracy falls from K 4 to 16",
                   "own-cell mixed test accuracy: dense_mixed_K16 on mixed_K16 vs dense_mixed_K4 on mixed_K4 (seed means)",
                   "K16 mean < K4 mean and seed ranges do not overlap (B8)",
                   {"K4": mmm(a4), "K16": mmm(a16), "difference": float(np.mean(a16) - np.mean(a4)),
                    "seed_ranges_overlap": overlap(a4, a16)},
                   float(np.mean(a16)) < float(np.mean(a4)) and not overlap(a4, a16),
                   "B8 decides: a difference counts only without overlap"))
    gaps = {c: float(np.mean(seed_acc(D, c, K=parse_config(c)[2], regime="near"))
                     - np.mean(seed_acc(D, c, K=parse_config(c)[2], regime="far")))
            for c in ("dense_mixed_K4", "dense_mixed_K8", "dense_mixed_K16")}
    P.append(_pred("B-H1c", "near-far gap < 10 points", "near minus far test accuracy at the training K (seed means), dense mixed",
                   "|gap| < 0.10 for K 4, 8, 16", gaps, all(abs(g) < 0.10 for g in gaps.values())))
    P.append(_pred("B-H1 rival", "dense fails at K 16 or far deficit > 10 points",
                   "dense_mixed_K16 learned seeds; near minus far gaps", "learned < 3/3 or some gap > 0.10",
                   {"dense_mixed_K16_learned": lc["dense_mixed_K16"], "gaps": gaps},
                   lc["dense_mixed_K16"] < 3 or any(g > 0.10 for g in gaps.values())))
    # B-H2
    d2 = {}
    for c in ("dense_mixed_K4", "dense_mixed_K8", "dense_mixed_K16", "dense_near_K16"):
        for K in KS:
            d2[f"{c}|near_K{K}"] = float(np.mean(seed_acc(D, c, 1, K=K, regime="near"))
                                         - np.mean(seed_acc(D, c, 0, K=K, regime="near")))
    P.append(_pred("B-H2", "Dense with test-time W = 64 mask within 5 points of unmasked dense on the near test set",
                   "window64 minus native near test accuracy (seed means), every dense configuration and test K",
                   "|difference| <= 0.05 everywhere", d2, all(abs(x) <= 0.05 for x in d2.values())))
    P.append(_pred("B-H2 rival", "drop > 10 points", "same differences", "some difference < -0.10", d2,
                   any(x < -0.10 for x in d2.values())))
    # B-H3
    wn = seed_acc(D, "window_near_K16", K=16, dmax=64)
    wm = seed_acc(D, "window_mixed_K16", K=16, dmax=64)
    diff3 = float(np.mean(wn) - np.mean(wm))
    P.append(_pred("B-H3", "Window trained on mixed K 16 more than 5 points below window trained on near K 16 on queries with distance < 64",
                   "test accuracy on all K16 test queries with distance < 64 (near, far and mixed cells pooled), seed means",
                   "near-trained minus mixed-trained > 0.05, seed ranges not overlapping (B8)",
                   {"window_near_K16": mmm(wn), "window_mixed_K16": mmm(wm), "difference": diff3,
                    "seed_ranges_overlap": overlap(wn, wm)}, diff3 > 0.05 and not overlap(wn, wm)))
    P.append(_pred("B-H3 rival", "within 5", "same difference", "|difference| <= 0.05", diff3, abs(diff3) <= 0.05))
    # B-H4
    g4 = {}
    for c in ("window_near_K4", "window_near_K8", "window_near_K16", "window_mixed_K16"):
        K = parse_config(c)[2]
        lo, hi = seed_acc(D, c, K=K, dmax=64), seed_acc(D, c, K=K, dmin=64, dmax=128)
        g4[c] = {"acc_0_64": float(np.mean(lo)), "acc_64_128": float(np.mean(hi)), "gap": float(np.mean(lo) - np.mean(hi)),
                 "seed_ranges_overlap": overlap(lo, hi)}
    P.append(_pred("B-H4", "Window models: [64,128) accuracy more than 30 points below [0,64)",
                   "test accuracy at the training K, all regimes pooled, by distance bin (seed means)",
                   "gap > 0.30 for every window configuration", g4, all(x["gap"] > 0.30 for x in g4.values()),
                   "window_mixed_K16 did not learn (B6) on any seed" if lc["window_mixed_K16"] == 0 else ""))
    P.append(_pred("B-H4 rival", "within 10", "same gaps", "|gap| <= 0.10 for some configuration",
                   {c: x["gap"] for c, x in g4.items()}, any(abs(x["gap"]) <= 0.10 for x in g4.values())))
    # B-H5
    l4, l16 = seed_acc(D, "linear_mixed_K4", K=4, regime="mixed"), seed_acc(D, "linear_mixed_K16", K=16, regime="mixed")
    dl = [a - b for a, b in zip(l4, l16)]
    dd = [a - b for a, b in zip(a4, a16)]
    lg = {c: float(np.mean(seed_acc(D, c, K=parse_config(c)[2], regime="near"))
                   - np.mean(seed_acc(D, c, K=parse_config(c)[2], regime="far")))
          for c in ("linear_mixed_K4", "linear_mixed_K8", "linear_mixed_K16")}
    P.append(_pred("B-H5a", "Linear accuracy drop from K 4 to 16 larger than dense's",
                   "own-cell mixed accuracy K4 minus K16, per seed (same seed index); linear vs dense",
                   "linear mean drop > dense mean drop and per-seed drop ranges not overlapping (B8)",
                   {"linear_drop": mmm(dl), "dense_drop": mmm(dd), "seed_ranges_overlap": overlap(dl, dd)},
                   float(np.mean(dl)) > float(np.mean(dd)) and not overlap(dl, dd),
                   f"linear learned seeds: K4 {lc['linear_mixed_K4']}/3, K8 {lc['linear_mixed_K8']}/3, K16 {lc['linear_mixed_K16']}/3; "
                   "B8 decides"))
    P.append(_pred("B-H5b", "near-far gap < 10 points", "near minus far test accuracy at the training K (seed means), linear mixed",
                   "|gap| < 0.10 for K 4, 8, 16", lg, all(abs(g) < 0.10 for g in lg.values())))
    linfail = all(lc[c] == 0 for c in ("linear_mixed_K4", "linear_mixed_K8", "linear_mixed_K16"))
    P.append(_pred("B-H5 rival", "drop not larger, or linear fails at every K", "B-H5a comparison; learned seeds per K",
                   "B-H5a does not hold or 0/3 learned at every K", {"linear_learned": [lc[c] for c in ("linear_mixed_K4", "linear_mixed_K8", "linear_mixed_K16")]},
                   not (float(np.mean(dl)) > float(np.mean(dd)) and not overlap(dl, dd)) or linfail))
    # B-H6
    ld, wd = seed_acc(D, "local_dilated_mixed_K4", K=4, regime="far"), seed_acc(D, "window_near_K4", K=4, regime="far")
    pre = (f"precondition not met: local_dilated_mixed_K4 learned on {lc['local_dilated_mixed_K4']}/3 seeds (B6)"
           if lc["local_dilated_mixed_K4"] < 3 else "")
    P.append(_pred("B-H6a", "Local/dilated far accuracy more than 20 points above window far accuracy at K 4",
                   "far_K4 test accuracy: local_dilated_mixed_K4 minus window_near_K4 (seed means)",
                   "> 0.20 and seed ranges not overlapping (B8)",
                   {"local_dilated": mmm(ld), "window": mmm(wd), "difference": float(np.mean(ld) - np.mean(wd)),
                    "seed_ranges_overlap": overlap(ld, wd)},
                   float(np.mean(ld) - np.mean(wd)) > 0.20 and not overlap(ld, wd), pre))
    r0 = seed_acc(D, "local_dilated_mixed_K4", K=4, regime="far", residue=0)
    rest = [float(np.mean(x)) for x in zip(*[seed_acc(D, "local_dilated_mixed_K4", K=4, regime="far", residue=r) for r in (1, 2, 3)])]
    P.append(_pred("B-H6b", "higher by more than 10 points when (distance - 1) mod 4 = 0",
                   "local_dilated_mixed_K4 far_K4 accuracy at residue 0 minus the mean over residues 1-3 (per seed)",
                   "> 0.10 (seed means) and per-seed ranges not overlapping (B8)",
                   {"residue_0": mmm(r0), "residues_1_3": mmm(rest), "difference": float(np.mean(r0) - np.mean(rest)),
                    "seed_ranges_overlap": overlap(r0, rest)},
                   float(np.mean(r0) - np.mean(rest)) > 0.10 and not overlap(r0, rest), pre))
    P.append(_pred("B-H6 rival", "no alignment difference", "residue 0 minus residues 1-3 (B-H6b)", "|difference| <= 0.10",
                   float(np.mean(r0) - np.mean(rest)), abs(float(np.mean(r0) - np.mean(rest))) <= 0.10, pre))
    # B3-H
    cur, cmp_ = "local_dilated_curriculum_K16", "local_dilated_mixed_K16"
    cm, km = seed_acc(D, cur, K=16, regime="mixed"), seed_acc(D, cmp_, K=16, regime="mixed")
    far16 = seed_acc(D, cur, K=16, regime="far")
    P.append(_pred("B3-H a", "local_dilated_curriculum learns (B6) on at least 2 of 3 seeds", "B6 learned seeds (mixed_K16 cell)",
                   ">= 2", lc[cur], lc[cur] >= 2))
    P.append(_pred("B3-H b", "its mixed K 16 test accuracy is more than 30 points above the comparator's (seed means; B8 applies)",
                   "mixed_K16 test accuracy, curriculum minus local_dilated_mixed_K16", "> 0.30 and seed ranges not overlapping",
                   {"curriculum": mmm(cm), "comparator": mmm(km), "difference": float(np.mean(cm) - np.mean(km)),
                    "seed_ranges_overlap": overlap(cm, km)}, float(np.mean(cm) - np.mean(km)) > 0.30 and not overlap(cm, km)))
    P.append(_pred("B3-H c", "its far K 16 test accuracy is above 0.5 (seed mean)", "far_K16 test accuracy", "> 0.5",
                   mmm(far16), float(np.mean(far16)) > 0.5))
    near16 = seed_acc(D, cur, K=16, regime="near")
    P.append(_pred("B3-H rival R1", "it learns near queries but far K 16 accuracy stays near 1/K",
                   "near_K16 and far_K16 test accuracy (seed means)", "near >= 0.5 and far within 0.05 of 1/16",
                   {"near": float(np.mean(near16)), "far": float(np.mean(far16))},
                   float(np.mean(near16)) >= 0.5 and abs(float(np.mean(far16)) - 1 / 16) <= 0.05,
                   "operationalization of 'near 1/K': within 5 points"))
    devmax = max(x["accuracy"] for s in SEEDS for x in D["runs"][(cur, s)]["dev"] if x["cell"] == "mixed_K16")
    P.append(_pred("B3-H rival R2", "it stays flat (development accuracy near 1/K at every checkpoint)",
                   "max own-cell development accuracy over checkpoints and seeds", "<= 1/16 + 0.05", devmax,
                   devmax <= 1 / 16 + 0.05, "operationalization of 'near 1/K': within 5 points"))
    # C-H1
    cells = {f"{lv}-{pos}": float(np.mean([x["correct"] for x in E["fp16"] if x["target_tokens"] == lv and x["position"] == pos]))
             for lv in LEVELS for pos in POSITIONS}
    P.append(_pred("C-H1", "Full FP16 cache exact match >= 0.9 in every length x location cell",
                   "fp16 exact match per length x location group (n 30 each)", ">= 0.9 in all 6", cells,
                   all(x >= 0.9 for x in cells.values())))
    P.append(_pred("C-H1 rival", "lower for middle evidence at 4k", "fp16 exact match 4096-middle vs the other groups",
                   "4096-middle below 0.9", cells["4096-middle"], cells["4096-middle"] < 0.9))
    C1 = {r["method"]: r for r in c1}
    gap2 = C1["question_aware"]["exact_match"] - C1["sink_recent"]["exact_match"]
    P.append(_pred("C-H2a", "Question-aware eviction exceeds sink-plus-recent by > 30 points overall",
                   "exact match difference over 180 prompts", "> 0.30", gap2, gap2 > 0.30))
    sr_ok = [x for x in E["sink_recent"] if x["correct"]]
    share = sum(x["evidence_fraction_retained"] == 1.0 for x in sr_ok) / len(sr_ok) if sr_ok else None
    P.append(_pred("C-H2b", "sink-plus-recent succeeds mainly when evidence lies in its recent window",
                   "share of correct sink_recent prompts whose evidence tokens are all retained", "> 0.5",
                   {"share": share, "n_correct": len(sr_ok),
                    "correct_by_position": {pos: sum(x["position"] == pos for x in sr_ok) for pos in POSITIONS}},
                   share is not None and share > 0.5, "operationalization of 'mainly': more than half"))
    P.append(_pred("C-H2 rival", "gap < 10 points", "same difference", "< 0.10", gap2, gap2 < 0.10))
    qa_d = {d: float(np.mean([x["correct"] for x in E["question_aware"] if x["distractors"] == d])) for d in DISTRACTORS}
    fp_d = {d: float(np.mean([x["correct"] for x in E["fp16"] if x["distractors"] == d])) for d in DISTRACTORS}
    P.append(_pred("C-H3", "Question-aware retains >= 0.9 of evidence tokens and stays within 5 points of FP16 exact match at 0, 2 and 8 distractors",
                   "mean evidence retained; exact match difference to fp16 per distractor count (n 60 each)",
                   "retained >= 0.9 and |difference| <= 0.05 at each count",
                   {"evidence_retained": C1["question_aware"]["mean_evidence_retained"],
                    "difference_by_distractors": {d: qa_d[d] - fp_d[d] for d in DISTRACTORS}},
                   C1["question_aware"]["mean_evidence_retained"] >= 0.9 and all(abs(qa_d[d] - fp_d[d]) <= 0.05 for d in DISTRACTORS)))
    P.append(_pred("C-H3 rival", "exact match falls > 10 points from 0 to 8 distractors", "question_aware EM at 0 minus at 8 distractors",
                   "> 0.10", qa_d[0] - qa_d[8], qa_d[0] - qa_d[8] > 0.10))
    tk, ck = C1["int4_token"], C1["int4_channel_k"]
    fpm = C1["fp16"]["exact_match"]
    P.append(_pred("C-H4a", "Channel-K has lower K reconstruction error and lower teacher-forced KL than token-wise",
                   "mean K relative reconstruction error; mean teacher-forced KL", "channel-K lower on both",
                   {"k_recon": [ck["mean_k_recon_rel_error"], tk["mean_k_recon_rel_error"]],
                    "teacher_forced_kl": [ck["mean_teacher_forced_kl"], tk["mean_teacher_forced_kl"]]},
                   ck["mean_k_recon_rel_error"] < tk["mean_k_recon_rel_error"] and ck["mean_teacher_forced_kl"] < tk["mean_teacher_forced_kl"],
                   "values listed as [int4_channel_k, int4_token]"))
    P.append(_pred("C-H4b", "both within 5 points of FP16 exact match", "exact match difference to fp16",
                   "|difference| <= 0.05 for both layouts",
                   {"int4_token": tk["exact_match"] - fpm, "int4_channel_k": ck["exact_match"] - fpm},
                   abs(tk["exact_match"] - fpm) <= 0.05 and abs(ck["exact_match"] - fpm) <= 0.05))
    P.append(_pred("C-H4 rival", "token-wise equal or better on KL", "mean teacher-forced KL", "token-wise <= channel-K",
                   [tk["mean_teacher_forced_kl"], ck["mean_teacher_forced_kl"]],
                   tk["mean_teacher_forced_kl"] <= ck["mean_teacher_forced_kl"], "values listed as [int4_token, int4_channel_k]"))
    bt = {(r["metric"], r["a"], r["b"]): r for r in boot}
    for m in ("int4_token", "int4_channel_k"):
        e, n = bt[("exact_match", "fp16", m)], bt[("reference_answer_nll", "fp16", m)]
        P.append(_pred(f"C-H5 {m}", "The INT4 layouts match FP16 exact match but answer NLL rises with a 95% interval excluding 0",
                       "paired bootstrap (AM2.7) method minus fp16: exact match and reference-answer NLL",
                       "EM interval includes 0; NLL point > 0 and interval excludes 0",
                       {"em_point": e["point"], "em_ci": [e["ci_low"], e["ci_high"]], "nll_point": n["point"],
                        "nll_ci": [n["ci_low"], n["ci_high"]]},
                       e["ci_low"] <= 0 <= e["ci_high"] and n["point"] > 0 and n["ci_low"] > 0,
                       "operationalization of 'match': the EM difference interval includes 0"))
        P.append(_pred(f"C-H5 rival {m}", "interval includes 0", "NLL difference interval", "interval includes 0",
                       [n["ci_low"], n["ci_high"]], n["ci_low"] <= 0 <= n["ci_high"]))
    hy, qa = C1["hybrid"], C1["question_aware"]
    P.append(_pred("C-H6", "Hybrid exact match within 5 points of question-aware eviction; hybrid teacher-forced KL above INT4 channel-K",
                   "exact match difference hybrid minus question_aware; mean teacher-forced KL hybrid vs int4_channel_k",
                   "|EM difference| <= 0.05 and hybrid KL > channel-K KL",
                   {"em_difference": hy["exact_match"] - qa["exact_match"],
                    "teacher_forced_kl": [hy["mean_teacher_forced_kl"], ck["mean_teacher_forced_kl"]]},
                   abs(hy["exact_match"] - qa["exact_match"]) <= 0.05 and hy["mean_teacher_forced_kl"] > ck["mean_teacher_forced_kl"],
                   "KL listed as [hybrid, int4_channel_k]"))
    P.append(_pred("C-H6 rival", "hybrid more than 10 points below eviction", "exact match hybrid minus question_aware", "< -0.10",
                   hy["exact_match"] - qa["exact_match"], hy["exact_match"] - qa["exact_match"] < -0.10))
    for x in bench:
        if x["L"] == 4096 and x["batch"] == 8 and x["config"] != "fp16":
            add = x["decode_step_ratio_to_fp16"] - 1
            P.append(_pred(f"C-H7 {x['config']}", "Per-step unpack-plus-dequantize adds >= 20% to FP16 decode-step time at batch 8 and L 4096",
                           "decode-step median with unpacking / fp16 decode-step median - 1 (C5 bench)", ">= 0.20",
                           {"added_fraction": add, "unpack_dequant_ms": x["unpack_dequant_ms_median"],
                            "decode_step_ms": x["decode_step_ms_median"]}, add >= 0.20))
            P.append(_pred(f"C-H7 rival {x['config']}", "< 5%", "same fraction", "< 0.05", add, add < 0.05))
    # AM4.5
    fp = per["fp16"]["accuracy"]
    P.append(_pred("AM4 (c)", "(c) exact match within 5 points of fp16 and mean teacher-forced KL below 0.05",
                   "kv_int8_token on the 18 diagnostic prompts", "|EM - fp16 EM| <= 0.05 and KL < 0.05",
                   {"em_difference": per["kv_int8_token"]["accuracy"] - fp, "teacher_forced_kl": per["kv_int8_token"]["mean_kl_teacher_forced"]},
                   am4["c_int8"], "post hoc diagnostic (AM4.1)"))
    P.append(_pred("AM4 (a)", "(a) fails like (d): exact match at most 0.1", "k_int4_token exact match", "<= 0.1",
                   per["k_int4_token"]["accuracy"], am4["a_k_only"], "post hoc diagnostic (AM4.1)"))
    P.append(_pred("AM4 (b)", "(b) exact match within 5 points of fp16", "v_int4_token exact match difference to fp16", "|difference| <= 0.05",
                   per["v_int4_token"]["accuracy"] - fp, am4["b_v_only"], "post hoc diagnostic (AM4.1)"))
    P.append(_pred("AM4 K ratio", "K shows channel max/median ratios far above V's (mean K ratio at least 3 times the mean V ratio)",
                   "mean over prompts, layers and KV heads of the largest channel max / median channel max", "K >= 3 x V",
                   {"k_ratio": overall["k_ratio"], "v_ratio": overall["v_ratio"], "three_times_v": 3 * overall["v_ratio"]},
                   am4["k_ratio"], "post hoc diagnostic (AM4.1)"))
    P.append(_pred("AM4 K rest error", "token-wise K error on the non-outlier channels at least 3 times channel-K's",
                   "overall non-outlier mean per-channel relative K error, token-wise vs channel-K", "token-wise >= 3 x channel-K",
                   {"token_wise": overall["k_err_token_rest"], "channel_k": overall["k_err_channel_rest"]},
                   am4["k_rest_error"], "post hoc diagnostic (AM4.1)"))
    P.append(_pred("AM4 rival", "(c) also fails, which would indicate a defect in the token-wise code path",
                   "AM4 (c) comparison", "AM4 (c) does not hold", am4["c_int8"], am4["rival_token_path_defect"],
                   "AM4.6: if this holds, int4_token is treated as a defect"))
    return P


# ------------------------------------------------------------------------------------------------ data sources
def data_sources(D):
    r = D["runs"][(CONFIGS[0], SEEDS[0])]["run"]
    e = D["eval_run"]
    return {"mqar_generator": {k: r["generator"][k] for k in ("sha256", "manifest_sha256", "T")},
            "mqar_development": {k: r["dev_file_opens"][0][k] for k in ("sha256", "manifest_sha256")},
            "mqar_test": {k: r["test_file_opens"][0][k] for k in ("sha256", "manifest_sha256")},
            "kv_retrieval": {k: e["config"]["data"][k] for k in ("rel", "sha256", "sha256_decompressed")},
            "model": {"id": e["model"]["model_id"], "revision": e["model"]["revision"]}}


# ------------------------------------------------------------------------------------------------ building
FIGURE_TABLES = ("A_errors.csv", "A_bench.csv", "B_dev_curves.csv", "B_bins.csv", "B3_sweep.csv", "C_cells.csv",
                 "C4_bench.csv")


def compute(inp: Inputs) -> dict:
    D = load(inp)
    E, ids = by_method(D)
    chk = {"part_a": check_part_a(D), "part_b": check_part_b(D), "part_c": check_part_c(D, E, ids)}
    per, per_layer, overall = check_diag(D)
    am4 = am4_readout(per, overall)
    stored = D["diag_run"]["summary"]["readout"]
    for k, v in am4.items():
        check(stored[k] == v, f"AM4 readout {k}")
    theo = b3_theoretical(D)
    c1 = c1_table(D, E)
    bench = c4_bench(D)
    ctrl, chan = c5_tables(per, per_layer, overall)
    T = {"A_errors.csv": a_errors(D), "A_bench.csv": a_bench(D), "A_backend.csv": a_backend(D),
         "A_backend_compact.csv": a_backend_compact(D), "A_unsupported.csv": a_unsupported(D),
         "A_storage.csv": a_storage(D),
         "B_dev_curves.csv": b_dev_curves(D), "B_test_cells.csv": b_test_cells(D), "B1_accuracy.csv": b1_accuracy(D),
         "B_bins.csv": b_bins(D), "B2_mask_controls.csv": b2_mask_controls(D), "B2_far_residue.csv": b2_far_residue(D),
         "B3_theoretical.csv": theo, "B3_measured_training.csv": b3_measured_training(D), "B3_sweep.csv": b3_sweep(D),
         "B3_items.csv": b3_items(D, theo),
         "C1_matched_budget.csv": c1, "C_rows.csv": c_rows(E), "C_cells.csv": c_cells(E),
         "C3_systems.csv": c3_systems(D, E), "C4_bench.csv": bench, "C5_am4_controls.csv": ctrl,
         "C5_am4_channels.csv": chan}
    return {"D": D, "E": E, "T": T, "checks": chk, "per": per, "overall": overall, "am4": am4}


def build(root, out, require_committed=True, figures=True):
    inp = Inputs(root)
    R = compute(inp)
    committed = committed_check(list(inp.read)) if require_committed else None
    D, E, T = R["D"], R["E"], R["T"]
    T["C2_bootstrap.csv"] = c2_bootstrap(E)
    T["P_predictions.csv"] = predictions(D, E, T["C1_matched_budget.csv"], T["C2_bootstrap.csv"], T["C4_bench.csv"],
                                         R["per"], R["overall"], R["am4"])
    qrecs, qtext = qualitative(D, E)
    out.mkdir(parents=True)
    for name, rows in T.items():
        write_csv(out / name, rows)
    with open(out / "C_qualitative.jsonl", "x", encoding="utf-8") as fh:
        for x in qrecs:
            fh.write(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n")
    with open(out / "C_qualitative.txt", "x", encoding="utf-8") as fh:
        fh.write(qtext)
    fig_files, info = [], {}
    if figures:
        from task2 import analysis_figures as F
        fig_files = F.make_all(T, out, info)
    summary = {
        "analysis_id": ANALYSIS_ID, "round_rule": ROUND_RULE,
        "bootstrap": {"method": B.METHOD, "resamples": BOOT_RESAMPLES, "seed": BOOT_SEED, "percentiles": list(B.PERCENTILES),
                      "unit": "prompt", "pairs": [list(p) for p in C_PAIRS], "rule": "protocol AM2.7"},
        "seed_rule": "protocol B8: a difference between conditions counts only when the seed min-max ranges do not overlap",
        "checks": R["checks"], "am4_readout": R["am4"], "data_sources": data_sources(D),
        "qualitative_rule": QUAL_RULE, "b3_selection_rule": B3_RULE,
        "tables": {k: len(v) for k, v in T.items()},
        "sanity_assertions": [
            "test accuracy, query counts, residue and pooled-bin accuracies recomputed from test_queries.npz equal run.json",
            "final development accuracy and the B6 learned flag recomputed from dev_queries_final.npz equal run.json",
            "mean training ms per update recomputed from train_log.jsonl equals run.json",
            "Part A row and status counts equal run.json; bench medians equal the median of the stored times",
            "Part C per-method summaries recomputed from eval rows equal run.json; extractor and correctness recomputed",
            "evidence retained recomputed from the per-prompt npz positions equals the eval rows",
            "context bytes equal the AM2.3 formula bytes and never exceed M4",
            "AM4 per-configuration summaries, channel statistics and readout recomputed from control rows and npz equal run.json",
            "bootstrap point estimates equal the direct differences"],
        "inputs": {str(Path(k).relative_to(Path(root))): v for k, v in sorted(inp.read.items())},
        "inputs_committed": committed, "figures": fig_files, "figure_info": info,
    }
    write_json_new(out / "summary.json", summary)
    return summary, inp


def _cell(v):
    return "" if v is None else (json.dumps(v) if isinstance(v, (list, dict)) else str(v))


def figures_only(results_root, analysis_dir, out, require_committed=True):
    """Figures recomputed from the committed results, checked against the analysis CSV files."""
    from task2 import analysis_figures as F
    inp = Inputs(results_root)
    R = compute(inp)
    analysis_dir = Path(analysis_dir)
    for name in FIGURE_TABLES:
        with open(analysis_dir / name, newline="", encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
        ref = R["T"][name]
        check(len(got) == len(ref), f"{name}: row count differs from the recomputed table")
        for g, r in zip(got, ref):
            check(all(g[k] == _cell(r.get(k)) for k in g), f"{name}: values differ from the recomputed table")
    committed = None
    if require_committed:
        committed = committed_check([*inp.read, *(analysis_dir / n for n in FIGURE_TABLES)])
    out.mkdir(parents=True)
    info = {}
    names = F.make_all(R["T"], out, info)
    sizes = {n: [round(v, 3) for v in F.pdf_size_in(out / n)] for n in names}
    return names, sizes, info, committed


def _staging(out):
    if out.exists():
        sys.exit(f"refusing existing output directory: {out}")
    staging = out.with_name(out.name + ".building")
    if staging.exists():
        sys.exit(f"refusing existing staging directory: {staging}")
    return staging


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="new output directory (refused if it exists)")
    ap.add_argument("--results", default=str(REPO_ROOT / "results" / "task2"), help="results root")
    ap.add_argument("--allow-uncommitted", action="store_true", help="skip the committed-inputs check (tests)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="only rebuild the figures")
    ap.add_argument("--analysis", default=str(REPO_ROOT / "results" / "task2" / "analysis"),
                    help="analysis outputs whose figure tables --figures-only checks")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    out = Path(a.out)
    staging = _staging(out)
    if a.figures_only:
        names, sizes, info, committed = figures_only(a.results, a.analysis, staging,
                                                     require_committed=not a.allow_uncommitted)
        rec = RunRecorder(staging / "run.json", {"entry_point": "task2.analysis --figures-only", "args": vars(a),
                                                  "analysis_id": ANALYSIS_ID})
        rec.finish("ok", inputs_committed=committed, figures=names, size_inches=sizes)
        staging.rename(out)
        for n in names:
            print(f"{n}: {sizes[n][0]:.2f} x {sizes[n][1]:.2f} in")
        print(f"wrote {out}")
        return out
    summary, inp = build(a.results, staging, require_committed=not a.allow_uncommitted, figures=not a.no_figures)
    rec = RunRecorder(staging / "run.json", {"entry_point": "task2.analysis", "args": vars(a), "analysis_id": ANALYSIS_ID})
    rec.finish("ok", inputs_committed=summary["inputs_committed"], n_inputs=len(inp.read),
               outputs=sorted(p.name for p in staging.iterdir() if p.name != "run.json"))
    staging.rename(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
