"""Task 1 analysis: tables, paired bootstrap intervals, diagnostics, figures and qualitative HellaSwag cases.

Reads committed result files under --results (default results/task1) and writes a new directory --out.

    python -m task1.analysis --out results/task1/analysis
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import bootstrap as B
from common.io import read_jsonl, sha256_file, write_json_new
from common.paths import REPO_ROOT
from common.runmeta import RunRecorder
from task1 import config as C

ANALYSIS_ID = "t1-analysis-v1"
SUITES = ("wikitext2_test", "c4_validation")
SUITE_LABEL = {"wikitext2_test": "WikiText-2 test", "c4_validation": "C4 validation"}
CONDITIONS = list(C.CONDITIONS)
FULL_CONDITIONS = [c for c in CONDITIONS if C.CONDITIONS[c]["hellaswag"] == C.HELLASWAG_FULL]
FAMILY_LABEL = {
    "baseline": "FP16", "P1": "Magnitude", "P2": "Wanda", "P3": "Wanda + OWL-style allocation",
    "P2_2of4": "Wanda 2:4", "Q1": "RTN", "Q2": "AWQ-style", "Q3": "Mixed precision",
}
# Matched comparisons (A = reference, B = adaptive); bootstrap reports B - A.
PAIRS = [("p2_s50", "p3_s50"), ("p2_s75", "p3_s75"), ("q2_b4", "q3_int4"), ("q2_b3", "q3_int3")]
# Selection directories of the allocations evaluated (Q3: fold-group design v2) and the block-level Q3 design.
SELECT_DIRS = {"p3_s50": "select_p3_s50", "p3_s75": "select_p3_s75",
               "q3_int4": "select_q3v2_int4", "q3_int3": "select_q3v2_int3"}
BLOCK_Q3_SELECT_DIRS = {"q3_int4": "select_q3_int4", "q3_int3": "select_q3_int3"}
PROJ_TYPES = list(C.ELIGIBLE)
Q3V2_UNIT_TYPES = [u for u, _, _ in C.Q3V2_UNITS]
QUAL_CONDITIONS = FULL_CONDITIONS
REL_TOL = 1e-12


class AnalysisError(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise AnalysisError(msg)


def close(a, b, tol=REL_TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


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

    def exists(self, rel):
        return (self.root / rel).exists()


def committed_check(paths) -> dict:
    """Every path is tracked by git and identical to HEAD."""
    rel = [str(Path(p).resolve().relative_to(REPO_ROOT)) for p in paths]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", *rel], cwd=REPO_ROOT, capture_output=True)
    check(tracked.returncode == 0, f"untracked inputs: {tracked.stderr.decode()[:400]}")
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *rel], cwd=REPO_ROOT)
    check(diff.returncode == 0, "inputs differ from HEAD")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"n_files": len(rel), "tracked": True, "identical_to_head": True, "head": head}


def run1_dir(spec_dir: str) -> str:
    """results/task1/run1b/q3_int3 -> run1b/q3_int3 (relative to the results root)."""
    parts = Path(spec_dir).parts
    return str(Path(*parts[-2:]))


def load(inp: Inputs, expect: dict) -> dict:
    D = {"cond": {}}
    for c in CONDITIONS:
        r2 = inp.json(f"run2/{c}/run.json")
        res = inp.json(f"run2_resources/{c}/run.json")
        check(r2["status"] == "ok" and res["status"] == "ok", f"status {c}")
        spec = r2["config"]["condition_spec"]
        check(spec["family"] == C.CONDITIONS[c]["family"], f"family {c}")
        src = "run1/fp16" if c == "fp16" else run1_dir(r2["steps"]["inputs"]["run1"]["dir"])
        r1 = inp.json(f"{src}/run.json")
        d = {"spec": spec, "run2": r2, "res": res, "run1": r1, "run1_dir": src, "ppl": {}}
        for s in SUITES:
            d["ppl"][s] = inp.jsonl(f"run2/{c}/ppl_{s}.jsonl.gz")
        hs = r2["steps"]["hellaswag"]
        d["hs"] = inp.jsonl(f"run2/{c}/{hs['file']}")
        d["dev"] = inp.jsonl(f"run2/{c}/calib_nll_c4_development.jsonl.gz")
        d["calib"] = inp.jsonl(f"{src}/calib_nll_c4_calibration.jsonl.gz")
        if c != "fp16":
            d["proj"] = inp.jsonl(f"{src}/projections.jsonl.gz")
        D["cond"][c] = d
    D["probe_prune"] = inp.json("probe_prune/sensitivity.json")
    D["probe_quant"] = inp.json("probe_quant/sensitivity.json")
    parts = sorted(p.parent.name for p in (inp.root / "probe_quant_v2").glob("*/sensitivity.json"))
    check(parts, "no probe_quant_v2 parts")
    D["probe_quant_v2"] = [inp.json(f"probe_quant_v2/{p}/sensitivity.json") for p in parts]
    D["probe_quant_v2_parts"] = parts
    D["calib_stats"] = inp.json("calib_stats/run.json")
    D["select"] = {}
    for cond, sd in {**SELECT_DIRS, **{f"{k}_block": v for k, v in BLOCK_Q3_SELECT_DIRS.items()}}.items():
        D["select"][cond] = {"dir": sd, "selected": inp.json(f"{sd}/selected_allocation.json"),
                             "candidates": inp.jsonl(f"{sd}/candidates.jsonl.gz"), "run": inp.json(f"{sd}/run.json"),
                             "sha256": sha256_file(inp.root / sd / "selected_allocation.json")}
    return D


# ------------------------------------------------------------------------------------------------ helpers
def pooled(rows):
    n = sum(r["n_scored"] for r in rows)
    s = math.fsum(r["nll_sum"] for r in rows)
    return n, s, s / n


def window_key(rows):
    return [(r["window_index"], r["source"], r["source_row"], r["token_offset"], r["n_scored"]) for r in rows]


def hs_ids(rows):
    return [(r["order"], r["source_row_index"]) for r in rows]


def budget_of(c):
    s = C.CONDITIONS[c]
    if "sparsity" in s:
        return s["sparsity"], "target sparsity"
    if "bits" in s:
        return s["bits"], "bits"
    if "budget" in s:
        return s["budget"], "packed-byte budget of"
    return None, "none"


def short_label(c):
    s = C.CONDITIONS[c]
    fam = s["family"]
    if fam == "baseline":
        return "FP16"
    if "sparsity" in s:
        tag = {"P1": "P1", "P2": "P2", "P3": "P3", "P2_2of4": "2:4"}[fam]
        return f"{tag} {int(round(s['sparsity'] * 100))}%"
    if "bits" in s:
        return f"{fam} {s['bits']}b"
    return f"Q3 {c.split('_')[1].upper()}"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p_value": float(r.pvalue), "n": int(x.size)}


def ranks_desc(v):
    """Rank 1 = largest value; ties share the lower rank (competition ranking)."""
    v = np.asarray(v, float)
    order = sorted(range(v.size), key=lambda i: -v[i])
    out, prev, rank = [0] * v.size, None, 0
    for pos, i in enumerate(order):
        if prev is None or v[i] != prev:
            rank = pos + 1
        out[i], prev = rank, v[i]
    return out


# ------------------------------------------------------------------------------------------------ tables
def quality(D, expect):
    fp = D["cond"]["fp16"]
    ref_w = {s: window_key(fp["ppl"][s]) for s in SUITES}
    fp_hs = {r["order"]: r for r in fp["hs"]}
    full_ids = hs_ids(fp["hs"])
    check(len(full_ids) == expect["hellaswag_full"], f"fp16 HellaSwag n {len(full_ids)}")
    fp_pool = {s: pooled(fp["ppl"][s]) for s in SUITES}
    rows = []
    for c in CONDITIONS:
        d = D["cond"][c]
        b, bu = budget_of(c)
        row = {"condition": c, "label": short_label(c), "family": d["spec"]["family"],
               "method": FAMILY_LABEL[d["spec"]["family"]], "budget": b, "budget_unit": bu}
        for s in SUITES:
            rows_s = d["ppl"][s]
            check(window_key(rows_s) == ref_w[s], f"window list differs: {c} {s}")
            N, tot, m = pooled(rows_s)
            st = d["run2"]["steps"]["perplexity"][s]
            check(N == st["N"] == expect["n_tokens"], f"N {c} {s}: {N} {st['N']}")
            check(close(m, st["mean_nll"]) and close(math.exp(m), st["ppl"]) and close(tot, st["nll_sum"]),
                  f"recomputed perplexity differs from stored summary: {c} {s}")
            row[f"{s}_N"] = N
            row[f"{s}_n_windows"] = len(rows_s)
            row[f"{s}_mean_nll"] = m
            row[f"{s}_ppl"] = math.exp(m)
            row[f"{s}_delta_mean_nll"] = m - fp_pool[s][2]
            row[f"{s}_delta_ppl"] = math.exp(m) - math.exp(fp_pool[s][2])
        hs = d["run2"]["steps"]["hellaswag"]
        ids = hs_ids(d["hs"])
        check(ids == full_ids[:len(ids)], f"HellaSwag IDs are not the manifest prefix: {c}")
        want = expect["hellaswag_full"] if hs["scope"] == C.HELLASWAG_FULL else expect["hellaswag_subset"]
        check(hs["scope"] == C.CONDITIONS[c]["hellaswag"] and len(ids) == want == hs["n"], f"HellaSwag scope {c}")
        n = len(ids)
        acc = sum(r["pred_acc"] == r["label"] for r in d["hs"]) / n
        accn = sum(r["pred_acc_norm"] == r["label"] for r in d["hs"]) / n
        for r in d["hs"]:
            check(r["correct_acc"] == (r["pred_acc"] == r["label"]) and
                  r["correct_acc_norm"] == (r["pred_acc_norm"] == r["label"]), f"correct flags {c} {r['order']}")
            check(r["pred_acc"] == int(np.argmax(r["loglik"])) and
                  r["pred_acc_norm"] == int(np.argmax(np.asarray(r["loglik"]) / np.asarray(r["cont_chars"]))),
                  f"prediction differs from argmax {c} {r['order']}")
        check(acc == hs["acc"] and accn == hs["acc_norm"], f"recomputed HellaSwag accuracy differs: {c}")
        fa = sum(fp_hs[o]["correct_acc"] for o, _ in ids) / n
        fn = sum(fp_hs[o]["correct_acc_norm"] for o, _ in ids) / n
        row.update({"hellaswag_scope": hs["scope"], "hellaswag_n": n, "acc": acc, "acc_norm": accn,
                    "fp16_acc_same_ids": fa, "fp16_acc_norm_same_ids": fn,
                    "delta_acc_pp": 100 * (acc - fa), "delta_acc_norm_pp": 100 * (accn - fn)})
        # development and calibration NLL (recomputed from per-sequence files)
        Nd, _, md = pooled(d["dev"])
        stored = (d["run2"]["steps"]["calib_nll"]["c4_development"]["mean_nll"] if c == "fp16"
                  else d["run2"]["steps"]["reproduction"]["mean_nll"])
        check(close(md, stored), f"development NLL differs: {c}")
        r1_dev = d["run1"]["steps"]["calib_nll"]["c4_development"]["mean_nll"]
        check(abs(md - r1_dev) <= 1e-4, f"development NLL not reproduced: {c}")
        Nc, _, mc = pooled(d["calib"])
        check(close(mc, d["run1"]["steps"]["calib_nll"]["c4_calibration"]["mean_nll"]), f"calibration NLL {c}")
        row.update({"development_N": Nd, "development_mean_nll": md, "development_mean_nll_run1": r1_dev,
                    "calibration_N": Nc, "calibration_mean_nll": mc,
                    "pooled_layer_output_error": (0.0 if c == "fp16" else
                                                  d["run1"]["steps"]["build"]["output_error"]["pooled_rel"])})
        rows.append(row)
    return rows


def _meas(res, key):
    return res["steps"]["resources"]["measurements"][key]


def resources(D, qrows):
    fp_res = D["cond"]["fp16"]["res"]
    fp_st = fp_res["steps"]["storage"]
    fp_whole = fp_st["fp16_dense"]["whole_model_bytes"]
    fp_lat = {k: _meas(fp_res, k)["median_ms"] for k in ("prefill_512", "prefill_2048", "decode_64_after_prefill_512")}
    fp_peak = max(_meas(fp_res, k)["peak_allocated_bytes_max"] for k in fp_lat)
    awq_wall = D["calib_stats"]["elapsed_wall_s"]
    rows = []
    for c in CONDITIONS:
        d = D["cond"][c]
        st = d["res"]["steps"]["storage"]
        elig = st["eligible_weights"]
        nz = st["sparse16"]["nonzeros"] if c == "fp16" else st["nonzeros"]["global"]
        logical = st.get("logical") or {"format": "fp16_dense", "eligible_bytes": st["fp16_dense"]["eligible_bytes"],
                                         "whole_model_bytes": st["fp16_dense"]["whole_model_bytes"],
                                         "bits_per_eligible_weight": 16.0}
        nominal = st.get("nominal_bits_per_eligible_weight", 16)
        art = st.get("artifact_files") or {}
        t1 = d["run1"].get("timings_s", {})
        fam = d["spec"]["family"]
        row = {"condition": c, "label": short_label(c), "family": fam,
               "eligible_weights": elig, "effective_nonzeros": nz, "zeros": elig - nz,
               "achieved_sparsity": (elig - nz) / elig, "nominal_bits_per_eligible_weight": nominal,
               "logical_format": logical["format"],
               "logical_bits_per_eligible_weight": logical["bits_per_eligible_weight"],
               "logical_eligible_bytes": logical["eligible_bytes"],
               "logical_whole_model_bytes": logical["whole_model_bytes"],
               "sparse16_eligible_bytes": (st.get("sparse16") or {}).get("eligible_bytes"),
               "sparse16_whole_model_bytes": (st.get("sparse16") or {}).get("whole_model_bytes"),
               "fp16_dense_eligible_bytes": st["fp16_dense"]["eligible_bytes"],
               "fp16_dense_whole_model_bytes": st["fp16_dense"]["whole_model_bytes"],
               "artifact_safetensors_bytes": art.get("safetensors_bytes"),
               "artifact_total_bytes": art.get("total_bytes"),
               "build_compression_s": t1.get("build_compression_s"),
               "build_calibration_stats_s": t1.get("build_calibration_stats_s"),
               "build_total_s": t1.get("build_total_s"),
               "run1_elapsed_wall_s": d["run1"]["elapsed_wall_s"],
               "shared_awq_scale_search_wall_s": awq_wall if fam in ("Q2", "Q3") else None,
               "execution_mode": d["res"]["steps"]["resources"]["execution"]["mode"],
               "device": d["res"]["steps"]["resources"]["device_name"],
               "run_peak_allocated_bytes": max(d["res"]["peak_vram_allocated_bytes"].values())}
        peak = 0
        for k, name in (("prefill_512", "prefill_512"), ("prefill_2048", "prefill_2048"),
                        ("decode_64_after_prefill_512", "decode_64")):
            m = _meas(d["res"], k)
            check(m["repetitions"] >= 5 and len(m["times_ms"]) == m["repetitions"], f"repetitions {c} {k}")
            check(close(float(np.median(m["times_ms"])), m["median_ms"], 1e-9), f"median {c} {k}")
            row[f"{name}_median_ms"] = m["median_ms"]
            row[f"{name}_repetitions"] = m["repetitions"]
            row[f"{name}_warmup"] = m["warmup"]
            row[f"{name}_change_vs_fp16"] = m["median_ms"] / fp_lat[k] - 1
            peak = max(peak, m["peak_allocated_bytes_max"])
        row["latency_peak_allocated_bytes"] = peak
        row["latency_peak_allocated_change_vs_fp16"] = peak / fp_peak - 1
        row["theoretical_storage_saving_vs_fp16"] = 1 - logical["whole_model_bytes"] / fp_whole
        rows.append(row)
    return rows


def run_bootstrap(D):
    rows = []
    for a, b in PAIRS:
        da, db = D["cond"][a], D["cond"][b]
        for s in SUITES:
            ra, rb = da["ppl"][s], db["ppl"][s]
            check(window_key(ra) == window_key(rb), "windows")
            r = B.paired_ppl_diff([x["nll_sum"] for x in ra], [x["n_scored"] for x in ra],
                                  [x["nll_sum"] for x in rb], [x["n_scored"] for x in rb], seed=B.DEFAULT_SEED,
                                  n_resamples=B.DEFAULT_RESAMPLES)
            pa, pb = pooled(ra)[2], pooled(rb)[2]
            check(close(r["point"], math.exp(pb) - math.exp(pa), 1e-9) and close(r["nll_point"], pb - pa, 1e-9),
                  f"bootstrap point differs from direct difference {a} {b} {s}")
            base = {"a": a, "b": b, "comparison": f"{b} - {a}", "suite": s, "resampled_unit": "window",
                    "n": r["n_units"], "n_resamples": r["n_resamples"], "seed": r["seed"]}
            rows.append({**base, "metric": "delta_ppl", "unit": "perplexity", "point": r["point"],
                         "ci_low": r["ci"][0], "ci_high": r["ci"][1]})
            rows.append({**base, "metric": "delta_mean_nll", "unit": "nats per token", "point": r["nll_point"],
                         "ci_low": r["nll_ci"][0], "ci_high": r["nll_ci"][1]})
        ha, hb = {x["order"]: x for x in da["hs"]}, {x["order"]: x for x in db["hs"]}
        ids = [o for o, _ in hs_ids(db["hs"])]
        check(ids == [o for o, _ in hs_ids(da["hs"])], f"HellaSwag IDs differ {a} {b}")
        for metric in ("acc", "acc_norm"):
            ca = [ha[o][f"correct_{metric}"] for o in ids]
            cb = [hb[o][f"correct_{metric}"] for o in ids]
            r = B.paired_acc_diff(ca, cb, seed=B.DEFAULT_SEED, n_resamples=B.DEFAULT_RESAMPLES)
            check(close(r["point"], np.mean(cb) - np.mean(ca), 1e-12), f"bootstrap point {a} {b} {metric}")
            rows.append({"a": a, "b": b, "comparison": f"{b} - {a}", "suite": f"hellaswag_{da['run2']['steps']['hellaswag']['scope']}",
                         "resampled_unit": "example", "n": r["n_units"], "n_resamples": r["n_resamples"],
                         "seed": r["seed"], "metric": f"delta_{metric}_pp", "unit": "percentage points",
                         "point": 100 * r["point"], "ci_low": 100 * r["ci"][0], "ci_high": 100 * r["ci"][1]})
    return rows


def matched(D, rrows):
    R = {r["condition"]: r for r in rrows}
    out = []
    for c in CONDITIONS:
        fam = C.CONDITIONS[c]["family"]
        if fam in ("P1", "P2", "P3", "P2_2of4"):
            b = D["cond"][c]["run1"]["steps"]["build"]
            t, mz = b["target_zeros_total"], b["mask_zeros_total"]
            out.append({"check": "achieved zeros equal target", "condition": c, "reference": None,
                        "value": R[c]["zeros"], "reference_value": t, "mask_zeros": mz,
                        "holds": R[c]["zeros"] == t == mz})
    for a, b in (("p2_s50", "p3_s50"), ("p2_s75", "p3_s75"), ("p2_s50", "p2_24_s50")):
        out.append({"check": "zeros equal reference zeros", "condition": b, "reference": a,
                    "value": R[b]["zeros"], "reference_value": R[a]["zeros"], "mask_zeros": None,
                    "holds": R[b]["zeros"] == R[a]["zeros"]})
    for a, b in (("q2_b4", "q3_int4"), ("q2_b3", "q3_int3")):
        for k in ("logical_eligible_bytes", "logical_whole_model_bytes"):
            out.append({"check": f"{k} within reference budget", "condition": b, "reference": a,
                        "value": R[b][k], "reference_value": R[a][k], "mask_zeros": None,
                        "holds": R[b][k] <= R[a][k]})
    for row in out:
        check(row["holds"], f"matched-budget check failed: {row}")
    return out


def effort(D):
    rows = []
    for name, probe in (("probe_prune", D["probe_prune"]), ("probe_quant", D["probe_quant"])):
        for m, e in probe["effort"].items():
            rows.append({"stage": name, "part": "all", "measure": m, **_effort_fields(e)})
    for part, probe in zip(D["probe_quant_v2_parts"], D["probe_quant_v2"]):
        for m, e in probe["effort"].items():
            rows.append({"stage": "probe_quant_v2", "part": part, "measure": m, **_effort_fields(e)})
    for key, s in D["select"].items():
        rows.append({"stage": s["dir"], "part": "all", "measure": "selection (candidate builds and NLL)",
                     "wall_s": s["run"]["elapsed_wall_s"], "full_model_calibration_passes": None,
                     "block_forward_passes": None, "restores": None,
                     "calibration_sequences_per_pass": None, "n_candidates": len(s["candidates"])})
    rows.append({"stage": "calib_stats", "part": "all", "measure": "AWQ scale search and dense statistics",
                 "wall_s": D["calib_stats"]["elapsed_wall_s"], "full_model_calibration_passes": None,
                 "block_forward_passes": None, "restores": None, "calibration_sequences_per_pass": None,
                 "n_candidates": None})
    cand = []
    for key, s in D["select"].items():
        design = "block (unsuccessful design, not evaluated)" if key.endswith("_block") else s["selected"].get("unit")
        for x in s["candidates"]:
            cand.append({"target": s["selected"]["condition"], "selection_dir": s["dir"], "design": design,
                         "candidate_id": x["candidate_id"], "measure": x["measure"], "lambda": x["lambda"],
                         "reference": x["reference"], "collapsed_to_uniform": x["collapsed_to_uniform"],
                         "selected": x["selected"], "development_mean_nll": x["development"]["mean_nll"],
                         "calibration_mean_nll": x["calibration"]["mean_nll"],
                         "output_error_pooled_rel": x.get("output_error_pooled_rel"),
                         "build_s": x.get("timings_s", {}).get("build"), "nll_s": x.get("timings_s", {}).get("nll")})
    return rows, cand


def _effort_fields(e):
    return {"wall_s": e.get("wall_s"), "full_model_calibration_passes": e.get("full_model_calibration_passes"),
            "block_forward_passes": e.get("block_forward_passes"), "restores": e.get("restores"),
            "calibration_sequences_per_pass": e.get("calibration_sequences_per_pass"), "n_candidates": None}


def transfer(D, qrows):
    Q = {r["condition"]: r for r in qrows}
    out = []
    for a, b in PAIRS:
        row = {"a": a, "b": b, "comparison": f"{b} - {a}",
               "development_delta_mean_nll": Q[b]["development_mean_nll"] - Q[a]["development_mean_nll"],
               "calibration_delta_mean_nll": Q[b]["calibration_mean_nll"] - Q[a]["calibration_mean_nll"]}
        for s in SUITES:
            row[f"{s}_delta_mean_nll"] = Q[b][f"{s}_mean_nll"] - Q[a][f"{s}_mean_nll"]
        for s in ("calibration", *SUITES):
            row[f"sign_agrees_development_{s}"] = bool(np.sign(row["development_delta_mean_nll"]) ==
                                                       np.sign(row[f"{s}_delta_mean_nll"]))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------ diagnostics
def proj_records(D, c):
    recs = sorted(D["cond"][c]["proj"], key=lambda r: (r["layer"], PROJ_TYPES.index(r["name"])))
    return recs


def block_sensitivities(D):
    pp, pq = D["probe_prune"]["values"], D["probe_quant"]["values"]
    nb = D["probe_prune"]["n_blocks"]
    cols = {"S1": pp["S1"]}
    for k, v in pp["S2"].items():
        cols[f"S2_s{k}"] = v
    for b, v in pq["QS1"].items():
        cols[f"QS1_block_b{b}"] = v
    v2 = q3v2_unit_values(D, "QS1v2")
    for b, m in v2.items():
        cols[f"QS1v2_block_sum_b{b}"] = list(m.sum(axis=0))
    for k, v in cols.items():
        check(len(v) == nb, f"block count {k}")
    rows = []
    rk = {k: ranks_desc(v) for k, v in cols.items()}
    for i in range(nb):
        row = {"block": i}
        for k in cols:
            row[k] = float(cols[k][i])
            row[f"{k}_rank"] = rk[k][i]
        rows.append(row)
    names = list(cols)
    corr = [{"a": a, "b": b, **spearman(cols[a], cols[b])} for i, a in enumerate(names) for b in names[i + 1:]]
    return rows, corr


def q3v2_unit_values(D, measure):
    """{bits: array [unit type, block]} from the probe parts."""
    nb = D["probe_quant_v2"][0]["n_blocks"]
    out = {}
    for probe in D["probe_quant_v2"]:
        for b, vals in probe["values"][measure].items():
            m = out.setdefault(b, np.full((len(Q3V2_UNIT_TYPES), nb), np.nan))
            for u, v in zip(probe["units"], vals):
                m[Q3V2_UNIT_TYPES.index(u["unit"]), u["layer"]] = v
    for b, m in out.items():
        check(not np.isnan(m).any(), f"missing QS1v2 values at {b} bits")
    return out


def diagnostics(D, qrows):
    Q = {r["condition"]: r for r in qrows}
    out = {}
    out["p1_minus_p2"] = [{"sparsity": s, **{f"{x}_{k}": Q[f"p1_s{int(s * 100)}"][f"{x}_{k}"] -
                                              Q[f"p2_s{int(s * 100)}"][f"{x}_{k}"]
                                              for x in SUITES for k in ("ppl", "mean_nll")}}
                          for s in C.SPARSITIES]
    out["q2_minus_q1"] = [{"bits": b, **{f"{x}_{k}": Q[f"q2_b{b}"][f"{x}_{k}"] - Q[f"q1_b{b}"][f"{x}_{k}"]
                                          for x in SUITES for k in ("ppl", "mean_nll")}} for b in C.BITS]
    out["two_four_minus_unstructured"] = {f"{x}_{k}": Q["p2_24_s50"][f"{x}_{k}"] - Q["p2_s50"][f"{x}_{k}"]
                                          for x in SUITES for k in ("ppl", "mean_nll")}
    dis = []
    for s in C.SPARSITIES:
        recs = proj_records(D, f"p2_s{int(s * 100)}")
        dis.append({"sparsity": s, "statistic": "mask disagreement rate vs coefficient of variation of input-channel "
                                                "norms (dense calibration statistics)",
                    **spearman([r["disagreement_with_magnitude"] for r in recs], [r["x_norm_cv_dense"] for r in recs])})
    out["spearman_disagreement_vs_input_norm_cv"] = dis
    hs = []
    for s in SUITES:
        hs.append({"suite": s, "conditions": FULL_CONDITIONS, "statistic": "delta mean NLL vs delta acc_norm (from FP16)",
                   **spearman([Q[c][f"{s}_delta_mean_nll"] for c in FULL_CONDITIONS],
                              [Q[c]["delta_acc_norm_pp"] for c in FULL_CONDITIONS])})
    out["spearman_delta_nll_vs_delta_acc_norm_full_hellaswag"] = hs
    cal = []
    for s in SUITES:
        cal.append({"suite": s, "x": "calibration mean NLL", **spearman([Q[c]["calibration_mean_nll"] for c in CONDITIONS],
                                                                       [Q[c][f"{s}_mean_nll"] for c in CONDITIONS])})
        cal.append({"suite": s, "x": "pooled layer-output error",
                    **spearman([Q[c]["pooled_layer_output_error"] for c in CONDITIONS],
                               [Q[c][f"{s}_mean_nll"] for c in CONDITIONS])})
    out["spearman_across_18_conditions"] = cal
    return out


def resolution(expect):
    return {"hellaswag_full_one_item_pp": 100 / expect["hellaswag_full"], "hellaswag_full_n": expect["hellaswag_full"],
            "hellaswag_subset_one_item_pp": 100 / expect["hellaswag_subset"],
            "hellaswag_subset_n": expect["hellaswag_subset"]}


def construction(D, qrows):
    """Allocation files evaluated equal the committed selections; fp16 lm_eval acceptance passed."""
    fp = D["cond"]["fp16"]["run2"]
    check(fp["acceptance"] == "PASS" and fp["steps"]["lmeval_verify"]["acceptance"] == "PASS", "fp16 acceptance")
    rows = {"fp16_lmeval_acceptance": fp["steps"]["lmeval_verify"]["acceptance"],
            "fp16_lmeval_acceptance_run": fp["steps"]["lmeval_verify"]["acceptance_run"], "frozen": {}}
    for c, sd in SELECT_DIRS.items():
        f = D["cond"][c]["run2"]["steps"]["inputs"]["freeze"]
        s = D["select"][c]
        ok = f["applies"] and f["clean"] and f["sha256"] == s["sha256"] and f["candidate_id"] == s["selected"]["candidate_id"]
        check(ok, f"evaluated allocation differs from the committed selection: {c}")
        rows["frozen"][c] = {"candidate_id": f["candidate_id"], "sha256": f["sha256"], "freeze_commit": f["freeze_commit"]}
    return rows


# ------------------------------------------------------------------------------------------------ qualitative
QUAL_RULES = [
    ("fp16_correct_p2_s50_wrong", 2, "first 2 IDs where fp16 is correct and p2_s50 is wrong"),
    ("fp16_correct_q2_b3_wrong_q3_int3_correct", 2, "first 2 IDs where fp16 is correct, q2_b3 is wrong and q3_int3 is correct"),
    ("all_10_correct", 2, "first 2 IDs where all 10 full-set conditions are correct"),
    ("fp16_wrong_any_compressed_correct", 1, "first ID where fp16 is wrong and any compressed condition is correct"),
]
QUAL_RULE_TEXT = ("Fixed before run 2: over the HellaSwag IDs in manifest order and the 10 full-set conditions, "
                  "using acc_norm: (1) first 2 IDs where fp16 is correct and p2_s50 is wrong; (2) first 2 IDs where "
                  "fp16 is correct, q2_b3 is wrong and q3_int3 is correct; (3) first 2 IDs where all 10 conditions "
                  "are correct; (4) first ID where fp16 is wrong and any compressed condition is correct.")


def qualitative(D, hs_records):
    by = {c: {r["order"]: r for r in D["cond"][c]["hs"]} for c in QUAL_CONDITIONS}
    orders = [o for o, _ in hs_ids(D["cond"]["fp16"]["hs"])]
    text = {r["order"]: r for r in hs_records}
    ok = {c: {o: by[c][o]["correct_acc_norm"] for o in orders} for c in QUAL_CONDITIONS}
    comp = [c for c in QUAL_CONDITIONS if c != "fp16"]
    preds = {
        "fp16_correct_p2_s50_wrong": lambda o: ok["fp16"][o] and not ok["p2_s50"][o],
        "fp16_correct_q2_b3_wrong_q3_int3_correct": lambda o: ok["fp16"][o] and not ok["q2_b3"][o] and ok["q3_int3"][o],
        "all_10_correct": lambda o: all(ok[c][o] for c in QUAL_CONDITIONS),
        "fp16_wrong_any_compressed_correct": lambda o: (not ok["fp16"][o]) and any(ok[c][o] for c in comp),
    }
    out = []
    for rule, k, desc in QUAL_RULES:
        hits = [o for o in orders if preds[rule](o)][:k]
        if not hits:
            out.append({"rule": rule, "rule_text": desc, "found": False, "message": "no ID satisfies this rule"})
            continue
        for rank, o in enumerate(hits):
            t = text[o]
            check(t["source_row_index"] == by["fp16"][o]["source_row_index"], f"HellaSwag text row {o}")
            conds = {}
            for c in QUAL_CONDITIONS:
                r = by[c][o]
                check(list(r["cont_chars"]) == [len(x) for x in t["choices"]], f"character counts {c} {o}")
                conds[c] = {"label": short_label(c), "loglik": r["loglik"],
                            "per_char": [ll / n for ll, n in zip(r["loglik"], r["cont_chars"])],
                            "cont_chars": r["cont_chars"], "pred_acc": r["pred_acc"],
                            "pred_acc_norm": r["pred_acc_norm"], "correct_acc_norm": r["correct_acc_norm"]}
            out.append({"rule": rule, "rule_text": desc, "found": True, "rank": rank, "order": o,
                        "source_row_index": t["source_row_index"], "context": t["context"],
                        "endings": t["choices"], "label": t["label"], "conditions": conds})
    return out


def qualitative_text(cases):
    L = ["HellaSwag qualitative cases (acc_norm).", QUAL_RULE_TEXT,
         "Scores: summed log likelihood of each ending (nats) / per-character score (log likelihood divided by "
         "ending length in characters); * marks the acc_norm prediction.", ""]
    for x in cases:
        L.append(f"=== Rule {x['rule']}: {x['rule_text']}")
        if not x["found"]:
            L += [x["message"], ""]
            continue
        L.append(f"ID order {x['order']} (source row {x['source_row_index']}), label {x['label']}")
        L.append(f"Context: {x['context']}")
        for i, e in enumerate(x["endings"]):
            L.append(f"  [{i}]{' (label)' if i == x['label'] else ''} {e}")
        for c, r in x["conditions"].items():
            cells = [f"{ll:9.3f}/{pc:7.4f}{'*' if i == r['pred_acc_norm'] else ' '}"
                     for i, (ll, pc) in enumerate(zip(r["loglik"], r["per_char"]))]
            L.append(f"  {r['label']:>10} pred {r['pred_acc_norm']} {'correct' if r['correct_acc_norm'] else 'wrong  '} | "
                     + " | ".join(cells))
        L.append("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ writing
def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k]))
                        for k in keys})


def build(root, out, expect, hs_records, require_committed=True, figures=True):
    inp = Inputs(root)
    D = load(inp, expect)
    committed = committed_check(list(inp.read)) if require_committed else None
    qrows = quality(D, expect)
    rrows = resources(D, qrows)
    boot = run_bootstrap(D)
    mrows = matched(D, rrows)
    erows, crows = effort(D)
    trows = transfer(D, qrows)
    srows, scorr = block_sensitivities(D)
    diag = diagnostics(D, qrows)
    cons = construction(D, qrows)
    cases = qualitative(D, hs_records)
    out.mkdir(parents=True)
    tables = {"T1_quality.csv": qrows, "T2_resources.csv": rrows, "T3_bootstrap.csv": boot,
              "T4_matched_budget.csv": mrows, "T5_selection_effort.csv": erows, "T5_candidates.csv": crows,
              "T6_dev_to_test.csv": trows, "D_block_sensitivity_ranks.csv": srows,
              "D_block_sensitivity_rank_correlations.csv": scorr}
    for name, rows in tables.items():
        write_csv(out / name, rows)
    with open(out / "qualitative.jsonl", "x", encoding="utf-8") as fh:
        for x in cases:
            fh.write(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n")
    with open(out / "qualitative.txt", "x", encoding="utf-8") as fh:
        fh.write(qualitative_text(cases))
    fig_files = []
    if figures:
        from task1 import analysis_figures as F
        fig_files = F.make_all(D, qrows, rrows, out)
    summary = {
        "analysis_id": ANALYSIS_ID, "eval_id": C.EVAL_ID, "bootstrap": {"method": B.METHOD, "resamples": B.DEFAULT_RESAMPLES,
                                                                        "seed": B.DEFAULT_SEED, "percentiles": list(B.PERCENTILES)},
        "expect": expect, "conditions": CONDITIONS, "full_hellaswag_conditions": FULL_CONDITIONS,
        "pairs": [list(p) for p in PAIRS], "family_labels": FAMILY_LABEL,
        "tables": {k: v for k, v in tables.items()}, "diagnostics": diag, "resolution": resolution(expect),
        "construction": cons, "qualitative_rule": QUAL_RULE_TEXT,
        "qualitative_found": {r: sum(1 for x in cases if x["rule"] == r and x["found"]) for r, _, _ in QUAL_RULES},
        "sanity_assertions": ["perplexity, mean NLL and N recomputed from per-window files equal the stored summaries",
                              "accuracy recomputed from per-example predictions equals the stored summaries",
                              "predictions equal the argmax of stored scores",
                              "bootstrap point estimates equal the direct differences",
                              "identical window lists and HellaSwag ID prefixes across conditions",
                              "P1/P2/P3/2:4 achieved zeros equal targets; P3 and 2:4 zeros equal P2",
                              "Q3 packed bytes within the Q2 budgets", "fp16 lm_eval acceptance PASS",
                              "evaluated P3/Q3 allocations equal the committed selections",
                              "development NLL reproduced within 1e-4 of run 1/1b"],
        "inputs": {str(Path(k).relative_to(Path(root))): v for k, v in sorted(inp.read.items())},
        "inputs_committed": committed, "figures": fig_files,
    }
    write_json_new(out / "summary.json", summary)
    return summary, inp


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="new output directory (refused if it exists)")
    ap.add_argument("--results", default=str(REPO_ROOT / "results" / "task1"), help="results root")
    ap.add_argument("--hellaswag-file", default=None,
                    help="HellaSwag records for qualitative text (default: the locked file, hash-checked)")
    ap.add_argument("--allow-uncommitted", action="store_true", help="skip the committed-inputs check (tests)")
    ap.add_argument("--expect-tokens", type=int, default=C.PPL_SCORED_TOKENS)
    ap.add_argument("--expect-hellaswag-full", type=int, default=C.HELLASWAG_FULL_N)
    ap.add_argument("--expect-hellaswag-subset", type=int, default=C.HELLASWAG_SUBSET_N)
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing existing output directory: {out}")
    from task1.data import format_hellaswag, read_hellaswag
    recs = read_hellaswag() if a.hellaswag_file is None else read_jsonl(a.hellaswag_file)
    hs_records = [format_hellaswag(r) for r in recs]
    expect = {"n_tokens": a.expect_tokens, "hellaswag_full": a.expect_hellaswag_full,
              "hellaswag_subset": a.expect_hellaswag_subset}
    staging = out.with_name(out.name + ".building")
    if staging.exists():
        sys.exit(f"refusing existing staging directory: {staging}")
    summary, inp = build(a.results, staging, expect, hs_records, require_committed=not a.allow_uncommitted,
                         figures=not a.no_figures)
    committed = summary["inputs_committed"]
    rec = RunRecorder(staging / "run.json", {"entry_point": "task1.analysis", "args": vars(a), "expect": expect,
                                              "analysis_id": ANALYSIS_ID})
    rec.finish("ok", inputs_committed=committed, n_inputs=len(inp.read),
               outputs=sorted(p.name for p in staging.iterdir() if p.name != "run.json"),
               hellaswag_text=("locked file (task1.data.read_hellaswag)" if a.hellaswag_file is None
                               else str(a.hellaswag_file)))
    staging.rename(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
