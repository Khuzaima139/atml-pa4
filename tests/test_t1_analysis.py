"""task1.analysis on a tiny synthetic results tree: recomputation assertions, bootstrap reproducibility,
the fixed qualitative rule (including a rule with no matching ID) and every output file."""
import csv
import hashlib
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from common import bootstrap as B
from common.io import read_jsonl, write_jsonl_gz
from common.paths import REPO_ROOT
from task1 import analysis as A
from task1 import config as C

NB = 2
N_WIN, TOK_PER_WIN = 4, 10
N_FULL, N_SUB = 8, 4
E = 1000  # eligible weights
FULL = A.FULL_CONDITIONS
QUANT_BITS = ("2", "3", "4", "8")
EXPECT_ARGS = ["--expect-tokens", str(N_WIN * TOK_PER_WIN), "--expect-hellaswag-full", str(N_FULL),
               "--expect-hellaswag-subset", str(N_SUB)]
OUTPUTS = {"T1_quality.csv", "T2_resources.csv", "T3_bootstrap.csv", "T4_matched_budget.csv",
           "T5_selection_effort.csv", "T5_candidates.csv", "T6_dev_to_test.csv", "D_block_sensitivity_ranks.csv",
           "D_block_sensitivity_rank_correlations.csv", "qualitative.jsonl", "qualitative.txt", "summary.json",
           "run.json", "F1_quality_vs_resource.pdf", "F2_p3_allocation.pdf", "F3_q3_allocation.pdf",
           "F4_calibration_vs_heldout.pdf", "F5_magnitude_vs_wanda.pdf"}
SUMMARY_KEYS = {"analysis_id", "eval_id", "bootstrap", "expect", "conditions", "full_hellaswag_conditions", "pairs",
                "family_labels", "tables", "diagnostics", "resolution", "construction", "qualitative_rule",
                "qualitative_found", "sanity_assertions", "inputs", "inputs_committed", "figures"}
DIAG_KEYS = {"p1_minus_p2", "q2_minus_q1", "two_four_minus_unstructured", "spearman_disagreement_vs_input_norm_cv",
             "spearman_delta_nll_vs_delta_acc_norm_full_hellaswag", "spearman_across_18_conditions"}

# acc_norm correctness per HellaSwag order for the full-set conditions (others follow fp16).
#   order 0, 4: all correct; 1: p2_s50 wrong; 2: p2_s50 and q2_b3 wrong; 3: fp16 wrong, p1_s50 correct;
#   5: q2_b3 wrong; 6, 7: all wrong.
EXPECTED_QUAL = {"fp16_correct_p2_s50_wrong": [1, 2], "fp16_correct_q2_b3_wrong_q3_int3_correct": [2, 5],
                 "all_10_correct": [0, 4], "fp16_wrong_any_compressed_correct": [3]}


def correct_pattern(c, o, no_rule2=False):
    if o in (6, 7):
        return False
    if o == 3:
        return c == "p1_s50"
    if o == 1 and c == "p2_s50":
        return False
    if o == 2 and c in ("p2_s50", "q2_b3"):
        return False
    if o == 5 and c == "q2_b3":
        return False
    if no_rule2 and o in (2, 5) and c == "q3_int3":
        return False
    return True


def wj(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


def endings(o):
    return [f"ending {o}{k} here." for k in range(4)]


def hellaswag_records():
    return [{"order": o, "source_row_index": 100 + o, "source_sha256": "x" * 64,
             "example": {"activity_label": "Test", "ctx_a": f"Context number {o}.", "ctx_b": "then",
                         "endings": endings(o), "label": str(o % 4)}} for o in range(N_FULL)]


def build_tree(root, no_rule2=False):
    rng = np.random.default_rng(0)
    for i, c in enumerate(A.CONDITIONS):
        spec = C.CONDITIONS[c]
        fam = spec["family"]
        src = f"run1b/{c}" if fam == "Q3" else f"run1/{c}"
        # perplexity windows
        steps = {"perplexity": {}}
        for j, s in enumerate(A.SUITES):
            rows = [{"window_index": w, "source": s, "source_row": 10 + w, "token_offset": 0, "n_scored": TOK_PER_WIN,
                     "nll_sum": TOK_PER_WIN * (2.7 + 0.05 * i + 0.01 * j + 0.02 * w + 0.003 * rng.random()),
                     "suite": s, "eval_id": C.EVAL_ID} for w in range(N_WIN)]
            write_jsonl_gz(root / f"run2/{c}/ppl_{s}.jsonl.gz", rows)
            tot = math.fsum(r["nll_sum"] for r in rows)
            N = N_WIN * TOK_PER_WIN
            steps["perplexity"][s] = {"N": N, "nll_sum": tot, "mean_nll": tot / N, "ppl": math.exp(tot / N),
                                      "n_windows": N_WIN}
        # HellaSwag
        scope = spec["hellaswag"]
        n = N_FULL if scope == C.HELLASWAG_FULL else N_SUB
        rows = []
        for o in range(n):
            label = o % 4
            ok = correct_pattern(c if c in FULL else "fp16", o, no_rule2)
            pred = label if ok else (label + 1) % 4
            ll = [-10.0 - 0.1 * k for k in range(4)]
            ll[pred] = -5.0
            rows.append({"order": o, "source_row_index": 100 + o, "label": label, "loglik": ll,
                         "cont_chars": [len(e) for e in endings(o)], "pred_acc": pred, "pred_acc_norm": pred,
                         "correct_acc": ok, "correct_acc_norm": ok, "eval_id": C.EVAL_ID})
        hfile = f"hellaswag_{scope}.jsonl.gz"
        write_jsonl_gz(root / f"run2/{c}/{hfile}", rows)
        acc = sum(r["correct_acc"] for r in rows) / n
        steps["hellaswag"] = {"file": hfile, "scope": scope, "n": n, "acc": acc, "acc_norm": acc}
        # development and calibration NLL
        dev_mean = 2.5 + 0.03 * i + (0.05 if c == "q3_int3" else 0.0)
        write_jsonl_gz(root / f"run2/{c}/calib_nll_c4_development.jsonl.gz",
                       [{"n_scored": 5, "nll_sum": 5 * dev_mean, "seq_index": k} for k in range(2)])
        cal_mean = 2.6 + 0.04 * i + 0.002 * rng.random()
        write_jsonl_gz(root / f"{src}/calib_nll_c4_calibration.jsonl.gz",
                       [{"n_scored": 5, "nll_sum": 5 * cal_mean, "seq_index": k} for k in range(2)])
        r1 = {"elapsed_wall_s": 100.0 + i, "timings_s": {"build_compression_s": 10.0 + i, "build_total_s": 20.0 + i,
                                                          "build_calibration_stats_s": 1.0},
              "steps": {"calib_nll": {"c4_calibration": {"mean_nll": cal_mean}, "c4_development": {"mean_nll": dev_mean}},
                        "build": {"output_error": {"pooled_rel": 0.001 * (i + 1) + 0.0001 * rng.random()},
                                  "score_distributions": {"quantile_levels": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                                                              0.9, 0.95, 0.99, 1.0]}}}}
        if fam == "baseline":
            r1["steps"].pop("build")
        if "sparsity" in spec:
            zeros = int(round(spec["sparsity"] * E))
            r1["steps"]["build"].update({"target_zeros_total": zeros, "mask_zeros_total": zeros})
        wj(root / f"{src}/run.json", r1)
        if fam != "baseline":
            proj = []
            for layer in range(NB):
                for t in C.ELIGIBLE:
                    q = sorted(rng.random(11) + 0.01)
                    proj.append({"layer": layer, "name": t, "qualname": f"model.layers.{layer}.{t}",
                                 "disagreement_with_magnitude": float(rng.random() * 0.2),
                                 "x_norm_cv_dense": float(0.5 + rng.random()),
                                 "dist_magnitude": {"quantiles": q}, "dist_wanda": {"quantiles": [x * 2 for x in q]}})
            write_jsonl_gz(root / f"{src}/projections.jsonl.gz", proj)
        # run-2 record
        r2 = {"status": "ok", "config": {"condition_spec": dict(spec)}, "steps": steps}
        if c == "fp16":
            steps["calib_nll"] = {"c4_development": {"mean_nll": dev_mean}}
            r2["acceptance"] = "PASS"
            steps["lmeval_verify"] = {"acceptance": "PASS", "acceptance_run": "fp32_model"}
        else:
            steps["reproduction"] = {"mean_nll": dev_mean}
            steps["inputs"] = {"run1": {"dir": f"results/task1/{src}"}, "freeze": {"applies": False}}
        wj(root / f"run2/{c}/run.json", r2)
        # resources
        if "sparsity" in spec:
            zeros = int(round(spec["sparsity"] * E))
            logical = {"format": "sparse16", "eligible_bytes": 125 + 2 * (E - zeros),
                       "whole_model_bytes": 125 + 2 * (E - zeros) + 400, "bits_per_eligible_weight": 1.0}
        elif fam == "baseline":
            zeros, logical = 0, None
        else:
            bits = spec.get("bits") or {"q2_b4": 4, "q2_b3": 3}[spec["budget"]]
            zeros = 100 + i
            eb = E * bits // 8 + 30 - (2 if fam == "Q3" else 0)
            logical = {"format": "packed_group_quant", "eligible_bytes": eb, "whole_model_bytes": eb + 400,
                       "bits_per_eligible_weight": 8 * eb / E}
        storage = {"eligible_weights": E, "nonzeros": {"global": E - zeros}, "nominal_bits_per_eligible_weight": 16,
                   "fp16_dense": {"eligible_bytes": 2 * E, "whole_model_bytes": 2 * E + 400},
                   "sparse16": {"eligible_bytes": 125 + 2 * (E - zeros), "whole_model_bytes": 525 + 2 * (E - zeros),
                                "nonzeros": E - zeros}}
        if logical:
            storage["logical"] = logical
            storage["artifact_files"] = {"safetensors_bytes": 999, "total_bytes": 1099}
        meas = {}
        for k, base in (("prefill_512", 100.0), ("prefill_2048", 700.0), ("decode_64_after_prefill_512", 2300.0)):
            t = [base + i + d for d in (0.0, 1.0, 2.0, 3.0, 4.0)]
            meas[k] = {"median_ms": float(np.median(t)), "times_ms": t, "repetitions": 5, "warmup": 3,
                       "peak_allocated_bytes_max": 3_000_000 + i}
        wj(root / f"run2_resources/{c}/run.json",
           {"status": "ok", "peak_vram_allocated_bytes": {"0": 3_100_000 + i},
            "steps": {"storage": storage, "resources": {"execution": {"mode": "dense_fp16"}, "device_name": "cpu",
                                                         "measurements": meas}}})
    # probes
    eff = {"wall_s": 5.0, "full_model_calibration_passes": 2, "block_forward_passes": 0, "restores": 2,
           "calibration_sequences_per_pass": 2}
    wj(root / "probe_prune/sensitivity.json",
       {"n_blocks": NB, "values": {"S1": [0.02, 0.01], "S2": {"0.50": [0.01, 0.03], "0.75": [0.1, 0.2]}},
        "effort": {"S1": eff, "S2": eff}})
    wj(root / "probe_quant/sensitivity.json",
       {"n_blocks": NB, "values": {"QS1": {b: [0.1 / int(b), 0.05 / int(b)] for b in QUANT_BITS}},
        "effort": {"QS1": eff}})
    for layer in range(NB):
        units = [{"layer": layer, "unit": u} for u in A.Q3V2_UNIT_TYPES]
        wj(root / f"probe_quant_v2/blocks_{layer:02d}_{layer:02d}/sensitivity.json",
           {"n_blocks": NB, "units": units,
            "values": {"QS1v2": {b: [0.01 * (k + 1) / int(b) + 0.001 * layer for k in range(4)] for b in QUANT_BITS}},
            "effort": {"QS1v2": eff}})
    wj(root / "calib_stats/run.json", {"elapsed_wall_s": 50.0})
    # selections
    for cond, sd in {**A.SELECT_DIRS, **{f"{k}_block": v for k, v in A.BLOCK_Q3_SELECT_DIRS.items()}}.items():
        base = cond.replace("_block", "")
        sel = {"condition": base, "candidate_id": f"{base}.X", "measure": "S1", "lambda": 0.08, "unit": "block"}
        if base.startswith("p3"):
            s = C.CONDITIONS[base]["sparsity"]
            sel["per_block"] = {"block_sparsity_achieved": [s - 0.05, s + 0.05]}
        elif not cond.endswith("_block"):
            sel["unit"] = "fold_group"
            sel["per_unit"] = {"ids": [f"{l}.{u}" for l in range(NB) for u in A.Q3V2_UNIT_TYPES],
                               "bits": [3, 4, 3, 8, 2, 3, 4, 3]}
        wj(root / f"{sd}/selected_allocation.json", sel)
        cands = [{"candidate_id": f"{base}.{k}", "measure": None if k == 0 else "S1", "lambda": 0.0 if k == 0 else 0.08,
                  "reference": k == 0, "collapsed_to_uniform": k == 0, "selected": k == 1,
                  "development": {"mean_nll": 2.8 - 0.01 * k}, "calibration": {"mean_nll": 2.9 - 0.01 * k},
                  "output_error_pooled_rel": 0.01, "timings_s": {"build": 1.0, "nll": 0.5}} for k in range(2)]
        write_jsonl_gz(root / f"{sd}/candidates.jsonl.gz", cands)
        wj(root / f"{sd}/run.json", {"elapsed_wall_s": 30.0})
    for c, sd in A.SELECT_DIRS.items():
        sha = hashlib.sha256((root / f"{sd}/selected_allocation.json").read_bytes()).hexdigest()
        p = root / f"run2/{c}/run.json"
        r = json.loads(p.read_text())
        r["steps"]["inputs"]["freeze"] = {"applies": True, "clean": True, "sha256": sha, "candidate_id": f"{c}.X",
                                          "freeze_commit": "0" * 40}
        p.write_text(json.dumps(r))
    hs = root.parent / f"{root.name}_hellaswag.jsonl"
    hs.write_text("".join(json.dumps(r) + "\n" for r in hellaswag_records()))
    return hs


def run_main(root, hs, out, extra=()):
    return A.main(["--results", str(root), "--out", str(out), "--hellaswag-file", str(hs), "--allow-uncommitted",
                   *EXPECT_ARGS, *extra])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("t1res") / "task1"
    hs = build_tree(root)
    return root, hs


@pytest.fixture(scope="module")
def analysed(tree, tmp_path_factory):
    root, hs = tree
    out = tmp_path_factory.mktemp("an") / "analysis"
    cmd = [sys.executable, "-m", "task1.analysis", "--results", str(root), "--out", str(out), "--hellaswag-file",
           str(hs), "--allow-uncommitted", *EXPECT_ARGS]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    return out


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_output_file_and_key(analysed):
    assert {p.name for p in analysed.iterdir()} == OUTPUTS
    for p in analysed.iterdir():
        assert p.stat().st_size > 0, p
        if p.suffix == ".pdf":
            assert p.read_bytes().startswith(b"%PDF")
    s = json.loads((analysed / "summary.json").read_text())
    assert set(s) == SUMMARY_KEYS
    assert set(s["diagnostics"]) == DIAG_KEYS
    assert s["inputs_committed"] is None
    assert s["resolution"] == {"hellaswag_full_one_item_pp": 100 / N_FULL, "hellaswag_full_n": N_FULL,
                               "hellaswag_subset_one_item_pp": 100 / N_SUB, "hellaswag_subset_n": N_SUB}
    assert sorted(s["figures"]) == sorted(p for p in OUTPUTS if p.endswith(".pdf"))
    run = json.loads((analysed / "run.json").read_text())
    assert run["status"] == "ok" and run["config"]["analysis_id"] == A.ANALYSIS_ID
    t1 = rows(analysed / "T1_quality.csv")
    assert [r["condition"] for r in t1] == A.CONDITIONS
    fp = t1[0]
    for r in t1:
        assert int(r["wikitext2_test_N"]) == N_WIN * TOK_PER_WIN
        assert math.isclose(float(r["wikitext2_test_delta_mean_nll"]),
                            float(r["wikitext2_test_mean_nll"]) - float(fp["wikitext2_test_mean_nll"]))
    assert len(rows(analysed / "T2_resources.csv")) == 18
    assert all(r["holds"] == "True" for r in rows(analysed / "T4_matched_budget.csv"))
    assert len(rows(analysed / "T6_dev_to_test.csv")) == 4
    assert len(rows(analysed / "D_block_sensitivity_ranks.csv")) == NB


def test_bootstrap_matches_common_bootstrap_and_is_reproducible(analysed, tree, tmp_path):
    root, hs = tree
    t3 = rows(analysed / "T3_bootstrap.csv")
    assert len(t3) == 4 * (2 * 2 + 2)
    r = next(x for x in t3 if x["comparison"] == "p3_s50 - p2_s50" and x["suite"] == "wikitext2_test"
             and x["metric"] == "delta_ppl")
    a = read_jsonl(root / "run2/p2_s50/ppl_wikitext2_test.jsonl.gz")
    b = read_jsonl(root / "run2/p3_s50/ppl_wikitext2_test.jsonl.gz")
    ref = B.paired_ppl_diff([x["nll_sum"] for x in a], [x["n_scored"] for x in a], [x["nll_sum"] for x in b],
                            [x["n_scored"] for x in b])
    assert (float(r["point"]), float(r["ci_low"]), float(r["ci_high"])) == (ref["point"], *ref["ci"])
    assert r["n_resamples"] == "10000" and r["seed"] == "6304"
    acc = next(x for x in t3 if x["comparison"] == "p3_s75 - p2_s75" and x["metric"] == "delta_acc_norm_pp")
    assert acc["n"] == str(N_SUB)



def test_rebuild_is_byte_identical(analysed, tree, tmp_path):
    root, hs = tree
    out2 = run_main(root, hs, tmp_path / "again")
    for name in OUTPUTS - {"run.json"}:
        assert (out2 / name).read_bytes() == (analysed / name).read_bytes(), name


def test_qualitative_rule(analysed):
    cases = read_jsonl(analysed / "qualitative.jsonl")
    got = {}
    for x in cases:
        assert x["found"]
        got.setdefault(x["rule"], []).append(x["order"])
        assert set(x["conditions"]) == set(FULL)
        assert x["endings"] == endings(x["order"]) and x["label"] == x["order"] % 4
        for c, v in x["conditions"].items():
            assert v["per_char"] == [ll / n for ll, n in zip(v["loglik"], v["cont_chars"])]
    assert got == EXPECTED_QUAL
    assert "Rule all_10_correct" in (analysed / "qualitative.txt").read_text()


def test_qualitative_rule_without_match(tmp_path):
    root = tmp_path / "task1"
    hs = build_tree(root, no_rule2=True)
    out = run_main(root, hs, tmp_path / "an", ["--no-figures"])
    cases = read_jsonl(out / "qualitative.jsonl")
    miss = [x for x in cases if x["rule"] == "fp16_correct_q2_b3_wrong_q3_int3_correct"]
    assert miss == [{"rule": "fp16_correct_q2_b3_wrong_q3_int3_correct", "rule_text": A.QUAL_RULES[1][2],
                     "found": False, "message": "no ID satisfies this rule"}]
    assert "no ID satisfies this rule" in (out / "qualitative.txt").read_text()
    assert json.loads((out / "summary.json").read_text())["qualitative_found"][
        "fp16_correct_q2_b3_wrong_q3_int3_correct"] == 0


def _tamper(root, rel, fn):
    p = root / rel
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))


@pytest.mark.parametrize("rel,fn,msg", [
    ("run2/q1_b4/run.json", lambda r: r["steps"]["perplexity"]["c4_validation"].update(ppl=1.0), "recomputed perplexity"),
    ("run2/p2_s50/run.json", lambda r: r["steps"]["hellaswag"].update(acc_norm=0.0), "recomputed HellaSwag accuracy"),
    ("run2_resources/p3_s50/run.json", lambda r: r["steps"]["storage"]["nonzeros"].update({"global": 1}),
     "achieved zeros"),
    ("run2/fp16/run.json", lambda r: r.update(acceptance="FAIL"), "fp16 acceptance"),
    ("run2/q3_int3/run.json", lambda r: r["steps"]["inputs"]["freeze"].update(sha256="0" * 64), "committed selection"),
])
def test_assertions_fail_on_inconsistent_inputs(tmp_path, rel, fn, msg):
    root = tmp_path / "task1"
    hs = build_tree(root)
    _tamper(root, rel, fn)
    with pytest.raises(A.AnalysisError, match=msg):
        run_main(root, hs, tmp_path / "an", ["--no-figures"])
    assert not (tmp_path / "an").exists()


def test_window_list_mismatch_fails(tmp_path):
    root = tmp_path / "task1"
    hs = build_tree(root)
    p = root / "run2/q2_b3/ppl_wikitext2_test.jsonl.gz"
    r = read_jsonl(p)
    r[0]["source_row"] += 1
    p.unlink()
    write_jsonl_gz(p, r)
    with pytest.raises(A.AnalysisError, match="window list differs"):
        run_main(root, hs, tmp_path / "an", ["--no-figures"])


def test_refuses_existing_out(tree, tmp_path):
    root, hs = tree
    (tmp_path / "an").mkdir()
    with pytest.raises(SystemExit, match="refusing existing output"):
        run_main(root, hs, tmp_path / "an", ["--no-figures"])


def test_committed_check_refuses_untracked(tree):
    root, _ = tree
    with pytest.raises((A.AnalysisError, ValueError)):
        A.committed_check([root / "calib_stats/run.json"])
