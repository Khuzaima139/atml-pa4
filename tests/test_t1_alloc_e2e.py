"""End-to-end CPU run of calib_stats, probe, select and compress for P3/Q3 with a tiny random 4-block model."""
import json
import os
import subprocess
import sys

import pytest

from common.io import read_jsonl, sha256_file
from common.paths import REPO_ROOT
from task1 import allocate as A
from task1 import config as C
from task1 import compress, probe, select, storage
from task1 import prune as P

LIM_C, LIM_D = 2, 2
N_LAYERS = 4
N_MATRICES = N_LAYERS * len(C.ELIGIBLE)
SHAPES = {"q_proj": (256, 256), "k_proj": (128, 256), "v_proj": (128, 256), "o_proj": (256, 256),
          "gate_proj": (512, 256), "up_proj": (512, 256), "down_proj": (256, 512)}
MATS = [{"layer": li, "name": n, "qualname": f"model.layers.{li}.{'self_attn' if n in C.ATTN_PROJ else 'mlp'}.{n}",
         "shape": list(SHAPES[n])} for li in range(N_LAYERS) for n in C.ELIGIBLE]

RUN_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "model",
           "eval_id", "timings_s", "steps"}
COMMON_CONFIG = {"entry_point", "settings", "device", "model_path", "limits", "limited", "test_only",
                 "output_files", "data_files", "batch_size"}
PROBE_CONFIG = COMMON_CONFIG | {"kind", "measures", "definitions", "alloc_settings", "calib_stats_dir"}
SELECT_CONFIG = COMMON_CONFIG | {"kind", "target", "target_spec", "condition", "alloc_settings", "probe_dir",
                                 "calib_stats_dir"}
COMPRESS_CONFIG = COMMON_CONFIG | {"condition", "condition_spec", "artifacts_dir", "calib_stats_dir", "allocation_file"}
CAL_ROW = {"eval_id", "split", "seq_index", "source_row", "token_offset", "nll_sum", "n_scored"}
CAL_SUMMARY = {"eval_id", "split", "n_sequences", "N", "nll_sum", "mean_nll", "batch_size"}
EFFORT = {"wall_s", "full_model_calibration_passes", "block_forward_passes", "calibration_sequences_per_pass"}
SENS = {"eval_id", "kind", "unit", "n_blocks", "measures", "definitions", "values", "fp16_calib_nll", "effort",
        "calib_stats", "model", "limits", "alloc_settings"}
ISOLATED_ROW = {"eval_id", "kind", "measure", "level", "layer", "value", "calib_mean_nll", "calib_nll_sum",
                "n_scored", "fp16_calib_mean_nll", "wall_s"}
CAND = {"eval_id", "candidate_id", "index", "condition", "kind", "target", "reference", "measure", "lambda", "rule",
        "development", "calibration", "output_error_pooled_rel", "timings_s", "per_block", "totals", "selected"}
CAND = CAND | {"collapsed_to_uniform"}
CAND_P3 = CAND | {"shift", "bisect_iterations", "bisect_residual", "n_largest_remainder_added", "per_matrix_zeros"}
CAND_Q3 = CAND | {"per_matrix_bits", "objective", "objective_exact", "dp_states_per_block"}
SELECTED = {"eval_id", "condition", "kind", "target", "unit", "n_blocks", "rule", "measure", "lambda", "candidate_id",
            "per_block", "totals", "development", "calibration", "selection_criterion", "n_candidates",
            "n_reference_candidates", "collapsed_to_uniform", "all_candidates_collapsed", "inputs", "model", "model_path", "limits", "code_commit", "code_dirty",
            "alloc_settings"}
MANIFEST = {"eval_id", "condition", "family", "kind", "model", "model_dtype", "conventions", "matrices",
            "extra_tensors", "logical_bytes", "tensors", "stored_tensor_bytes", "file", "allocation"}
ROW = {"eval_id", "layer", "name", "qualname", "shape", "err_num", "err_den", "rel_output_error", "nonzeros"}
ROW_P3 = ROW | {"target_zeros", "mask_zeros", "x_norm_cv_dense", "dist_magnitude", "dist_wanda",
                "disagreement_with_magnitude", "x_norm_cv_sequential"}
ROW_Q3 = ROW | {"bits", "groups", "n_guarded_groups", "n_clipped_codes", "n_one_sided_groups"}
DIST = {"n", "n_zero", "quantiles", "mean", "log10_hist"}


def _env():
    return dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")


def _run(args):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True,
                          timeout=1800)


@pytest.fixture(scope="module")
def runs(tiny_compress4_dir, tmp_path_factory):
    b = tmp_path_factory.mktemp("alloc_e2e")
    t = ["--model-path", str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", str(LIM_C)]
    steps = [
        ["task1.calib_stats", "--out", str(b / "calib"), *t],
        ["task1.probe", "--kind", "prune", "--calib-stats", str(b / "calib"), "--out", str(b / "probe_prune"), *t],
        ["task1.probe", "--kind", "quant", "--calib-stats", str(b / "calib"), "--out", str(b / "probe_quant"), *t],
        ["task1.select", "--kind", "p3", "--target", "s50", "--probe", str(b / "probe_prune"), "--calib-stats",
         str(b / "calib"), "--out", str(b / "sel_p3"), *t, "--limit-dev", str(LIM_D)],
        ["task1.select", "--kind", "q3", "--target", "int3", "--probe", str(b / "probe_quant"), "--calib-stats",
         str(b / "calib"), "--out", str(b / "sel_q3"), *t, "--limit-dev", str(LIM_D)],
        ["task1.compress", "--condition", "p3_s50", "--allocation", str(b / "sel_p3" / "selected_allocation.json"),
         "--out", str(b / "p3_s50"), "--artifacts", str(b / "art_p3_s50"), *t, "--limit-dev", str(LIM_D)],
        ["task1.compress", "--condition", "q3_int3", "--allocation", str(b / "sel_q3" / "selected_allocation.json"),
         "--calib-stats", str(b / "calib"), "--out", str(b / "q3_int3"), "--artifacts", str(b / "art_q3_int3"), *t,
         "--limit-dev", str(LIM_D)],
    ]
    for a in steps:
        p = _run(a)
        assert p.returncode == 0, (a[0], p.stderr[-4000:])
    return b


def _load(p):
    return json.loads(p.read_text())


def _check_guard(steps):
    g = steps["held_out_guard"]
    assert g["attempts"] == [] and set(g["forbidden"]) == {
        "wikitext2_test_windows_locked.jsonl.gz", "c4_validation_windows_locked.jsonl.gz",
        "hellaswag_val_locked.jsonl.gz"}


@pytest.mark.parametrize("kind", ["prune", "quant"])
def test_probe_outputs(runs, kind):
    d = runs / f"probe_{kind}"
    assert sorted(x.name for x in d.iterdir()) == sorted(probe.OUTPUT_FILES.values())
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == PROBE_CONFIG
    assert r["config"]["limits"] == {"calib": LIM_C, "dev": None}
    assert set(r["timings_s"]) == {"load_s", "fp16_calib_nll_s"}
    s = r["steps"]
    assert set(s) == {"restore", "calib_stats_source", "fp16_calib_nll", "pristine_after", "sensitivity", "effort",
                      "held_out_guard"} | ({"awq_alphas"} if kind == "quant" else set())
    _check_guard(s)
    cs = s["calib_stats_source"]
    for k in ("dense_stats", "awq_scales"):
        assert cs[f"{k}_sha256"] == sha256_file(runs / "calib" / f"{k}.safetensors")
    sens = _load(d / "sensitivity.json")
    assert set(sens) == SENS and sens["n_blocks"] == N_LAYERS and sens["unit"] == "block"
    assert s["sensitivity"]["sha256"] == sha256_file(d / "sensitivity.json")
    v = sens["values"]
    m1, m2 = probe.KINDS[kind]
    assert set(v) == {m1, m2} and set(sens["effort"]) == {m1, m2, "fp16_reference"}
    assert all(EFFORT <= set(e) for e in sens["effort"].values())
    if kind == "prune":
        assert len(v["S1"]) == N_LAYERS and all(0 <= x < 1 for x in v["S1"])
        assert set(v["S2"]) == {"0.50", "0.75"} and all(len(x) == N_LAYERS for x in v["S2"].values())
        passes = {"S1": 0, "S2": 2 * N_LAYERS}
    else:
        for m in ("QS1", "QS2"):
            assert set(v[m]) == {"2", "3", "4", "8"} and all(len(x) == N_LAYERS for x in v[m].values())
        for li in range(N_LAYERS):
            assert v["QS2"]["2"][li] > v["QS2"]["3"][li] > v["QS2"]["4"][li] > v["QS2"]["8"][li] > 0
        passes = {"QS1": 4 * N_LAYERS, "QS2": 0}
        assert sens["effort"]["QS2"]["block_forward_passes"] == 4 * N_LAYERS
    for m, n in passes.items():
        assert sens["effort"][m]["full_model_calibration_passes"] == n
    rows = read_jsonl(d / "probe_rows.jsonl.gz")
    iso = [x for x in rows if x["measure"] in ("S2", "QS1")]
    assert len(iso) == sum(passes.values())
    extra = {"zeros"} if kind == "prune" else {"bits"}
    assert all(set(x) == ISOLATED_ROW | extra for x in iso)
    for x in iso:
        assert x["value"] == pytest.approx(x["calib_mean_nll"] - sens["fp16_calib_nll"]["mean_nll"], abs=1e-12)
    other = [x for x in rows if x not in iso]
    assert len(other) == (N_LAYERS if kind == "prune" else 4 * N_LAYERS)
    if kind == "prune":
        assert all(set(x) == {"eval_id", "kind", "measure", "level", "layer", "value", "n_scores", "n_outliers",
                              "mean_score", "threshold", "owl_m"} for x in other)
        assert all(x["zeros"] == sum(n[0] * n[1] for n in SHAPES.values()) * float(x["level"]) for x in iso)
    else:
        assert all(set(x) == {"eval_id", "kind", "measure", "level", "layer", "value", "projections"} for x in other)
        assert all(set(x["projections"]) == set(C.ELIGIBLE) for x in other)
    nll = read_jsonl(d / "probe_calib_nll.jsonl.gz")
    assert len(nll) == LIM_C * (1 + sum(passes.values()))
    assert all(set(x) == CAL_ROW | {"measure", "level", "layer"} for x in nll)
    for p in d.glob("*.gz"):
        assert p.read_bytes()[4:8] == b"\x00\x00\x00\x00"


@pytest.mark.parametrize("kind", ["p3", "q3"])
def test_select_outputs(runs, kind):
    d = runs / f"sel_{kind}"
    assert sorted(x.name for x in d.iterdir()) == sorted(select.OUTPUT_FILES.values())
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == SELECT_CONFIG
    s = r["steps"]
    assert set(s) == {"restore", "inputs", "n_candidates", "selection", "pristine_after", "held_out_guard"}
    _check_guard(s)
    n_cand = 11 if kind == "p3" else 3
    assert s["n_candidates"] == {"total": n_cand, "reference": 1}
    pdir = runs / ("probe_prune" if kind == "p3" else "probe_quant")
    assert s["inputs"]["probe"]["sensitivity_sha256"] == sha256_file(pdir / "sensitivity.json")
    rows = read_jsonl(d / "candidates.jsonl.gz")
    assert len(rows) == n_cand and sum(x["selected"] for x in rows) == 1
    assert all(set(x) == (CAND_P3 if kind == "p3" else CAND_Q3) for x in rows)
    win = [x for x in rows if x["selected"]][0]
    assert not win["reference"]
    pool = [x for x in rows if not x["reference"]]
    fresh = [x for x in pool if not x["collapsed_to_uniform"]]
    assert fresh, "the tiny model gives at least one allocation that differs from the reference"
    assert not win["collapsed_to_uniform"]
    assert win["development"]["mean_nll"] == min(x["development"]["mean_nll"] for x in fresh)
    assert s["selection"]["collapsed_to_uniform"] == [x["candidate_id"] for x in pool if x["collapsed_to_uniform"]]
    assert s["selection"]["winner_collapsed_to_uniform"] is False and s["selection"]["all_candidates_collapsed"] is False
    for x in rows:
        assert set(x["development"]) == CAL_SUMMARY and x["development"]["n_sequences"] == LIM_D
        assert set(x["calibration"]) == CAL_SUMMARY and x["calibration"]["n_sequences"] == LIM_C
    if kind == "p3":
        p2 = A.p2_zeros(MATS, 0.5)
        ref = [x for x in rows if x["reference"]][0]
        assert ref["per_matrix_zeros"] == p2 and ref["lambda"] == 0.0 and ref["collapsed_to_uniform"] is True
        for x in pool:
            assert x["collapsed_to_uniform"] == (x["per_matrix_zeros"] == p2)
        lams = sorted({(x["measure"], x["lambda"]) for x in pool})
        assert lams == sorted((m, lam) for m in C.P3_MEASURES for lam in C.P3_LAMBDAS)
        for x in rows:
            assert x["totals"]["zeros"] == x["totals"]["p2_zeros"] == sum(p2.values())
            assert sum(x["per_matrix_zeros"].values()) == sum(p2.values())
            assert all(C.P3_S_MIN - 1e-3 <= v <= C.P3_S_MAX + 1e-3 for v in x["per_block"]["block_sparsity_achieved"])
    else:
        budget = storage.packed_quant_bytes([tuple(m["shape"]) for m in MATS], 3)
        for x in rows:
            assert x["totals"]["eligible_packed_bytes"] <= x["totals"]["budget_bytes"] == budget
            assert x["collapsed_to_uniform"] == (x["per_block"]["bits"] == [3] * N_LAYERS)
        assert [x for x in rows if x["reference"]][0]["per_block"]["bits"] == [3] * N_LAYERS
    nll = read_jsonl(d / "candidate_nll.jsonl.gz")
    assert len(nll) == n_cand * (LIM_C + LIM_D) and all(set(x) == CAL_ROW | {"candidate_id"} for x in nll)
    sel = _load(d / "selected_allocation.json")
    extra = {"sparsity", "per_matrix_zeros", "shift"} if kind == "p3" else {
        "ref_bits", "per_matrix_bits", "objective"}
    assert set(sel) == SELECTED | extra
    assert sel["candidate_id"] == win["candidate_id"] == s["selection"]["winner"]
    assert sel["collapsed_to_uniform"] is False and sel["all_candidates_collapsed"] is False
    assert sel["condition"] == ("p3_s50" if kind == "p3" else "q3_int3") and sel["n_candidates"] == n_cand
    assert sel["inputs"]["calib_stats"]["awq_scales_sha256"] == sha256_file(runs / "calib" / "awq_scales.safetensors")
    assert sel["inputs"]["probe"]["run_sha256"] == sha256_file(pdir / "run.json")
    assert sel["code_commit"] == r["meta"]["git_commit"] and len(sel["code_commit"]) == 40
    assert s["selection"]["sha256"] == sha256_file(d / "selected_allocation.json")


@pytest.mark.parametrize("cond", ["p3_s50", "q3_int3"])
def test_compress_with_allocation(runs, cond):
    d = runs / cond
    fam = C.CONDITIONS[cond]["family"]
    sel_file = runs / ("sel_p3" if fam == "P3" else "sel_q3") / "selected_allocation.json"
    sel = _load(sel_file)
    assert sorted(x.name for x in d.iterdir()) == sorted(compress.OUTPUT_FILES.values())
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == COMPRESS_CONFIG
    s = r["steps"]
    assert set(s) == {"restore", "allocation", "build", "storage", "calib_nll", "artifact",
                      "peak_gpu_allocated_bytes", "held_out_guard"} | ({"awq_source"} if fam == "Q3" else set())
    _check_guard(s)
    al = s["allocation"]
    assert set(al) == {"file", "sha256", "candidate_id", "measure", "lambda", "selection_code_commit", "totals",
                       "inputs", "git"}
    assert al["sha256"] == sha256_file(sel_file) and al["candidate_id"] == sel["candidate_id"]
    rows = read_jsonl(d / "projections.jsonl.gz")
    assert len(rows) == N_MATRICES
    man = _load(runs / f"art_{cond}" / "manifest.json")
    assert set(man) == MANIFEST and man["allocation"]["sha256"] == al["sha256"]
    assert s["artifact"]["reload_matches_executed"] is True
    # Development NLL of the saved condition equals the one recorded at selection.
    assert s["calib_nll"]["c4_development"]["mean_nll"] == sel["development"]["mean_nll"]
    if fam == "P3":
        assert all(set(x) == ROW_P3 for x in rows)
        z = {x["qualname"]: x["mask_zeros"] for x in rows}
        assert z == sel["per_matrix_zeros"] and sum(z.values()) == sum(A.p2_zeros(MATS, 0.5).values())
        assert s["build"]["mask_zeros_total"] == s["build"]["target_zeros_total"] == sum(z.values())
        sd = s["build"]["score_distributions"]
        assert sd["wanda_norm_source"] == "sequential" and len(sd["wanda_log10_edges"]) == P.N_BINS + 1
        assert all(set(x["dist_wanda"]) == DIST for x in rows)
    else:
        assert all(set(x) == ROW_Q3 for x in rows)
        assert {x["qualname"]: x["bits"] for x in rows} == sel["per_matrix_bits"]
        pq = s["storage"]["packed_quant"]
        assert pq["total"] <= pq["budget_bytes"] == sel["totals"]["budget_bytes"]
        assert pq["total"] == s["artifact"]["logical_bytes"]["eligible"] == sel["totals"]["eligible_packed_bytes"]
        assert len(s["artifact"]["extra_tensors"]) == 3 * N_LAYERS
        assert set(s["awq_source"]) == {str(b) for b in set(sel["per_block"]["bits"])}


def test_allocation_refusals(runs, tiny_compress4_dir, tmp_path):
    t = ["--model-path", str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", str(LIM_C),
         "--limit-dev", str(LIM_D)]
    p = _run(["task1.compress", "--condition", "p3_s50", "--out", str(tmp_path / "a"), "--artifacts",
              str(tmp_path / "aa"), *t])
    assert p.returncode != 0 and "requires an allocation file" in p.stderr and not (tmp_path / "a").exists()
    sel = runs / "sel_p3" / "selected_allocation.json"
    p = _run(["task1.compress", "--condition", "p3_s75", "--allocation", str(sel), "--out", str(tmp_path / "b"),
              "--artifacts", str(tmp_path / "bb"), *t])
    assert p.returncode != 0 and "is for condition p3_s50" in p.stderr
    bad = _load(sel)
    q = next(iter(bad["per_matrix_zeros"]))
    bad["per_matrix_zeros"][q] += 1
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    p = _run(["task1.compress", "--condition", "p3_s50", "--allocation", str(tmp_path / "bad.json"), "--out",
              str(tmp_path / "c"), "--artifacts", str(tmp_path / "cc"), *t])
    assert p.returncode != 0 and "failed validation" in p.stderr
    with pytest.raises(SystemExit, match="only used by P3 and Q3"):
        compress.main(["--condition", "p2_s50", "--allocation", str(sel), "--out", str(tmp_path / "d"),
                       "--artifacts", str(tmp_path / "dd")])
    p = _run(["task1.select", "--kind", "q3", "--target", "int3", "--probe", str(runs / "probe_prune"),
              "--calib-stats", str(runs / "calib"), "--out", str(tmp_path / "e"), *t])
    assert p.returncode != 0 and "does not serve q3" in p.stderr
    with pytest.raises(SystemExit, match="--target for p3"):
        select.main(["--kind", "p3", "--target", "int4", "--probe", "x", "--calib-stats", "y",
                     "--out", str(tmp_path / "f")])
