"""End-to-end CPU runs of python -m task1.calib_stats and python -m task1.compress with a tiny random model."""
import json
import os
import subprocess
import sys

import pytest

from common.io import read_jsonl, sha256_file
from common.paths import REPO_ROOT, RESULTS_DIR
from task1 import awq, calib_stats, compress, storage
from task1 import config as C
from task1 import prune as P

LIM_C, LIM_D = 2, 2
CONDS = ("fp16", "p1_s50", "p2_s50", "p2_24_s50", "q1_b4", "q2_b3")
N_LAYERS, N_MATRICES = 2, 14

RUN_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "model",
           "eval_id", "timings_s", "steps"}
COMMON_CONFIG = {"entry_point", "settings", "device", "model_path", "limits", "limited", "test_only",
                 "output_files", "data_files", "batch_size"}
COMPRESS_CONFIG = COMMON_CONFIG | {"condition", "condition_spec", "artifacts_dir", "calib_stats_dir", "allocation_file"}
CALIB_CONFIG = COMMON_CONFIG | {"awq"}
BUILD_TIMINGS = {f"build_{k}" for k in ("capture_s", "calibration_stats_s", "construct_s", "error_pass_s",
                                        "apply_s", "rerun_s", "compression_s", "measurement_s")} | {"build_total_s"}
COMPRESS_TIMINGS = BUILD_TIMINGS | {"load_s", "calib_nll_c4_calibration_s", "calib_nll_c4_development_s",
                                    "artifact_save_s", "artifact_verify_s"}
BUILD = {"family", "calibration_sequences", "nonzeros", "output_error", "score_distributions",
         "peak_gpu_allocated_bytes"}
STORAGE = {"convention", "eval_id", "eligible_weights", "eligible_matrices", "non_eligible_params", "fp16_dense",
           "sparse16", "nominal_bits_per_eligible_weight", "representation"}
ARTIFACT = {"dir", "file", "manifest_bytes", "manifest_sha256", "logical_bytes", "stored_tensor_bytes", "n_tensors",
            "extra_tensors", "reload_matches_executed", "executed_tensors_checked"}
ROW = {"eval_id", "layer", "name", "qualname", "shape", "err_num", "err_den", "rel_output_error", "nonzeros"}
ROW_PRUNE = ROW | {"target_zeros", "mask_zeros", "x_norm_cv_dense", "dist_magnitude", "dist_wanda"}
ROW_QUANT = ROW | {"bits", "groups", "n_guarded_groups", "n_clipped_codes", "n_one_sided_groups"}
ROW_KEYS = {"fp16": ROW, "p1_s50": ROW_PRUNE,
            "p2_s50": ROW_PRUNE | {"disagreement_with_magnitude", "x_norm_cv_sequential"},
            "p2_24_s50": ROW_PRUNE | {"disagreement_with_magnitude", "disagreement_with_unstructured_wanda",
                                      "x_norm_cv_sequential"},
            "q1_b4": ROW_QUANT, "q2_b3": ROW_QUANT}
CAL_ROW = {"eval_id", "split", "seq_index", "source_row", "token_offset", "nll_sum", "n_scored"}
CAL_SUMMARY = {"eval_id", "split", "n_sequences", "N", "nll_sum", "mean_nll", "batch_size", "file"}
DIST = {"n", "n_zero", "quantiles", "mean", "log10_hist"}
MANIFEST = {"eval_id", "condition", "family", "kind", "model", "model_dtype", "conventions", "matrices",
            "extra_tensors", "logical_bytes", "tensors", "stored_tensor_bytes", "file"}


def _env():
    return dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")


def _run(args):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True,
                          timeout=900)


@pytest.fixture(scope="module")
def runs(tiny_compress_dir, tmp_path_factory):
    base = tmp_path_factory.mktemp("compress_e2e")
    test_args = ["--model-path", str(tiny_compress_dir), "--device", "cpu", "--limit-calib", str(LIM_C)]
    p = _run(["task1.calib_stats", "--out", str(base / "calib"), *test_args])
    assert p.returncode == 0, p.stderr[-4000:]
    for cond in CONDS:
        extra = ["--calib-stats", str(base / "calib")] if cond.startswith("q2") else []
        p = _run(["task1.compress", "--condition", cond, "--out", str(base / cond), "--artifacts",
                  str(base / f"art_{cond}"), *extra, *test_args, "--limit-dev", str(LIM_D)])
        assert p.returncode == 0, (cond, p.stderr[-4000:])
    return base


def test_calib_stats_outputs(runs):
    d = runs / "calib"
    assert sorted(x.name for x in d.iterdir()) == sorted(calib_stats.OUTPUT_FILES.values())
    r = json.loads((d / "run.json").read_text())
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == CALIB_CONFIG
    assert r["config"]["limits"] == {"calib": LIM_C, "dev": None} and r["config"]["test_only"] is True
    assert set(r["timings_s"]) == {"load_s", "capture_s", "search_total_s", "save_s"}
    s = r["steps"]
    assert set(s) == {"restore", "calibration", "block_timings", "search", "preservation", "files", "pristine_after"}
    assert s["calibration"]["n_sequences"] == LIM_C and s["calibration"]["n_tokens"] == LIM_C * C.CALIB_SEQ_LEN
    assert len(s["block_timings"]) == N_LAYERS
    assert set(s["block_timings"][0]) == {"layer", "capture_s", "preservation_s"} | {
        f"search_{g}_s" for g in awq.GROUPS}
    pres = s["preservation"]
    assert len(pres["per_block"]) == N_LAYERS * len(awq.SEARCH_BITS)
    assert pres["max_over_blocks"]["max_abs_err_over_max_abs_out"] < 1e-2  # FP16 folding on CPU
    assert set(s["search"]["chosen_alpha"]) == {str(b) for b in awq.SEARCH_BITS}
    for b in awq.SEARCH_BITS:
        for g in awq.GROUPS:
            assert all(a in awq.ALPHA_GRID for a in s["search"]["chosen_alpha"][str(b)][g])
    for k in ("dense_stats", "awq_scales"):
        assert s["files"][k]["sha256"] == sha256_file(d / calib_stats.OUTPUT_FILES[k])
    srch = json.loads((d / "awq_search.json").read_text())
    assert set(srch) == {"eval_id", "alpha_grid", "bits", "groups", "convention", "geometry", "search_matmul_dtype",
                         "n_sequences", "n_tokens", "scales_file", "entries"}
    assert len(srch["entries"]) == N_LAYERS * 4 * len(awq.SEARCH_BITS)
    for e in srch["entries"]:
        assert set(e) == {"layer", "group", "bits", "alpha", "alpha_index", "losses", "loss_ref", "rel_loss_chosen",
                          "rel_loss_alpha0", "scale_tensor"}
        assert len(e["losses"]) == 20 and e["losses"][e["alpha_index"]] == min(e["losses"])
    sm = json.loads((d / "awq_scales.manifest.json").read_text())
    assert len(sm["tensors"]) == N_LAYERS * 4 * len(awq.SEARCH_BITS)
    assert sm["tensors"]["layer0.o.b4"]["shape"] == [128] and sm["tensors"]["layer0.down.b4"]["shape"] == [512]
    dm = json.loads((d / "dense_stats.manifest.json").read_text())
    assert len(dm["tensors"]) == 2 * N_MATRICES and dm["n_tokens"] == LIM_C * C.CALIB_SEQ_LEN
    for man in (sm, dm):
        assert {"tensors", "file", "stored_tensor_bytes", "eval_id"} <= set(man)
        assert all(set(t) == {"shape", "dtype", "sha256", "nbytes"} for t in man["tensors"].values())


@pytest.mark.parametrize("cond", CONDS)
def test_compress_outputs(runs, cond):
    d = runs / cond
    fam = C.CONDITIONS[cond]["family"]
    assert sorted(x.name for x in d.iterdir()) == sorted(compress.OUTPUT_FILES.values())
    r = json.loads((d / "run.json").read_text())
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == COMPRESS_CONFIG
    assert r["config"]["condition"] == cond and r["config"]["limits"] == {"calib": LIM_C, "dev": LIM_D}
    assert r["model"]["parameter_dtypes"] == ["torch.float16"]
    prune_like = fam in ("P1", "P2", "P2_2of4")
    assert set(r["timings_s"]) == COMPRESS_TIMINGS | ({"build_distributions_s"} if prune_like else set())
    s = r["steps"]
    assert set(s) == {"restore", "build", "storage", "calib_nll", "artifact", "peak_gpu_allocated_bytes",
                      "held_out_guard"} | ({"awq_source"} if fam == "Q2" else set())
    assert s["held_out_guard"]["attempts"] == [] and len(s["held_out_guard"]["forbidden"]) == 3
    b = s["build"]
    assert set(b) == BUILD | ({"mask_zeros_total", "target_zeros_total"} if prune_like else set())
    nz = b["nonzeros"]
    assert len(nz["per_projection"]) == N_MATRICES and nz["global"] == sum(nz["per_projection"].values())
    if prune_like:
        assert b["mask_zeros_total"] == b["target_zeros_total"] == nz["eligible_weights"] // 2
        assert nz["nonzero_fraction"] <= 0.5
        sd = b["score_distributions"]
        assert len(sd["magnitude_log10_edges"]) == len(sd["wanda_log10_edges"]) == P.N_BINS + 1
        assert sd["wanda_norm_source"] == ("dense" if fam == "P1" else "sequential")
    else:
        assert b["score_distributions"] is None
    if cond == "fp16":
        assert b["output_error"]["pooled_rel"] == 0.0 and nz["nonzero_fraction"] > 0.99
    else:
        assert 0 < b["output_error"]["pooled_rel"] < 1
    st = s["storage"]
    q = fam in ("Q1", "Q2")
    assert set(st) == STORAGE | ({"packed_quant"} if q else set())
    assert st["sparse16"]["nonzeros"] == nz["global"]
    if q:
        bits = C.CONDITIONS[cond]["bits"]
        assert st["nominal_bits_per_eligible_weight"] == bits and st["packed_quant"]["bits"] == bits
        assert st["packed_quant"]["total"] == s["artifact"]["logical_bytes"]["eligible"]
    assert set(s["calib_nll"]) == {"c4_calibration", "c4_development"}
    for split, summ in s["calib_nll"].items():
        assert set(summ) == CAL_SUMMARY and summ["n_sequences"] == (LIM_C if split == "c4_calibration" else LIM_D)
        rows = read_jsonl(d / summ["file"])
        assert len(rows) == summ["n_sequences"] and all(set(x) == CAL_ROW for x in rows)
    a = s["artifact"]
    assert set(a) == ARTIFACT and a["reload_matches_executed"] is True
    assert a["executed_tensors_checked"] == N_MATRICES + (3 * N_LAYERS if fam == "Q2" else 0)
    art = runs / f"art_{cond}"
    assert sorted(x.name for x in art.iterdir()) == ["artifact.safetensors", "manifest.json"]
    assert a["file"]["bytes"] == (art / "artifact.safetensors").stat().st_size
    assert a["file"]["sha256"] == sha256_file(art / "artifact.safetensors")
    man = json.loads((art / "manifest.json").read_text())
    assert set(man) == MANIFEST and man["condition"] == cond and len(man["matrices"]) == N_MATRICES
    assert man["logical_bytes"] == a["logical_bytes"]
    shapes = [tuple(m["shape"]) for m in man["matrices"]]
    if prune_like:
        assert a["logical_bytes"]["eligible"] == storage.sparse16_bytes(shapes, [m["nonzeros"] for m in man["matrices"]])
        assert a["n_tensors"] == N_MATRICES
    if fam == "Q2":
        assert len(a["extra_tensors"]) == 3 * N_LAYERS and s["awq_source"]["bits"] == 3
        assert s["awq_source"]["awq_scales_sha256"] == sha256_file(runs / "calib" / "awq_scales.safetensors")
    rows = read_jsonl(d / "projections.jsonl.gz")
    assert len(rows) == N_MATRICES and all(set(x) == ROW_KEYS[cond] for x in rows)
    assert {x["qualname"]: x["nonzeros"] for x in rows} == nz["per_projection"]
    if prune_like:
        for x in rows:
            assert set(x["dist_magnitude"]) == set(x["dist_wanda"]) == DIST
            assert len(x["dist_wanda"]["quantiles"]) == len(P.QUANTILES)
            assert len(x["dist_wanda"]["log10_hist"]) == P.N_BINS
    for p in d.glob("*.gz"):
        assert p.read_bytes()[4:8] == b"\x00\x00\x00\x00"


def test_allocation_conditions_refused(tiny_compress_dir, tmp_path):
    for cond in ("p3_s50", "q3_int4"):
        assert cond in C.CONDITIONS
        p = _run(["task1.compress", "--condition", cond, "--out", str(tmp_path / cond), "--artifacts",
                  str(tmp_path / f"art_{cond}"), "--model-path", str(tiny_compress_dir), "--device", "cpu"])
        assert p.returncode != 0 and "requires an allocation file" in p.stderr
        assert not (tmp_path / cond).exists()


def test_argument_refusals(tmp_path):
    with pytest.raises(SystemExit, match="requires --calib-stats"):
        compress.main(["--condition", "q2_b4", "--out", str(tmp_path / "a"), "--artifacts", str(tmp_path / "b")])
    with pytest.raises(SystemExit, match="only used by Q2"):
        compress.main(["--condition", "p1_s50", "--out", str(tmp_path / "a"), "--artifacts", str(tmp_path / "b"),
                       "--calib-stats", str(tmp_path)])
    with pytest.raises(SystemExit, match="unknown condition"):
        compress.main(["--condition", "p9", "--out", str(tmp_path / "a"), "--artifacts", str(tmp_path / "b")])
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        compress.main(["--condition", "p1_s50", "--out", str(tmp_path / "exists"), "--artifacts", str(tmp_path / "b")])
    real = RESULTS_DIR / "task1" / "refusal_check"
    with pytest.raises(SystemExit, match="refusing"):
        compress.main(["--condition", "p1_s50", "--out", str(real), "--artifacts", str(tmp_path / "b"),
                       "--limit-calib", "2"])
    with pytest.raises(SystemExit, match="refusing"):
        calib_stats.main(["--out", str(real), "--model-path", str(tmp_path)])
    assert not real.exists()


def test_q2_refuses_mismatched_calib_stats(runs, tiny_compress_dir, tmp_path):
    p = _run(["task1.compress", "--condition", "q2_b4", "--out", str(tmp_path / "q"), "--artifacts",
              str(tmp_path / "art"), "--calib-stats", str(runs / "calib"), "--model-path", str(tiny_compress_dir),
              "--device", "cpu", "--limit-calib", "1", "--limit-dev", "1"])
    assert p.returncode != 0 and "different model source or calibration limit" in p.stderr
