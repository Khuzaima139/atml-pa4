"""End-to-end CPU runs of python -m task2.part_a verify and bench with limited lengths and repetitions."""
import json
import os
import subprocess
import sys

import pytest

from common.io import read_jsonl
from common.paths import REPO_ROOT, RESULTS_DIR
from task2 import part_a

VERIFY_LENGTHS = (64, 100)
BENCH_LENGTHS = (32, 48)
WARM, TIMED = 1, 2

TOP_KEYS = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "environment",
            "status_counts", "implementation_orders", "n_rows", "step"}
CONFIG_COMMON = {"version", "mode", "device", "B", "H", "d", "seed", "lengths", "dtypes", "limits", "limited",
                 "projection_limit_gib", "output_files", "reference", "reference_inputs", "secondary_error_prefix",
                 "score_scaling", "input_generation", "impls"}
CONFIG_VERIFY = CONFIG_COMMON | {"verify_tiles", "edge_L", "edge_causality_t", "extreme_targets", "masked_row"}
CONFIG_BENCH = CONFIG_COMMON | {"bench_tile", "passes", "warmup", "repetitions", "forward_pass_note", "order_rule"}
ENV_KEYS = {"device", "env", "matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision",
            "matmul_allow_fp16_reduced_precision_reduction", "sdpa_backends_enumerated", "cpu_threads",
            "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled", "gpu_name",
            "gpu_capability", "gpu_total_memory_bytes"}
META_KEYS = {"git_commit", "git_dirty", "torch_version", "package_versions", "device_names", "cuda_available"}
SPEC_KEYS = {"impl", "label", "tile", "backend_forced"}
ERR_KEYS = {f"{n}_{s}" for n in ("out", "dq", "dk", "dv") for s in ("max_abs_err", "mean_abs_err", "finite")}
SEC_KEYS = {f"vs_fp32_inputs_{n}_{s}" for n in ("out", "dq", "dk", "dv") for s in ("max_abs_err", "mean_abs_err")}
ACC_KEYS = {"version", "kind", "L", "dtype", "status", "error", "wall_s", "peak_allocated_bytes",
            "peak_reserved_bytes", "reference_inputs"} | SPEC_KEYS
REF_KEYS = {"version", "kind", "L", "dtype", "for_rows_dtype", "reference_inputs", "status", "error",
            "dense_projected_peak_bytes", "peak_allocated_bytes", "peak_reserved_bytes"}
CAUSAL_KEYS = {"version", "kind", "L", "dtype", "t", "status", "bitwise_equal_upto_t", "max_abs_change_upto_t",
               "max_abs_change_after_t", "required_bitwise"} | SPEC_KEYS
EXTREME_KEYS = {"version", "kind", "L", "dtype", "status", "error", "target_max_abs_score", "q_scale_factor",
                "achieved_max_abs_score", "q_cast_finite", "reference_inputs"} | SPEC_KEYS
MASKED_KEYS = {"version", "kind", "L", "dtype", "status", "error", "masked_row", "reference_inputs"} | SPEC_KEYS
MASKED_OK_KEYS = {"row_out_max_abs", "row_out_nan_count", "row_dq_max_abs", "row_dq_nan_count", "out_all_finite",
                  "grads_all_finite", "other_rows_max_abs_err", "matches_documented_zero_row"}
BENCH_KEYS = {"version", "kind", "L", "dtype", "pass", "order_position", "status", "error",
              "baseline_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "timer", "warmup",
              "repetitions", "unit"} | SPEC_KEYS
TIME_KEYS = {"times_ms", "median_ms", "mean_ms", "min_ms", "max_ms"}
TABLE_KEYS = {"version", "L", "dtype", "pass", "default", "default_error", "forced", "forced_error"}
SCORE_KEYS = {"L", "dtype", "B", "H", "d", "score_matrix_bytes", "causal_mask_bytes", "dense_projected_peak_bytes",
              "dense_projected_peak_gib", "dense_attempted", "tiled_block_bytes_fp32_128x128", "projection_formula"}


def _run(args, out):
    cmd = [sys.executable, "-m", "task2.part_a", *args, "--device", "cpu", "--out", str(out)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return out


@pytest.fixture(scope="module")
def verify_dir(tmp_path_factory):
    return _run(["verify", "--limit-lengths", ",".join(map(str, VERIFY_LENGTHS))],
                tmp_path_factory.mktemp("t2a") / "verify")


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory):
    return _run(["bench", "--limit-lengths", ",".join(map(str, BENCH_LENGTHS)), "--limit-reps", f"{WARM},{TIMED}"],
                tmp_path_factory.mktemp("t2a") / "bench")


def _check_run_json(path, mode, lengths):
    run = json.loads((path / "run.json").read_text())
    assert set(run) == TOP_KEYS
    assert run["status"] == "ok" and run["step"] == "done"
    assert set(run["config"]) == (CONFIG_VERIFY if mode == "verify" else CONFIG_BENCH)
    assert run["config"]["version"] == part_a.VERSION and run["config"]["mode"] == mode
    assert run["config"]["lengths"] == list(lengths) and run["config"]["limited"] is True
    assert set(run["environment"]) == ENV_KEYS
    assert META_KEYS <= set(run["meta"])
    assert run["environment"]["matmul_allow_tf32"] is False
    assert set(os.listdir(path)) == set(part_a.OUTPUT_FILES[mode].values())
    return run


def _check_score_bytes(path, lengths):
    sb = json.loads((path / "score_bytes.json").read_text())
    assert sb["version"] == part_a.VERSION
    assert len(sb["rows"]) == 2 * len(lengths)
    for r in sb["rows"]:
        assert set(r) == SCORE_KEYS
        assert r["score_matrix_bytes"] == part_a.B * part_a.H * r["L"] ** 2 * part_a.ELEM_BYTES[r["dtype"]]


def test_verify_outputs(verify_dir):
    run = _check_run_json(verify_dir, "verify", VERIFY_LENGTHS)
    _check_score_bytes(verify_dir, VERIFY_LENGTHS)
    rows = read_jsonl(verify_dir / "verify_rows.jsonl")
    assert run["n_rows"] == len(rows)
    assert all(r["version"] == part_a.VERSION for r in rows)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"reference", "accuracy", "edge_causality", "edge_extreme", "edge_masked_row"}
    n_specs = len(part_a.verify_specs())
    assert run["config"]["edge_L"] == 100

    refs = [r for r in rows if r["kind"] == "reference"]
    assert [(r["L"], r["for_rows_dtype"]) for r in refs] == [(L, d) for L in VERIFY_LENGTHS for d in ("fp32", "fp16")]
    assert all(set(r) == REF_KEYS and r["status"] == "ok" and r["dtype"] == "fp32" for r in refs)
    assert all(r["reference_inputs"] == part_a.REFERENCE_INPUTS[r["for_rows_dtype"]] for r in refs)

    acc = [r for r in rows if r["kind"] == "accuracy"]
    assert len(acc) == len(VERIFY_LENGTHS) * 2 * n_specs
    for r in acc:
        assert r["status"] in part_a.STATUSES
        assert r["reference_inputs"] == part_a.REFERENCE_INPUTS[r["dtype"]]
        if r["status"] == "ok":
            assert set(r) == ACC_KEYS | ERR_KEYS | (SEC_KEYS if r["dtype"] == "fp16" else set())
            assert r["out_finite"] and r["dq_finite"] and r["dk_finite"] and r["dv_finite"]
            if r["dtype"] == "fp32":
                assert r["out_max_abs_err"] < 1e-5 and r["dq_max_abs_err"] < 1e-5
            else:
                assert r["out_max_abs_err"] < 1e-2
        else:
            assert r["status"] == "unsupported" and r["impl"] == "sdpa" and r["error"]
            assert set(r) == ACC_KEYS
    labels_ok = {r["label"] for r in acc if r["status"] == "ok"}
    assert {"dense", "tiled_32x32", "tiled_64x64", "tiled_128x128", "tiled_64x128", "sdpa_default",
            "sdpa_MATH"} <= labels_ok
    dense32 = [r for r in acc if r["label"] == "dense" and r["dtype"] == "fp32"]
    assert all(r["out_max_abs_err"] == 0 for r in dense32)
    acc16 = [r for r in acc if r["dtype"] == "fp16" and r["status"] == "ok"]
    assert any(r["out_max_abs_err"] != r["vs_fp32_inputs_out_max_abs_err"] for r in acc16)
    assert all(r["vs_fp32_inputs_out_max_abs_err"] < 1e-2 for r in acc16)

    caus = [r for r in rows if r["kind"] == "edge_causality"]
    assert caus and all(set(r) == CAUSAL_KEYS for r in caus)
    assert {r["t"] for r in caus} == set(part_a.edge_ts(100))
    assert all(r["bitwise_equal_upto_t"] for r in caus if r["required_bitwise"])
    assert all(r["max_abs_change_after_t"] > 0 for r in caus)

    ext = [r for r in rows if r["kind"] == "edge_extreme"]
    assert {r["target_max_abs_score"] for r in ext} == set(part_a.EXTREME_TARGETS)
    for r in ext:
        assert r["reference_inputs"] == part_a.REFERENCE_INPUTS[r["dtype"]]
        if r["status"] == "ok":
            assert set(r) == EXTREME_KEYS | ERR_KEYS | (SEC_KEYS if r["dtype"] == "fp16" else set())
            assert r["out_finite"] and r["dq_finite"]
        else:
            assert set(r) == EXTREME_KEYS

    masked = [r for r in rows if r["kind"] == "edge_masked_row"]
    assert len(masked) == 2 * n_specs
    for r in masked:
        if r["status"] == "ok":
            extra = {"vs_fp32_inputs_other_rows_max_abs_err"} if r["dtype"] == "fp16" else set()
            assert set(r) == MASKED_KEYS | MASKED_OK_KEYS | extra
            if r["impl"] in ("dense", "tiled"):
                assert r["matches_documented_zero_row"] is True
        else:
            assert set(r) == MASKED_KEYS and r["impl"] == "sdpa"


def test_bench_outputs(bench_dir):
    run = _check_run_json(bench_dir, "bench", BENCH_LENGTHS)
    _check_score_bytes(bench_dir, BENCH_LENGTHS)
    rows = read_jsonl(bench_dir / "bench_rows.jsonl")
    n_specs = len(part_a.bench_specs())
    assert len(rows) == run["n_rows"] == len(BENCH_LENGTHS) * n_specs * 2 * 2
    for r in rows:
        assert r["status"] in part_a.STATUSES
        if r["status"] == "ok":
            assert set(r) == BENCH_KEYS | TIME_KEYS
            assert len(r["times_ms"]) == TIMED and r["warmup"] == WARM
            assert r["min_ms"] <= r["median_ms"] <= r["max_ms"]
        else:
            assert r["status"] == "unsupported" and r["error"] and set(r) == BENCH_KEYS
    labels = [s["label"] for s in part_a.bench_specs()]
    orders = run["implementation_orders"]
    assert orders[str(BENCH_LENGTHS[0])] == labels
    assert orders[str(BENCH_LENGTHS[1])] == labels[1:] + labels[:1]
    for L in BENCH_LENGTHS:
        for r in rows:
            if r["L"] == L:
                assert orders[str(L)][r["order_position"]] == r["label"]
    table = read_jsonl(bench_dir / "backend_table.jsonl")
    assert len(table) == len(BENCH_LENGTHS) * 2 * 2
    for t in table:
        assert set(t) == TABLE_KEYS
        assert set(t["default"]) == {"aten_ops", "cuda_kernels", "inferred_backend"}
        assert set(t["forced"]) == set(run["environment"]["sdpa_backends_enumerated"])


def test_reference_inputs_per_dtype():
    inp = part_a.make_inputs(16)
    assert part_a.reference_inputs(inp, "fp32") is inp
    r16 = part_a.reference_inputs(inp, "fp16")
    for n in ("q", "k", "v", "g"):
        assert r16[n].dtype == part_a.torch.float32
        assert part_a.torch.equal(r16[n], inp[n].half().float())
        assert not part_a.torch.equal(r16[n], inp[n])
    for n in ("pk", "pv"):
        assert r16[n] is inp[n]


def test_fp16_primary_reference_excludes_input_rounding(tmp_path, monkeypatch):
    """An implementation that computes FP32 dense attention on its FP16 inputs and rounds the result to FP16: its
    primary output error equals the output rounding against the FP16-cast reference, and its vs_fp32_inputs_ error
    equals the distance of the same output from the FP32-input reference."""
    real = part_a.run_spec
    torch = part_a.torch

    def exact_on_inputs(spec, q, k, v, mask=None):
        if spec["label"] != "tiled_32x32" or q.dtype != torch.float16:
            return real(spec, q, k, v, mask=mask)
        return part_a.ops.dense_causal(q.float(), k.float(), v.float(), mask=mask).half()

    monkeypatch.setattr(part_a, "run_spec", exact_on_inputs)
    out = tmp_path / "exact"
    assert part_a.main(["verify", "--device", "cpu", "--out", str(out), "--limit-lengths", "64"]) == 0
    rows = [r for r in read_jsonl(out / "verify_rows.jsonl")
            if r["kind"] == "accuracy" and r["label"] == "tiled_32x32" and r["dtype"] == "fp16"]
    assert len(rows) == 1
    r = rows[0]
    inp = part_a.make_inputs(64)
    ref16 = part_a.fwd_bwd(part_a.spec_dense(), part_a.reference_inputs(inp, "fp16"), torch.float32, "cpu")[0]
    ref32 = part_a.fwd_bwd(part_a.spec_dense(), inp, torch.float32, "cpu")[0]
    mine = ref16.half().float()
    assert r["out_max_abs_err"] == float((mine - ref16).abs().max())
    assert r["vs_fp32_inputs_out_max_abs_err"] == float((mine - ref32).abs().max())
    assert r["vs_fp32_inputs_out_mean_abs_err"] > r["out_mean_abs_err"]


def test_skipped_projection_rows(tmp_path):
    out = tmp_path / "skip"
    assert part_a.main(["verify", "--device", "cpu", "--out", str(out), "--limit-lengths", "16",
                        "--limit-projection-gib", "1e-9"]) == 0
    rows = read_jsonl(out / "verify_rows.jsonl")
    ref = [r for r in rows if r["kind"] == "reference"]
    assert ref[0]["status"] == "skipped_projected" and ref[0]["error"].startswith("skipped: projected")
    acc = [r for r in rows if r["kind"] == "accuracy"]
    assert acc and all(r["status"] == "no_reference" for r in acc)


def test_refuses_existing_out(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        part_a.main(["verify", "--device", "cpu", "--out", str(out), "--limit-lengths", "16"])


@pytest.mark.parametrize("flag", [["--limit-lengths", "16"], ["--limit-reps", "1,1"],
                                  ["--limit-projection-gib", "1"]])
def test_refuses_limits_on_real_result_path(flag):
    out = RESULTS_DIR / "task2" / "part_a_limit_refusal_check"
    with pytest.raises(SystemExit, match="refusing"):
        part_a.main(["bench", "--device", "cpu", "--out", str(out), *flag])
    assert not out.exists()


def test_oom_rows(tmp_path, monkeypatch):
    real = part_a.run_spec

    def fake(spec, q, k, v, mask=None):
        if spec["label"] in ("dense", "tiled_128x128"):
            raise part_a.torch.cuda.OutOfMemoryError("simulated out of memory")
        return real(spec, q, k, v, mask=mask)

    monkeypatch.setattr(part_a, "run_spec", fake)
    out = tmp_path / "oom"
    part_a.main(["bench", "--device", "cpu", "--out", str(out), "--limit-lengths", "16", "--limit-reps", "0,1"])
    rows = read_jsonl(out / "bench_rows.jsonl")
    oom = [r for r in rows if r["status"] == "oom"]
    assert {r["label"] for r in oom} == {"dense", "tiled_128x128"}
    assert all(r["error"] == "simulated out of memory" and "times_ms" not in r for r in oom)
    assert any(r["status"] == "ok" and r["label"] == "sdpa_default" for r in rows)
