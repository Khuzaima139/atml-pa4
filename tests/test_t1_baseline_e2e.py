"""End-to-end CPU run of python -m task1.baseline with a tiny random model and limited locked data."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common.io import read_jsonl
from common.paths import REPO_ROOT, RESULTS_DIR
from task1 import baseline
from task1 import config as C

LIM_W, LIM_H, LIM_C = 3, 6, 2

TOP_KEYS = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "model",
            "tokenizer", "eval_id", "timings_s", "steps"}
CONFIG_KEYS = {"condition", "condition_spec", "settings", "device", "model_path", "limits", "limited",
               "hellaswag_scope", "output_files", "data_files"}
STEP_KEYS = {"enumeration", "perplexity", "hellaswag", "calib_nll", "lmeval_verify", "resources", "storage",
             "pristine_after"}
TIMING_KEYS = {"load_s", "enumeration_s", "ppl_wikitext2_test_s", "ppl_c4_validation_s", "hellaswag_s",
               "calib_nll_c4_calibration_s", "calib_nll_c4_development_s", "lmeval_verify_s", "resources_s"}
PPL_SUMMARY = {"eval_id", "suite", "n_windows", "N", "nll_sum", "mean_nll", "ppl", "batch_size", "file"}
PPL_ROW = {"eval_id", "suite", "window_index", "source", "source_row", "token_offset", "nll_sum", "n_scored"}
HS_SUMMARY = {"eval_id", "scope", "n", "acc", "acc_norm", "n_correct_acc", "n_correct_acc_norm",
              "n_context_not_prefix", "orders_first_last", "task", "prompt_format", "batch_size", "file"}
HS_ROW = {"eval_id", "order", "source_row_index", "label", "loglik", "cont_chars", "cont_tokens", "ctx_tokens",
          "pred_acc", "pred_acc_norm", "correct_acc", "correct_acc_norm"}
CAL_SUMMARY = {"eval_id", "split", "n_sequences", "N", "nll_sum", "mean_nll", "batch_size", "file"}
CAL_ROW = {"eval_id", "split", "seq_index", "source_row", "token_offset", "nll_sum", "n_scored"}
VERIFY_RUN = {"n_examples", "n_candidates", "max_abs_diff", "mean_abs_diff", "argmax_disagreements_acc",
              "argmax_disagreements_acc_norm", "lmeval_acc", "lmeval_acc_norm", "ours_acc", "ours_acc_norm",
              "wall_s"}
MEAS = {"what", "unit", "repetitions", "warmup", "times_ms", "median_ms", "min_ms", "max_ms",
        "peak_allocated_bytes", "peak_reserved_bytes", "peak_allocated_bytes_max", "peak_reserved_bytes_max"}


@pytest.fixture(scope="module")
def run_dir(tiny_model_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e") / "fp16"
    cmd = [sys.executable, "-m", "task1.baseline", "--model-path", str(tiny_model_dir), "--device", "cpu",
           "--limit-windows", str(LIM_W), "--limit-hellaswag", str(LIM_H), "--limit-calib", str(LIM_C),
           "--out", str(out)]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return out


def test_output_files(run_dir):
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(baseline.OUTPUT_FILES.values())


def test_run_json_keys(run_dir):
    r = json.loads((run_dir / "run.json").read_text())
    assert set(r) == TOP_KEYS
    assert r["status"] == "ok" and r["eval_id"] == C.EVAL_ID
    assert set(r["config"]) == CONFIG_KEYS
    assert r["config"]["limits"] == {"windows": LIM_W, "hellaswag": LIM_H, "calib": LIM_C}
    assert r["config"]["limited"] is True and r["config"]["condition"] == "fp16"
    assert r["config"]["hellaswag_scope"] == "full"
    assert r["config"]["settings"] == C.settings_dict()
    assert len(r["config"]["data_files"]) == 5
    assert {"git_commit", "git_dirty", "package_versions", "torch_version"} <= set(r["meta"])
    assert r["meta"]["package_versions"]["lm_eval"] == "0.4.13"
    assert r["model"]["parameter_dtypes"] == ["torch.float16"]
    assert r["model"]["attn_implementation_effective"] == "sdpa"
    assert set(r["tokenizer"]) == {"class", "name_or_path", "len", "bos_token_id", "eos_token_id", "pad_token_id"}
    assert set(r["timings_s"]) == TIMING_KEYS
    s = r["steps"]
    assert set(s) == STEP_KEYS

    en = s["enumeration"]
    assert set(en) == {"n_eligible_matrices", "first", "last", "per_block", "param_counts",
                       "pristine_tensors_verified"}
    assert en["n_eligible_matrices"] == 14 and en["per_block"] == 7
    assert set(en["param_counts"]) == {"eligible", "embedding", "norms", "biases", "other", "total",
                                       "non_eligible", "lm_head_tied", "eligible_matrices",
                                       "eligible_exact_zeros", "other_names"}

    assert set(s["perplexity"]) == {"wikitext2_test", "c4_validation"}
    for suite, summ in s["perplexity"].items():
        assert set(summ) == PPL_SUMMARY and summ["n_windows"] == LIM_W and summ["suite"] == suite
        assert summ["file"] == baseline.OUTPUT_FILES[f"ppl_{suite}"]

    assert set(s["hellaswag"]) == HS_SUMMARY and s["hellaswag"]["n"] == LIM_H
    assert s["hellaswag"]["task"]["version"] == "1.0"
    assert s["hellaswag"]["file"] == baseline.OUTPUT_FILES["hellaswag"]

    assert set(s["calib_nll"]) == {"c4_calibration", "c4_development"}
    for split, summ in s["calib_nll"].items():
        assert set(summ) == CAL_SUMMARY and summ["N"] == LIM_C * 511

    lv = s["lmeval_verify"]
    assert set(lv) == {"file", "n_examples", "runs"} and lv["n_examples"] == LIM_H
    assert set(lv["runs"]) == {"softmax_dtype=default", "softmax_dtype=float32"}
    for run in lv["runs"].values():
        assert set(run) == VERIFY_RUN and run["n_candidates"] == 4 * LIM_H
        assert run["max_abs_diff"] < 0.5  # FP16 model on CPU; agreement is checked at FP32 in unit tests

    res = s["resources"]
    assert set(res) == {"execution", "device", "timer", "batch_size", "eval_id", "device_name",
                        "attn_implementation", "prompts", "measurements"}
    assert res["execution"]["mode"] == "dense_fp16" and res["timer"] == "perf_counter"
    assert set(res["prompts"]) == {"512", "2048"} and len(res["prompts"]["2048"]["source_rows"]) == 4
    m = res["measurements"]
    assert set(m) == {"prefill_512", "prefill_2048", "decode_64_after_prefill_512"}
    for k in ("prefill_512", "prefill_2048"):
        assert set(m[k]) == MEAS and m[k]["repetitions"] == C.LATENCY_REPS
    assert set(m["decode_64_after_prefill_512"]) == MEAS | {"median_ms_per_token", "decoded_token_ids"}
    assert len(m["decode_64_after_prefill_512"]["decoded_token_ids"]) == 64

    st = s["storage"]
    assert set(st) == {"convention", "eval_id", "eligible_weights", "eligible_matrices", "non_eligible_params",
                       "fp16_dense", "sparse16"}
    assert st["eligible_weights"] == en["param_counts"]["eligible"]
    assert st["fp16_dense"]["eligible_bytes"] == 2 * st["eligible_weights"]
    assert st["sparse16"]["nonzeros"] == st["eligible_weights"] - en["param_counts"]["eligible_exact_zeros"]
    assert s["pristine_after"]["tensors_verified"] == en["pristine_tensors_verified"]


def test_per_example_files(run_dir):
    r = json.loads((run_dir / "run.json").read_text())
    for suite in ("wikitext2_test", "c4_validation"):
        rows = read_jsonl(run_dir / f"ppl_{suite}.jsonl.gz")
        assert len(rows) == LIM_W and all(set(x) == PPL_ROW for x in rows)
        assert [x["window_index"] for x in rows] == list(range(LIM_W))
        assert sum(x["n_scored"] for x in rows) == r["steps"]["perplexity"][suite]["N"]
    hs = read_jsonl(run_dir / "hellaswag_full.jsonl.gz")
    assert len(hs) == LIM_H and all(set(x) == HS_ROW for x in hs)
    assert [x["order"] for x in hs] == list(range(LIM_H))
    for split in ("c4_calibration", "c4_development"):
        rows = read_jsonl(run_dir / f"calib_nll_{split}.jsonl.gz")
        assert len(rows) == LIM_C and all(set(x) == CAL_ROW for x in rows)
    rep = json.loads((run_dir / "lmeval_verify.json").read_text())
    assert set(rep) == {"eval_id", "lm_eval_version", "batch_size", "prompt_format", "orders", "runs"}
    assert rep["orders"] == list(range(LIM_H))
    for run in rep["runs"].values():
        assert set(run) == VERIFY_RUN | {"per_example"} and len(run["per_example"]) == LIM_H
    for p in run_dir.glob("*.gz"):
        assert p.read_bytes()[4:8] == b"\x00\x00\x00\x00"  # gzip mtime 0


def test_refuses_existing_out(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        baseline.main(["--out", str(out), "--limit-windows", "2"])


def test_refuses_limits_under_results(tmp_path):
    out = RESULTS_DIR / "task1" / "fp16_limited_refusal_check"
    with pytest.raises(SystemExit, match="refusing limits"):
        baseline.main(["--out", str(out), "--limit-hellaswag", "3"])
    assert not out.exists()
