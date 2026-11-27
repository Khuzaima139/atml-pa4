"""End-to-end CPU runs of python -m task2.train_mqar (every mechanism) and python -m task2.mqar_sweep."""
import json
import os
import subprocess
import sys

import pytest
import torch

from common.io import read_jsonl, sha256_file
from common.paths import REPO_ROOT, RESULTS_DIR
from task2 import mqar_data as D
from task2 import mqar_model as M
from task2 import mqar_sweep as SW
from task2 import train_mqar as TR

UPDATES, EVAL = 9, 2
CHECKPOINTS = [1, 2, 3, 4, 6, 9]
RUNS = {"dense": ("4", "near", "6305"), "window": ("4", "near", "6305"), "local_dilated": ("8", "mixed", "6304"),
        "linear": ("4", "near", "6305")}

TOP_KEYS = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "environment",
            "step", "generator", "dev_file", "stream", "dev_curve", "updates_done", "train_wall_s", "learned",
            "test_summary", "test_pooled_bins", "test_opened_after_update", "train_wall_note", "mean_ms_per_update",
            "eval_wall_s", "peak_allocated_bytes_train", "peak_allocated_bytes_eval", "parameter_count", "n_rows",
            "checkpoint_sha256", "dev_eval_sequences_per_cell", "test_eval_sequences_per_cell"}
CONFIG_KEYS = {"version", "scorer", "mechanism", "K", "regime", "seed", "device", "dtype", "cache_dir", "limits",
               "limited", "output_files", "updates", "batch", "train_index_range", "optimizer", "schedule",
               "grad_clip_global_norm", "checkpoints", "eval_batch", "distance_bins", "mask_conditions", "model",
               "theoretical_costs", "theoretical_costs_all", "learned_rule"}
STREAM_KEYS = {"built", "rebuild_reason", "build_seconds", "path", "sha256", "K", "regime", "start", "stop",
               "n_sequences", "shape", "cache_version", "spot_checked_indices", "load_seconds", "validation",
               "disjointness"}
LOG_KEYS = {"version", "update", "loss", "accuracy", "lr", "grad_norm", "clipped", "step_ms", "index_range"}
STATS_KEYS = {"n_queries", "accuracy", "loss"}
CURVE_KEYS = {"version", "update", "split", "cell", "K", "regime", "mask_condition", "n_sequences", "bins",
              "residue_mod4"} | STATS_KEYS
LEARNED_KEYS = {"cell", "n_queries", "accuracy", "threshold", "max_distance_exclusive", "learned", "rule"}
BIN_NAMES = {"[0,32)", "[32,64)", "[64,128)", "[128,256)"}


def _run_train(mech, out, cache):
    K, regime, seed = RUNS[mech]
    cmd = [sys.executable, "-m", "task2.train_mqar", "--mechanism", mech, "--K", K, "--regime", regime,
           "--seed", seed, "--out", str(out), "--cache-dir", str(cache), "--device", "cpu",
           "--limit-updates", str(UPDATES), "--limit-eval", str(EVAL)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return out


@pytest.fixture(scope="module")
def train_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("t2b")
    cache = base / "cache"
    return {m: _run_train(m, base / m, cache) for m in M.MECHANISMS}, cache


@pytest.mark.parametrize("mech", M.MECHANISMS)
def test_train_outputs(train_dirs, mech):
    dirs, cache = train_dirs
    out = dirs[mech]
    K, regime, seed = int(RUNS[mech][0]), RUNS[mech][1], int(RUNS[mech][2])
    assert set(os.listdir(out)) == set(TR.OUTPUT_FILES.values())
    run = json.loads((out / "run.json").read_text())
    assert set(run) == TOP_KEYS
    assert run["status"] == "ok" and run["step"] == "done"
    cfg = run["config"]
    assert set(cfg) == CONFIG_KEYS
    assert (cfg["version"], cfg["mechanism"], cfg["K"], cfg["regime"], cfg["seed"]) == (TR.VERSION, mech, K,
                                                                                         regime, seed)
    assert cfg["limited"] is True and cfg["updates"] == UPDATES and cfg["checkpoints"] == CHECKPOINTS
    start = (seed - 6304) * 144_000
    assert cfg["train_index_range"] == [start, start + UPDATES * 32]
    assert cfg["model"]["parameters"] == run["parameter_count"] == 462_336
    assert cfg["mask_conditions"] == (["native", "window64"] if mech == "dense" else ["native"])
    assert cfg["theoretical_costs"] == M.theoretical_costs(mech, 256)
    assert set(cfg["theoretical_costs_all"]) == set(M.MECHANISMS)
    assert {"git_commit", "git_dirty", "torch_version", "package_versions"} <= set(run["meta"])
    assert run["environment"]["matmul_allow_tf32"] is False

    st = run["stream"]
    assert set(st) == STREAM_KEYS
    assert (st["K"], st["regime"], st["start"], st["stop"]) == (K, regime, start, start + UPDATES * 32)
    assert st["sha256"] == sha256_file(st["path"])
    assert st["validation"]["batches_checked"] == UPDATES
    dj = st["disjointness"]
    assert dj["collisions"] == 0 and dj["n_train_hashed"] == UPDATES * 32 and dj["n_reference_hashes"] == 11_520
    assert run["generator"]["sha256"] == run["generator"]["manifest_sha256"]
    assert run["test_opened_after_update"] == UPDATES == run["updates_done"]

    log = read_jsonl(out / "train_log.jsonl")
    assert [r["update"] for r in log] == list(range(1, UPDATES + 1))
    assert all(set(r) == LOG_KEYS and r["version"] == TR.VERSION for r in log)
    assert log[0]["lr"] == pytest.approx(8e-4) and log[-1]["lr"] == pytest.approx(0.15 * 8e-4)
    assert log[0]["index_range"] == [start, start + 32]

    curve = read_jsonl(out / "dev_curve.jsonl")
    assert len(curve) == len(CHECKPOINTS) * 9
    assert [r["update"] for r in curve[::9]] == CHECKPOINTS
    for r in curve:
        assert set(r) == CURVE_KEYS and r["n_sequences"] == EVAL and r["n_queries"] == EVAL * r["K"]
        assert set(r["bins"]) == BIN_NAMES and set(r["residue_mod4"]) == {"0", "1", "2", "3"}
        assert sum(b["n_queries"] for b in r["bins"].values()) == r["n_queries"]
    assert [p["update"] for p in run["dev_curve"]] == CHECKPOINTS

    assert set(run["learned"]) == LEARNED_KEYS and run["learned"]["cell"] == f"{regime}_K{K}"
    conds = cfg["mask_conditions"]
    assert set(run["test_summary"]) == set(conds) == set(run["test_pooled_bins"])
    for cond in conds:
        assert set(run["test_summary"][cond]) == set(D.cell_order())
        assert set(run["test_pooled_bins"][cond]) == BIN_NAMES
    n_q = EVAL * (4 + 8 + 16) * 3
    assert run["n_rows"] == {"train_log": UPDATES, "dev_curve": len(curve), "dev_queries": n_q,
                             "test_queries": n_q * len(conds)}
    for name, split, n in (("dev_queries_final.jsonl.gz", "development", n_q),
                           ("test_queries.jsonl.gz", "test", n_q * len(conds))):
        rows = read_jsonl(out / name)
        assert len(rows) == n
        for r in rows:
            assert set(r) == set(TR.QUERY_ROW_KEYS) and r["split"] == split and r["version"] == TR.VERSION
            assert r["distance"] == r["query_position"] - r["source_position"]
            assert r["correct"] == (r["predicted_id"] == r["answer_id"]) and 128 <= r["predicted_id"] < 192
            assert r["id"].startswith(f"{split}-{r['regime']}-K{r['K']}-")
        assert {r["mask_condition"] for r in rows} == (set(conds) if split == "test" else {"native"})
    assert run["checkpoint_sha256"] == sha256_file(out / "model_final.safetensors")


def test_cache_shared_between_runs(train_dirs):
    dirs, cache = train_dirs
    built = {m: json.loads((dirs[m] / "run.json").read_text())["stream"] for m in ("dense", "window", "linear")}
    assert built["dense"]["built"] is True
    assert built["window"]["built"] is False and built["linear"]["built"] is False
    assert len({b["sha256"] for b in built.values()}) == 1
    assert not any(p.name.endswith(".npy") and ".tmp" in p.name for p in cache.iterdir())


def test_checkpoint_reproduces_recorded_predictions(train_dirs):
    from safetensors.torch import load_file

    out = train_dirs[0]["dense"]
    model = M.MQARModel("dense")
    model.load_state_dict(load_file(str(out / "model_final.safetensors")))
    model.eval()
    rows = read_jsonl(out / "test_queries.jsonl.gz")
    test = D.read_eval_file(D.TEST_FILE, "test", limit_per_cell=EVAL)
    for cond in ("native", "window64"):
        for r in [x for x in rows if x["mask_condition"] == cond and x["cell"] == "mixed_K16"][:20]:
            c = test[r["cell"]]
            n = list(c["ids"]).index(r["id"])
            tok = torch.from_numpy(c["tokens"][n:n + 1]).long()
            with torch.no_grad(), model.mask_override(cond):
                logits = model(tok)[0, r["query_position"], 128:192]
            assert int(logits.argmax()) + 128 == r["predicted_id"]
            assert float(logits[r["answer_id"] - 128]) == pytest.approx(r["answer_logit"], abs=1e-5)


def test_test_file_opened_only_after_training(tmp_path, monkeypatch):
    calls = {"updates": 0, "test_read_at": None}
    real_loss, real_read = TR.query_loss, D.read_eval_file

    def counting_loss(*a, **k):
        calls["updates"] += 1
        return real_loss(*a, **k)

    def watched_read(path, split, **k):
        if split == "test":
            calls["test_read_at"] = calls["updates"]
        return real_read(path, split, **k)

    monkeypatch.setattr(TR, "query_loss", counting_loss)
    monkeypatch.setattr(D, "read_eval_file", watched_read)
    assert TR.main(["--mechanism", "window", "--K", "4", "--regime", "near", "--seed", "6304", "--out",
                    str(tmp_path / "w"), "--cache-dir", str(tmp_path / "c"), "--device", "cpu",
                    "--limit-updates", "3", "--limit-eval", "1"]) == 0
    assert calls["test_read_at"] == 3 == calls["updates"]


def test_train_refuses_existing_out(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        TR.main(["--mechanism", "dense", "--K", "4", "--regime", "near", "--seed", "6304", "--out", str(out),
                 "--device", "cpu", "--limit-updates", "1"])


@pytest.mark.parametrize("flag", [["--limit-updates", "2"], ["--limit-eval", "1"]])
def test_train_refuses_limits_on_real_result_path(flag):
    out = RESULTS_DIR / "task2" / "train_limit_refusal_check"
    with pytest.raises(SystemExit, match="refusing"):
        TR.main(["--mechanism", "dense", "--K", "4", "--regime", "near", "--seed", "6304", "--out", str(out),
                 "--device", "cpu", *flag])
    assert not out.exists()


# ---------------------------------------------------------------------------------------------------------------
# Sweep

SWEEP_LENGTHS = (32, 80)
SWEEP_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "environment",
             "step", "status_counts", "n_rows"}
SWEEP_CONFIG = {"version", "device", "dtype", "B", "H", "head_dim", "d_model", "layers", "seed", "lengths",
                "mechanisms", "warmup", "repetitions", "limits", "limited", "output_files", "input", "model_init",
                "forward_only"}
SWEEP_ROW = {"version", "mechanism", "T", "B", "H", "head_dim", "d_model", "dtype", "materialization", "status",
             "error", "baseline_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "timer", "warmup",
             "repetitions", "unit"}
TIME_KEYS = {"times_ms", "median_ms", "mean_ms", "min_ms", "max_ms"}
MAT_KEYS = {"dense_mask", "dense_mask_bytes", "dense_mask_shape", "linear_prefix", "linear_prefix_bytes_per_layer",
            "linear_prefix_shape", "note"}


def test_sweep_outputs(tmp_path):
    out = tmp_path / "sweep"
    cmd = [sys.executable, "-m", "task2.mqar_sweep", "--out", str(out), "--device", "cpu",
           "--limit-lengths", ",".join(map(str, SWEEP_LENGTHS)), "--limit-reps", "1,2"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert set(os.listdir(out)) == set(SW.OUTPUT_FILES.values())
    run = json.loads((out / "run.json").read_text())
    assert set(run) == SWEEP_TOP and run["status"] == "ok" and set(run["config"]) == SWEEP_CONFIG
    assert run["config"]["lengths"] == list(SWEEP_LENGTHS) and run["config"]["limited"] is True
    rows = read_jsonl(out / "sweep_rows.jsonl")
    assert len(rows) == run["n_rows"] == len(SWEEP_LENGTHS) * 4
    assert [(r["T"], r["mechanism"]) for r in rows] == [(T, m) for T in SWEEP_LENGTHS for m in M.MECHANISMS]
    for r in rows:
        assert set(r) == SWEEP_ROW | TIME_KEYS and r["status"] == "ok" and r["version"] == SW.VERSION
        assert len(r["times_ms"]) == 2 and r["min_ms"] <= r["median_ms"] <= r["max_ms"]
        mat = r["materialization"]
        assert set(mat) == MAT_KEYS
        assert mat == M.materialization(r["mechanism"], 2, r["T"])
    lin = [r for r in rows if r["mechanism"] == "linear" and r["T"] == 80][0]
    assert lin["materialization"]["linear_prefix_bytes_per_layer"] == 2 * 4 * 80 * 32 * 32 * 4


def test_sweep_input_fixed():
    a, b = SW.make_input(32), SW.make_input(32)
    assert torch.equal(a, b) and a.shape == (2, 32, 128) and a.dtype == torch.float32


def test_sweep_oom_and_unsupported_rows(tmp_path, monkeypatch):
    real = M.MQARModel.forward_embeddings

    def fake(self, x):
        if self.mechanism == "window":
            raise torch.cuda.OutOfMemoryError("simulated out of memory")
        if self.mechanism == "local_dilated":
            raise RuntimeError("simulated unsupported operator")
        return real(self, x)

    monkeypatch.setattr(M.MQARModel, "forward_embeddings", fake)
    out = tmp_path / "s"
    assert SW.main(["--out", str(out), "--device", "cpu", "--limit-lengths", "16", "--limit-reps", "0,1"]) == 0
    rows = {r["mechanism"]: r for r in read_jsonl(out / "sweep_rows.jsonl")}
    assert rows["window"]["status"] == "oom" and rows["window"]["error"] == "simulated out of memory"
    assert rows["local_dilated"]["status"] == "unsupported" and "simulated" in rows["local_dilated"]["error"]
    assert rows["dense"]["status"] == rows["linear"]["status"] == "ok"
    assert "times_ms" not in rows["window"] and "times_ms" in rows["dense"]


def test_sweep_refusals(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        SW.main(["--out", str(out), "--device", "cpu", "--limit-lengths", "16"])
    real = RESULTS_DIR / "task2" / "sweep_limit_refusal_check"
    for flag in (["--limit-lengths", "16"], ["--limit-reps", "1,1"]):
        with pytest.raises(SystemExit, match="refusing"):
            SW.main(["--out", str(real), "--device", "cpu", *flag])
    assert not real.exists()
