"""End-to-end CPU run of python -m task1.smoke, and the run-1 projection computed from its output."""
import copy
import json
import os
import subprocess
import sys

import pytest

from common.paths import REPO_ROOT, RESULTS_DIR
from task1 import config as C
from task1 import kaggle_run1 as K
from task1 import smoke

UNIT = {"wall_s", "peak_allocated_bytes", "peak_reserved_bytes", "host_maxrss_bytes"}
UNIT_EXTRA = {"load": {"includes_pristine_snapshot"}, "calib_pass": {"n_sequences"}, "dev_pass": {"n_sequences"},
              "worst_batch_calib": {"batch"}, "awq_block": {"bits", "n_alphas", "block_timings"},
              "qs2_block": {"bits"}, "p1_build": {"condition", "build_timings"},
              "wanda_build": {"condition", "build_timings"}, "quant_build": {"condition", "build_timings"}}


@pytest.fixture(scope="module")
def smoke_run(tiny_compress4_dir, tmp_path_factory):
    b = tmp_path_factory.mktemp("timing")
    cmd = [sys.executable, "-m", "task1.smoke", "--out", str(b / "out"), "--artifacts", str(b / "art"),
           "--model-path", str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", "2", "--limit-dev", "2"]
    p = subprocess.run(cmd, cwd=REPO_ROOT, env=dict(os.environ, HF_HUB_OFFLINE="1"), capture_output=True, text=True,
                       timeout=900)
    assert p.returncode == 0, p.stderr[-4000:]
    return b


def test_smoke_outputs(smoke_run):
    assert sorted(x.name for x in (smoke_run / "out").iterdir()) == ["run.json"]
    assert sorted(x.name for x in (smoke_run / "art").iterdir()) == ["artifact.safetensors", "manifest.json"]
    r = json.loads((smoke_run / "out" / "run.json").read_text())
    assert r["status"] == "ok" and r["eval_id"] == C.EVAL_ID
    assert set(r["config"]) == {"entry_point", "units", "settings", "device", "model_path", "limits", "limited",
                                "test_only", "output_files", "artifacts_dir", "data_files", "batch_size"}
    s = r["steps"]
    assert set(s) == {"units", "counts", "artifact", "pristine_after", "held_out_guard", "files_written"}
    assert s["held_out_guard"]["attempts"] == []
    assert set(s["units"]) == set(smoke.UNITS)
    for name, u in s["units"].items():
        assert set(u) == UNIT | UNIT_EXTRA.get(name, set()), name
        assert u["wall_s"] > 0 and u["peak_allocated_bytes"] is None and u["host_maxrss_bytes"] > 0
    c = s["counts"]
    assert c["n_layers"] == 4 and c["n_calibration"] == 2 and c["n_development"] == 2 and c["seq_len"] == 512
    assert len(c["eligible_shapes"]) == 4 * len(C.ELIGIBLE)
    assert len(s["units"]["awq_block"]["block_timings"]) == 1
    assert s["units"]["worst_batch_calib"]["batch"] == [2, 512]
    a = s["artifact"]
    man = json.loads((smoke_run / "art" / "manifest.json").read_text())
    assert a["stored_tensor_bytes"] == man["stored_tensor_bytes"] and a["file_bytes"] == man["file"]["bytes"]
    assert a["stored_tensor_bytes"] == K.artifact_bytes("q1_b8", [tuple(x) for x in c["eligible_shapes"]], 4)
    assert s["files_written"] == {str(smoke_run / "art" / "artifact.safetensors"): a["file_bytes"],
                                  str(smoke_run / "art" / "manifest.json"): a["manifest_bytes"]}


def test_smoke_refuses_test_args_under_results(tmp_path):
    out = RESULTS_DIR / "task1" / "smoke_refusal_check"
    with pytest.raises(SystemExit, match="refusing"):
        smoke.main(["--out", str(out), "--artifacts", str(tmp_path / "a"), "--limit-calib", "1"])
    assert not out.exists()


def test_plan_from_smoke(smoke_run):
    r = json.loads((smoke_run / "out" / "run.json").read_text())
    p = K.project(r, working_bytes=1_000_000_000)
    assert [x["name"] for x in p["rows"]] == [j["name"] for j in K.stage_jobs("run1")]
    assert p["over"] == [] and set(p["queue_end_h"]) == {"0", "1"}
    by = p["jobs_by_name"]
    assert all(x["projected_s"] == pytest.approx(x["raw_s"] * K.SAFETY) for x in p["rows"])
    u = {k: v["wall_s"] for k, v in r["steps"]["units"].items()}
    assert by["probe_prune"]["raw_s"] == pytest.approx(
        u["load"] + u["calib_pass"] + 8 * (u["restore"] + u["p1_build"] / 4 + u["calib_pass"]))
    assert by["select_p3_s50"]["raw_s"] == pytest.approx(
        u["load"] + 11 * (u["restore"] + u["wanda_build"] + u["dev_pass"] + u["calib_pass"]))
    assert by["compress_fp16"]["artifact_bytes"] > 0
    gpu1 = [x for x in p["rows"] if x["gpu"] == "1"]
    assert p["queue_end_s"]["1"] >= p["stagger_s"] + sum(x["projected_s"] for x in gpu1) - 1e-6
    assert p["disk_bytes"] == 1_000_000_000 + sum(x["artifact_bytes"] for x in p["rows"]) + \
        K.RESULTS_BYTES_PER_JOB * len(p["rows"])
    slow = copy.deepcopy(r)
    slow["steps"]["units"]["calib_pass"]["wall_s"] = 2000.0
    assert any("GPU 0 queue" in x for x in K.project(slow, 0)["over"])
    assert any("/kaggle/working" in x for x in K.project(r, 18 * 10**9)["over"])


def test_plan_cli(smoke_run, tmp_path):
    run = str(smoke_run / "out" / "run.json")
    assert K.main(["plan", "--smoke", run, "--working-bytes", "0", "--out", str(tmp_path / "plan.json")]) == 0
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert set(saved) >= {"rows", "queue_end_h", "disk_bytes", "limits", "over"}
    assert K.main(["plan", "--smoke", run, "--working-bytes", str(10**11)]) == 1


def test_full_model_artifact_bytes():
    sh = C.full_model_matrix_shapes()
    assert K.artifact_bytes("p2_s50", sh, 28) == C.EXPECTED_ELIGIBLE_PARAMS // 8
    extras = 28 * 2 * (2 * 1536 + 256)
    assert K.artifact_bytes("q1_b4", sh, 28) == C.Q3_FULL_MODEL_BUDGET_BYTES["int4"]
    assert K.artifact_bytes("q2_b3", sh, 28) == C.Q3_FULL_MODEL_BUDGET_BYTES["int3"] + extras
    assert K.artifact_bytes("q3_int4", sh, 28) == C.Q3_FULL_MODEL_BUDGET_BYTES["int4"] + extras
    assert K.artifact_bytes("fp16", sh, 28) == 0
