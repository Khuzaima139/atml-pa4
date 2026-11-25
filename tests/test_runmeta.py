import json

import pytest

from common.runmeta import PACKAGES, RunRecorder, collect_meta

META_KEYS = {
    "git_commit", "git_dirty", "hostname", "platform", "python_version", "torch_version",
    "torch_cuda_build", "torch_cudnn_build", "device_count", "device_names", "package_versions",
}
EXPECTED_PACKAGES = {
    "torch", "transformers", "tokenizers", "huggingface_hub", "datasets", "accelerate",
    "peft", "safetensors", "numpy", "scipy", "matplotlib", "lm_eval",
}


def test_recorder_refuses_existing_path(tmp_path):
    p = tmp_path / "run.json"
    p.write_text("{}")
    with pytest.raises(FileExistsError):
        RunRecorder(p, {"seed": 0})
    assert p.read_text() == "{}"


def test_update_rewrites_file(tmp_path):
    p = tmp_path / "run.json"
    rec = RunRecorder(p, {"seed": 1})
    first = json.loads(p.read_text())
    assert first["config"] == {"seed": 1} and first["status"] == "running"
    rec.update(step=1, loss=0.5)
    rec.update(step=2, loss=0.25)
    data = json.loads(p.read_text())
    assert data["step"] == 2 and data["loss"] == 0.25
    assert data["elapsed_wall_s"] >= 0
    rec.finish("ok")
    data = json.loads(p.read_text())
    assert data["status"] == "ok" and "end_time" in data
    assert sorted(x.name for x in tmp_path.iterdir()) == ["run.json"]


def test_meta_has_every_key():
    meta = collect_meta()
    assert META_KEYS <= set(meta)
    assert set(PACKAGES) == EXPECTED_PACKAGES
    assert set(meta["package_versions"]) == EXPECTED_PACKAGES
    assert meta["package_versions"]["transformers"] is not None
