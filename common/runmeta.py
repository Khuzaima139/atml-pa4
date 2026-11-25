"""Run metadata collection and a per-run result recorder."""
import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from .paths import REPO_ROOT

PACKAGES = [
    "torch", "transformers", "tokenizers", "huggingface_hub", "datasets",
    "accelerate", "peft", "safetensors", "numpy", "scipy", "matplotlib", "lm_eval",
]


def _pkg_version(name):
    for candidate in (name, name.replace("_", "-")):
        try:
            return _md.version(candidate)
        except _md.PackageNotFoundError:
            continue
    return None


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_meta() -> dict:
    """Collect git state, host, platform and library versions."""
    import torch

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    cuda = torch.cuda.is_available()
    n_dev = torch.cuda.device_count() if cuda else 0
    return {
        "git_commit": commit.strip() if commit else None,
        "git_dirty": (bool(status.strip()) if status is not None else None),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn_build": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        "cuda_available": cuda,
        "device_count": n_dev,
        "device_names": [torch.cuda.get_device_name(i) for i in range(n_dev)],
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
    }


def _now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class RunRecorder:
    """One result JSON per run; created exclusively and rewritten atomically on every update."""

    def __init__(self, path, config: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self.record = {
            "config": config,
            "meta": collect_meta(),
            "start_time": _now_iso(),
            "status": "running",
        }
        with open(self.path, "x", encoding="utf-8") as fh:
            json.dump(self.record, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    def _resources(self):
        out = {"elapsed_wall_s": time.perf_counter() - self._t0}
        try:
            import torch
            if torch.cuda.is_available():
                out["peak_vram_allocated_bytes"] = {
                    str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}
                out["peak_vram_reserved_bytes"] = {
                    str(i): torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())}
        except ImportError:
            pass
        return out

    def _write(self):
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.record, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def update(self, **fields):
        self.record.update(fields)
        self.record.update(self._resources())
        self.record["updated_time"] = _now_iso()
        self._write()
        return self.record

    def finish(self, status: str = "ok", **fields):
        fields["status"] = status
        fields["end_time"] = _now_iso()
        return self.update(**fields)
