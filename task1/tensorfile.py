"""Safetensors files with a JSON manifest (per-tensor shape, dtype, sha256; file bytes and sha256).

The sha256 of a tensor is taken over its contiguous little-endian raw bytes on CPU.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

from common.io import sha256_file, write_json_new


def tensor_sha256(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()


def tensor_entry(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "sha256": tensor_sha256(t),
            "nbytes": t.numel() * t.element_size()}


def save_tensors(path, tensors: dict, extra: dict = None, manifest_path=None) -> dict:
    """Write tensors to a new safetensors file and a new manifest; return the manifest."""
    from safetensors.torch import save_file

    path = Path(path)
    manifest_path = Path(manifest_path) if manifest_path else path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(path if path.exists() else manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    tmp = path.with_name(path.name + ".tmp")
    save_file(cpu, str(tmp))
    os.replace(tmp, path)
    manifest = dict(extra or {})
    manifest["tensors"] = {k: tensor_entry(v) for k, v in sorted(cpu.items())}
    manifest["stored_tensor_bytes"] = sum(e["nbytes"] for e in manifest["tensors"].values())
    manifest["file"] = {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json_new(manifest_path, manifest)
    return manifest


class ArtifactIntegrityError(RuntimeError):
    pass


def load_tensors(path, manifest_path=None, device="cpu") -> tuple:
    """Load a safetensors file after checking its file sha256 and every tensor sha256 against the manifest."""
    from safetensors.torch import load_file

    path = Path(path)
    manifest_path = Path(manifest_path) if manifest_path else path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    got = sha256_file(path)
    if got != manifest["file"]["sha256"]:
        raise ArtifactIntegrityError(f"{path.name}: file sha256 {got} != manifest {manifest['file']['sha256']}")
    tensors = load_file(str(path))
    if set(tensors) != set(manifest["tensors"]):
        raise ArtifactIntegrityError(f"{path.name}: tensor names differ from the manifest")
    for k, t in tensors.items():
        e = manifest["tensors"][k]
        if tensor_sha256(t) != e["sha256"] or list(t.shape) != e["shape"]:
            raise ArtifactIntegrityError(f"{path.name}: tensor {k} does not match its manifest sha256/shape")
    return {k: v.to(device) for k, v in tensors.items()}, manifest
