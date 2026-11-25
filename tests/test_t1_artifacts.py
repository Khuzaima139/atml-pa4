"""Artifact save/load round trip and integrity checks."""
import json

import pytest
import torch

from task1 import awq, storage
from task1.artifacts import ARTIFACT_FILE, MANIFEST_FILE, executed_hashes, load_artifact, save_artifact
from task1.blocks import capture_block_inputs, layers
from task1.build import build_condition
from task1.data import read_sequences
from task1.model import Pristine, enumerate_eligible
from task1.tensorfile import ArtifactIntegrityError

SRC = {"model_path": "tiny"}


def _awq_scales(model, seqs, bits):
    res = awq.search_model(model, capture_block_inputs(model, seqs, 2), bits_list=(bits,), alphas=(0.0, 0.5))
    return {(li, g): res["scales"][(li, g, bits)] for li in range(len(layers(model))) for g in awq.GROUPS}


@pytest.mark.parametrize("cond", ["fp16", "p1_s75", "p2_24_s50", "q1_b3", "q2_b4"])
def test_round_trip_reproduces_executed_weights(compress_fp32, tmp_path, cond):
    model = compress_fp32
    seqs = read_sequences("c4_calibration", limit=2)
    pristine = Pristine(model)
    sc = _awq_scales(model, seqs, 4) if cond == "q2_b4" else None
    res = build_condition(model, seqs, cond, batch_size=2, awq_scales=sc, pristine=pristine)
    params = dict(model.named_parameters())
    executed = {n: p.detach().clone() for n, p in params.items()}
    before = executed_hashes(model, sorted(res.extra_tensors))
    man = save_artifact(tmp_path / "art", cond, SRC, res.payload, res.extra_tensors, model)
    pristine.restore(model)
    assert executed_hashes(model, sorted(res.extra_tensors)) != before or cond == "fp16"
    load_artifact(model, pristine, tmp_path / "art")
    assert executed_hashes(model, sorted(res.extra_tensors)) == before
    for n, p in model.named_parameters():
        assert torch.equal(p, executed[n]), n
    assert man["file"]["bytes"] == (tmp_path / "art" / ARTIFACT_FILE).stat().st_size
    shapes = [tuple(e.module.weight.shape) for e in enumerate_eligible(model)]
    if cond.startswith("q"):
        assert man["logical_bytes"]["eligible"] == storage.packed_quant_bytes(shapes, int(cond[-1]))
        assert man["stored_tensor_bytes"] == man["logical_bytes"]["eligible"] + sum(
            man["tensors"][n]["nbytes"] for n in man["extra_tensors"])
    if cond == "q2_b4":
        assert len(man["extra_tensors"]) == 3 * len(layers(model))
    if cond.startswith("p"):
        nnz = [m["nonzeros"] for m in man["matrices"]]
        assert man["logical_bytes"]["eligible"] == storage.sparse16_bytes(shapes, nnz)
    if cond == "fp16":
        assert man["tensors"] == {} and man["logical_bytes"]["eligible"] == storage.fp16_bytes(shapes)


def _saved_p1(model, tmp_path):
    seqs = read_sequences("c4_calibration", limit=1)
    pristine = Pristine(model)
    res = build_condition(model, seqs, "p1_s50", batch_size=1, pristine=pristine)
    save_artifact(tmp_path / "art", "p1_s50", SRC, res.payload, res.extra_tensors, model)
    return pristine


def test_manifest_sha256_mismatch_refused(compress_fp32, tmp_path):
    pristine = _saved_p1(compress_fp32, tmp_path)
    mp = tmp_path / "art" / MANIFEST_FILE
    man = json.loads(mp.read_text())
    name = sorted(man["tensors"])[0]
    man["tensors"][name]["sha256"] = "0" * 64
    mp.write_text(json.dumps(man))
    with pytest.raises(ArtifactIntegrityError, match="sha256"):
        load_artifact(compress_fp32, pristine, tmp_path / "art")


def test_corrupted_file_refused(compress_fp32, tmp_path):
    pristine = _saved_p1(compress_fp32, tmp_path)
    fp = tmp_path / "art" / ARTIFACT_FILE
    data = bytearray(fp.read_bytes())
    data[-1] ^= 0xFF
    fp.write_bytes(bytes(data))
    with pytest.raises(ArtifactIntegrityError, match="file sha256"):
        load_artifact(compress_fp32, pristine, tmp_path / "art")


def test_save_refuses_existing(compress_fp32, tmp_path):
    _saved_p1(compress_fp32, tmp_path)
    with pytest.raises(FileExistsError):
        save_artifact(tmp_path / "art", "fp16", SRC, [], {}, compress_fp32)
