"""Part C diagnostic (protocol AM4): INT8 and FP16 blocks, the control caches (each quantizes exactly the intended
tensor at the intended bit width), channel statistics, the control pipeline against the evaluated one, and an
end-to-end CPU run of python -m task2.kv_diag with a tiny random Qwen2 (head_dim 128, 2 KV heads, 2 layers, real
vocabulary and tokenizer) on real retrieval prompts."""
import gzip
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common.io import sha256_file, sha256_gz_content
from common.paths import REPO_ROOT, RESULTS_DIR
from task2 import kv_budget as KB
from task2 import kv_cache_ops as KO
from task2 import kv_diag as KD
from task2 import kv_run as KR
from task2.kv_quant import (GROUP, PackedKVCache, QuantBlock, dequantize_groups, pack_codes, quantize_groups,
                            relative_error)

PROMPTS = ["KV-2048-end-0-00", "KV-4096-middle-8-00"]
LIMIT = ",".join(PROMPTS)
ENV = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
H, D = 2, 128


def _kv(L, n_layers=2, seed=6304, scale_channels=None):
    g = torch.Generator().manual_seed(seed + L)
    ks = [torch.randn(1, H, L, D, generator=g) for _ in range(n_layers)]
    vs = [torch.randn(1, H, L, D, generator=g) for _ in range(n_layers)]
    if scale_channels:
        for k in ks:
            k[..., scale_channels] *= 100.0
    return [k.half() for k in ks], [v.half() for v in vs]


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------------------------------------------------
# Blocks


@pytest.mark.parametrize("layout", ["token", "channel"])
@pytest.mark.parametrize("bits", [2, 4])
def test_group_view_matches_quant_block(layout, bits):
    x = _kv(128)[0][0][:, :, 3:131] if layout == "token" else _kv(192)[0][0][:, :, :128]
    qb = QuantBlock.encode(x, layout, bits)
    codes, scale, offset = quantize_groups(KD.group_view(x, layout), bits)
    assert torch.equal(pack_codes(codes, bits), qb.payload)
    assert torch.equal(scale, qb.scale) and torch.equal(offset, qb.offset)


@pytest.mark.parametrize("layout", ["token", "channel"])
def test_int8_block(layout):
    x = _kv(256)[0][0][:, :, :128]
    b = KD.Int8Block.encode(x, layout)
    codes, scale, offset = quantize_groups(KD.group_view(x, layout), 8)
    assert b.bits == 8 and b.payload.dtype == torch.uint8 and torch.equal(b.codes(), codes)
    assert torch.equal(b.payload, codes)  # one code per byte
    n_groups = x.numel() // GROUP
    assert b.payload_bytes() == 64 * n_groups and b.metadata_bytes() == 4 * n_groups
    assert int(codes.max()) == 255 and len(torch.unique(codes)) > 16
    rec = b.decode(torch.float32)
    assert rec.shape == x.shape
    bound = (scale.float() / 2 * 1.001 + 1e-6).unsqueeze(-1)
    err = (dequantize_groups(codes, scale, offset, torch.float32) - KD.group_view(x, layout).float()).abs()
    assert bool((err <= bound).all())
    with pytest.raises(ValueError):
        KD.Int8Block.encode(x, layout, 4)
    b2 = KD.Int8Block.encode(x, layout)
    b.extend(b2)
    assert b.n_tokens == 256


def test_fp16_block():
    x = _kv(64)[0][0]
    b = KD.Fp16Block.encode(x)
    assert torch.equal(b.decode(), x) and b.payload_bytes() == 0 and b.fp16_bytes() == x.numel() * 2
    b.extend(KD.Fp16Block.encode(x))
    assert b.n_tokens == 128


# ---------------------------------------------------------------------------------------------------------------------
# Control caches


def _block_check(block, spec, x):
    """The block holds x as intended: exact FP16 for spec None, else the AM2.1 quantizer at spec's bit width on spec's
    groups (codes within range, decoded value = offset + scale * code, error within half a step)."""
    if spec is None:
        assert isinstance(block, KD.Fp16Block) and torch.equal(block.decode(), x)
        return
    layout, bits = spec
    assert type(block) is (KD.Int8Block if bits == 8 else QuantBlock)
    assert (block.layout, block.bits) == (layout, bits)
    codes, scale, offset = quantize_groups(KD.group_view(x, layout), bits)
    assert torch.equal(block.codes(), codes) and int(codes.max()) == (1 << bits) - 1
    assert block.payload_bytes() == x.numel() * bits // 8
    rec = block.decode(torch.float32)
    assert not torch.equal(rec, x.float())
    assert float((rec - x.float()).abs().max()) <= float(scale.float().max()) / 2 * 1.001 + 1e-6


@pytest.mark.parametrize("method", list(KD.SPECS))
def test_control_quantizes_intended_tensor(method):
    L = 2 * GROUP + 21
    ks, vs = _kv(L)
    spec = KD.SPECS[method]
    packed = KD.DiagPackedKVCache.control(method, ks, vs, **spec)
    n_q = 2 * GROUP
    for i, (k, v) in enumerate(zip(ks, vs)):
        lay = packed.layers[i]
        _block_check(lay.k_q, spec["k_spec"], k[:, :, :n_q])
        _block_check(lay.v_q, spec["v_spec"], v[:, :, :n_q])
        kt, vt = packed.layer_tensors(i, torch.float16)
        assert torch.equal(kt[:, :, n_q:], k[:, :, n_q:]) and torch.equal(vt[:, :, n_q:], v[:, :, n_q:])
        if spec["k_spec"] is None:
            assert torch.equal(kt, k)
        if spec["v_spec"] is None:
            assert torch.equal(vt, v)
    nb = packed.nbytes()
    per_tensor = len(ks) * H * n_q * D  # elements of one of K or V in the quantized prefix
    exp_payload = sum(per_tensor * s[1] // 8 for s in spec.values() if s is not None)
    exp_fp16 = sum(per_tensor * 2 for s in spec.values() if s is None)
    assert nb["payload"] == exp_payload and nb["fp16_retained"] == exp_fp16 and nb["indices"] == 0
    assert nb["metadata"] == sum(per_tensor // GROUP * 4 for s in spec.values() if s is not None)
    assert nb["fp16_tail"] == len(ks) * 2 * H * 21 * D * 2 and nb["total"] == sum(v for k, v in nb.items()
                                                                                  if k != "total")
    assert packed.context_bytes == nb
    rec = packed.reconstruction_errors(ks, vs)
    assert (rec["k_mean"] == 0) == (spec["k_spec"] is None) and (rec["v_mean"] == 0) == (spec["v_spec"] is None)
    # appended entries: the tail group completed at 64 tokens goes into the blocks of the same specs
    g = torch.Generator().manual_seed(1)
    nk = [torch.randn(1, H, 50, D, generator=g).half() for _ in ks]
    nv = [torch.randn(1, H, 50, D, generator=g).half() for _ in ks]
    assert packed.append(nk, nv, torch.arange(L, L + 50)) == 1
    assert torch.equal(packed.positions(), torch.arange(L + 50))
    for i in range(len(ks)):
        fk, fv = torch.cat([ks[i], nk[i]], 2), torch.cat([vs[i], nv[i]], 2)
        lay = packed.layers[i]
        assert lay.k_q.n_tokens == 3 * GROUP and lay.k_tail.shape[2] == 21 + 50 - GROUP
        new_k = KD.encode_block(fk[:, :, 2 * GROUP:3 * GROUP], spec["k_spec"]).decode()
        new_v = KD.encode_block(fv[:, :, 2 * GROUP:3 * GROUP], spec["v_spec"]).decode()
        kt, vt = packed.layer_tensors(i, torch.float16)
        assert torch.equal(kt[:, :, 2 * GROUP:3 * GROUP], new_k) and torch.equal(vt[:, :, 2 * GROUP:3 * GROUP], new_v)
        assert torch.equal(kt[:, :, 3 * GROUP:], fk[:, :, 3 * GROUP:])
        if spec["k_spec"] is None:
            assert torch.equal(kt, fk)
        if spec["v_spec"] is None:
            assert torch.equal(vt, fv)


@pytest.mark.parametrize("method,layout", [("int4_token", "token"), ("int4_channel_k", "channel")])
def test_control_cache_equals_evaluated_layout(method, layout):
    """With the evaluated specs, the control cache holds the same tensors and bytes as the evaluated packed cache, before
    and after appends: the controls share the evaluated quantizer, groups and trailing rule."""
    L = 2 * GROUP + 37
    ks, vs = _kv(L)
    ref = PackedKVCache.uniform(method, ks, vs, layout, 4)
    dg = KD.DiagPackedKVCache.control(method, ks, vs, (layout, 4), ("token", 4))
    assert dg.nbytes() == ref.nbytes()
    g = torch.Generator().manual_seed(2)
    for step in range(3):
        nk = [torch.randn(1, H, 20, D, generator=g).half() for _ in ks]
        assert ref.append(nk, nk, torch.arange(L + 20 * step, L + 20 * (step + 1))) == \
            dg.append(nk, nk, torch.arange(L + 20 * step, L + 20 * (step + 1)))
        for i in range(len(ks)):
            a, b = ref.layer_tensors(i), dg.layer_tensors(i)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert dg.nbytes() == ref.nbytes() and torch.equal(dg.positions(), ref.positions())


def test_clone_keeps_blocks_and_specs():
    L = 3 * GROUP + 5
    ks, vs = _kv(L)
    for method, spec in KD.SPECS.items():
        p = KD.DiagPackedKVCache.control(method, ks, vs, **spec)
        for c in (p.clone(), p.clone("cpu")):
            assert type(c) is KD.DiagPackedKVCache and (c.k_spec, c.v_spec) == (p.k_spec, p.v_spec)
            assert [type(lay.k_q) for lay in c.layers] == [type(lay.k_q) for lay in p.layers]
            assert c.nbytes() == p.nbytes() and c.context_bytes == p.context_bytes
            for i in range(len(ks)):
                assert all(torch.equal(x, y) for x, y in zip(c.layer_tensors(i), p.layer_tensors(i)))
            nk = [torch.zeros(1, H, 70, D).half() for _ in ks]
            c.append(nk, nk, torch.arange(L, L + 70))
            assert p.n_positions == L and c.n_positions == L + 70


def test_int8_error_below_int4():
    ks, vs = _kv(4 * GROUP, seed=11)
    p8 = KD.DiagPackedKVCache.control("kv_int8_token", ks, vs, ("token", 8), ("token", 8))
    p4 = KD.DiagPackedKVCache.control("int4", ks, vs, ("token", 4), ("token", 4))
    r8, r4 = p8.reconstruction_errors(ks, vs), p4.reconstruction_errors(ks, vs)
    assert 0 < r8["k_mean"] < r4["k_mean"] / 8 and 0 < r8["v_mean"] < r4["v_mean"] / 8
    k8 = p8.layer_tensors(0)[0]
    assert relative_error(ks[0], k8) == pytest.approx(r8["k_per_layer"][0])


# ---------------------------------------------------------------------------------------------------------------------
# Channel statistics


def test_channel_stats_on_outlier_keys():
    """A few key channels scaled by 100: token-wise INT4 error on the other channels exceeds channel-K's, the outlier
    channels form the top group, and the K max/median ratio is far above V's."""
    out_ch = [5, 77]
    ks, vs = _kv(4 * GROUP + 13, scale_channels=out_ch)
    st = KD.channel_stats(ks, vs)
    assert int(st["n_top"]) == 2 and int(st["context_len"]) == 4 * GROUP + 13
    assert st["k_top"].shape == (2, H, D)
    for i in range(2):
        for h in range(H):
            assert np.flatnonzero(st["k_top"][i, h]).tolist() == out_ch
    assert (st["k_err_token_rest"] > 3 * st["k_err_channel_rest"]).all()
    assert (st["k_err_token_rest"] > 0.5).all() and (st["k_err_channel_rest"] < 0.2).all()
    assert (st["k_ratio"] > 30).all() and (st["v_ratio"] < 3).all()
    s = KD.prompt_channel_summary(st)
    assert set(s) == {f"{k}_mean" for k in KD.HEAD_ARRAYS} | {"n_zero_norm_channels"}
    assert s["n_zero_norm_channels"] == 0


def test_channel_stats_match_direct_computation():
    ks, vs = _kv(2 * GROUP + 9, seed=3)
    st = KD.channel_stats(ks, vs)
    assert set(st) == set(KD.NPZ_KEYS)
    tok = PackedKVCache.uniform("int4_token", ks, vs, "token", 4)
    chan = PackedKVCache.uniform("int4_channel_k", ks, vs, "channel", 4)
    i, h = 1, 0
    K = ks[i][0, h].float().numpy()
    V = vs[i][0, h].float().numpy()
    kt = tok.layer_tensors(i, torch.float32)[0][0, h].numpy()
    vt = tok.layer_tensors(i, torch.float32)[1][0, h].numpy()
    kc = chan.layer_tensors(i, torch.float32)[0][0, h].numpy()
    np.testing.assert_allclose(st["k_abs_max"][i, h], np.abs(K).max(0), rtol=1e-6)
    np.testing.assert_allclose(st["k_abs_median"][i, h], np.median(np.abs(K), 0), rtol=1e-5)
    np.testing.assert_allclose(st["v_abs_median"][i, h], np.median(np.abs(V), 0), rtol=1e-5)
    e_kt = np.linalg.norm(kt - K, axis=0) / np.linalg.norm(K, axis=0)
    e_kc = np.linalg.norm(kc - K, axis=0) / np.linalg.norm(K, axis=0)
    e_vt = np.linalg.norm(vt - V, axis=0) / np.linalg.norm(V, axis=0)
    np.testing.assert_allclose(st["k_err_token"][i, h], e_kt, rtol=1e-4)
    np.testing.assert_allclose(st["k_err_channel"][i, h], e_kc, rtol=1e-4)
    np.testing.assert_allclose(st["v_err_token"][i, h], e_vt, rtol=1e-4)
    kmax = np.abs(K).max(0)
    assert st["k_ratio"][i, h] == pytest.approx(kmax.max() / np.median(kmax), rel=1e-5)
    top = np.argsort(-kmax, kind="stable")[:2]
    assert np.flatnonzero(st["k_top"][i, h]).tolist() == sorted(top.tolist())
    rest = np.setdiff1d(np.arange(D), top)
    assert st["k_err_token_rest"][i, h] == pytest.approx(e_kt[rest].mean(), rel=1e-4)
    assert st["k_err_channel_top"][i, h] == pytest.approx(e_kc[top].mean(), rel=1e-4)
    vtop = np.argsort(-np.abs(V).max(0), kind="stable")[:2]
    assert st["v_err_token_top"][i, h] == pytest.approx(e_vt[vtop].mean(), rel=1e-4)


def test_top_mask_ties_go_to_lower_index():
    m = KD._top_mask(torch.tensor([[1.0, 3.0, 3.0, 3.0, 0.0]]), 2)
    assert m[0].tolist() == [False, True, True, False, False]


def test_readout_and_summary():
    per = {"fp16": {"accuracy": 1.0}, "kv_int8_token": {"accuracy": 1.0, "mean_kl_teacher_forced": 0.001},
           "k_int4_token": {"accuracy": 0.0}, "v_int4_token": {"accuracy": 17 / 18}}
    ch = {"overall": {"k_ratio": 30.0, "v_ratio": 3.0, "k_err_token_rest": 0.9, "k_err_channel_rest": 0.1}}
    r = KD.readout(per, ch)
    assert r["c_int8"] and not r["rival_token_path_defect"] and r["a_k_only"] and not r["b_v_only"]
    assert r["k_ratio"] and r["k_rest_error"]
    per["kv_int8_token"]["mean_kl_teacher_forced"] = 0.06
    assert KD.readout(per, ch)["rival_token_path_defect"]


# ---------------------------------------------------------------------------------------------------------------------
# Selection, refusals, gold separation


def test_prompt_selection():
    cases = KR.load_cases()
    diag = KR.first_per_cell(cases)
    assert len(diag) == 18 and all(i.endswith("-00") for i in diag)
    assert [c["id"] for c in KD.select_prompts(cases, None)] == diag
    assert [c["id"] for c in KD.select_prompts(cases, "2")] == diag[:2]
    assert [c["id"] for c in KD.select_prompts(cases, "KV-4096-end-8-03")] == ["KV-4096-end-8-03"]
    assert KD.config_order(0) == list(KD.CONFIGS) and KD.config_order(7) == list(KD.CONFIGS[1:] + KD.CONFIGS[:1])


def test_refuses_existing_out(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="output path exists"):
        KD.main(["--out", str(out), "--device", "cpu"])
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("opt", [["--limit-prompts", "2"], ["--model-path", "x"]])
def test_refuses_limits_under_results(opt):
    out = RESULTS_DIR / "task2" / "part_c" / "diag_refusal_check_never_created"
    with pytest.raises(SystemExit, match="refusing test-only options"):
        KD.main(["--out", str(out), "--device", "cpu", *opt])
    assert not out.exists()


def test_generation_apis_take_no_gold():
    banned = {"code", "gold", "gold_ids", "answer", "evidence", "evidence_positions", "target", "label"}
    for fn in (KD.generation_stage, KD.run_control, KD.run_config, KD.channel_stats, KD.DiagPackedKVCache.control):
        params = set(inspect.signature(fn).parameters)
        assert not (params & banned), (fn.__qualname__, params & banned)


# ---------------------------------------------------------------------------------------------------------------------
# Tiny model: the control pipeline against the evaluated one, and the entry point end to end


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(KR.MODEL_ID, revision=KR.locked_revision(), local_files_only=True)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(6304)
    d = tmp_path_factory.mktemp("tiny_qwen2_diag")
    cfg = Qwen2Config(hidden_size=128, num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      head_dim=128, intermediate_size=256, vocab_size=151_936, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


def test_control_pipeline_equals_evaluated_pipeline(tiny_dir, tokenizer, monkeypatch):
    """run_control with the int4_token specs gives the same tokens, logits and teacher-forced logits as
    task2.kv_cache_ops.run_method for int4_token."""
    monkeypatch.setitem(KD.SPECS, "int4_token_copy", {"k_spec": ("token", 4), "v_spec": ("token", 4)})
    case = next(c for c in KR.load_cases() if c["id"] == PROMPTS[0])
    inputs, gold = KR.prepare_prompt(tokenizer, case)
    model, _ = KR.load_model("cpu", tiny_dir)
    shape = KB.KVShape.from_config(model.config)
    ids = inputs.ids_tensor("cpu")
    prefill = KO.full_prefill(model, ids, inputs.context_len)
    ref, ref_s, _ = KO.run_method(model, shape, prefill, ids, "int4_token", tf_device="cpu")
    got, got_s = KD.run_control(model, prefill, ids, "int4_token_copy", tf_device="cpu")
    assert got.tokens == ref.tokens and got.stop == ref.stop and torch.equal(got.first_logits, ref.first_logits)
    assert got.context_bytes == ref.context_bytes and got.recon == ref.recon and got.tail_rebuilds == ref.tail_rebuilds
    assert torch.equal(got.final_positions, ref.final_positions) and got.n_appended == ref.n_appended
    a = KO.teacher_forced_logits(ref_s.clone("cpu"), ref.first_logits, gold.code_ids)
    b = KO.teacher_forced_logits(got_s.clone("cpu"), got.first_logits, gold.code_ids)
    assert torch.equal(a, b)
    run8, _ = KD.run_config(model, shape, prefill, ids, "kv_int8_token", tf_device="cpu")
    assert run8.info["k_spec"] == "int8_token" and run8.recon["k_mean"] < ref.recon["k_mean"]


RUN_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "environment",
           "kv_shape", "model", "model_facts", "row_files", "step", "tokenizer", "timing", "summary", "prompts_done"}
CONFIG_KEYS = {"version", "scorer", "protocol", "pipeline_version", "device", "model_id", "model_revision",
               "model_path", "dtype", "attn_implementation", "limits", "limited", "output_files", "data", "n_prompts",
               "prompt_ids", "diagnostic_prompt_ids", "configs", "labels", "quant", "evaluated_code_path",
               "control_specs", "group_size", "int8_storage", "top_fraction", "top_channels_rule", "median_rule",
               "order_rule", "seed", "predictions", "generation", "teacher_forced_session_device", "npz_arrays",
               "npz_format_version"}
ROW_KEYS = {"version", "scorer", "id", "prompt_index", "config", "label", "k_quant", "v_quant", "evaluated_code_path",
            "config_order", "config_slot", "target_tokens", "position", "distractors", "context_len", "suffix_len",
            "generated_token_ids", "generated_text", "n_generated", "stop", "hit_max_new_tokens", "prediction",
            "correct", "no_match", "answer_nll_mean", "answer_nll_per_token", "kl_post_prefill",
            "kl_teacher_forced_mean", "kl_teacher_forced_per_position", "k_recon_mean", "v_recon_mean",
            "k_recon_per_layer", "v_recon_per_layer", "context_bytes", "n_appended", "live_bytes_end",
            "positions_preserved", "tail_rebuilds", "seconds", "peak_allocated_bytes", "start_allocated_bytes"}
PROMPT_KEYS = {"version", "scorer", "id", "prompt_index", "target_tokens", "position", "distractors", "code",
               "context_len", "suffix_len", "total_len", "config_order", "prefill_seconds",
               "prefill_peak_allocated_bytes", "channel_stats_seconds", "fp16_correct", "npz", "channel_summary"}
PER_CONFIG_KEYS = {"label", "n", "n_correct", "accuracy", "n_no_match", "n_hit_max_new_tokens", "mean_answer_nll",
                   "mean_kl_post_prefill", "mean_kl_teacher_forced", "mean_k_recon", "mean_v_recon",
                   "n_positions_preserved"}
BYTE_KEYS = {"payload", "metadata", "fp16_retained", "fp16_tail", "indices", "total"}


@pytest.fixture(scope="module")
def diag_dir(tmp_path_factory, tiny_dir):
    out = tmp_path_factory.mktemp("kv_diag") / "out"
    cmd = [sys.executable, "-m", "task2.kv_diag", "--out", str(out), "--device", "cpu", "--limit-prompts", LIMIT,
           "--model-path", str(tiny_dir)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=ENV, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert f"kv_diag done: {out}" in proc.stdout
    return out


def test_diag_outputs(diag_dir, tokenizer):
    root = Path(diag_dir)
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    assert files == sorted(["run.json", "control_rows.jsonl.gz", "prompt_rows.jsonl.gz"]
                           + [f"npz/{p}.npz" for p in PROMPTS])
    run = json.loads((root / "run.json").read_text())
    assert run["status"] == "ok" and run["step"] == "done" and set(run) == RUN_TOP
    c = run["config"]
    assert set(c) == CONFIG_KEYS and c["version"] == KD.VERSION and c["limited"] is True
    assert c["prompt_ids"] == PROMPTS and c["configs"] == list(KD.CONFIGS) and len(c["diagnostic_prompt_ids"]) == 18
    assert c["data"]["sha256"] == c["data"]["manifest_sha256"] == sha256_file(KR.DATA_FILE)
    assert run["model_facts"]["parameter_dtypes"] == ["torch.float16"]
    assert run["kv_shape"] == {"n_layers": 2, "n_kv_heads": 2, "head_dim": 128, "channels": 1024}
    assert run["prompts_done"] == 2 and set(run["timing"]["prompt_s"]) == set(PROMPTS) and run["timing"]["load_s"] > 0
    for k in ("transformers", "torch", "numpy"):
        assert run["meta"]["package_versions"][k]
    for key, name in (("rows", "control_rows.jsonl.gz"), ("prompts", "prompt_rows.jsonl.gz")):
        info = run["row_files"][key]
        assert info["path"] == name and sha256_file(root / name) == info["sha256"]
        assert sha256_gz_content(root / name) == info["sha256_decompressed"]

    rows = _rows(root / "control_rows.jsonl.gz")
    assert run["row_files"]["rows"]["n_rows"] == len(rows) == 12
    assert {(r["id"], r["config"]) for r in rows} == {(p, m) for p in PROMPTS for m in KD.CONFIGS}
    cases = {c["id"]: c for c in KR.load_cases()}
    for r in rows:
        assert set(r) == ROW_KEYS and r["version"] == KD.VERSION and r["positions_preserved"] is True
        assert set(r["context_bytes"]) == BYTE_KEYS == set(r["live_bytes_end"])
        assert (r["k_quant"], r["v_quant"]) == KD.QUANT[r["config"]]
        assert r["evaluated_code_path"] == (r["config"] in KD.EVALUATED)
        assert len(r["kl_teacher_forced_per_position"]) == 5 == len(r["answer_nll_per_token"])
        assert len(r["k_recon_per_layer"]) == 2 and r["n_generated"] == len(r["generated_token_ids"])
        assert r["prediction"] is None or len(r["prediction"]) == 5
        assert r["correct"] == (r["prediction"] == cases[r["id"]]["code"])
        if r["config"] == "fp16":
            assert r["kl_post_prefill"] == 0 and r["kl_teacher_forced_mean"] == 0 and r["k_recon_mean"] == 0
        assert (r["k_recon_mean"] == 0) == (r["k_quant"] == "fp16")
        assert (r["v_recon_mean"] == 0) == (r["v_quant"] == "fp16")
        L = r["context_len"]
        cb = r["context_bytes"]
        if r["config"] in ("int4_token", "int4_channel_k"):
            assert cb["total"] == KB.m4(KB.KVShape(2, 2, 128), L)
        if r["config"] == "kv_int8_token":
            assert cb["payload"] == 2 * 2 * 2 * 128 * (L // 64) * 64 and cb["fp16_retained"] == 0
        if r["config"] in ("k_int4_token", "v_int4_token"):
            assert cb["fp16_retained"] == 2 * 2 * 128 * (L // 64) * 64 * 2
    by = {(r["id"], r["config"]): r for r in rows}
    for p in PROMPTS:
        assert by[(p, "kv_int8_token")]["k_recon_mean"] < by[(p, "int4_token")]["k_recon_mean"]
        assert by[(p, "k_int4_token")]["k_recon_mean"] == pytest.approx(by[(p, "int4_token")]["k_recon_mean"])
        assert by[(p, "v_int4_token")]["v_recon_mean"] == pytest.approx(by[(p, "int4_token")]["v_recon_mean"])

    prs = _rows(root / "prompt_rows.jsonl.gz")
    assert [r["id"] for r in prs] == PROMPTS
    for r in prs:
        assert set(r) == PROMPT_KEYS and r["npz"] == f"npz/{r['id']}.npz"
        assert set(r["channel_summary"]) == {f"{k}_mean" for k in KD.HEAD_ARRAYS} | {"n_zero_norm_channels"}
        with np.load(root / r["npz"], allow_pickle=False) as z:
            assert set(z.files) == set(KD.NPZ_KEYS)
            for k in KD.CHANNEL_ARRAYS:
                assert z[k].shape == (2, 2, 128) and z[k].dtype == np.float32
            for k in KD.HEAD_ARRAYS:
                assert z[k].shape == (2, 2) and z[k].dtype == np.float32
            for k in KD.MASK_ARRAYS:
                assert z[k].dtype == np.uint8 and (z[k].sum(-1) == 2).all()
            assert int(z["context_len"]) == r["context_len"] and int(z["n_top"]) == 2
            assert bytes(z["version_ascii"]).decode() == KD.VERSION
            assert r["channel_summary"]["k_ratio_mean"] == pytest.approx(float(z["k_ratio"].mean()), rel=1e-6)

    s = run["summary"]
    assert s["n_rows"] == 12 and set(s["per_config"]) == set(KD.CONFIGS)
    for m, v in s["per_config"].items():
        assert set(v) == PER_CONFIG_KEYS and v["n"] == 2 and v["n_positions_preserved"] == 2
    ch = s["channel"]
    assert set(ch) == {"per_layer", "overall", "n_prompts", "n_top_channels", "n_zero_norm_channels"}
    assert set(ch["per_layer"]) == set(KD.HEAD_ARRAYS) and all(len(v) == 2 for v in ch["per_layer"].values())
    assert ch["n_prompts"] == 2 and ch["n_top_channels"] == 2
    ro = s["readout"]
    assert set(ro) == {"definitions", "c_int8", "rival_token_path_defect", "a_k_only", "b_v_only", "k_ratio",
                       "k_rest_error", "values"}
    assert ro["rival_token_path_defect"] == (not ro["c_int8"])
