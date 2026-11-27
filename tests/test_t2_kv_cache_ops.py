"""Part C cache pipeline on a tiny random Qwen2 (FP32, CPU, 2 layers, 12 query heads, 2 KV heads, head_dim 128)."""
import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from common.seeding import seed_everything
from task2 import kv_budget as KB
from task2 import kv_cache_ops as KO
from task2 import kv_metrics as KM
from task2 import kv_select as KS

L_CTX = 600  # r = 24: the suffix leaves the tail at 58, decoding completes a 64-token group
S_SUF = 34
VOCAB = 512


def _config(attn="sdpa"):
    return Qwen2Config(hidden_size=192, num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                       head_dim=128, intermediate_size=256, vocab_size=VOCAB, max_position_embeddings=2048,
                       tie_word_embeddings=True, attn_implementation=attn)


@pytest.fixture(scope="module")
def tiny():
    seed_everything(6304)
    model = Qwen2ForCausalLM(_config()).to(torch.float32).eval()
    assert model.config._attn_implementation == "sdpa"
    return model


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(6304)
    return torch.randint(0, VOCAB, (1, L_CTX + S_SUF), generator=g)


@pytest.fixture(scope="module")
def shape(tiny):
    s = KB.KVShape.from_config(tiny.config)
    assert s == KB.KVShape(2, 2, 128)
    return s


@pytest.fixture(scope="module")
def prefill(tiny, ids):
    return KO.full_prefill(tiny, ids, L_CTX)


def _mask(n, evicted, first_row):
    m = torch.ones(n, n, dtype=torch.bool).tril()
    if len(evicted):
        m[first_row:, torch.as_tensor(evicted)] = False
    return m[None, None]


def test_question_scores_equal_eager_attention(tiny, ids, prefill):
    eager = Qwen2ForCausalLM(_config("eager")).to(torch.float32).eval()
    eager.load_state_dict(tiny.state_dict())
    with torch.no_grad():
        att = eager(input_ids=ids, output_attentions=True).attentions
    probs = KS.suffix_attention(tiny, prefill.queries, prefill.keys, L_CTX, per_layer=True)
    assert len(probs) == 2
    for p, a in zip(probs, att):
        assert p.shape == (1, 12, S_SUF, L_CTX + S_SUF) and p.dtype == torch.float32
        torch.testing.assert_close(p, a[:, :, L_CTX:], atol=2e-6, rtol=1e-4)
    scores, secs = KO.score_prefill(tiny, prefill)
    expect = torch.stack([a[0, :, L_CTX:].mean(dim=(0, 1)) for a in att]).mean(0)[:L_CTX]
    torch.testing.assert_close(scores, expect, atol=1e-7, rtol=1e-4)
    assert scores.shape == (L_CTX,) and secs >= 0


def test_prefill_captures_suffix_queries_and_keeps_sdpa(tiny, prefill):
    assert tiny.config._attn_implementation == "sdpa"
    assert all(q.shape == (1, S_SUF, 12 * 128) for q in prefill.queries)
    assert all(k.shape == (1, 2, L_CTX + S_SUF, 128) for k in prefill.keys)
    assert not any(m._forward_hooks for layer in tiny.model.layers for m in [layer.self_attn.q_proj])


def test_identity_compression_reproduces_full_forward_and_generate(tiny, ids, shape, prefill):
    with torch.no_grad():
        full = tiny(input_ids=ids).logits[0]
    assert torch.isfinite(full).all()
    torch.testing.assert_close(prefill.last_logits, full[-1], atol=1e-4, rtol=1e-4)
    run, tf, suffix = KO.run_method(tiny, shape, prefill, ids, "fp16", eos_ids=(), suffix_logits_all=True)
    assert torch.isfinite(suffix).all()
    torch.testing.assert_close(suffix, full[L_CTX:], atol=1e-4, rtol=1e-4)
    gc = GenerationConfig(do_sample=False, max_new_tokens=32, repetition_penalty=1.0, eos_token_id=None,
                          pad_token_id=0)
    with torch.no_grad():
        gen = tiny.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gc)
    assert run.tokens == gen[0, ids.shape[1]:].tolist()
    assert len(run.tokens) == 32 and run.stop == "max_new_tokens"
    es = prefill.keys[0].element_size()
    assert run.live_bytes["total"] == (L_CTX + run.n_appended) * shape.channels * es
    assert run.n_appended == S_SUF + 31
    assert run.recon == {"k_per_layer": [0.0, 0.0], "v_per_layer": [0.0, 0.0], "k_mean": 0.0, "v_mean": 0.0}
    # teacher-forced logits equal a full forward over prompt + gold tokens
    gold = [5, 7, 11, 13, 17]
    tfl = KO.teacher_forced_logits(tf, run.first_logits, gold)
    with torch.no_grad():
        ref = tiny(input_ids=torch.cat([ids, torch.tensor([gold[:-1]])], dim=1)).logits[0, -5:]
    torch.testing.assert_close(tfl, ref, atol=1e-4, rtol=1e-4)
    nll = KM.nll_rows(tfl, gold)
    assert nll.shape == (5,) and bool((nll > 0).all())
    assert float(KM.kl_rows(tfl, tfl).abs().max()) < 1e-6


def test_greedy_stops_at_eos(tiny, ids, shape, prefill):
    run, _, _ = KO.run_method(tiny, shape, prefill, ids, "fp16", eos_ids=())
    first = run.tokens[0]
    run2, _, _ = KO.run_method(tiny, shape, prefill, ids, "fp16", eos_ids=(first,))
    assert run2.tokens == [first] and run2.stop == "eos" and run2.n_appended == S_SUF


@pytest.mark.parametrize("method", ["sink_recent", "question_aware"])
def test_eviction_equals_masked_full_forward(tiny, ids, shape, prefill, method):
    scores, _ = KO.score_prefill(tiny, prefill)
    run, tf, suffix = KO.run_method(tiny, shape, prefill, ids, method, scores, eos_ids=(), max_new_tokens=4,
                                    suffix_logits_all=True)
    kept = run.info["fp16_positions"]
    assert kept.numel() == run.info["n"]
    evicted = sorted(set(range(L_CTX)) - set(kept.tolist()))
    assert len(evicted) > 0
    T = L_CTX + S_SUF
    with torch.no_grad():
        ref = tiny(input_ids=ids, attention_mask=_mask(T, evicted, L_CTX)).logits[0]
    torch.testing.assert_close(suffix, ref[L_CTX:], atol=1e-4, rtol=1e-4)
    # first decode step: the session copy taken after the suffix re-run feeds the first generated token
    tok = run.tokens[0]
    phys_before = tf.physical_len
    step = tf.forward(torch.tensor([[tok]]))[-1]
    assert tf.physical_len == phys_before + 1 == run.info["n"] + S_SUF + 1
    ids2 = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    with torch.no_grad():
        ref2 = tiny(input_ids=ids2, attention_mask=_mask(T + 1, evicted, L_CTX)).logits[0, -1]
    torch.testing.assert_close(step, ref2, atol=1e-4, rtol=1e-4)
    es = prefill.keys[0].element_size()
    assert run.live_bytes["indices"] == (8 if method == "sink_recent" else 4 * run.info["n"])
    assert run.live_bytes["total"] == run.live_bytes["indices"] + (run.info["n"] + run.n_appended) * shape.channels * es


def test_quantized_session_uses_stored_values_and_tail_rule(tiny, ids, shape, prefill):
    run, tf, _ = KO.run_method(tiny, shape, prefill, ids, "int4_channel_k", eos_ids=())
    assert run.tail_rebuilds == (L_CTX % 64 + run.n_appended) // 64 >= 1
    assert 0 < run.recon["k_mean"] < 0.5 and 0 < run.recon["v_mean"] < 0.5
    for i, layer in enumerate(tf.cache.layers):
        k, v = tf.packed.layer_tensors(i, layer.keys.dtype)
        assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
    # the stored context differs from FP16 only through quantization
    k0, _ = tf.packed.layer_tensors(0, torch.float32)
    assert not torch.equal(k0[:, :, :L_CTX - L_CTX % 64], prefill.keys[0][:, :, :L_CTX - L_CTX % 64])
    assert torch.equal(k0[:, :, L_CTX - L_CTX % 64:L_CTX], prefill.keys[0][:, :, L_CTX - L_CTX % 64:L_CTX])
    # feeding tokens until a tail group completes rebuilds the session cache from the packed codes
    before = tf.packed.layers[0].k_q.n_tokens
    for t in range(64 - (L_CTX % 64 + S_SUF)):
        tf.forward(torch.tensor([[t]]))
    assert tf.packed.layers[0].k_q.n_tokens == before + 64 and tf.packed.layers[0].k_tail.shape[2] == 0
    for i, layer in enumerate(tf.cache.layers):
        k, _ = tf.packed.layer_tensors(i, layer.keys.dtype)
        assert torch.equal(layer.keys, k)


def test_all_methods_run_with_metrics(tiny, ids, shape, prefill):
    scores, _ = KO.score_prefill(tiny, prefill)
    runs = {}
    for method in KO.METHODS:
        run, tf, _ = KO.run_method(tiny, shape, prefill, ids, method, scores, eos_ids=(), max_new_tokens=3)
        assert set(run.seconds) == {"compression", "suffix", "decode"}
        assert torch.isfinite(run.first_logits).all() and len(run.tokens) == 3
        runs[method] = run
    assert runs["hybrid"].info["m"] >= 1 and runs["hybrid"].info["f"] >= 72
    assert set(runs["hybrid"].info["fp16_positions"].tolist()) <= set(
        runs["question_aware"].info["fp16_positions"].tolist())
    assert runs["sink_recent"].recon is None and runs["question_aware"].recon is None
    assert runs["hybrid"].recon["k_mean"] > runs["int4_channel_k"].recon["k_mean"]
    assert float(KM.kl_rows(runs["fp16"].first_logits[None], runs["fp16"].first_logits[None])) == 0.0


def test_metrics_helpers():
    assert KM.extract_code("The code is 26836.") == "26836"
    assert KM.extract_code("268361 then 12345") == "12345"
    assert KM.exact_match("no digits", "12345") == {"prediction": None, "no_match": True, "correct": False}
    offsets = [(0, 3), (3, 8), (8, 12), (12, 12), (12, 20)]
    assert KM.evidence_tokens(offsets, 5, 13) == [1, 2, 4]
    assert KM.evidence_fraction([1, 4, 9], [1, 2, 4]) == pytest.approx(2 / 3)


def test_fp16_model_bytes_equal_formulas(tiny, ids, shape):
    """Byte accounting of every method on an FP16 copy of the tiny model (CPU)."""
    m16 = Qwen2ForCausalLM(_config()).eval()
    m16.load_state_dict(tiny.state_dict())
    m16 = m16.half()
    pre = KO.full_prefill(m16, ids, L_CTX)
    assert pre.keys[0].dtype == torch.float16
    scores, _ = KO.score_prefill(m16, pre)
    assert scores.dtype == torch.float32
    M4 = KB.m4(shape, L_CTX)
    for method in KO.METHODS:
        run, _, _ = KO.run_method(m16, shape, pre, ids, method, scores, eos_ids=(), max_new_tokens=32)
        assert torch.isfinite(run.first_logits).all()
        cb, lb, n = run.context_bytes, run.live_bytes, run.n_appended
        assert cb["total"] == run.info["formula_bytes"]
        assert cb["total"] == KB.m16(shape, L_CTX) if method == "fp16" else cb["total"] <= M4
        if method in ("int4_token", "int4_channel_k"):
            assert cb["total"] == M4 and lb["total"] == KB.m4(shape, L_CTX + n)
        else:
            assert lb["total"] == cb["total"] + KB.decode_time_fp16_bytes(shape, n)
