"""Exact speculative sampler with tiny random Qwen2 target and draft (CPU, fp32) through the real cache path."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task3.lm import HFCachedLM, hf_full_logits
from task3.sampling import EOS_IDS, VOCAB_SIZE, make_generator, sample_token
from task3.spec import cache_next_logits, speculative_generate, target_only_generate

PROMPT = [151644, 8948, 198, 2610, 525, 1207, 16948, 13, 151645, 198, 151644, 77091, 198]
FORBIDDEN = 1000


def _tiny(seed, layers, hidden):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=2 * hidden, vocab_size=VOCAB_SIZE, tie_word_embeddings=True,
                      max_position_embeddings=4096, initializer_range=0.2)
    return Qwen2ForCausalLM(cfg).to(torch.float32).eval()


@pytest.fixture(scope="module")
def target_model():
    return _tiny(6304, 2, 64)


@pytest.fixture(scope="module")
def draft_model():
    return _tiny(6305, 1, 32)


class Hooked:
    """Cached LM whose logits pass through fn(logits, predicted absolute positions)."""

    def __init__(self, lm, fn):
        self.lm, self.fn, self.device = lm, fn, lm.device

    def reset(self):
        self.lm.reset()

    @property
    def length(self):
        return self.lm.length

    def crop(self, n):
        self.lm.crop(n)

    def forward(self, ids):
        start = self.lm.length
        logits = self.lm.forward(ids).clone()
        return self.fn(logits, torch.arange(start + 1, start + 1 + len(ids)))


def _base(lm):
    return lm.lm if isinstance(lm, Hooked) else lm


class CacheProbe:
    """Round callback: cache lengths and next-token logits from both caches versus a fresh full forward."""

    def __init__(self, target, draft, t_model, d_model):
        self.target, self.draft, self.t_model, self.d_model = target, draft, t_model, d_model
        self.diffs = []

    def __call__(self, rec, ctx, out):
        n_end = len(ctx)
        assert self.target.length == n_end - 1 == rec["target_cache_len_end"]
        expect_d = min(rec["ctx_len_start"] + rec["proposed"] - 1, n_end - 1)
        assert self.draft.length == expect_d == rec["draft_cache_len_end"]
        for lm, model in ((self.target, self.t_model), (self.draft, self.d_model)):
            base = _base(lm)
            with torch.inference_mode():
                fresh = hf_full_logits(model, ctx)[-1]
            got = cache_next_logits(base, ctx)
            self.diffs.append(float((got - fresh).abs().max()))
        assert self.target.length == n_end - 1 and self.draft.length == expect_d


def _check_identities(res, gamma):
    total_emitted = 0
    for r in res["rounds"]:
        assert r["accepted"] <= r["examined"] <= r["proposed"] <= gamma
        assert r["draft_forward_calls"] == r["proposed"]
        assert r["target_tokens_fed"] == r["proposed"] + 1
        total_emitted += r["emitted"]
    assert total_emitted == res["output_tokens"]
    assert res["accepted"] <= res["examined"] <= res["proposed"]
    assert res["target_verification_passes"] == res["n_rounds"]
    assert res["target_passes_per_output_token"] >= 1.0 / (gamma + 1) - 1e-12


def test_full_acceptance_when_draft_equals_target(target_model):
    t, d = HFCachedLM(target_model), HFCachedLM(target_model)
    probe = CacheProbe(t, d, target_model, target_model)
    res = speculative_generate(t, d, PROMPT, 4, 13, (), make_generator(11, "cpu"), round_callback=probe)
    assert [r["emitted"] for r in res["rounds"]] == [5, 5, 3]
    assert all(r["accepted"] == r["proposed"] for r in res["rounds"])
    assert [r["bonus"] for r in res["rounds"]] == [True, True, False]
    assert res["rounds"][-1]["bonus_skipped_cap"] is True
    assert res["rounds"][1]["draft_catchup_tokens"] == 2
    assert res["termination"] == "cap" and res["output_tokens"] == 13
    _check_identities(res, 4)
    print("full acceptance: max abs cache-vs-fresh logit diff", max(probe.diffs))
    assert max(probe.diffs) < 1e-4


def _forbid(logits, pos):
    logits[:, FORBIDDEN] = float("-inf")
    return logits


def _only_forbidden(logits, pos):
    out = torch.full_like(logits, float("-inf"))
    out[:, FORBIDDEN] = 0.0
    return out


def test_forced_rejection_at_first_proposal(target_model, draft_model):
    t = Hooked(HFCachedLM(target_model), _forbid)
    d = Hooked(HFCachedLM(draft_model), _only_forbidden)
    probe = CacheProbe(t, d, target_model, draft_model)
    res = speculative_generate(t, d, PROMPT, 4, 6, (), make_generator(12, "cpu"), round_callback=probe)
    assert res["n_rounds"] == 6
    for r in res["rounds"]:
        assert r["emitted"] == 1 and r["accepted"] == 0 and r["examined"] == 1 and r["rejected"]
        assert r["accept_flags"] == [False]
        assert not r["residual_fallback"]
    assert FORBIDDEN not in res["output_ids"]
    assert [r["proposed"] for r in res["rounds"]] == [4, 4, 4, 3, 2, 1]
    _check_identities(res, 4)
    print("forced rejection: max abs cache-vs-fresh logit diff", max(probe.diffs))
    assert max(probe.diffs) < 1e-4


@pytest.mark.parametrize("gamma", [2, 4])
def test_cache_consistency_random_draft(target_model, draft_model, gamma):
    t, d = HFCachedLM(target_model), HFCachedLM(draft_model)
    probe = CacheProbe(t, d, target_model, draft_model)
    res = speculative_generate(t, d, PROMPT, gamma, 20, EOS_IDS, make_generator(13 + gamma, "cpu"),
                               round_callback=probe)
    assert res["output_tokens"] == 20 or res["termination"] == "eos"
    _check_identities(res, gamma)
    for key in ("prefill_target_s", "prefill_draft_s", "draft_s_total", "verify_s_total", "accept_s_total",
                "cache_s_total", "host_s", "total_s", "mean_accepted_prefix_per_verification"):
        assert key in res
    print(f"gamma {gamma}: max abs cache-vs-fresh logit diff", max(probe.diffs))
    assert max(probe.diffs) < 1e-4


def _eos_at(abs_index):
    def fn(logits, pos):
        logits[:, list(EOS_IDS)] = float("-inf")
        hit = pos == abs_index
        if hit.any():
            logits[hit] = float("-inf")
            logits[hit, EOS_IDS[0]] = 0.0
        return logits
    return fn


def test_eos_inside_round_ends_generation(target_model):
    fn = _eos_at(len(PROMPT) + 2)
    t, d = Hooked(HFCachedLM(target_model), fn), Hooked(HFCachedLM(target_model), fn)
    probe = CacheProbe(t, d, target_model, target_model)
    res = speculative_generate(t, d, PROMPT, 4, 20, EOS_IDS, make_generator(14, "cpu"), round_callback=probe)
    assert res["n_rounds"] == 1
    r = res["rounds"][0]
    assert r["accepted"] == 4 and r["bonus"] and r["eos"] and r["discarded"] == 2 and r["emitted"] == 3
    assert res["output_tokens"] == 3 and res["output_ids"][-1] == EOS_IDS[0]
    assert res["termination"] == "eos"
    ref = target_only_generate(Hooked(HFCachedLM(target_model), fn), PROMPT, 20, EOS_IDS, make_generator(14, "cpu"))
    assert ref["output_tokens"] == 3 and ref["termination"] == "eos" and ref["output_ids"][-1] == EOS_IDS[0]


def test_cap_handling(target_model, draft_model):
    t, d = HFCachedLM(target_model), HFCachedLM(target_model)
    res = speculative_generate(t, d, PROMPT, 4, 4, (), make_generator(15, "cpu"))
    assert [r["proposed"] for r in res["rounds"]] == [4]
    assert res["rounds"][0]["bonus_skipped_cap"] and res["output_tokens"] == 4 and res["termination"] == "cap"
    res = speculative_generate(t, d, PROMPT, 4, 6, (), make_generator(16, "cpu"))
    assert [r["proposed"] for r in res["rounds"]] == [4, 1]
    assert [r["emitted"] for r in res["rounds"]] == [5, 1]
    assert res["rounds"][1]["bonus_skipped_cap"] and res["output_tokens"] == 6
    t2 = Hooked(HFCachedLM(target_model), _forbid)
    d2 = Hooked(HFCachedLM(draft_model), _only_forbidden)
    res = speculative_generate(t2, d2, PROMPT, 4, 3, (), make_generator(17, "cpu"))
    assert [r["proposed"] for r in res["rounds"]] == [3, 2, 1] and res["output_tokens"] == 3
    ref = target_only_generate(HFCachedLM(target_model), PROMPT, 5, (), make_generator(18, "cpu"))
    assert ref["output_tokens"] == 5 and ref["termination"] == "cap" and ref["target_cache_len_end"] == len(PROMPT) + 4


def test_same_seed_same_output(target_model, draft_model):
    def run(seed):
        return speculative_generate(HFCachedLM(target_model), HFCachedLM(draft_model), PROMPT, 4, 16, EOS_IDS,
                                    make_generator(seed, "cpu"))
    a, b, c = run(21), run(21), run(22)
    assert a["output_ids"] == b["output_ids"]
    assert [r["accept_flags"] for r in a["rounds"]] == [r["accept_flags"] for r in b["rounds"]]
    assert a["output_ids"] != c["output_ids"]
    ra = target_only_generate(HFCachedLM(target_model), PROMPT, 8, EOS_IDS, make_generator(23, "cpu"))
    rb = target_only_generate(HFCachedLM(target_model), PROMPT, 8, EOS_IDS, make_generator(23, "cpu"))
    assert ra["output_ids"] == rb["output_ids"]


def test_target_only_matches_manual_loop(target_model):
    ref = target_only_generate(HFCachedLM(target_model), PROMPT, 6, (), make_generator(31, "cpu"))
    gen = make_generator(31, "cpu")
    ctx = list(PROMPT)
    with torch.inference_mode():
        for _ in range(6):
            ctx.append(sample_token(hf_full_logits(target_model, ctx)[-1], gen))
    assert ref["output_ids"] == ctx[len(PROMPT):]
