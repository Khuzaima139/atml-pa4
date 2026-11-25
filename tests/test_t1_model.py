import pytest
import torch

from conftest import TINY_CONFIG
from task1 import config as C
from task1.model import (Pristine, check_tied, count_params, enumerate_eligible, fingerprint, load_model,
                         nonzeros_per_matrix)

H, I, L = TINY_CONFIG["hidden_size"], TINY_CONFIG["intermediate_size"], TINY_CONFIG["num_hidden_layers"]
HD = H // TINY_CONFIG["num_attention_heads"]
KV = TINY_CONFIG["num_key_value_heads"] * HD
V = TINY_CONFIG["vocab_size"]


def test_load_model_fp16_sdpa(tiny_model_dir):
    model, info = load_model("cpu", tiny_model_dir)
    assert info["parameter_dtypes"] == ["torch.float16"]
    assert info["attn_implementation_effective"] == "sdpa"
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_enumeration_order_count_and_shapes(tiny_fp32):
    el = enumerate_eligible(tiny_fp32)
    assert len(el) == L * 7
    assert [(e.layer, e.name) for e in el] == [(li, n) for li in range(L) for n in C.ELIGIBLE]
    assert el[0].qualname == "model.layers.0.self_attn.q_proj"
    assert el[-1].qualname == f"model.layers.{L - 1}.mlp.down_proj"
    want = {"q_proj": (H, H), "k_proj": (KV, H), "v_proj": (KV, H), "o_proj": (H, H),
            "gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    for e in el:
        assert tuple(e.module.weight.shape) == want[e.name]


def test_exclusions_and_tied_storage(tiny_fp32):
    el = enumerate_eligible(tiny_fp32)
    ptrs = {e.module.weight.data_ptr() for e in el}
    emb = tiny_fp32.get_input_embeddings().weight
    head = tiny_fp32.lm_head.weight
    assert head.data_ptr() == emb.data_ptr() and check_tied(tiny_fp32)
    excluded = [p for n, p in tiny_fp32.named_parameters() if n.endswith(".bias") or "norm" in n or "embed" in n]
    assert excluded and all(p.data_ptr() not in ptrs for p in excluded)
    assert head.data_ptr() not in ptrs


def test_untied_head_detected(tiny_fp32):
    tiny_fp32.lm_head.weight = torch.nn.Parameter(tiny_fp32.lm_head.weight.detach().clone())
    with pytest.raises(AssertionError):
        check_tied(tiny_fp32)


def test_count_params(tiny_fp32):
    c = count_params(tiny_fp32)
    per_block = H * H + 2 * KV * H + H * H + 3 * I * H
    assert c["eligible"] == L * per_block
    assert c["embedding"] == V * H
    assert c["norms"] == (2 * L + 1) * H
    assert c["biases"] == L * (H + 2 * KV)
    assert c["other"] == 0 and c["other_names"] == []
    assert c["total"] == c["eligible"] + c["embedding"] + c["norms"] + c["biases"]
    assert c["non_eligible"] == c["total"] - c["eligible"]
    assert c["eligible_matrices"] == L * 7 and c["eligible_exact_zeros"] == 0
    el = enumerate_eligible(tiny_fp32)
    el[0].module.weight.data[0, :5] = 0
    assert count_params(tiny_fp32)["eligible_exact_zeros"] == 5
    assert nonzeros_per_matrix(tiny_fp32)[0] == H * H - 5


def test_restore_returns_identical_fingerprints(tiny_fp32):
    original = {n: p.detach().clone() for n, p in tiny_fp32.named_parameters()}
    pr = Pristine(tiny_fp32)
    assert pr.verify(tiny_fp32) == len(original)
    el = enumerate_eligible(tiny_fp32)
    with torch.no_grad():
        el[3].module.weight.mul_(0.5)
        el[5].module.weight[:, :7] = 0
        tiny_fp32.model.norm.weight.add_(1.0)
        tiny_fp32.model.layers[0].self_attn.q_proj.bias.zero_()
        tiny_fp32.get_input_embeddings().weight[0].zero_()
    with pytest.raises(AssertionError):
        pr.verify(tiny_fp32)
    assert pr.restore(tiny_fp32) == len(original)
    for n, p in tiny_fp32.named_parameters():
        assert torch.equal(p, original[n]), n
        assert fingerprint(p) == pr.fp[n]
    assert check_tied(tiny_fp32)


def test_fingerprint_counts_zeros():
    t = torch.tensor([0.0, 1.0, -2.0, 0.0])
    assert fingerprint(t) == (-1.0, 5.0, 2)
