import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.seeding import seed_everything


def _tiny():
    cfg = Qwen2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, vocab_size=512, tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(cfg).to(torch.float32).eval()


def test_fp32_forward_finite_logits():
    seed_everything(0)
    model = _tiny()
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.shape == (2, 16, 512)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_lm_head_tied_to_embeddings():
    model = _tiny()
    emb = model.get_input_embeddings().weight
    head = model.lm_head.weight
    assert head is emb or head.data_ptr() == emb.data_ptr()
