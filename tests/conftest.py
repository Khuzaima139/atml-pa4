import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import pytest  # noqa: E402

TINY_CONFIG = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=151_936, tie_word_embeddings=True,
    max_position_embeddings=4096, initializer_range=0.2,
)


@pytest.fixture(scope="session")
def real_tokenizer():
    """Tokenizer of the checkpoint at the locked revision, from the local Hugging Face cache."""
    from transformers import AutoTokenizer

    from task1 import config as C

    return AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.MODEL_REVISION, local_files_only=True)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, real_tokenizer):
    """Tiny random Qwen2 model with the real vocabulary size and tokenizer, saved once (FP32)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(6304)
    d = tmp_path_factory.mktemp("tiny_qwen2")
    Qwen2ForCausalLM(Qwen2Config(**TINY_CONFIG)).to(torch.float32).save_pretrained(d)
    real_tokenizer.save_pretrained(d)
    return d


@pytest.fixture()
def tiny_fp32(tiny_model_dir):
    from task1.model import load_model

    model, _ = load_model("cpu", tiny_model_dir, dtype="float32")
    return model

# Compression tests: widths that are multiples of the quantization group (128) with integer per-row zeros.
COMPRESS_CONFIG = dict(
    hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=512, vocab_size=151_936, tie_word_embeddings=True,
    max_position_embeddings=4096, initializer_range=0.05,
)


def _save_compress_model(d, tokenizer, n_layers: int):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(6305)
    model = Qwen2ForCausalLM(Qwen2Config(**{**COMPRESS_CONFIG, "num_hidden_layers": n_layers})).to(torch.float32)
    g = torch.Generator().manual_seed(6306)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            elif name.endswith(".bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def tiny_compress_dir(tmp_path_factory, real_tokenizer):
    """Tiny random Qwen2 (head_dim 64, 2 query heads per KV head) with random norm weights and biases (FP32)."""
    return _save_compress_model(tmp_path_factory.mktemp("tiny_qwen2_compress"), real_tokenizer, 2)


@pytest.fixture(scope="session")
def tiny_compress4_dir(tmp_path_factory, real_tokenizer):
    """The same recipe with 4 blocks, for block-allocation tests."""
    return _save_compress_model(tmp_path_factory.mktemp("tiny_qwen2_compress4"), real_tokenizer, 4)


@pytest.fixture()
def compress_fp32(tiny_compress_dir):
    from task1.model import load_model

    model, _ = load_model("cpu", tiny_compress_dir, dtype="float32")
    return model


@pytest.fixture()
def compress4_fp32(tiny_compress4_dir):
    from task1.model import load_model

    model, _ = load_model("cpu", tiny_compress4_dir, dtype="float32")
    return model
