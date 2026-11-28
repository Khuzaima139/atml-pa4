"""The single sampling function of Task 3 (protocol D3) and a CUDA-synchronised timer.

Every Task 3 sampler (teacher collection, rollouts, banks, speculative decoding, target-only reference, standalone
evaluation) draws tokens through ``probs_from_logits`` and ``sample_from_weights``: an fp32 softmax of logits / 1.0
over every lm_head row, no top-k, no top-p, no repetition penalty, and ``torch.multinomial`` with an explicit
``torch.Generator`` per sequence. Batch size is 1 everywhere, so no padding exists.
"""
import time

import torch

SAMPLER_ID = "t3-sampler-v1"
VOCAB_SIZE = 151_936
EOS_IDS = (151645, 151643)
TEMPERATURE = 1.0

GENERATION_SETTINGS = {
    "sampler_id": SAMPLER_ID,
    "temperature": TEMPERATURE,
    "top_p": 1.0,
    "top_k": None,
    "repetition_penalty": None,
    "softmax_dtype": "float32",
    "vocab_rows": VOCAB_SIZE,
    "draw": "torch.multinomial with an explicit torch.Generator per sequence",
    "model_generate_used": False,
    "batch_size": 1,
    "padding": "none (batch size 1, no padding exists)",
    "eos_ids": list(EOS_IDS),
}


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """fp32 softmax of logits / TEMPERATURE over the last dimension (all rows)."""
    return torch.softmax(logits.float() / TEMPERATURE, dim=-1)


def sample_from_weights(weights: torch.Tensor, generator: torch.Generator) -> int:
    """One draw from a non-negative 1-D weight vector (normalised internally by torch.multinomial)."""
    return int(torch.multinomial(weights, 1, generator=generator).item())


def sample_token(logits: torch.Tensor, generator: torch.Generator) -> int:
    """Draw one token from a 1-D logit row with the shared law."""
    return sample_from_weights(probs_from_logits(logits), generator)


def uniform(generator: torch.Generator, device) -> float:
    """One U[0, 1) draw from the sequence generator (acceptance tests)."""
    return float(torch.rand((), generator=generator, device=device).item())


def make_generator(seed: int, device) -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


class Timer:
    """Wall-clock reads; synchronises CUDA before each read on a CUDA device, plain perf_counter otherwise."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"

    def now(self) -> float:
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()
