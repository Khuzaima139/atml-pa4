"""Small cached language-model interface used by the Task 3 samplers.

Interface (duck-typed): ``reset()`` empties the cache; ``forward(ids)`` feeds new tokens after the cached ones and
returns fp32 logits of shape [len(ids), V] (row i predicts the token after ids[i]); ``crop(n)`` keeps the first n
cached positions; ``length`` is the number of cached positions; ``device``.
"""
import torch


class HFCachedLM:
    """Hugging Face causal LM (Qwen2) with a DynamicCache, batch size 1, no padding."""

    def __init__(self, model, expect_vocab=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.expect_vocab = expect_vocab
        self.cache = None
        self.reset()

    def reset(self):
        from transformers import DynamicCache

        self.cache = DynamicCache()

    @property
    def length(self) -> int:
        return int(self.cache.get_seq_length())

    def forward(self, ids) -> torch.Tensor:
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("forward needs at least one token")
        start = self.length
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        pos = torch.arange(start, start + len(ids), device=self.device)
        out = self.model(input_ids=inp, past_key_values=self.cache, use_cache=True, cache_position=pos)
        self.cache = out.past_key_values
        logits = out.logits[0].float()
        if self.expect_vocab is not None and logits.shape[-1] != self.expect_vocab:
            raise ValueError(f"logit rows {logits.shape[-1]} != expected vocabulary {self.expect_vocab}")
        return logits

    def crop(self, n: int):
        if n < 0 or n > self.length:
            raise ValueError(f"crop to {n} with {self.length} cached positions")
        if n < self.length:
            self.cache.crop(n)


def hf_full_logits(model, ids) -> torch.Tensor:
    """Fresh full forward without a cache: fp32 logits [len(ids), V]."""
    device = next(model.parameters()).device
    inp = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=device)
    return model(input_ids=inp, use_cache=False).logits[0].float()


class MarkovLM:
    """First-order Markov model over S symbols as a cached LM.

    Input ids are symbols 0..S-1 or the start symbol S. The row after the start symbol is log(initial), the row
    after symbol s is log(transition[s]). The cache is the list of fed tokens.
    """

    def __init__(self, initial, transition):
        init = torch.as_tensor(initial, dtype=torch.float64)
        trans = torch.as_tensor(transition, dtype=torch.float64)
        self.n_symbols = int(init.numel())
        self.start = self.n_symbols
        rows = torch.cat([trans, init[None, :]], dim=0)
        self.logrows = torch.log(rows).float()
        self.device = torch.device("cpu")
        self.tokens = []

    def reset(self):
        self.tokens = []

    @property
    def length(self) -> int:
        return len(self.tokens)

    def forward(self, ids) -> torch.Tensor:
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("forward needs at least one token")
        self.tokens.extend(ids)
        return self.logrows[torch.tensor(ids, dtype=torch.long)]

    def crop(self, n: int):
        if n < 0 or n > self.length:
            raise ValueError(f"crop to {n} with {self.length} cached positions")
        del self.tokens[n:]
