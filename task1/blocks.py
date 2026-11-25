"""Block-by-block execution on calibration sequences: block-0 input capture, per-projection input hooks.

A stream is a list of (hidden_states, layer_kwargs) batches. Block-0 inputs are captured by running the
model's embedding and mask/rotary preparation with a pre-hook on layer 0 that stops the forward; every
later block input is the previous block's output on the same batch with the same keyword arguments.
"""
from contextlib import contextmanager

import torch

from . import config as C


class _Stop(Exception):
    pass


def layers(model):
    return model.model.layers


def projections(block) -> dict:
    """The seven eligible projections of one decoder block in enumeration order."""
    return {n: getattr(block.self_attn if n in C.ATTN_PROJ else block.mlp, n) for n in C.ELIGIBLE}


@torch.inference_mode()
def capture_block_inputs(model, sequences, batch_size: int = C.CALIB_BATCH_SIZE) -> list:
    """[(hidden, kwargs)] for block 0, batches of consecutive sequences of equal length (no padding)."""
    lengths = {len(s["token_ids"]) for s in sequences}
    assert len(lengths) == 1, lengths
    dev = next(model.parameters()).device
    store = []

    def pre(_mod, args, kwargs):
        h = args[0] if args else kwargs.pop("hidden_states")
        kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        store.append((h, kw))
        raise _Stop

    handle = layers(model)[0].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        for a in range(0, len(sequences), batch_size):
            ids = torch.tensor([s["token_ids"] for s in sequences[a:a + batch_size]], dtype=torch.long, device=dev)
            try:
                model.model(input_ids=ids, use_cache=False)
            except _Stop:
                pass
            else:
                raise RuntimeError("layer-0 pre-hook did not fire")
    finally:
        handle.remove()
    assert len(store) == -(-len(sequences) // batch_size)
    return store


@torch.inference_mode()
def run_block(block, stream) -> list:
    return [(block(h, **kw), kw) for h, kw in stream]


@contextmanager
def input_hooks(block, fn, names=C.ELIGIBLE):
    """Call fn(name, x2d) with the [tokens, in] input of each named projection during forwards."""
    projs = projections(block)
    handles = []
    for n in names:
        def pre(_mod, args, n=n):
            x = args[0]
            fn(n, x.reshape(-1, x.shape[-1]))
        handles.append(projs[n].register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


class ChannelStats:
    """Per-input-channel sum of squares and sum of absolute values (FP32) over tokens."""

    def __init__(self, dim: int, device):
        self.sumsq = torch.zeros(dim, dtype=torch.float32, device=device)
        self.sumabs = torch.zeros(dim, dtype=torch.float32, device=device)
        self.n = 0

    def update(self, x2d: torch.Tensor):
        xf = x2d.float()
        self.sumsq += (xf * xf).sum(dim=0)
        self.sumabs += xf.abs().sum(dim=0)
        self.n += xf.shape[0]

    def l2(self) -> torch.Tensor:
        return self.sumsq.sqrt()

    def absmean(self) -> torch.Tensor:
        return self.sumabs / self.n


@torch.inference_mode()
def collect_stats(block, stream, names=C.ELIGIBLE):
    """Run the block on a stream; return ({name: ChannelStats}, output stream)."""
    projs = projections(block)
    stats = {n: ChannelStats(projs[n].in_features, projs[n].weight.device) for n in names}
    with input_hooks(block, lambda n, x: stats[n].update(x), names):
        out = run_block(block, stream)
    return stats, out
