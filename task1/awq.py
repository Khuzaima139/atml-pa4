"""Activation-aware per-channel scale search (AWQ-style), folding, and the original-basis effective weights.

Four scale groups per block, searched on dense-model calibration inputs and the original weights:
  qkv    q,k,v share s over the input_layernorm output; inverse folded into input_layernorm.weight;
  o      o_proj input, one scale per (KV head, head dim) shared by the query heads of that KV head;
         inverse folded into v_proj rows and v_proj bias;
  gateup gate,up share s over the post_attention_layernorm output; inverse folded into its weight;
  down   down_proj input; inverse folded into up_proj rows.
s = (mean|x|)^alpha clamped at 1e-4, divided by sqrt(max(s) * min(s)) (AWQ reference normalization);
alpha on ALPHA_GRID; loss = sum over the group's projections of ||X (Q(W diag(s)) diag(1/s) - W)^T||_F^2
on all calibration tokens, with Q the task1.quant quantizer at the target bit width.
"""
import time

import torch

from . import config as C
from .blocks import ChannelStats, input_hooks, layers, projections, run_block
from .quant import fake_quant

ALPHA_GRID = tuple(round(0.05 * i, 2) for i in range(20))
SEARCH_BITS = (2, 3, 4, 8)
SCALE_FLOOR = 1e-4
TOKEN_CHUNK = 8192
GROUPS = {
    "qkv": {"projs": ("q_proj", "k_proj", "v_proj"), "input_of": "q_proj", "fold": "input_layernorm.weight"},
    "o": {"projs": ("o_proj",), "input_of": "o_proj", "fold": "self_attn.v_proj (rows and bias)"},
    "gateup": {"projs": ("gate_proj", "up_proj"), "input_of": "gate_proj", "fold": "post_attention_layernorm.weight"},
    "down": {"projs": ("down_proj",), "input_of": "down_proj", "fold": "mlp.up_proj (rows)"},
}
AWQ_CONVENTION = (
    "groups qkv/o/gateup/down; importance mean|x| over calibration tokens (o: also over the query heads of each "
    "KV head); s=clamp(imp^alpha, min=1e-4)/sqrt(max*min); alpha grid 0..0.95 step 0.05; loss = summed squared "
    "output error of the group's projections on dense calibration inputs with original weights; argmin, ties "
    "to the lower alpha; folds: input_layernorm/=s_qkv, v rows and bias/=s_o, post_attention_layernorm/=s_gateup, "
    "up rows/=s_down, consumer columns*=s; no clipping search"
)


def geometry(config) -> dict:
    hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    nh, nkv = config.num_attention_heads, config.num_key_value_heads
    assert nh % nkv == 0
    return {"n_heads": nh, "n_kv": nkv, "head_dim": hd, "n_rep": nh // nkv}


def importance(group: str, absmean: torch.Tensor, geom) -> torch.Tensor:
    """Per-scale importance from the per-channel mean |x| of the group's input."""
    if group == "o":
        return absmean.float().view(geom["n_kv"], geom["n_rep"], geom["head_dim"]).mean(dim=1).reshape(-1)
    return absmean.float()


def expand(group: str, s: torch.Tensor, geom) -> torch.Tensor:
    """Distinct scales -> one scale per input column of the group's projections."""
    if group == "o":
        return s.view(geom["n_kv"], geom["head_dim"]).repeat_interleave(geom["n_rep"], dim=0).reshape(-1)
    return s


def scales_for(imp: torch.Tensor, alpha: float) -> torch.Tensor:
    s = imp.float().pow(alpha).clamp(min=SCALE_FLOOR)
    return s / (s.max() * s.min()).sqrt()


def _prefix(name: str) -> str:
    return ("self_attn." if name in C.ATTN_PROJ else "mlp.") + name


def col_row_scales(block_scales: dict, geom) -> tuple:
    """Input-column and output-row scale vectors of each projection from the four distinct scale vectors."""
    sa, sb, sc, sd = (block_scales[g].float() for g in ("qkv", "o", "gateup", "down"))
    col = {"q_proj": sa, "k_proj": sa, "v_proj": sa, "o_proj": expand("o", sb, geom),
           "gate_proj": sc, "up_proj": sc, "down_proj": sd}
    row = {"v_proj": sb, "up_proj": sd}
    return col, row


def fold_weight(w: torch.Tensor, col, row=None) -> torch.Tensor:
    out = w.float() * col.to(w.device)[None, :]
    if row is not None:
        out = out / row.to(w.device)[:, None]
    return out


def unfold_weight(e: torch.Tensor, col, row=None) -> torch.Tensor:
    """Executed weight mapped back to the original basis."""
    out = e.float()
    if row is not None:
        out = out * row.to(e.device)[:, None]
    return out / col.to(e.device)[None, :]


@torch.no_grad()
def folded_params(block, block_scales: dict, geom) -> dict:
    """FP32 folded tensors of one block, keyed by the block-relative parameter name."""
    col, row = col_row_scales(block_scales, geom)
    out = {}
    for n, m in projections(block).items():
        out[_prefix(n) + ".weight"] = fold_weight(m.weight, col[n], row.get(n))
    dev = block.input_layernorm.weight.device
    out["input_layernorm.weight"] = block.input_layernorm.weight.float() / col["q_proj"].to(dev)
    out["post_attention_layernorm.weight"] = block.post_attention_layernorm.weight.float() / col["gate_proj"].to(dev)
    out["self_attn.v_proj.bias"] = block.self_attn.v_proj.bias.float() / row["v_proj"].to(dev)
    return out


NON_ELIGIBLE_FOLDED = ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.v_proj.bias")


def snapshot(block, names) -> dict:
    params = dict(block.named_parameters())
    return {n: params[n].detach().clone() for n in names}


@torch.no_grad()
def assign(block, tensors: dict):
    params = dict(block.named_parameters())
    for n, t in tensors.items():
        params[n].copy_(t.to(params[n].dtype))


@torch.inference_mode()
def capture_group_inputs(block, stream):
    """Dense inputs of each scale group ([tokens, in], model dtype), channel stats of all 7 inputs, block output."""
    projs = projections(block)
    T = sum(h.shape[0] * h.shape[1] for h, _ in stream)
    dt, dev = stream[0][0].dtype, stream[0][0].device
    bufs = {g: torch.empty((T, projs[s["input_of"]].in_features), dtype=dt, device=dev) for g, s in GROUPS.items()}
    offs = {g: 0 for g in GROUPS}
    stats = {n: ChannelStats(projs[n].in_features, dev) for n in C.ELIGIBLE}
    by_input = {s["input_of"]: g for g, s in GROUPS.items()}

    def fn(n, x):
        stats[n].update(x)
        g = by_input.get(n)
        if g is not None:
            bufs[g][offs[g]:offs[g] + x.shape[0]] = x
            offs[g] += x.shape[0]

    with input_hooks(block, fn):
        out = run_block(block, stream)
    assert all(o == T for o in offs.values())
    return bufs, stats, out


def _sq_out(X, D, compute_dtype, chunk) -> float:
    Dc = D.to(compute_dtype)
    total = 0.0
    for a in range(0, X.shape[0], chunk):
        y = X[a:a + chunk].to(compute_dtype) @ Dc.T
        total += float(y.float().square().sum(dtype=torch.float64))
    return total


@torch.inference_mode()
def search_group(weights, X, imp, group: str, geom, bits_list=SEARCH_BITS, alphas=ALPHA_GRID,
                 compute_dtype=torch.float32, chunk=TOKEN_CHUNK) -> dict:
    """Loss curve over alphas for each bit width; returns {b: {alpha, index, losses, scales}} and the reference."""
    Wcat = torch.cat([w.float() for w in weights], dim=0)
    ref = _sq_out(X, Wcat, compute_dtype, chunk)
    losses = {b: [] for b in bits_list}
    for alpha in alphas:
        s = scales_for(imp, alpha)
        sc = expand(group, s, geom)
        for b in bits_list:
            D = torch.cat([fake_quant(w.float() * sc[None, :], b) / sc[None, :] - w.float() for w in weights], dim=0)
            loss = _sq_out(X, D, compute_dtype, chunk)
            assert loss == loss and loss != float("inf"), (group, alpha, b)
            losses[b].append(loss)
    out = {}
    for b in bits_list:
        i = min(range(len(alphas)), key=lambda k: (losses[b][k], k))
        out[b] = {"alpha": alphas[i], "index": i, "losses": losses[b], "scales": scales_for(imp, alphas[i]).cpu()}
    return {"by_bits": out, "loss_ref": ref}


def _out_err(y, y0, h) -> dict:
    e = (y.float() - y0.float())
    return {"max_abs_err_over_max_abs_out": float(e.abs().max() / y0.float().abs().max()),
            "rel_fro_out": float(e.norm() / y0.float().norm()),
            "rel_fro_update": float(e.norm() / (y0.float() - h.float()).norm())}


@torch.inference_mode()
def check_preservation(block, batch, ref_out, block_scales: dict, geom) -> dict:
    """Block output after folding (no quantization) versus the original output on one batch."""
    fold = folded_params(block, block_scales, geom)
    saved = snapshot(block, list(fold))
    try:
        assign(block, fold)
        h, kw = batch
        y = block(h, **kw)
        return _out_err(y, ref_out, h)
    finally:
        assign(block, saved)


@torch.inference_mode()
def search_model(model, stream, bits_list=SEARCH_BITS, alphas=ALPHA_GRID, chunk=TOKEN_CHUNK, log=None) -> dict:
    """Dense statistics, scale search for every block and bit width, and the preservation check."""
    geom = geometry(model.config)
    dev = next(model.parameters()).device
    # Search matmuls run in FP16 on CUDA with an FP16 model and in FP32 otherwise; squares are summed in FP64.
    cdt = torch.float16 if dev.type == "cuda" and stream[0][0].dtype == torch.float16 else torch.float32
    stats, entries, scales, preservation, timings = {}, [], {}, [], []
    for li, block in enumerate(layers(model)):
        tb = {"layer": li}
        t = time.perf_counter()
        X, st, out = capture_group_inputs(block, stream)
        tb["capture_s"] = time.perf_counter() - t
        for n, s in st.items():
            stats[(li, n)] = {"l2": s.l2().cpu(), "absmean": s.absmean().cpu(), "n_tokens": s.n}
        projs = projections(block)
        block_res = {}
        for g, spec in GROUPS.items():
            t = time.perf_counter()
            imp = importance(g, st[spec["input_of"]].absmean(), geom)
            res = search_group([projs[n].weight for n in spec["projs"]], X[g], imp, g, geom, bits_list, alphas,
                               cdt, chunk)
            block_res[g] = res
            for b, r in res["by_bits"].items():
                scales[(li, g, b)] = r["scales"]
                entries.append({"layer": li, "group": g, "bits": b, "alpha": r["alpha"], "alpha_index": r["index"],
                                "losses": r["losses"], "loss_ref": res["loss_ref"],
                                "rel_loss_chosen": r["losses"][r["index"]] / res["loss_ref"],
                                "rel_loss_alpha0": r["losses"][0] / res["loss_ref"]})
            tb[f"search_{g}_s"] = time.perf_counter() - t
        del X
        t = time.perf_counter()
        for b in bits_list:
            bs = {g: block_res[g]["by_bits"][b]["scales"].to(dev) for g in GROUPS}
            preservation.append({"layer": li, "bits": b, **check_preservation(block, stream[0], out[0][0], bs, geom)})
        tb["preservation_s"] = time.perf_counter() - t
        timings.append(tb)
        if log:
            log(li, tb)
        stream = out
    return {"stats": stats, "entries": entries, "scales": scales, "preservation": preservation,
            "block_timings": timings, "geometry": geom, "search_matmul_dtype": str(cdt).replace("torch.", "")}
