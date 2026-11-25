"""Perplexity, HellaSwag and calibration NLL scoring shared by every Task 1 condition.

Logits are lm_head(final hidden state), as in Qwen2ForCausalLM.forward; the LM head is applied
per sequence in chunks of positions and the log-softmax is computed in FP32. Sums are float64.
"""
import math
from pathlib import Path

import numpy as np
import torch

from common.io import write_jsonl_gz

from . import config as C
from .data import HELLASWAG_PROMPT_FORMAT, HELLASWAG_TASK, encode_pair, format_hellaswag


def _device(model):
    return next(model.parameters()).device


@torch.inference_mode()
def score_spans(model, sequences, spans, batch_size: int, chunk: int = C.LOGPROB_CHUNK) -> list:
    """Sum of log p(x_t | x_<t) over t in [a, b) for each token sequence; returns [(logprob, n)].

    Each sequence is evaluated independently with input x[:b-1], right padding and an attention
    mask; batches are formed from sequences sorted by input length (longest first).
    """
    assert len(sequences) == len(spans)
    dev = _device(model)
    head = model.get_output_embeddings()
    items = []
    for i, (seq, (a, b)) in enumerate(zip(sequences, spans)):
        assert 1 <= a < b <= len(seq), (i, a, b, len(seq))  # the first token is never a target
        items.append((i, list(seq[: b - 1]), list(seq[a:b]), a))
    order = sorted(range(len(items)), key=lambda k: (-len(items[k][1]), k))
    out = [None] * len(items)
    for s in range(0, len(order), batch_size):
        batch = [items[k] for k in order[s:s + batch_size]]
        L = max(len(x[1]) for x in batch)
        ids = torch.zeros((len(batch), L), dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, (_, inp, _, _) in enumerate(batch):
            ids[r, :len(inp)] = torch.tensor(inp)
            mask[r, :len(inp)] = 1
        hidden = model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).last_hidden_state
        for r, (idx, _, tgt, a) in enumerate(batch):
            pos = torch.arange(a - 1, a - 1 + len(tgt), device=dev)  # logits at t-1 predict x_t
            tg = torch.tensor(tgt, device=dev)
            total = 0.0
            for c in range(0, len(tgt), chunk):
                h = hidden[r, pos[c:c + chunk]]
                lp = torch.log_softmax(head(h).float(), dim=-1)
                total += float(lp.gather(-1, tg[c:c + chunk, None]).double().sum())
            out[idx] = (total, len(tgt))
    return out


def perplexity(model, windows, batch_size: int = C.PPL_BATCH_SIZE, suite: str = None, expected_n=C.PPL_SCORED_TOKENS):
    """Per-window NLL and pooled PPL = exp(sum NLL / N); windows are independent sequences."""
    spans = [tuple(w["score_target_offsets"]) for w in windows]
    res = score_spans(model, [w["token_ids"] for w in windows], spans, batch_size)
    rows = []
    for w, (lp, n) in zip(windows, res):
        assert n == w["scored_tokens"]
        rows.append({"eval_id": C.EVAL_ID, "suite": suite or w["source"], "window_index": w["window_index"],
                     "source": w["source"], "source_row": w["source_row"], "token_offset": w["token_offset"],
                     "nll_sum": -lp, "n_scored": n})
    N = sum(r["n_scored"] for r in rows)
    if expected_n is not None:
        assert N == expected_n, (N, expected_n)
    nll = sum(r["nll_sum"] for r in rows)
    mean = nll / N
    summary = {"eval_id": C.EVAL_ID, "suite": suite or windows[0]["source"], "n_windows": len(rows),
               "N": N, "nll_sum": nll, "mean_nll": mean, "ppl": math.exp(mean), "batch_size": batch_size}
    return rows, summary


def hellaswag_requests(tokenizer, records) -> list:
    """Formatted examples with the token encoding of each (context, continuation) pair."""
    out = []
    for rec in records:
        ex = format_hellaswag(rec)
        enc = [encode_pair(tokenizer, ex["context"], cont) for cont in ex["continuations"]]
        ex["context_tokens"] = [e[0] for e in enc]
        ex["continuation_tokens"] = [e[1] for e in enc]
        ex["context_is_prefix"] = [e[2] for e in enc]
        out.append(ex)
    return out


def hellaswag(model, tokenizer, records, scope: str, batch_size: int = C.HELLASWAG_BATCH_SIZE, expected_n=None):
    """Zero-shot HellaSwag: summed continuation log likelihoods, acc and character-normalized acc."""
    reqs = hellaswag_requests(tokenizer, records)
    if expected_n is not None:
        assert len(reqs) == expected_n, (len(reqs), expected_n)
    seqs, spans = [], []
    for ex in reqs:
        for ctx, cont in zip(ex["context_tokens"], ex["continuation_tokens"]):
            assert len(ctx) >= 1 and len(cont) >= 1
            seqs.append(ctx + cont)
            spans.append((len(ctx), len(ctx) + len(cont)))
    res = score_spans(model, seqs, spans, batch_size)
    rows = []
    n_acc = n_norm = 0
    for i, ex in enumerate(reqs):
        lls = [res[4 * i + j][0] for j in range(4)]
        chars = [len(c) for c in ex["choices"]]
        ntok = [len(t) for t in ex["continuation_tokens"]]
        ll = np.array(lls, dtype=np.float64)
        pred = int(np.argmax(ll))
        pred_norm = int(np.argmax(ll / np.array(chars, dtype=np.float64)))
        n_acc += pred == ex["label"]
        n_norm += pred_norm == ex["label"]
        rows.append({"eval_id": C.EVAL_ID, "order": ex["order"], "source_row_index": ex["source_row_index"],
                     "label": ex["label"], "loglik": lls, "cont_chars": chars, "cont_tokens": ntok,
                     "ctx_tokens": len(ex["context_tokens"][0]), "pred_acc": pred, "pred_acc_norm": pred_norm,
                     "correct_acc": pred == ex["label"], "correct_acc_norm": pred_norm == ex["label"]})
    n = len(rows)
    summary = {"eval_id": C.EVAL_ID, "scope": scope, "n": n, "acc": n_acc / n, "acc_norm": n_norm / n,
               "n_correct_acc": n_acc, "n_correct_acc_norm": n_norm,
               "n_context_not_prefix": sum(not x for ex in reqs for x in ex["context_is_prefix"]),
               "orders_first_last": [rows[0]["order"], rows[-1]["order"]],
               "task": HELLASWAG_TASK, "prompt_format": HELLASWAG_PROMPT_FORMAT, "batch_size": batch_size}
    return rows, summary


def calib_nll(model, sequences, batch_size: int = C.CALIB_BATCH_SIZE, split: str = None):
    """Pooled NLL over sequences with positions 1..len-1 scored."""
    spans = [(1, len(s["token_ids"])) for s in sequences]
    res = score_spans(model, [s["token_ids"] for s in sequences], spans, batch_size)
    rows = [{"eval_id": C.EVAL_ID, "split": s["split"], "seq_index": i, "source_row": s["source_row"],
             "token_offset": s["token_offset"], "nll_sum": -lp, "n_scored": n}
            for i, (s, (lp, n)) in enumerate(zip(sequences, res))]
    N = sum(r["n_scored"] for r in rows)
    nll = sum(r["nll_sum"] for r in rows)
    summary = {"eval_id": C.EVAL_ID, "split": split or sequences[0]["split"], "n_sequences": len(rows), "N": N,
               "nll_sum": nll, "mean_nll": nll / N, "batch_size": batch_size}
    return rows, summary


def write_rows(path, rows) -> str:
    """Write per-example rows as gzip JSONL (mtime 0); refuses an existing path."""
    write_jsonl_gz(Path(path), rows)
    return str(path)
