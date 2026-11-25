import numpy as np
import pytest

from task1 import config as C
from task1 import evaluate as E
from task1.data import encode_pair, format_hellaswag, read_hellaswag
from task1.verify_lmeval import compare, lmeval_loglikelihoods, verify


@pytest.fixture(scope="module")
def records():
    return read_hellaswag(limit=5)


def test_formatting_matches_lm_eval_process_docs(records):
    from datasets import Dataset
    from lm_eval.tasks.hellaswag.utils import process_docs

    docs = process_docs(Dataset.from_list([r["example"] for r in records]))
    for rec, doc in zip(records, docs):
        ex = format_hellaswag(rec)
        assert ex["context"] == doc["query"]
        assert ex["choices"] == doc["choices"]
        assert ex["label"] == doc["gold"] == int(rec["example"]["label"])
        assert ex["continuations"] == [" " + c for c in doc["choices"]]
    first = format_hellaswag(records[0])
    assert first["context"] == ("Clean and jerk: A lady walks to a barbell. She bends down and grabs the pole. "
                                "The lady")
    assert first["label"] == 3


def test_encode_pair_matches_hflm(tiny_fp32, real_tokenizer, records):
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=tiny_fp32, tokenizer=real_tokenizer, batch_size=1)
    for rec in records:
        ex = format_hellaswag(rec)
        for cont in ex["continuations"]:
            ctx, cnt, is_prefix = encode_pair(real_tokenizer, ex["context"], cont)
            assert (ctx, cnt) == tuple(lm._encode_pair(ex["context"], cont))
            assert is_prefix
    # Trailing whitespace of the context moves to the continuation.
    assert encode_pair(real_tokenizer, "A cat ", "sat")[:2] == tuple(lm._encode_pair("A cat ", "sat"))


def test_loglik_matches_hflm(tiny_fp32, real_tokenizer, records):
    rows, summ = E.hellaswag(tiny_fp32, real_tokenizer, records, scope="full", batch_size=3)
    theirs = lmeval_loglikelihoods(tiny_fp32, real_tokenizer, records, batch_size=4)
    ours = [x for r in rows for x in r["loglik"]]
    assert np.max(np.abs(np.array(ours) - np.array(theirs))) < 1e-4
    cmp = compare(rows, theirs)
    assert cmp["argmax_disagreements_acc"] == 0 and cmp["argmax_disagreements_acc_norm"] == 0
    assert cmp["max_abs_diff"] < 1e-4


def test_verify_report(tiny_fp32, real_tokenizer, records):
    rows, _ = E.hellaswag(tiny_fp32, real_tokenizer, records, scope="full", batch_size=8)
    rep = verify(tiny_fp32, real_tokenizer, records, rows, batch_size=4)
    assert rep["lm_eval_version"] == "0.4.13" and rep["eval_id"] == C.EVAL_ID
    assert rep["acceptance_run"] == C.LMEVAL_ACCEPTANCE_RUN == "softmax_dtype=float32"
    assert set(rep["runs"]) == {"softmax_dtype=default", "softmax_dtype=float32"}
    for run in rep["runs"].values():
        assert run["n_candidates"] == 20 and run["max_abs_diff"] < 1e-4


def test_rows_and_summary(tiny_fp32, real_tokenizer, records):
    rows, summ = E.hellaswag(tiny_fp32, real_tokenizer, records, scope="full", batch_size=8, expected_n=5)
    keys = {"eval_id", "order", "source_row_index", "label", "loglik", "cont_chars", "cont_tokens", "ctx_tokens",
            "pred_acc", "pred_acc_norm", "correct_acc", "correct_acc_norm"}
    for r, rec in zip(rows, records):
        assert set(r) == keys
        ex = format_hellaswag(rec)
        assert r["order"] == rec["order"] and r["source_row_index"] == rec["source_row_index"]
        assert r["cont_chars"] == [len(c) for c in ex["choices"]]
        assert len(r["loglik"]) == 4 and all(isinstance(x, float) for x in r["loglik"])
        assert r["pred_acc"] == int(np.argmax(r["loglik"]))
        assert r["pred_acc_norm"] == int(np.argmax(np.array(r["loglik"]) / np.array(r["cont_chars"])))
    assert summ["n"] == 5 and summ["n_context_not_prefix"] == 0
    assert summ["acc"] == sum(r["correct_acc"] for r in rows) / 5
    assert summ["acc_norm"] == sum(r["correct_acc_norm"] for r in rows) / 5
    assert "character" in summ["prompt_format"] or "characters" in summ["prompt_format"]
    with pytest.raises(AssertionError):
        E.hellaswag(tiny_fp32, real_tokenizer, records, scope="full", expected_n=2000)


def test_acc_norm_uses_characters_not_tokens(monkeypatch, real_tokenizer):
    rec = read_hellaswag(limit=1)
    ex = E.hellaswag_requests(real_tokenizer, rec)[0]
    chars = np.array([len(c) for c in ex["choices"]], dtype=float)
    toks = np.array([len(t) for t in ex["continuation_tokens"]], dtype=float)
    ratio = chars / toks
    j = int(np.argmax(ratio))
    assert ratio.min() < 0.9 * ratio.max()  # precondition: char and token normalization can disagree
    r = np.ones(4)
    r[j] = 0.9
    lls = -chars * r  # per-character score is best for j; per-token score is best elsewhere
    assert int(np.argmax(lls / toks)) != j

    def fake(model, seqs, spans, batch_size, chunk=None):
        return [(float(x), b - a) for x, (a, b) in zip(lls, spans)]

    monkeypatch.setattr(E, "score_spans", fake)
    rows, _ = E.hellaswag(None, real_tokenizer, rec, scope="full")
    assert rows[0]["pred_acc_norm"] == j
    assert rows[0]["pred_acc"] == int(np.argmax(lls))
