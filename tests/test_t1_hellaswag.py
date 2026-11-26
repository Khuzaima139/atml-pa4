import numpy as np
import pytest

from task1 import config as C
from task1 import evaluate as E
from task1.data import encode_pair, format_hellaswag, read_hellaswag
from task1.verify_lmeval import FP32_RUN, compare, lmeval_loglikelihoods, verify


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
    assert rep["acceptance_run"] == C.LMEVAL_ACCEPTANCE_RUN == FP32_RUN == "fp32_model"
    assert set(rep["runs"]) == {"softmax_dtype=default", "softmax_dtype=float32", "fp32_model"}
    for run in rep["runs"].values():
        assert run["n_candidates"] == 20 and run["max_abs_diff"] < 1e-4
        assert run["model_dtype"] == "torch.float32" and run["rule_check"]["result"] == "PASS"
        assert all(set(ex) == {"order", "ours_loglik", "lmeval_loglik", "max_abs_diff"} for ex in run["per_example"])
    assert rep["acceptance"] == "PASS" and rep["acceptance_detail"]["max_abs_diff"] == rep["runs"][FP32_RUN]["max_abs_diff"]
    assert rep["batch_size"] == 4 and rep["fp32_batch_size"] == C.LMEVAL_FP32_BATCH_SIZE
    assert {n: r["batch_size"] for n, r in rep["runs"].items()} == {
        "softmax_dtype=default": 4, "softmax_dtype=float32": 4, FP32_RUN: C.LMEVAL_FP32_BATCH_SIZE}
    assert rep["runs"][FP32_RUN]["ours_batch_size"] == C.LMEVAL_FP32_BATCH_SIZE


def test_memory_projection_formula(tiny_fp32, real_tokenizer, records):
    from task1.verify_lmeval import memory_line, memory_projection

    reqs = E.hellaswag_requests(real_tokenizer, records)
    max_len = max(len(c) + len(t) for ex in reqs for c, t in zip(ex["context_tokens"], ex["continuation_tokens"]))
    vocab = tiny_fp32.get_output_embeddings().weight.shape[0]
    m = memory_projection(tiny_fp32, real_tokenizer, records, 8, "float32")
    assert (m["max_length"], m["vocab"], m["element_bytes"]) == (max_len, vocab, 4)
    assert m["projected_bytes"] == 8 * max_len * vocab * 4 * 2
    assert m["device"] == "cpu" and m["free_bytes"] is None and m["total_bytes"] is None
    half = tiny_fp32.half()
    try:
        assert memory_projection(half, real_tokenizer, records, 8, None)["element_bytes"] == 2
        assert memory_projection(half, real_tokenizer, records, 8, "float32")["element_bytes"] == 4
    finally:
        tiny_fp32.float()
    line = memory_line("fp32_model", m)
    assert line.startswith("[lm_eval check] memory before run fp32_model: projected logits+log_softmax bytes ")
    assert f"(batch 8 x max length {max_len} x vocab {vocab} x 4 bytes x 2); free None of None bytes on cpu" in line


def _spy_fp32_run(monkeypatch, model, fail=False):
    """Record model.to calls and the batch sizes and model dtypes seen by both scorers in fp32_copy_run."""
    from task1 import verify_lmeval as V

    events = []
    cls = type(model)
    orig_to = cls.to

    def to(self, *a, **k):  # class-level, so the deep copy does not inherit a bound spy
        if self is model:
            events.append(("to", str(a[0]) if a else str(k.get("device"))))
        return orig_to(self, *a, **k)

    monkeypatch.setattr(cls, "to", to)
    orig_hs, orig_ll = E.hellaswag, V.lmeval_loglikelihoods

    def hs(m, tok, recs, scope, batch_size=C.HELLASWAG_BATCH_SIZE, expected_n=None):
        events.append(("ours", batch_size, str(next(m.parameters()).dtype)))
        return orig_hs(m, tok, recs, scope, batch_size=batch_size, expected_n=expected_n)

    def ll(m, tok, recs, batch_size, softmax_dtype=None):
        events.append(("lmeval", batch_size, str(next(m.parameters()).dtype), softmax_dtype))
        if fail:
            raise RuntimeError("simulated out of memory")
        return orig_ll(m, tok, recs, batch_size, softmax_dtype=softmax_dtype)

    monkeypatch.setattr(E, "hellaswag", hs)
    monkeypatch.setattr(V, "lmeval_loglikelihoods", ll)
    return events


def test_fp32_copy_run_offloads_loaded_model_and_uses_fp32_batch(tiny_model_dir, real_tokenizer, records,
                                                                  monkeypatch):
    """The loaded model goes to CPU before the copy is made and back to its device afterwards; both scorers on the
    float32 copy use LMEVAL_FP32_BATCH_SIZE."""
    from task1.model import Pristine, load_model
    from task1.verify_lmeval import fp32_copy_run

    model, _ = load_model("cpu", tiny_model_dir)
    pristine = Pristine(model)
    events = _spy_fp32_run(monkeypatch, model)
    cmp, rows32 = fp32_copy_run(model, real_tokenizer, records)
    b = C.LMEVAL_FP32_BATCH_SIZE
    assert events == [("to", "cpu"), ("ours", b, "torch.float32"), ("lmeval", b, "torch.float32", "float32"),
                      ("to", "cpu")]
    assert cmp["batch_size"] == cmp["ours_batch_size"] == cmp["memory_projection"]["batch_size"] == b
    assert cmp["device"] == "cpu" and cmp["loaded_model_offloaded_to"] == "cpu"
    assert sorted({str(p.dtype) for p in model.parameters()}) == ["torch.float16"]
    assert pristine.verify(model) == len(pristine.names) and len(rows32) == len(records)


def test_fp32_copy_run_restores_loaded_model_on_error(tiny_model_dir, real_tokenizer, records, monkeypatch):
    from task1.model import Pristine, load_model
    from task1.verify_lmeval import fp32_copy_run

    model, _ = load_model("cpu", tiny_model_dir)
    pristine = Pristine(model)
    events = _spy_fp32_run(monkeypatch, model, fail=True)
    with pytest.raises(RuntimeError, match="simulated out of memory"):
        fp32_copy_run(model, real_tokenizer, records)
    assert events[0] == ("to", "cpu") and events[-1] == ("to", "cpu") and len(events) == 4
    assert sorted({str(p.dtype) for p in model.parameters()}) == ["torch.float16"]
    assert pristine.verify(model) == len(pristine.names)


def test_fp32_copy_acceptance_leaves_fp16_model_unchanged(tiny_model_dir, real_tokenizer, records):
    """Acceptance uses the float32 copy (its own task1.evaluate rows); the FP16 model keeps dtype and fingerprints."""
    from task1.model import Pristine, load_model

    model, _ = load_model("cpu", tiny_model_dir)
    pristine = Pristine(model)
    rows, _ = E.hellaswag(model, real_tokenizer, records, scope="full", batch_size=8)
    rep = verify(model, real_tokenizer, records, rows, batch_size=4)
    assert sorted({str(p.dtype) for p in model.parameters()}) == ["torch.float16"]
    assert pristine.verify(model) == len(pristine.names)
    fp = rep["runs"][FP32_RUN]
    assert fp["model_dtype"] == "torch.float32" and fp["ours_from"].endswith("float32 copy")
    for name in ("softmax_dtype=default", "softmax_dtype=float32"):
        assert rep["runs"][name]["model_dtype"] == "torch.float16"
        assert [ex["ours_loglik"] for ex in rep["runs"][name]["per_example"]] == [r["loglik"] for r in rows]
    # The FP32-copy run compares task1.evaluate on the copy (not the FP16 rows) with lm_eval on the copy.
    assert [ex["ours_loglik"] for ex in fp["per_example"]] != [r["loglik"] for r in rows]
    assert fp["max_abs_diff"] < 1e-4 and rep["acceptance"] == "PASS"


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
