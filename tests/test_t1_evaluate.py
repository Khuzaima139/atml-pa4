import math

import pytest
import torch

from common.io import read_jsonl
from task1 import config as C
from task1.data import read_sequences, read_windows
from task1.evaluate import calib_nll, perplexity, score_spans, write_rows


@pytest.fixture(scope="module")
def windows():
    """Tiny fixture cut from the real WikiText-2 windows: short and full-length windows, in file order."""
    ws = read_windows("wikitext2_test")
    short = [w for w in ws if len(w["token_ids"]) <= 40][:4]
    long = [w for w in ws if len(w["token_ids"]) == 512][:1]
    mid = [w for w in ws if 100 <= len(w["token_ids"]) <= 200][:2]
    return sorted(short + long + mid, key=lambda w: w["window_index"])


@torch.no_grad()
def slow_reference(model, w):
    """Unbatched, token by token from the full-sequence logits."""
    ids = torch.tensor([w["token_ids"]])
    logits = model(input_ids=ids).logits[0].float()
    a, b = w["score_target_offsets"]
    total = 0.0
    for t in range(a, b):
        total -= float(torch.log_softmax(logits[t - 1], dim=-1)[w["token_ids"][t]])
    return total, b - a


def test_perplexity_matches_slow_reference(tiny_fp32, windows):
    rows, summ = perplexity(tiny_fp32, windows, batch_size=3, expected_n=None)
    ref_nll = ref_n = 0
    for r, w in zip(rows, windows):
        nll, n = slow_reference(tiny_fp32, w)
        assert r["n_scored"] == n
        assert r["nll_sum"] / n == pytest.approx(nll / n, abs=1e-5)
        ref_nll += nll
        ref_n += n
    assert summ["N"] == ref_n
    assert summ["mean_nll"] == pytest.approx(ref_nll / ref_n, abs=1e-5)
    assert summ["ppl"] == pytest.approx(math.exp(summ["mean_nll"]), rel=1e-12)


def test_padding_invariance(tiny_fp32, windows):
    batched, sb = perplexity(tiny_fp32, windows, batch_size=len(windows), expected_n=None)
    single, ss = perplexity(tiny_fp32, windows, batch_size=1, expected_n=None)
    for a, b in zip(batched, single):
        assert a["n_scored"] == b["n_scored"]
        assert a["nll_sum"] / a["n_scored"] == pytest.approx(b["nll_sum"] / b["n_scored"], abs=1e-5)
    assert sb["mean_nll"] == pytest.approx(ss["mean_nll"], abs=1e-5)


def test_first_token_never_scored(tiny_fp32, windows):
    for w in windows:
        assert w["score_target_offsets"][0] == 1
    rows, _ = perplexity(tiny_fp32, windows, batch_size=2, expected_n=None)
    assert [r["n_scored"] for r in rows] == [len(w["token_ids"]) - 1 for w in windows]
    with pytest.raises(AssertionError):
        score_spans(tiny_fp32, [[1, 2, 3]], [(0, 3)], batch_size=1)
    # Two-token sequence: exactly one target, -log p(x1 | x0).
    (lp, n), = score_spans(tiny_fp32, [[100, 200]], [(1, 2)], batch_size=1)
    with torch.no_grad():
        ref = torch.log_softmax(tiny_fp32(input_ids=torch.tensor([[100]])).logits[0, 0].float(), -1)[200]
    assert n == 1 and lp == pytest.approx(float(ref), abs=1e-5)


def test_n_matches_and_expected_n_checked(tiny_fp32, windows):
    rows, summ = perplexity(tiny_fp32, windows, batch_size=4, expected_n=None)
    assert summ["N"] == sum(w["scored_tokens"] for w in windows)
    assert summ["n_windows"] == len(windows)
    with pytest.raises(AssertionError):
        perplexity(tiny_fp32, windows, batch_size=4, expected_n=C.PPL_SCORED_TOKENS)


def test_full_files_have_50000_scored_tokens():
    for suite in ("wikitext2_test", "c4_validation"):
        assert sum(w["scored_tokens"] for w in read_windows(suite)) == C.PPL_SCORED_TOKENS


def test_window_rows_keys(tiny_fp32, windows, tmp_path):
    rows, summ = perplexity(tiny_fp32, windows[:2], batch_size=2, expected_n=None)
    assert set(rows[0]) == {"eval_id", "suite", "window_index", "source", "source_row", "token_offset",
                            "nll_sum", "n_scored"}
    assert all(r["eval_id"] == C.EVAL_ID for r in rows) and summ["eval_id"] == C.EVAL_ID
    assert [r["window_index"] for r in rows] == [w["window_index"] for w in windows[:2]]
    p = tmp_path / "ppl.jsonl.gz"
    write_rows(p, rows)
    assert read_jsonl(p) == rows
    with pytest.raises(FileExistsError):
        write_rows(p, rows)


def test_calib_nll(tiny_fp32):
    seqs = read_sequences("c4_development", limit=3)
    rows, summ = calib_nll(tiny_fp32, seqs, batch_size=2)
    assert [r["n_scored"] for r in rows] == [511, 511, 511]
    assert summ["N"] == 3 * 511 and summ["split"] == "c4_development"
    assert summ["mean_nll"] == pytest.approx(sum(r["nll_sum"] for r in rows) / (3 * 511))
    assert [r["source_row"] for r in rows] == [s["source_row"] for s in seqs]
    nll, n = slow_reference(tiny_fp32, {"token_ids": seqs[0]["token_ids"], "score_target_offsets": [1, 512]})
    assert rows[0]["nll_sum"] / 511 == pytest.approx(nll / n, abs=1e-5)
