"""Readers for the locked Task 1 files and the HellaSwag prompt format of lm_eval 0.4.13."""
import json

from lm_eval.tasks.hellaswag.utils import preprocess

from common.io import read_jsonl, sha256_gz_content
from common.paths import DATA_REPO_PUBLIC_LOCKED, PUBLIC_ASSETS_LOCK

WINDOW_FILES = {
    "wikitext2_test": "wikitext2_test_windows_locked.jsonl.gz",
    "c4_validation": "c4_validation_windows_locked.jsonl.gz",
}
SEQUENCE_FILES = {
    "c4_calibration": "c4_calibration_locked.jsonl.gz",
    "c4_development": "c4_development_locked.jsonl.gz",
}
HELLASWAG_FILE = "hellaswag_val_locked.jsonl.gz"

WINDOW_KEYS = ("source", "source_row", "token_offset", "token_ids", "score_target_offsets", "scored_tokens")

HELLASWAG_TASK = {"name": "hellaswag", "version": "1.0", "harness": "lm_eval==0.4.13", "num_fewshot": 0,
                  "target_delimiter": " "}
HELLASWAG_PROMPT_FORMAT = (
    "lm_eval 0.4.13 hellaswag (task version 1.0), zero-shot. "
    "context = preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()); "
    "choices = [preprocess(e) for e in endings]; continuation = ' ' + choice; label = int(label); "
    "preprocess = lm_eval.tasks.hellaswag.utils.preprocess. "
    "Encoding (causal encode-pair): trailing whitespace of the context moves to the continuation; "
    "whole = tokenizer.encode(context + continuation); ctx = tokenizer.encode(context); "
    "continuation tokens = whole[len(ctx):]; no BOS (tokenizer default); model input = whole[:-1]. "
    "Score = sum of continuation-token log probabilities (FP32 log-softmax, float64 sum). "
    "acc = argmax score; acc_norm = argmax score / len(choice) in characters, without the leading space."
)


def _lock():
    with open(PUBLIC_ASSETS_LOCK, encoding="utf-8") as fh:
        return json.load(fh)["files"]


def locked_path(filename, verify: bool = True):
    """Path of a locked file; checks the decompressed-content hash against data_lock."""
    path = DATA_REPO_PUBLIC_LOCKED / filename
    if verify:
        want = _lock()[filename]["content_sha256"]
        got = sha256_gz_content(path)
        assert got == want, f"{filename}: content sha256 {got} != locked {want}"
    return path


def file_info(filename) -> dict:
    entry = _lock()[filename]
    return {"file": filename, "content_sha256": entry["content_sha256"], "records": entry["records"]}


def read_windows(suite: str, limit=None) -> list:
    """Perplexity windows in file order; row identity is (source, source_row)."""
    rows = read_jsonl(locked_path(WINDOW_FILES[suite]))
    out = []
    for i, r in enumerate(rows):
        w = {k: r[k] for k in WINDOW_KEYS}
        assert w["source"] == suite, (suite, w["source"])
        a, b = w["score_target_offsets"]
        assert 1 <= a < b <= len(w["token_ids"]) and b - a == w["scored_tokens"], (suite, i)
        w["window_index"] = i
        out.append(w)
    return out[:limit] if limit is not None else out


def read_sequences(split: str, limit=None) -> list:
    """Calibration or development sequences; row identity is (split, source_row)."""
    rows = read_jsonl(locked_path(SEQUENCE_FILES[split]))
    out = [{"split": split, "source_row": r["source_row"], "token_offset": r["token_offset"],
            "token_ids": r["token_ids"], "raw_text_sha256": r["raw_text_sha256"]} for r in rows]
    return out[:limit] if limit is not None else out


def read_hellaswag(limit=None) -> list:
    rows = read_jsonl(locked_path(HELLASWAG_FILE))
    assert [r["order"] for r in rows] == list(range(len(rows)))
    return rows[:limit] if limit is not None else rows


def format_hellaswag(record) -> dict:
    """lm_eval 0.4.13 hellaswag formatting of one locked record."""
    ex = record["example"]
    context = preprocess(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
    choices = [preprocess(e) for e in ex["endings"]]
    assert len(choices) == 4
    return {
        "order": record["order"],
        "source_row_index": record["source_row_index"],
        "label": int(ex["label"]),
        "context": context,
        "choices": choices,
        "continuations": [HELLASWAG_TASK["target_delimiter"] + c for c in choices],
    }


def encode_pair(tokenizer, context: str, continuation: str) -> tuple:
    """Causal encode-pair of lm_eval 0.4.13 (api/model.py _encode_pair) with the tokenizer default."""
    assert context, "empty context"
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer.encode(context + continuation)
    ctx = tokenizer.encode(context)
    return ctx, whole[len(ctx):], whole[:len(ctx)] == ctx
