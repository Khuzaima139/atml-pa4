"""Standalone-accuracy scorer: every gold answer passes, realistic malformed responses fail; held-out hash list."""
import pytest

from common.freeze import require_committed
from common.io import read_jsonl
from task3 import data as D
from task3 import heldout as H
from task3 import scorer as S


def _gsm8k_test():
    return sorted(read_jsonl(D.GSM8K_TEST_LOCKED), key=lambda r: r["order"])


def _short_all():
    return read_jsonl(D.SHORT_TRAIN) + read_jsonl(D.SHORT_TEST)


def test_gsm8k_golds_extract():
    rows = _gsm8k_test()
    assert len(rows) == 60
    for r in rows:
        sol = r["example"]["answer"]
        g = S.gold_math(sol)
        assert S.score_math(sol, g)["correct"], r["order"]
        assert S.score_math(g, g)["correct"]
        assert S.score_math(f"The answer is {g}.", g)["correct"]
        body = sol.rsplit("####", 1)[0]
        assert S.score_math(body + f"So the total is {g}.", g)["correct"]


def test_short_golds_exact_and_format():
    rows = _short_all()
    assert len(rows) == 360
    seen = set()
    for r in rows:
        res = S.score_short(r["answer"], r["answer"], r["family"], r["split"])
        assert res["correct"] and res["format_ok"], r["id"]
        assert S.score_short(f"  {r['answer']}\n", r["answer"], r["family"], r["split"])["correct"]
        seen.add(S.template_key(r["family"], r["split"]))
    assert seen == set(S.TEMPLATES)


def test_template_instructions_match_prompts():
    for r in _short_all():
        key = S.template_key(r["family"], r["split"])
        assert S.TEMPLATES[key]["instruction"] in r["input"], r["id"]


@pytest.mark.parametrize("resp,gold,correct,extractable", [
    ("I am not sure how to solve this.", "18", False, False),
    ("She makes 9 * 2 = 18 dollars, plus 4 tips.", "18", False, True),
    ("#### 1,200", "1200", True, True),
    ("Total 18.00 dollars", "18", True, True),
    ("The answer is 18\n#### 17", "18", False, True),
    ("#### $18", "18", True, True),
    ("Answer: -3", "3", False, True),
])
def test_math_malformed(resp, gold, correct, extractable):
    r = S.score_math(resp, gold)
    assert r["correct"] is correct and r["extractable"] is extractable and r["scorer_id"] == S.SCORER_ID


@pytest.mark.parametrize("resp,gold,family,split,correct,fmt", [
    ("The month is May.", "May", 0, "familiar", False, False),
    ("may", "May", 0, "familiar", False, False),
    ("Mar", "March", 0, "shifted", False, False),
    ("05", "May", 0, "shifted", False, False),
    ("8,441", "8441", 1, "familiar", False, False),
    ("8441 crates", "8441", 1, "familiar", False, False),
    ('{"count": 37}', "37", 1, "shifted", False, False),
    ("-5", "5", 1, "shifted", False, False),
    ("PK.", "PK", 2, "familiar", False, False),
    ("pk", "PK", 2, "shifted", False, False),
    ("PAK", "PK", 2, "shifted", False, False),
    ("JP", "PK", 2, "familiar", False, True),
    ("June", "May", 0, "train", False, True),
])
def test_short_malformed(resp, gold, family, split, correct, fmt):
    r = S.score_short(resp, gold, family, split)
    assert r["correct"] is correct and r["format_ok"] is fmt


def test_template_key_rejects_unknown():
    with pytest.raises(ValueError):
        S.template_key(0, "test")
    with pytest.raises(ValueError):
        S.template_key(3, "train")


def test_heldout_hash_list_matches_release():
    rec = require_committed(H.HELDOUT_LIST, "held-out hash list")
    assert rec["sha256"]
    obj = H.read_heldout_list()
    built = H.build_heldout_list()
    assert obj == built
    assert obj["n"] == 120 and len(obj["prompts"]) == 120 and obj["heldout_id"] == H.HELDOUT_ID
    prompts, _ = D.heldout_prompts()
    assert [p["prompt_id"] for p in obj["prompts"]] == [p["prompt_id"] for p in prompts]
    listed = {h for p in obj["prompts"] for h in (p["sha256_exact"], p["sha256_norm"])}
    train, _ = D.training_items()
    assert not any(H.text_hashes(t["prompt"])["sha256_exact"] in listed for t in train)
    assert not any(H.text_hashes(t["prompt"])["sha256_norm"] in listed for t in train)


def test_heldout_guard_refuses_and_records(tmp_path):
    g = H.HeldoutGuard()
    prompts, _ = D.heldout_prompts()
    with pytest.raises(SystemExit):
        g.check([{"prompt_id": "x", "prompt": prompts[70]["prompt"].upper() + "  "}])
    train, _ = D.training_items()
    g.check(train[:5])
    rep = g.report()
    assert rep["items_checked"] == 6 and rep["n_listed_prompts"] == 120
    assert any(o.endswith("short_test.jsonl") for o in rep["heldout_file_opens"])
