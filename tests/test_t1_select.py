"""Selection criterion, candidate grid and the held-out file guard."""
import pytest

from task1 import config as C
from task1 import data
from task1.select import candidate_id, choose, p3_candidates, q3_candidates
from tests.test_t1_allocate import SMALL4


def _row(cid, nll, measure, lam, ref=False):
    return {"candidate_id": cid, "development": {"mean_nll": nll}, "measure": measure, "lambda": lam, "reference": ref}


def test_choose_minimum_and_tie_rule():
    rows = [_row("ref", 1.0, None, 0.0, ref=True), _row("a", 3.0, "S1", 0.04), _row("b", 2.5, "S2", 0.20),
            _row("c", 2.6, "S1", 0.08)]
    assert choose(rows, "p3")["candidate_id"] == "b"  # reference is lower but never selected
    rows = [_row("s2_small", 2.0, "S2", 0.04), _row("s1_big", 2.0, "S1", 0.12), _row("s1_small", 2.0, "S1", 0.04)]
    assert choose(rows, "p3")["candidate_id"] == "s1_small"
    rows = [_row("s2", 2.0, "S2", 0.04), _row("s1", 2.0, "S1", 0.08)]
    assert choose(rows, "p3")["candidate_id"] == "s2"  # smaller lambda first, then measure order
    rows = [_row("u", 0.5, None, None, ref=True), _row("qs2", 2.0, "QS2", None), _row("qs1", 2.0, "QS1", None)]
    assert choose(rows, "q3")["candidate_id"] == "qs1"
    with pytest.raises(AssertionError):
        choose([_row("u", 0.5, None, None, ref=True)], "q3")


def test_candidate_grid():
    sens = {"values": {"S1": [0.1, 0.4, 0.2, 0.3], "S2": {"0.50": [1.0, 2.0, 3.0, 4.0], "0.75": [4, 3, 2, 1]}}}
    c = p3_candidates(sens, 0.5, SMALL4)
    assert len(c) == 1 + len(C.P3_MEASURES) * len(C.P3_LAMBDAS) == 11
    assert [x["reference"] for x in c] == [True] + [False] * 10
    assert candidate_id("p3_s50", c[0]) == "p3_s50.reference"
    assert candidate_id("p3_s50", c[1]) == "p3_s50.S1.lambda0.04"
    assert candidate_id("p3_s50", c[-1]) == "p3_s50.S2.lambda0.20"
    assert c[6]["alloc"]["sensitivity"] == [1.0, 2.0, 3.0, 4.0]
    qs = {"values": {m: {str(b): [float(9 - b)] * 4 for b in C.Q3_BITS} for m in C.Q3_MEASURES}}
    q = q3_candidates(qs, 3, SMALL4)
    assert len(q) == 3 and q[0]["reference"] and q[0]["alloc"]["bits"] == [3] * 4
    assert candidate_id("q3_int3", q[1]) == "q3_int3.QS1"


def test_held_out_guard_refuses_reads(tmp_path):
    assert set(data.HELD_OUT_FILES) == {"wikitext2_test_windows_locked.jsonl.gz",
                                        "c4_validation_windows_locked.jsonl.gz", "hellaswag_val_locked.jsonl.gz"}
    with data.forbid_held_out() as g:
        assert data.held_out_guard_active()
        assert len(data.read_sequences("c4_development", limit=1)) == 1
        with pytest.raises(data.HeldOutFileAccess):
            data.read_windows("wikitext2_test")
        with pytest.raises(data.HeldOutFileAccess):
            data.read_hellaswag(limit=1)
        with pytest.raises(data.HeldOutFileAccess):  # any open() of the name, even outside the data directory
            open(tmp_path / "c4_validation_windows_locked.jsonl.gz", "wb")
    assert not data.held_out_guard_active()
    assert len(g["attempts"]) == 3
    with data.forbid_held_out() as g2:
        pass
    assert g2["attempts"] == []
    (tmp_path / "hellaswag_val_locked.jsonl.gz").write_bytes(b"")  # outside the guard: allowed
