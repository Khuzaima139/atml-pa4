"""Acceptance rule of the lm_eval comparison."""
from task1 import baseline
from task1.verify_lmeval import ACCEPT_MAX_ABS_DIFF, acceptance, compare


def _row(order, loglik, chars, label=0):
    pred = max(range(4), key=lambda j: loglik[j])
    pred_n = max(range(4), key=lambda j: loglik[j] / chars[j])
    return {"order": order, "loglik": loglik, "cont_chars": chars, "label": label, "pred_acc": pred,
            "pred_acc_norm": pred_n, "correct_acc": pred == label, "correct_acc_norm": pred_n == label}


ROWS = [_row(0, [-10.0, -12.0, -15.0, -11.0], [10, 10, 10, 10]),
        _row(1, [-20.0, -20.004, -25.0, -30.0], [20, 20, 20, 20])]


def test_pass_small_differences_and_explained_disagreement():
    # Example 1 flips (our margin 0.004) under a 0.006 perturbation: explained by the max difference.
    theirs = [-10.001, -12.0, -15.0, -11.0, -20.006, -20.0, -25.0, -30.0]
    run = compare(ROWS, theirs)
    det = acceptance(ROWS, run)
    assert run["argmax_disagreements_acc"] == 1
    assert det["result"] == "PASS" and det["n_unexplained"] == 0
    d = [x for x in det["disagreements"] if x["metric"] == "acc"][0]
    assert d["order"] == 1 and abs(d["margin_ours"] - 0.004) < 1e-9 and d["explained"]
    assert abs(d["margin_lmeval"] - 0.006) < 1e-9


def test_fail_by_difference():
    theirs = [-10.03, -12.0, -15.0, -11.0, -20.0, -20.004, -25.0, -30.0]
    det = acceptance(ROWS, compare(ROWS, theirs))
    assert det["max_abs_diff"] > ACCEPT_MAX_ABS_DIFF
    assert det["result"] == "FAIL" and not det["max_abs_diff_ok"] and det["n_disagreements"] == 0


def test_fail_by_unexplained_disagreement():
    # Our margin 0.015 between candidates 0 and 1; lm_eval flips them with differences of only 0.008.
    rows = [_row(0, [-10.0, -10.015, -15.0, -16.0], [10, 10, 10, 10])]
    theirs = [-10.008, -10.007, -15.0, -16.0]
    det = acceptance(rows, compare(rows, theirs))
    assert det["max_abs_diff_ok"] and abs(det["max_abs_diff"] - 0.008) < 1e-9
    assert det["result"] == "FAIL" and det["n_unexplained"] >= 1
    d = [x for x in det["disagreements"] if x["metric"] == "acc"][0]
    assert not d["explained"] and abs(d["margin_ours"] - 0.015) < 1e-9


def test_acc_norm_margin_uses_per_character_scores():
    # acc agrees; acc_norm flips with a per-character margin of 0.0005, below the max difference.
    rows = [_row(0, [-10.0, -20.01, -35.0, -50.0], [10, 20, 30, 40])]
    theirs = [-10.0, -19.995, -35.0, -50.0]
    det = acceptance(rows, compare(rows, theirs))
    assert [x["metric"] for x in det["disagreements"]] == ["acc_norm"]
    assert abs(det["disagreements"][0]["margin_ours"] - 0.0005) < 1e-12 and det["result"] == "PASS"


def test_baseline_status_and_exit_code():
    assert baseline.run_status("PASS") == "ok" and baseline.exit_code("PASS") == 0
    assert baseline.run_status("FAIL") == "acceptance_failed" and baseline.exit_code("FAIL") == 1
