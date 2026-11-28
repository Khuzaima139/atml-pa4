import json

import pytest
import torch

from common.paths import RESULTS_DIR
from task3 import sampler_check as SC
from task3.lm import MarkovLM
from task3.sampling import make_generator, probs_from_logits
from task3.spec import speculative_generate


def test_toy_target_joint_bounds():
    joint = SC.target_joint()
    assert len(joint) == 27
    assert abs(sum(joint.values()) - 1.0) < 1e-12
    assert min(joint.values()) >= SC.MIN_JOINT
    assert SC.N_SEQUENCES * min(joint.values()) >= SC.MIN_EXPECTED


@pytest.mark.parametrize("gamma", [1, 2, 3, 4])
def test_exact_law_of_positive_residual_equals_target(gamma):
    joint = SC.target_joint()
    law = SC.sampler_joint(gamma, "target_minus_draft")
    assert max(abs(law.get(s, 0.0) - joint[s]) for s in joint) < 1e-12
    bad = SC.sampler_joint(gamma, "target")
    assert abs(sum(bad.values()) - 1.0) < 1e-12
    assert max(abs(bad.get(s, 0.0) - joint[s]) for s in joint) > 1e-3


@pytest.mark.parametrize("gamma", [2, 4])
def test_negative_control_expected_statistic_above_200(gamma):
    joint = SC.target_joint()
    assert SC.expected_statistic(SC.sampler_joint(gamma, "target"), joint, SC.N_SEQUENCES) > 200
    assert SC.expected_statistic(SC.sampler_joint(gamma, "target_minus_draft"), joint, SC.N_SEQUENCES) == \
        pytest.approx(26.0)


def test_markov_lm_rows_and_crop():
    lm = MarkovLM(SC.TARGET["initial"], SC.TARGET["transition"])
    rows = probs_from_logits(lm.forward([lm.start, 0, 2]))
    assert torch.allclose(rows[0], torch.tensor(SC.TARGET["initial"]), atol=1e-6)
    assert torch.allclose(rows[1], torch.tensor(SC.TARGET["transition"][0]), atol=1e-6)
    assert torch.allclose(rows[2], torch.tensor(SC.TARGET["transition"][2]), atol=1e-6)
    lm.crop(1)
    assert lm.length == 1 and lm.tokens == [lm.start]
    with pytest.raises(ValueError):
        lm.crop(5)


def test_toy_round_structure_gamma4_at_length3():
    t = MarkovLM(SC.TARGET["initial"], SC.TARGET["transition"])
    d = MarkovLM(SC.DRAFT["initial"], SC.DRAFT["transition"])
    res = speculative_generate(t, d, [t.start], 4, 3, (), make_generator(1, "cpu"))
    assert res["output_tokens"] == 3
    assert res["rounds"][0]["proposed"] == 3
    for r in res["rounds"]:
        assert r["accepted"] <= r["examined"] <= r["proposed"]


def test_sampler_check_entry_point_small(tmp_path):
    out = tmp_path / "sc.json"
    rc = SC.main(["--out", str(out), "--n", "2000"])
    d = json.loads(out.read_text())
    assert rc == 0 and d["status"] == "ok" and d["all_pass"] is True
    assert len(d["table"]) == 27
    row = d["table"][0]
    for key in ("sequence", "symbols", "target_probability", "expected_count", "observed_correct_gamma2",
                "observed_incorrect_gamma2", "observed_correct_gamma4", "observed_incorrect_gamma4",
                "exact_probability_correct_gamma2", "exact_probability_incorrect_gamma4"):
        assert key in row
    for g in (2, 4):
        for name in ("correct", "incorrect"):
            assert sum(r[f"observed_{name}_gamma{g}"] for r in d["table"]) == 2000
            assert d["statistics"][f"{name}_gamma{g}"]["df"] == 26
    assert set(d["passes"]) == {
        "min_expected_count_ge_10", "min_target_joint_ge_2e-4", "exact_correct_law_equals_target",
        "expected_statistic_incorrect_gamma2_gt_200", "expected_statistic_incorrect_gamma4_gt_200",
        "correct_gamma2_not_rejected", "correct_gamma4_not_rejected",
        "incorrect_gamma2_rejected", "incorrect_gamma4_rejected"}
    assert d["config"]["target"] == SC.TARGET and d["config"]["draft"] == SC.DRAFT
    assert d["config"]["seed_base"] == {"2": SC.SEED_BASE[2], "4": SC.SEED_BASE[4]}
    for key in ("git_commit", "git_dirty", "package_versions"):
        assert key in d["meta"]
    assert d["pooled_round_counts"]["correct_gamma2"]["proposed"] > 0
    with pytest.raises(SystemExit):
        SC.main(["--out", str(out), "--n", "2000"])


def test_sampler_check_refuses_reduced_n_under_results():
    target = RESULTS_DIR / "task3" / "never_written_sampler_check.json"
    with pytest.raises(SystemExit):
        SC.main(["--out", str(target), "--n", "10"])
    assert not target.exists()
