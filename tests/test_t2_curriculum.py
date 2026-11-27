"""Protocol AM3 curriculum of python -m task2.train_mqar: schedule values, batch composition in index order,
argument refusals and an end-to-end CPU run with --limit-updates (scaled schedule)."""
import json
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch

from common.io import read_jsonl, sha256_file
from common.paths import REPO_ROOT
from task2 import mqar_data as D
from task2 import train_mqar as TR

UPDATES, EVAL, SEED = 9, 2, 6305


@pytest.mark.parametrize("u,f,n", [(1, 1, 32), (1000, 1, 32), (1001, Fraction(999, 1000), 32), (1500, Fraction(1, 2), 16),
                                   (1999, Fraction(1, 1000), 0), (2000, 0, 0), (4500, 0, 0)])
def test_schedule_values(u, f, n):
    assert TR.curriculum_f(u) == f
    assert TR.n_near_at(u) == n


def test_schedule_totals_and_no_ties():
    plan = TR.curriculum_plan()
    assert len(plan) == 4500 and TR.curriculum_breaks() == (1000, 2000)
    assert plan[-1]["near"][1] == 47_984 and plan[-1]["mixed"][1] == 96_016
    assert all((32 * r["f"]).denominator != 2 for r in plan)
    ns = [r["n_near"] for r in plan]
    assert all(a >= b for a, b in zip(ns, ns[1:]))
    with pytest.raises(ValueError):
        TR.curriculum_f(0)
    with pytest.raises(ValueError):
        TR.curriculum_f(4501)


def test_scaled_schedule():
    assert TR.curriculum_breaks(9) == (2, 4)
    assert [TR.n_near_at(u, 9) for u in range(1, 10)] == [32, 32, 16, 0, 0, 0, 0, 0, 0]
    assert TR.curriculum_breaks(100) == (22, 44)
    assert TR.curriculum_breaks(4) == (1, 2) and [TR.n_near_at(u, 4) for u in range(1, 5)] == [32, 0, 0, 0]


def test_batches_drawn_from_streams_in_index_order():
    n_upd = 50
    near = np.arange(n_upd * 32)[:, None] + np.zeros((1, 3), dtype=np.int64)
    mixed = -1 - near
    plan = TR.curriculum_plan(n_upd)
    seen_near, seen_mixed = [], []
    for r in plan:
        b = TR.curriculum_batch(near, mixed, r)
        assert b.shape == (32, 3)
        nn = r["n_near"]
        assert (b[:nn, 0] >= 0).all() and (b[nn:, 0] < 0).all()
        seen_near += b[:nn, 0].tolist()
        seen_mixed += (-1 - b[nn:, 0]).tolist()
        tb = TR.curriculum_batch(torch.from_numpy(near), torch.from_numpy(mixed), r)
        assert torch.equal(tb, torch.from_numpy(b))
    assert seen_near == list(range(len(seen_near))) and seen_mixed == list(range(len(seen_mixed)))
    assert len(seen_near) + len(seen_mixed) == n_upd * 32
    a, bb = TR.curriculum_breaks(n_upd)
    assert len(seen_near) == sum(TR.n_near_at(u, n_upd) for u in range(1, n_upd + 1)) and a == 11 and bb == 22


def test_run_names():
    assert TR.run_name("local_dilated", "mixed", 16, 6304, "near_to_mixed") == "local_dilated_curriculum_K16_s6304"
    assert TR.run_name("local_dilated", "mixed", 16, 6304) == "local_dilated_mixed_K16_s6304"


@pytest.mark.parametrize("mech,K,regime", [("dense", "16", "mixed"), ("window", "16", "mixed"),
                                           ("linear", "16", "mixed"), ("local_dilated", "4", "mixed"),
                                           ("local_dilated", "8", "mixed"), ("local_dilated", "16", "near")])
def test_curriculum_refuses_other_combinations(tmp_path, mech, K, regime):
    out = tmp_path / "o"
    with pytest.raises(SystemExit, match="--curriculum near_to_mixed needs"):
        TR.main(["--mechanism", mech, "--K", K, "--regime", regime, "--seed", "6304", "--out", str(out),
                 "--device", "cpu", "--limit-updates", "2", "--curriculum", "near_to_mixed"])
    assert not out.exists()


def test_unknown_curriculum_refused(tmp_path):
    with pytest.raises(SystemExit):
        TR.main(["--mechanism", "local_dilated", "--K", "16", "--regime", "mixed", "--seed", "6304", "--out",
                 str(tmp_path / "o"), "--device", "cpu", "--curriculum", "mixed_to_near"])


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("cur")
    out = base / "local_dilated_curriculum_K16_s6305"
    cmd = [sys.executable, "-m", "task2.train_mqar", "--mechanism", "local_dilated", "--K", "16", "--regime", "mixed",
           "--seed", str(SEED), "--curriculum", "near_to_mixed", "--out", str(out), "--cache-dir", str(base / "c"),
           "--device", "cpu", "--limit-updates", str(UPDATES), "--limit-eval", str(EVAL)]
    p = subprocess.run(cmd, cwd=REPO_ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), capture_output=True,
                       text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-4000:]
    assert f"train_mqar done: {out}" in p.stdout
    return out


def test_e2e_curriculum_run(run_dir):
    from tests.test_t2_mqar_e2e import CONFIG_KEYS, CURVE_KEYS, LOG_KEYS, STREAM_KEYS, TOP_KEYS
    out = run_dir
    assert set(os.listdir(out)) == set(TR.OUTPUT_FILES.values())
    run = json.loads((out / "run.json").read_text())
    assert set(run) == TOP_KEYS | {"stream_near", "curriculum_consumed"}
    assert run["status"] == "ok" and run["step"] == "done"
    cfg = run["config"]
    assert set(cfg) == CONFIG_KEYS | {"run_name", "curriculum"}
    assert cfg["run_name"] == f"local_dilated_curriculum_K16_s{SEED}"
    assert (cfg["mechanism"], cfg["K"], cfg["regime"], cfg["limited"]) == ("local_dilated", 16, "mixed", True)
    cur = cfg["curriculum"]
    assert cur["name"] == "near_to_mixed" and cur["breaks"] == [2, 4] and cur["breaks_full"] == [1000, 2000]
    assert cur["scaled"] is True and cur["streams"] == {"near": "near_K16", "mixed": "mixed_K16"}
    start = (SEED - 6304) * 144_000
    stop = start + UPDATES * 32
    for key, regime in (("stream", "mixed"), ("stream_near", "near")):
        st = run[key]
        assert set(st) == STREAM_KEYS
        assert (st["K"], st["regime"], st["start"], st["stop"]) == (16, regime, start, stop)
        assert st["sha256"] == sha256_file(st["path"])
        assert st["validation"]["sequences_checked"] == UPDATES * 32
        assert st["disjointness"]["collisions"] == 0 and st["disjointness"]["n_train_hashed"] == UPDATES * 32
        assert st["disjointness"]["heldout_list"]["sha256"] == sha256_file(D.HELDOUT_LIST)
    assert run["stream"]["sha256"] != run["stream_near"]["sha256"]
    n_near = [32, 32, 16, 0, 0, 0, 0, 0, 0]
    assert run["curriculum_consumed"] == {"near_index_range": [start, start + 80],
                                          "mixed_index_range": [start, start + 208], "n_near": 80, "n_mixed": 208,
                                          "n_near_by_update_sum": 80}
    log = read_jsonl(out / "train_log.jsonl")
    want_keys = (LOG_KEYS - {"index_range"}) | {"curriculum_f", "n_near", "near_index_range", "mixed_index_range"}
    assert all(set(r) == want_keys for r in log)
    assert [r["n_near"] for r in log] == n_near
    assert [r["curriculum_f"] for r in log] == [1.0, 1.0, 0.5, 0, 0, 0, 0, 0, 0]
    pn = pm = start
    for r in log:
        assert r["near_index_range"] == [pn, pn + r["n_near"]]
        assert r["mixed_index_range"] == [pm, pm + 32 - r["n_near"]]
        pn, pm = r["near_index_range"][1], r["mixed_index_range"][1]
    curve = read_jsonl(out / "dev_curve.jsonl")
    assert [r["update"] for r in curve[::9]] == [1, 2, 3, 4, 6, 9]
    extra = {"curriculum_f", "n_near", "near_consumed", "mixed_consumed"}
    assert all(set(r) == CURVE_KEYS | extra for r in curve)
    by_u = {r["update"]: r for r in curve}
    assert (by_u[3]["n_near"], by_u[3]["near_consumed"], by_u[3]["mixed_consumed"]) == (16, 80, 16)
    assert (by_u[9]["near_consumed"], by_u[9]["mixed_consumed"]) == (80, 208)
    assert all(extra <= set(p) for p in run["dev_curve"])
    assert run["learned"]["cell"] == "mixed_K16"
    assert run["test_file_opens"][0]["after_update"] == UPDATES and run["updates_done"] == UPDATES


def test_e2e_batches_equal_generator_rows(run_dir):
    """The first update's near rows and the fourth update's mixed rows are the generator's examples at the
    recorded indices."""
    gen, _ = D.load_generator()
    run = json.loads((run_dir / "run.json").read_text())
    log = read_jsonl(run_dir / "train_log.jsonl")
    near = np.load(run["stream_near"]["path"])
    mixed = np.load(run["stream"]["path"])
    start = run["stream"]["start"]
    lo, hi = log[0]["near_index_range"]
    for i in (lo, hi - 1):
        assert np.array_equal(near[i - start], D.encode_example(gen.make_example("train", 16, "near", i)))
    lo, hi = log[3]["mixed_index_range"]
    for i in (lo, hi - 1):
        assert np.array_equal(mixed[i - start], D.encode_example(gen.make_example("train", 16, "mixed", i)))
