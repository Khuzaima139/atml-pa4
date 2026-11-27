"""Task 2 notebook 1 helper: inventory and job lists, the job wrapper's concurrency record, queue and wait failure paths
through the wrapper, and an end-to-end CPU run of every stage (prebuild, preflight, smoke, plan, main and timing
queues, idle check, summary, check, zip, final) through the real entry points with tiny limits."""
import json
import os
import signal
import subprocess
import sys
import time
import zipfile

import pytest

from common import kaggle_queue as KQ
from common.paths import REPO_ROOT
from task2 import kaggle_nb1 as N

PY = sys.executable


def _nb(*args, timeout=900):
    return subprocess.run([PY, "-m", "task2.kaggle_nb1", *map(str, args)], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=timeout)


def _queue_cmd(tmp, name, gpu, jobs_file, logs, status):
    return [PY, "-m", "common.kaggle_queue", "--name", name, "--gpu", str(gpu), "--jobs", str(jobs_file),
            "--logs", str(logs), "--status", str(status), "--poll", "0.1"]


# ---------------------------------------------------------------------------------------------------------------
# Inventory and job lists

def test_inventory_and_streams():
    runs = [(m, r, K, s) for m, r, K in N.CONFIGS for s in N.SEEDS]
    assert len(N.CONFIGS) == 13 and len(runs) == 39
    assert sorted(N.CONFIGS) == sorted(set(N.CONFIGS))
    st = N.streams_needed()
    assert len(st) == 18
    assert {(K, r) for K, r, _ in st} == {(4, "mixed"), (8, "mixed"), (16, "mixed"), (4, "near"), (8, "near"),
                                          (16, "near")}


def test_main_jobs_without_plan():
    jobs = N.stage_jobs("main", python="py")
    names = [j["name"] for j in jobs]
    assert len(names) == 42 and len(set(names)) == 42
    ev = [j for j in jobs if j["phase"] == "eval"]
    assert len(ev) == 40 and {j["queue"] for j in ev} == {"gpu0", "gpu1"}
    assert all(j["queue"] == f"gpu{j['gpu']}" for j in ev)
    tm = [j for j in jobs if j["phase"] == "timing"]
    assert [j["name"] for j in tm] == ["part_a_bench", "mqar_sweep"] and {j["gpu"] for j in tm} == {"0"}
    assert [j["queue"] for j in tm] == ["timing_gpu0"] * 2
    for j in jobs:
        assert not any(str(c).startswith("--limit") for c in j["cmd"]), j["name"]
        assert j["expect"] == {}
        assert j["cmd"][:4] == ["py", "-m", "task2.kaggle_nb1", "job"]
        assert j["cmd"][j["cmd"].index("--status") + 1] == "notes/kaggle_status"
    by = {j["name"]: j for j in jobs}
    d = by["train_dense_mixed_K4_s6305"]
    assert d["out"] == "results/task2/mqar/dense_mixed_K4_s6305"
    inner = d["cmd"][d["cmd"].index("--") + 1:]
    assert inner == ["py", "-m", "task2.train_mqar", "--mechanism", "dense", "--K", "4", "--regime", "mixed",
                     "--seed", "6305", "--cache-dir", N.CACHE_DIR, "--out", "results/task2/mqar/dense_mixed_K4_s6305"]
    assert by["part_a_verify"]["out"] == "results/task2/part_a/verify"
    assert by["part_a_bench"]["out"] == "results/task2/part_a/bench"
    assert by["mqar_sweep"]["out"] == "results/task2/mqar_sweep"
    # each queue runs its jobs longest first
    for q in ("gpu0", "gpu1"):
        w = [N.static_seconds(j["name"]) for j in ev if j["queue"] == q]
        assert w == sorted(w, reverse=True)


def test_main_jobs_follow_plan_assignment():
    names = ["part_a_verify"] + [f"train_{N.run_name(m, r, K, s)}" for m, r, K in N.CONFIGS for s in N.SEEDS]
    plan = {"assignment": {n: ("1" if i % 3 == 0 else "0") for i, n in enumerate(names)},
            "rows": [{"name": n, "projected_s": float(i)} for i, n in enumerate(names)]}
    jobs = N.stage_jobs("main", plan=plan, python="py")
    for j in jobs:
        if j["phase"] == "eval":
            assert j["gpu"] == plan["assignment"][j["name"]]
    ev0 = [j["name"] for j in jobs if j["queue"] == "gpu0"]
    assert ev0 == sorted(ev0, key=lambda n: -names.index(n))
    with pytest.raises(SystemExit):
        N.stage_jobs("main", plan={"assignment": {}, "rows": []}, python="py")


def test_preflight_and_smoke_jobs():
    pf = N.stage_jobs("preflight", python="py")
    assert [j["name"] for j in pf if j["queue"] == "gpu0"] == \
        ["pf_part_a_verify"] + [f"pf_train_{m}_gpu0" for m in N.MECHANISMS]
    assert [j["name"] for j in pf if j["queue"] == "gpu1"] == [f"pf_train_{m}_gpu1" for m in N.MECHANISMS] + \
        ["pf_sweep"]
    by = {j["name"]: j for j in pf}
    v = by["pf_part_a_verify"]["cmd"]
    assert v[v.index("--limit-lengths") + 1] == "256,8192"
    s = by["pf_sweep"]["cmd"]
    assert s[s.index("--limit-lengths") + 1] == "8192" and s[s.index("--limit-reps") + 1] == "1,2"
    t = by["pf_train_linear_gpu1"]["cmd"]
    assert t[t.index("--limit-updates") + 1] == "20" and t[t.index("--limit-eval") + 1] == "64"
    assert all(j["out"].startswith("/kaggle/tmp/preflight/") for j in pf)
    sm = N.stage_jobs("smoke", python="py")
    assert [j["name"] for j in sm if j["queue"] == "gpu0"] == ["sm_train_dense", "sm_train_window",
                                                               "sm_part_a_verify"]
    assert [j["name"] for j in sm if j["queue"] == "gpu1"] == ["sm_train_local_dilated", "sm_train_linear"]
    for j in sm:
        if j["kind"] == "train":
            assert j["cmd"][j["cmd"].index("--limit-updates") + 1] == "100" and "--limit-eval" not in j["cmd"]
            assert j["run"] == {"mechanism": j["run"]["mechanism"], "regime": "mixed", "K": 16, "seed": 6304}
    v = {j["name"]: j for j in sm}["sm_part_a_verify"]["cmd"]
    assert v[v.index("--limit-lengths") + 1] == "256,2048"
    assert all(j["out"].startswith("/kaggle/tmp/smoke/") for j in sm)


def test_expected_part_a_counts():
    from task2 import part_a as P
    nv = 6 + 4
    full = N.expected_part_a("verify", P.LENGTHS, 4)
    assert full["rows"] == 6 * (2 + 2 * nv) + 2 * 6 * 4 + 2 * 2 * nv + 2 * nv
    b = N.expected_part_a("bench", P.LENGTHS, 4)
    assert b == {"rows": 6 * 7 * 2 * 2, "backend_table": 6 * 2 * 2}
    # edge cases move to the largest length when 256 is absent
    assert N.expected_part_a("verify", (64,), 0)["rows"] == (2 + 2 * 6) + 2 * 6 * len(P.edge_ts(64)) + 4 * 6 + 2 * 6


def test_balance_lpt():
    a = N.balance({"x": 1000.0, "y": 900.0, "z": 200.0}, ["x", "y", "z"])
    assert a == {"x": "0", "y": "1", "z": "1"}


# ---------------------------------------------------------------------------------------------------------------
# Job wrapper

FAKE = ("import json, os, sys, time; out = sys.argv[1]; os.makedirs(out); "
        "json.dump({{'status': 'ok', 'config': {{}}}}, open(os.path.join(out, 'run.json'), 'w')); "
        "time.sleep({sleep}); print('fake done', flush=True); sys.exit({code})")


def _wrapped(name, status, out, sleep=0.0, code=0):
    return [PY, "-m", "task2.kaggle_nb1", "job", "--name", name, "--status", str(status), "--out", str(out), "--",
            PY, "-c", FAKE.format(sleep=sleep, code=code), str(out)]


def test_wrapper_records_concurrent_jobs(tmp_path):
    st, logs = tmp_path / "status", tmp_path / "logs"
    jobs0 = [{"name": "a", "cmd": _wrapped("a", st, tmp_path / "out_a", sleep=2.0)},
             {"name": "c", "cmd": _wrapped("c", st, tmp_path / "out_c", sleep=0.0)}]
    jobs1 = [{"name": "b", "cmd": _wrapped("b", st, tmp_path / "out_b", sleep=1.0)}]
    f0, f1 = tmp_path / "j0.json", tmp_path / "j1.json"
    f0.write_text(json.dumps(jobs0))
    f1.write_text(json.dumps(jobs1))
    q0 = subprocess.Popen(_queue_cmd(tmp_path, "q0", 0, f0, logs, st), cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    time.sleep(0.5)
    q1 = subprocess.Popen(_queue_cmd(tmp_path, "q1", 1, f1, logs, st), cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    assert q0.wait(120) == 0 and q1.wait(120) == 0
    ra = json.loads((tmp_path / "out_a" / "run.json").read_text())[N.CONCURRENCY_KEY]
    rb = json.loads((tmp_path / "out_b" / "run.json").read_text())[N.CONCURRENCY_KEY]
    rc = json.loads((tmp_path / "out_c" / "run.json").read_text())[N.CONCURRENCY_KEY]
    assert [o["name"] for o in ra["overlapping"]] == ["b"] and ra["gpu"] == "0"
    assert [o["name"] for o in rb["overlapping"]] == ["a"] and rb["alive_at_start"] == ["a"] and rb["gpu"] == "1"
    assert rb["overlapping"][0]["gpu"] == "0" and rb["overlapping_same_gpu"] == []
    assert rc["overlapping"] == [] and rc["alive_at_start"] == []
    assert "[job] a: concurrent jobs ['b']" in (logs / "a.log").read_text()
    iv = json.loads((st / "a.interval.json").read_text())
    assert iv["exit"] == 0 and iv["end_unix"] > iv["start_unix"]
    assert json.loads((tmp_path / "out_a" / "run.json").read_text())["status"] == "ok"


def test_overlapping_rule():
    ivs = [{"name": "x", "start_unix": 0.0, "end_unix": 5.0}, {"name": "y", "start_unix": 5.0, "end_unix": 9.0},
           {"name": "z", "start_unix": 1.0, "end_unix": None, "pid": None}, {"name": "me", "start_unix": 4.0}]
    got = N.overlapping(ivs, "me", 4.0, 6.0)
    assert [o["name"] for o in got] == ["x", "y", "z"]
    assert [o["name"] for o in N.overlapping(ivs, "me", 9.0, 10.0)] == ["z"]
    assert got[2]["ended_without_record"] is True


# ---------------------------------------------------------------------------------------------------------------
# Queue and wait failure paths through the wrapper (success, nonzero exit, killed job)

def _wait(st, logs, names, mode="--wait", out=None, timeout=60):
    cmd = [PY, "-m", "common.kaggle_queue", mode, str(st), "--names", *names, "--logs", str(logs), "--poll", "0.1",
           "--report", "1", "--timeout", str(timeout)]
    if out:
        cmd += ["--out", str(out)]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout + 30)


def test_wait_success_nonzero_and_killed_through_wrapper(tmp_path):
    st, logs = tmp_path / "status", tmp_path / "logs"
    # success
    f = tmp_path / "ok.json"
    f.write_text(json.dumps([{"name": "ok1", "cmd": _wrapped("ok1", st, tmp_path / "o1")}]))
    assert subprocess.run(_queue_cmd(tmp_path, "qa", 0, f, logs, st), cwd=REPO_ROOT, capture_output=True,
                          timeout=120).returncode == 0
    w = _wait(st, logs, ["ok1"])
    assert w.returncode == 0 and "all 1 jobs done" in w.stdout
    # nonzero exit: the wrapper returns the code and prints the run.json status lines; the queue goes on
    f = tmp_path / "bad.json"
    f.write_text(json.dumps([{"name": "bad1", "cmd": _wrapped("bad1", st, tmp_path / "o2", code=3)},
                             {"name": "after", "cmd": _wrapped("after", st, tmp_path / "o3")}]))
    p = subprocess.run(_queue_cmd(tmp_path, "qb", 0, f, logs, st), cwd=REPO_ROOT, capture_output=True, timeout=120)
    assert p.returncode == 1
    assert KQ.read_int(st / "bad1.exit") == 3 and KQ.read_int(st / "after.exit") == 0
    w = _wait(st, logs, ["ok1", "bad1", "after"])
    assert w.returncode == 1 and "job bad1 failed with exit code 3" in w.stdout
    assert "[job status] exit code 3" in w.stdout and "fake done" in w.stdout
    # killed: SIGTERM to the wrapper is forwarded; the wrapper ends by the same signal
    f = tmp_path / "kill.json"
    f.write_text(json.dumps([{"name": "slow", "cmd": _wrapped("slow", st, tmp_path / "o4", sleep=120)},
                             {"name": "next", "cmd": _wrapped("next", st, tmp_path / "o5")}]))
    q = subprocess.Popen(_queue_cmd(tmp_path, "qc", 0, f, logs, st), cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    for _ in range(400):
        if (tmp_path / "o4" / "run.json").exists() and (st / "slow.pid").exists():
            break
        time.sleep(0.05)
    os.kill(KQ.read_int(st / "slow.pid"), signal.SIGTERM)
    assert q.wait(120) == 1
    assert KQ.read_int(st / "slow.exit") == -signal.SIGTERM and KQ.read_int(st / "next.exit") == 0
    rec = json.loads((tmp_path / "o4" / "run.json").read_text())
    assert N.CONCURRENCY_KEY in rec
    w = _wait(st, logs, ["slow", "next"])
    assert w.returncode == 1 and f"exit code {-signal.SIGTERM}" in w.stdout
    wa = _wait(st, logs, ["ok1", "bad1", "after", "slow", "next"], mode="--wait-all", out=tmp_path / "wa.json")
    assert wa.returncode == 1
    states = {r["name"]: (r["state"], r["exit"]) for r in json.loads((tmp_path / "wa.json").read_text())["jobs"]}
    assert states == {"ok1": ("ok", 0), "bad1": ("failed", 3), "after": ("ok", 0), "slow": ("failed", -15),
                      "next": ("ok", 0)}


def test_wrapper_sigkill_of_wrapper_is_failure(tmp_path):
    st, logs = tmp_path / "status", tmp_path / "logs"
    f = tmp_path / "k.json"
    f.write_text(json.dumps([{"name": "victim", "cmd": _wrapped("victim", st, tmp_path / "o", sleep=3)}]))
    q = subprocess.Popen(_queue_cmd(tmp_path, "qk", 1, f, logs, st), cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    for _ in range(400):
        if (st / "victim.pid").exists() and (st / "victim.interval.json").exists():
            break
        time.sleep(0.05)
    os.kill(KQ.read_int(st / "victim.pid"), signal.SIGKILL)
    assert q.wait(120) == 1
    w = _wait(st, logs, ["victim"])
    assert w.returncode == 1 and "exit code -9" in w.stdout


# ---------------------------------------------------------------------------------------------------------------
# End to end on CPU through every stage

TEST_OPTS = ["--device", "cpu", "--limit-updates", "3", "--limit-eval", "2", "--verify-lengths", "32,64",
             "--bench-lengths", "32", "--sweep-lengths", "16", "--reps", "0,1"]
CONFIGS = ["dense_mixed_K4", "linear_mixed_K4"]


def _stage(tmp, stage, jobs_dir, logs, status, extra=()):
    p = _nb("jobs", "--stage", stage, "--jobs-dir", jobs_dir, "--results", tmp / "res" / stage, "--status", status,
            "--cache-dir", tmp / "cache", *TEST_OPTS, *extra)
    assert p.returncode == 0, p.stdout + p.stderr
    meta = N.load_meta(jobs_dir, stage)
    procs = {}
    for q in N.queues(meta):
        if q.startswith("timing"):
            continue
        procs[q] = subprocess.Popen(_queue_cmd(tmp, f"{stage}_{q}", q[-1], jobs_dir / f"{stage}_{q}.json", logs,
                                               status), cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    return meta, procs


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nb1")
    jobs_dir, logs_s, st_s = tmp / "jobs", tmp / "logs_smoke", tmp / "status_smoke"
    logs, st = tmp / "logs", tmp / "status"
    out = {"tmp": tmp, "jobs": jobs_dir, "logs": logs, "status": st}
    pb = _nb("prebuild", "--cache-dir", tmp / "cache", "--workers", "2", "--limit-updates", "3", "--streams",
             "4:mixed:6304", "8:mixed:6305", "16:mixed:6306", "4:near:6304", "8:near:6304", "16:near:6304",
             "--out", tmp / "prebuild.json")
    assert pb.returncode == 0, pb.stdout + pb.stderr
    out["prebuild"] = pb
    for stage in ("preflight", "smoke"):
        meta, procs = _stage(tmp, stage, jobs_dir, logs_s, st_s)
        w = _wait(st_s, logs_s, [j["name"] for j in meta], timeout=900)
        assert w.returncode == 0, w.stdout
        for p in procs.values():
            assert p.wait(60) == 0
        out[stage] = _nb(f"{'preflight-check' if stage == 'preflight' else 'smoke-report'}", "--jobs-dir", jobs_dir)
    out["plan"] = _nb("plan", "--jobs-dir", jobs_dir, "--prebuild", tmp / "prebuild.json", "--elapsed-s", "60",
                      "--out", tmp / "plan.json")
    # the main stage uses the smaller test inventory and a plan written for it
    plan = json.loads((tmp / "plan.json").read_text())
    names = ["part_a_verify"] + [f"train_{c}_s6304" for c in CONFIGS]
    sub = {"assignment": {n: plan["assignment"].get(n, "1") for n in names},
           "rows": [{"name": n, "projected_s": 1.0} for n in names]}
    sub["assignment"]["part_a_verify"] = "0"
    (tmp / "plan_sub.json").write_text(json.dumps(sub))
    meta, procs = _stage(tmp, "main", jobs_dir, logs, st, extra=("--configs", *CONFIGS, "--seeds", "6304",
                                                                 "--plan", tmp / "plan_sub.json"))
    out["meta"] = meta
    ev = [j["name"] for j in meta if j["phase"] == "eval"]
    out["wait"] = _wait(st, logs, ev, mode="--wait-all", out=tmp / "wait.json", timeout=900)
    for p in procs.values():
        p.wait(60)
    out["idle"] = _nb("idle-check", "--jobs-dir", jobs_dir, "--status", st)
    tq = subprocess.run(_queue_cmd(tmp, "timing_gpu0", 0, jobs_dir / "main_timing_gpu0.json", logs, st),
                        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    out["timing"] = tq
    out["summary"] = _nb("summary", "--jobs-dir", jobs_dir, "--status", st)
    out["check"] = _nb("check", "--jobs-dir", jobs_dir, "--status", st, "--logs", logs, "--out", tmp / "check.json")
    out["zip"] = _nb("zip", "--jobs-dir", jobs_dir, "--out", tmp / "nb1.zip", "--include", tmp / "res" / "main",
                     logs, st, jobs_dir)
    out["final"] = _nb("final", "--jobs-dir", jobs_dir, "--status", st, "--check-report", tmp / "check.json",
                       "--zip", tmp / "nb1.zip")
    return out


def test_e2e_prebuild(e2e):
    rec = json.loads((e2e["tmp"] / "prebuild.json").read_text())
    assert rec["n_failed"] == 0 and len(rec["streams"]) == 6
    for r in rec["streams"]:
        assert r["built"] is True and r["stop"] - r["start"] == 96 and r["validation"]["sequences_checked"] == 96
        assert r["problems"] == [] and len(r["sha256"]) == 64
    assert "stream K4  mixed s6304 [0, 96): built True" in e2e["prebuild"].stdout
    assert "stream K8  mixed s6305 [144000, 144096)" in e2e["prebuild"].stdout


def test_e2e_preflight_and_smoke(e2e):
    assert e2e["preflight"].returncode == 0, e2e["preflight"].stdout + e2e["preflight"].stderr
    assert "preflight-check: 10 jobs, 0 problems" in e2e["preflight"].stdout
    assert "verify L 64 fp32 dense" in e2e["preflight"].stdout
    assert e2e["smoke"].returncode == 0, e2e["smoke"].stdout + e2e["smoke"].stderr
    assert "smoke-report: 5 jobs, 0 problems" in e2e["smoke"].stdout
    assert "sm_train_dense: 3 updates" in e2e["smoke"].stdout


def test_e2e_plan(e2e):
    p = e2e["plan"]
    assert p.returncode == 0, p.stdout + p.stderr
    plan = json.loads((e2e["tmp"] / "plan.json").read_text())
    assert len(plan["rows"]) == 40 and set(plan["assignment"].values()) == {"0", "1"}
    assert plan["over"] is False and plan["reasons"] == [] and plan["elapsed_s"] == 60
    r = {x["name"]: x for x in plan["rows"]}["train_dense_near_K16_s6305"]
    assert set(r["parts"]) == {"train", "dev_eval", "test_eval", "fixed", "stream_check_and_validation"}
    assert r["projected_s"] == pytest.approx(N.SAFETY * sum(r["parts"].values()))
    assert set(plan["timing_s"]) == {"part_a_bench", "mqar_sweep"}
    assert "plan: within limits" in p.stdout
    # a tiny budget stops the plan
    q = _nb("plan", "--jobs-dir", e2e["jobs"], "--prebuild", e2e["tmp"] / "prebuild.json", "--budget-h", "0.00001")
    assert q.returncode == 1 and "OVER:" in q.stdout


def test_e2e_main_timing_and_outputs(e2e):
    assert e2e["wait"].returncode == 0, e2e["wait"].stdout
    assert e2e["idle"].returncode == 0, e2e["idle"].stdout + e2e["idle"].stderr
    assert e2e["timing"].returncode == 0, e2e["timing"].stdout[-3000:]
    res = e2e["tmp"] / "res" / "main"
    for c in CONFIGS:
        d = res / "mqar" / f"{c}_s6304"
        assert sorted(os.listdir(d)) == sorted(["run.json", "train_log.jsonl", "dev_curve.jsonl",
                                                "dev_queries_final.npz", "test_queries.npz",
                                                "model_final.safetensors"])
        run = json.loads((d / "run.json").read_text())
        conc = run[N.CONCURRENCY_KEY]
        assert set(conc) == {"method", "job", "gpu", "start", "end", "alive_at_start", "overlapping",
                             "overlapping_same_gpu", "status_dir"}
    for sub, files in (("part_a/verify", {"run.json", "verify_rows.jsonl", "score_bytes.json"}),
                       ("part_a/bench", {"run.json", "bench_rows.jsonl", "backend_table.jsonl", "score_bytes.json"}),
                       ("mqar_sweep", {"run.json", "sweep_rows.jsonl"})):
        assert set(os.listdir(res / sub)) == files
    for sub in ("part_a/bench", "mqar_sweep"):
        assert json.loads((res / sub / "run.json").read_text())[N.CONCURRENCY_KEY]["overlapping"] == []


def test_e2e_summary_check_zip_final(e2e):
    s = e2e["summary"]
    assert s.returncode == 0, s.stderr
    lines = s.stdout.rstrip("\n").split("\n")
    assert len(lines) <= 60
    assert lines[0].startswith("run ")
    assert any(x.startswith("dense_mixed_K4_s6304 ") and " | " in x for x in lines)
    assert any(x.startswith("linear_mixed_K4_s6304 ") for x in lines)
    assert any(x.startswith("Part A default SDPA fp16") for x in lines)
    assert any(x.startswith("sweep T 16:") for x in lines)
    c = e2e["check"]
    assert c.returncode == 0, c.stdout + c.stderr
    assert json.loads((e2e["tmp"] / "check.json").read_text()) == {"n_jobs": 5, "problems": [], "failed_jobs": []}
    assert "==== CHECK: PASS: 5 jobs, 0 problems ====" in c.stdout
    z = e2e["zip"]
    assert z.returncode == 0, z.stdout + z.stderr
    with zipfile.ZipFile(e2e["tmp"] / "nb1.zip") as zf:
        names = zf.namelist()
    assert any(n.endswith("dense_mixed_K4_s6304/test_queries.npz") for n in names)
    assert any(n.endswith("logs/train_dense_mixed_K4_s6304.log") for n in names)
    assert not any(w in n.split("/nb1")[-1] for n in names for w in ("smoke", "preflight", "cache"))
    f = e2e["final"]
    assert f.returncode == 0 and "notebook complete" in f.stdout, f.stdout


def test_e2e_check_finds_problems(e2e, tmp_path):
    """A copied run directory with a tampered record and a missing npz is reported."""
    import shutil
    meta = [j for j in e2e["meta"] if j["name"] == "train_linear_mixed_K4_s6304"]
    src = meta[0]["out"]
    dst = tmp_path / "run"
    shutil.copytree(src, dst)
    run = json.loads((dst / "run.json").read_text())
    run["parameter_count"] = 1
    run["test_file_opens"].append({"after_update": 0})
    (dst / "run.json").write_text(json.dumps(run))
    (dst / "dev_queries_final.npz").unlink()
    j = dict(meta[0], out=str(dst))
    probs = N.check([j], e2e["status"], e2e["logs"])
    text = "\n".join(probs)
    assert "parameter count 1" in text and "test_file_opens" in text and "dev_queries_final.npz unreadable" in text


def test_evaluation_failures_do_not_stop_later_phases(tmp_path):
    """The notebook path after the queues start: wait-all over both GPU queues waits for every job despite a nonzero
    exit and a killed job, prints their exit codes and log tails; the idle check then passes, the timing queue runs
    with no overlapping job, check prints a FAIL list and final reports the run as not complete (nonzero exit codes,
    no exception)."""
    st, logs, jd = tmp_path / "status", tmp_path / "logs", tmp_path / "jobs"
    jd.mkdir()

    def spec(name, phase, kind, queue, sleep=0.0, code=0):
        out = tmp_path / "res" / name
        return {"name": name, "phase": phase, "kind": kind, "queue": queue, "gpu": queue[-1], "out": str(out),
                "run": {"mechanism": "dense", "regime": "mixed", "K": 4, "seed": 6304}, "expect": {},
                "cmd": _wrapped(name, st, out, sleep=sleep, code=code), "wait_for": []}

    meta = [spec("t_bad", "eval", "train", "gpu0", code=3), spec("t_late", "eval", "train", "gpu0", sleep=2.0),
            spec("t_slow", "eval", "train", "gpu1", sleep=120), spec("t_ok", "eval", "train", "gpu1"),
            spec("part_a_bench", "timing", "bench", "timing_gpu0"), spec("mqar_sweep", "timing", "sweep",
                                                                         "timing_gpu0")]
    N.write_jobs("main", jd, meta)
    q0 = subprocess.Popen(_queue_cmd(tmp_path, "gpu0", 0, jd / "main_gpu0.json", logs, st), cwd=REPO_ROOT,
                          stdout=subprocess.DEVNULL, start_new_session=True)
    q1 = subprocess.Popen(_queue_cmd(tmp_path, "gpu1", 1, jd / "main_gpu1.json", logs, st), cwd=REPO_ROOT,
                          stdout=subprocess.DEVNULL, start_new_session=True)
    for _ in range(400):
        if (st / "t_slow.pid").exists() and (tmp_path / "res" / "t_slow" / "run.json").exists():
            break
        time.sleep(0.05)
    os.kill(KQ.read_int(st / "t_slow.pid"), signal.SIGTERM)
    ev = [j["name"] for j in meta if j["phase"] == "eval"]
    w = _wait(st, logs, ev, mode="--wait-all", out=tmp_path / "wait.json", timeout=120)
    assert w.returncode == 1
    states = {r["name"]: (r["state"], r["exit"]) for r in json.loads((tmp_path / "wait.json").read_text())["jobs"]}
    assert states == {"t_bad": ("failed", 3), "t_late": ("ok", 0), "t_slow": ("failed", -15), "t_ok": ("ok", 0)}
    assert "--- t_bad: failed, exit 3" in w.stdout and "[job status] exit code 3" in w.stdout
    assert "--- t_slow: failed, exit -15" in w.stdout and "2 of 4 jobs ok" in w.stdout
    assert q0.wait(60) == 1 and q1.wait(60) == 1
    idle = _nb("idle-check", "--jobs-dir", jd, "--status", st)
    assert idle.returncode == 0, idle.stdout + idle.stderr
    tq = subprocess.run(_queue_cmd(tmp_path, "timing_gpu0", 0, jd / "main_timing_gpu0.json", logs, st),
                        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert tq.returncode == 0
    for n in ("part_a_bench", "mqar_sweep"):
        assert json.loads((tmp_path / "res" / n / "run.json").read_text())[N.CONCURRENCY_KEY]["overlapping"] == []
    c = _nb("check", "--jobs-dir", jd, "--status", st, "--logs", logs, "--out", tmp_path / "check.json")
    assert c.returncode == 1 and c.stderr == ""
    assert "FAIL: t_bad: exit 3, done False" in c.stdout and "==== CHECK: FAIL:" in c.stdout
    assert "FAILED JOBS:" in c.stdout and "t_slow" in c.stdout.split("FAILED JOBS:")[1]
    rep = json.loads((tmp_path / "check.json").read_text())
    assert {"t_bad", "t_slow"} <= set(rep["failed_jobs"])
    z = _nb("zip", "--jobs-dir", jd, "--out", tmp_path / "z.zip", "--include", tmp_path / "res", logs, st)
    assert z.returncode == 0 and (tmp_path / "z.zip").is_file()
    f = _nb("final", "--jobs-dir", jd, "--status", st, "--check-report", tmp_path / "check.json",
            "--zip", tmp_path / "z.zip")
    assert f.returncode == 1 and f.stderr == "" and "notebook not complete" in f.stdout
    assert "NOT OK: job t_bad: exit 3" in f.stdout


def test_idle_check_reports_live_job(tmp_path):
    st, logs, jd = tmp_path / "status", tmp_path / "logs", tmp_path / "jobs"
    jd.mkdir()
    out = tmp_path / "res" / "t_run"
    meta = [{"name": "t_run", "phase": "eval", "kind": "train", "queue": "gpu0", "gpu": "0", "out": str(out),
             "run": None, "expect": {}, "cmd": _wrapped("t_run", st, out, sleep=120), "wait_for": []}]
    N.write_jobs("main", jd, meta)
    q = subprocess.Popen(_queue_cmd(tmp_path, "gpu0", 0, jd / "main_gpu0.json", logs, st), cwd=REPO_ROOT,
                         stdout=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(400):
            if (st / "t_run.interval.json").exists():
                break
            time.sleep(0.05)
        idle = _nb("idle-check", "--jobs-dir", jd, "--status", st)
        assert idle.returncode == 1 and "NOT IDLE: t_run: state running" in idle.stdout
        assert "a queue process is alive" in idle.stdout
    finally:
        os.killpg(q.pid, signal.SIGTERM)
        q.wait(60)
