"""Run-1 job lists, and an end-to-end CPU run of both run-1 queues through common.kaggle_queue with a tiny model.

The e2e run uses the real job lists of task1.kaggle_run1 with outputs redirected to a temporary directory and the
test options (--model-path, --device cpu, --limit-calib, --limit-dev) appended; then summary, check and zip.
"""
import json
import os
import subprocess
import sys
import zipfile

import pytest

from common import kaggle_queue as KQ
from common.paths import REPO_ROOT
from task1 import config as C
from task1 import data as D
from task1 import kaggle_run1 as K

LIM_C, LIM_D = 2, 2
HELD_OUT = set(D.HELD_OUT_FILES)


def _deps_ok(jobs):
    order = {j["name"]: i for i, j in enumerate(jobs)}
    names = set(order)
    for j in jobs:
        for w in j["wait_for"]:
            dep = os.path.basename(w)[:-len(".done")]
            assert dep in names, (j["name"], dep)
            same = [x for x in jobs if x["name"] == dep][0]["gpu"] == j["gpu"]
            if same:
                assert order[dep] < order[j["name"]], (j["name"], dep)


def test_run1_jobs_structure():
    jobs = K.stage_jobs("run1")
    g0 = [j["name"] for j in jobs if j["gpu"] == "0"]
    g1 = [j["name"] for j in jobs if j["gpu"] == "1"]
    assert g0 == ["calib_stats", "probe_quant", "select_q3_int4", "select_q3_int3", "compress_q1_b8", "compress_q1_b4",
                  "compress_q1_b3", "compress_q2_b8", "compress_q2_b4", "compress_q2_b3", "compress_q3_int4",
                  "compress_q3_int3"]
    assert g1 == ["compress_fp16", "compress_p1_s25", "compress_p1_s50", "compress_p1_s75", "compress_p2_s25",
                  "compress_p2_s50", "compress_p2_s75", "compress_p2_24_s50", "probe_prune", "select_p3_s50",
                  "select_p3_s75", "compress_p3_s50", "compress_p3_s75"]
    comp = [j for j in jobs if j["kind"] == "compress"]
    assert sorted(j["condition"] for j in comp) == sorted(C.CONDITIONS)
    _deps_ok(jobs)
    pp = [j for j in jobs if j["name"] == "probe_prune"][0]
    assert pp["wait_for"] == ["notes/kaggle_status/calib_stats.done"]
    for j in jobs:
        flat = " ".join(map(str, j["cmd"]))
        assert not any(h in flat for h in HELD_OUT) and "--limit" not in flat and "--model-path" not in flat
        assert j["cmd"][:2] == [sys.executable, "-m"] and j["cmd"][2] == j["entry"]
        if j["kind"] == "compress":
            c = j["condition"]
            assert j["out"] == f"results/task1/run1/{c}" and j["artifacts"] == f"/kaggle/working/artifacts/{c}"
            fam = C.CONDITIONS[c]["family"]
            assert ("--calib-stats" in j["cmd"]) == (fam in ("Q2", "Q3"))
            assert ("--allocation" in j["cmd"]) == (fam in ("P3", "Q3"))
    sel = {j["name"]: j for j in jobs if j["kind"] == "select"}
    assert sel["select_p3_s75"]["out"] == "results/task1/select_p3_s75"
    a = [x for x in jobs if x["name"] == "compress_p3_s75"][0]["cmd"]
    assert a[a.index("--allocation") + 1] == "results/task1/select_p3_s75/selected_allocation.json"


def test_preflight_jobs_structure():
    jobs = K.stage_jobs("preflight")
    _deps_ok(jobs)
    assert all(j["name"].startswith("pf_") for j in jobs)
    kinds = {C.CONDITIONS[j["condition"]]["family"] for j in jobs if j["kind"] == "compress"}
    assert kinds == {"P1", "P2", "P2_2of4", "P3", "Q1", "Q2", "Q3"}
    assert {j["name"] for j in jobs if j["kind"] != "compress"} == {
        "pf_calib_stats", "pf_probe_quant", "pf_probe_prune", "pf_select_q3_int3", "pf_select_p3_s50"}
    for j in jobs:
        assert j["out"].startswith("results_smoke/preflight/")
        assert j["cmd"][j["cmd"].index("--limit-calib") + 1] == "1"
        assert ("--limit-dev" in j["cmd"]) == (j["entry"] in ("task1.select", "task1.compress"))
        assert all(w.startswith("notes/kaggle_status_smoke/pf_") for w in j["wait_for"])
        if j["kind"] == "compress":
            assert j["artifacts"].startswith("/kaggle/working/artifacts_smoke/")


def test_jobs_cli_refuses_existing(tmp_path):
    assert K.main(["jobs", "--stage", "run1", "--jobs-dir", str(tmp_path)]) == 0
    q0 = KQ.load_jobs(tmp_path / "run1_gpu0.json")
    assert q0[0]["name"] == "calib_stats" and set(q0[0]) == {"name", "cmd", "wait_for"}
    assert len(K.load_meta(tmp_path, "run1")) == 25
    with pytest.raises(FileExistsError):
        K.main(["jobs", "--stage", "run1", "--jobs-dir", str(tmp_path)])


@pytest.fixture(scope="module")
def queue_run(tiny_compress4_dir, tmp_path_factory):
    b = tmp_path_factory.mktemp("run1_queues")
    res, art, st, logs, jd = b / "results" / "task1", b / "artifacts", b / "status", b / "logs", b / "jobs"
    rc = K.main(["jobs", "--stage", "run1", "--jobs-dir", str(jd), "--results", str(res), "--artifacts", str(art),
                 "--status", str(st), "--model-path", str(tiny_compress4_dir), "--device", "cpu",
                 "--limit-calib", str(LIM_C), "--limit-dev", str(LIM_D)])
    assert rc == 0
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    procs = []
    for gpu in ("0", "1"):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "common.kaggle_queue", "--name", f"gpu{gpu}", "--gpu", gpu, "--jobs",
             str(jd / f"run1_gpu{gpu}.json"), "--logs", str(logs), "--status", str(st), "--poll", "0.5"],
            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
    meta = K.load_meta(jd, "run1")
    w = subprocess.run([sys.executable, "-m", "common.kaggle_queue", "--wait", str(st), "--names",
                        *[j["name"] for j in meta], "--logs", str(logs), "--poll", "1", "--report", "60",
                        "--timeout", "3000"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=3100)
    for p in procs:
        p.wait(timeout=600)
    assert w.returncode == 0, w.stdout[-4000:]
    assert [p.returncode for p in procs] == [0, 0]
    return {"base": b, "results": res, "artifacts": art, "status": st, "logs": logs, "jobs": jd, "meta": meta}


def test_queue_outputs_check(queue_run):
    meta = queue_run["meta"]
    assert K.check(meta, queue_run["status"]) == []
    for j in meta:
        assert (queue_run["logs"] / f"{j['name']}.log").is_file()
        assert KQ.read_int(queue_run["status"] / f"{j['name']}.exit") == 0
    p3 = json.loads((queue_run["results"] / "run1" / "p3_s75" / "run.json").read_text())
    assert p3["steps"]["allocation"]["file"].endswith("select_p3_s75/selected_allocation.json")
    q3 = json.loads((queue_run["results"] / "run1" / "q3_int4" / "run.json").read_text())
    assert q3["config"]["calib_stats_dir"] == str(queue_run["results"] / "calib_stats")


def test_queue_summary(queue_run, capsys):
    sm = K.summary(queue_run["meta"], queue_run["status"])
    assert len(sm["jobs"]) == 25 and all(r["exit"] == 0 and r["run_status"] == "ok" for r in sm["jobs"])
    assert all(r["wall_s"] > 0 for r in sm["jobs"])
    conds = {c["condition"]: c for c in sm["conditions"]}
    assert set(conds) == set(C.CONDITIONS)
    assert conds["p2_s50"]["zero_fraction"] == pytest.approx(0.5, abs=1e-3)
    assert conds["q1_b3"]["nominal_bits"] == 3 and conds["fp16"]["nominal_bits"] == 16
    assert conds["q3_int3"]["logical_eligible_bytes"] <= conds["q2_b3"]["logical_eligible_bytes"]
    K.print_summary(sm)
    out = capsys.readouterr().out
    assert "compress_p3_s75" in out and "q3_int4" in out


def test_queue_check_detects_problems(queue_run, tmp_path):
    meta = [dict(j) for j in queue_run["meta"]]
    bad = [j for j in meta if j["name"] == "compress_q1_b4"][0]
    bad["artifacts"] = str(tmp_path / "missing")
    probs = K.check(meta, queue_run["status"], hash_artifacts=False)
    assert probs == [f"compress_q1_b4: artifact files missing in {tmp_path / 'missing'}"]
    sel = [j for j in meta if j["name"] == "select_p3_s50"][0]
    sel["condition"] = "p3_s75"
    assert any("selected allocation is for p3_s50" in x for x in K.check(meta, queue_run["status"], False))


def test_queue_zip(queue_run, tmp_path):
    out = tmp_path / "t1_run1_results.zip"
    res = K.make_zip(out, queue_run["meta"], [queue_run["logs"], queue_run["status"], queue_run["jobs"]],
                     queue_run["results"])
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert res["n_files"] == len(names) and res["bytes"] == out.stat().st_size and len(res["sha256"]) == 64
    for j in queue_run["meta"]:
        assert any(n.endswith(f"/{j['name']}.log") for n in names)
        if j["kind"] == "compress":
            arc = (queue_run["results"] / "run1" / j["condition"] / K.RUN1_MANIFEST_COPY).as_posix().lstrip("/")
            assert arc in names
            with zipfile.ZipFile(out) as zf:
                assert zf.read(arc) == (queue_run["artifacts"] / j["condition"] / "manifest.json").read_bytes()
    assert not any(n.endswith(".safetensors") and "/artifacts/" in n for n in names)
    assert any(n.endswith("calib_stats/awq_scales.safetensors") for n in names)
    with pytest.raises(FileExistsError):
        K.make_zip(out, queue_run["meta"], [], queue_run["results"])
