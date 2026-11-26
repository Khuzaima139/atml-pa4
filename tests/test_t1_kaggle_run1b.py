"""Run-1b (Q3 design v2) job lists, smoke and projection, calib_stats check, and an end-to-end CPU run of both
run-1b queues through common.kaggle_queue with a tiny 4-block model, followed by summary, check, zip and final.
A second run with stand-in jobs (one probe part failing) checks dependency failure, summary, check, zip and final."""
import copy
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from common import kaggle_queue as KQ
from common.paths import REPO_ROOT, RESULTS_DIR
from task1 import config as C
from task1 import data as D
from task1 import kaggle_run1b as K
from task1 import smoke_run1b

LIM_C, LIM_D = 2, 2
HELD_OUT = set(D.HELD_OUT_FILES)
TINY_PARTS = ["0-1", "2-3"]
UNIT = {"wall_s", "peak_allocated_bytes", "peak_reserved_bytes", "host_maxrss_bytes"}
UNIT_EXTRA = {"load": {"includes_pristine_snapshot"}, "calib_pass": {"n_sequences"}, "dev_pass": {"n_sequences"},
              "worst_batch_calib": {"batch"}, "qs2v2_block": {"bits"}, "isolated_unit": {"unit", "bits"},
              "quant_build": {"condition", "build_timings"}}


def _deps_ok(jobs):
    order = {j["name"]: i for i, j in enumerate(jobs)}
    for j in jobs:
        for w in j["wait_for"]:
            dep = os.path.basename(w)[:-len(".done")]
            assert dep in order, (j["name"], dep)
            if [x for x in jobs if x["name"] == dep][0]["gpu"] == j["gpu"]:
                assert order[dep] < order[j["name"]], (j["name"], dep)


def test_run1b_jobs_structure():
    jobs = K.stage_jobs("run1b")
    assert [j["name"] for j in jobs if j["gpu"] == "0"] == ["probe_quant_v2_a", "select_q3v2_int4", "compress_q3_int4"]
    assert [j["name"] for j in jobs if j["gpu"] == "1"] == ["probe_quant_v2_b", "select_q3v2_int3", "compress_q3_int3"]
    _deps_ok(jobs)
    by = {j["name"]: j for j in jobs}
    S = "notes/kaggle_status/"
    assert by["probe_quant_v2_a"]["wait_for"] == [] and by["probe_quant_v2_b"]["wait_for"] == []
    for t in ("int4", "int3"):
        assert by[f"select_q3v2_{t}"]["wait_for"] == [S + "probe_quant_v2_a.done", S + "probe_quant_v2_b.done"]
    assert by["compress_q3_int4"]["wait_for"] == [S + "select_q3v2_int4.done"]
    assert by["compress_q3_int3"]["wait_for"] == [S + "select_q3v2_int3.done"]
    assert by["probe_quant_v2_a"]["out"] == "results/task1/probe_quant_v2/blocks_00_13"
    assert by["probe_quant_v2_b"]["out"] == "results/task1/probe_quant_v2/blocks_14_27"
    assert by["probe_quant_v2_a"]["blocks"] == [0, 13] and by["probe_quant_v2_b"]["blocks"] == [14, 27]
    assert by["select_q3v2_int3"]["out"] == "results/task1/select_q3v2_int3"
    for c in ("q3_int4", "q3_int3"):
        j = by[f"compress_{c}"]
        assert j["out"] == f"results/task1/run1b/{c}" and j["artifacts"] == f"/kaggle/working/artifacts/{c}"
        assert j["condition"] == c
    a = by["compress_q3_int3"]["cmd"]
    assert a[a.index("--allocation") + 1] == "results/task1/select_q3v2_int3/selected_allocation.json"
    s = by["select_q3v2_int4"]["cmd"]
    i = s.index("--probe")
    assert s[i + 1:i + 3] == ["results/task1/probe_quant_v2/blocks_00_13", "results/task1/probe_quant_v2/blocks_14_27"]
    assert s[s.index("--kind") + 1] == "q3v2" and s[s.index("--target") + 1] == "int4"
    for j in jobs:
        flat = " ".join(map(str, j["cmd"]))
        assert not any(h in flat for h in HELD_OUT) and "--limit" not in flat and "--model-path" not in flat
        assert j["cmd"][j["cmd"].index("--calib-stats") + 1] == "results/task1/calib_stats"
        assert j["cmd"][:3] == [sys.executable, "-m", j["entry"]]
    p = by["probe_quant_v2_b"]["cmd"]
    assert p[p.index("--blocks") + 1] == "14-27" and p[p.index("--kind") + 1] == "quant_v2"
    assert K.PARTS["a"][1] + 1 == K.PARTS["b"][0] and K.PARTS["b"][1] == C.N_LAYERS - 1


def test_preflight_jobs_structure():
    jobs = K.stage_jobs("preflight")
    _deps_ok(jobs)
    assert [j["name"] for j in jobs if j["gpu"] == "0"] == ["pf_calib_stats", "pf_probe_quant_v2_a",
                                                           "pf_select_q3v2_int4", "pf_compress_q3_int4"]
    assert [j["name"] for j in jobs if j["gpu"] == "1"] == ["pf_probe_quant_v2_b", "pf_select_q3v2_int3",
                                                           "pf_compress_q3_int3"]
    for j in jobs:
        assert j["out"].startswith("results_smoke/preflight/")
        assert j["cmd"][j["cmd"].index("--limit-calib") + 1] == "1"
        assert ("--limit-dev" in j["cmd"]) == (j["entry"] in ("task1.select", "task1.compress"))
        assert all(w.startswith("notes/kaggle_status_smoke/pf_") for w in j["wait_for"])
        if j["kind"] != "calib_stats":
            assert j["cmd"][j["cmd"].index("--calib-stats") + 1] == "results_smoke/preflight/calib_stats"
            assert "notes/kaggle_status_smoke/pf_calib_stats.done" in j["wait_for"] or j["kind"] == "select"
        if j["kind"] == "compress":
            assert j["artifacts"].startswith("/kaggle/working/artifacts_smoke/")


def test_jobs_cli_refuses_existing(tmp_path):
    assert K.main(["jobs", "--stage", "run1b", "--jobs-dir", str(tmp_path)]) == 0
    q1 = KQ.load_jobs(tmp_path / "run1b_gpu1.json")
    assert [x["name"] for x in q1] == ["probe_quant_v2_b", "select_q3v2_int3", "compress_q3_int3"]
    assert len(K.K.load_meta(tmp_path, "run1b")) == 6
    with pytest.raises(FileExistsError):
        K.main(["jobs", "--stage", "run1b", "--jobs-dir", str(tmp_path)])
    with pytest.raises(AssertionError):
        K.parse_parts(["0-1", "3-3"])


def test_calib_check_committed_files():
    rec = K.calib_check(RESULTS_DIR / "task1" / "calib_stats")
    assert rec["problems"] == [] and set(rec["files"]) == set(K.CALIB_FILES)
    assert rec["run_status"] == "ok" and rec["run_limits"]["calib"] is None
    assert rec["provenance"]["awq_scales_sha256"] == rec["files"]["awq_scales.safetensors"]["sha256"]


def test_calib_check_refuses_untracked_or_altered(tmp_path):
    d = tmp_path / "cs"
    shutil.copytree(RESULTS_DIR / "task1" / "calib_stats", d)
    rec = K.calib_check(d)
    assert len(rec["problems"]) == len(K.CALIB_FILES) and all("not tracked" in x for x in rec["problems"])
    (d / "awq_search.json").unlink()
    assert any("missing" in x for x in K.calib_check(d)["problems"])


# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_calib(tiny_compress4_dir, tmp_path_factory):
    b = tmp_path_factory.mktemp("run1b_calib")
    p = subprocess.run([sys.executable, "-m", "task1.calib_stats", "--out", str(b / "calib_stats"), "--model-path",
                        str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", str(LIM_C)], cwd=REPO_ROOT,
                       env=dict(os.environ, HF_HUB_OFFLINE="1"), capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-4000:]
    return b / "calib_stats"


@pytest.fixture(scope="module")
def smoke_run(tiny_compress4_dir, tiny_calib, tmp_path_factory):
    b = tmp_path_factory.mktemp("run1b_smoke")
    cmd = [sys.executable, "-m", "task1.smoke_run1b", "--out", str(b / "out"), "--calib-stats", str(tiny_calib),
           "--artifacts", str(b / "art"), "--model-path", str(tiny_compress4_dir), "--device", "cpu",
           "--limit-calib", str(LIM_C), "--limit-dev", str(LIM_D)]
    p = subprocess.run(cmd, cwd=REPO_ROOT, env=dict(os.environ, HF_HUB_OFFLINE="1"), capture_output=True, text=True,
                       timeout=900)
    assert p.returncode == 0, p.stderr[-4000:]
    return b


def test_smoke_outputs(smoke_run):
    r = json.loads((smoke_run / "out" / "run.json").read_text())
    assert r["status"] == "ok" and r["config"]["design"] == C.Q3V2_DESIGN
    assert sorted(x.name for x in (smoke_run / "out").iterdir()) == ["run.json"]
    s = r["steps"]
    assert set(s) == {"units", "calib_stats_source", "counts", "artifact", "pristine_after", "held_out_guard",
                      "files_written"}
    assert s["held_out_guard"]["attempts"] == []
    assert set(s["units"]) == set(smoke_run1b.UNITS)
    for k, u in s["units"].items():
        assert set(u) == UNIT | UNIT_EXTRA.get(k, set()), k
        assert u["wall_s"] > 0 and u["peak_allocated_bytes"] is None
    assert s["units"]["isolated_unit"] == {**s["units"]["isolated_unit"], "unit": "0.mlp_in", "bits": 2}
    c = s["counts"]
    assert c["n_layers"] == 4 and c["n_units"] == 16 and c["chunk_groups"] == 512
    assert c["unit_chunks_block0"] == [2, 1, 4, 2] and c["n_calibration"] == LIM_C and c["n_development"] == LIM_D
    man = json.loads((smoke_run / "art" / "manifest.json").read_text())
    assert man["design"] == C.Q3V2_DESIGN and {m["bits"] for m in man["matrices"]} == {4}
    assert s["artifact"]["file_bytes"] == man["file"]["bytes"]
    assert str(smoke_run / "art" / "artifact.safetensors") in s["files_written"]


def test_smoke_refuses_test_args_under_results(tmp_path):
    with pytest.raises(SystemExit, match="refusing"):
        smoke_run1b.main(["--out", str(RESULTS_DIR / "task1" / "smoke_refusal"), "--calib-stats", "x",
                          "--artifacts", str(tmp_path / "a"), "--limit-calib", "1"])


def test_plan_from_smoke(smoke_run):
    sm = json.loads((smoke_run / "out" / "run.json").read_text())
    u = {k: v["wall_s"] for k, v in sm["steps"]["units"].items()}
    pl = K.project(sm, 1_000_000, budget_h=10.5, disk_gb=18)
    rows = {r["name"]: r for r in pl["rows"]}
    assert list(rows) == [j["name"] for j in K.stage_jobs("run1b")]
    raw_a = (u["load"] + u["calib_pass"] + u["capture"] + 0 * u["block_forward"] + 14 * u["qs2v2_block"]
             + 4 * 4 * 14 * u["isolated_unit"])
    raw_b = (u["load"] + u["calib_pass"] + u["capture"] + 14 * u["block_forward"] + 14 * u["qs2v2_block"]
             + 4 * 4 * 14 * u["isolated_unit"])
    assert rows["probe_quant_v2_a"]["raw_s"] == pytest.approx(raw_a)
    assert rows["probe_quant_v2_b"]["raw_s"] == pytest.approx(raw_b)
    sel = u["load"] + 3 * (u["restore"] + u["quant_build"] + u["dev_pass"] + u["calib_pass"])
    assert rows["select_q3v2_int4"]["raw_s"] == pytest.approx(sel)
    f = K.K.SAFETY
    both = max(raw_a * f, 60 + raw_b * f)
    assert pl["queue_end_s"]["0"] == pytest.approx(both + rows["select_q3v2_int4"]["projected_s"]
                                                   + rows["compress_q3_int4"]["projected_s"])
    assert pl["queue_end_s"]["1"] == pytest.approx(both + rows["select_q3v2_int3"]["projected_s"]
                                                   + rows["compress_q3_int3"]["projected_s"])
    assert pl["over"] == []
    slow = copy.deepcopy(sm)
    slow["steps"]["units"]["isolated_unit"]["wall_s"] = 200.0  # 224 passes x 200 s x 1.15 > 10.5 h
    assert any("GPU 0 queue" in x for x in K.project(slow, 0)["over"])
    assert any("/kaggle/working" in x for x in K.project(sm, 10 ** 12)["over"])


def test_plan_cli(smoke_run, tmp_path):
    out = tmp_path / "plan.json"
    assert K.main(["plan", "--smoke", str(smoke_run / "out" / "run.json"), "--working-bytes", "0",
                   "--out", str(out)]) == 0
    assert set(json.loads(out.read_text())) >= {"rows", "queue_end_h", "over", "limits", "safety"}


# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def queue_run(tiny_compress4_dir, tiny_calib, tmp_path_factory):
    b = tmp_path_factory.mktemp("run1b_queues")
    res, art, st, logs, jd = b / "results" / "task1", b / "artifacts", b / "status", b / "logs", b / "jobs"
    rc = K.main(["jobs", "--stage", "run1b", "--jobs-dir", str(jd), "--results", str(res), "--artifacts", str(art),
                 "--status", str(st), "--calib-stats", str(tiny_calib), "--parts", *TINY_PARTS,
                 "--model-path", str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", str(LIM_C),
                 "--limit-dev", str(LIM_D)])
    assert rc == 0
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    procs = []
    for gpu in ("0", "1"):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "common.kaggle_queue", "--name", f"gpu{gpu}", "--gpu", gpu, "--jobs",
             str(jd / f"run1b_gpu{gpu}.json"), "--logs", str(logs), "--status", str(st), "--poll", "0.5"],
            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
    meta = json.loads((jd / "run1b_meta.json").read_text())
    w = subprocess.run([sys.executable, "-m", "common.kaggle_queue", "--wait-all", str(st), "--names",
                        *[j["name"] for j in meta], "--logs", str(logs), "--poll", "1", "--report", "60",
                        "--timeout", "3000", "--out", str(logs / "wait_run1b.json")],
                       cwd=REPO_ROOT, capture_output=True, text=True, timeout=3100)
    for p in procs:
        p.wait(timeout=600)
    assert w.returncode == 0, w.stdout[-4000:]
    assert f"{len(meta)} of {len(meta)} jobs ok" in w.stdout
    return {"base": b, "results": res, "artifacts": art, "status": st, "logs": logs, "jobs": jd, "meta": meta,
            "calib": tiny_calib}


def test_queue_outputs_check(queue_run):
    meta = queue_run["meta"]
    assert len(meta) == 6 and K.check(meta, queue_run["status"]) == []
    res = queue_run["results"]
    for t, c in (("int4", "q3_int4"), ("int3", "q3_int3")):
        sel = json.loads((res / f"select_q3v2_{t}" / "selected_allocation.json").read_text())
        assert sel["design"] == C.Q3V2_DESIGN and sel["probe_union"]["n_parts"] == 2
        assert sel["probe_union"]["block_ranges"] == [[0, 1], [2, 3]]
        run = json.loads((res / "run1b" / c / "run.json").read_text())
        assert run["steps"]["allocation"]["file"].endswith(f"select_q3v2_{t}/selected_allocation.json")
        assert run["config"]["calib_stats_dir"] == str(queue_run["calib"])
        man = json.loads((queue_run["artifacts"] / c / "manifest.json").read_text())
        assert man["design"] == C.Q3V2_DESIGN and man["condition"] == c
    for name in ("blocks_00_01", "blocks_02_03"):
        assert (res / "probe_quant_v2" / name / "sensitivity.json").is_file()


def test_queue_summary(queue_run, capsys):
    sm = K.summary(queue_run["meta"], queue_run["status"])
    assert len(sm["jobs"]) == 6 and all(r["exit"] == 0 and r["run_status"] == "ok" for r in sm["jobs"])
    assert sm["incomplete"] == []
    assert [b["target"] for b in sm["budgets"]] == ["int4", "int3"]
    for b in sm["budgets"]:
        assert [c["candidate_id"] for c in b["candidates"]] == [f"{b['condition']}.reference",
                                                                f"{b['condition']}.QS1v2", f"{b['condition']}.QS2v2"]
        assert sum(c["selected"] for c in b["candidates"]) == 1
        assert all(c["eligible_packed_bytes"] <= c["budget_bytes"] == b["budget_bytes"] for c in b["candidates"])
        assert all(set(c["bits_histogram"]) == {"attn_in", "attn_out", "mlp_in", "mlp_out"} for c in b["candidates"])
        win = [c for c in b["candidates"] if c["selected"]][0]
        assert win["candidate_id"] == b["winner"] and b["compress"]["dev_mean_nll"] == win["dev_mean_nll"]
    K.print_summary(sm)
    out = capsys.readouterr().out
    assert "int3 (q3_int3): winner" in out and "attn_in:" in out and "compress q3_int4" in out


def test_queue_check_detects_problems(queue_run, tmp_path):
    meta = [dict(j) for j in queue_run["meta"]]
    bad = [j for j in meta if j["name"] == "probe_quant_v2_b"][0]
    bad["blocks"] = [2, 2]
    probs = K.check(meta, queue_run["status"], hash_artifacts=False)
    assert any(x.startswith("probe_quant_v2_b: sensitivity design") for x in probs)
    meta = [dict(j) for j in queue_run["meta"]]
    art = tmp_path / "art"
    shutil.copytree(queue_run["artifacts"] / "q3_int4", art)
    man = json.loads((art / "manifest.json").read_text())
    man.pop("design")
    (art / "manifest.json").write_text(json.dumps(man))
    [j for j in meta if j["name"] == "compress_q3_int4"][0]["artifacts"] = str(art)
    probs = K.check(meta, queue_run["status"], hash_artifacts=False)
    assert "compress_q3_int4: manifest sha256 differs from run.json" in probs


def test_queue_zip_and_final(queue_run, tmp_path):
    jd, st = str(queue_run["jobs"]), str(queue_run["status"])
    report = tmp_path / "check_run1b.json"
    assert K.main(["check", "--jobs-dir", jd, "--status", st, "--out", str(report)]) == 0
    assert json.loads(report.read_text())["problems"] == []
    out = tmp_path / "t1_run1b_results.zip"
    assert K.main(["zip", "--out", str(out), "--jobs-dir", jd, "--status", st, "--include",
                   str(queue_run["logs"]), st, jd]) == 0
    res = queue_run["results"]
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        for c in ("q3_int4", "q3_int3"):
            arc = (res / "run1b" / c / "artifact_manifest.json").as_posix().lstrip("/")
            assert zf.read(arc) == (queue_run["artifacts"] / c / "manifest.json").read_bytes()
    for sub in ("probe_quant_v2/blocks_00_01/sensitivity.json", "probe_quant_v2/blocks_02_03/run.json",
                "select_q3v2_int4/selected_allocation.json", "select_q3v2_int3/candidates.jsonl.gz",
                "run1b/q3_int3/run.json", "run1b/q3_int4/projections.jsonl.gz"):
        assert (res / sub).as_posix().lstrip("/") in names, sub
    assert not any(n.endswith(".safetensors") for n in names)
    assert not any("calib_stats" in n for n in names)
    assert any(n.endswith("/wait_run1b.json") for n in names) and any(n.endswith("/run1b_meta.json") for n in names)
    assert K.main(["final", "--jobs-dir", jd, "--status", st, "--check-report", str(report), "--zip", str(out)]) == 0


def _cli(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def test_failed_part_fails_dependents_and_final(tmp_path):
    res, st, logs, jd = tmp_path / "results" / "task1", tmp_path / "status", tmp_path / "logs", tmp_path / "jobs"
    jobs = K.stage_jobs("run1b", results=res, artifacts=tmp_path / "art", status=st, calib_stats=tmp_path / "cs",
                        parts=K.parse_parts(TINY_PARTS))
    by = {j["name"]: j for j in jobs}
    by["probe_quant_v2_a"]["cmd"] = [sys.executable, "-c", "print('part a stand-in')"]
    by["probe_quant_v2_b"]["cmd"] = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]
    for n in ("select_q3v2_int4", "select_q3v2_int3", "compress_q3_int4", "compress_q3_int3"):
        by[n]["cmd"] = [sys.executable, "-c", "print('must not run')"]
    K.K.write_jobs("run1b", jd, jobs)
    procs = [subprocess.Popen([sys.executable, "-m", "common.kaggle_queue", "--name", f"gpu{g}", "--gpu", g, "--jobs",
                               str(jd / f"run1b_gpu{g}.json"), "--logs", str(logs), "--status", str(st), "--poll",
                               "0.1"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
             for g in ("0", "1")]
    names = [j["name"] for j in jobs]
    w = _cli("common.kaggle_queue", "--wait-all", str(st), "--names", *names, "--logs", str(logs), "--poll", "0.1",
             "--report", "1", "--timeout", "120", "--out", str(logs / "wait_run1b.json"))
    assert [p.wait(timeout=60) for p in procs] == [1, 1]
    assert w.returncode == 1, w.stdout + w.stderr
    assert KQ.read_int(st / "probe_quant_v2_a.exit") == 0 and KQ.read_int(st / "probe_quant_v2_b.exit") == 3
    for n in ("select_q3v2_int4", "select_q3v2_int3", "compress_q3_int4", "compress_q3_int3"):
        assert KQ.read_int(st / f"{n}.exit") == KQ.DEP_FAILED, n
        assert "must not run" not in (logs / f"{n}.log").read_text()
    sm = _cli("task1.kaggle_run1b", "summary", "--jobs-dir", str(jd), "--status", str(st))
    assert sm.returncode == 0 and "select_q3v2_int4: selection outputs missing" in sm.stdout, sm.stdout + sm.stderr
    report = logs / "check_run1b.json"
    ck = _cli("task1.kaggle_run1b", "check", "--jobs-dir", str(jd), "--status", str(st), "--out", str(report))
    assert ck.returncode == 1 and "Traceback" not in ck.stderr, ck.stdout + ck.stderr
    probs = json.loads(report.read_text())["problems"]
    assert "probe_quant_v2_b: exit 3, done False" in probs and f"select_q3v2_int3: exit {KQ.DEP_FAILED}, done False" \
        in probs
    zp = tmp_path / "t1_run1b_results.zip"
    z = _cli("task1.kaggle_run1b", "zip", "--out", str(zp), "--jobs-dir", str(jd), "--status", str(st), "--include",
             str(logs), str(st), str(jd))
    assert z.returncode == 0, z.stdout + z.stderr
    assert "not in the zip (missing)" in z.stdout and zp.is_file()
    f = _cli("task1.kaggle_run1b", "final", "--jobs-dir", str(jd), "--status", str(st), "--check-report", str(report),
             "--zip", str(zp))
    assert f.returncode == 1 and "job probe_quant_v2_b: exit 3" in f.stdout
