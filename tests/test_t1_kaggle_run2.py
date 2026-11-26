"""Run-2 (held-out evaluation) artifact location, job lists, projection and balancing, and an end-to-end CPU run
through common.kaggle_queue with a tiny 4-block model: locate and verify the run-1 / run-1b artifacts in a stand-in
notebook-output tree, smoke stage, plan, both evaluation queues and the GPU-0 resources queue (which waits for every
evaluation .exit file), then summary, check, zip and final. A second run with stand-in jobs checks that a failed
evaluation job does not block the resources phase and is reported by check and final."""
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from common import kaggle_queue as KQ
from common.io import sha256_file
from common.paths import REPO_ROOT, RESULTS_DIR
from task1 import config as C
from task1 import data as D
from task1 import kaggle_run2 as K
from task1 import smoke_run2

HELD_OUT = set(D.HELD_OUT_FILES)
LIM_C, LIM_D, LIM_W, LIM_H = 2, 2, 2, 4
E2E = ("fp16", "p1_s25", "p3_s50", "q3_int3")


def _fake_located(conds=K.CONDITIONS, root="results/task1"):
    return {"run1_root": root, "problems": [], "conditions": {
        c: {"dir": f"/kaggle/input/nb/{'atml-pa4-t1-run1b' if c in K.RUN1B_CONDITIONS else 'atml-pa4-t1-run1'}"
                   f"/artifacts/{c}", "manifest_sha256": f"sha-{c}", "file_bytes": 1000 + i}
        for i, c in enumerate(conds)}}


def test_run2_jobs_structure():
    loc = _fake_located()
    jobs = K.stage_jobs("run2", loc)
    ev = [j for j in jobs if j["phase"] == "eval"]
    res = [j for j in jobs if j["phase"] == "resources"]
    assert sorted(j["condition"] for j in ev) == sorted(C.CONDITIONS) and len(ev) == 18
    assert [j["condition"] for j in res] == ["fp16"] + [c for c in C.CONDITIONS if c != "fp16"]
    assert {j["queue"] for j in ev} == {"gpu0", "gpu1"} and {j["queue"] for j in res} == {"res_gpu0"}
    assert all(j["gpu"] == "0" for j in res)
    exits = [f"notes/kaggle_status/eval_{c}.exit" for c in C.CONDITIONS]
    for j in res:
        assert sorted(j["wait_for"]) == sorted(exits) and "--resources-only" in j["cmd"]
        assert j["out"] == f"results/task1/run2_resources/{j['condition']}"
    for j in ev:
        assert j["wait_for"] == [] and "--no-resources" in j["cmd"] and "--resources-only" not in j["cmd"]
        assert j["out"] == f"results/task1/run2/{j['condition']}"
        assert j["expect"]["hellaswag_n"] == C.hellaswag_n(C.CONDITIONS[j["condition"]]["hellaswag"])
    for j in jobs:
        flat = " ".join(map(str, j["cmd"]))
        assert "--limit" not in flat and "--model-path" not in flat and "--repo-root" not in flat
        assert not any(h in flat for h in HELD_OUT)
        c = j["condition"]
        if c == "fp16":
            assert j["entry"] == "task1.baseline" and "--condition" not in j["cmd"]
        else:
            a = j["cmd"]
            assert j["entry"] == "task1.evaluate_condition" and a[a.index("--condition") + 1] == c
            assert a[a.index("--artifacts") + 1] == loc["conditions"][c]["dir"]
            run1 = "run1b" if c in ("q3_int4", "q3_int3") else "run1"
            assert a[a.index("--run1") + 1] == f"results/task1/{run1}/{c}" == j["run1"]
            assert j["manifest_sha256"] == f"sha-{c}"


def test_balance_longest_first():
    secs = {"eval_a": 100.0, "eval_b": 90.0, "eval_c": 50.0, "eval_d": 45.0, "eval_e": 10.0}
    a = K.balance(secs, list("abcde"))
    # loads (GPU 1 starts at 60 s): a -> 0 (100), b -> 1 (150), c -> 0 (150), d -> 0 on the tie (195), e -> 1 (160)
    assert a == {"eval_a": "0", "eval_b": "1", "eval_c": "0", "eval_d": "0", "eval_e": "1"}
    loads = {g: sum(s for n, s in secs.items() if a[n] == g) for g in "01"}
    assert abs(loads["0"] - (loads["1"] + K.STAGGER_S)) <= max(secs.values())


def test_run2_jobs_follow_plan():
    loc = _fake_located()
    rows = [{"name": f"eval_{c}", "phase": "eval", "projected_s": float(100 + i)} for i, c in enumerate(C.CONDITIONS)]
    assign = K.balance({r["name"]: r["projected_s"] for r in rows}, list(C.CONDITIONS))
    jobs = K.stage_jobs("run2", loc, plan={"assignment": assign, "rows": rows})
    for j in jobs:
        if j["phase"] == "eval":
            assert j["gpu"] == assign[f"eval_{j['condition']}"]
    g0 = [j["condition"] for j in jobs if j["queue"] == "gpu0"]
    assert g0 == sorted(g0, key=lambda c: -list(C.CONDITIONS).index(c))  # longest (here: latest) first
    with pytest.raises(SystemExit, match="no GPU"):
        K.stage_jobs("run2", loc, plan={"assignment": {}, "rows": rows})


def test_preflight_and_smoke_jobs():
    loc = _fake_located()
    pf = K.stage_jobs("preflight", loc)
    assert [(j["name"], j["gpu"]) for j in pf] == [("pf_eval_fp16", "0"), ("pf_eval_q3_int3", "0"),
                                                  ("pf_eval_p3_s50", "1")]
    for j in pf:
        a = j["cmd"]
        assert j["out"].startswith("results_smoke/preflight/run2/") and "--no-resources" in a
        assert a[a.index("--limit-windows") + 1] == "1" and a[a.index("--limit-hellaswag") + 1] == "1"
        assert ("--limit-calib" in a) == (j["condition"] == "fp16") and "--limit-dev" not in a
    sm = K.stage_jobs("smoke", loc)
    assert [(j["name"], j["queue"]) for j in sm] == [("sm_eval_fp16", "gpu0"), ("sm_worst_batch", "gpu0"),
                                                    ("sm_eval_q3_int4", "gpu1"), ("sm_res_q3_int4", "gpu1")]
    by = {j["name"]: j for j in sm}
    for n in ("sm_eval_fp16", "sm_eval_q3_int4"):
        a = by[n]["cmd"]
        assert a[a.index("--limit-hellaswag") + 1] == "100" and "--limit-windows" not in a
    assert "--limit-hellaswag" not in by["sm_res_q3_int4"]["cmd"] and "--resources-only" in by["sm_res_q3_int4"]["cmd"]
    assert by["sm_res_q3_int4"]["out"] == "results_smoke/smoke/resources/q3_int4"
    assert by["sm_worst_batch"]["entry"] == "task1.smoke_run2"


def test_jobs_cli_refuses_problems_and_existing(tmp_path):
    loc = tmp_path / "loc.json"
    loc.write_text(json.dumps(_fake_located()))
    jd = tmp_path / "jobs"
    assert K.main(["jobs", "--stage", "run2", "--jobs-dir", str(jd), "--located", str(loc)]) == 0
    assert sorted(p.name for p in jd.iterdir()) == ["run2_gpu0.json", "run2_gpu1.json", "run2_meta.json",
                                                    "run2_res_gpu0.json"]
    assert len(KQ.load_jobs(jd / "run2_res_gpu0.json")) == 18
    with pytest.raises(FileExistsError):
        K.main(["jobs", "--stage", "run2", "--jobs-dir", str(jd), "--located", str(loc)])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**_fake_located(), "problems": ["x"]}))
    with pytest.raises(SystemExit, match="problems"):
        K.main(["jobs", "--stage", "run2", "--jobs-dir", str(tmp_path / "j2"), "--located", str(bad)])


def test_locate_committed_manifests_pick_run1b_for_q3(tmp_path):
    inp = tmp_path / "input"
    conds = ("p1_s25", "q3_int4", "q3_int3")
    for run, nb in (("run1", "atml-pa4-t1-run1"), ("run1b", "atml-pa4-t1-run1b")):
        for c in conds:
            src = RESULTS_DIR / "task1" / run / c / "artifact_manifest.json"
            if src.is_file():
                d = inp / "notebooks" / "u" / nb / "artifacts" / c
                d.mkdir(parents=True)
                shutil.copyfile(src, d / "manifest.json")
                (d / "artifact.safetensors").write_bytes(b"")
    rec = K.locate(inp, RESULTS_DIR / "task1", conds, hash_files=False)
    assert rec["roots"]["run1"][0].endswith("atml-pa4-t1-run1") and rec["roots"]["run1b"][0].endswith("-run1b")
    for c in ("q3_int4", "q3_int3"):
        cc = rec["conditions"][c]
        assert "/atml-pa4-t1-run1b/" in cc["dir"] and len(cc["not_used"]) == 1 and "-run1/" in cc["not_used"][0] + "/"
        assert cc["design"] == C.Q3V2_DESIGN
        assert cc["allocation_sha256"] == sha256_file(REPO_ROOT / "task1" / "frozen" / f"{c}.json")
        assert cc["manifest_copy_git"]["tracked"] and cc["manifest_copy_git"]["clean"]
    assert "/atml-pa4-t1-run1/" in rec["conditions"]["p1_s25"]["dir"]
    assert all("artifact size 0" in x for x in rec["problems"]) and len(rec["problems"]) == 3
    shutil.copytree(inp / "notebooks" / "u" / "atml-pa4-t1-run1" / "artifacts" / "p1_s25",
                    inp / "other" / "artifacts" / "p1_s25")
    rec = K.locate(inp, RESULTS_DIR / "task1", ("p1_s25",), hash_files=False)
    assert any("2 artifact directories match" in x for x in rec["problems"])
    rec = K.locate(inp, RESULTS_DIR / "task1", ("q2_b4",), hash_files=False)
    assert any("0 artifact directories match" in x for x in rec["problems"])


def _smoke_record(path, timings, steps, peak=1 << 30):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "ok", "timings_s": timings, "steps": steps,
                                "peak_vram_allocated_bytes": {"0": peak}}))


def _fake_smoke(d, peak=1 << 30):
    _smoke_record(d / "eval" / "fp16" / "run.json",
                  {"load_s": 20, "enumeration_s": 1, "ppl_wikitext2_test_s": 30, "ppl_c4_validation_s": 30,
                   "hellaswag_s": 10, "calib_nll_c4_calibration_s": 20, "calib_nll_c4_development_s": 5,
                   "lmeval_verify_s": 40}, {"hellaswag": {"n": 100}, "lmeval_verify": {"n_examples": 100}}, peak)
    _smoke_record(d / "eval" / "q3_int4" / "run.json",
                  {"load_s": 20, "artifact_load_s": 10, "calib_nll_c4_development_s": 5, "ppl_wikitext2_test_s": 30,
                   "ppl_c4_validation_s": 30, "hellaswag_s": 10}, {"hellaswag": {"n": 100}}, peak)
    _smoke_record(d / "resources" / "q3_int4" / "run.json", {"load_s": 20, "artifact_load_s": 10, "resources_s": 60},
                  {}, peak)
    _smoke_record(d / "worst_batch" / "run.json", {}, {"units": {"load": {"peak_allocated_bytes": peak},
                                                               "worst_batch_hellaswag": {"peak_allocated_bytes": peak}}})


def test_plan_formulas_and_limits(tmp_path):
    _fake_smoke(tmp_path)
    loc = _fake_located()
    for c in loc["conditions"].values():
        c["file_bytes"] = 1000
    loc["conditions"]["q1_b8"]["file_bytes"] = 3000
    pl = K.project(tmp_path, loc, 0)
    rows = {r["name"]: r for r in pl["rows"]}
    assert len(rows) == 36
    assert rows["eval_fp16"]["raw_s"] == pytest.approx(20 + 1 + 60 + 0.1 * 2000 + 25 + 0.4 * 512)
    assert rows["eval_q3_int3"]["raw_s"] == pytest.approx(20 + 10 + 5 + 60 + 0.1 * 2000)
    assert rows["eval_p1_s25"]["raw_s"] == pytest.approx(20 + 10 + 5 + 60 + 0.1 * 512)
    assert rows["eval_q1_b8"]["raw_s"] == pytest.approx(20 + 30 + 5 + 60 + 0.1 * 512)  # artifact load x 3
    assert rows["res_fp16"]["raw_s"] == pytest.approx(20 + 1 + 60)
    assert rows["res_q3_int3"]["raw_s"] == pytest.approx(20 + 10 + 60)
    f = K.K.SAFETY
    ev = {n: r["projected_s"] for n, r in rows.items() if n.startswith("eval_")}
    assert pl["assignment"] == K.balance(ev, list(C.CONDITIONS))
    e0 = sum(s for n, s in ev.items() if pl["assignment"][n] == "0")
    e1 = 60 + sum(s for n, s in ev.items() if pl["assignment"][n] == "1")
    assert pl["queue_end_s"]["1"] == pytest.approx(e1)
    assert pl["queue_end_s"]["0"] == pytest.approx(max(e0, e1) + sum(r["projected_s"] for n, r in rows.items()
                                                                    if n.startswith("res_")))
    assert rows["eval_fp16"]["projected_s"] == pytest.approx(rows["eval_fp16"]["raw_s"] * f)
    assert pl["over"] == []
    assert any("GPU 0 queue" in x for x in K.project(tmp_path, loc, 0, budget_h=0.5)["over"])
    hot = tmp_path / "hot"
    _fake_smoke(hot, peak=15 * 2**30)
    assert any("peak allocated VRAM" in x for x in K.project(hot, loc, 0)["over"])


def test_smoke_run2_refuses_results_path(tmp_path):
    with pytest.raises(SystemExit, match="refusing"):
        smoke_run2.main(["--out", str(RESULTS_DIR / "task1" / "smoke_refusal")])


def test_longest_selection():
    reqs = [{"context_tokens": [[1] * 3, [1] * 3], "continuation_tokens": [[1], [1] * 5]},
            {"context_tokens": [[1] * 7, [1] * 7], "continuation_tokens": [[1], [1] * 2]}]
    assert smoke_run2.longest_sequences(reqs, 3) == [(1, 1, 9), (0, 1, 8), (1, 0, 8)]
    assert smoke_run2.longest_examples(reqs, 1) == [1]


# ---------------------------------------------------------------------------------------------------------------
def _env():
    return dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")


def _run(args, timeout=1800):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True,
                          timeout=timeout)


def _git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


@pytest.fixture(scope="module")
def built(tiny_compress4_dir, tmp_path_factory):
    """A stand-in repository (run-1 / run-1b records, committed manifest copies, frozen P3/Q3 allocations) and a
    stand-in notebook-output tree holding the artifacts."""
    b = tmp_path_factory.mktemp("run2")
    repo = b / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    rr = repo / "results" / "task1"
    nb = {"run1": b / "input" / "notebooks" / "u" / "atml-pa4-t1-run1",
          "run1b": b / "input" / "notebooks" / "u" / "atml-pa4-t1-run1b"}
    t = ["--model-path", str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", str(LIM_C)]
    td = [*t, "--limit-dev", str(LIM_D)]

    def art(c):
        return nb["run1b" if c in K.RUN1B_CONDITIONS else "run1"] / "artifacts" / c

    steps = [
        ["task1.calib_stats", "--out", str(b / "calib"), *t],
        ["task1.probe", "--kind", "prune", "--calib-stats", str(b / "calib"), "--out", str(b / "probe_prune"), *t],
        ["task1.probe", "--kind", "quant", "--calib-stats", str(b / "calib"), "--out", str(b / "probe_quant"), *t],
        ["task1.select", "--kind", "p3", "--target", "s50", "--probe", str(b / "probe_prune"), "--calib-stats",
         str(b / "calib"), "--out", str(b / "sel_p3"), *td],
        ["task1.select", "--kind", "q3", "--target", "int3", "--probe", str(b / "probe_quant"), "--calib-stats",
         str(b / "calib"), "--out", str(b / "sel_q3"), *td],
        ["task1.compress", "--condition", "fp16", "--out", str(rr / "run1" / "fp16"), "--artifacts", str(art("fp16")),
         *td],
        ["task1.compress", "--condition", "p1_s25", "--out", str(rr / "run1" / "p1_s25"), "--artifacts",
         str(art("p1_s25")), *td],
        ["task1.compress", "--condition", "p3_s50", "--allocation", str(b / "sel_p3" / "selected_allocation.json"),
         "--out", str(rr / "run1" / "p3_s50"), "--artifacts", str(art("p3_s50")), *td],
        ["task1.compress", "--condition", "q3_int3", "--allocation", str(b / "sel_q3" / "selected_allocation.json"),
         "--calib-stats", str(b / "calib"), "--out", str(rr / "run1b" / "q3_int3"), "--artifacts",
         str(art("q3_int3")), *td],
    ]
    for a in steps:
        p = _run(a)
        assert p.returncode == 0, (a[:3], p.stderr[-4000:])
    for c in E2E:
        shutil.copyfile(art(c) / "manifest.json", K.run1_dir(rr, c) / K.RUN1_MANIFEST_COPY)
    _git(repo, "add", "--", "results")
    _git(repo, "commit", "-q", "-m", "Import run-1 outputs")
    fz = repo / "task1" / "frozen"
    fz.mkdir(parents=True)
    shutil.copyfile(b / "sel_p3" / "selected_allocation.json", fz / "p3_s50.json")
    shutil.copyfile(b / "sel_q3" / "selected_allocation.json", fz / "q3_int3.json")
    _git(repo, "add", "--", "task1/frozen")
    _git(repo, "commit", "-q", "-m", "Freeze allocations")
    loc = b / "located.json"
    assert K.main(["locate", "--inputs", str(b / "input"), "--out", str(loc), "--run1-root", str(rr),
                   "--conditions", *E2E]) == 0
    return {"base": b, "repo": repo, "rr": rr, "model": tiny_compress4_dir, "located": loc,
            "test": ["--model-path", str(tiny_compress4_dir), "--device", "cpu", "--repo-root", str(repo),
                     "--limit-windows", str(LIM_W), "--limit-hellaswag", str(LIM_H), "--limit-dev", str(LIM_D),
                     "--limit-calib", str(LIM_C)]}


def test_located_record(built):
    rec = json.loads(built["located"].read_text())
    assert rec["problems"] == [] and set(rec["conditions"]) == set(E2E)
    for c in E2E:
        cc = rec["conditions"][c]
        assert cc["file_sha256_verified"] is True and cc["not_used"] == []
        assert cc["manifest_sha256"] == sha256_file(K.run1_dir(built["rr"], c) / K.RUN1_MANIFEST_COPY)
        assert ("/atml-pa4-t1-run1b/" in cc["dir"]) == (c == "q3_int3")
    bad = built["base"] / "tampered_input"
    shutil.copytree(built["base"] / "input", bad)
    st = next(bad.rglob("p1_s25/artifact.safetensors"))
    data = bytearray(st.read_bytes())
    data[-1] ^= 1
    st.write_bytes(bytes(data))
    rec = K.locate(bad, built["rr"], ("p1_s25",))
    assert rec["problems"] == [f"p1_s25: artifact sha256 {sha256_file(st)} != manifest "
                               f"{rec['conditions']['p1_s25']['file_sha256_manifest']}"]


def _queues(jd, stage, qnames, st, logs):
    return [subprocess.Popen([sys.executable, "-m", "common.kaggle_queue", "--name", q, "--gpu",
                              "0" if q.endswith("0") else "1", "--jobs", str(jd / f"{stage}_{q}.json"), "--logs",
                              str(logs), "--status", str(st), "--poll", "0.5"],
                             cwd=REPO_ROOT, env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            for q in qnames]


def _wait_all(st, names, logs, out, timeout=3000):
    return subprocess.run([sys.executable, "-m", "common.kaggle_queue", "--wait-all", str(st), "--names", *names,
                           "--logs", str(logs), "--poll", "1", "--report", "60", "--timeout", str(timeout),
                           "--out", str(out)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout + 100)


@pytest.fixture(scope="module")
def smoke(built):
    b = built["base"] / "smoke"
    jd, st, logs = b / "jobs", b / "status", b / "logs"
    test = [x for x in built["test"] if True]
    i = test.index("--limit-windows")
    del test[i:i + 2]  # full (tiny-data) suites as in the smoke stage
    assert K.main(["jobs", "--stage", "smoke", "--jobs-dir", str(jd), "--located", str(built["located"]),
                   "--results", str(b / "results"), "--status", str(st), "--conditions", *E2E,
                   "--smoke-condition", "q3_int3", *test]) == 0
    meta = K.K.load_meta(jd, "smoke")
    procs = _queues(jd, "smoke", ["gpu0", "gpu1"], st, logs)
    w = _wait_all(st, [j["name"] for j in meta], logs, logs / "wait_smoke.json")
    for p in procs:
        p.wait(timeout=600)
    assert w.returncode == 0, w.stdout[-4000:]
    return {"base": b, "meta": meta, "status": st, "results": b / "results"}


def test_smoke_outputs_and_plan(built, smoke):
    assert K.check(smoke["meta"], smoke["status"]) == []
    r = json.loads((smoke["results"] / "worst_batch" / "run.json").read_text())
    assert r["status"] == "ok" and set(r["steps"]["units"]) == set(smoke_run2.UNITS)
    u = r["steps"]["units"]
    assert u["worst_batch_hellaswag"]["batch"][0] == min(C.HELLASWAG_BATCH_SIZE, 4 * LIM_H)
    assert u["worst_batch_lmeval"]["n_requests"] == 4 * min(C.LMEVAL_BATCH_SIZE // 4, LIM_H)
    assert "loglik" not in json.dumps(r) and "acc" not in json.dumps(r["steps"])
    out = smoke["base"] / "plan.json"
    assert K.main(["plan", "--smoke-dir", str(smoke["results"]), "--located", str(built["located"]),
                   "--working-bytes", "0", "--smoke-condition", "q3_int3", "--conditions", *E2E,
                   "--out", str(out)]) == 0
    pl = json.loads(out.read_text())
    assert set(pl["assignment"]) == {f"eval_{c}" for c in E2E} and pl["over"] == []
    assert len(pl["rows"]) == 2 * len(E2E)


@pytest.fixture(scope="module")
def run2(built, smoke):
    b = built["base"] / "run2"
    jd, st, logs = b / "jobs", b / "status", b / "logs"
    assert K.main(["jobs", "--stage", "run2", "--jobs-dir", str(jd), "--located", str(built["located"]),
                   "--plan", str(smoke["base"] / "plan.json"), "--results", str(b / "results" / "task1"),
                   "--status", str(st), "--conditions", *E2E, *built["test"]]) == 0
    meta = K.K.load_meta(jd, "run2")
    qn = K.queues(meta)
    assert set(qn) <= {"gpu0", "gpu1", "res_gpu0"} and "res_gpu0" in qn
    procs = _queues(jd, "run2", qn, st, logs)
    w = _wait_all(st, [j["name"] for j in meta], logs, logs / "wait_run2.json")
    for p in procs:
        p.wait(timeout=600)
    assert w.returncode == 0, w.stdout[-4000:]
    return {"base": b, "jobs": jd, "status": st, "logs": logs, "meta": meta, "results": b / "results" / "task1"}


def test_run2_outputs_and_order(run2):
    meta, st = run2["meta"], run2["status"]
    assert K.check(meta, st) == []
    ev_end = max(json.loads((st / f"{j['name']}.json").read_text())["end"] for j in meta if j["phase"] == "eval")
    res = [json.loads((st / f"{j['name']}.json").read_text()) for j in meta if j["phase"] == "resources"]
    assert all(r["start"] >= ev_end for r in res)  # resources start after every evaluation job has ended
    assert [r["name"] for r in res][0] == "res_fp16"
    R = run2["results"]
    for c in E2E:
        e = json.loads((R / "run2" / c / "run.json").read_text())
        r = json.loads((R / "run2_resources" / c / "run.json").read_text())
        assert e["config"]["resources_mode"] == "skip" and r["config"]["resources_mode"] == "only"
        assert r["steps"]["held_out_guard"]["attempts"] == [] and "resources" not in e["steps"]
        if c != "fp16":
            assert e["steps"]["reproduction"]["reproduces_run1"] is True
            assert e["config"]["run1_dir"] == str(K.run1_dir(run2["base"].parent / "repo" / "results" / "task1", c))
            assert e["steps"]["inputs"]["freeze"]["applies"] is (c in ("p3_s50", "q3_int3"))
        else:
            assert e["steps"]["lmeval_verify"]["acceptance"] == "PASS"


def test_run2_summary(run2, capsys):
    sm = K.summary(run2["meta"], run2["status"])
    assert sm["incomplete"] == [] and len(sm["jobs"]) == 2 * len(E2E)
    rows = {c["condition"]: c for c in sm["conditions"]}
    assert set(rows) == set(E2E)
    for c, r in rows.items():
        assert r["wikitext2_test"]["N"] > 0 and r["hellaswag"]["n"] == LIM_H
        assert r["prefill_512_ms"] > 0 and r["decode_ms"] > 0 and r["logical_whole_model_bytes"] > 0
        assert (r["file_bytes"] is None) == (c == "fp16") and (r["lmeval_acceptance"] == "PASS") == (c == "fp16")
    K.print_summary(sm)
    out = capsys.readouterr().out
    assert "q3_int3" in out and "PASS" in out


def test_run2_check_detects_problems(run2, tmp_path):
    meta = [dict(j) for j in run2["meta"]]
    j = [x for x in meta if x["name"] == "eval_p1_s25"][0]
    j["manifest_sha256"] = "0" * 64
    j["expect"] = {**j["expect"], "hellaswag_n": 7}
    probs = K.check(meta, run2["status"])
    assert "eval_p1_s25: artifact manifest sha256 differs from the located artifact" in probs
    assert f"eval_p1_s25: HellaSwag n {LIM_H} != 7" in probs


def test_run2_zip_and_final(run2, tmp_path):
    jd, st = str(run2["jobs"]), str(run2["status"])
    report = tmp_path / "check_run2.json"
    assert K.main(["check", "--jobs-dir", jd, "--status", st, "--out", str(report)]) == 0
    out = tmp_path / "t1_run2_results.zip"
    assert K.main(["zip", "--out", str(out), "--jobs-dir", jd, "--status", st, "--include",
                   str(run2["logs"]), st, jd]) == 0
    R = run2["results"]
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    for c in E2E:
        for sub in ("run2/%s/run.json" % c, "run2_resources/%s/run.json" % c):
            assert (R / sub).as_posix().lstrip("/") in names
    assert (R / "run2/fp16/lmeval_verify.json").as_posix().lstrip("/") in names
    assert not any(n.endswith(".safetensors") or n.endswith("manifest.json") and "artifacts" in n for n in names)
    assert any(n.endswith("/wait_run2.json") for n in names) and any(n.endswith("/run2_meta.json") for n in names)
    assert K.main(["final", "--jobs-dir", jd, "--status", st, "--check-report", str(report), "--zip", str(out)]) == 0


def test_failed_evaluation_does_not_block_resources(tmp_path):
    loc = _fake_located(("fp16", "p1_s25"))
    st, logs, jd = tmp_path / "status", tmp_path / "logs", tmp_path / "jobs"
    jobs = K.stage_jobs("run2", loc, results=tmp_path / "r", status=st, conditions=["fp16", "p1_s25"])
    by = {j["name"]: j for j in jobs}
    by["eval_fp16"]["cmd"] = [sys.executable, "-c", "print('eval fp16 stand-in')"]
    by["eval_p1_s25"]["cmd"] = [sys.executable, "-c", "import sys; print('boom'); sys.exit(4)"]
    for n in ("res_fp16", "res_p1_s25"):
        by[n]["cmd"] = [sys.executable, "-c", f"print('{n} ran')"]
    K.write_jobs("run2", jd, jobs)
    procs = _queues(jd, "run2", K.queues(jobs), st, logs)
    names = [j["name"] for j in jobs]
    w = _wait_all(st, names, logs, logs / "wait_run2.json", timeout=120)
    [p.wait(timeout=60) for p in procs]
    assert w.returncode == 1
    assert KQ.read_int(st / "eval_p1_s25.exit") == 4
    assert KQ.read_int(st / "res_fp16.exit") == 0 and KQ.read_int(st / "res_p1_s25.exit") == 0
    assert "res_p1_s25 ran" in (logs / "res_p1_s25.log").read_text()
    probs = K.check(jobs, st)
    assert "eval_p1_s25: exit 4, done False" in probs
    zp = tmp_path / "z.zip"
    assert K.main(["zip", "--out", str(zp), "--jobs-dir", str(jd), "--status", str(st)]) == 0
    report = tmp_path / "check.json"
    assert K.main(["check", "--jobs-dir", str(jd), "--status", str(st), "--out", str(report)]) == 1
    assert K.main(["final", "--jobs-dir", str(jd), "--status", str(st), "--check-report", str(report),
                   "--zip", str(zp)]) == 1
