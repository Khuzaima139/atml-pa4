"""Task 3 notebook 1 helper: the smoke stage through the real queue runner on CPU with tiny random models, the smoke
check, the plan from the smoke JSONs, the main job list built from the plan, and the failure path (success, nonzero
exit, kill) through wait-all, check, summary, manifest, zip and final."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from common import kaggle_queue as Q
from common.paths import REPO_ROOT
from task3 import data as D
from task3 import kaggle_nb1 as K
from task3 import preflight as P
from task3.sampling import VOCAB_SIZE


def _save_tiny(d, seed, layers, hidden, tok):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=2 * hidden, vocab_size=VOCAB_SIZE, tie_word_embeddings=True,
                      max_position_embeddings=4096, initializer_range=0.2)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def _run_queues(jobs_dir, stage, status, logs, names, timeout=1800, out=None):
    procs = []
    for q in ("gpu0", "gpu1"):
        f = Path(jobs_dir) / f"{stage}_{q}.json"
        if f.is_file():
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "common.kaggle_queue", "--name", f"{stage}_{q}", "--gpu", q[-1], "--jobs",
                 str(f), "--logs", str(logs), "--status", str(status), "--poll", "1"],
                cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
    rc = Q.wait_all(status, names, logs, poll=1, report=600, timeout=timeout, out=out)
    for p in procs:
        p.wait(timeout=60)
    return rc


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    rv = D.revisions()["model_target"]
    tok = AutoTokenizer.from_pretrained(rv["repo"], revision=rv["revision"], local_files_only=True)
    root = tmp_path_factory.mktemp("t3_nb1")
    e = {"root": root, "target": _save_tiny(root / "target", 6304, 2, 64, tok),
         "draft": _save_tiny(root / "draft", 6305, 1, 32, tok), "jobs": root / "jobs"}
    pf = root / "preflight.json"
    assert P.main(["--out", str(pf), "--target-path", str(e["target"]), "--draft-path", str(e["draft"]),
                   "--device", "cpu", "--cap", "12"]) == 0
    e["preflight"] = pf
    return e


def _test_args(env):
    return ["--device", "cpu", "--dtype", "float32", "--target-path", str(env["target"]), "--draft-path",
            str(env["draft"]), "--preflight", str(env["preflight"])]


@pytest.fixture(scope="module")
def smoke(env):
    r = env["root"]
    st, lg = r / "status_smoke", r / "logs_smoke"
    assert K.main(["jobs", "--stage", "smoke", "--jobs-dir", str(env["jobs"]), "--results", str(r / "smoke"),
                   "--status", str(st), "--artifacts", str(r / "smoke" / "artifacts"), *_test_args(env)]) == 0
    meta = K.load_meta(env["jobs"], "smoke")
    rc = _run_queues(env["jobs"], "smoke", st, lg, [j["name"] for j in meta])
    return {"meta": meta, "rc": rc, "status": st, "logs": lg}


@pytest.mark.parametrize("break_dtype", ["float32", "float16"])
def test_jobs_refuse_failing_preflight(env, break_dtype, tmp_path):
    """Every stage needs a preflight whose FP32 and FP16 rules both pass (protocol AM3)."""
    d = json.loads(Path(env["preflight"]).read_text())
    assert d["all_pass"] is True
    s = d["precisions"][break_dtype]["summary"]["target"]["split_prefill"]
    s["max_tv"] = 0.5
    verdict = P.evaluate_record(d["precisions"])
    assert verdict["all_pass"] is False and verdict["precision_pass"][break_dtype] is False
    d.update(verdict)
    bad = tmp_path / "preflight_bad.json"
    bad.write_text(json.dumps(d))
    jobs = tmp_path / "jobs"
    for stage in ("smoke", "main"):
        args = ["jobs", "--stage", stage, "--jobs-dir", str(jobs), "--results", str(tmp_path / "r"),
                "--status", str(tmp_path / "st"), "--artifacts", str(tmp_path / "a"), "--device", "cpu",
                "--preflight", str(bad)]
        if stage == "main":
            args += ["--plan", str(tmp_path / "no_plan.json")]
        with pytest.raises(SystemExit, match="preflight"):
            K.main(args)
    assert not jobs.exists()


def test_smoke_jobs_and_check(env, smoke):
    names = [j["name"] for j in smoke["meta"]]
    assert set(names) == {"smoke_collect", "smoke_train_off_policy_6304", "smoke_train_on_policy_6304",
                          "smoke_merge_on_policy_6304", "smoke_bank_own_on_policy_6304", "smoke_bank_teacher",
                          "smoke_bank_original_draft", "smoke_bank_exact_spec_original_gamma4"}
    assert smoke["rc"] == 0
    off = next(j for j in smoke["meta"] if j["name"] == "smoke_train_off_policy_6304")
    assert off["wait_for"] == [str(smoke["status"] / "smoke_collect.done")]
    assert "--limit-updates" in off["cmd"] and "--collection" in off["cmd"]
    rc = K.main(["smoke-check", "--jobs-dir", str(env["jobs"]), "--status", str(smoke["status"]), "--logs",
                 str(smoke["logs"])])
    assert rc == 0
    tr = json.loads(Path(next(j for j in smoke["meta"] if j["kind"] == "train")["run_json"]).read_text())
    assert [u["positions"] for u in tr["updates"]] == [256, 256]


@pytest.fixture(scope="module")
def plan(env, smoke):
    out = env["root"] / "plan.json"
    rc = K.main(["plan", "--jobs-dir", str(env["jobs"]), "--elapsed-s", "0", "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text())


def test_plan(plan, capsys):
    assert set(plan["assignment"]) == set(K.UNITS) and plan["assignment"]["collect"] == "gpu0"
    assert plan["orders"]["gpu0"][0] == "collect" and not plan["over"]
    assert set(plan["job_seconds"]) == {"collect", *[f"{p}_{r}" for r in K.RUNS for p in ("train", "merge",
                                                                                         "bank_own")],
                                        *[f"bank_{k}" for k in K.BANK_KINDS]}
    assert all(v >= K.JOB_OVERHEAD_S for v in plan["job_seconds"].values())
    assert plan["makespan_s"] == max(plan["queue_s"].values())
    for r in K.RUNS:
        if r.startswith("off_"):
            assert plan["unit_end_s"][r] >= plan["unit_end_s"]["collect"] + plan["unit_seconds"][r] - 1e-6
    K.print_plan(plan)
    text = capsys.readouterr().out
    assert "plan: within limits" in text
    print("synthetic CPU plan (placeholder): " + ", ".join(f"{q} {h:.3f} h" for q, h in plan["queue_h"].items()))


def test_plan_over_limit(env, smoke):
    p = K.project(env["jobs"], budget_h=1e-6)
    assert p["over"] and any("above" in x for x in p["reasons"])
    out = env["root"] / "plan_over.json"
    K._write_new(out, p)
    with pytest.raises(SystemExit):
        K.main(["jobs", "--stage", "main", "--jobs-dir", str(env["root"] / "jobs_over"), "--plan", str(out)])


def test_balance_prefers_short_makespan():
    us = {u: 100.0 for u in K.UNITS}
    us["collect"] = 1000.0
    bal = K.balance(us)
    assert bal["assignment"]["collect"] == "gpu0"
    # both off-policy units wait for the collection: best is one off-policy unit after it on each queue
    assert bal["makespan_s"] == 1100.0
    assert sorted(bal["assignment"][u] for u in ("off_policy_6304", "off_policy_6305")) == ["gpu0", "gpu1"]
    us2 = {u: 100.0 for u in K.UNITS}
    assert K.balance(us2)["makespan_s"] <= (sum(us2.values()) + K.STAGGER_S) / 2 + 100


@pytest.fixture(scope="module")
def main_jobs(env, plan):
    r = env["root"]
    jd = r / "jobs_main"
    pf = r / "plan.json"
    assert K.main(["jobs", "--stage", "main", "--jobs-dir", str(jd), "--plan", str(pf), "--results",
                   str(r / "results"), "--status", str(r / "status"), "--artifacts", str(r / "art"),
                   *_test_args(env)]) == 0
    return {"dir": jd, "meta": K.load_meta(jd, "main")}


def test_main_job_list(env, plan, main_jobs):
    meta = main_jobs["meta"]
    names = [j["name"] for j in meta]
    want = {"collect", *[f"bank_{k}" for k in K.BANK_KINDS]}
    for r in K.RUNS:
        want |= {f"train_{r}", f"merge_{r}", f"bank_own_{r}"}
    assert set(names) == want and len(names) == len(want)
    by = {j["name"]: j for j in meta}
    assert by["collect"]["queue"] == "gpu0"
    assert [j["name"] for j in meta if j["queue"] == "gpu0"][0] == "collect"
    st = env["root"] / "status"
    for r in K.RUNS:
        assert by[f"merge_{r}"]["wait_for"] == [str(st / f"train_{r}.done")]
        assert by[f"bank_own_{r}"]["wait_for"] == [str(st / f"merge_{r}.done")]
        assert by[f"train_{r}"]["queue"] == by[f"merge_{r}"]["queue"] == by[f"bank_own_{r}"]["queue"]
        assert by[f"train_{r}"]["queue"] == plan["assignment"][r]
        if r.startswith("off_"):
            assert by[f"train_{r}"]["wait_for"] == [str(st / "collect.done")]
    for j in meta:
        assert j["expect"] == {} and not any(str(c).startswith("--limit") for c in j["cmd"])
    spec = by["bank_exact_spec_original_gamma4"]["cmd"]
    assert spec[spec.index("--preflight") + 1] == str(env["preflight"])
    for q in ("gpu0", "gpu1"):
        rows = json.loads((main_jobs["dir"] / f"main_{q}.json").read_text())
        assert Q.load_jobs(main_jobs["dir"] / f"main_{q}.json")
        assert [x["name"] for x in rows] == [j["name"] for j in meta if j["queue"] == q]


def test_failure_path(env, plan, main_jobs, capsys):
    """Success, nonzero exit and kill: wait-all waits for every job, later cells run and report, nothing raises."""
    r = env["root"]
    jd = r / "jobs_fail"
    st, lg = r / "status_fail", r / "logs_fail"
    meta = K.stage_jobs("main", r / "fres", st, r / "fart", env["preflight"], plan,
                        {"device": "cpu"})
    py = sys.executable
    for j in meta:
        if j["name"] == "train_on_policy_6304":
            j["cmd"] = [py, "-c", "import sys; print('failing'); sys.exit(3)"]
        elif j["name"] == "bank_teacher":
            j["cmd"] = [py, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]
        else:
            j["cmd"] = [py, "-c", "print('ok')"]
    K.write_jobs("main", jd, meta)
    names = [j["name"] for j in meta]
    rc = _run_queues(jd, "main", st, lg, names, timeout=600, out=r / "wait_fail.json")
    assert rc == 1
    table = {x["name"]: x for x in json.loads((r / "wait_fail.json").read_text())["jobs"]}
    assert table["train_on_policy_6304"]["exit"] == 3 and table["bank_teacher"]["exit"] == -9
    assert table["merge_on_policy_6304"]["exit"] == Q.DEP_FAILED
    assert table["bank_own_on_policy_6304"]["exit"] == Q.DEP_FAILED
    assert table["collect"]["state"] == "ok" and table["train_off_policy_6304"]["state"] == "ok"
    common = ["--jobs-dir", str(jd), "--status", str(st), "--logs", str(lg)]
    assert K.main(["summary", *common]) == 0
    assert K.main(["timing", *common, "--plan", str(r / "plan.json")]) == 0
    assert K.main(["check", *common, "--out", str(r / "check_fail.json")]) == 1
    rep = json.loads((r / "check_fail.json").read_text())
    assert set(rep["failed_jobs"]) == set(names)
    assert any("train_on_policy_6304: exit 3" in p for p in rep["problems"])
    assert any("bank_teacher: exit -9" in p for p in rep["problems"])
    assert K.main(["manifest", *common, "--out", str(r / "manifest_fail.json"), "--artifacts",
                   str(r / "fart")]) == 1
    z = r / "fail.zip"
    assert K.main(["zip", *common, "--out", str(z), "--include", str(lg), str(st), str(jd),
                   str(r / "fres")]) == 0
    with zipfile.ZipFile(z) as zf:
        got = zf.namelist()
    assert any(n.endswith("bank_teacher.log") for n in got) and not any("smoke" in n for n in got)
    assert K.main(["final", *common, "--check-report", str(r / "check_fail.json"), "--zip", str(z)]) == 1
    out = capsys.readouterr().out
    assert "==== CHECK: FAIL" in out and "notebook not complete" in out


def test_zip_excludes_smoke(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("{}")
    (tmp_path / "a" / "smoke_x.json").write_text("{}")
    res = K.make_zip(tmp_path / "o.zip", [tmp_path / "a", tmp_path / "missing"])
    assert res["n_files"] == 1 and res["missing"] == [str(tmp_path / "missing")]


def _regzip(path, mtime):
    import gzip

    with gzip.open(path, "rb") as fh:
        data = fh.read()
    path.unlink()
    with open(path, "xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime, compresslevel=6) as gz:
            gz.write(data)


@pytest.mark.skipif(not D.GSM8K_TEST_LOCKED.is_file(), reason="course data repository not prepared")
def test_data_check_passes_after_regzip_with_new_mtime(tmp_path, monkeypatch):
    """Data preparation regenerates the locked gzip files with a new header; the data check compares content."""
    import shutil

    from common.io import sha256_file
    from task3 import heldout as H

    root = tmp_path / "datasets"
    names = {"GSM8K_TEST_LOCKED": D.GSM8K_TEST_LOCKED, "GSM8K_TRAIN_LOCKED": D.GSM8K_TRAIN_LOCKED,
             "GSM8K_TEST_IDS": D.GSM8K_TEST_IDS, "SHORT_TEST": D.SHORT_TEST, "SHORT_TRAIN": D.SHORT_TRAIN,
             "RELEASE_SUMS": D.RELEASE_SUMS}
    for attr, src in names.items():
        dst = root / Path(src).relative_to(D.DATA_REPO_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        monkeypatch.setattr(D, attr, dst)
    monkeypatch.setattr(D, "DATA_REPO_DIR", root)
    for attr in ("GSM8K_TEST_LOCKED", "GSM8K_TRAIN_LOCKED"):
        before = sha256_file(getattr(D, attr))
        _regzip(getattr(D, attr), mtime=1_700_000_000)
        assert sha256_file(getattr(D, attr)) != before
    res = K.data_check()
    assert res["heldout_list_equals_rebuild"] is True
    assert res["training_items"] == {"math": 500, "short": 300}
    gz = [f for f in H.read_heldout_list()["files"] if f["name"].endswith(".gz")]
    assert gz and all("content_sha256" in f and "sha256" not in f for f in gz)
