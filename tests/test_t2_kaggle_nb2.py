"""Task 2 notebook 2 helper: inventory, job lists of every stage, data and model-file records, the projection rule, the
failure path after the queues start, and an end-to-end CPU run of every stage (prebuild, preflight, smoke, plan,
phase 1 queues, wait-all, idle check, phase 2 queue, summary, check, zip, final) through the real entry points with a
tiny random Qwen2, real retrieval prompts and tiny training limits."""
import json
import os
import subprocess
import sys
import zipfile

import pytest
import torch

from common.paths import REPO_ROOT
from task2 import kaggle_nb1 as N1
from task2 import kaggle_nb2 as N2
from task2 import kv_run as KR
from tests.test_t2_kaggle_nb1 import FAKE, _queue_cmd, _wait

PY = sys.executable
PROMPTS = "KV-2048-end-8-00,KV-4096-middle-8-00"
ENV = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")


def _nb(*args, timeout=1200):
    return subprocess.run([PY, "-m", "task2.kaggle_nb2", *map(str, args)], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=timeout, env=ENV)


# ---------------------------------------------------------------------------------------------------------------
# Inventory and job lists

def test_inventory():
    assert N2.streams_needed() == [(16, r, s) for r in ("mixed", "near") for s in (6304, 6305, 6306)]
    assert [N2.run_name(s) for s in N2.SEEDS] == [f"local_dilated_curriculum_K16_s{s}" for s in (6304, 6305, 6306)]


def test_longest_case_ids():
    cases = KR.load_cases()
    ids = N2.longest_case_ids(cases)
    for lv in (2048, 4096):
        cs = [c for c in cases if int(c["target_tokens"]) == lv]
        assert ids[lv] in {c["id"] for c in cs}
        top = max(len(c["user_text"]) for c in cs)
        assert len(next(c for c in cs if c["id"] == ids[lv])["user_text"]) == top
        assert ids[lv] == min(c["id"] for c in cs if len(c["user_text"]) == top)


def test_main_jobs():
    jobs = N2.stage_jobs("main")
    by_q = {q: [j["name"] for j in jobs if j["queue"] == q] for q in N1.queues(jobs)}
    assert by_q == {"gpu0": [f"train_local_dilated_curriculum_K16_s{s}" for s in (6304, 6305, 6306)],
                    "gpu1": ["kv_validate"], "phase2_gpu0": ["kv_eval", "kv_bench"]}
    assert all(j["expect"] == {} for j in jobs)
    out = {j["name"]: j["out"] for j in jobs}
    assert out["kv_validate"] == "results/task2/part_c/validate" and out["kv_bench"] == "results/task2/part_c/bench"
    assert out["train_local_dilated_curriculum_K16_s6305"] == "results/task2/mqar/local_dilated_curriculum_K16_s6305"
    tr = jobs[0]["cmd"]
    inner = tr[tr.index("--") + 1:]
    assert inner[1:3] == ["-m", "task2.train_mqar"]
    assert inner[inner.index("--curriculum") + 1] == "near_to_mixed" and inner[inner.index("--mechanism") + 1] == \
        "local_dilated" and inner[inner.index("--K") + 1] == "16" and inner[inner.index("--regime") + 1] == "mixed"
    assert not any(a.startswith("--limit") or a == "--model-path" for j in jobs for a in j["cmd"])
    assert {j["phase"] for j in jobs if j["queue"] == "phase2_gpu0"} == {"phase2"}
    ev = next(j for j in jobs if j["name"] == "kv_eval")["cmd"]
    assert ev[ev.index("--") + 1:][1:5] == ["-m", "task2.kv_eval", "--mode", "eval"]


def test_preflight_and_smoke_jobs():
    ids = N2.longest_case_ids()
    pf = {j["name"]: j for j in N2.stage_jobs("preflight")}
    assert set(pf) == {"pf_train_curriculum", "pf_kv_eval", "pf_kv_bench", "pf_kv_validate"}
    assert [n for n, j in pf.items() if j["queue"] == "gpu1"] == ["pf_kv_validate"]
    assert pf["pf_train_curriculum"]["expect"] == {"limit_updates": 20, "limit_eval": 64}
    assert pf["pf_kv_eval"]["expect"] == {"limit_prompts": ids[4096]} == pf["pf_kv_validate"]["expect"]
    assert pf["pf_kv_bench"]["expect"] == {"limit_shapes": "4096:8", "limit_configs": "hybrid", "limit_reps": "1,2"}
    assert all(j["out"].startswith("/kaggle/tmp/preflight") for j in pf.values())
    sm = {j["name"]: j for j in N2.stage_jobs("smoke")}
    assert sm["sm_train_curriculum"]["expect"] == {"limit_updates": 100} and sm["sm_train_curriculum"]["gpu"] == "0"
    assert sm["sm_kv_eval"]["expect"] == {"limit_prompts": f"{ids[2048]},{ids[4096]}"} and sm["sm_kv_eval"]["gpu"] == "1"
    t = {"device": "cpu", "model_path": "/m", "limit_updates": 3, "limit_eval": 2, "limit_prompts": PROMPTS,
         "limit_reps": "0,1"}
    pt = {j["name"]: j for j in N2.stage_jobs("preflight", test=t)}
    assert pt["pf_kv_bench"]["expect"] == {"limit_shapes": "4096:8", "limit_configs": "hybrid", "limit_reps": "0,1",
                                           "limit_prompts": PROMPTS, "model_path": "/m"}
    assert pt["pf_train_curriculum"]["expect"] == {"limit_updates": 3, "limit_eval": 2}
    assert "--device" in pt["pf_kv_eval"]["cmd"] and "--model-path" in pt["pf_kv_eval"]["cmd"]


def test_data_check_and_model_files(tmp_path):
    rec = N2.data_check()
    assert rec["problems"] == [] and rec["retrieval"]["n_cases"] == 180
    assert rec["retrieval"]["sha256"] == rec["retrieval"]["manifest_sha256"]
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"abc")
    (d / "config.json").write_text("{}")
    p = _nb("model-files", "--model-path", d, "--out", tmp_path / "mf.json")
    assert p.returncode == 0, p.stdout + p.stderr
    mf = json.loads((tmp_path / "mf.json").read_text())
    assert [f["name"] for f in mf["files"]] == ["config.json", "model.safetensors"] and mf["problems"] == []
    assert mf["files"][1]["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (d / "model.safetensors").unlink()
    q = _nb("model-files", "--model-path", d, "--out", tmp_path / "mf2.json")
    assert q.returncode == 1 and "no safetensors file" in q.stdout


# ---------------------------------------------------------------------------------------------------------------
# End to end on CPU

@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(KR.MODEL_ID, revision=KR.locked_revision(), local_files_only=True)
    torch.manual_seed(6304)
    d = tmp_path_factory.mktemp("tiny_qwen2_nb2")
    cfg = Qwen2Config(hidden_size=128, num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      head_dim=128, intermediate_size=256, vocab_size=151_936, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def _test_opts(tiny_dir):
    return ["--device", "cpu", "--model-path", tiny_dir, "--limit-updates", "3", "--limit-eval", "2",
            "--limit-prompts", PROMPTS, "--limit-reps", "0,1"]


def _stage(tmp, stage, jobs_dir, logs, status, tiny_dir, extra=()):
    p = _nb("jobs", "--stage", stage, "--jobs-dir", jobs_dir, "--results", tmp / "res" / stage, "--status", status,
            "--cache-dir", tmp / "cache", *_test_opts(tiny_dir), *extra)
    assert p.returncode == 0, p.stdout + p.stderr
    meta = N1.load_meta(jobs_dir, stage)
    procs = {}
    for q in N1.queues(meta):
        if q.startswith("phase2"):
            continue
        procs[q] = subprocess.Popen(_queue_cmd(tmp, f"{stage}_{q}", q[-1], jobs_dir / f"{stage}_{q}.json", logs,
                                               status), cwd=REPO_ROOT, stdout=subprocess.DEVNULL, env=ENV)
    return meta, procs


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, tiny_dir):
    tmp = tmp_path_factory.mktemp("nb2")
    jobs_dir, logs_s, st_s = tmp / "jobs", tmp / "logs_smoke", tmp / "status_smoke"
    logs, st = tmp / "logs", tmp / "status"
    out = {"tmp": tmp, "jobs": jobs_dir, "logs": logs, "status": st}
    pb = _nb("prebuild", "--cache-dir", tmp / "cache", "--workers", "2", "--limit-updates", "3",
             "--out", tmp / "prebuild.json")
    assert pb.returncode == 0, pb.stdout + pb.stderr
    out["prebuild"] = pb
    for stage in ("preflight", "smoke"):
        meta, procs = _stage(tmp, stage, jobs_dir, logs_s, st_s, tiny_dir)
        w = _wait(st_s, logs_s, [j["name"] for j in meta], timeout=1200)
        assert w.returncode == 0, w.stdout
        for p in procs.values():
            assert p.wait(60) == 0
        out[stage] = _nb("preflight-check" if stage == "preflight" else "smoke-report", "--jobs-dir", jobs_dir)
    out["plan"] = _nb("plan", "--jobs-dir", jobs_dir, "--prebuild", tmp / "prebuild.json", "--elapsed-s", "60",
                      "--out", tmp / "plan.json")
    meta, procs = _stage(tmp, "main", jobs_dir, logs, st, tiny_dir,
                         extra=("--seeds", "6304", "--plan", tmp / "plan.json"))
    out["meta"] = meta
    p1 = [j["name"] for j in meta if j["phase"] == "phase1"]
    out["wait"] = _wait(st, logs, p1, mode="--wait-all", out=tmp / "wait.json", timeout=1200)
    for p in procs.values():
        p.wait(60)
    out["idle"] = _nb("idle-check", "--jobs-dir", jobs_dir, "--status", st)
    out["phase2"] = subprocess.run(_queue_cmd(tmp, "phase2_gpu0", 0, jobs_dir / "main_phase2_gpu0.json", logs, st),
                                   cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200, env=ENV)
    out["summary"] = _nb("summary", "--jobs-dir", jobs_dir, "--status", st)
    out["check"] = _nb("check", "--jobs-dir", jobs_dir, "--status", st, "--logs", logs, "--out", tmp / "check.json")
    out["zip"] = _nb("zip", "--jobs-dir", jobs_dir, "--out", tmp / "nb2.zip", "--include", tmp / "res" / "main",
                     logs, st, jobs_dir)
    out["final"] = _nb("final", "--jobs-dir", jobs_dir, "--status", st, "--check-report", tmp / "check.json",
                       "--zip", tmp / "nb2.zip")
    return out


def test_e2e_prebuild(e2e):
    rec = json.loads((e2e["tmp"] / "prebuild.json").read_text())
    assert rec["n_failed"] == 0 and len(rec["streams"]) == 6
    assert {(r["K"], r["regime"], r["seed"]) for r in rec["streams"]} == set(N2.streams_needed())
    assert all(r["stop"] - r["start"] == 96 and r["validation"]["sequences_checked"] == 96 for r in rec["streams"])


def test_e2e_preflight_and_smoke(e2e):
    pf, sm = e2e["preflight"], e2e["smoke"]
    assert pf.returncode == 0, pf.stdout + pf.stderr
    assert "preflight-check: 4 jobs, 0 problems" in pf.stdout
    assert "bench L 4096 B 8 hybrid" in pf.stdout and "bench L 4096 B 8 fp16" in pf.stdout
    assert "    validate: " in pf.stdout and "/2 identical" in pf.stdout
    assert sm.returncode == 0, sm.stdout + sm.stderr
    assert "smoke-report: 2 jobs, 0 problems" in sm.stdout and "sm_train_curriculum: 3 updates" in sm.stdout


def test_e2e_plan(e2e):
    p = e2e["plan"]
    assert p.returncode == 0, p.stdout + p.stderr
    plan = json.loads((e2e["tmp"] / "plan.json").read_text())
    names = [r["name"] for r in plan["rows"]]
    assert names == [f"train_local_dilated_curriculum_K16_s{s}" for s in (6304, 6305, 6306)] + \
        ["kv_validate", "kv_eval", "kv_bench"]
    for r in plan["rows"]:
        assert r["projected_s"] == pytest.approx(N2.SAFETY * sum(r["parts"].values()))
    tr = plan["rows"][0]
    assert set(tr["parts"]) == {"train", "dev_eval", "test_eval", "fixed", "streams_check_and_validation"}
    rows = {r["name"]: r for r in plan["rows"]}
    assert plan["queue_s"]["gpu0"] == pytest.approx(sum(rows[n]["projected_s"] for n in names[:3]))
    assert plan["queue_s"]["gpu1"] == pytest.approx(60 + rows["kv_validate"]["projected_s"])
    assert plan["phase1_s"] == pytest.approx(max(plan["queue_s"].values()))
    assert plan["phase2_s"] == pytest.approx(rows["kv_eval"]["projected_s"] + rows["kv_bench"]["projected_s"])
    assert plan["total_s"] == pytest.approx(plan["phase1_s"] + plan["phase2_s"])
    assert rows["kv_validate"]["parts"]["prompts"] == pytest.approx(180 * plan["validate_prompt_s"])
    assert set(plan["eval_prompt_s_by_level"]) == {"2048", "4096"}
    assert plan["over"] is False and "plan: within limits" in p.stdout
    q = _nb("plan", "--jobs-dir", e2e["jobs"], "--prebuild", e2e["tmp"] / "prebuild.json", "--budget-h", "0.00001")
    assert q.returncode == 1 and "OVER:" in q.stdout
    r = _nb("plan", "--jobs-dir", e2e["jobs"], "--prebuild", e2e["tmp"] / "prebuild.json", "--vram-gib", "-1")
    assert r.returncode == 1 and "above -1.0 GiB" in r.stdout


def test_e2e_main_outputs(e2e):
    assert e2e["wait"].returncode == 0, e2e["wait"].stdout
    assert e2e["idle"].returncode == 0, e2e["idle"].stdout + e2e["idle"].stderr
    assert e2e["phase2"].returncode == 0, e2e["phase2"].stdout[-3000:]
    res = e2e["tmp"] / "res" / "main"
    d = res / "mqar" / "local_dilated_curriculum_K16_s6304"
    assert sorted(os.listdir(d)) == sorted(["run.json", "train_log.jsonl", "dev_curve.jsonl", "dev_queries_final.npz",
                                            "test_queries.npz", "model_final.safetensors"])
    run = json.loads((d / "run.json").read_text())
    assert run["config"]["curriculum"]["name"] == "near_to_mixed" and run["config"]["limited"] is True
    assert set(os.listdir(res / "part_c" / "validate")) == {"run.json", "validate_rows.jsonl.gz",
                                                            "full_forward_rows.jsonl.gz"}
    ev = res / "part_c" / "eval"
    assert set(os.listdir(ev)) == {"run.json", "eval_rows.jsonl.gz", "prompt_rows.jsonl.gz", "npz", "qualitative"}
    assert sorted(os.listdir(ev / "npz")) == sorted(f"{p}.npz" for p in PROMPTS.split(","))
    assert set(os.listdir(res / "part_c" / "bench")) == {"run.json", "bench_rows.jsonl.gz"}
    for sub in ("part_c/eval", "part_c/bench"):
        conc = json.loads((res / sub / "run.json").read_text())[N1.CONCURRENCY_KEY]
        assert conc["overlapping"] == []
    conc = json.loads((res / "part_c" / "validate" / "run.json").read_text())[N1.CONCURRENCY_KEY]
    assert "train_local_dilated_curriculum_K16_s6304" in [o["name"] for o in conc["overlapping"]] or \
        conc["overlapping"] == []


def test_e2e_summary_check_zip_final(e2e):
    s = e2e["summary"]
    assert s.returncode == 0, s.stderr
    lines = s.stdout.rstrip("\n").split("\n")
    assert len(lines) <= 60
    assert lines[0].startswith("B3 local_dilated curriculum K16")
    assert any(x.startswith("  s6304 ") and " dev " in x and " | test " in x for x in lines)
    assert any(x.startswith("  comparator local_dilated_mixed_K16_s6304 ") for x in lines)
    assert any(x.startswith("validate: ") and "generations identical" in x for x in lines)
    for head in ("eval exact match: ", "eval fp16 by length x location: ", "eval mean evidence retained: ",
                 "eval realized byte fraction: ", "bench B8 L4096 "):
        assert any(x.startswith(head) for x in lines), head
    assert "decode ratio" in next(x for x in lines if x.startswith("bench B8 L4096 "))
    c = e2e["check"]
    assert c.returncode == 0, c.stdout + c.stderr
    assert json.loads((e2e["tmp"] / "check.json").read_text()) == {"n_jobs": 4, "problems": [], "failed_jobs": []}
    assert "==== CHECK: PASS: 4 jobs, 0 problems ====" in c.stdout
    z = e2e["zip"]
    assert z.returncode == 0, z.stdout + z.stderr
    with zipfile.ZipFile(e2e["tmp"] / "nb2.zip") as zf:
        names = zf.namelist()
    assert any(n.endswith("part_c/eval/eval_rows.jsonl.gz") for n in names)
    assert any(n.endswith("local_dilated_curriculum_K16_s6304/test_queries.npz") for n in names)
    assert any(n.endswith("logs/kv_bench.log") for n in names)
    assert not any(w in n.split("/nb2")[-1] for n in names for w in ("smoke", "preflight", "cache"))
    f = e2e["final"]
    assert f.returncode == 0 and "notebook complete" in f.stdout, f.stdout


def test_e2e_check_finds_problems(e2e, tmp_path):
    import shutil
    js = {j["name"]: j for j in e2e["meta"]}
    probs = []
    for name, tamper in (("train_local_dilated_curriculum_K16_s6304", "curriculum"), ("kv_eval", "rows")):
        dst = tmp_path / name
        shutil.copytree(js[name]["out"], dst)
        run = json.loads((dst / "run.json").read_text())
        if tamper == "curriculum":
            run["curriculum_consumed"]["n_near"] = 1
            run["config"]["curriculum"]["breaks"] = [0, 0]
        else:
            (dst / "npz" / "KV-2048-end-8-00.npz").unlink()
        (dst / "run.json").write_text(json.dumps(run))
        probs += N2.check([dict(js[name], out=str(dst))], e2e["status"], e2e["logs"])
    text = "\n".join(probs)
    assert "consumed 1/" in text and "curriculum breaks [0, 0]" in text and "npz 1" in text


# ---------------------------------------------------------------------------------------------------------------
# Failure path after the queues start

def _wrapped(name, status, out, sleep=0.0, code=0):
    return [PY, "-m", "task2.kaggle_nb2", "job", "--name", name, "--status", str(status), "--out", str(out), "--",
            PY, "-c", FAKE.format(sleep=sleep, code=code), str(out)]


def test_phase1_failure_does_not_stop_phase2(tmp_path):
    """A failed phase 1 job: wait-all waits for every job and reports it, the idle check passes, phase 2 runs with no
    overlapping job, summary prints, check prints a FAIL list and final reports not complete; no command raises."""
    st, logs, jd = tmp_path / "status", tmp_path / "logs", tmp_path / "jobs"
    jd.mkdir()

    def spec(name, phase, kind, queue, sleep=0.0, code=0, run=None):
        out = tmp_path / "res" / name
        return {"name": name, "phase": phase, "kind": kind, "queue": queue, "gpu": queue[-1], "out": str(out),
                "run": run or {"mode": kind}, "expect": {}, "cmd": _wrapped(name, st, out, sleep=sleep, code=code),
                "wait_for": []}

    meta = [spec("train_x", "phase1", "train", "gpu0", code=4, run={"mechanism": "local_dilated", "regime": "mixed",
                                                                    "K": 16, "seed": 6304}),
            spec("kv_validate", "phase1", "validate", "gpu1", sleep=1.0),
            spec("kv_eval", "phase2", "eval", "phase2_gpu0"), spec("kv_bench", "phase2", "bench", "phase2_gpu0")]
    N1.write_jobs("main", jd, meta)
    q0 = subprocess.Popen(_queue_cmd(tmp_path, "gpu0", 0, jd / "main_gpu0.json", logs, st), cwd=REPO_ROOT,
                          stdout=subprocess.DEVNULL, start_new_session=True)
    q1 = subprocess.Popen(_queue_cmd(tmp_path, "gpu1", 1, jd / "main_gpu1.json", logs, st), cwd=REPO_ROOT,
                          stdout=subprocess.DEVNULL, start_new_session=True)
    w = _wait(st, logs, ["train_x", "kv_validate"], mode="--wait-all", out=tmp_path / "wait.json", timeout=120)
    assert w.returncode == 1 and "--- train_x: failed, exit 4" in w.stdout and "1 of 2 jobs ok" in w.stdout
    assert q0.wait(60) == 1 and q1.wait(60) == 0
    idle = _nb("idle-check", "--jobs-dir", jd, "--status", st)
    assert idle.returncode == 0, idle.stdout + idle.stderr
    p2 = subprocess.run(_queue_cmd(tmp_path, "phase2_gpu0", 0, jd / "main_phase2_gpu0.json", logs, st),
                        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert p2.returncode == 0
    for n in ("kv_eval", "kv_bench"):
        assert json.loads((tmp_path / "res" / n / "run.json").read_text())[N1.CONCURRENCY_KEY]["overlapping"] == []
    s = _nb("summary", "--jobs-dir", jd, "--status", st)
    assert s.returncode == 0 and s.stderr == ""
    c = _nb("check", "--jobs-dir", jd, "--status", st, "--logs", logs, "--out", tmp_path / "check.json")
    assert c.returncode == 1 and c.stderr == "" and "FAIL: train_x: exit 4, done False" in c.stdout
    z = _nb("zip", "--jobs-dir", jd, "--out", tmp_path / "z.zip", "--include", tmp_path / "res", logs, st)
    assert z.returncode == 0 and (tmp_path / "z.zip").is_file()
    f = _nb("final", "--jobs-dir", jd, "--status", st, "--check-report", tmp_path / "check.json",
            "--zip", tmp_path / "z.zip")
    assert f.returncode == 1 and f.stderr == "" and "NOT OK: job train_x: exit 4" in f.stdout


def test_jobs_refuse_over_plan(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"over": True}))
    p = _nb("jobs", "--stage", "main", "--jobs-dir", tmp_path / "j", "--plan", tmp_path / "plan.json")
    assert p.returncode != 0 and "over the limits" in p.stderr
    assert not (tmp_path / "j").exists()
