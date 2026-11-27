"""Task 2 notebook 3 helper: job lists, the projection rule, the failure path after the queue starts, and an
end-to-end CPU run of every stage (preflight, preflight check, plan, main queue, wait-all, summary, check, zip,
final) through the real entry points with a tiny random Qwen2 and real retrieval prompts."""
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest
import torch

from common.paths import REPO_ROOT
from task2 import kaggle_nb1 as N1
from task2 import kaggle_nb2 as N2
from task2 import kaggle_nb3 as N3
from task2 import kv_run as KR
from tests.test_t2_kaggle_nb1 import FAKE, _queue_cmd, _wait

PY = sys.executable
PF_PROMPT = "KV-4096-middle-8-00"
PROMPTS = "KV-2048-end-0-00,KV-4096-middle-8-00"
ENV = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")


def _nb(*args, timeout=1200):
    return subprocess.run([PY, "-m", "task2.kaggle_nb3", *map(str, args)], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=timeout, env=ENV)


# ---------------------------------------------------------------------------------------------------------------
# Job lists

def test_main_and_preflight_jobs():
    (m,) = N3.stage_jobs("main")
    assert (m["name"], m["queue"], m["gpu"], m["out"], m["expect"]) == \
        ("kv_diag", "gpu0", "0", "results/task2/part_c/diag", {})
    inner = m["cmd"][m["cmd"].index("--") + 1:]
    assert inner[1:5] == ["-m", "task2.kv_diag", "--out", "results/task2/part_c/diag"]
    assert not any(a.startswith("--limit") or a == "--model-path" for a in m["cmd"])
    assert m["cmd"][1:4] == ["-m", "task2.kaggle_nb3", "job"]
    (p,) = N3.stage_jobs("preflight")
    longest = N2.longest_case_ids()[4096]
    assert p["name"] == "pf_kv_diag" and p["expect"] == {"limit_prompts": longest}
    assert p["out"] == "/kaggle/tmp/preflight/kv_diag" and p["cmd"][-1] == longest
    (t,) = N3.stage_jobs("preflight", test={"device": "cpu", "model_path": "/m", "limit_prompts": "2"})
    assert t["expect"] == {"limit_prompts": "2", "model_path": "/m"} and "--device" in t["cmd"]


def test_main_jobs_need_a_plan_within_limits(tmp_path):
    p = _nb("jobs", "--stage", "main", "--jobs-dir", tmp_path / "j")
    assert p.returncode != 0 and "need --plan" in p.stderr
    (tmp_path / "plan.json").write_text(json.dumps({"over": True}))
    q = _nb("jobs", "--stage", "main", "--jobs-dir", tmp_path / "j", "--plan", tmp_path / "plan.json")
    assert q.returncode != 0 and "over the limits" in q.stderr
    assert not (tmp_path / "j").exists()


# ---------------------------------------------------------------------------------------------------------------
# End to end on CPU

@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(KR.MODEL_ID, revision=KR.locked_revision(), local_files_only=True)
    torch.manual_seed(6304)
    d = tmp_path_factory.mktemp("tiny_qwen2_nb3")
    cfg = Qwen2Config(hidden_size=128, num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      head_dim=128, intermediate_size=256, vocab_size=151_936, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, tiny_dir):
    tmp = tmp_path_factory.mktemp("nb3")
    jobs_dir, logs_s, st_s, logs, st = tmp / "jobs", tmp / "logs_smoke", tmp / "status_smoke", tmp / "logs", \
        tmp / "status"
    out = {"tmp": tmp, "jobs": jobs_dir, "logs": logs, "status": st}
    opts = ["--device", "cpu", "--model-path", tiny_dir]
    p = _nb("jobs", "--stage", "preflight", "--jobs-dir", jobs_dir, "--results", tmp / "pf", "--status", st_s, *opts,
            "--limit-prompts", PF_PROMPT)
    assert p.returncode == 0, p.stdout + p.stderr
    q = subprocess.run(_queue_cmd(tmp, "pf_gpu0", 0, jobs_dir / "preflight_gpu0.json", logs_s, st_s), cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=1200, env=ENV)
    assert q.returncode == 0, q.stdout[-3000:]
    w = _wait(st_s, logs_s, ["pf_kv_diag"], timeout=1200)
    assert w.returncode == 0, w.stdout
    out["preflight"] = _nb("preflight-check", "--jobs-dir", jobs_dir)
    out["plan"] = _nb("plan", "--jobs-dir", jobs_dir, "--elapsed-s", "60", "--out", tmp / "plan.json")
    p = _nb("jobs", "--stage", "main", "--jobs-dir", jobs_dir, "--results", tmp / "res", "--status", st, *opts,
            "--limit-prompts", PROMPTS, "--plan", tmp / "plan.json")
    assert p.returncode == 0, p.stdout + p.stderr
    out["meta"] = N1.load_meta(jobs_dir, "main")
    proc = subprocess.Popen(_queue_cmd(tmp, "gpu0", 0, jobs_dir / "main_gpu0.json", logs, st), cwd=REPO_ROOT,
                            stdout=subprocess.DEVNULL, env=ENV)
    out["wait"] = _wait(st, logs, ["kv_diag"], mode="--wait-all", out=tmp / "wait.json", timeout=1200)
    proc.wait(60)
    out["summary"] = _nb("summary", "--jobs-dir", jobs_dir, "--status", st)
    out["check"] = _nb("check", "--jobs-dir", jobs_dir, "--status", st, "--logs", logs, "--out", tmp / "check.json")
    out["zip"] = _nb("zip", "--jobs-dir", jobs_dir, "--out", tmp / "nb3.zip", "--include", tmp / "res", logs, st,
                     jobs_dir)
    out["final"] = _nb("final", "--jobs-dir", jobs_dir, "--status", st, "--check-report", tmp / "check.json",
                       "--zip", tmp / "nb3.zip")
    return out


def test_e2e_preflight_and_plan(e2e):
    pf = e2e["preflight"]
    assert pf.returncode == 0, pf.stdout + pf.stderr
    assert "preflight-check: 1 jobs, 0 problems" in pf.stdout and f"    {PF_PROMPT} kv_int8_token " in pf.stdout
    assert " EM " not in pf.stdout and "accuracy" not in pf.stdout
    p = e2e["plan"]
    assert p.returncode == 0, p.stdout + p.stderr
    plan = json.loads((e2e["tmp"] / "plan.json").read_text())
    (r,) = plan["rows"]
    assert r["name"] == "kv_diag" and set(r["parts"]) == {"load", "prompts"}
    assert r["parts"]["prompts"] == pytest.approx(18 * plan["prompt_s"])
    assert plan["total_s"] == pytest.approx(N3.SAFETY * sum(r["parts"].values()))
    assert plan["over"] is False and "plan: within limits" in p.stdout
    q = _nb("plan", "--jobs-dir", e2e["jobs"], "--budget-h", "0.00001")
    assert q.returncode == 1 and "OVER:" in q.stdout
    v = _nb("plan", "--jobs-dir", e2e["jobs"], "--vram-gib", "-1")
    assert v.returncode == 1 and "above -1.0 GiB" in v.stdout


def test_e2e_main_outputs(e2e):
    assert e2e["wait"].returncode == 0, e2e["wait"].stdout
    d = e2e["tmp"] / "res" / "diag"
    assert sorted(os.listdir(d)) == ["control_rows.jsonl.gz", "npz", "prompt_rows.jsonl.gz", "run.json"]
    run = json.loads((d / "run.json").read_text())
    assert run["status"] == "ok" and run["config"]["limited"] is True
    assert run[N1.CONCURRENCY_KEY]["overlapping"] == []


def test_e2e_summary_check_zip_final(e2e):
    s = e2e["summary"]
    assert s.returncode == 0, s.stderr
    lines = s.stdout.rstrip("\n").split("\n")
    assert len(lines) <= 40
    assert lines[0].startswith("kv_diag: 2 prompts")
    for label in ("fp16 reference", "(a) K INT4 token-wise, V FP16", "(b) K FP16, V INT4 token-wise",
                  "(c) K and V INT8 token-wise", "(d) int4_token as evaluated", "(e) int4_channel_k as evaluated"):
        assert any(x.startswith("  " + label) and " TF KL " in x for x in lines), label
    for head in ("channel max/median ratio per layer", "K relative error per layer", "overall: K ratio",
                 "AM4.5 read-out (descriptive): ", "tokens equal to the committed eval rows: "):
        assert any(x.startswith(head) for x in lines), head
    assert any(x.startswith("  L0 ") and "|" in x for x in lines)
    c = e2e["check"]
    assert c.returncode == 0, c.stdout + c.stderr
    assert json.loads((e2e["tmp"] / "check.json").read_text()) == {"n_jobs": 1, "problems": [], "failed_jobs": []}
    assert "==== CHECK: PASS: 1 jobs, 0 problems ====" in c.stdout
    z = e2e["zip"]
    assert z.returncode == 0, z.stdout + z.stderr
    with zipfile.ZipFile(e2e["tmp"] / "nb3.zip") as zf:
        names = zf.namelist()
    assert any(n.endswith("diag/control_rows.jsonl.gz") for n in names)
    assert any(n.endswith("diag/npz/KV-4096-middle-8-00.npz") for n in names)
    assert any(n.endswith("logs/kv_diag.log") for n in names)
    assert not any(w in n.split("/nb3")[-1] for n in names for w in ("smoke", "preflight", "cache"))
    f = e2e["final"]
    assert f.returncode == 0 and "notebook complete" in f.stdout, f.stdout


def test_e2e_check_finds_problems(e2e, tmp_path):
    (j,) = e2e["meta"]
    dst = tmp_path / "diag"
    shutil.copytree(j["out"], dst)
    (dst / "npz" / "KV-2048-end-0-00.npz").unlink()
    run = json.loads((dst / "run.json").read_text())
    run["summary"]["per_config"]["fp16"]["n"] = 1
    run["config"]["limited"] = False
    (dst / "run.json").write_text(json.dumps(run))
    text = "\n".join(N3.check([dict(j, out=str(dst))], e2e["status"], e2e["logs"]))
    assert "npz files 1 != 2" in text and "summary per-config counts" in text and "config.limited False" in text


def test_reproduction_counts(tmp_path):
    from common.io import write_jsonl_gz

    write_jsonl_gz(tmp_path / "e.jsonl.gz", [{"id": "a", "method": "int4_token", "generated_token_ids": [1, 2]},
                                             {"id": "a", "method": "fp16", "generated_token_ids": [3]}])
    rows = [{"id": "a", "config": "int4_token", "generated_token_ids": [1, 2]},
            {"id": "a", "config": "fp16", "generated_token_ids": [4]}]
    assert N3.reproduction(rows, tmp_path / "e.jsonl.gz") == {"fp16": (0, 1), "int4_token": (1, 1),
                                                              "int4_channel_k": (0, 0)}
    assert N3.reproduction(rows, tmp_path / "missing.jsonl.gz") == {}


# ---------------------------------------------------------------------------------------------------------------
# Failure path after the queue starts

def test_failed_job_does_not_raise(tmp_path):
    """A failed kv_diag job: wait-all waits and reports it, summary prints its status, check prints FAIL lines and
    final reports not complete; no command raises."""
    st, logs, jd = tmp_path / "status", tmp_path / "logs", tmp_path / "jobs"
    jd.mkdir()
    out = tmp_path / "res" / "diag"
    cmd = [PY, "-m", "task2.kaggle_nb3", "job", "--name", "kv_diag", "--status", str(st), "--out", str(out), "--",
           PY, "-c", FAKE.format(sleep=0.0, code=3), str(out)]
    meta = [{"name": "kv_diag", "stage": "main", "gpu": "0", "queue": "gpu0", "phase": "main", "kind": "diag",
             "out": str(out), "cmd": cmd, "wait_for": [], "run": {}, "expect": {}}]
    N1.write_jobs("main", jd, meta)
    q = subprocess.Popen(_queue_cmd(tmp_path, "gpu0", 0, jd / "main_gpu0.json", logs, st), cwd=REPO_ROOT,
                         stdout=subprocess.DEVNULL, start_new_session=True)
    w = _wait(st, logs, ["kv_diag"], mode="--wait-all", out=tmp_path / "wait.json", timeout=120)
    assert w.returncode == 1 and "--- kv_diag: failed, exit 3" in w.stdout
    assert q.wait(60) == 1
    s = _nb("summary", "--jobs-dir", jd, "--status", st)
    assert s.returncode == 0 and s.stderr == "" and "kv_diag: summary unavailable" in s.stdout
    c = _nb("check", "--jobs-dir", jd, "--status", st, "--logs", logs, "--out", tmp_path / "check.json")
    assert c.returncode == 1 and c.stderr == "" and "FAIL: kv_diag: exit 3, done False" in c.stdout
    z = _nb("zip", "--jobs-dir", jd, "--out", tmp_path / "z.zip", "--include", tmp_path / "res", logs, st)
    assert z.returncode == 0 and (tmp_path / "z.zip").is_file()
    f = _nb("final", "--jobs-dir", jd, "--status", st, "--check-report", tmp_path / "check.json",
            "--zip", tmp_path / "z.zip")
    assert f.returncode == 1 and f.stderr == "" and "NOT OK: job kv_diag: exit 3" in f.stdout
