"""Queue runner and wait helper of common.kaggle_queue: success, nonzero exit, killed job, dependency failure."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from common import kaggle_queue as KQ
from common.paths import REPO_ROOT

PY = sys.executable


def _py(code):
    return [PY, "-c", code]


def _jobs(path, jobs):
    path.write_text(json.dumps(jobs))
    return path


def _queue(tmp, name, gpu, jobs, background=False):
    cmd = [PY, "-m", "common.kaggle_queue", "--name", name, "--gpu", str(gpu), "--jobs",
           str(_jobs(tmp / f"{name}.json", jobs)), "--logs", str(tmp / "logs"), "--status", str(tmp / "status"),
           "--poll", "0.1"]
    if background:
        return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)


def _wait(tmp, names, timeout=60):
    return subprocess.run([PY, "-m", "common.kaggle_queue", "--wait", str(tmp / "status"), "--names", *names,
                           "--logs", str(tmp / "logs"), "--poll", "0.1", "--report", "1", "--timeout", str(timeout)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout + 30)


def _st(tmp, name, ext):
    return tmp / "status" / f"{name}.{ext}"


def test_success(tmp_path):
    p = _queue(tmp_path, "q0", 1, [
        {"name": "a", "cmd": _py("import os; print('gpu', os.environ['CUDA_VISIBLE_DEVICES'], "
                                 "os.environ['PYTHONUNBUFFERED'])"), "wait_for": []},
        {"name": "b", "cmd": "echo from-shell | tr a-z A-Z"},
    ])
    assert p.returncode == 0, p.stdout + p.stderr
    for n in ("a", "b"):
        assert KQ.read_int(_st(tmp_path, n, "exit")) == 0 and _st(tmp_path, n, "done").exists()
        assert KQ.read_int(_st(tmp_path, n, "pid")) > 0
        rec = json.loads(_st(tmp_path, n, "json").read_text())
        assert rec["exit"] == 0 and rec["gpu"] == "1" and rec["wall_s"] >= 0
    assert "gpu 1 1" in (tmp_path / "logs" / "a.log").read_text()
    assert "FROM-SHELL" in (tmp_path / "logs" / "b.log").read_text()
    assert "gpu 1 1" in p.stdout  # tee to stdout
    assert json.loads(_st(tmp_path, "q0", "queue.exit").read_text()) == {"a": 0, "b": 0}
    w = _wait(tmp_path, ["a", "b"])
    assert w.returncode == 0 and "all 2 jobs done" in w.stdout


def test_nonzero_exit_does_not_stop_queue(tmp_path):
    p = _queue(tmp_path, "q0", 0, [
        {"name": "bad", "cmd": _py("import sys; print('line-one'); print('boom'); sys.exit(3)")},
        {"name": "good", "cmd": _py("print('ran after failure')")},
        {"name": "bad_shell", "cmd": "false | cat"},
    ])
    assert p.returncode == 1
    assert KQ.read_int(_st(tmp_path, "bad", "exit")) == 3 and not _st(tmp_path, "bad", "done").exists()
    assert KQ.read_int(_st(tmp_path, "good", "exit")) == 0 and _st(tmp_path, "good", "done").exists()
    assert KQ.read_int(_st(tmp_path, "bad_shell", "exit")) == 1  # pipefail
    t0 = time.monotonic()
    w = _wait(tmp_path, ["good", "bad"])
    assert w.returncode == 1 and time.monotonic() - t0 < 20
    assert "job bad failed with exit code 3" in w.stdout and "boom" in w.stdout and "line-one" in w.stdout


def test_killed_job(tmp_path):
    q = _queue(tmp_path, "q0", 0, [
        {"name": "sleeper", "cmd": _py("import time; print('sleeping', flush=True); time.sleep(120)")},
        {"name": "next", "cmd": _py("print('next ran')")},
    ], background=True)
    pid_file = _st(tmp_path, "sleeper", "pid")
    for _ in range(300):
        if pid_file.exists():
            break
        time.sleep(0.05)
    os.kill(KQ.read_int(pid_file), signal.SIGKILL)
    out, _ = q.communicate(timeout=60)
    assert q.returncode == 1
    assert KQ.read_int(_st(tmp_path, "sleeper", "exit")) == -signal.SIGKILL
    assert KQ.read_int(_st(tmp_path, "next", "exit")) == 0
    w = _wait(tmp_path, ["sleeper", "next"])
    assert w.returncode == 1 and "exit code -9" in w.stdout


def test_dependency_failure(tmp_path):
    p = _queue(tmp_path, "q0", 0, [
        {"name": "dep", "cmd": _py("import sys; sys.exit(5)")},
        {"name": "user", "cmd": _py("print('must not run')"), "wait_for": [str(_st(tmp_path, "dep", "done"))]},
        {"name": "after", "cmd": _py("print('after ran')")},
    ])
    assert p.returncode == 1
    assert KQ.read_int(_st(tmp_path, "user", "exit")) == KQ.DEP_FAILED
    assert not _st(tmp_path, "user", "pid").exists()
    log = (tmp_path / "logs" / "user.log").read_text()
    assert "dependency failed" in log and "dependency exit code 5" in log and "must not run" not in log
    assert KQ.read_int(_st(tmp_path, "after", "exit")) == 0
    w = _wait(tmp_path, ["user"])
    assert w.returncode == 1 and f"exit code {KQ.DEP_FAILED}" in w.stdout


def test_dependency_across_queues(tmp_path):
    marker = tmp_path / "order.txt"
    q1 = _queue(tmp_path, "q1", 1, [
        {"name": "late_user", "cmd": _py(f"open({str(marker)!r}, 'a').write('user\\n')"),
         "wait_for": [str(_st(tmp_path, "slow_dep", "done"))]},
    ], background=True)
    q0 = _queue(tmp_path, "q0", 0, [
        {"name": "slow_dep", "cmd": _py(f"import time; time.sleep(1); open({str(marker)!r}, 'a').write('dep\\n')")},
    ], background=True)
    q0.communicate(timeout=60)
    q1.communicate(timeout=60)
    assert marker.read_text() == "dep\nuser\n"
    assert _wait(tmp_path, ["slow_dep", "late_user"]).returncode == 0


def test_wait_detects_dead_process_and_dead_queue(tmp_path):
    st = tmp_path / "status"
    st.mkdir()
    dead = subprocess.Popen(_py("pass"))
    dead.wait()
    (st / "ghost.pid").write_text(f"{dead.pid}\n")
    w = _wait(tmp_path, ["ghost"])
    assert w.returncode == 1 and "ended without an exit file" in w.stdout
    (st / "ghost.pid").unlink()
    (st / "q9.queue.pid").write_text(f"{dead.pid}\n")
    w = _wait(tmp_path, ["never_started"])
    assert w.returncode == 1 and "no queue process alive" in w.stdout


def test_refuses_existing_log_and_bad_jobs(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.log").write_text("old")
    p = _queue(tmp_path, "q0", 0, [{"name": "a", "cmd": _py("print(1)")}])
    assert p.returncode == 1 and KQ.read_int(_st(tmp_path, "a", "exit")) == KQ.EXISTS
    assert (tmp_path / "logs" / "a.log").read_text() == "old"
    with pytest.raises(AssertionError, match="duplicate"):
        KQ.load_jobs(_jobs(tmp_path / "d.json", [{"name": "x", "cmd": "true"}, {"name": "x", "cmd": "true"}]))
    with pytest.raises(AssertionError, match="unknown job keys"):
        KQ.load_jobs(_jobs(tmp_path / "e.json", [{"name": "x", "cmd": "true", "gpu": 1}]))


def test_exit_written_within_grace_is_not_lost(tmp_path):
    import threading

    dead = subprocess.Popen(_py("pass"))
    dead.wait()
    (tmp_path / "j.pid").write_text(f"{dead.pid}\n")
    threading.Timer(0.5, lambda: (tmp_path / "j.exit").write_text("0\n")).start()
    assert KQ.ended_without_exit(tmp_path / "j") is False
    (tmp_path / "k.pid").write_text(f"{dead.pid}\n")
    assert KQ.ended_without_exit(tmp_path / "k") is True
