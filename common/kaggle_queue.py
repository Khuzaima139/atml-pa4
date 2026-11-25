"""Sequential job queue on one GPU with per-job logs and status files, and a wait helper.

Run a queue:  python -m common.kaggle_queue --name NAME --gpu N --jobs FILE [--logs DIR] [--status DIR] [--poll S]
Wait:         python -m common.kaggle_queue --wait DIR --names A B ... [--logs DIR] [--poll S] [--report S]
              [--timeout S]

FILE is a JSON list of jobs {"name": str, "cmd": [args] or str, "wait_for": [paths]}; a str cmd runs under
bash -c with pipefail. Jobs run one after another with CUDA_VISIBLE_DEVICES=N and PYTHONUNBUFFERED=1; output
is written to <logs>/<job>.log and to stdout (tee). Before a job starts, every wait_for path is polled every
--poll seconds; for a path ending in .done the sibling .exit is read, and a nonzero value (or a process that
ended without one) fails the job without running it. A failed job does not stop the queue.

Status files in <status>: <job>.pid (child pid, written at start), <job>.json (record), <job>.done (only after
exit code 0), <job>.exit (exit code, written last). The queue itself writes <NAME>.queue.pid and, when it ends,
<NAME>.queue.exit. Exit codes set by the queue: 90 dependency failed, 91 log or status file exists,
92 command could not be started; a child killed by signal k has exit code -k.

The wait helper returns 1 at once (with the last 30 log lines) if any named job has a nonzero .exit, has a .pid
of an ended process and no .exit, or is not finished while no queue process is alive; 0 when every .done exists;
2 on --timeout.
"""
import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

DEFAULT_LOGS = Path("notes") / "kaggle_logs"
DEFAULT_STATUS = Path("notes") / "kaggle_status"
DEP_FAILED = 90
EXISTS = 91
START_FAILED = 92
TAIL_LINES = 30
# A job whose process has ended gets this long for its queue to write .exit before it counts as lost.
EXIT_GRACE_S = 5.0
JOB_KEYS = {"name", "cmd", "wait_for"}
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _write(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_jobs(path) -> list:
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(jobs, list) and jobs, "jobs file must hold a non-empty list"
    names = set()
    for j in jobs:
        extra = set(j) - JOB_KEYS
        assert not extra, f"unknown job keys {sorted(extra)}"
        assert {"name", "cmd"} <= set(j), f"job needs name and cmd: {j}"
        assert _NAME.match(j["name"]), f"bad job name {j['name']!r}"
        assert j["name"] not in names, f"duplicate job name {j['name']}"
        assert isinstance(j["cmd"], (str, list)) and j["cmd"], j
        assert isinstance(j.get("wait_for", []), list), j
        names.add(j["name"])
    return jobs


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def tail(path: Path, n: int = TAIL_LINES) -> str:
    if not path.is_file():
        return f"(no log at {path})"
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return "".join(deque(fh, maxlen=n))


def ended_without_exit(stem: Path) -> bool:
    """True if <stem>.pid names an ended process and no .exit appears within EXIT_GRACE_S."""
    pid = read_int(stem.with_name(stem.name + ".pid"))
    if pid is None or pid_alive(pid):
        return False
    deadline = time.monotonic() + EXIT_GRACE_S
    while time.monotonic() < deadline:
        if stem.with_name(stem.name + ".exit").exists():
            return False
        time.sleep(0.1)
    return not stem.with_name(stem.name + ".exit").exists()


def dependency_state(path: Path) -> tuple:
    """('ok' | 'wait' | 'failed', detail) for one wait_for path."""
    if path.exists():
        return "ok", None
    if path.name.endswith(".done"):
        stem = path.with_name(path.name[:-len(".done")])
        code = read_int(stem.with_name(stem.name + ".exit"))
        if code is not None and code != 0:
            return "failed", f"{path}: dependency exit code {code}"
        if code == 0:
            return "wait", None
        if ended_without_exit(stem):
            return "failed", f"{path}: dependency process ended without an exit file"
    return "wait", None


def _say(msg, fh=None):
    line = f"[queue {_now()}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if fh is not None:
        fh.write(line.encode())
        fh.flush()


def run_job(job: dict, gpu: str, logs: Path, status: Path, poll: float) -> int:
    name = job["name"]
    files = {k: status / f"{name}.{k}" for k in ("pid", "exit", "done", "json")}
    log = logs / f"{name}.log"
    rec = {"name": name, "cmd": job["cmd"], "wait_for": job.get("wait_for", []), "gpu": gpu, "log": str(log)}
    existing = [str(p) for p in (log, *files.values()) if p.exists()]
    if existing:
        _say(f"{name}: refusing, files exist: {existing}")
        if not files["exit"].exists():
            _write(files["exit"], f"{EXISTS}\n")
        return EXISTS
    with open(log, "xb") as lf:
        rec["wait_start"] = _now()
        failed = None
        for dep in job.get("wait_for", []):
            while True:
                state, detail = dependency_state(Path(dep))
                if state == "ok":
                    break
                if state == "failed":
                    failed = detail
                    break
                time.sleep(poll)
            if failed:
                break
        rec["wait_end"] = _now()
        if failed:
            _say(f"{name}: dependency failed, not run: {failed}", lf)
            rec.update(exit=DEP_FAILED, error=failed)
            _write(files["json"], json.dumps(rec, indent=2, sort_keys=True) + "\n")
            _write(files["exit"], f"{DEP_FAILED}\n")
            return DEP_FAILED
        cmd = job["cmd"]
        argv = ["bash", "-c", "set -o pipefail; " + cmd] if isinstance(cmd, str) else [str(c) for c in cmd]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        _say(f"{name}: start on GPU {gpu}: {argv}", lf)
        rec["start"] = _now()
        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError as exc:
            _say(f"{name}: could not start: {exc!r}", lf)
            rec.update(exit=START_FAILED, error=repr(exc))
            _write(files["json"], json.dumps(rec, indent=2, sort_keys=True) + "\n")
            _write(files["exit"], f"{START_FAILED}\n")
            return START_FAILED
        _write(files["pid"], f"{proc.pid}\n")
        rec["pid"] = proc.pid
        for line in iter(proc.stdout.readline, b""):
            lf.write(line)
            lf.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
        code = proc.wait()
        rec.update(end=_now(), wall_s=time.perf_counter() - t0, exit=code)
        _say(f"{name}: exit code {code} after {rec['wall_s']:.1f} s", lf)
    _write(files["json"], json.dumps(rec, indent=2, sort_keys=True) + "\n")
    if code == 0:
        _write(files["done"], rec["end"] + "\n")
    _write(files["exit"], f"{code}\n")
    return code


def run_queue(name: str, gpu: str, jobs_file, logs=DEFAULT_LOGS, status=DEFAULT_STATUS, poll: float = 30.0) -> int:
    jobs = load_jobs(jobs_file)
    logs, status = Path(logs), Path(status)
    logs.mkdir(parents=True, exist_ok=True)
    status.mkdir(parents=True, exist_ok=True)
    qpid, qexit = status / f"{name}.queue.pid", status / f"{name}.queue.exit"
    if qpid.exists() or qexit.exists():
        raise SystemExit(f"queue status files exist for {name}")
    _write(qpid, f"{os.getpid()}\n")
    codes = {}
    try:
        for job in jobs:
            codes[job["name"]] = run_job(job, gpu, logs, status, poll)
    finally:
        _write(qexit, json.dumps(codes, sort_keys=True) + "\n")
    failed = {k: v for k, v in codes.items() if v != 0}
    _say(f"queue {name} finished: {len(codes) - len(failed)} ok, failed: {failed or 'none'}")
    return 1 if failed else 0


def _queues_alive(status: Path) -> bool:
    for p in status.glob("*.queue.pid"):
        if (p.with_name(p.name[:-len(".pid")] + ".exit")).exists():
            continue
        pid = read_int(p)
        if pid is not None and pid_alive(pid):
            return True
    return False


def check(status: Path, logs: Path, names) -> tuple:
    """(state, message): state 'failed', 'done' or 'running'."""
    done, running, pending = [], [], []
    for n in names:
        code = read_int(status / f"{n}.exit")
        if code is not None and code != 0:
            return "failed", f"job {n} failed with exit code {code}\n--- last {TAIL_LINES} lines of {n}.log ---\n" + \
                tail(logs / f"{n}.log")
        if (status / f"{n}.done").exists():
            done.append(n)
            continue
        pid = read_int(status / f"{n}.pid")
        if code is None and pid is not None:
            if ended_without_exit(status / n):
                return "failed", f"job {n}: process {pid} ended without an exit file\n" + tail(logs / f"{n}.log")
            code = read_int(status / f"{n}.exit")
            if code is not None and code != 0:
                return "failed", f"job {n} failed with exit code {code}\n--- last {TAIL_LINES} lines of {n}.log ---\n" + \
                    tail(logs / f"{n}.log")
            if (status / f"{n}.done").exists():
                done.append(n)
                continue
            running.append(n)
        else:
            pending.append(n)
    if len(done) == len(names):
        return "done", f"all {len(names)} jobs done"
    if (pending or running) and not _queues_alive(status):
        stuck = pending + running
        return "failed", f"no queue process alive; unfinished jobs: {stuck}"
    return "running", f"done {len(done)}/{len(names)}; running: {running or '-'}; pending: {len(pending)}"


def wait(status, names, logs=DEFAULT_LOGS, poll: float = 30.0, report: float = 600.0, timeout=None) -> int:
    status, logs = Path(status), Path(logs)
    t0 = time.monotonic()
    last = None
    while True:
        state, msg = check(status, logs, names)
        now = time.monotonic()
        if state == "failed":
            print(f"[wait {_now()}] FAILED: {msg}", flush=True)
            return 1
        if state == "done":
            print(f"[wait {_now()}] {msg}", flush=True)
            return 0
        if last is None or now - last >= report:
            print(f"[wait {_now()}] {msg} ({(now - t0) / 60:.1f} min)", flush=True)
            last = now
        if timeout is not None and now - t0 > timeout:
            print(f"[wait {_now()}] TIMEOUT after {timeout} s: {msg}", flush=True)
            return 2
        time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name")
    ap.add_argument("--gpu")
    ap.add_argument("--jobs")
    ap.add_argument("--wait", metavar="STATUS_DIR")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--logs", default=str(DEFAULT_LOGS))
    ap.add_argument("--status", default=str(DEFAULT_STATUS))
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--report", type=float, default=600.0)
    ap.add_argument("--timeout", type=float, default=None)
    a = ap.parse_args(argv)
    if a.wait:
        if not a.names or a.name or a.jobs:
            ap.error("--wait needs --names and takes no --name/--jobs")
        return wait(a.wait, a.names, a.logs, a.poll, a.report, a.timeout)
    if not (a.name and a.gpu is not None and a.jobs):
        ap.error("a queue needs --name, --gpu and --jobs")
    return run_queue(a.name, a.gpu, a.jobs, a.logs, a.status, a.poll)


if __name__ == "__main__":
    sys.exit(main())
