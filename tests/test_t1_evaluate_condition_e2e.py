"""End-to-end CPU run of python -m task1.evaluate_condition on tiny artifacts built through task1.compress.

A temporary git repository plays the role of the project repository: it holds the run-1 output directories with the
committed artifact manifest copies and the frozen P3 allocation (task1/frozen/p3_s50.json, one commit).
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from common.io import read_jsonl, sha256_file
from common.paths import REPO_ROOT, RESULTS_DIR
from task1 import config as C
from task1 import evaluate_condition as EC

LIM_C, LIM_D, LIM_W, LIM_H = 2, 2, 2, 4
CONDS = ("p3_s50", "q2_b3", "p1_s25")
N_MATRICES = 4 * len(C.ELIGIBLE)

RUN_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "model",
           "tokenizer", "eval_id", "timings_s", "steps"}
CONFIG = {"entry_point", "condition", "condition_spec", "settings", "device", "model_path", "limits", "limited",
          "test_only", "hellaswag_scope", "output_files", "artifacts_dir", "run1_dir", "repo_root", "repro_tol",
          "data_files"}
STEPS = {"inputs", "executed_model", "reproduction", "perplexity", "hellaswag", "resources", "storage"}
TIMINGS = {"load_s", "artifact_load_s", "calib_nll_c4_development_s", "ppl_wikitext2_test_s", "ppl_c4_validation_s",
           "hellaswag_s", "resources_s"}
PPL_SUMMARY = {"eval_id", "suite", "n_windows", "N", "nll_sum", "mean_nll", "ppl", "batch_size", "file"}
PPL_ROW = {"eval_id", "suite", "window_index", "source", "source_row", "token_offset", "nll_sum", "n_scored"}
HS_SUMMARY = {"eval_id", "scope", "n", "acc", "acc_norm", "n_correct_acc", "n_correct_acc_norm",
              "n_context_not_prefix", "orders_first_last", "task", "prompt_format", "batch_size", "file"}
HS_ROW = {"eval_id", "order", "source_row_index", "label", "loglik", "cont_chars", "cont_tokens", "ctx_tokens",
          "pred_acc", "pred_acc_norm", "correct_acc", "correct_acc_norm"}
CAL_ROW = {"eval_id", "split", "seq_index", "source_row", "token_offset", "nll_sum", "n_scored"}
REPRO = {"eval_id", "split", "n_sequences", "N", "nll_sum", "mean_nll", "batch_size", "file", "run1_mean_nll",
         "diff_vs_run1", "reproduces_run1"}
ART = {"dir", "file", "n_tensors_verified", "stored_tensor_bytes", "manifest_bytes", "manifest_sha256",
       "file_sha256_verified", "tensor_sha256_verified"}
RUN1 = {"dir", "run_sha256", "run_git_commit", "run_git_dirty", "manifest_sha256", "manifest_copy",
        "development_mean_nll", "calibration_mean_nll"}
GIT_REC = {"path", "exists", "in_repo", "tracked", "clean", "commits", "repo_root", "rel_path", "head",
           "status_porcelain", "sha256"}
FREEZE = GIT_REC | {"applies", "file", "freeze_commit", "freeze_commit_date", "freeze_author_date",
                    "allocation_sha256_in_manifest", "candidate_id", "totals"}
STORAGE = {"convention", "eval_id", "eligible_weights", "eligible_matrices", "non_eligible_params", "fp16_dense",
           "nonzeros", "nominal_bits_per_eligible_weight", "representation", "logical", "param_counts",
           "artifact_files"}
MEAS = {"what", "unit", "repetitions", "warmup", "times_ms", "median_ms", "min_ms", "max_ms",
        "peak_allocated_bytes", "peak_reserved_bytes", "peak_allocated_bytes_max", "peak_reserved_bytes_max"}


def _env():
    return dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")


def _run(args):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True,
                          timeout=1800)


def _git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def _init_repo(path):
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    return path


def _freeze(repo, cond, src):
    dst = repo / "task1" / "frozen" / f"{cond}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    _git(repo, "add", "--", str(dst.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", f"Freeze {cond}")
    return dst, _git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def built(tiny_compress4_dir, tmp_path_factory):
    b = tmp_path_factory.mktemp("evalcond")
    repo = _init_repo(b / "repo")
    run1 = repo / "results" / "task1" / "run1"
    t = ["--model-path", str(tiny_compress4_dir), "--device", "cpu", "--limit-calib", str(LIM_C)]
    steps = [
        ["task1.calib_stats", "--out", str(b / "calib"), *t],
        ["task1.probe", "--kind", "prune", "--calib-stats", str(b / "calib"), "--out", str(b / "probe_prune"), *t],
        ["task1.select", "--kind", "p3", "--target", "s50", "--probe", str(b / "probe_prune"), "--calib-stats",
         str(b / "calib"), "--out", str(b / "sel_p3"), *t, "--limit-dev", str(LIM_D)],
        ["task1.compress", "--condition", "p3_s50", "--allocation", str(b / "sel_p3" / "selected_allocation.json"),
         "--out", str(run1 / "p3_s50"), "--artifacts", str(b / "art" / "p3_s50"), *t, "--limit-dev", str(LIM_D)],
        ["task1.compress", "--condition", "q2_b3", "--calib-stats", str(b / "calib"), "--out", str(run1 / "q2_b3"),
         "--artifacts", str(b / "art" / "q2_b3"), *t, "--limit-dev", str(LIM_D)],
        ["task1.compress", "--condition", "p1_s25", "--out", str(run1 / "p1_s25"), "--artifacts",
         str(b / "art" / "p1_s25"), *t, "--limit-dev", str(LIM_D)],
    ]
    for a in steps:
        p = _run(a)
        assert p.returncode == 0, (a[0], p.stderr[-4000:])
    for cond in CONDS:
        shutil.copyfile(b / "art" / cond / "manifest.json", run1 / cond / EC.RUN1_MANIFEST_COPY)
    _git(repo, "add", "--", "results")
    _git(repo, "commit", "-q", "-m", "Import run-1 outputs")
    frozen, freeze_commit = _freeze(repo, "p3_s50", b / "sel_p3" / "selected_allocation.json")
    return {"base": b, "repo": repo, "run1": run1, "model": tiny_compress4_dir, "frozen": frozen,
            "freeze_commit": freeze_commit}


def _eval_args(built, cond, out, art=None, run1=None, repo=None, extra=()):
    return ["--condition", cond, "--artifacts", str(art or built["base"] / "art" / cond),
            "--run1", str(run1 or built["run1"] / cond), "--out", str(out), "--model-path", str(built["model"]),
            "--device", "cpu", "--limit-windows", str(LIM_W), "--limit-hellaswag", str(LIM_H),
            "--limit-dev", str(LIM_D), "--repo-root", str(repo or built["repo"]), *extra]


@pytest.fixture(scope="module")
def evals(built):
    outs = {}
    for cond in CONDS:
        out = built["base"] / "eval" / cond
        p = _run(["task1.evaluate_condition", *_eval_args(built, cond, out)])
        assert p.returncode == 0, (cond, p.stderr[-4000:])
        outs[cond] = out
    return outs


def _load(p):
    return json.loads(p.read_text())


@pytest.mark.parametrize("cond", CONDS)
def test_outputs(built, evals, cond):
    d = evals[cond]
    spec = C.CONDITIONS[cond]
    scope = spec["hellaswag"]
    files = EC.output_files(scope)
    assert sorted(x.name for x in d.iterdir()) == sorted(files.values())
    assert files["hellaswag"] == ("hellaswag_full.jsonl.gz" if cond != "p1_s25" else "hellaswag_subset512.jsonl.gz")
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and r["eval_id"] == C.EVAL_ID
    assert set(r["config"]) == CONFIG and r["config"]["hellaswag_scope"] == scope
    assert r["config"]["limits"] == {"windows": LIM_W, "hellaswag": LIM_H, "dev": LIM_D}
    assert r["config"]["test_only"] is True and r["config"]["settings"] == C.settings_dict()
    assert len(r["config"]["data_files"]) == 5
    assert set(r["timings_s"]) == TIMINGS
    s = r["steps"]
    assert set(s) == STEPS
    inp = s["inputs"]
    assert set(inp) == {"artifact", "run1", "freeze"}
    art_dir = built["base"] / "art" / cond
    assert set(inp["artifact"]) == ART
    assert inp["artifact"]["file"]["sha256"] == sha256_file(art_dir / "artifact.safetensors")
    assert inp["artifact"]["manifest_sha256"] == sha256_file(art_dir / "manifest.json")
    man = _load(art_dir / "manifest.json")
    assert inp["artifact"]["n_tensors_verified"] == len(man["tensors"])
    assert set(inp["run1"]) == RUN1 and set(inp["run1"]["manifest_copy"]) == GIT_REC
    assert inp["run1"]["manifest_copy"]["tracked"] and inp["run1"]["manifest_copy"]["clean"]
    r1 = _load(built["run1"] / cond / "run.json")
    assert inp["run1"]["manifest_sha256"] == r1["steps"]["artifact"]["manifest_sha256"]
    fz = inp["freeze"]
    if spec["family"] == "P3":
        assert set(fz) == FREEZE and fz["applies"] is True
        assert fz["freeze_commit"] == built["freeze_commit"] and len(fz["commits"]) == 1
        assert fz["freeze_commit_date"] == fz["commits"][0]["committer_date"]
        assert fz["sha256"] == fz["allocation_sha256_in_manifest"] == man["allocation"]["sha256"]
        assert fz["rel_path"] == "task1/frozen/p3_s50.json"
    else:
        assert fz == {"applies": False}

    rp = s["reproduction"]
    assert set(rp) == REPRO and rp["reproduces_run1"] is True and rp["n_sequences"] == LIM_D
    assert rp["diff_vs_run1"] == 0.0 and rp["run1_mean_nll"] == r1["steps"]["calib_nll"]["c4_development"]["mean_nll"]
    assert set(s["perplexity"]) == {"wikitext2_test", "c4_validation"}
    for suite, summ in s["perplexity"].items():
        assert set(summ) == PPL_SUMMARY and summ["n_windows"] == LIM_W and summ["file"] == files[f"ppl_{suite}"]
    assert set(s["hellaswag"]) == HS_SUMMARY and s["hellaswag"]["n"] == LIM_H and s["hellaswag"]["scope"] == scope
    res = s["resources"]
    assert res["execution"]["mode"] == "dense_fp16" and res["eval_id"] == C.EVAL_ID
    assert set(res["measurements"]) == {"prefill_512", "prefill_2048", "decode_64_after_prefill_512"}
    assert all(MEAS <= set(m) for m in res["measurements"].values())

    st = s["storage"]
    fam = spec["family"]
    extra = {"sparse16", "mask"} if fam.startswith("P") else {"packed_quant"}
    assert set(st) == STORAGE | extra
    nz = st["nonzeros"]
    assert len(nz["per_matrix"]) == N_MATRICES and nz["global"] == sum(nz["per_matrix"].values())
    assert nz["per_matrix"] == {m["qualname"]: m["nonzeros"] for m in man["matrices"]}
    assert st["logical"]["eligible_bytes"] == man["logical_bytes"]["eligible"]
    assert st["logical"]["whole_model_bytes"] == man["logical_bytes"]["whole_model"]
    assert st["logical"]["tied_lm_head_counted_once"] is True
    assert st["artifact_files"]["safetensors_bytes"] == (art_dir / "artifact.safetensors").stat().st_size
    assert st["artifact_files"]["manifest_bytes"] == (art_dir / "manifest.json").stat().st_size
    if fam.startswith("P"):
        assert st["nominal_bits_per_eligible_weight"] == 16 and st["representation"] == "sparse16"
        assert st["sparse16"]["eligible_bytes"] == st["logical"]["eligible_bytes"]
        target = 0.5 if cond == "p3_s50" else 0.25
        assert st["mask"]["zero_fraction"] == pytest.approx(target, abs=1e-3)
        if fam == "P3":
            sel = _load(built["frozen"])
            assert st["mask"]["zeros"] == st["mask"]["p2_zeros"] == sel["totals"]["p2_zeros"]
    else:
        assert st["nominal_bits_per_eligible_weight"] == 3 and st["representation"] == "packed_quant"
        assert st["packed_quant"]["total"] == st["logical"]["eligible_bytes"]
        assert set(st["packed_quant"]["bits_per_matrix"].values()) == {3}

    for suite in ("wikitext2_test", "c4_validation"):
        rows = read_jsonl(d / files[f"ppl_{suite}"])
        assert len(rows) == LIM_W and all(set(x) == PPL_ROW for x in rows)
        assert sum(x["n_scored"] for x in rows) == s["perplexity"][suite]["N"]
    hs = read_jsonl(d / files["hellaswag"])
    assert len(hs) == LIM_H and all(set(x) == HS_ROW for x in hs) and [x["order"] for x in hs] == list(range(LIM_H))
    cal = read_jsonl(d / files["calib_nll_c4_development"])
    assert len(cal) == LIM_D and all(set(x) == CAL_ROW for x in cal)
    for p in d.glob("*.gz"):
        assert p.read_bytes()[4:8] == b"\x00\x00\x00\x00"
        assert all(x["eval_id"] == C.EVAL_ID for x in read_jsonl(p))


def _refused(built, tmp_path, cond, match, **kw):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=match):
        EC.main(_eval_args(built, cond, out, **kw))
    assert not out.exists()


def test_refuses_tampered_tensor(built, tmp_path):
    art = tmp_path / "art"
    shutil.copytree(built["base"] / "art" / "p3_s50", art)
    f = art / "artifact.safetensors"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))
    _refused(built, tmp_path, "p3_s50", "integrity check failed.*file sha256", art=art)


def test_verify_refuses_tensor_with_matching_file_hash(built, tmp_path):
    art = tmp_path / "art"
    shutil.copytree(built["base"] / "art" / "q2_b3", art)
    f = art / "artifact.safetensors"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))
    man = _load(art / "manifest.json")
    man["file"]["sha256"] = sha256_file(f)
    (art / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="does not match its manifest sha256"):
        EC.verify_artifact(art, "q2_b3", {"model_path": str(built["model"])})


def test_refuses_manifest_differing_from_committed_copy(built, tmp_path):
    art = tmp_path / "art"
    shutil.copytree(built["base"] / "art" / "q2_b3", art)
    man = _load(art / "manifest.json")
    man["condition_note"] = "edited"
    (art / "manifest.json").write_text(json.dumps(man))
    _refused(built, tmp_path, "q2_b3", "committed run-1 copy", art=art)


def test_refuses_modified_frozen_file(built, tmp_path):
    f = built["frozen"]
    orig = f.read_bytes()
    try:
        f.write_bytes(orig + b" ")
        _refused(built, tmp_path, "p3_s50", "modified in the working tree")
    finally:
        f.write_bytes(orig)
    assert _git(built["repo"], "status", "--porcelain") == ""


def test_refuses_frozen_file_with_two_commits(built, tmp_path):
    repo = _init_repo(tmp_path / "repo2")
    src = built["base"] / "sel_p3" / "selected_allocation.json"
    dst = repo / "task1" / "frozen" / "p3_s50.json"
    dst.parent.mkdir(parents=True)
    dst.write_text(json.dumps(_load(src)))
    _git(repo, "add", "--", "task1/frozen/p3_s50.json")
    _git(repo, "commit", "-q", "-m", "First version")
    shutil.copyfile(src, dst)
    _git(repo, "add", "--", "task1/frozen/p3_s50.json")
    _git(repo, "commit", "-q", "-m", "Second version")
    assert sha256_file(dst) == sha256_file(src)
    _refused(built, tmp_path, "p3_s50", "touched by 2 commits", repo=repo)


def test_refuses_untracked_or_missing_frozen_file(built, tmp_path):
    repo = _init_repo(tmp_path / "repo3")
    _refused(built, tmp_path, "p3_s50", "file not found", repo=repo)
    dst = repo / "task1" / "frozen" / "p3_s50.json"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(built["frozen"], dst)
    _refused(built, tmp_path, "p3_s50", "not tracked by git", repo=repo)


def test_refuses_other_inputs(built, tmp_path):
    _refused(built, tmp_path, "p3_s50", "not task1.compress p3_s50", run1=built["run1"] / "q2_b3")
    with pytest.raises(SystemExit, match="evaluated by task1.baseline"):
        EC.main(["--condition", "fp16", "--artifacts", "a", "--run1", "b", "--out", str(tmp_path / "o")])
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="output directory exists"):
        EC.main(_eval_args(built, "q2_b3", out))
    real = RESULTS_DIR / "task1" / "evaluate_condition_refusal_check"
    with pytest.raises(SystemExit, match="refusing test options"):
        EC.main(_eval_args(built, "q2_b3", real))
    assert not real.exists()
