"""End-to-end CPU test of the prefix-bank generator and the preflight with tiny random Qwen2 models."""
import json
import subprocess

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from common.io import read_jsonl, sha256_gz_content
from common.paths import RESULTS_DIR
from task3 import banks as B
from task3 import data as D
from task3 import fixed64 as F
from task3 import preflight as P
from task3.lm import HFCachedLM
from task3.sampling import EOS_IDS, SAMPLER_ID, VOCAB_SIZE, make_generator
from task3.spec import target_only_generate


def _save_tiny(d, seed, layers, hidden, tok):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=2 * hidden, vocab_size=VOCAB_SIZE, tie_word_embeddings=True,
                      max_position_embeddings=4096, initializer_range=0.2)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_dirs(tmp_path_factory):
    rv = D.revisions()["model_target"]
    tok = AutoTokenizer.from_pretrained(rv["repo"], revision=rv["revision"], local_files_only=True)
    root = tmp_path_factory.mktemp("t3_tiny")
    return {"target": _save_tiny(root / "target", 6304, 2, 64, tok),
            "draft": _save_tiny(root / "draft", 6305, 1, 32, tok),
            "own": _save_tiny(root / "own", 6306, 1, 32, tok), "tok": tok}


@pytest.fixture(scope="module")
def committed_check(tmp_path_factory):
    """A passing sampler check committed in a throwaway git repository."""
    repo = tmp_path_factory.mktemp("t3_gitrepo")
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    good = repo / "sampler_check.json"
    good.write_text(json.dumps({"status": "ok", "passes": {"a": True, "b": True}, "all_pass": True}))
    bad = repo / "sampler_check_bad.json"
    bad.write_text(json.dumps({"status": "ok", "passes": {"a": True, "b": False}, "all_pass": False}))
    subprocess.run(git + ["add", "sampler_check.json", "sampler_check_bad.json"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "check"], check=True)
    uncommitted = repo / "sampler_check_new.json"
    uncommitted.write_text(good.read_text())
    return {"good": good, "bad": bad, "uncommitted": uncommitted}


@pytest.fixture(scope="module")
def preflight_json(tiny_dirs, tmp_path_factory):
    out = tmp_path_factory.mktemp("t3_pre") / "preflight.json"
    rc = P.main(["--out", str(out), "--target-path", str(tiny_dirs["target"]), "--draft-path",
                 str(tiny_dirs["draft"]), "--device", "cpu", "--dtype", "float32", "--cap", "12"])
    assert rc == 0
    return out


def test_preflight_outputs(preflight_json):
    d = json.loads(preflight_json.read_text())
    assert d["status"] == "ok" and d["all_pass"] is True
    checks = ("split_prefill", "verify_windows", "crop_reforward", "speculative_cache")
    assert set(d["passes"]) == {f"{m}_{c}" for m in ("target", "draft") for c in checks}
    for m in ("target", "draft"):
        for c in checks:
            s = d["summary"][m][c]
            assert set(s) == {"max_abs_diff", "max_tv", "argmax_agreement", "rows", "pass"}
            assert s["max_abs_diff"] < 1e-4 and s["argmax_agreement"] == 1.0 and s["rows"] > 0
    assert len(d["per_prompt"]) == 4
    assert [r["prompt_id"] for r in d["per_prompt"]] == d["config"]["prompt_ids"]
    for row in d["per_prompt"]:
        assert {"prompt_tokens", "target", "draft", "speculative"} <= set(row)
    for key in ("logit_tol", "tv_tol", "data_provenance", "generation_settings"):
        assert key in d["config"]
    assert set(d["model_revisions"]) == {"target", "draft"}


def _run(kind, out, tiny_dirs, extra=()):
    args = ["--kind", kind, "--out", str(out), "--target-path", str(tiny_dirs["target"]), "--device", "cpu",
            "--dtype", "float32", "--limit-prompts", "2", *extra]
    if kind in ("original_draft", "exact_spec_original_gamma4"):
        args += ["--draft-path", str(tiny_dirs["draft"])]
    return B.main(args)


POSITION_KEYS = {"record_type", "prompt_id", "domain", "group", "bank", "position", "prompt",
                 "response_prefix_token_ids", "source_length", "source_generation_seed", "sampled_token_id",
                 "termination_reason", "model_revision", "sampler_id", "bank_id"}
SEQUENCE_KEYS = {"record_type", "prompt_id", "domain", "group", "bank", "prompt", "source_length",
                 "source_generation_seed", "termination_reason", "model_revision", "sampler_id", "bank_id",
                 "prompt_index", "prompt_token_ids", "response_token_ids", "positions"}


@pytest.mark.parametrize("kind", B.KINDS)
def test_bank_end_to_end(kind, tiny_dirs, committed_check, preflight_json, tmp_path):
    out = tmp_path / f"{kind}.jsonl.gz"
    extra = []
    run_name = None
    if kind == "exact_spec_original_gamma4":
        extra = ["--sampler-check", str(committed_check["good"]), "--preflight", str(preflight_json)]
    if kind == "own":
        run_name = "on_policy_6305"
        extra = ["--draft-path", str(tiny_dirs["own"]), "--run-name", run_name]
    assert _run(kind, out, tiny_dirs, extra) == 0
    bank = B.bank_name(kind, run_name)
    recs = read_jsonl(out)
    seqs = [r for r in recs if r["record_type"] == "sequence"]
    poss = [r for r in recs if r["record_type"] == "position"]
    assert len(seqs) == 2 and len(seqs) + len(poss) == len(recs)
    prompts, _ = D.heldout_prompts()
    for j, s in enumerate(seqs):
        want_keys = SEQUENCE_KEYS | ({"speculative_counts"} if kind == "exact_spec_original_gamma4" else set())
        assert set(s) == want_keys
        assert s["prompt_index"] == j and s["prompt_id"] == prompts[j]["prompt_id"]
        assert s["prompt"] == prompts[j]["prompt"] and s["domain"] == "math" and s["group"] == "math"
        assert s["bank"] == bank and s["sampler_id"] == SAMPLER_ID and s["bank_id"] == B.BANK_ID
        assert s["prompt_token_ids"] == D.chat_ids(tiny_dirs["tok"], prompts[j]["prompt"])
        seq = s["response_token_ids"]
        assert 1 <= len(seq) <= B.MAX_NEW_TOKENS and s["source_length"] == len(seq)
        if s["termination_reason"] == "eos":
            assert seq[-1] in EOS_IDS and not any(t in EOS_IDS for t in seq[:-1])
        else:
            assert s["termination_reason"] == "cap" and len(seq) == B.MAX_NEW_TOKENS
            assert not any(t in EOS_IDS for t in seq)
        assert s["source_generation_seed"] == 6304 + 10007 * j + B.seed_offset(kind, run_name)
        assert s["positions"] == B.select_positions(s["prompt_id"], bank, len(seq))
        mine = [p for p in poss if p["prompt_id"] == s["prompt_id"]]
        assert [p["position"] for p in mine] == s["positions"]
        for p in mine:
            assert set(p) == POSITION_KEYS
            assert p["response_prefix_token_ids"] == seq[: p["position"] - 1]
            assert p["sampled_token_id"] == seq[p["position"] - 1]
            for key in ("bank", "source_length", "source_generation_seed", "termination_reason", "model_revision"):
                assert p[key] == s[key]
        if kind == "exact_spec_original_gamma4":
            c = s["speculative_counts"]
            assert c["accepted"] <= c["examined"] <= c["proposed"]
            assert c["target_verification_passes"] == c["n_rounds"]
        else:
            path = tiny_dirs["target"] if kind == "teacher" else (
                tiny_dirs["own"] if kind == "own" else tiny_dirs["draft"])
            model, _ = D.load_model(str(path), "cpu", "float32")
            again = target_only_generate(HFCachedLM(model), s["prompt_token_ids"], B.MAX_NEW_TOKENS, EOS_IDS,
                                         make_generator(s["source_generation_seed"], "cpu"))
            assert again["output_ids"] == seq
    models = set(seqs[0]["model_revision"])
    assert models == ({"target", "draft"} if kind == "exact_spec_original_gamma4"
                      else {"target"} if kind == "teacher" else {"draft"})
    for pv in seqs[0]["model_revision"].values():
        assert pv["source"] == "local" and "config.json" in pv["file_sha256"] and pv["dtype"] == "float32"

    run = json.loads((tmp_path / f"{kind}.json").read_text())
    assert run["status"] == "ok"
    assert run["n_sequences"] == 2 and run["n_positions"] == len(poss)
    assert sum(run["terminations"].values()) == 2
    assert run["output_content_sha256"] == sha256_gz_content(out)
    cfg = run["config"]
    assert cfg["kind"] == kind and cfg["bank"] == bank and cfg["max_new_tokens"] == 192
    assert cfg["generation_settings"]["batch_size"] == 1 and cfg["generation_settings"]["top_k"] is None
    assert cfg["generation_settings"]["padding"].startswith("none")
    assert cfg["limit_prompts"] == 2 and cfg["sampler_id"] == SAMPLER_ID
    assert cfg["gamma"] == (4 if kind == "exact_spec_original_gamma4" else None)
    if kind == "exact_spec_original_gamma4":
        assert cfg["gate"]["sampler_check"]["passes"] == {"a": True, "b": True}
        assert cfg["gate"]["preflight"]["all_pass"] is True
    for key in ("git_commit", "git_dirty", "package_versions", "device_names"):
        assert key in run["meta"]
    assert "elapsed_wall_s" in run and run["progress"]["prompts_done"] == 2
    with pytest.raises(SystemExit):
        _run(kind, out, tiny_dirs, extra)


def test_bank_refusals(tiny_dirs, committed_check, preflight_json, tmp_path):
    spec = "exact_spec_original_gamma4"
    with pytest.raises(SystemExit):
        _run(spec, tmp_path / "a.jsonl.gz", tiny_dirs, ["--sampler-check", str(committed_check["bad"]),
                                                        "--preflight", str(preflight_json)])
    with pytest.raises(Exception):
        _run(spec, tmp_path / "b.jsonl.gz", tiny_dirs, ["--sampler-check", str(committed_check["uncommitted"]),
                                                        "--preflight", str(preflight_json)])
    with pytest.raises(SystemExit):
        _run(spec, tmp_path / "c.jsonl.gz", tiny_dirs, ["--sampler-check", str(committed_check["good"])])
    with pytest.raises(SystemExit):
        _run("own", tmp_path / "d.jsonl.gz", tiny_dirs, ["--draft-path", str(tiny_dirs["own"])])
    with pytest.raises(SystemExit):
        _run("teacher", tmp_path / "e.jsonl.gz", tiny_dirs, ["--run-name", "off_policy_6304"])
    with pytest.raises(SystemExit):
        _run("teacher", tmp_path / "f.json", tiny_dirs)
    under = RESULTS_DIR / "task3" / "banks" / "never_written.jsonl.gz"
    with pytest.raises(SystemExit):
        _run("teacher", under, tiny_dirs)
    assert not under.exists()
    assert not list(tmp_path.glob("*.json"))


def test_seed_offsets_and_positions():
    assert [B.seed_offset(k) for k in ("teacher", "original_draft", "exact_spec_original_gamma4")] == [0, 123, 456]
    assert [B.seed_offset("own", r) for r in ("off_policy_6304", "off_policy_6305", "on_policy_6304",
                                              "on_policy_6305")] == [1001, 1002, 1003, 1004]
    assert B.generation_seed(3, "original_draft") == 6304 + 3 * 10007 + 123
    pos = B.select_positions("GSM8K-0000", "teacher", 192)
    assert len(pos) == 48 and sum(1 <= p <= 16 for p in pos) == 8 and sum(65 <= p for p in pos) == 16
    assert B.select_positions("X", "teacher", 5) == sorted(B.select_positions("X", "teacher", 5))
    assert len(B.select_positions("X", "teacher", 5)) == 5


def test_fixed64_ids_file():
    ids = F.fixed64_ids()
    assert len(ids) == 24 and len(set(ids)) == 24
    assert all(i.startswith("GSM8K-") and 0 <= int(i[6:]) < 60 for i in ids)
    d = json.loads((RESULTS_DIR.parent / "task3" / "fixed64_ids.json").read_text())
    assert d["ids"] == ids and d["n"] == 24 and d["fixed64_id"] == F.FIXED64_ID
    assert d["source_sha256"] == "b0ad559360962ffe9445161d9e01d3f1f2ab75aa54cba1ad2d91346366b6dd55"
