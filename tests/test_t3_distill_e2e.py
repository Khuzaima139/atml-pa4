"""End-to-end CPU test of the Task 3 teacher collection, both training methods, the merge and an own-student bank,
through the real entry points with tiny random Qwen2 models and tiny quotas."""
import json

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from common.io import read_jsonl, sha256_gz_content
from common.paths import RESULTS_DIR
from task3 import banks as B
from task3 import collect as C
from task3 import data as D
from task3 import distill as T
from task3 import merge as M
from task3 import train as TR
from task3.lm import HFCachedLM, hf_full_logits
from task3.sampling import EOS_IDS, SAMPLER_ID, VOCAB_SIZE, make_generator
from task3.spec import target_only_generate

LIM = {"math": 3, "short": 3, "k_short": 2}
QUOTA = {"math": 100, "short": 220}


def _save_tiny(d, seed, layers, hidden, tok):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=2 * hidden, vocab_size=VOCAB_SIZE, tie_word_embeddings=True,
                      max_position_embeddings=4096, initializer_range=0.2)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    rv = D.revisions()["model_target"]
    tok = AutoTokenizer.from_pretrained(rv["repo"], revision=rv["revision"], local_files_only=True)
    root = tmp_path_factory.mktemp("t3_distill")
    return {"teacher": _save_tiny(root / "teacher", 6304, 2, 64, tok),
            "student": _save_tiny(root / "student", 6305, 1, 32, tok), "tok": tok, "root": root}


def _dev(tiny):
    return ["--device", "cpu", "--dtype", "float32"]


@pytest.fixture(scope="module")
def collection(tiny):
    out = tiny["root"] / "coll" / "teacher_collection.jsonl.gz"
    rc = C.main(["--out", str(out), "--teacher-path", str(tiny["teacher"]), *_dev(tiny),
                 "--limit-prompts-math", str(LIM["math"]), "--limit-prompts-short", str(LIM["short"]),
                 "--limit-k-short", str(LIM["k_short"])])
    assert rc == 0
    return out


def _train(tiny, method, out, adapter, collection=None, extra=()):
    args = ["--method", method, "--seed", "6304", "--out", str(out), "--adapter-dir", str(adapter),
            "--teacher-path", str(tiny["teacher"]), "--student-path", str(tiny["student"]), *_dev(tiny),
            "--limit-prompts-math", str(LIM["math"]), "--limit-prompts-short", str(LIM["short"]),
            "--limit-quota-math", str(QUOTA["math"]), "--limit-quota-short", str(QUOTA["short"]), *extra]
    if collection is not None:
        args += ["--collection", str(collection)]
    return TR.main(args)


@pytest.fixture(scope="module")
def runs(tiny, collection):
    out = {}
    for method in T.METHODS:
        d = tiny["root"] / "train" / f"{method}_6304"
        a = tiny["root"] / "adapters" / f"{method}_6304"
        assert _train(tiny, method, d, a, collection if method == "off_policy" else None) == 0
        out[method] = {"dir": d, "adapter": a}
    return out


COLL_KEYS = {"source_id", "category", "k", "rollout_seed", "prompt_tokens", "response_token_ids", "termination",
             "length", "generation_s", "prefill_s", "collection_id", "sampler_id"}


def test_collection_outputs(tiny, collection):
    recs = read_jsonl(collection)
    assert len(recs) == LIM["math"] + LIM["short"] * LIM["k_short"]
    items, _ = D.training_items(LIM["math"], LIM["short"])
    want = [(x["prompt_id"], k) for x in items for k in range(1 if x["category"] == "math" else LIM["k_short"])]
    assert [(r["source_id"], r["k"]) for r in recs] == want
    for r in recs:
        assert set(r) == COLL_KEYS and r["sampler_id"] == SAMPLER_ID and r["collection_id"] == T.COLLECTION_ID
        assert r["rollout_seed"] == T.collection_seed(r["source_id"], r["k"])
        assert r["length"] == len(r["response_token_ids"]) <= T.CAPS[r["category"]]
        if r["termination"] == "cap":
            assert r["length"] == T.CAPS[r["category"]] and not set(r["response_token_ids"]) & set(EOS_IDS)
        else:
            assert r["termination"] == "eos" and r["response_token_ids"][-1] in EOS_IDS
    model, _ = D.load_model(str(tiny["teacher"]), "cpu", "float32")
    r = recs[-1]
    ids = D.chat_ids(tiny["tok"], items[-1]["prompt"])
    again = target_only_generate(HFCachedLM(model), ids, T.CAPS["short"], EOS_IDS,
                                 make_generator(r["rollout_seed"], "cpu"))
    assert again["output_ids"] == r["response_token_ids"]
    run = json.loads(collection.with_name("teacher_collection.json").read_text())
    assert run["status"] == "ok" and run["n_responses"] == len(recs)
    assert run["output_content_sha256"] == sha256_gz_content(collection)
    assert run["heldout_guard"]["heldout_file_opens"] == [] and run["heldout_guard"]["items_checked"] == 6
    assert run["config"]["limited"] is True and run["config"]["k"] == {"math": 1, "short": 2}
    for key in ("load_s", "teacher_generation_s", "teacher_generation_s_total", "mean_prefill_s", "median_step_s"):
        assert key in run["timing"]
    assert run["timing"]["teacher_generation_s_total"] > 0
    for key in ("git_commit", "git_dirty", "package_versions", "elapsed_wall_s"):
        assert key in run["meta"] or key in run
    assert run["meta"]["package_versions"]["peft"] == "0.22.0"


def test_collection_refusals(tiny, collection, tmp_path):
    with pytest.raises(SystemExit):
        C.main(["--out", str(collection), "--teacher-path", str(tiny["teacher"]), *_dev(tiny)])
    under = RESULTS_DIR / "task3" / "collection" / "never.jsonl.gz"
    with pytest.raises(SystemExit):
        C.main(["--out", str(under), "--teacher-path", str(tiny["teacher"]), *_dev(tiny), "--limit-k-short", "1"])
    assert not under.exists() and not under.with_name("never.json").exists()


PRES_KEYS = {"presentation", "category", "source_id", "n_of_prompt", "prompt_tokens", "method", "k", "rollout_seed",
             "wrap_around", "producing_update", "response_token_ids", "termination", "length", "scored", "truncated",
             "teacher_s", "rollout_s", "updates", "consumed"}
UPDATE_KEYS = {"update", "positions", "positions_by_category", "loss", "kl_mean_by_category", "lr", "scale_before",
               "scale_after", "clip_norm", "skipped", "cumulative_scored", "cumulative_scored_by_category",
               "presentations_started", "teacher_s", "rollout_s", "fwd_bwd_s", "step_s", "wall_s"}


@pytest.mark.parametrize("method", T.METHODS)
def test_training_outputs(method, tiny, runs, collection):
    d, adapter = runs[method]["dir"], runs[method]["adapter"]
    run = json.loads((d / "run.json").read_text())
    pres = read_jsonl(d / "presentations.jsonl.gz")
    assert run["status"] == "ok" and run["config"]["method"] == method and run["config"]["limited"] is True
    ups = run["updates"]
    assert [u["positions"] for u in ups] == [256, 64] == T.update_sizes(sum(QUOTA.values()))
    assert run["summary"]["n_updates"] == 2 and run["summary"]["scored_by_category"] == QUOTA
    assert run["summary"]["scored_total"] == 320 and run["summary"]["unconsumed_presentations"] == []
    for u in ups:
        assert set(u) == UPDATE_KEYS and u["lr"] == T.LR and u["skipped"] is False
        assert sum(u["positions_by_category"].values()) == u["positions"] and u["loss"] > 0
        assert isinstance(u["clip_norm"], float) and u["scale_after"] == u["scale_before"] == 65536.0
    assert ups[-1]["cumulative_scored_by_category"] == QUOTA
    for p in pres:
        assert set(p) == PRES_KEYS and p["consumed"] == p["scored"] <= p["length"]
        assert p["truncated"] == (p["scored"] < p["length"])
    # quota truncation: the response that crosses each quota is scored only up to it
    for c in T.CATEGORIES:
        mine = [p for p in pres if p["category"] == c]
        assert sum(p["scored"] for p in mine) == QUOTA[c]
        if sum(p["length"] for p in mine) > QUOTA[c]:
            assert [p["truncated"] for p in mine] == [False] * (len(mine) - 1) + [True]
    # 256-position split: the boundary falls inside a response, which is consumed by both updates
    split = [p for p in pres if p["updates"] and p["updates"][0] != p["updates"][-1]]
    assert len(split) == 1 and split[0]["updates"] == [1, 2]
    assert sum(len(p["updates"]) for p in pres) == len(pres) + 1
    # D7 schedule: permutation with the training seed, cycled, mathematics first on ties
    items, _ = D.training_items(LIM["math"], LIM["short"])
    by_cat = T.by_category(items)
    for c in T.CATEGORIES:
        order = T.permutation(len(by_cat[c]), 6304)
        got = [p["source_id"] for p in pres if p["category"] == c]
        assert got == [by_cat[c][order[i % len(order)]]["prompt_id"] for i in range(len(got))]
    assert pres[0]["category"] == "math"
    if method == "off_policy":
        coll = {(r["source_id"], r["k"]): r for r in read_jsonl(collection)}
        for p in pres:
            K = 1 if p["category"] == "math" else LIM["k_short"]
            assert p["k"] == p["n_of_prompt"] % K and p["wrap_around"] == (p["n_of_prompt"] >= K)
            assert p["response_token_ids"] == coll[(p["source_id"], p["k"])]["response_token_ids"]
            assert p["producing_update"] is None
        assert run["summary"]["wrap_arounds"] >= 1
        assert run["config"]["collection"]["content_sha256"] == sha256_gz_content(collection)
    else:
        for p in pres:
            assert p["rollout_seed"] == T.rollout_seed(6304, p["presentation"]) and p["k"] is None
            assert p["producing_update"] == (p["updates"][0] - 1) and p["rollout_s"] > 0
    assert run["adapter_sha256"] == T.dir_sha256(adapter) and "adapter_model.safetensors" in run["adapter_sha256"]
    assert run["heldout_guard"]["heldout_file_opens"] == []
    assert run["presentations_file"]["content_sha256"] == sha256_gz_content(d / "presentations.jsonl.gz")
    assert run["lora"]["trainable_parameters"] == 1 * 8 * ((32 + 32) + 2 * (32 + 16) + (32 + 32))
    assert run["lora"]["adapter_dtype"] == "torch.float32"
    tm = run["timing"]
    for key in ("load_s", "train_loop_s", "teacher_scoring_s", "student_rollout_s", "forward_backward_s",
                "optimizer_step_s", "adapter_save_s"):
        assert key in tm
    assert (tm["student_rollout_s"] > 0) == (method == "on_policy")
    assert "peak_gpu_memory_bytes" in run


def test_off_and_on_share_schedule(runs):
    a = read_jsonl(runs["off_policy"]["dir"] / "presentations.jsonl.gz")
    b = read_jsonl(runs["on_policy"]["dir"] / "presentations.jsonl.gz")
    assert a[0]["source_id"] == b[0]["source_id"]


def test_adapter_trained(runs):
    from safetensors.torch import load_file

    w = load_file(str(runs["on_policy"]["adapter"] / "adapter_model.safetensors"))
    bs = [v for k, v in w.items() if "lora_B" in k]
    assert bs and any(float(v.abs().max()) > 0 for v in bs) and all(v.dtype == torch.float32 for v in bs)


def test_objective_pieces(tiny):
    model, _ = D.load_model(str(tiny["teacher"]), "cpu", "float32")
    prompt = D.chat_ids(tiny["tok"], "What is 2 + 2?")
    resp = [9, 17, 33, 151645]
    t = TR.teacher_logprobs(model, prompt, resp, 3)
    full = torch.log_softmax(hf_full_logits(model, prompt + resp), -1)
    assert t.shape == (3, VOCAB_SIZE) and t.dtype == torch.float32
    assert torch.allclose(t, full[len(prompt) - 1: len(prompt) + 2], atol=1e-5)
    kl_self = TR.kl_part(model, prompt, resp, 1, 2, t[1:3], autocast=False)
    assert kl_self.shape == (2,) and float(kl_self.abs().max()) < 1e-4
    other, _ = D.load_model(str(tiny["student"]), "cpu", "float32")
    assert float(TR.kl_part(other, prompt, resp, 0, 3, t, autocast=False).min()) > 0


def test_training_refusals(tiny, runs, collection, tmp_path):
    d = runs["on_policy"]["dir"]
    with pytest.raises(SystemExit):
        _train(tiny, "on_policy", d, tmp_path / "a1")
    with pytest.raises(SystemExit):
        _train(tiny, "on_policy", RESULTS_DIR / "task3" / "train" / "never", tmp_path / "a2")
    with pytest.raises(SystemExit):
        _train(tiny, "on_policy", tmp_path / "o3", RESULTS_DIR / "task3" / "never_adapter")
    with pytest.raises(SystemExit):
        _train(tiny, "off_policy", tmp_path / "o4", tmp_path / "a4")
    assert not (RESULTS_DIR / "task3" / "train" / "never").exists()


def test_heldout_guard_refuses(tiny, collection, tmp_path, monkeypatch):
    prompts, _ = D.heldout_prompts()
    real = D.training_items

    def leaky(limit_math=None, limit_short=None):
        items, prov = real(limit_math, limit_short)
        h = prompts[0]
        return items + [{"prompt_id": "GSM8K-train-9999", "domain": "math", "category": "math",
                         "prompt": h["prompt"], "gold": "", "family": None, "split": "train"}], prov

    monkeypatch.setattr(D, "training_items", leaky)
    with pytest.raises(SystemExit, match="held-out guard"):
        _train(tiny, "on_policy", tmp_path / "g1", tmp_path / "ga1")
    with pytest.raises(SystemExit, match="held-out guard"):
        C.main(["--out", str(tmp_path / "g.jsonl.gz"), "--teacher-path", str(tiny["teacher"]), *_dev(tiny),
                "--limit-k-short", "1"])
    assert not (tmp_path / "g1").exists() and not (tmp_path / "g.json").exists()


@pytest.fixture(scope="module")
def merged(tiny, runs):
    d = runs["off_policy"]
    out = tiny["root"] / "merge" / "off_policy_6304.json"
    md = tiny["root"] / "merged" / "off_policy_6304"
    rc = M.main(["--train-json", str(d["dir"] / "run.json"), "--adapter-dir", str(d["adapter"]),
                 "--merged-dir", str(md), "--out", str(out), "--backbone-path", str(tiny["student"]),
                 "--tokenizer-path", str(tiny["teacher"]), *_dev(tiny)])
    assert rc == 0
    return {"json": out, "dir": md}


def test_merge_outputs(tiny, runs, merged):
    run = json.loads(merged["json"].read_text())
    assert run["status"] == "ok" and run["config"]["run_name"] == "off_policy_6304"
    assert run["config"]["train_limited"] is True
    d20 = run["d20"]
    assert len(d20["per_prompt"]) == 8 and d20["max_abs_logit_diff"] < 1e-4
    assert [p["category"] for p in d20["per_prompt"]] == ["math"] * 4 + ["short"] * 4
    items, _ = D.training_items()
    want = [x["prompt_id"] for x in T.first_items(T.by_category(items), 6304)]
    assert run["config"]["d20_prompt_ids"] == want
    assert run["merged_files_sha256"] == T.dir_sha256(merged["dir"])
    assert "model.safetensors" in run["merged_files_sha256"] and "config.json" in run["merged_files_sha256"]
    assert run["timing"]["merge_s"] >= 0 and run["heldout_guard"]["heldout_file_opens"] == []
    from peft import PeftModel

    base, _ = D.load_model(str(tiny["student"]), "cpu", "float32")
    peft_model = PeftModel.from_pretrained(base, str(runs["off_policy"]["adapter"])).eval()
    mm, _ = D.load_model(str(merged["dir"]), "cpu", "float32")
    ids = D.chat_ids(tiny["tok"], items[0]["prompt"])
    with torch.no_grad():
        assert float((hf_full_logits(peft_model, ids) - hf_full_logits(mm, ids)).abs().max()) < 1e-4
    with pytest.raises(SystemExit):
        M.main(["--train-json", str(runs["off_policy"]["dir"] / "run.json"), "--adapter-dir",
                str(runs["on_policy"]["adapter"]), "--merged-dir", str(tiny["root"] / "m2"), "--out",
                str(tiny["root"] / "m2.json"), "--backbone-path", str(tiny["student"]), *_dev(tiny)])


def test_own_bank_from_merged(tiny, merged, tmp_path):
    out = tmp_path / "own_off_policy_6304.jsonl.gz"
    rc = B.main(["--kind", "own", "--run-name", "off_policy_6304", "--draft-path", str(merged["dir"]), "--out",
                 str(out), "--target-path", str(tiny["teacher"]), *_dev(tiny), "--limit-prompts", "1"])
    assert rc == 0
    recs = read_jsonl(out)
    seq = [r for r in recs if r["record_type"] == "sequence"]
    assert len(seq) == 1 and seq[0]["bank"] == "own_off_policy_6304"
    assert seq[0]["source_generation_seed"] == 6304 + 1001
    assert seq[0]["model_revision"]["draft"]["path"] == str(merged["dir"])
    run = json.loads(out.with_name("own_off_policy_6304.json").read_text())
    assert run["timing"]["load_s"] > 0 and len(run["timing"]["prompt_s"]) == 1
