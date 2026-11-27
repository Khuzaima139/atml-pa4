"""Part C entry points on CPU: python -m task2.kv_eval (validate, eval) and python -m task2.kv_bench, run end to end
with a tiny random Qwen2 (head_dim 128, 12 query heads, 2 KV heads, 2 layers, real vocabulary and tokenizer) on three
real retrieval prompts, plus unit checks of the shared pieces."""
import dataclasses
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common.io import sha256_file, sha256_gz_content
from common.paths import REPO_ROOT, RESULTS_DIR
from task2 import kv_bench as BN
from task2 import kv_budget as KB
from task2 import kv_cache_ops as KO
from task2 import kv_eval as EV
from task2 import kv_run as KR
from task2.kv_quant import PackedKVCache

PROMPTS = ["KV-2048-end-8-00", "KV-2048-start-0-00", "KV-4096-middle-8-00"]
LIMIT = ",".join(PROMPTS)
QUAL = ["KV-2048-end-8-00", "KV-4096-middle-8-00"]
REPS = (1, 2)
ENV = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")

RUN_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "environment",
           "kv_shape", "model", "model_facts", "row_files", "step", "tokenizer"}
COMMON_CONFIG = {"version", "scorer", "device", "model_id", "model_revision", "model_path", "dtype",
                 "attn_implementation", "limits", "limited", "output_files", "data", "seed", "generation"}
EVAL_COMMON_CONFIG = COMMON_CONFIG | {"mode", "n_prompts", "prompt_ids", "methods", "prompt_format",
                                      "teacher_forced_session_device"}
VALIDATE_CONFIG = EVAL_COMMON_CONFIG | {"cell_prompt_ids", "reference"}
EVAL_CONFIG = EVAL_COMMON_CONFIG | {"method_order_rule", "qualitative_ids", "npz_arrays", "npz_format_version",
                                    "qualitative_window_tokens"}
BENCH_CONFIG = COMMON_CONFIG | {"configs", "reference", "batches", "truncate", "warmup", "repetitions", "phases",
                                "order_rule", "prompt_ids_per_level", "statuses", "timing"}
VALIDATE_ROW = {"version", "scorer", "id", "prompt_index", "target_tokens", "position", "distractors", "context_len",
                "suffix_len", "total_len", "generate_token_ids", "pipeline_token_ids", "tokens_identical",
                "first_divergence", "n_steps_compared", "max_abs_logit_diff", "mean_abs_logit_diff",
                "generate_logits_finite", "pipeline_logits_finite", "prefill_logits_finite", "all_logits_finite",
                "positions_preserved", "physical_len", "next_pos", "n_appended", "generate_stop", "pipeline_stop",
                "generate_text", "pipeline_text", "generate_prediction", "generate_correct", "generate_no_match",
                "pipeline_prediction", "pipeline_correct", "generate_seconds", "pipeline_seconds",
                "full_forward_check"}
FF_ROW = {"version", "id", "target_tokens", "position", "distractors", "context_len", "suffix_len", "answer_len",
          "answer_token_ids", "split_prefill_decode", "pipeline_identity", "logits_finite", "eviction_mask_check"}
DIFF_KEYS = {"max_abs_diff", "mean_abs_diff", "argmax_agree", "n_rows"}
EVICT_KEYS = {"n_kept", "n_evicted", "suffix_max_abs_diff", "suffix_mean_abs_diff", "suffix_argmax_agree",
              "last_position_max_abs_diff", "positions_preserved", "finite"}
EVAL_ROW = {"version", "scorer", "id", "prompt_index", "method", "method_order", "method_slot", "target_tokens",
            "position", "distractors", "context_len", "suffix_len", "generated_token_ids", "generated_text",
            "n_generated", "stop", "hit_max_new_tokens", "prediction", "correct", "no_match", "answer_nll_mean",
            "answer_nll_per_token", "kl_post_prefill", "kl_teacher_forced_mean", "kl_teacher_forced_per_position",
            "k_recon_mean", "v_recon_mean", "k_recon_per_layer", "v_recon_per_layer", "evidence_token_count",
            "evidence_retained_count", "evidence_fraction_retained", "n_fp16_positions", "eviction_n", "hybrid_m",
            "hybrid_f", "M4", "M16", "formula_bytes", "context_bytes", "realized_fraction", "n_appended",
            "decode_time_bytes", "live_bytes_end", "total_live_bytes_end", "positions_preserved", "tail_rebuilds",
            "seconds", "peak_allocated_bytes", "start_allocated_bytes"}
PROMPT_ROW = {"version", "scorer", "id", "prompt_index", "target_tokens", "position", "distractors", "code",
              "context_len", "suffix_len", "total_len", "evidence_token_count", "method_order", "prefill_seconds",
              "scoring_seconds", "prefill_peak_allocated_bytes", "budget", "fp16_correct", "npz", "qualitative"}
BYTE_KEYS = {"payload", "metadata", "fp16_retained", "fp16_tail", "indices", "total"}
NPZ_KEYS = {"question_scores", "evidence_positions", "sink_recent_positions", "question_aware_positions",
            "hybrid_fp16_positions", "context_len", "suffix_len", "format_version", "version_ascii"}
BENCH_ROW = {"version", "L", "batch", "config", "status", "error", "cache_ids", "repeated_caches", "n_warmup",
             "n_reps", "times_ms", "stats_ms", "bytes", "bytes_per_sequence", "formula_bytes_per_sequence",
             "fp16_bytes_per_sequence", "realized_fraction", "hybrid_m", "hybrid_f", "recon_rel_error",
             "peak_allocated_bytes", "peak_reserved_bytes"}
STAT_KEYS = {"median", "mean", "min", "max", "std"}


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _files(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())


def _run(module, out, *args):
    cmd = [sys.executable, "-m", module, "--out", str(out), "--device", "cpu", *args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=ENV, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(KR.MODEL_ID, revision=KR.locked_revision(), local_files_only=True)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(6304)
    d = tmp_path_factory.mktemp("tiny_qwen2_kv")
    cfg = Qwen2Config(hidden_size=128, num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      head_dim=128, intermediate_size=256, vocab_size=151_936, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).to(torch.float32).save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def cases():
    return KR.load_cases()


@pytest.fixture(scope="module")
def validate_dir(tmp_path_factory, tiny_dir):
    out = tmp_path_factory.mktemp("kv_validate") / "out"
    return _run("task2.kv_eval", out, "--mode", "validate", "--limit-prompts", LIMIT, "--model-path", str(tiny_dir))


@pytest.fixture(scope="module")
def eval_dir(tmp_path_factory, tiny_dir):
    out = tmp_path_factory.mktemp("kv_eval") / "out"
    return _run("task2.kv_eval", out, "--mode", "eval", "--limit-prompts", LIMIT, "--model-path", str(tiny_dir))


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory, tiny_dir):
    out = tmp_path_factory.mktemp("kv_bench") / "out"
    return _run("task2.kv_bench", out, "--limit-prompts", LIMIT, "--model-path", str(tiny_dir),
                "--limit-reps", f"{REPS[0]},{REPS[1]}")


def _check_run(run, config_keys, extra_top=()):
    assert run["status"] == "ok" and run["step"] == "done"
    assert RUN_TOP | set(extra_top) == set(run)
    assert set(run["config"]) == config_keys
    c = run["config"]
    assert c["dtype"] == "float16" and c["attn_implementation"] == "sdpa" and c["limited"] is True
    assert c["model_id"] is None and c["model_revision"] is None and c["model_path"]
    assert c["data"]["sha256"] == c["data"]["manifest_sha256"] == sha256_file(KR.DATA_FILE)
    assert c["data"]["sha256_decompressed"] == sha256_gz_content(KR.DATA_FILE)
    assert run["model_facts"]["attn_implementation_effective"] == "sdpa"
    assert run["model_facts"]["parameter_dtypes"] == ["torch.float16"]
    assert run["kv_shape"] == {"n_layers": 2, "n_kv_heads": 2, "head_dim": 128, "channels": 1024}
    tok = run["tokenizer"]
    assert set(tok["files_sha256"]) == set(KR.TOKENIZER_FILES) and all(tok["files_sha256"].values())
    assert len(tok["chat_template_sha256"]) == 64 and tok["eos_ids_used"] == [151645, 151643]
    for k in ("transformers", "torch", "tokenizers", "huggingface_hub", "numpy"):
        assert run["meta"]["package_versions"][k]
    assert {"gpu_name", "gpu_capability", "seed"} <= set(run["environment"])


def _check_row_file(root, info, name):
    assert info["path"] == name and (root / name).exists()
    assert sha256_file(root / name) == info["sha256"]
    assert sha256_gz_content(root / name) == info["sha256_decompressed"]
    assert len(_rows(root / name)) == info["n_rows"]


# ---------------------------------------------------------------------------------------------------------------------
# Unit checks


def test_prepare_prompt_separates_gold(tokenizer, cases):
    case = next(c for c in cases if c["id"] == PROMPTS[0])
    inputs, gold = KR.prepare_prompt(tokenizer, case)
    assert not set(KR.INPUT_FIELDS) & {"code", "code_ids", "evidence_positions", "evidence_text",
                                      "evidence_char_range", "evidence_char_start", "evidence_char_end"}
    altered = dict(case, code="00000", evidence_char_start=0, evidence_char_end=5, evidence_text="xxxxx")
    inputs2, gold2 = KR.prepare_prompt(tokenizer, altered)
    assert inputs2 == inputs and gold2 != gold
    assert inputs.suffix_len == 33 and len(gold.evidence_positions) == 26 and len(gold.code_ids) == 5
    assert inputs.rendered[inputs.offsets[inputs.context_len][0]:].startswith(KR.DELIMITER)


def test_case_selection_and_orders(cases):
    assert [c["id"] for c in KR.select_cases(cases, "2")] == sorted(c["id"] for c in cases)[:2]
    assert [c["id"] for c in KR.select_cases(cases, LIMIT)] == sorted(PROMPTS)
    with pytest.raises(SystemExit):
        KR.select_cases(cases, "KV-9999-x")
    assert len(KR.qualitative_ids(cases)) == 6 and len(KR.first_per_cell(cases)) == 18
    assert EV.method_order(0) == list(KO.METHODS)
    assert EV.method_order(1) == list(KO.METHODS[1:] + KO.METHODS[:1])
    assert EV.method_order(6) == list(KO.METHODS)


def test_compare_steps():
    lg = torch.zeros(4, 3)
    assert EV.compare_steps([1, 2, 3], lg[:3], [1, 2, 3], lg[:3])["tokens_identical"]
    c = EV.compare_steps([1, 2, 3], lg[:3], [1, 5, 3], lg[:3] + 1)
    assert (c["tokens_identical"], c["first_divergence"], c["n_steps_compared"], c["max_abs_logit_diff"]) == \
        (False, 1, 2, 1.0)
    c = EV.compare_steps([1, 2], lg[:2], [1, 2, 3], lg[:3])
    assert (c["first_divergence"], c["n_steps_compared"]) == (2, 2)


def test_fp16_assertion_unit():
    t16 = torch.zeros(1, 2, 3, 4, dtype=torch.float16)
    KR.assert_fp16_cache([t16], [t16])
    with pytest.raises(KR.CacheDtypeError):
        KR.assert_fp16_cache([t16.float()], [t16])


def test_finalize_rows(tmp_path):
    from task2.rows import RowWriter

    p = tmp_path / "r.jsonl"
    with RowWriter(p) as w:
        w.write({"a": 1, "b": [1.5, None], "c": "x"})
        w.write({"a": 2})
    info = KR.finalize_rows(p)
    assert not p.exists() and info["n_rows"] == 2
    assert _rows(tmp_path / "r.jsonl.gz") == [{"a": 1, "b": [1.5, None], "c": "x"}, {"a": 2}]


def test_npz_writer_refuses_non_numeric(tmp_path):
    with pytest.raises(TypeError):
        KR.write_npz_new(tmp_path / "a.npz", {"x": np.array(["a"])})


def test_guarded_statuses():
    def oom():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    def unsup():
        raise NotImplementedError("no kernel")

    def bad():
        raise ValueError("bug")

    assert BN.guarded(lambda: 3) == ("ok", None, 3)
    assert BN.guarded(oom)[0] == "oom" and "out of memory" in BN.guarded(oom)[1]
    assert BN.guarded(unsup)[:2] == ("unsupported", "NotImplementedError: no kernel")
    with pytest.raises(ValueError):
        BN.guarded(bad)


def test_bench_packing_matches_packed_cache():
    g = torch.Generator().manual_seed(6304)
    L, H, d, n_layers = 2048 + 37, 2, 128, 2
    shape = KB.KVShape(n_layers, H, d)
    keys = [torch.randn(2, H, L, d, generator=g).half() for _ in range(n_layers)]
    values = [torch.randn(2, H, L, d, generator=g).half() for _ in range(n_layers)]
    one_k, one_v = [k[:1] for k in keys], [v[:1] for v in values]
    for cfg, layout in (("int4_token", "token"), ("int4_channel_k", "channel")):
        bp = BN.pack(cfg, one_k, one_v)
        ref = PackedKVCache.uniform(cfg, one_k, one_v, layout, 4)
        assert BN.nbytes(bp) == ref.nbytes() and BN.nbytes(bp)["total"] == KB.m4(shape, L)
        for (k, v), i in zip(BN.unpack(bp), range(n_layers)):
            rk, rv = ref.layer_tensors(i)
            assert torch.equal(k, rk) and torch.equal(v, rv)
    m, f = KB.hybrid_split(shape, L)
    scores = [torch.rand(L, generator=g) for _ in range(2)]
    fp = BN.select_hybrid(scores, L, f, "cpu")
    assert fp.shape == (2, f) and not torch.equal(fp[0], fp[1])
    bp = BN.pack("hybrid", keys, values, fp)
    assert BN.nbytes(bp)["total"] == 2 * KB.hybrid_bytes(shape, L, m)
    dec = BN.unpack(bp)
    for b in range(2):
        ref = PackedKVCache.hybrid([k[b:b + 1] for k in keys], [v[b:b + 1] for v in values], fp[b], 2)
        assert ref.nbytes()["total"] == KB.hybrid_bytes(shape, L, m)
        for i in range(n_layers):
            rk, rv = ref.layer_tensors(i)
            assert torch.equal(dec[i][0][b:b + 1], rk) and torch.equal(dec[i][1][b:b + 1], rv)


# ---------------------------------------------------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize("entry", ["validate", "eval", "bench"])
def test_refuses_existing_out(tmp_path, entry):
    out = tmp_path / "exists"
    out.mkdir()
    main, mode = (BN.main, []) if entry == "bench" else (EV.main, ["--mode", entry])
    with pytest.raises(SystemExit, match="output path exists"):
        main([*mode, "--out", str(out), "--device", "cpu"])
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("entry,opt", [("validate", ["--limit-prompts", "2"]), ("eval", ["--limit-prompts", "2"]),
                                       ("eval", ["--model-path", "x"]), ("bench", ["--limit-prompts", "2"]),
                                       ("bench", ["--limit-reps", "1,1"])])
def test_refuses_limits_under_results(entry, opt):
    out = RESULTS_DIR / "task2" / "kv_refusal_check_never_created"
    main, mode = (BN.main, []) if entry == "bench" else (EV.main, ["--mode", entry])
    with pytest.raises(SystemExit, match="refusing test-only options"):
        main([*mode, "--out", str(out), "--device", "cpu", *opt])
    assert not out.exists()


@pytest.mark.parametrize("entry", ["eval", "bench"])
def test_non_fp16_cache_is_refused(tmp_path, tiny_dir, monkeypatch, entry):
    orig = KR.load_model

    def load_fp32(device, model_path=None):
        model, info = orig(device, model_path)
        return model.float(), info

    monkeypatch.setattr(KR, "load_model", load_fp32)
    out = tmp_path / "out"
    args = ["--out", str(out), "--device", "cpu", "--limit-prompts", PROMPTS[0], "--model-path", str(tiny_dir)]
    with pytest.raises(KR.CacheDtypeError):
        if entry == "eval":
            EV.main(["--mode", "eval", *args])
        else:
            BN.main([*args, "--limit-reps", "0,1"])
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and "float16" in run["error"]
    assert not list(out.glob("*.gz")) and not (out / "npz").exists()


def test_gold_enters_only_after_generation(tmp_path, tiny_dir, tokenizer, cases, monkeypatch):
    case = next(c for c in cases if c["id"] == PROMPTS[0])
    _, gold = KR.prepare_prompt(tokenizer, case)
    forbidden = [gold.code, gold.evidence_text, list(gold.code_ids), tuple(gold.code_ids),
                 list(gold.evidence_positions), tuple(gold.evidence_positions)]
    ev_t = torch.tensor(gold.evidence_positions)
    log = []

    def carries_gold(x, depth=0):
        if depth > 4:
            return False
        if isinstance(x, KR.Gold):
            return True
        if isinstance(x, (str, list, tuple)) and any(type(x) is type(f) and x == f for f in forbidden):
            return True
        if isinstance(x, torch.Tensor):
            return x.dim() == 1 and x.numel() == ev_t.numel() and torch.equal(x.cpu().long(), ev_t)
        if isinstance(x, (list, tuple)):
            return any(carries_gold(e, depth + 1) for e in x[:64])
        if isinstance(x, dict):
            return any(carries_gold(e, depth + 1) for e in x.values())
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return any(carries_gold(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x))
        return False

    def spy(mod, name):
        fn = getattr(mod, name)

        def wrapped(*a, **k):
            log.append((name, any(carries_gold(v) for v in list(a) + list(k.values()))))
            return fn(*a, **k)

        monkeypatch.setattr(mod, name, wrapped)

    for name in ("full_prefill", "score_prefill", "compress", "run_method", "greedy_decode", "teacher_forced_logits"):
        spy(KO, name)
    spy(KR, "assert_fp16_cache")
    spy(EV, "generation_stage")
    out = tmp_path / "out"
    EV.main(["--mode", "eval", "--out", str(out), "--device", "cpu", "--limit-prompts", PROMPTS[0],
             "--model-path", str(tiny_dir)])
    names = [n for n, _ in log]
    first_tf = names.index("teacher_forced_logits")
    before = log[:first_tf]
    assert {"full_prefill", "score_prefill", "compress", "run_method", "greedy_decode", "assert_fp16_cache",
            "generation_stage"} <= {n for n, _ in before}
    assert not any(g for _, g in before)
    assert names.count("run_method") == 6 and max(i for i, n in enumerate(names) if n == "run_method") < first_tf
    assert all(g for n, g in log if n == "teacher_forced_logits") and names.count("teacher_forced_logits") == 6
    assert set(names[first_tf:]) == {"teacher_forced_logits"}


# ---------------------------------------------------------------------------------------------------------------------
# validate end to end


def test_validate_outputs(validate_dir, cases):
    root = validate_dir
    assert _files(root) == ["full_forward_rows.jsonl.gz", "run.json", "validate_rows.jsonl.gz"]
    run = json.loads((root / "run.json").read_text())
    _check_run(run, VALIDATE_CONFIG, extra_top=("summary", "prompts_done", "effective_generation_config"))
    c = run["config"]
    assert c["mode"] == "validate" and c["prompt_ids"] == sorted(PROMPTS) and c["cell_prompt_ids"] == sorted(PROMPTS)
    assert c["generation"]["settings"] == KR.GREEDY
    eff = run["effective_generation_config"]["settings"]
    assert eff["do_sample"] is False and eff["repetition_penalty"] == 1.0 and eff["max_new_tokens"] == 32
    assert eff["eos_token_id"] == [151645, 151643] and eff["top_k"] is None and eff["temperature"] is None
    _check_row_file(root, run["row_files"]["rows"], "validate_rows.jsonl.gz")
    _check_row_file(root, run["row_files"]["full_forward"], "full_forward_rows.jsonl.gz")
    rows = _rows(root / "validate_rows.jsonl.gz")
    assert [r["id"] for r in rows] == sorted(PROMPTS)
    for i, r in enumerate(rows):
        assert set(r) == VALIDATE_ROW and r["prompt_index"] == i and r["version"] == KR.VERSION
        assert r["tokens_identical"] and r["first_divergence"] is None and r["generate_token_ids"]
        assert r["generate_token_ids"] == r["pipeline_token_ids"] and r["n_steps_compared"] == len(
            r["generate_token_ids"])
        assert r["max_abs_logit_diff"] < 0.05 and r["all_logits_finite"] and r["positions_preserved"]
        assert r["next_pos"] == r["physical_len"] == r["context_len"] + r["n_appended"]
        assert r["n_appended"] == r["suffix_len"] + len(r["pipeline_token_ids"]) - 1
        assert r["generate_correct"] is False and r["full_forward_check"] is True
    ff = _rows(root / "full_forward_rows.jsonl.gz")
    assert [r["id"] for r in ff] == sorted(PROMPTS)
    for r in ff:
        assert set(r) == FF_ROW and r["logits_finite"]
        assert r["answer_token_ids"] == next(v["generate_token_ids"] for v in rows if v["id"] == r["id"])
        for k in ("split_prefill_decode", "pipeline_identity"):
            assert set(r[k]) == DIFF_KEYS and r[k]["max_abs_diff"] < 0.05 and r[k]["n_rows"] == r["answer_len"]
        assert set(r["eviction_mask_check"]) == {"sink_recent", "question_aware"}
        for m, e in r["eviction_mask_check"].items():
            assert set(e) == EVICT_KEYS and e["positions_preserved"] and e["finite"]
            assert e["suffix_max_abs_diff"] < 0.05 and e["n_kept"] + e["n_evicted"] == r["context_len"]
    s = run["summary"]
    assert s["n_prompts"] == 3 and s["n_tokens_identical"] == 3 and s["n_positions_preserved"] == 3
    assert set(s["generate_accuracy_by_cell"]) == {"2048-end-8", "2048-start-0", "4096-middle-8"}
    assert s["full_forward"]["n_prompts"] == 3


# ---------------------------------------------------------------------------------------------------------------------
# eval end to end


def test_eval_outputs(eval_dir, tokenizer, cases):
    root = eval_dir
    expected = sorted(["eval_rows.jsonl.gz", "prompt_rows.jsonl.gz", "run.json"]
                      + [f"npz/{p}.npz" for p in PROMPTS] + [f"qualitative/{p}.txt" for p in QUAL])
    assert _files(root) == expected
    run = json.loads((root / "run.json").read_text())
    _check_run(run, EVAL_CONFIG, extra_top=("summary", "prompts_done"))
    assert run["config"]["qualitative_ids"] == QUAL
    _check_row_file(root, run["row_files"]["rows"], "eval_rows.jsonl.gz")
    _check_row_file(root, run["row_files"]["prompts"], "prompt_rows.jsonl.gz")
    rows = _rows(root / "eval_rows.jsonl.gz")
    prompts = _rows(root / "prompt_rows.jsonl.gz")
    assert len(rows) == 18 and [p["id"] for p in prompts] == sorted(PROMPTS)
    shape = KB.KVShape(2, 2, 128)
    by_case = {c["id"]: c for c in cases}
    for idx, p in enumerate(prompts):
        assert set(p) == PROMPT_ROW and p["prompt_index"] == idx
        order = EV.method_order(idx)
        assert p["method_order"] == order == run["summary"]["method_orders"][p["id"]]
        prow = [r for r in rows if r["id"] == p["id"]]
        assert [r["method"] for r in prow] == order  # written in the rotated order
        assert p["code"] == by_case[p["id"]]["code"] and p["budget"] == KB.budget_table(shape, p["context_len"])
        assert p["npz"] == f"npz/{p['id']}.npz"
        assert p["qualitative"] == (f"qualitative/{p['id']}.txt" if p["id"] in QUAL else None)
        _, gold = KR.prepare_prompt(tokenizer, by_case[p["id"]])
        rm = {r["method"]: r for r in prow}
        assert p["fp16_correct"] == rm["fp16"]["correct"]
        for m, r in rm.items():
            assert set(r) == EVAL_ROW and r["method_slot"] == order.index(m) and r["method_order"] == order
            assert set(r["context_bytes"]) == BYTE_KEYS and set(r["live_bytes_end"]) == BYTE_KEYS
            assert set(r["seconds"]) == {"compression", "suffix", "decode", "teacher_forced"}
            assert r["context_bytes"]["total"] == r["formula_bytes"]
            assert r["M16"] == KB.m16(shape, r["context_len"]) and r["M4"] == KB.m4(shape, r["context_len"])
            assert r["realized_fraction"] == r["context_bytes"]["total"] / r["M16"]
            assert r["decode_time_bytes"] == shape.fp16_bytes_per_position * r["n_appended"]
            assert r["n_appended"] == r["suffix_len"] + r["n_generated"] - 1
            assert r["positions_preserved"] and len(r["answer_nll_per_token"]) == 5
            assert len(r["kl_teacher_forced_per_position"]) == 5 and r["evidence_token_count"] == len(
                gold.evidence_positions)
            assert r["prediction"] is None or len(r["prediction"]) == 5
            assert r["hit_max_new_tokens"] == (r["stop"] == "max_new_tokens")
            if m in ("sink_recent", "question_aware", "hybrid"):
                assert 0.0 <= r["evidence_fraction_retained"] <= 1.0 and r["n_fp16_positions"] > 0
            else:
                assert r["evidence_fraction_retained"] is None and r["evidence_retained_count"] is None
            if m in ("sink_recent", "question_aware"):
                assert r["k_recon_mean"] is None and r["context_bytes"]["total"] <= r["M4"]
            else:
                assert len(r["k_recon_per_layer"]) == 2 and len(r["v_recon_per_layer"]) == 2
        assert rm["fp16"]["kl_post_prefill"] == 0.0 and rm["fp16"]["kl_teacher_forced_mean"] == 0.0
        assert rm["fp16"]["k_recon_mean"] == 0.0 and rm["fp16"]["realized_fraction"] == 1.0
        for m in ("int4_token", "int4_channel_k"):
            assert rm[m]["context_bytes"]["total"] == rm[m]["M4"] and rm[m]["k_recon_mean"] > 0
        assert rm["int4_token"]["v_recon_mean"] == rm["int4_channel_k"]["v_recon_mean"]
        assert rm["hybrid"]["hybrid_m"] >= 1 and rm["hybrid"]["hybrid_f"] >= 72
        assert rm["hybrid"]["context_bytes"]["total"] <= rm["hybrid"]["M4"]
        assert rm["sink_recent"]["context_bytes"]["indices"] == 8
        assert rm["question_aware"]["context_bytes"]["indices"] == 4 * rm["question_aware"]["eviction_n"]
        # npz
        with np.load(root / p["npz"], allow_pickle=False) as z:
            assert set(z.files) == NPZ_KEYS
            arr = {k: z[k] for k in z.files}
        for k, dt in EV.NPZ_ARRAYS.items():
            assert arr[k].dtype == np.dtype(dt)
        assert arr["question_scores"].shape == (p["context_len"],) and int(arr["context_len"]) == p["context_len"]
        assert arr["evidence_positions"].tolist() == list(gold.evidence_positions)
        assert arr["version_ascii"].tobytes().decode("ascii") == KR.VERSION
        sr, qa, hy = (arr[k].tolist() for k in ("sink_recent_positions", "question_aware_positions",
                                                 "hybrid_fp16_positions"))
        n = rm["sink_recent"]["eviction_n"]
        assert sr == list(range(8)) + list(range(p["context_len"] - (n - 8), p["context_len"]))
        assert len(qa) == rm["question_aware"]["eviction_n"] and len(hy) == rm["hybrid"]["hybrid_f"]
        assert set(hy) <= set(qa) and set(range(8)) <= set(hy) and sorted(qa) == qa
        ev = set(gold.evidence_positions)
        assert rm["question_aware"]["evidence_retained_count"] == len(ev & set(qa))
        assert rm["hybrid"]["evidence_retained_count"] == len(ev & set(hy))
    for q in QUAL:
        text = (root / "qualitative" / f"{q}.txt").read_text(encoding="utf-8")
        case = by_case[q]
        assert case["code"] in text and case["evidence_text"] in text and "token-selection mask" in text
        assert all(m in text for m in KO.METHODS)
        assert sum(1 for line in text.splitlines() if line[:6].strip().isdigit() and line[7:8] == "E") == len(
            KR.prepare_prompt(tokenizer, case)[1].evidence_positions)
    s = run["summary"]
    assert s["n_rows"] == 18 and set(s["per_method"]) == set(KO.METHODS)
    assert all(v["n"] == 3 and v["n_positions_preserved"] == 3 for v in s["per_method"].values())


# ---------------------------------------------------------------------------------------------------------------------
# bench end to end


def test_bench_outputs(bench_dir):
    root = bench_dir
    assert _files(root) == ["bench_rows.jsonl.gz", "run.json"]
    run = json.loads((root / "run.json").read_text())
    _check_run(run, BENCH_CONFIG, extra_top=("block_orders", "n_rows"))
    c = run["config"]
    assert (c["warmup"], c["repetitions"]) == REPS and c["batches"] == [1, 8]
    assert c["prompt_ids_per_level"] == {"2048": sorted(PROMPTS[:2]), "4096": [PROMPTS[2]]}
    _check_row_file(root, run["row_files"]["rows"], "bench_rows.jsonl.gz")
    rows = _rows(root / "bench_rows.jsonl.gz")
    assert len(rows) == run["n_rows"] == 16
    shape = KB.KVShape(2, 2, 128)
    seen = set()
    for r in rows:
        assert set(r) == BENCH_ROW and r["status"] == "ok" and r["error"] is None
        seen.add((r["L"], r["batch"], r["config"]))
        B, L = r["batch"], r["L"]
        assert set(r["times_ms"]) == set(BN.PHASES[r["config"]])
        assert all(len(v) == REPS[1] and all(t > 0 for t in v) for v in r["times_ms"].values())
        assert all(set(s) == STAT_KEYS for s in r["stats_ms"].values())
        assert set(r["bytes"]) == BYTE_KEYS and r["bytes"]["total"] == B * r["formula_bytes_per_sequence"]
        assert r["fp16_bytes_per_sequence"] == KB.m16(shape, L)
        assert len(r["cache_ids"]) == B and r["repeated_caches"] == (len(set(r["cache_ids"])) < B)
        if r["config"] in ("int4_token", "int4_channel_k"):
            assert r["formula_bytes_per_sequence"] == KB.m4(shape, L) and r["recon_rel_error"]["k_mean"] > 0
        elif r["config"] == "hybrid":
            m, f = KB.hybrid_split(shape, L)
            assert (r["hybrid_m"], r["hybrid_f"]) == (m, f) and r["bytes"]["indices"] == 4 * f * B
        else:
            assert r["bytes"]["fp16_retained"] == B * KB.m16(shape, L) and r["recon_rel_error"] is None
    assert seen == {(L, B, cfg) for L in (2048, 4096) for B in (1, 8) for cfg in BN.ALL_CONFIGS}
    assert set(run["block_orders"]) == {"L2048_B1", "L2048_B8", "L4096_B1", "L4096_B8"}
    for blocks in run["block_orders"].values():
        assert [b["kind"] for b in blocks] == ["warmup"] * REPS[0] + ["timed"] * REPS[1]
        assert [b["order"] for b in blocks] == [BN.rotate(BN.ALL_CONFIGS, i) for i in range(sum(REPS))]
