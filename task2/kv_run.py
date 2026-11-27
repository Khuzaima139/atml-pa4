"""Shared pieces of the Part C entry points (task2.kv_eval, task2.kv_bench).

Model: Qwen/Qwen2.5-1.5B-Instruct at the revision locked in data_lock/local_source_revisions.json (model_target),
loaded in FP16 with attn_implementation "sdpa" (both asserted). Prompts: the supplied retrieval file
external/atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz, checked against the data repository's
SHA256SUMS.json, rendered as one user message with the supplied chat template (default system message included) and
add_generation_prompt=True.

Prompt split. The context is every token before the delimiter line "QUESTION_BEGIN" (which occurs once and starts a
token); the question suffix is the rest of the rendered prompt. The split uses only the prompt text.

Gold separation. prepare_prompt returns (PromptInputs, Gold). PromptInputs holds the token ids, the split and the
cell labels; Gold holds the code, its token ids and the evidence token positions. The generation stage of each entry
point receives PromptInputs only; Gold is used after decoding (teacher-forced pass and metrics).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from common.io import read_jsonl, sha256_file, sha256_gz_content, write_jsonl_gz
from common.paths import DATA_REPO_DIR, RESULTS_DIR, SOURCE_REVISIONS_LOCK

from . import kv_cache_ops as KO
from . import kv_metrics as KM
from .mqar_data import manifest_sha

VERSION = "task2-part-c-v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECTED_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DTYPE = torch.float16
ATTN_IMPLEMENTATION = "sdpa"
DATA_REL = "data/kv_retrieval/kv_retrieval_cases.jsonl.gz"
DATA_FILE = DATA_REPO_DIR / DATA_REL
N_CASES = 180
DELIMITER = "QUESTION_BEGIN"
REAL_ROOT = RESULTS_DIR / "task2"
SEED = 6304
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
LEVELS = (2048, 4096)
POSITIONS = ("start", "middle", "end")
DISTRACTORS = (0, 2, 8)
PAD_ID = 151643

# Settings of every greedy decode (protocol C6). The Part C pipeline decodes with an explicit argmax loop using
# exactly these settings; C1 passes them to model.generate as a GenerationConfig with use_model_defaults=False.
GREEDY = {"do_sample": False, "repetition_penalty": 1.0, "temperature": None, "top_p": None, "top_k": None,
          "max_new_tokens": KO.MAX_NEW_TOKENS, "eos_token_id": list(KO.EOS_IDS), "pad_token_id": PAD_ID,
          "num_beams": 1}


def locked_revision() -> str:
    lock = json.loads(Path(SOURCE_REVISIONS_LOCK).read_text())
    assert lock["repos"]["model_target"] == MODEL_ID, lock["repos"]["model_target"]
    rev = lock["revisions"]["model_target"]
    assert rev == EXPECTED_REVISION, rev
    return rev


# ---------------------------------------------------------------------------------------------------------------------
# Arguments


def under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_out(out, limits: dict) -> bool:
    """Refuse test-only options with an output under results/task2, and an existing output path. Returns limited."""
    limited = any(v is not None for v in limits.values())
    if limited and under(out, REAL_ROOT):
        raise SystemExit(f"refusing test-only options {sorted(k for k, v in limits.items() if v is not None)} "
                         f"with an output path under {REAL_ROOT}: {out}")
    if Path(out).exists():
        raise SystemExit(f"output path exists: {out}")
    return limited


def select_cases(cases: list, spec: str | None) -> list:
    """All cases (sorted by id), or a test subset: an integer N (the first N ids) or a comma-separated id list."""
    if spec is None:
        return cases
    spec = spec.strip()
    if spec.isdigit():
        n = int(spec)
        if n < 1:
            raise SystemExit("--limit-prompts needs N >= 1")
        return cases[:n]
    by_id = {c["id"]: c for c in cases}
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [i for i in ids if i not in by_id]
    if missing or not ids:
        raise SystemExit(f"--limit-prompts: unknown ids {missing}")
    return [by_id[i] for i in sorted(set(ids))]


# ---------------------------------------------------------------------------------------------------------------------
# Data


def data_record(path=DATA_FILE) -> dict:
    path = Path(path)
    return {"path": str(path), "rel": DATA_REL, "sha256": sha256_file(path),
            "sha256_decompressed": sha256_gz_content(path), "manifest_sha256": manifest_sha(DATA_REL)}


def load_cases(path=DATA_FILE) -> list:
    """The 180 retrieval cases sorted by id, after the file sha256 is checked against SHA256SUMS.json."""
    rec = data_record(path)
    if rec["sha256"] != rec["manifest_sha256"]:
        raise RuntimeError(f"{path}: sha256 {rec['sha256']} != manifest {rec['manifest_sha256']}")
    cases = sorted(read_jsonl(path), key=lambda c: c["id"])
    if len(cases) != N_CASES or len({c["id"] for c in cases}) != N_CASES:
        raise RuntimeError(f"{path}: expected {N_CASES} distinct cases, got {len(cases)}")
    return cases


def cell_of(case: dict) -> dict:
    return {"target_tokens": int(case["target_tokens"]), "position": case["position"],
            "distractors": int(case["distractors"])}


def cell_key(case: dict) -> str:
    c = cell_of(case)
    return f"{c['target_tokens']}-{c['position']}-{c['distractors']}"


def qualitative_ids(cases: list) -> list:
    """Protocol AM2.9: per context length, the lowest id in the start, middle and end cells with 8 distractors."""
    out = []
    for level in LEVELS:
        for pos in POSITIONS:
            ids = sorted(c["id"] for c in cases
                         if int(c["target_tokens"]) == level and c["position"] == pos and int(c["distractors"]) == 8)
            if ids:
                out.append(ids[0])
    return out


def first_per_cell(cases: list) -> list:
    """The lowest id in each grid cell present in cases."""
    best = {}
    for c in cases:
        k = cell_key(c)
        if k not in best or c["id"] < best[k]:
            best[k] = c["id"]
    return sorted(best.values())


@dataclass(frozen=True)
class PromptInputs:
    """Everything the generation stage may see: token ids, the context/suffix split and the cell labels."""

    id: str
    target_tokens: int
    position: str
    distractors: int
    input_ids: tuple
    context_len: int
    suffix_len: int
    offsets: tuple  # (start, end) character range of each token in the rendered prompt
    rendered: str

    def ids_tensor(self, device) -> torch.Tensor:
        return torch.tensor([list(self.input_ids)], dtype=torch.long, device=device)

    @property
    def total_len(self) -> int:
        return len(self.input_ids)

    def cell(self) -> dict:
        return {"target_tokens": self.target_tokens, "position": self.position, "distractors": self.distractors}


@dataclass(frozen=True)
class Gold:
    """Evaluation-only fields: the code, its token ids and the evidence token positions (protocol C4)."""

    code: str
    code_ids: tuple
    evidence_positions: tuple
    evidence_text: str
    evidence_char_range: tuple  # in the rendered prompt


GOLD_FIELDS = tuple(f.name for f in fields(Gold))
INPUT_FIELDS = tuple(f.name for f in fields(PromptInputs))


def render(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True,
                                         tokenize=False)


def prepare_prompt(tokenizer, case: dict) -> tuple[PromptInputs, Gold]:
    """Tokenize one case; the split and the ids depend only on the prompt text."""
    rendered = render(tokenizer, case["user_text"])
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    direct = tokenizer.apply_chat_template([{"role": "user", "content": case["user_text"]}],
                                           add_generation_prompt=True, tokenize=True)
    if list(direct) != ids:
        raise RuntimeError(f"{case['id']}: chat-template ids differ from tokenizing the rendered prompt")
    offsets = tuple((int(a), int(b)) for a, b in enc["offset_mapping"])
    if rendered.count(DELIMITER) != 1:
        raise RuntimeError(f"{case['id']}: delimiter count {rendered.count(DELIMITER)} != 1")
    b = rendered.index(DELIMITER)
    L = sum(1 for a, _ in offsets if a < b)
    if L >= len(ids) or offsets[L][0] != b:
        raise RuntimeError(f"{case['id']}: the delimiter does not start a token")
    inputs = PromptInputs(case["id"], int(case["target_tokens"]), case["position"], int(case["distractors"]),
                          tuple(ids), L, len(ids) - L, offsets, rendered)
    # evaluation-only fields
    if rendered.count(case["user_text"]) != 1:
        raise RuntimeError(f"{case['id']}: user text not found once in the rendered prompt")
    u0 = rendered.index(case["user_text"])
    a, e = u0 + int(case["evidence_char_start"]), u0 + int(case["evidence_char_end"])
    ev = KM.evidence_tokens(offsets, a, e)
    if not ev or max(ev) >= L:
        raise RuntimeError(f"{case['id']}: evidence tokens {ev[:3]}... not inside the context (L={L})")
    code_ids = tuple(tokenizer(case["code"], add_special_tokens=False)["input_ids"])
    if len(code_ids) != len(case["code"]):
        raise RuntimeError(f"{case['id']}: code {case['code']} is not one token per digit")
    gold = Gold(case["code"], code_ids, tuple(ev), case["evidence_text"], (a, e))
    return inputs, gold


# ---------------------------------------------------------------------------------------------------------------------
# Model and tokenizer


def load_tokenizer(model_path=None):
    from transformers import AutoTokenizer

    if model_path is not None:
        return AutoTokenizer.from_pretrained(model_path)
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=locked_revision())


def load_model(device, model_path=None):
    """FP16 causal LM with SDPA, eval mode, no gradients. Returns (model, info)."""
    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": DTYPE, "attn_implementation": ATTN_IMPLEMENTATION}
    if model_path is None:
        rev = locked_revision()
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=rev, **kwargs)
        source = {"model_id": MODEL_ID, "revision": rev, "model_path": None}
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        source = {"model_id": None, "revision": None, "model_path": str(model_path)}
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    assert dtypes == [str(DTYPE)], dtypes
    info = {**source, "parameter_dtypes": dtypes, "model_class": type(model).__name__,
            "attn_implementation_requested": ATTN_IMPLEMENTATION, "device": str(next(model.parameters()).device)}
    return model, info


def check_model(model) -> dict:
    """Assert SDPA attention and return the effective model facts recorded in run.json."""
    attn = model.config._attn_implementation
    if attn != ATTN_IMPLEMENTATION:
        raise RuntimeError(f"attention implementation {attn!r} != {ATTN_IMPLEMENTATION!r}")
    cfg = model.config
    return {"attn_implementation_effective": attn,
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "num_hidden_layers": cfg.num_hidden_layers, "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
            "hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size,
            "rope_theta": getattr(cfg, "rope_theta", None),
            "model_generation_config_file_values": model.generation_config.to_diff_dict()}


def _tokenizer_file(name, model_path):
    if model_path is not None:
        p = Path(model_path) / name
        return p if p.exists() else None
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(MODEL_ID, name, revision=locked_revision()))
    except Exception:  # a file the repository does not have
        return None


def tokenizer_meta(tokenizer, model_path=None) -> dict:
    files = {}
    for name in TOKENIZER_FILES:
        p = _tokenizer_file(name, model_path)
        files[name] = sha256_file(p) if p is not None else None
    template = tokenizer.chat_template or ""
    eos = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    if tuple(eos) != tuple(KO.EOS_IDS):
        raise RuntimeError(f"tokenizer EOS ids {eos} != {KO.EOS_IDS}")
    return {"tokenizer_class": type(tokenizer).__name__, "files_sha256": files,
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token, "eos_ids_used": list(KO.EOS_IDS)}


class CacheDtypeError(RuntimeError):
    pass


def assert_fp16_cache(keys: list, values: list) -> None:
    """Refuse a cache that is not FP16 before any byte is counted."""
    bad = sorted({str(t.dtype) for t in list(keys) + list(values) if t.dtype != DTYPE})
    if bad:
        raise CacheDtypeError(f"cache dtype {bad} is not float16; bytes are counted only for an FP16 cache")


# ---------------------------------------------------------------------------------------------------------------------
# Outputs


def finalize_rows(plain: Path) -> dict:
    """Compress a finished plain JSONL row file to <name>.gz (mtime 0), check that the decompressed bytes equal the
    plain file, then remove the plain file. Returns the gz path, its sha256 and the row count."""
    plain = Path(plain)
    rows = read_jsonl(plain)
    gz = plain.with_name(plain.name + ".gz")
    write_jsonl_gz(gz, rows)
    if sha256_gz_content(gz) != sha256_file(plain):
        raise RuntimeError(f"{gz}: decompressed content differs from {plain}")
    os.remove(plain)
    return {"path": gz.name, "sha256": sha256_file(gz), "sha256_decompressed": sha256_gz_content(gz),
            "n_rows": len(rows)}


def write_npz_new(path: Path, arrays: dict) -> Path:
    """Numeric arrays only, to a new path; readable with np.load(allow_pickle=False)."""
    path = Path(path)
    for k, a in arrays.items():
        if not isinstance(a, np.ndarray) or a.dtype.kind not in "biuf":
            raise TypeError(f"{k}: only numeric numpy arrays are written, got {type(a).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    return path


def write_text_new(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(text)
    return path


def is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def sync(device) -> None:
    if is_cuda(device):
        torch.cuda.synchronize(device)


def reset_peak(device) -> None:
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def peak_allocated(device):
    return int(torch.cuda.max_memory_allocated(device)) if is_cuda(device) else None


def peak_reserved(device):
    return int(torch.cuda.max_memory_reserved(device)) if is_cuda(device) else None


def environment(device) -> dict:
    from .part_a import environment as part_a_environment

    env = part_a_environment(device)
    env["seed"] = SEED
    return env
