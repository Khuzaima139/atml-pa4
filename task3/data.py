"""Task 3 prompt sets, chat formatting and model loading."""
import json
from pathlib import Path

from common.io import read_jsonl, sha256_file, sha256_gz_content
from common.paths import DATA_REPO_DIR, PUBLIC_ASSETS_LOCK, SOURCE_REVISIONS_LOCK

GSM8K_TEST_LOCKED = DATA_REPO_DIR / "data" / "public_locked" / "gsm8k_test_locked.jsonl.gz"
GSM8K_TRAIN_LOCKED = DATA_REPO_DIR / "data" / "public_locked" / "gsm8k_train_locked.jsonl.gz"
GSM8K_TEST_IDS = DATA_REPO_DIR / "data" / "public_manifests" / "gsm8k_test_ids.jsonl"
SHORT_TEST = DATA_REPO_DIR / "data" / "short_form" / "short_test.jsonl"
SHORT_TRAIN = DATA_REPO_DIR / "data" / "short_form" / "short_train.jsonl"
RELEASE_SUMS = DATA_REPO_DIR / "SHA256SUMS.json"


def revisions() -> dict:
    lock = json.loads(Path(SOURCE_REVISIONS_LOCK).read_text())
    return {key: {"repo": lock["repos"][key], "revision": lock["revisions"][key]}
            for key in ("model_target", "model_draft")}


def _check_locked_gz(path: Path) -> str:
    lock = json.loads(Path(PUBLIC_ASSETS_LOCK).read_text())
    want = lock["files"][path.name]["content_sha256"]
    got = sha256_gz_content(path)
    if got != want:
        raise RuntimeError(f"{path}: content sha256 {got} != data_lock {want}")
    return got


def _check_release_file(path: Path) -> str:
    sums = json.loads(Path(RELEASE_SUMS).read_text())
    rel = path.relative_to(DATA_REPO_DIR).as_posix()
    got = sha256_file(path)
    want = sums.get(rel)
    if want is None:
        raise RuntimeError(f"{rel} not listed in {RELEASE_SUMS}")
    if got != want:
        raise RuntimeError(f"{path}: sha256 {got} != release SHA256SUMS {want}")
    return got


def heldout_prompts():
    """The 120 held-out prompts in release order: 60 GSM8K test questions, then the 60 short_test inputs.

    Returns (prompts, provenance); each prompt is dict(prompt_id, domain, group, prompt).
    """
    manifest = read_jsonl(GSM8K_TEST_IDS)
    locked = {r["order"]: r for r in read_jsonl(GSM8K_TEST_LOCKED)}
    prompts = []
    for m in manifest:
        row = locked[m["order"]]
        if row["source_row_index"] != m["row_index"]:
            raise RuntimeError(f"GSM8K test order {m['order']}: row {row['source_row_index']} != {m['row_index']}")
        prompts.append({"prompt_id": f"GSM8K-{m['order']:04d}", "domain": "math", "group": "math",
                        "prompt": row["example"]["question"]})
    for x in read_jsonl(SHORT_TEST):
        prompts.append({"prompt_id": x["id"], "domain": "short", "group": x["split"], "prompt": x["input"]})
    prov = {
        "gsm8k_test_locked_content_sha256": _check_locked_gz(GSM8K_TEST_LOCKED),
        "gsm8k_test_ids_sha256": _check_release_file(GSM8K_TEST_IDS),
        "short_test_sha256": _check_release_file(SHORT_TEST),
    }
    return prompts, prov


def training_prompts_head(n_math=2, n_short=2):
    """First n_math GSM8K training questions (manifest order) and first n_short short-form training inputs."""
    rows = sorted(read_jsonl(GSM8K_TRAIN_LOCKED), key=lambda r: r["order"])[:n_math]
    out = [{"prompt_id": f"GSM8K-train-{r['order']:04d}", "domain": "math", "prompt": r["example"]["question"]}
           for r in rows]
    out += [{"prompt_id": x["id"], "domain": "short", "prompt": x["input"]} for x in read_jsonl(SHORT_TRAIN)[:n_short]]
    prov = {"gsm8k_train_locked_content_sha256": _check_locked_gz(GSM8K_TRAIN_LOCKED),
            "short_train_sha256": _check_release_file(SHORT_TRAIN)}
    return out, prov


def training_items(limit_math=None, limit_short=None):
    """Training prompts (protocol D11): all 500 GSM8K training questions in manifest order (GSM8K-train-{order:04d})
    and all 300 short-form training inputs in file order, duplicates included. limit_* keep the first n (tests only).

    Returns (items, provenance); each item is dict(prompt_id, domain, category, prompt, gold, family, split).
    """
    rows = sorted(read_jsonl(GSM8K_TRAIN_LOCKED), key=lambda r: r["order"])
    math = [{"prompt_id": f"GSM8K-train-{r['order']:04d}", "domain": "math", "category": "math",
             "prompt": r["example"]["question"], "gold": r["example"]["answer"], "family": None, "split": "train"}
            for r in rows]
    short = [{"prompt_id": x["id"], "domain": "short", "category": "short", "prompt": x["input"], "gold": x["answer"],
              "family": x["family"], "split": x["split"]} for x in read_jsonl(SHORT_TRAIN)]
    if len(math) != 500 or len(short) != 300:
        raise RuntimeError(f"training sets have {len(math)} / {len(short)} items, expected 500 / 300")
    if any(x["split"] != "train" for x in short):
        raise RuntimeError("short-form training file holds a non-training split")
    if limit_math is not None:
        math = math[:limit_math]
    if limit_short is not None:
        short = short[:limit_short]
    prov = {"gsm8k_train_locked_content_sha256": _check_locked_gz(GSM8K_TRAIN_LOCKED),
            "short_train_sha256": _check_release_file(SHORT_TRAIN)}
    return math + short, prov


def chat_ids(tokenizer, text) -> list:
    """Single user message, chat template, add_generation_prompt=True, no explicit system message (D4)."""
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def _dir_record(path: Path) -> dict:
    files = sorted(p for p in path.iterdir() if p.is_file() and (p.suffix in (".safetensors", ".json")))
    return {"source": "local", "path": str(path), "file_sha256": {p.name: sha256_file(p) for p in files}}


def load_model(key_or_path, device, dtype):
    """Load a causal LM from the locked hub revision ("model_target" / "model_draft") or a local directory.

    Returns (model, provenance record).
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    if key_or_path in ("model_target", "model_draft"):
        rv = revisions()[key_or_path]
        model = AutoModelForCausalLM.from_pretrained(rv["repo"], revision=rv["revision"], dtype=torch_dtype,
                                                     attn_implementation="sdpa")
        prov = {"source": "hub", **rv}
    else:
        path = Path(key_or_path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype, attn_implementation="sdpa")
        prov = _dir_record(path)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prov["dtype"] = dtype
    prov["vocab_size"] = int(model.config.vocab_size)
    return model, prov


def load_tokenizer(key_or_path):
    from transformers import AutoTokenizer

    if key_or_path in ("model_target", "model_draft"):
        rv = revisions()[key_or_path]
        return AutoTokenizer.from_pretrained(rv["repo"], revision=rv["revision"])
    return AutoTokenizer.from_pretrained(Path(key_or_path))
