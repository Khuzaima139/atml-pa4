"""Shared pieces of Task 3 distillation: settings fixed by the manual and the protocol, derived seeds, the training
schedule (D7), update sizes (D10), the LoRA student (D12) and the collection reader (D8)."""
import hashlib
import json
from pathlib import Path

import numpy as np

from common.io import read_jsonl, sha256_gz_content
from common.paths import RESULTS_DIR

DISTILL_ID = "t3-distill-v1"
COLLECTION_ID = "t3-collection-v1"
COLLECTION_SEED = 6304
TRAIN_SEEDS = (6304, 6305)
METHODS = ("off_policy", "on_policy")
CATEGORIES = ("math", "short")
QUOTA = {"math": 30_000, "short": 10_000}
CAPS = {"math": 192, "short": 32}
K = {"math": 1, "short": 16}
UPDATE_POSITIONS = 256
LR = 3e-4
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none", "task_type": "CAUSAL_LM"}
SCALER = {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000}
D20_N = 4

SETTINGS = {
    "distill_id": DISTILL_ID, "quota": QUOTA, "caps": CAPS, "update_positions": UPDATE_POSITIONS,
    "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "schedule": "constant (protocol AM1)"},
    "clip_global_norm": CLIP_NORM, "microbatch": 1, "lora": LORA, "grad_scaler": SCALER,
    "loss": "forward KL(p_teacher || q_student) over all 151,936 rows at every scored response position; summed per "
            "update and divided by that update's scored positions",
    "teacher_logprobs": "fp32 log_softmax over all rows, response positions only",
    "precision": "FP16 frozen backbones, FP32 LoRA adapters and AdamW states, FP16 autocast on CUDA, dynamic "
                 "gradient scaling, unscale before clipping",
}


def derive_seed(*parts) -> int:
    """Deterministic 63-bit seed from the parts (sha256 of their '|'-joined text)."""
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:15], 16)


def collection_seed(prompt_id, k) -> int:
    return derive_seed("t3-collect", COLLECTION_SEED, prompt_id, k)


def rollout_seed(train_seed, presentation_index) -> int:
    return derive_seed("t3-rollout", train_seed, presentation_index)


def update_sizes(total: int) -> list:
    """Scored positions per update: 256 each, the final update holds the remainder (protocol D10)."""
    sizes = [UPDATE_POSITIONS] * (total // UPDATE_POSITIONS)
    if total % UPDATE_POSITIONS:
        sizes.append(total % UPDATE_POSITIONS)
    return sizes


def permutation(n: int, seed: int) -> list:
    return [int(i) for i in np.random.default_rng(seed).permutation(n)]


class Schedule:
    """Protocol D7: per category a permutation drawn with the training seed, cycled; each presentation takes the next
    prompt of the category with the smaller fraction of its quota assigned (mathematics on ties); a category stops
    when its quota is met; the response that crosses a quota is scored only up to it (truncated)."""

    def __init__(self, items_by_cat: dict, seed: int, quota: dict):
        self.items = items_by_cat
        self.quota = dict(quota)
        self.order = {c: permutation(len(items_by_cat[c]), seed) for c in CATEGORIES}
        self.next_index = {c: 0 for c in CATEGORIES}
        self.assigned = {c: 0 for c in CATEGORIES}
        self.presented = {}

    def done(self) -> bool:
        return all(self.assigned[c] >= self.quota[c] for c in CATEGORIES)

    def next_category(self):
        open_ = [c for c in CATEGORIES if self.assigned[c] < self.quota[c]]
        if not open_:
            return None
        return min(open_, key=lambda c: (self.assigned[c] / self.quota[c], CATEGORIES.index(c)))

    def next_item(self, cat):
        """(item, n) with n the number of earlier presentations of that prompt in this run."""
        order = self.order[cat]
        item = self.items[cat][order[self.next_index[cat] % len(order)]]
        self.next_index[cat] += 1
        n = self.presented.get(item["prompt_id"], 0)
        self.presented[item["prompt_id"]] = n + 1
        return item, n

    def assign(self, cat, length: int) -> tuple:
        """(scored, truncated) for a response of `length` tokens."""
        remaining = self.quota[cat] - self.assigned[cat]
        scored = min(length, remaining)
        self.assigned[cat] += scored
        return scored, scored < length


def first_items(items_by_cat: dict, seed: int, n: int = D20_N) -> list:
    """First n prompts of each category in the seed's schedule order (protocol D20 uses seed 6304)."""
    out = []
    for c in CATEGORIES:
        order = permutation(len(items_by_cat[c]), seed)
        out += [items_by_cat[c][i] for i in order[:n]]
    return out


def by_category(items) -> dict:
    return {c: [x for x in items if x["category"] == c] for c in CATEGORIES}


def under_results(path) -> bool:
    return Path(path).resolve().is_relative_to(RESULTS_DIR.resolve())


def load_collection(path) -> dict:
    """Collection records keyed by (source_id, k), the run JSON and the content sha256 (checked against the run JSON)."""
    path = Path(path)
    run_json = path.with_name(path.name[: -len(".jsonl.gz")] + ".json")
    run = json.loads(run_json.read_text())
    if run.get("status") != "ok":
        raise SystemExit(f"collection run {run_json} status {run.get('status')!r}")
    sha = sha256_gz_content(path)
    if sha != run.get("output_content_sha256"):
        raise SystemExit(f"collection {path}: content sha256 {sha} != run JSON {run.get('output_content_sha256')}")
    recs = {}
    for r in read_jsonl(path):
        recs[(r["source_id"], r["k"])] = r
    return {"records": recs, "run": run, "content_sha256": sha, "path": str(path), "run_json": str(run_json)}


def add_lora(model, seed: int):
    """peft LoRA (D12) with default initialisation (B = 0) under torch seed = training seed; adapters in FP32."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(int(seed))
    cfg = LoraConfig(**LORA)
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in n:
                raise RuntimeError(f"unexpected trainable parameter {n}")
            p.data = p.data.float()
    return model


def trainable(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def dir_sha256(path) -> dict:
    from common.io import sha256_file

    return {p.name: sha256_file(p) for p in sorted(Path(path).iterdir()) if p.is_file()}
