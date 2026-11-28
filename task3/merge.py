"""Merge a trained LoRA adapter into a fresh FP16 0.5B backbone and save the merged draft (protocol D12, D15, D20).

Loads a fresh backbone (the locked original student unless --backbone-path is given), attaches the adapter
(unmerged), computes the logits of the D20 prompts (the first 4 mathematics and first 4 short-form prompts of the
seed-6304 schedule), merges with merge_and_unload (timed), computes the same logits with the merged model, records
the max abs logit difference per prompt and overall, and saves the merged draft (safetensors) to --merged-dir
(outside results/). The training run JSON is checked for status ok and for the adapter sha256.

    python -m task3.merge --train-json results/task3/train/on_policy_6304/run.json \
        --adapter-dir /kaggle/working/t3_artifacts/adapters/on_policy_6304 \
        --merged-dir /kaggle/working/t3_artifacts/merged/on_policy_6304 --out results/task3/merge/on_policy_6304.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import data as D
from . import distill as T
from .heldout import HeldoutGuard
from .lm import hf_full_logits
from .sampling import VOCAB_SIZE

MERGE_ID = "t3-merge-v1"
DONE = "merge done: "
D20_SEED = 6304


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--out", required=True, help="merge record JSON (must not exist)")
    ap.add_argument("--backbone-path", default="model_draft")
    ap.add_argument("--tokenizer-path", default="model_target")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args(argv)
    out, merged_dir, adapter_dir = Path(a.out), Path(a.merged_dir), Path(a.adapter_dir)
    for p in (out, merged_dir):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    if T.under_results(merged_dir):
        raise SystemExit("--merged-dir must be outside results/")
    train = json.loads(Path(a.train_json).read_text())
    if train.get("status") != "ok":
        raise SystemExit(f"training run {a.train_json} status {train.get('status')!r}")
    adapter_sha = T.dir_sha256(adapter_dir)
    if adapter_sha != train.get("adapter_sha256"):
        raise SystemExit(f"adapter files in {adapter_dir} differ from the sha256 recorded in {a.train_json}")

    seed_everything(D20_SEED)
    guard = HeldoutGuard()
    items, prov = D.training_items()
    guard.check(items, "training prompt")
    d20 = T.first_items(T.by_category(items), D20_SEED)
    config = {"merge_id": MERGE_ID, "distill_id": T.DISTILL_ID, "run_name": train["config"]["run_name"],
              "train_json": a.train_json, "train_git_commit": train["meta"]["git_commit"],
              "train_limited": bool(train["config"].get("limited")), "adapter_dir": str(adapter_dir),
              "adapter_sha256": adapter_sha, "merged_dir": str(merged_dir), "backbone_path": a.backbone_path,
              "device": a.device, "dtype": a.dtype, "d20_seed": D20_SEED,
              "d20_prompt_ids": [x["prompt_id"] for x in d20], "data_provenance": prov}
    rec = RunRecorder(out, config)
    device = torch.device(a.device)
    tok = D.load_tokenizer(a.tokenizer_path)
    t0 = time.perf_counter()
    base, prov_b = D.load_model(a.backbone_path, a.device, a.dtype)
    if prov_b["vocab_size"] != VOCAB_SIZE:
        raise SystemExit(f"vocabulary {prov_b['vocab_size']} != {VOCAB_SIZE}")
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    _sync(device)
    load_s = time.perf_counter() - t0
    ids = [D.chat_ids(tok, x["prompt"]) for x in d20]
    with torch.no_grad():
        unmerged = [hf_full_logits(model, s) for s in ids]
    _sync(device)
    t1 = time.perf_counter()
    merged = model.merge_and_unload()
    _sync(device)
    merge_s = time.perf_counter() - t1
    leftover = [n for n, _ in merged.named_parameters() if "lora_" in n]
    if leftover:
        raise SystemExit(f"merged model still holds LoRA parameters: {leftover[:3]}")
    merged.eval()
    with torch.no_grad():
        merged_logits = [hf_full_logits(merged, s) for s in ids]
    per = []
    for x, s, u, m in zip(d20, ids, unmerged, merged_logits):
        per.append({"prompt_id": x["prompt_id"], "category": x["category"], "tokens": len(s),
                    "max_abs_logit_diff": float((u - m).abs().max().item()),
                    "argmax_agreement": float((u.argmax(-1) == m.argmax(-1)).float().mean().item())})
    t2 = time.perf_counter()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    save_s = time.perf_counter() - t2
    d20_rec = {"rule": "first 4 mathematics and first 4 short-form prompts of the seed-6304 schedule; full forward "
                       "over the templated prompt; unmerged (adapter attached) versus merged",
               "max_abs_logit_diff": max(p["max_abs_logit_diff"] for p in per), "per_prompt": per}
    rec.finish("ok", d20=d20_rec, timing={"load_s": load_s, "merge_s": merge_s, "save_s": save_s},
               merged_files_sha256=T.dir_sha256(merged_dir), heldout_guard=guard.report(),
               model_revisions={"backbone": prov_b})
    print(f"{DONE}{config['run_name']}: merge {merge_s:.2f} s, D20 max |logit diff| {d20_rec['max_abs_logit_diff']:.3g} "
          f"-> {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
