"""Task 3 distillation run (off-policy or on-policy) with the forward-KL objective (manual Section 3 steps 3-4;
protocol Section 1, D7, D9, D10, D12, AM1).

Each presentation takes the next prompt of the D7 schedule; its response is the collection response (n mod K) for
off-policy or a fresh rollout of the current student (eval mode, no gradient, seed from (training seed, presentation
index)) for on-policy. The teacher scores the response once: fp32 log_softmax over all 151,936 rows at the scored
response positions only. Scored positions are consumed in schedule order; every update holds exactly 256 positions
(the final one the remainder); a response crossing an update boundary is split and its remaining positions are
recomputed with the updated student in the next update. The loss of an update is the forward KL summed over its
positions divided by its position count; FP16 autocast (CUDA), dynamic gradient scaling, unscale before clipping at
global norm 1.0, AdamW at a constant LR 3e-4. Skipped updates (gradient scale drop or non-finite clip norm) are
recorded and still consume their positions.

Outputs in --out (a new directory): run.json (RunRecorder, rewritten after every update, with one record per update)
and presentations.jsonl.gz (one record per presentation). The adapter after the final update is saved to
--adapter-dir (outside results/), its sha256 recorded in run.json.

    python -m task3.train --method on_policy --seed 6304 --out results/task3/train/on_policy_6304 \
        --adapter-dir /kaggle/working/t3_artifacts/adapters/on_policy_6304
    python -m task3.train --method off_policy --seed 6304 --collection results/task3/collection/... ...
"""
import argparse
import math
import time
from collections import Counter
from pathlib import Path

import torch

from common.io import sha256_gz_content, write_jsonl_gz
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import data as D
from . import distill as T
from .heldout import HeldoutGuard
from .lm import HFCachedLM
from .sampling import EOS_IDS, GENERATION_SETTINGS, SAMPLER_ID, VOCAB_SIZE, Timer, make_generator
from .spec import target_only_generate

DONE = "train done: "


def _float(x):
    x = float(x)
    return x if math.isfinite(x) else str(x)


def teacher_logprobs(teacher, prompt_ids, response, scored) -> torch.Tensor:
    """fp32 log-probabilities [scored, V] at the first `scored` response positions."""
    ids = list(prompt_ids) + list(response[: scored - 1])
    inp = torch.tensor([ids], dtype=torch.long, device=next(teacher.parameters()).device)
    with torch.no_grad():
        logits = teacher(input_ids=inp, use_cache=False, logits_to_keep=scored).logits[0].float()
        return torch.log_softmax(logits, dim=-1)


def kl_part(student, prompt_ids, response, offset, m, tlogp_part, autocast) -> torch.Tensor:
    """Forward KL(p_T || q_S) [m] at response positions offset..offset+m-1 (full vocabulary, fp32)."""
    ids = list(prompt_ids) + list(response[: offset + m - 1])
    device = next(student.parameters()).device
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
        logits = student(input_ids=inp, use_cache=False, logits_to_keep=m).logits[0]
    slogp = torch.log_softmax(logits.float(), dim=-1)
    if slogp.shape != tlogp_part.shape:
        raise RuntimeError(f"student rows {tuple(slogp.shape)} != teacher rows {tuple(tlogp_part.shape)}")
    return (tlogp_part.exp() * (tlogp_part - slogp)).sum(-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=T.METHODS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True, help="new output directory")
    ap.add_argument("--adapter-dir", required=True, help="new directory outside results/ for the final adapter")
    ap.add_argument("--collection", default=None, help="teacher collection .jsonl.gz (off_policy)")
    ap.add_argument("--teacher-path", default="model_target")
    ap.add_argument("--student-path", default="model_draft")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    for name in ("prompts-math", "prompts-short", "quota-math", "quota-short", "updates"):
        ap.add_argument(f"--limit-{name}", type=int, default=None, help="tests only; refused under results/")
    a = ap.parse_args(argv)

    out, adapter_dir = Path(a.out), Path(a.adapter_dir)
    for p in (out, adapter_dir):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    if T.under_results(adapter_dir):
        raise SystemExit("--adapter-dir must be outside results/")
    limits = {"limit_prompts_math": a.limit_prompts_math, "limit_prompts_short": a.limit_prompts_short,
              "limit_quota_math": a.limit_quota_math, "limit_quota_short": a.limit_quota_short,
              "limit_updates": a.limit_updates}
    limited = any(v is not None for v in limits.values())
    if limited and T.under_results(out):
        raise SystemExit("--limit-* options are refused under results/")
    if a.method == "off_policy" and a.collection is None:
        raise SystemExit("off_policy needs --collection")
    if a.method == "on_policy" and a.collection is not None:
        raise SystemExit("--collection is only for off_policy")
    if a.seed not in T.TRAIN_SEEDS and not limited:
        raise SystemExit(f"training seed must be one of {T.TRAIN_SEEDS}")

    seed_everything(a.seed)
    guard = HeldoutGuard()
    items, prov = D.training_items(a.limit_prompts_math, a.limit_prompts_short)
    guard.check(items, "training prompt")
    by_cat = T.by_category(items)
    quota = {"math": a.limit_quota_math or T.QUOTA["math"], "short": a.limit_quota_short or T.QUOTA["short"]}
    sizes = T.update_sizes(quota["math"] + quota["short"])
    n_updates_full = len(sizes)
    if a.limit_updates is not None:
        sizes = sizes[: a.limit_updates]

    coll, coll_rec = None, None
    if a.method == "off_policy":
        coll = T.load_collection(a.collection)
        ccfg = coll["run"]["config"]
        if ccfg.get("collection_id") != T.COLLECTION_ID:
            raise SystemExit(f"collection version {ccfg.get('collection_id')!r} != {T.COLLECTION_ID}")
        known = {x["prompt_id"] for x in items}
        stray = sorted({sid for sid, _ in coll["records"]} - known)
        if stray:
            raise SystemExit(f"collection holds {len(stray)} sources outside the training set, first {stray[:5]}")
        k_per = {c: int(ccfg["k"][c]) for c in T.CATEGORIES}
        missing = [(x["prompt_id"], k) for x in items for k in range(k_per[x["category"]])
                   if (x["prompt_id"], k) not in coll["records"]]
        if missing:
            raise SystemExit(f"collection lacks {len(missing)} responses, first {missing[:3]}")
        coll_rec = {"path": coll["path"], "run_json": coll["run_json"], "content_sha256": coll["content_sha256"],
                    "k": k_per, "limited": bool(ccfg.get("limited")), "git_commit": coll["run"]["meta"]["git_commit"]}

    name = f"{a.method}_{a.seed}"
    config = {
        "distill_id": T.DISTILL_ID, "sampler_id": SAMPLER_ID, "run_name": name, "method": a.method, "seed": a.seed,
        "settings": T.SETTINGS, "quota": quota, "update_sizes": sizes, "n_updates_full": n_updates_full,
        "limits": limits, "limited": limited, "collection": coll_rec, "teacher_path": a.teacher_path,
        "student_path": a.student_path, "device": a.device, "dtype": a.dtype, "data_provenance": prov,
        "n_prompts": {c: len(by_cat[c]) for c in T.CATEGORIES}, "schedule_seed": a.seed,
        "rollout_seed_rule": "sha256('t3-rollout|<seed>|<presentation index>') first 15 hex digits",
        "generation_settings": GENERATION_SETTINGS, "out": str(out), "adapter_dir": str(adapter_dir),
    }
    out.mkdir(parents=True)
    rec = RunRecorder(out / "run.json", config)

    device = torch.device(a.device)
    timer = Timer(device)
    t0 = timer.now()
    tok = D.load_tokenizer(a.teacher_path)
    teacher, prov_t = D.load_model(a.teacher_path, a.device, a.dtype)
    student, prov_s = D.load_model(a.student_path, a.device, a.dtype)
    for pv in (prov_t, prov_s):
        if pv["vocab_size"] != VOCAB_SIZE:
            raise SystemExit(f"vocabulary {pv['vocab_size']} != {VOCAB_SIZE}")
    student = T.add_lora(student, a.seed)
    params = T.trainable(student)
    n_trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type, **T.SCALER)
    autocast = device.type == "cuda"
    load_s = timer.now() - t0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rec.update(model_revisions={"teacher": prov_t, "student_backbone": prov_s},
               lora={"trainable_parameters": n_trainable, "adapter_dtype": str(params[0].dtype)},
               autocast_fp16=autocast, timing={"load_s": load_s})

    sched = T.Schedule(by_cat, a.seed, quota)
    pres, updates = [], []
    totals = Counter()
    cum = Counter()
    cur = None
    student.train()
    opt.zero_grad(set_to_none=True)

    def new_update():
        return {"n": 0, "by_cat": Counter(), "kl_sum": 0.0, "kl_by_cat": Counter(), "teacher_s": 0.0,
                "rollout_s": 0.0, "fwd_bwd_s": 0.0, "started": 0, "t0": timer.now()}

    def start_presentation(upd):
        cat = sched.next_category()
        if cat is None:
            raise RuntimeError("schedule exhausted before the last update")
        item, n = sched.next_item(cat)
        prompt_ids = D.chat_ids(tok, item["prompt"])
        p_idx = len(pres)
        r = {"presentation": p_idx, "category": cat, "source_id": item["prompt_id"], "n_of_prompt": n,
             "prompt_tokens": len(prompt_ids), "method": a.method}
        t_r = 0.0
        if a.method == "off_policy":
            k = n % coll_rec["k"][cat]
            c = coll["records"][(item["prompt_id"], k)]
            response, term, seed = list(c["response_token_ids"]), c["termination"], c["rollout_seed"]
            r.update(k=k, rollout_seed=seed, wrap_around=n >= coll_rec["k"][cat], producing_update=None)
        else:
            seed = T.rollout_seed(a.seed, p_idx)
            student.eval()
            ta = timer.now()
            res = target_only_generate(HFCachedLM(student, expect_vocab=VOCAB_SIZE), prompt_ids, T.CAPS[cat],
                                       EOS_IDS, make_generator(seed, a.device), timer=timer)
            t_r = timer.now() - ta
            student.train()
            response, term = res["output_ids"], res["termination"]
            r.update(k=None, rollout_seed=seed, wrap_around=False, producing_update=len(updates))
        scored, truncated = sched.assign(cat, len(response))
        ta = timer.now()
        tlogp = teacher_logprobs(teacher, prompt_ids, response, scored)
        t_t = timer.now() - ta
        r.update(response_token_ids=response, termination=term, length=len(response), scored=scored,
                 truncated=truncated, teacher_s=t_t, rollout_s=t_r, updates=[], consumed=0)
        upd["teacher_s"] += t_t
        upd["rollout_s"] += t_r
        upd["started"] += 1
        totals["teacher_s"] += t_t
        totals["rollout_s"] += t_r
        pres.append(r)
        return {"rec": r, "prompt_ids": prompt_ids, "response": response, "tlogp": tlogp, "offset": 0,
                "scored": scored, "cat": cat}

    upd = new_update()
    t_train = timer.now()
    while len(updates) < len(sizes):
        size = sizes[len(updates)]
        if cur is None or cur["offset"] == cur["scored"]:
            cur = start_presentation(upd)
        m = min(cur["scored"] - cur["offset"], size - upd["n"])
        ta = timer.now()
        kl = kl_part(student, cur["prompt_ids"], cur["response"], cur["offset"], m,
                     cur["tlogp"][cur["offset"]: cur["offset"] + m], autocast)
        loss = kl.sum() / size
        scaler.scale(loss).backward()
        tb = timer.now()
        upd["fwd_bwd_s"] += tb - ta
        totals["fwd_bwd_s"] += tb - ta
        kl_sum = float(kl.detach().sum().item())
        upd["kl_sum"] += kl_sum
        upd["kl_by_cat"][cur["cat"]] += kl_sum
        upd["n"] += m
        upd["by_cat"][cur["cat"]] += m
        cur["offset"] += m
        cur["rec"]["consumed"] = cur["offset"]
        cur["rec"]["updates"].append(len(updates) + 1)
        if upd["n"] < size:
            continue
        ta = timer.now()
        scale_before = float(scaler.get_scale())
        scaler.unscale_(opt)
        norm = float(torch.nn.utils.clip_grad_norm_(params, T.CLIP_NORM).item())
        scaler.step(opt)
        scaler.update()
        scale_after = float(scaler.get_scale())
        opt.zero_grad(set_to_none=True)
        tb = timer.now()
        totals["step_s"] += tb - ta
        skipped = bool(scale_after < scale_before or not math.isfinite(norm))
        for c in T.CATEGORIES:
            cum[c] += upd["by_cat"][c]
        updates.append({
            "update": len(updates) + 1, "positions": upd["n"], "positions_by_category": dict(upd["by_cat"]),
            "loss": upd["kl_sum"] / upd["n"],
            "kl_mean_by_category": {c: upd["kl_by_cat"][c] / upd["by_cat"][c] for c in upd["by_cat"]},
            "lr": float(opt.param_groups[0]["lr"]), "scale_before": scale_before, "scale_after": scale_after,
            "clip_norm": _float(norm), "skipped": skipped, "cumulative_scored": sum(cum.values()),
            "cumulative_scored_by_category": dict(cum), "presentations_started": upd["started"],
            "teacher_s": upd["teacher_s"], "rollout_s": upd["rollout_s"], "fwd_bwd_s": upd["fwd_bwd_s"],
            "step_s": tb - ta, "wall_s": tb - upd["t0"]})
        rec.update(updates=updates, progress={"updates_done": len(updates), "presentations": len(pres),
                                              "skipped_updates": [u["update"] for u in updates if u["skipped"]]})
        upd = new_update()
    train_s = timer.now() - t_train
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    ta = timer.now()
    student.save_pretrained(str(adapter_dir))
    save_s = timer.now() - ta
    adapter_sha = T.dir_sha256(adapter_dir)
    write_jsonl_gz(out / "presentations.jsonl.gz", pres)
    unconsumed = [p["presentation"] for p in pres if p["consumed"] < p["scored"]]
    summary = {
        "n_updates": len(updates), "n_presentations": len(pres),
        "skipped_updates": [u["update"] for u in updates if u["skipped"]],
        "scored_by_category": dict(cum), "scored_total": sum(cum.values()),
        "assigned_by_category": dict(sched.assigned),
        "presentations_by_category": dict(Counter(p["category"] for p in pres)),
        "truncated_presentations": [p["presentation"] for p in pres if p["truncated"]],
        "unconsumed_presentations": unconsumed,
        "wrap_arounds": sum(1 for p in pres if p["wrap_around"]),
        "terminations": dict(Counter(f"{p['category']}_{p['termination']}" for p in pres)),
        "response_tokens_by_category": {c: sum(p["length"] for p in pres if p["category"] == c)
                                        for c in T.CATEGORIES},
        "final_scale": float(scaler.get_scale()),
    }
    timing = {"load_s": load_s, "train_loop_s": train_s, "teacher_scoring_s": totals["teacher_s"],
              "student_rollout_s": totals["rollout_s"], "forward_backward_s": totals["fwd_bwd_s"],
              "optimizer_step_s": totals["step_s"], "adapter_save_s": save_s,
              "mean_update_wall_s": (sum(u["wall_s"] for u in updates) / len(updates)) if updates else None}
    rec.finish("ok", summary=summary, timing=timing, peak_gpu_memory_bytes=peak, adapter_sha256=adapter_sha,
               heldout_guard=guard.report(),
               presentations_file={"path": "presentations.jsonl.gz",
                                   "content_sha256": sha256_gz_content(out / "presentations.jsonl.gz")})
    print(f"{DONE}{name}: {len(updates)} updates, {sum(cum.values())} scored positions {dict(cum)}, "
          f"{len(pres)} presentations, skipped {summary['skipped_updates']}, loop {train_s:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
