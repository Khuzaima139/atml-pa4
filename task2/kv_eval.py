"""Task 2 Part C runner: C1 validation of the FP16 cache path and the C2-C5 quality evaluation.

Usage: python -m task2.kv_eval --mode validate --out DIR [--device cuda:0]
       python -m task2.kv_eval --mode eval     --out DIR [--device cuda:0]
Test-only options (refused for any output under results/task2):
       --limit-prompts N | ID,ID,...   the first N prompt ids, or the listed ids
       --model-path DIR                a local model directory instead of the locked checkpoint

validate (C1): for every prompt, the identity-compression pipeline (full FP16 prefill, cache of the context kept as
stored, question-suffix re-run with absolute position_ids and physical cache_position, argmax decoding) against
model.generate with the same greedy settings: token identity, first divergence, max abs logit difference over the
steps whose inputs agree, finiteness, and the absolute positions of the physical cache entries. For the lowest-id
prompt of each grid cell: direct full-forward logits over the prompt plus the FP16 reference answer (the generate
output) against (a) one prefill of the prompt plus single-token decoding of the answer and (b) the identity pipeline
teacher-forced on the answer; and, for both eviction rules, the suffix logits after eviction against a full forward
whose 4D attention mask hides the evicted context positions.

eval (C2-C5): per prompt one full FP16 prefill and one question-aware scoring pass, then the six cache methods in an
order rotated by one position per prompt; each method compresses the context, re-runs the suffix, decodes greedily
(at most 32 new tokens, stop at EOS) and is then teacher-forced on the gold code. The gold code and evidence positions
are used only after every method of the prompt has finished decoding.

Outputs under --out:
  validate: run.json, validate_rows.jsonl.gz (one row per prompt), full_forward_rows.jsonl.gz (one row per cell)
  eval:     run.json, eval_rows.jsonl.gz (one row per prompt and method), prompt_rows.jsonl.gz (one row per prompt),
            npz/<id>.npz (question scores, evidence positions, retained FP16 positions), qualitative/<id>.txt
Row files are written as plain JSONL while running and compressed when the run finishes. run.json "timing" holds the
wall seconds of model loading and of each prompt (by id).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import kv_budget as KB
from . import kv_cache_ops as KO
from . import kv_metrics as KM
from . import kv_run as KR
from .rows import RowWriter

VERSION = KR.VERSION
NPZ_FORMAT_VERSION = 1
TF_DEVICE = "cpu"
QUAL_WINDOW = 16
EVICTION_METHODS = ("sink_recent", "question_aware")
FP16_SET_METHODS = ("sink_recent", "question_aware", "hybrid")
OUTPUT_FILES = {
    "validate": {"run": "run.json", "rows": "validate_rows.jsonl", "full_forward": "full_forward_rows.jsonl"},
    "eval": {"run": "run.json", "rows": "eval_rows.jsonl", "prompts": "prompt_rows.jsonl", "npz_dir": "npz",
             "qualitative_dir": "qualitative"},
}
NPZ_ARRAYS = {"question_scores": np.float16, "evidence_positions": np.int32, "sink_recent_positions": np.int32,
              "question_aware_positions": np.int32, "hybrid_fp16_positions": np.int32, "context_len": np.int32,
              "suffix_len": np.int32, "format_version": np.int32, "version_ascii": np.uint8}
METHOD_ORDER_RULE = "task2.kv_cache_ops.METHODS rotated left by (prompt_index mod 6); prompt_index = rank of the id"


def method_order(prompt_index: int) -> list:
    k = prompt_index % len(KO.METHODS)
    return list(KO.METHODS[k:] + KO.METHODS[:k])


def _timer(device):
    return KO._Timer(device)


def positions_preserved(run: KO.MethodRun, L: int) -> bool:
    """The physical entries carry the retained context positions (ascending) followed by the appended positions, and
    the next absolute position and the physical length agree with them."""
    kept = run.info["fp16_positions"] if run.method in EVICTION_METHODS else torch.arange(L)
    expected = torch.cat([kept.cpu().long(), torch.arange(L, L + run.n_appended)])
    return bool(torch.equal(run.final_positions.long(), expected) and run.physical_len == expected.numel()
                and run.next_pos == L + run.n_appended)


def decode_text(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True)


# ---------------------------------------------------------------------------------------------------------------------
# validate (C1)


def generation_config():
    from transformers import GenerationConfig

    return GenerationConfig(**KR.GREEDY)


def effective_generation_config(model, gc) -> dict:
    """The settings generate uses after merging (use_model_defaults=False keeps the passed values)."""
    try:
        eff, _ = model._prepare_generation_config(gc, use_model_defaults=False)
        full = eff.to_dict()
        return {"settings": {k: full.get(k) for k in KR.GREEDY}, "non_default": eff.to_diff_dict()}
    except Exception as exc:  # private API; recorded instead of failing the run
        return {"error": f"{type(exc).__name__}: {exc}"}


@torch.no_grad()
def reference_generate(model, ids, gc) -> dict:
    with _timer(ids.device) as t:
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gc,
                             use_model_defaults=False, output_logits=True, return_dict_in_generate=True)
    tokens = out.sequences[0, ids.shape[1]:].tolist()
    logits = torch.stack(out.logits)[:, 0].float()
    stop = "eos" if tokens and tokens[-1] in KO.EOS_IDS else "max_new_tokens"
    return {"tokens": tokens, "logits": logits, "stop": stop, "seconds": t.seconds}


def compare_steps(ref_tokens, ref_logits, tokens, logits) -> dict:
    """Token identity, the first differing step, and logit differences over the steps whose inputs agree (every step
    up to and including the first divergence)."""
    n = min(len(ref_tokens), len(tokens))
    div = next((i for i in range(n) if ref_tokens[i] != tokens[i]), None)
    if div is None and len(ref_tokens) != len(tokens):
        div = n
    identical = div is None
    n_cmp = min(len(ref_tokens), len(tokens)) if identical else min(div + 1, len(ref_tokens), len(tokens))
    d = (ref_logits[:n_cmp] - logits[:n_cmp]).abs()
    return {"tokens_identical": identical, "first_divergence": div, "n_steps_compared": int(n_cmp),
            "max_abs_logit_diff": float(d.max()) if n_cmp else None,
            "mean_abs_logit_diff": float(d.mean()) if n_cmp else None}


def _diff(a, b) -> dict:
    d = (a.float() - b.float()).abs()
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "argmax_agree": int((a.argmax(-1) == b.argmax(-1)).sum()), "n_rows": int(a.shape[0])}


@torch.no_grad()
def direct_logits(model, ids, answer) -> torch.Tensor:
    """Full forward over prompt + answer[:-1]; FP32 logits of the len(answer) positions predicting the answer."""
    seq = torch.cat([ids, torch.tensor([answer[:-1]], dtype=torch.long, device=ids.device)], dim=1)
    return model(input_ids=seq, logits_to_keep=len(answer)).logits[0].float()


@torch.no_grad()
def prefill_decode_logits(model, ids, answer) -> torch.Tensor:
    """One prefill of the prompt, then single-token decoding of answer[:-1] on the same cache."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    rows = [model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1].float()]
    for tok in answer[:-1]:
        out = model(input_ids=torch.tensor([[tok]], device=ids.device), past_key_values=cache, use_cache=True)
        rows.append(out.logits[0, -1].float())
    return torch.stack(rows)


def eviction_mask(T: int, evicted: list, first_row: int, device) -> torch.Tensor:
    m = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    if evicted:
        m[first_row:, torch.as_tensor(evicted, device=device)] = False
    return m[None, None]


@torch.no_grad()
def eviction_check(model, shape, prefill, ids, scores, method) -> dict:
    """Suffix logits after eviction against a full forward with the evicted context hidden by a 4D mask."""
    L, T = prefill.context_len, ids.shape[1]
    run, _, suffix = KO.run_method(model, shape, prefill, ids, method, scores, max_new_tokens=1,
                                   suffix_logits_all=True)
    kept = run.info["fp16_positions"].tolist()
    evicted = sorted(set(range(L)) - set(kept))
    ref = model(input_ids=ids, attention_mask=eviction_mask(T, evicted, L, ids.device),
                logits_to_keep=T - L).logits[0].float()
    d = _diff(ref, suffix)
    return {"n_kept": len(kept), "n_evicted": len(evicted), "suffix_max_abs_diff": d["max_abs_diff"],
            "suffix_mean_abs_diff": d["mean_abs_diff"], "suffix_argmax_agree": d["argmax_agree"],
            "last_position_max_abs_diff": float((ref[-1] - suffix[-1]).abs().max()),
            "positions_preserved": positions_preserved(run, L),
            "finite": bool(torch.isfinite(suffix).all() and torch.isfinite(ref).all())}


def run_validate(model, tokenizer, shape, cases, device, writers, rec, timing) -> dict:
    gc = generation_config()
    cell_ids = set(KR.first_per_cell(cases))
    rows, ff_rows = [], []
    for idx, case in enumerate(cases):
        t_prompt = time.perf_counter()
        inputs, gold = KR.prepare_prompt(tokenizer, case)
        ids = inputs.ids_tensor(device)
        L = inputs.context_len
        ref = reference_generate(model, ids, gc)
        with _timer(device) as t_pipe:
            prefill = KO.full_prefill(model, ids, L)
            KR.assert_fp16_cache(prefill.keys, prefill.values)
            run, tf, _ = KO.run_method(model, shape, prefill, ids, "fp16", collect_logits=True,
                                       tf_device=TF_DEVICE)
        cmp = compare_steps(ref["tokens"], ref["logits"], run.tokens, run.step_logits)
        finite = {"generate_logits_finite": bool(torch.isfinite(ref["logits"]).all()),
                  "pipeline_logits_finite": bool(torch.isfinite(run.step_logits).all()),
                  "prefill_logits_finite": bool(torch.isfinite(prefill.last_logits).all())}
        # evaluation-only fields below this line (after both decodes)
        gtext, ptext = decode_text(tokenizer, ref["tokens"]), decode_text(tokenizer, run.tokens)
        gem, pem = KM.exact_match(gtext, gold.code), KM.exact_match(ptext, gold.code)
        row = {"version": VERSION, "scorer": KM.SCORER, "id": inputs.id, "prompt_index": idx, **inputs.cell(),
               "context_len": L, "suffix_len": inputs.suffix_len, "total_len": inputs.total_len,
               "generate_token_ids": ref["tokens"], "pipeline_token_ids": run.tokens, **cmp, **finite,
               "all_logits_finite": all(finite.values()), "positions_preserved": positions_preserved(run, L),
               "physical_len": run.physical_len, "next_pos": run.next_pos, "n_appended": run.n_appended,
               "generate_stop": ref["stop"], "pipeline_stop": run.stop, "generate_text": gtext,
               "pipeline_text": ptext, "generate_prediction": gem["prediction"], "generate_correct": gem["correct"],
               "generate_no_match": gem["no_match"], "pipeline_prediction": pem["prediction"],
               "pipeline_correct": pem["correct"], "generate_seconds": ref["seconds"],
               "pipeline_seconds": t_pipe.seconds, "full_forward_check": inputs.id in cell_ids}
        writers["rows"].write(row)
        rows.append(row)
        if inputs.id in cell_ids:
            answer = ref["tokens"]
            direct = direct_logits(model, ids, answer)
            split = prefill_decode_logits(model, ids, answer)
            pipe = KO.teacher_forced_logits(tf.clone(device), run.first_logits, answer)
            scores, _ = KO.score_prefill(model, prefill)
            ev = {m: eviction_check(model, shape, prefill, ids, scores, m) for m in EVICTION_METHODS}
            ff = {"version": VERSION, "id": inputs.id, **inputs.cell(), "context_len": L,
                  "suffix_len": inputs.suffix_len, "answer_len": len(answer), "answer_token_ids": answer,
                  "split_prefill_decode": _diff(direct, split), "pipeline_identity": _diff(direct, pipe),
                  "logits_finite": bool(torch.isfinite(direct).all() and torch.isfinite(split).all()
                                        and torch.isfinite(pipe).all()),
                  "eviction_mask_check": ev}
            writers["full_forward"].write(ff)
            ff_rows.append(ff)
            del direct, split, pipe, scores
        del prefill, run, tf, ref
        timing["prompt_s"][inputs.id] = time.perf_counter() - t_prompt
        rec.update(step="validate", prompts_done=idx + 1, timing=timing)
    return validate_summary(rows, ff_rows)


def _cell_accuracy(rows, flag) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[f"{r['target_tokens']}-{r['position']}-{r['distractors']}"].append(bool(r[flag]))
    return {k: {"n": len(v), "n_correct": sum(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by.items())}


def _maxes(vals):
    vals = [v for v in vals if v is not None]
    return {"max": max(vals), "median": statistics.median(vals)} if vals else None


def validate_summary(rows, ff_rows) -> dict:
    return {
        "n_prompts": len(rows),
        "n_tokens_identical": sum(r["tokens_identical"] for r in rows),
        "diverged": [{"id": r["id"], "first_divergence": r["first_divergence"]} for r in rows
                     if not r["tokens_identical"]],
        "max_abs_logit_diff": _maxes([r["max_abs_logit_diff"] for r in rows]),
        "n_all_logits_finite": sum(r["all_logits_finite"] for r in rows),
        "n_positions_preserved": sum(r["positions_preserved"] for r in rows),
        "generate_accuracy_overall": sum(r["generate_correct"] for r in rows) / len(rows),
        "generate_accuracy_by_cell": _cell_accuracy(rows, "generate_correct"),
        "pipeline_accuracy_overall": sum(r["pipeline_correct"] for r in rows) / len(rows),
        "full_forward": {
            "n_prompts": len(ff_rows),
            "split_prefill_decode_max_abs_diff": _maxes([r["split_prefill_decode"]["max_abs_diff"] for r in ff_rows]),
            "pipeline_identity_max_abs_diff": _maxes([r["pipeline_identity"]["max_abs_diff"] for r in ff_rows]),
            "argmax_agree": {k: [sum(r[k]["argmax_agree"] for r in ff_rows), sum(r[k]["n_rows"] for r in ff_rows)]
                             for k in ("split_prefill_decode", "pipeline_identity")},
            "eviction_suffix_max_abs_diff": {
                m: _maxes([r["eviction_mask_check"][m]["suffix_max_abs_diff"] for r in ff_rows])
                for m in EVICTION_METHODS},
            "n_logits_finite": sum(r["logits_finite"] for r in ff_rows),
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# eval (C2-C5)


def generation_stage(model, shape, inputs: KR.PromptInputs, device, order) -> dict:
    """Prefill, scoring and every method's compression, suffix re-run and decoding. Receives no gold field."""
    ids = inputs.ids_tensor(device)
    prefill = KO.full_prefill(model, ids, inputs.context_len)
    KR.assert_fp16_cache(prefill.keys, prefill.values)
    scores, t_score = KO.score_prefill(model, prefill)
    runs, sessions = {}, {}
    for m in order:
        runs[m], sessions[m], _ = KO.run_method(model, shape, prefill, ids, m, scores, tf_device=TF_DEVICE)
    return {"prefill": prefill, "scores": scores, "scoring_seconds": t_score, "runs": runs, "sessions": sessions}


def teacher_forced_stage(gen: dict, gold: KR.Gold, device, order) -> dict:
    """Teacher-forced logits on the gold code for every method, from its own post-suffix cache state."""
    out = {}
    for m in order:
        session = gen["sessions"][m].clone(device)
        with _timer(device) as t:
            logits = KO.teacher_forced_logits(session, gen["runs"][m].first_logits, gold.code_ids)
        out[m] = {"logits": logits, "seconds": t.seconds}
        del session
    return out


def method_row(inputs, gold, shape, idx, order, m, run, tf, ref_run, ref_tf, tokenizer) -> dict:
    L = inputs.context_len
    text = decode_text(tokenizer, run.tokens)
    em = KM.exact_match(text, gold.code)
    nll = KM.nll_rows(tf["logits"], gold.code_ids)
    kl_tf = KM.kl_rows(ref_tf["logits"], tf["logits"])
    kl_post = KM.kl_rows(ref_run.first_logits[None], run.first_logits[None])[0]
    fp_pos = run.info.get("fp16_positions")
    ev_kept = None if fp_pos is None else sum(int(p) in set(fp_pos.tolist()) for p in gold.evidence_positions)
    cb = run.context_bytes
    if cb["total"] != run.info["formula_bytes"]:
        raise RuntimeError(f"{inputs.id} {m}: counted bytes {cb['total']} != formula {run.info['formula_bytes']}")
    recon = run.recon
    return {
        "version": VERSION, "scorer": KM.SCORER, "id": inputs.id, "prompt_index": idx, "method": m,
        "method_order": order, "method_slot": order.index(m), **inputs.cell(), "context_len": L,
        "suffix_len": inputs.suffix_len,
        "generated_token_ids": run.tokens, "generated_text": text, "n_generated": len(run.tokens), "stop": run.stop,
        "hit_max_new_tokens": run.stop == "max_new_tokens", "prediction": em["prediction"], "correct": em["correct"],
        "no_match": em["no_match"],
        "answer_nll_mean": float(nll.mean()), "answer_nll_per_token": [float(x) for x in nll],
        "kl_post_prefill": float(kl_post), "kl_teacher_forced_mean": float(kl_tf.mean()),
        "kl_teacher_forced_per_position": [float(x) for x in kl_tf],
        "k_recon_mean": None if recon is None else recon["k_mean"],
        "v_recon_mean": None if recon is None else recon["v_mean"],
        "k_recon_per_layer": None if recon is None else recon["k_per_layer"],
        "v_recon_per_layer": None if recon is None else recon["v_per_layer"],
        "evidence_token_count": len(gold.evidence_positions), "evidence_retained_count": ev_kept,
        "evidence_fraction_retained": None if ev_kept is None else ev_kept / len(gold.evidence_positions),
        "n_fp16_positions": None if fp_pos is None else int(fp_pos.numel()),
        "eviction_n": run.info.get("n"), "hybrid_m": run.info.get("m"), "hybrid_f": run.info.get("f"),
        "M4": run.info["M4"], "M16": run.info["M16"], "formula_bytes": run.info["formula_bytes"],
        "context_bytes": dict(cb), "realized_fraction": cb["total"] / run.info["M16"],
        "n_appended": run.n_appended, "decode_time_bytes": KB.decode_time_fp16_bytes(shape, run.n_appended),
        "live_bytes_end": dict(run.live_bytes), "total_live_bytes_end": run.live_bytes["total"],
        "positions_preserved": positions_preserved(run, L), "tail_rebuilds": run.tail_rebuilds,
        "seconds": {**run.seconds, "teacher_forced": tf["seconds"]},
        "peak_allocated_bytes": run.peak_bytes, "start_allocated_bytes": run.start_allocated_bytes,
    }


def ranges(positions) -> str:
    """Compact ascending ranges, e.g. '0-7, 100, 2540-2603'."""
    pos = sorted(int(p) for p in positions)
    if not pos:
        return "(none)"
    out, start, prev = [], pos[0], pos[0]
    for p in pos[1:]:
        if p != prev + 1:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = p
        prev = p
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(out)


def qualitative_text(inputs, gold, tokenizer, rows: dict, sets: dict) -> str:
    """Plain-text view: evidence span, target code, each method's generation, and an aligned token-selection mask
    around the evidence span (K kept in FP16, . evicted; hybrid F FP16, 2 INT2)."""
    ev = list(gold.evidence_positions)
    lo, hi = max(0, min(ev) - QUAL_WINDOW), min(inputs.context_len - 1, max(ev) + QUAL_WINDOW)
    sr, qa, hy = (set(int(p) for p in sets[k]) for k in FP16_SET_METHODS)
    evs = set(ev)
    ev_decoded = tokenizer.decode([inputs.input_ids[p] for p in ev])
    lines = [
        f"prompt {inputs.id}  version {VERSION}",
        f"cell: context level {inputs.target_tokens}, evidence location {inputs.position}, "
        f"distractors {inputs.distractors}; context tokens {inputs.context_len}, suffix tokens {inputs.suffix_len}",
        f"target code: {gold.code}",
        f"evidence span text: {gold.evidence_text}",
        f"evidence tokens ({len(ev)}): positions {ranges(ev)}; decoded: {ev_decoded!r}",
        "",
        "generations (method: correct / extracted / text):",
    ]
    for m in KO.METHODS:
        r = rows[m]
        lines.append(f"  {m:15s} {'correct' if r['correct'] else 'wrong  '} {str(r['prediction']):6s} "
                     f"{r['generated_text']!r}")
    lines += ["", "FP16 positions kept over the whole context:"]
    for m, s in zip(FP16_SET_METHODS, (sr, qa, hy)):
        kept = sum(p in s for p in ev)
        lines.append(f"  {m:15s} {len(s)} positions, evidence kept {kept}/{len(ev)}: {ranges(s)}")
    lines += ["", f"token-selection mask, positions {lo}-{hi} (E evidence token; sink_recent and question_aware: "
                  "K kept in FP16, . evicted; hybrid: F FP16, 2 INT2)",
              f"{'pos':>6} E SR QA HY token"]
    for p in range(lo, hi + 1):
        tok = tokenizer.decode([inputs.input_ids[p]])
        lines.append(f"{p:6d} {'E' if p in evs else ' '} {'K' if p in sr else '.':>2} {'K' if p in qa else '.':>2} "
                     f"{'F' if p in hy else '2':>2} {tok!r}")
    return "\n".join(lines) + "\n"


def npz_arrays(inputs, gold, gen) -> dict:
    sets = {m: gen["runs"][m].info["fp16_positions"].numpy() for m in FP16_SET_METHODS}
    return {"question_scores": gen["scores"].float().cpu().numpy().astype(np.float16),
            "evidence_positions": np.asarray(gold.evidence_positions, dtype=np.int32),
            "sink_recent_positions": sets["sink_recent"].astype(np.int32),
            "question_aware_positions": sets["question_aware"].astype(np.int32),
            "hybrid_fp16_positions": sets["hybrid"].astype(np.int32),
            "context_len": np.array(inputs.context_len, dtype=np.int32),
            "suffix_len": np.array(inputs.suffix_len, dtype=np.int32),
            "format_version": np.array(NPZ_FORMAT_VERSION, dtype=np.int32),
            "version_ascii": np.frombuffer(VERSION.encode("ascii"), dtype=np.uint8).copy()}


def run_eval(model, tokenizer, shape, cases, device, out, files, writers, rec, timing) -> dict:
    qual_ids = set(KR.qualitative_ids(cases))
    orders = {}
    summary_rows = []
    for idx, case in enumerate(cases):
        t_prompt = time.perf_counter()
        inputs, gold = KR.prepare_prompt(tokenizer, case)
        order = method_order(idx)
        orders[inputs.id] = order
        gen = generation_stage(model, shape, inputs, device, order)
        # gold fields enter here, after every method has finished decoding
        tf = teacher_forced_stage(gen, gold, device, order)
        runs = gen["runs"]
        rows = {m: method_row(inputs, gold, shape, idx, order, m, runs[m], tf[m], runs["fp16"], tf["fp16"],
                              tokenizer) for m in order}
        for m in order:
            writers["rows"].write(rows[m])
            summary_rows.append(rows[m])
        npz_name = f"{files['npz_dir']}/{inputs.id}.npz"
        KR.write_npz_new(out / npz_name, npz_arrays(inputs, gold, gen))
        qual_name = None
        if inputs.id in qual_ids:
            qual_name = f"{files['qualitative_dir']}/{inputs.id}.txt"
            sets = {m: runs[m].info["fp16_positions"].tolist() for m in FP16_SET_METHODS}
            KR.write_text_new(out / qual_name, qualitative_text(inputs, gold, tokenizer, rows, sets))
        pre = gen["prefill"]
        writers["prompts"].write({
            "version": VERSION, "scorer": KM.SCORER, "id": inputs.id, "prompt_index": idx, **inputs.cell(),
            "code": gold.code, "context_len": inputs.context_len, "suffix_len": inputs.suffix_len,
            "total_len": inputs.total_len, "evidence_token_count": len(gold.evidence_positions),
            "method_order": order, "prefill_seconds": pre.seconds, "scoring_seconds": gen["scoring_seconds"],
            "prefill_peak_allocated_bytes": pre.peak_bytes, "budget": KB.budget_table(shape, inputs.context_len),
            "fp16_correct": rows["fp16"]["correct"], "npz": npz_name, "qualitative": qual_name,
        })
        del gen, tf, runs, pre
        timing["prompt_s"][inputs.id] = time.perf_counter() - t_prompt
        rec.update(step="eval", prompts_done=idx + 1, timing=timing)
    return {"method_orders": orders, **eval_summary(summary_rows)}


def eval_summary(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    per = {}
    for m in KO.METHODS:
        rs = by.get(m, [])
        if not rs:
            continue
        per[m] = {"n": len(rs), "n_correct": sum(r["correct"] for r in rs),
                  "accuracy": sum(r["correct"] for r in rs) / len(rs),
                  "n_no_match": sum(r["no_match"] for r in rs),
                  "n_hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in rs),
                  "mean_answer_nll": statistics.fmean(r["answer_nll_mean"] for r in rs),
                  "mean_kl_post_prefill": statistics.fmean(r["kl_post_prefill"] for r in rs),
                  "mean_kl_teacher_forced": statistics.fmean(r["kl_teacher_forced_mean"] for r in rs),
                  "mean_realized_fraction": statistics.fmean(r["realized_fraction"] for r in rs),
                  "n_positions_preserved": sum(r["positions_preserved"] for r in rs)}
    return {"n_rows": len(rows), "per_method": per,
            "fp16_accuracy_by_cell": _cell_accuracy(by.get("fp16", []), "correct") if by.get("fp16") else {}}


# ---------------------------------------------------------------------------------------------------------------------
# Entry point


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("validate", "eval"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-prompts", default=None, help="N or ID,ID,... (tests only)")
    ap.add_argument("--model-path", default=None, help="local model directory (tests only)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    limits = {"prompts": args.limit_prompts, "model_path": args.model_path}
    limited = KR.check_out(args.out, limits)
    out = Path(args.out)
    device = torch.device(args.device)
    if KR.is_cuda(device):
        torch.cuda.set_device(device)
    seed_everything(KR.SEED)
    files = OUTPUT_FILES[args.mode]
    data = KR.data_record()
    cases = KR.select_cases(KR.load_cases(), args.limit_prompts)
    config = {
        "version": VERSION, "scorer": KM.SCORER, "mode": args.mode, "device": str(device),
        "model_id": KR.MODEL_ID if args.model_path is None else None,
        "model_revision": KR.locked_revision() if args.model_path is None else None,
        "model_path": args.model_path, "dtype": "float16", "attn_implementation": KR.ATTN_IMPLEMENTATION,
        "limits": limits, "limited": limited, "output_files": files, "data": data, "n_prompts": len(cases),
        "prompt_ids": [c["id"] for c in cases], "methods": list(KO.METHODS), "seed": KR.SEED,
        "generation": {"settings": KR.GREEDY, "decoder": "argmax loop task2.kv_cache_ops.greedy_decode; the last "
                       "generated token is not fed back", "stop": "EOS ids or max_new_tokens"},
        "prompt_format": "one user message with the supplied chat template (default system message), "
                         "add_generation_prompt=True; context = tokens before the line QUESTION_BEGIN",
        "teacher_forced_session_device": TF_DEVICE,
    }
    if args.mode == "validate":
        config.update(cell_prompt_ids=KR.first_per_cell(cases),
                      reference="model.generate with GenerationConfig(settings), use_model_defaults=False, "
                                "output_logits=True")
    else:
        config.update(method_order_rule=METHOD_ORDER_RULE, qualitative_ids=KR.qualitative_ids(cases),
                      npz_arrays={k: np.dtype(v).name for k, v in NPZ_ARRAYS.items()},
                      npz_format_version=NPZ_FORMAT_VERSION, qualitative_window_tokens=QUAL_WINDOW)
    out.mkdir(parents=True)
    rec = RunRecorder(out / files["run"], config)
    timing = {"load_s": None, "prompt_s": {}}
    try:
        rec.update(environment=KR.environment(device), step="loading")
        t_load = time.perf_counter()
        tokenizer = KR.load_tokenizer(args.model_path)
        model, info = KR.load_model(device, args.model_path)
        model_facts = KR.check_model(model)
        shape = KB.KVShape.from_config(model.config)
        timing["load_s"] = time.perf_counter() - t_load
        extra = {}
        if args.mode == "validate":
            extra["effective_generation_config"] = effective_generation_config(model, generation_config())
        rec.update(model=info, model_facts=model_facts, kv_shape=vars(shape) | {"channels": shape.channels},
                   tokenizer=KR.tokenizer_meta(tokenizer, args.model_path), timing=timing, step="loaded", **extra)
        plain = {k: out / files[k] for k in ("rows", "full_forward", "prompts") if k in files}
        with RowWriter(plain["rows"]) as w_rows:
            second_key = "full_forward" if args.mode == "validate" else "prompts"
            with RowWriter(plain[second_key]) as w_second:
                writers = {"rows": w_rows, second_key: w_second}
                if args.mode == "validate":
                    summary = run_validate(model, tokenizer, shape, cases, device, writers, rec, timing)
                else:
                    summary = run_eval(model, tokenizer, shape, cases, device, out, files, writers, rec, timing)
        row_files = {k: KR.finalize_rows(p) for k, p in plain.items()}
        rec.finish("ok", summary=summary, row_files=row_files, timing=timing, step="done")
    except BaseException as exc:
        rec.finish("failed", timing=timing, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    print(f"{args.mode} done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
