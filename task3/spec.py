"""Exact speculative sampling and the target-only reference, sharing the D3 sampling function.

Cache invariant at the start of every round (and of every target-only step), with ctx = prompt + output so far:
the target cache holds ctx[:-1] (length len(ctx) - 1) and the draft cache holds a prefix of ctx[:-1]. The last
context token is fed with the next forward call, so the prefill feeds prompt[:-1].

One round (protocol D16, Section 3): with r the remaining allowance, propose k = min(gamma, r) draft tokens (the
first draft call also feeds the context tokens the draft cache lacks, i.e. the catch-up after a fully accepted
round); verify with one target forward over the pending context token and the k proposals; accept proposal i with
probability min(1, p_i(x_i) / q_i(x_i)); at the first rejection draw from the positive residual
norm(max(0, p_i - q_i)); after full acceptance draw the bonus token from p_{k+1} unless it would exceed the cap.
Tokens after the first EOS are discarded. Both caches are then cropped to len(ctx) - 1.

``residual="target"`` replaces the residual by p_i; it exists only as the negative control of the sampler check.
"""
import torch

from .sampling import Timer, probs_from_logits, sample_from_weights, sample_token, uniform

RESIDUALS = ("target_minus_draft", "target")


def _prefill(lm, prompt_ids):
    lm.reset()
    if len(prompt_ids) > 1:
        lm.forward(prompt_ids[:-1])


def _termination(out, eos, cap):
    if out and out[-1] in eos:
        return "eos"
    return "cap"


def target_only_generate(target, prompt_ids, cap, eos_ids, generator, timer=None):
    """Plain autoregressive sampling from the target with the shared law; one target forward per output token."""
    prompt_ids = [int(t) for t in prompt_ids]
    if not prompt_ids:
        raise ValueError("empty prompt")
    eos = set(int(e) for e in eos_ids)
    timer = timer or Timer(target.device)
    ctx = list(prompt_ids)
    out, fwd_s, smp_s = [], [], []
    with torch.inference_mode():
        t_start = timer.now()
        _prefill(target, ctx)
        t_prefill = timer.now()
        while len(out) < cap:
            assert target.length == len(ctx) - 1
            a = timer.now()
            logits = target.forward(ctx[target.length:])
            b = timer.now()
            tok = sample_token(logits[-1], generator)
            c = timer.now()
            out.append(tok)
            ctx.append(tok)
            fwd_s.append(b - a)
            smp_s.append(c - b)
            if tok in eos:
                break
        t_end = timer.now()
    total = t_end - t_start
    prefill = t_prefill - t_start
    return {
        "output_ids": out,
        "termination": _termination(out, eos, cap),
        "output_tokens": len(out),
        "prompt_tokens": len(prompt_ids),
        "target_forward_calls": len(out),
        "prefill_s": prefill,
        "step_forward_s": fwd_s,
        "step_sample_s": smp_s,
        "decode_s": t_end - t_prefill,
        "total_s": total,
        "host_s": total - prefill - sum(fwd_s) - sum(smp_s),
        "target_cache_len_end": target.length,
    }


def speculative_generate(target, draft, prompt_ids, gamma, cap, eos_ids, generator,
                         residual="target_minus_draft", timer=None, round_callback=None):
    """Exact speculative sampling; returns the output tokens, one record per round and pooled counts."""
    if residual not in RESIDUALS:
        raise ValueError(f"residual must be one of {RESIDUALS}")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    prompt_ids = [int(t) for t in prompt_ids]
    if not prompt_ids:
        raise ValueError("empty prompt")
    eos = set(int(e) for e in eos_ids)
    device = target.device
    timer = timer or Timer(device)
    ctx = list(prompt_ids)
    out, rounds = [], []
    termination = None
    with torch.inference_mode():
        t_start = timer.now()
        _prefill(target, ctx)
        t_pt = timer.now()
        _prefill(draft, ctx)
        t_pd = timer.now()
        while termination is None:
            remaining = cap - len(out)
            if remaining <= 0:
                termination = "cap"
                break
            n = len(ctx)
            if target.length != n - 1 or draft.length > n - 1:
                raise RuntimeError(f"cache invariant broken: ctx {n}, target {target.length}, draft {draft.length}")
            k = min(gamma, remaining)

            ta = timer.now()
            catchup = n - draft.length
            feed = ctx[draft.length:]
            qs, xs = [], []
            draft_calls = draft_fed = 0
            for _ in range(k):
                logits = draft.forward(feed)
                draft_calls += 1
                draft_fed += len(feed)
                q = probs_from_logits(logits[-1])
                x = sample_from_weights(q, generator)
                qs.append(q)
                xs.append(x)
                feed = [x]
            tb = timer.now()

            vfeed = ctx[target.length:] + xs
            ps = probs_from_logits(target.forward(vfeed)[-(k + 1):])
            tc = timer.now()

            flags, emitted = [], []
            rejected = fallback = False
            for i in range(k):
                x = xs[i]
                px = float(ps[i, x].item())
                qx = float(qs[i][x].item())
                if uniform(generator, device) * qx < px:
                    flags.append(True)
                    emitted.append(x)
                    continue
                flags.append(False)
                rejected = True
                if residual == "target_minus_draft":
                    w = (ps[i] - qs[i]).clamp_min(0.0)
                    if not float(w.sum().item()) > 0.0:
                        w, fallback = ps[i], True
                else:
                    w = ps[i]
                emitted.append(sample_from_weights(w, generator))
                break
            bonus = bonus_skipped = False
            if not rejected:
                if k < remaining:
                    emitted.append(sample_from_weights(ps[k], generator))
                    bonus = True
                else:
                    bonus_skipped = True
            n_raw = len(emitted)
            eos_hit = False
            for j, tok in enumerate(emitted):
                if tok in eos:
                    emitted = emitted[:j + 1]
                    eos_hit = True
                    break
            td = timer.now()

            out.extend(emitted)
            ctx.extend(emitted)
            keep = len(ctx) - 1
            if target.length > keep:
                target.crop(keep)
            if draft.length > keep:
                draft.crop(keep)
            te = timer.now()

            accepted = sum(flags)
            rec = {
                "round": len(rounds),
                "ctx_len_start": n,
                "remaining": remaining,
                "proposed": k,
                "accepted": accepted,
                "examined": accepted + (1 if rejected else 0),
                "rejected": rejected,
                "accept_flags": flags,
                "bonus": bonus,
                "bonus_skipped_cap": bonus_skipped,
                "residual_fallback": fallback,
                "emitted": len(emitted),
                "discarded": n_raw - len(emitted),
                "eos": eos_hit,
                "draft_forward_calls": draft_calls,
                "draft_tokens_fed": draft_fed,
                "draft_catchup_tokens": catchup,
                "target_verification_passes": 1,
                "target_tokens_fed": len(vfeed),
                "draft_s": tb - ta,
                "verify_s": tc - tb,
                "accept_s": td - tc,
                "cache_s": te - td,
                "target_cache_len_end": target.length,
                "draft_cache_len_end": draft.length,
            }
            rounds.append(rec)
            if round_callback is not None:
                round_callback(rec, list(ctx), list(out))
            if eos_hit:
                termination = "eos"
            elif len(out) >= cap:
                termination = "cap"
        t_end = timer.now()
    total = t_end - t_start
    prefill_t, prefill_d = t_pt - t_start, t_pd - t_pt
    res = {
        "output_ids": out,
        "termination": termination,
        "output_tokens": len(out),
        "prompt_tokens": len(prompt_ids),
        "gamma": gamma,
        "cap": cap,
        "residual": residual,
        "rounds": rounds,
        "prefill_target_s": prefill_t,
        "prefill_draft_s": prefill_d,
        "decode_s": t_end - t_pd,
        "total_s": total,
    }
    res.update(summarize_rounds(rounds, len(out)))
    res["host_s"] = total - prefill_t - prefill_d - sum(
        res[f"{c}_s_total"] for c in ("draft", "verify", "accept", "cache"))
    return res


def summarize_rounds(rounds, output_tokens) -> dict:
    """Pooled counts over rounds (protocol Section 3)."""
    s = {key: sum(r[key] for r in rounds) for key in (
        "proposed", "accepted", "examined", "target_verification_passes", "draft_forward_calls",
        "draft_tokens_fed", "target_tokens_fed", "discarded")}
    s["n_rounds"] = len(rounds)
    s["residual_fallbacks"] = sum(1 for r in rounds if r["residual_fallback"])
    s["bonus_tokens"] = sum(1 for r in rounds if r["bonus"])
    s["bonus_skipped_cap"] = sum(1 for r in rounds if r["bonus_skipped_cap"])
    for c in ("draft", "verify", "accept", "cache"):
        s[f"{c}_s_total"] = sum(r[f"{c}_s"] for r in rounds)
    v = s["target_verification_passes"]
    s["mean_accepted_prefix_per_verification"] = (s["accepted"] / v) if v else None
    s["target_passes_per_output_token"] = (v / output_tokens) if output_tokens else None
    s["accepted_per_examined"] = (s["accepted"] / s["examined"]) if s["examined"] else None
    s["accepted_per_proposed"] = (s["accepted"] / s["proposed"]) if s["proposed"] else None
    return s


def cache_next_logits(lm, ctx) -> torch.Tensor:
    """Next-token logits after ctx from the cache (feeds the missing tokens, then restores the cache length)."""
    before = lm.length
    with torch.inference_mode():
        logits = lm.forward(ctx[before:])[-1]
    lm.crop(before)
    return logits
