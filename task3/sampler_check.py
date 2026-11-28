"""Exactness check of the speculative sampler on a three-symbol Markov toy (protocol Section 1, order of work 1).

Target and draft are first-order Markov chains (initial distribution plus transition matrix). Sequences of length 3
are drawn with ``task3.spec.speculative_generate`` at gamma 2 and 4, once with the positive residual (the exact
sampler) and once with the residual replaced by the target distribution (negative control), 100,000 fixed-seed
sequences each; sequence i uses a fresh generator seeded with SEED_BASE[gamma] + i, shared by both samplers.
The Pearson chi-squared statistic over the 27 sequences (df 26) is computed against the exact target joint law.
The exact output law of each sampler variant is also computed by enumeration of the round process.

    python -m task3.sampler_check --out results/task3/sampler_check.json
"""
import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.paths import RESULTS_DIR
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from .lm import MarkovLM
from .sampling import SAMPLER_ID, make_generator
from .spec import speculative_generate, summarize_rounds

CHECK_ID = "t3-sampler-check-v1"
LENGTH = 3
N_SEQUENCES = 100_000
GAMMAS = (2, 4)
ALPHA = 0.01
MIN_EXPECTED = 10
MIN_JOINT = 2e-4
MIN_EXPECTED_STAT_INCORRECT = 200.0
SEED_BASE = {2: 6_306_000_000, 4: 6_308_000_000}
SAMPLERS = {"correct": "target_minus_draft", "incorrect": "target"}

TARGET = {
    "initial": [0.5, 0.3, 0.2],
    "transition": [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.25, 0.25, 0.5]],
}
DRAFT = {
    "initial": [0.2, 0.3, 0.5],
    "transition": [[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.4, 0.4, 0.2]],
}


def _next(model, ctx):
    return np.asarray(model["initial"] if not ctx else model["transition"][ctx[-1]], dtype=np.float64)


def target_joint(target=TARGET, length=LENGTH) -> dict:
    out = {}
    for seq in itertools.product(range(3), repeat=length):
        pr = 1.0
        for t in range(length):
            pr *= _next(target, seq[:t])[seq[t]]
        out[seq] = pr
    return out


def _round_law(prefix, gamma, length, residual, target, draft):
    """Law of the tokens emitted by one round from ``prefix`` (no EOS), as {emitted tuple: probability}."""
    remaining = length - len(prefix)
    k = min(gamma, remaining)
    res = defaultdict(float)

    def rec(i, props, prob):
        ctx = prefix + props
        p = _next(target, ctx)
        if i == k:
            if k < remaining:
                for y in range(3):
                    res[props + (y,)] += prob * p[y]
            else:
                res[props] += prob
            return
        q = _next(draft, ctx)
        acc = np.minimum(1.0, p / q)
        for x in range(3):
            rec(i + 1, props + (x,), prob * q[x] * acc[x])
        rej = float(np.sum(q * (1.0 - acc)))
        if rej > 0:
            w = np.maximum(p - q, 0.0) if residual == "target_minus_draft" else p
            w = w / w.sum()
            for y in range(3):
                res[props + (y,)] += prob * rej * w[y]

    rec(0, (), 1.0)
    return res


def sampler_joint(gamma, residual, target=TARGET, draft=DRAFT, length=LENGTH) -> dict:
    """Exact joint law of the length-``length`` output of the speculative round process."""
    law = defaultdict(float)
    frontier = {(): 1.0}
    while frontier:
        nxt = defaultdict(float)
        for prefix, pr in frontier.items():
            for em, pe in _round_law(prefix, gamma, length, residual, target, draft).items():
                seq = prefix + em
                if len(seq) >= length:
                    law[seq[:length]] += pr * pe
                else:
                    nxt[seq] += pr * pe
        frontier = nxt
    return dict(law)


def run_sampler(gamma, residual, n, target=TARGET, draft=DRAFT, length=LENGTH):
    """Draw n sequences through the real sampler; returns sequence counts and pooled round counts."""
    t_lm = MarkovLM(target["initial"], target["transition"])
    d_lm = MarkovLM(draft["initial"], draft["transition"])
    counts = Counter()
    pooled = Counter()
    base = SEED_BASE[gamma]
    for i in range(n):
        gen = make_generator(base + i, "cpu")
        res = speculative_generate(t_lm, d_lm, [t_lm.start], gamma, length, (), gen, residual=residual)
        if len(res["output_ids"]) != length:
            raise RuntimeError(f"sequence {i}: length {len(res['output_ids'])}")
        counts[tuple(res["output_ids"])] += 1
        s = summarize_rounds(res["rounds"], res["output_tokens"])
        for key in ("proposed", "accepted", "examined", "target_verification_passes", "draft_forward_calls",
                    "bonus_tokens", "bonus_skipped_cap", "residual_fallbacks"):
            pooled[key] += s[key]
    return counts, dict(pooled)


def pearson(counts, joint, n):
    from scipy.stats import chisquare

    seqs = sorted(joint)
    obs = np.array([counts.get(s, 0) for s in seqs], dtype=np.float64)
    p = np.array([joint[s] for s in seqs], dtype=np.float64)
    exp = n * p / p.sum()
    r = chisquare(obs, exp)
    return float(r.statistic), float(r.pvalue), len(seqs) - 1


def expected_statistic(law, joint, n):
    """Mean of the Pearson statistic when sequences follow ``law``: df + n * sum((law - joint)^2 / joint)."""
    seqs = sorted(joint)
    lam = n * sum((law.get(s, 0.0) - joint[s]) ** 2 / joint[s] for s in seqs)
    return float(len(seqs) - 1 + lam)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_SEQUENCES)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if a.n != N_SEQUENCES and out.resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise SystemExit(f"--n other than {N_SEQUENCES} is refused under results/")
    seed_everything(6304)
    n = a.n
    joint = target_joint()
    seqs = sorted(joint)
    config = {
        "check_id": CHECK_ID, "sampler_id": SAMPLER_ID, "symbols": 3, "length": LENGTH, "n_sequences": n,
        "gammas": list(GAMMAS), "alpha": ALPHA, "target": TARGET, "draft": DRAFT,
        "start_symbol": 3, "eos_ids": [], "cap": LENGTH,
        "seed_rule": "sequence i at gamma g uses torch.Generator('cpu').manual_seed(seed_base[g] + i), "
                     "the same seeds for both samplers",
        "seed_base": {str(g): b for g, b in SEED_BASE.items()},
        "samplers": {"correct": "positive residual norm(max(0, p - q))",
                     "incorrect": "residual replaced by p (negative control); otherwise the same code"},
        "test": "Pearson chi-squared over the 27 length-3 sequences, df 26, scipy.stats.chisquare",
        "batch_size": 1, "padding": "none (batch size 1, no padding exists)",
        "note_gamma4": "at length 3 a gamma-4 round proposes min(4, remaining) tokens (D16), so its first round "
                       "proposes 3 and the bonus token is skipped at the cap",
    }
    rec = RunRecorder(out, config)
    exact = {}
    for g in GAMMAS:
        for name, residual in SAMPLERS.items():
            exact[(name, g)] = sampler_joint(g, residual)
    min_joint = min(joint.values())
    exact_correct_maxdiff = max(abs(exact[("correct", g)].get(s, 0.0) - joint[s]) for g in GAMMAS for s in seqs)
    exp_stat_incorrect = {str(g): expected_statistic(exact[("incorrect", g)], joint, n) for g in GAMMAS}
    rec.update(min_target_joint=min_joint, min_expected_count=n * min_joint,
               exact_correct_law_max_abs_diff=exact_correct_maxdiff,
               expected_statistic_incorrect=exp_stat_incorrect)

    counts, pooled, stats = {}, {}, {}
    for g in GAMMAS:
        for name, residual in SAMPLERS.items():
            c, pc = run_sampler(g, residual, n)
            counts[(name, g)] = c
            stat, pval, df = pearson(c, joint, n)
            key = f"{name}_gamma{g}"
            stats[key] = {"statistic": stat, "p_value": pval, "df": df}
            pooled[key] = pc
            rec.update(statistics=stats, pooled_round_counts=pooled)

    table = []
    for s in seqs:
        row = {"sequence": "".join("ABC"[t] for t in s), "symbols": list(s), "target_probability": joint[s],
               "expected_count": n * joint[s]}
        for g in GAMMAS:
            for name in SAMPLERS:
                row[f"observed_{name}_gamma{g}"] = counts[(name, g)].get(s, 0)
                row[f"exact_probability_{name}_gamma{g}"] = exact[(name, g)].get(s, 0.0)
        table.append(row)

    passes = {
        "min_expected_count_ge_10": bool(n * min_joint >= MIN_EXPECTED),
        "min_target_joint_ge_2e-4": bool(min_joint >= MIN_JOINT),
        "exact_correct_law_equals_target": bool(exact_correct_maxdiff < 1e-12),
    }
    for g in GAMMAS:
        passes[f"expected_statistic_incorrect_gamma{g}_gt_200"] = bool(
            exp_stat_incorrect[str(g)] > MIN_EXPECTED_STAT_INCORRECT)
        passes[f"correct_gamma{g}_not_rejected"] = bool(stats[f"correct_gamma{g}"]["p_value"] >= ALPHA)
        passes[f"incorrect_gamma{g}_rejected"] = bool(stats[f"incorrect_gamma{g}"]["p_value"] < ALPHA)
    all_pass = all(passes.values())
    rec.finish("ok", table=table, passes=passes, all_pass=all_pass)
    print(f"all_pass={all_pass} " + " ".join(f"{k}: p={v['p_value']:.4g} chi2={v['statistic']:.1f}"
                                              for k, v in stats.items()))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
