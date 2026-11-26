"""Compare task1.evaluate HellaSwag scores with lm_eval 0.4.13 HFLM.loglikelihood on identical requests.

Usage: python -m task1.verify_lmeval --out report.json [--model-path PATH] [--device cuda:0] [--n 512]

Runs (each compares our summed continuation log likelihoods with HFLM on the same model):
  softmax_dtype=default  the loaded FP16 model, lm_eval default log-softmax (model dtype); informational
  softmax_dtype=float32  the loaded FP16 model, lm_eval FP32 log-softmax; informational
  fp32_model             a deep copy of the loaded model cast to float32 on the same device: task1.evaluate.hellaswag
                         and HFLM (FP32 log-softmax) both on that copy; the acceptance run (config.LMEVAL_ACCEPTANCE_RUN)
On the FP16 model the two scorers batch and pad differently (our scorer passes an attention mask with padding, HFLM
none), so FP16 attention kernels can differ between them; the FP32 copy tests the scoring logic itself. The FP32 copy
is freed (and the CUDA cache emptied) before returning; the loaded model is not modified.
"""
import argparse
import copy
import gc
import time

import numpy as np

from common.io import write_json_new

from . import config as C
from .data import HELLASWAG_PROMPT_FORMAT, format_hellaswag


def lmeval_loglikelihoods(model, tokenizer, records, batch_size: int, softmax_dtype=None) -> list:
    """HFLM(pretrained=model).loglikelihood on the (context, ' ' + choice) requests, 4 per record."""
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, softmax_dtype=softmax_dtype)
    reqs = []
    for rec in records:
        ex = format_hellaswag(rec)
        for j, cont in enumerate(ex["continuations"]):
            reqs.append(Instance(request_type="loglikelihood", doc={"order": ex["order"]},
                                 arguments=(ex["context"], cont), idx=j))
    res = lm.loglikelihood(reqs, disable_tqdm=True)
    return [float(r[0]) for r in res]


def compare(ours_rows, lmeval_lls) -> dict:
    """Per-candidate absolute differences and argmax disagreements for acc and acc_norm."""
    assert len(lmeval_lls) == 4 * len(ours_rows)
    diffs = []
    dis_acc = dis_norm = 0
    lm_acc = lm_norm = 0
    per_example = []
    for i, row in enumerate(ours_rows):
        theirs = np.array(lmeval_lls[4 * i:4 * i + 4], dtype=np.float64)
        ours = np.array(row["loglik"], dtype=np.float64)
        chars = np.array(row["cont_chars"], dtype=np.float64)
        d = np.abs(theirs - ours)
        diffs.extend(d.tolist())
        p, pn = int(np.argmax(theirs)), int(np.argmax(theirs / chars))
        dis_acc += p != row["pred_acc"]
        dis_norm += pn != row["pred_acc_norm"]
        lm_acc += p == row["label"]
        lm_norm += pn == row["label"]
        per_example.append({"order": row["order"], "ours_loglik": ours.tolist(), "lmeval_loglik": theirs.tolist(),
                            "max_abs_diff": float(d.max())})
    n = len(ours_rows)
    diffs = np.array(diffs)
    return {"n_examples": n, "n_candidates": int(diffs.size), "max_abs_diff": float(diffs.max()),
            "mean_abs_diff": float(diffs.mean()), "argmax_disagreements_acc": dis_acc,
            "argmax_disagreements_acc_norm": dis_norm, "lmeval_acc": lm_acc / n, "lmeval_acc_norm": lm_norm / n,
            "ours_acc": sum(r["correct_acc"] for r in ours_rows) / n,
            "ours_acc_norm": sum(r["correct_acc_norm"] for r in ours_rows) / n, "per_example": per_example}


ACCEPT_MAX_ABS_DIFF = 2e-2
ACCEPTANCE_RULE = (
    "PASS if the acceptance run's max absolute per-candidate log-likelihood difference is <= 2e-2 nats and every "
    "acc or acc_norm argmax disagreement has a top-two margin of our scores in that metric (log likelihood for acc, "
    "log likelihood per character for acc_norm) smaller than the observed max difference; otherwise FAIL"
)


def _top_two_margin(scores) -> float:
    s = sorted((float(x) for x in scores), reverse=True)
    return s[0] - s[1]


def acceptance(ours_rows, run: dict, threshold: float = ACCEPT_MAX_ABS_DIFF) -> dict:
    """Acceptance decision for one lm_eval comparison run (see ACCEPTANCE_RULE)."""
    max_diff = run["max_abs_diff"]
    assert len(run["per_example"]) == len(ours_rows)
    dis = []
    for row, ex in zip(ours_rows, run["per_example"]):
        assert ex["order"] == row["order"]
        ours = np.array(row["loglik"], dtype=np.float64)
        theirs = np.array(ex["lmeval_loglik"], dtype=np.float64)
        chars = np.array(row["cont_chars"], dtype=np.float64)
        for metric, so, st in (("acc", ours, theirs), ("acc_norm", ours / chars, theirs / chars)):
            if int(np.argmax(so)) != int(np.argmax(st)):
                m_o, m_t = _top_two_margin(so), _top_two_margin(st)
                dis.append({"order": row["order"], "metric": metric, "ours_argmax": int(np.argmax(so)),
                            "lmeval_argmax": int(np.argmax(st)), "margin_ours": m_o, "margin_lmeval": m_t,
                            "explained": m_o < max_diff})
    diff_ok = max_diff <= threshold
    unexplained = [d for d in dis if not d["explained"]]
    return {"result": "PASS" if diff_ok and not unexplained else "FAIL", "rule": ACCEPTANCE_RULE,
            "threshold_max_abs_diff": threshold, "max_abs_diff": max_diff, "max_abs_diff_ok": diff_ok,
            "n_disagreements": len(dis), "n_unexplained": len(unexplained), "disagreements": dis}


FP32_RUN = "fp32_model"
INFO_SOFTMAX_DTYPES = (None, "float32")
LOG_PREFIX = "[lm_eval check]"
N_LOG_ROWS = 10


def run_name(softmax_dtype) -> str:
    return "softmax_dtype=" + (softmax_dtype or "default")


def _param_dtype(model) -> str:
    return str(next(model.parameters()).dtype)


def lmeval_run(model, tokenizer, records, ours_rows, batch_size: int, softmax_dtype, ours_from: str) -> dict:
    """One HFLM pass compared with ours_rows; the acceptance rule is evaluated for it as well (rule_check)."""
    t0 = time.perf_counter()
    lls = lmeval_loglikelihoods(model, tokenizer, records, batch_size, softmax_dtype=softmax_dtype)
    cmp = compare(ours_rows, lls)
    det = acceptance(ours_rows, cmp)
    cmp.update(model_dtype=_param_dtype(model), softmax_dtype=softmax_dtype or "default", ours_from=ours_from,
               rule_check={k: det[k] for k in ("result", "max_abs_diff_ok", "n_disagreements", "n_unexplained")},
               wall_s=time.perf_counter() - t0)
    return cmp


def fp32_copy_run(model, tokenizer, records, batch_size: int) -> tuple:
    """task1.evaluate.hellaswag and HFLM (FP32 log-softmax) on a float32 deep copy of model on the same device.

    Returns (comparison, rows of task1.evaluate.hellaswag on the copy). The copy is deleted and the CUDA cache emptied
    before returning, also on error."""
    import torch

    from .evaluate import hellaswag

    t0 = time.perf_counter()
    m32 = None
    try:
        m32 = copy.deepcopy(model).to(torch.float32)
        m32.eval()
        rows32, _ = hellaswag(m32, tokenizer, records, scope=f"{FP32_RUN}_first{len(records)}")
        cmp = lmeval_run(m32, tokenizer, records, rows32, batch_size, "float32",
                         ours_from="task1.evaluate.hellaswag on the float32 copy")
    finally:
        del m32
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cmp["wall_s"] = time.perf_counter() - t0
    return cmp, rows32


def verify(model, tokenizer, records, ours_rows, batch_size: int = C.LMEVAL_BATCH_SIZE,
           softmax_dtypes=INFO_SOFTMAX_DTYPES, fp32_model: bool = True) -> dict:
    """HFLM on the given model for each softmax dtype (None = lm_eval default, the model dtype) against ours_rows, then
    (fp32_model) the FP32-copy run; the acceptance rule is applied to config.LMEVAL_ACCEPTANCE_RUN."""
    import lm_eval

    assert [r["order"] for r in ours_rows] == [r["order"] for r in records]
    report = {"eval_id": C.EVAL_ID, "lm_eval_version": lm_eval.__version__, "batch_size": batch_size,
              "prompt_format": HELLASWAG_PROMPT_FORMAT, "orders": [r["order"] for r in records], "runs": {}}
    rows_of = {}
    for sd in softmax_dtypes:
        name = run_name(sd)
        report["runs"][name] = lmeval_run(model, tokenizer, records, ours_rows, batch_size, sd,
                                          ours_from="task1.evaluate.hellaswag on the loaded model")
        rows_of[name] = ours_rows
    if fp32_model:
        report["runs"][FP32_RUN], rows_of[FP32_RUN] = fp32_copy_run(model, tokenizer, records, batch_size)
    assert C.LMEVAL_ACCEPTANCE_RUN in report["runs"], "acceptance run missing"
    report["acceptance_run"] = C.LMEVAL_ACCEPTANCE_RUN
    det = acceptance(rows_of[C.LMEVAL_ACCEPTANCE_RUN], report["runs"][C.LMEVAL_ACCEPTANCE_RUN])
    report["acceptance"] = det["result"]
    report["acceptance_detail"] = det
    return report


def largest_differences(run: dict, k: int = N_LOG_ROWS) -> list:
    """The k largest per-candidate absolute differences of one run: (order, choice, ours, lm_eval, abs diff)."""
    items = []
    for ex in run["per_example"]:
        for j, (o, t) in enumerate(zip(ex["ours_loglik"], ex["lmeval_loglik"])):
            items.append((abs(t - o), ex["order"], j, o, t))
    items.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [{"order": o, "choice": j, "ours": a, "lmeval": b, "abs_diff": d} for d, o, j, a, b in items[:k]]


def report_lines(report: dict) -> list:
    """Log lines: one per run, the verdict, and on FAIL up to N_LOG_ROWS disagreements (unexplained first) and the
    N_LOG_ROWS largest per-candidate differences of the acceptance run. Every line starts with LOG_PREFIX."""
    P = LOG_PREFIX
    lines = []
    for name, r in report["runs"].items():
        role = "acceptance" if name == report["acceptance_run"] else "informational"
        lines.append(f"{P} run {name} ({role}; model {r['model_dtype']}, softmax {r['softmax_dtype']}): "
                     f"max_abs_diff={r['max_abs_diff']:.6g} mean_abs_diff={r['mean_abs_diff']:.6g} "
                     f"disagreements acc={r['argmax_disagreements_acc']} "
                     f"acc_norm={r['argmax_disagreements_acc_norm']} unexplained={r['rule_check']['n_unexplained']} "
                     f"rule={r['rule_check']['result']} n={r['n_examples']}")
    det = report["acceptance_detail"]
    lines.append(f"{P} acceptance {report['acceptance']} (run {report['acceptance_run']}; max_abs_diff "
                 f"{det['max_abs_diff']:.6g} vs threshold {det['threshold_max_abs_diff']:g}; disagreements "
                 f"{det['n_disagreements']}, unexplained {det['n_unexplained']})")
    if report["acceptance"] != "PASS":
        dis = sorted(det["disagreements"], key=lambda d: (d["explained"], d["order"], d["metric"]))
        for d in dis[:N_LOG_ROWS]:
            lines.append(f"{P} disagreement order={d['order']} metric={d['metric']} ours_argmax={d['ours_argmax']} "
                         f"lmeval_argmax={d['lmeval_argmax']} margin_ours={d['margin_ours']:.6g} "
                         f"margin_lmeval={d['margin_lmeval']:.6g} explained={d['explained']}")
        for x in largest_differences(report["runs"][report["acceptance_run"]]):
            lines.append(f"{P} difference order={x['order']} choice={x['choice']} ours={x['ours']:.6f} "
                         f"lmeval={x['lmeval']:.6f} abs_diff={x['abs_diff']:.6g}")
    return lines


def main(argv=None):
    from common.seeding import seed_everything

    from .data import read_hellaswag
    from .evaluate import hellaswag
    from .model import load_model, load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=C.HELLASWAG_SUBSET_N)
    args = ap.parse_args(argv)
    seed_everything(C.SEED)
    model, info = load_model(args.device, args.model_path)
    tok = load_tokenizer(args.model_path)
    records = read_hellaswag(limit=args.n)
    rows, _ = hellaswag(model, tok, records, scope=f"first{args.n}")
    rep = verify(model, tok, records, rows)
    rep["model"] = info
    write_json_new(args.out, rep)
    for line in report_lines(rep):
        print(line, flush=True)


if __name__ == "__main__":
    main()
