"""Compare task1.evaluate HellaSwag scores with lm_eval 0.4.13 HFLM.loglikelihood on identical requests.

Usage: python -m task1.verify_lmeval --out report.json [--model-path PATH] [--device cuda:0] [--n 512]
"""
import argparse
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
        per_example.append({"order": row["order"], "lmeval_loglik": theirs.tolist(), "max_abs_diff": float(d.max())})
    n = len(ours_rows)
    diffs = np.array(diffs)
    return {"n_examples": n, "n_candidates": int(diffs.size), "max_abs_diff": float(diffs.max()),
            "mean_abs_diff": float(diffs.mean()), "argmax_disagreements_acc": dis_acc,
            "argmax_disagreements_acc_norm": dis_norm, "lmeval_acc": lm_acc / n, "lmeval_acc_norm": lm_norm / n,
            "ours_acc": sum(r["correct_acc"] for r in ours_rows) / n,
            "ours_acc_norm": sum(r["correct_acc_norm"] for r in ours_rows) / n, "per_example": per_example}


def verify(model, tokenizer, records, ours_rows, batch_size: int = C.LMEVAL_BATCH_SIZE,
           softmax_dtypes=(None, "float32")) -> dict:
    """Run HFLM for each softmax dtype setting (None = lm_eval default, the model dtype) and compare."""
    import lm_eval

    assert [r["order"] for r in ours_rows] == [r["order"] for r in records]
    report = {"eval_id": C.EVAL_ID, "lm_eval_version": lm_eval.__version__, "batch_size": batch_size,
              "prompt_format": HELLASWAG_PROMPT_FORMAT, "orders": [r["order"] for r in records], "runs": {}}
    for sd in softmax_dtypes:
        t0 = time.perf_counter()
        lls = lmeval_loglikelihoods(model, tokenizer, records, batch_size, softmax_dtype=sd)
        cmp = compare(ours_rows, lls)
        cmp["wall_s"] = time.perf_counter() - t0
        report["runs"]["softmax_dtype=" + (sd or "default")] = cmp
    return report


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


if __name__ == "__main__":
    main()
