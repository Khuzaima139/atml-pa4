"""Development-set selection of the P3 or Q3 allocation from the fixed candidate grid.

Usage: python -m task1.select --kind p3 --target s50|s75 --probe DIR --calib-stats DIR --out DIR [--device cuda:0]
       python -m task1.select --kind q3 --target int4|int3 --probe DIR --calib-stats DIR --out DIR [--device cuda:0]
       python -m task1.select --kind q3v2 --target int4|int3 --probe DIR [DIR ...] --calib-stats DIR --out DIR
       [--model-path PATH --limit-calib N --limit-dev N]   (tests only; refused for outputs under results/task1)
Candidates: P3 = {S1, S2} x P3_LAMBDAS plus lambda 0 (reference, equal to P2); Q3 = {QS1, QS2} knapsack plus
uniform (reference, equal to Q2); Q3 design v2 (q3v2) = {QS1v2, QS2v2} knapsack over fold-group units plus uniform,
from one or more quant_v2 probe parts that together cover every unit exactly once with one FP16 reference value.
Every candidate is built from restore() and scored by its calibration and
development NLL; the winner follows config.SELECTION_CRITERION. Reads only the C4 calibration and development
files; no held-out file can be opened while it runs (task1.data.forbid_held_out).
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

from common.io import sha256_file, write_json_new
from common.runmeta import RunRecorder
from common.seeding import seed_everything

from . import config as C
from . import runargs

OUTPUT_FILES = {
    "run": "run.json",
    "candidates": "candidates.jsonl.gz",
    "candidate_nll": "candidate_nll.jsonl.gz",
    "selected": "selected_allocation.json",
}
KIND_TARGETS = {"p3": C.P3_TARGETS, "q3": C.Q3_TARGETS, "q3v2": C.Q3_TARGETS}
PROBE_KIND = {"p3": "prune", "q3": "quant", "q3v2": "quant_v2"}
MEASURES = {"p3": C.P3_MEASURES, "q3": C.Q3_MEASURES, "q3v2": C.Q3V2_MEASURES}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", required=True, choices=sorted(KIND_TARGETS))
    ap.add_argument("--target", required=True)
    ap.add_argument("--probe", required=True, nargs="+", help="probe directory (q3v2: one or more parts)")
    ap.add_argument("--calib-stats", required=True)
    ap.add_argument("--out", required=True)
    runargs.add_test_args(ap)
    return ap.parse_args(argv)


def matrices_of(model) -> list:
    from .model import enumerate_eligible

    return [{"layer": e.layer, "name": e.name, "qualname": e.qualname, "shape": list(e.module.weight.shape)}
            for e in enumerate_eligible(model)]


def load_probe(probe_dir, kind: str, args, calib_prov: dict) -> tuple:
    """Sensitivity file of a finished probe run on the same model, calibration limit and calib_stats outputs."""
    import json

    from .probe import OUTPUT_FILES as PF

    d = Path(probe_dir)
    prun = json.loads((d / PF["run"]).read_text(encoding="utf-8"))
    if prun.get("status") != "ok":
        raise SystemExit(f"probe run is not finished ok: {prun.get('status')}")
    if prun["config"]["kind"] != PROBE_KIND[kind]:
        raise SystemExit(f"probe kind {prun['config']['kind']} does not serve {kind}")
    if prun["config"]["model_path"] != args.model_path or prun["config"]["limits"]["calib"] != args.limit_calib:
        raise SystemExit("probe run used a different model source or calibration limit")
    sha = sha256_file(d / PF["sensitivity"])
    if sha != prun["steps"]["sensitivity"]["sha256"]:
        raise SystemExit("probe sensitivity sha256 differs from its run record")
    sens = json.loads((d / PF["sensitivity"]).read_text(encoding="utf-8"))
    for k in ("dense_stats_sha256", "awq_scales_sha256"):
        if sens["calib_stats"][k] != calib_prov[k]:
            raise SystemExit(f"probe used different calib_stats outputs ({k})")
    prov = {"dir": str(d), "run_sha256": sha256_file(d / PF["run"]), "sensitivity_sha256": sha,
            "probe_git_commit": prun["meta"]["git_commit"]}
    return sens, prov


def join_probe_parts(parts: list, matrices) -> tuple:
    """Values per unit (unit order) from quant_v2 probe parts; asserts exact coverage and one FP16 reference."""
    from . import q3v2

    units = q3v2.units_of(matrices)
    ids = [u["id"] for u in units]
    seen = {}
    refs = []
    for k, (sens, _) in enumerate(parts):
        if sens.get("design") != C.Q3V2_DESIGN or sens.get("unit") != C.Q3V2_UNIT:
            raise SystemExit(f"probe part {k} is not a {C.Q3V2_DESIGN} probe")
        if sens["n_units_total"] != len(units):
            raise SystemExit(f"probe part {k} was made for {sens['n_units_total']} units, this model has {len(units)}")
        refs.append(float(sens["fp16_calib_nll"]["mean_nll"]))
        for j, u in enumerate(sens["units"]):
            if u["id"] in seen:
                raise SystemExit(f"unit {u['id']} is covered by probe parts {seen[u['id']][0]} and {k}")
            seen[u["id"]] = (k, j)
    missing = [i for i in ids if i not in seen]
    extra = sorted(set(seen) - set(ids))
    if missing or extra:
        raise SystemExit(f"probe parts do not cover the {len(ids)} units exactly once: missing {missing}, unknown "
                         f"{extra}")
    spread = max(refs) - min(refs)
    if spread > C.Q3V2_FP16_REF_TOL:
        raise SystemExit(f"probe parts record different FP16 calibration NLL values {refs} (spread {spread})")
    values = {}
    for m in C.Q3V2_MEASURES:
        values[m] = []
        for i in ids:
            k, j = seen[i]
            v = parts[k][0]["values"][m]
            values[m].append({b: float(v[str(b)][j]) for b in C.Q3V2_BITS})
    return values, {"n_units": len(ids), "n_parts": len(parts), "fp16_calib_mean_nll": refs, "fp16_spread": spread,
                    "part_of_unit": [seen[i][0] for i in ids],
                    "block_ranges": [p[0]["block_range"] for p in parts]}


def q3v2_candidates(values: dict, ref_bits: int, matrices) -> list:
    from . import q3v2

    out = [{"measure": None, "lambda": None, "reference": True, "alloc": q3v2.uniform(matrices, ref_bits)}]
    for m in C.Q3V2_MEASURES:
        out.append({"measure": m, "lambda": None, "reference": False,
                    "alloc": q3v2.allocation(values[m], matrices, ref_bits)})
    return out


def p3_candidates(sens: dict, S: float, matrices) -> list:
    from . import allocate as A
    from .probe import level_key

    vec = {"S1": sens["values"]["S1"], "S2": sens["values"]["S2"][level_key(S)]}
    ref = A.p3_allocation(vec["S1"], S, C.P3_REFERENCE_LAMBDA, matrices)
    assert ref["zeros"] == A.p2_zeros(matrices, S), "lambda 0 does not reproduce P2"
    out = [{"measure": None, "lambda": C.P3_REFERENCE_LAMBDA, "reference": True, "alloc": ref}]
    for m in C.P3_MEASURES:
        for lam in C.P3_LAMBDAS:
            out.append({"measure": m, "lambda": lam, "reference": False, "alloc": A.p3_allocation(vec[m], S, lam, matrices)})
    return out


def q3_candidates(sens: dict, ref_bits: int, matrices) -> list:
    from . import allocate as A

    L = A.n_blocks(matrices)
    out = [{"measure": None, "lambda": None, "reference": True, "alloc": A.uniform_q3(matrices, ref_bits)}]
    for m in C.Q3_MEASURES:
        per_block = [{b: sens["values"][m][str(b)][li] for b in C.Q3_BITS} for li in range(L)]
        out.append({"measure": m, "lambda": None, "reference": False, "alloc": A.q3_allocation(per_block, matrices, ref_bits)})
    return out


def v2_scales(al: dict, scales: dict) -> dict:
    """{(layer, group): scales at the bit width of the group's unit} for a v2 allocation."""
    from . import q3v2

    out = {}
    for uid, b in zip(al["unit_ids"], al["bits"]):
        li, unit = uid.split(".")
        out[(int(li), q3v2.UNIT_GROUP[unit])] = scales[(int(li), q3v2.UNIT_GROUP[unit], int(b))]
    return out


def selected_v2(win, rows, cond, args, tgt, steps, info, lim, meta, mats) -> dict:
    return {"eval_id": C.EVAL_ID, "design": C.Q3V2_DESIGN, "condition": cond, "kind": args.kind,
            "target": args.target, "unit": C.Q3V2_UNIT, "n_units": len(win["per_unit"]["ids"]),
            "n_blocks": len({m["layer"] for m in mats}), "rule": win["rule"], "measure": win["measure"],
            "lambda": None, "candidate_id": win["candidate_id"], "per_unit": win["per_unit"],
            "bits_histogram": win["bits_histogram"], "chunking": win["chunking"], "totals": win["totals"],
            "development": win["development"], "calibration": win["calibration"],
            "selection_criterion": C.Q3V2_SELECTION_CRITERION, "n_candidates": len(rows),
            "n_reference_candidates": sum(r["reference"] for r in rows),
            "collapsed_to_uniform": win["collapsed_to_uniform"],
            "all_candidates_collapsed": all(r["collapsed_to_uniform"] for r in rows if not r["reference"]),
            "inputs": steps["inputs"], "probe_union": steps["probe_union"], "model": info,
            "model_path": args.model_path, "limits": lim["limits"], "code_commit": meta["git_commit"],
            "code_dirty": meta["git_dirty"], "q3v2_settings": C.q3v2_settings(), "ref_bits": tgt["bits"],
            "per_matrix_bits": win["per_matrix_bits"], "objective": win["objective"],
            "objective_exact": win["objective_exact"]}


def candidate_id(cond: str, c: dict) -> str:
    if c["reference"]:
        return f"{cond}.reference"
    return f"{cond}.{c['measure']}" + (f".lambda{c['lambda']:.2f}" if c["lambda"] is not None else "")


def choose(rows: list, kind: str) -> dict:
    """Fixed criterion (config.SELECTION_CRITERION).

    Lowest development mean NLL among non-reference candidates that differ from the reference; ties: smaller
    lambda, then measure order. Only if every non-reference candidate equals the reference is the rule applied
    to those collapsed candidates.
    """
    order = MEASURES[kind]
    pool = [r for r in rows if not r["reference"]]
    assert pool, "no non-reference candidate"
    pool = [r for r in pool if not r["collapsed_to_uniform"]] or pool
    return min(pool, key=lambda r: (r["development"]["mean_nll"], r["lambda"] or 0.0, order.index(r["measure"])))


def run(args):
    if args.target not in KIND_TARGETS[args.kind]:
        raise SystemExit(f"--target for {args.kind} must be one of {sorted(KIND_TARGETS[args.kind])}")
    if args.kind != "q3v2" and len(args.probe) != 1:
        raise SystemExit(f"--kind {args.kind} takes one probe directory")
    lim = runargs.check(args, [args.out])

    import torch

    from . import allocate as A
    from . import storage
    from .build import SCALE_GROUPS, build_condition
    from .calib_stats import load_run, provenance
    from .compress import load_awq_scales
    from .data import SEQUENCE_FILES, file_info, forbid_held_out, read_sequences
    from .evaluate import calib_nll, write_rows
    from .model import Pristine, count_params, load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    tgt = KIND_TARGETS[args.kind][args.target]
    cond = tgt["condition"]
    cfg = {"entry_point": "task1.select", "kind": args.kind, "target": args.target, "target_spec": tgt,
           "condition": cond, "settings": C.settings_dict(), "alloc_settings": C.alloc_settings(),
           "device": args.device, "model_path": args.model_path, **lim, "output_files": OUTPUT_FILES,
           "probe_dir": args.probe if args.kind == "q3v2" else args.probe[0], "calib_stats_dir": args.calib_stats,
           "data_files": [file_info(SEQUENCE_FILES[s]) for s in ("c4_calibration", "c4_development")],
           "batch_size": C.CALIB_BATCH_SIZE}
    if args.kind == "q3v2":
        cfg.update(design=C.Q3V2_DESIGN, q3v2_settings=C.q3v2_settings())
    rec = RunRecorder(out / OUTPUT_FILES["run"], cfg)
    steps, timings = {}, {}
    try:
        with forbid_held_out() as guard:
            t = time.perf_counter()
            model, info = load_model(args.device, args.model_path)
            pristine = Pristine(model)
            steps["restore"] = {"tensors_verified": pristine.restore(model)}
            timings["load_s"] = time.perf_counter() - t
            rec.update(model=info, eval_id=C.EVAL_ID, steps=steps, timings_s=timings)
            dev = next(model.parameters()).device

            load_run(args.calib_stats, args)
            calib_prov = provenance(args.calib_stats)
            if args.kind == "q3v2":
                parts = [load_probe(d, args.kind, args, calib_prov) for d in args.probe]
                probe_prov = [pp for _, pp in parts]
            else:
                sens, probe_prov = load_probe(args.probe[0], args.kind, args, calib_prov)
            steps["inputs"] = {"calib_stats": calib_prov, "probe": probe_prov}
            calib = read_sequences("c4_calibration", limit=args.limit_calib)
            devel = read_sequences("c4_development", limit=args.limit_dev)
            if args.limit_calib is None:
                assert len(calib) == C.N_CALIBRATION
            if args.limit_dev is None:
                assert len(devel) == C.N_DEVELOPMENT

            mats = matrices_of(model)
            shapes = [tuple(m["shape"]) for m in mats]
            counts = count_params(model)
            full_model = shapes == [tuple(s) for s in C.full_model_matrix_shapes()]
            scales = {}
            if args.kind == "p3":
                cands = p3_candidates(sens, tgt["sparsity"], mats)
            elif args.kind == "q3v2":
                values, steps["probe_union"] = join_probe_parts(parts, mats)
                cands = q3v2_candidates(values, tgt["bits"], mats)
                if full_model:
                    assert cands[0]["alloc"]["budget"] == C.Q3_FULL_MODEL_BUDGET_BYTES[args.target]
                    ch = cands[0]["alloc"]["chunking"]
                    assert ch["full_model_expected"], ch
                for b in C.Q3V2_BITS:
                    sc, _ = load_awq_scales(args.calib_stats, b, args, dev)
                    scales.update({(li, g, b): v for (li, g), v in sc.items()})
            else:
                cands = q3_candidates(sens, tgt["bits"], mats)
                if full_model:
                    assert cands[0]["alloc"]["budget"] == C.Q3_FULL_MODEL_BUDGET_BYTES[args.target]
                for b in C.Q3_BITS:
                    sc, _ = load_awq_scales(args.calib_stats, b, args, dev)
                    scales.update({(li, g, b): v for (li, g), v in sc.items()})
            steps["n_candidates"] = {"total": len(cands), "reference": sum(c["reference"] for c in cands)}

            rows, nll_rows = [], []
            for i, c in enumerate(cands):
                cid = candidate_id(cond, c)
                al = c["alloc"]
                t0 = time.perf_counter()
                pristine.restore(model)
                t_restore = time.perf_counter() - t0
                t0 = time.perf_counter()
                if args.kind == "p3":
                    res = build_condition(model, calib, cond, C.CALIB_BATCH_SIZE, zeros_per_matrix=al["zeros"],
                                          pristine=pristine, with_distributions=False)
                    assert sum(r["mask_zeros"] for r in res.records) == al["p2_total_zeros"]
                    assert {r["qualname"]: r["mask_zeros"] for r in res.records} == al["zeros"]
                elif args.kind == "q3v2":
                    aw = v2_scales(al, scales)
                    res = build_condition(model, calib, cond, C.CALIB_BATCH_SIZE, awq_scales=aw,
                                          bits_per_matrix=al["per_matrix_bits"], pristine=pristine)
                    assert [r["bits"] for r in res.records] == list(al["per_matrix_bits"].values())
                else:
                    bits = al["bits"]
                    aw = {(li, g): scales[(li, g, b)] for li, b in enumerate(bits) for g in SCALE_GROUPS}
                    res = build_condition(model, calib, cond, C.CALIB_BATCH_SIZE, awq_scales=aw,
                                          bits_per_block=bits, pristine=pristine)
                    assert [r["bits"] for r in res.records] == list(al["per_matrix_bits"].values())
                build_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                summ = {}
                for split, seqs in (("c4_development", devel), ("c4_calibration", calib)):
                    nr, summ[split] = calib_nll(model, seqs, split=split)
                    nll_rows.extend({**r, "candidate_id": cid} for r in nr)
                nll_s = time.perf_counter() - t0
                err_num = sum(r["err_num"] for r in res.records)
                err_den = sum(r["err_den"] for r in res.records)
                row = {"eval_id": C.EVAL_ID, "candidate_id": cid, "index": i, "condition": cond, "kind": args.kind,
                       "target": args.target, "reference": c["reference"], "measure": c["measure"],
                       "lambda": c["lambda"], "rule": al["rule"], "development": summ["c4_development"],
                       "calibration": summ["c4_calibration"], "output_error_pooled_rel": err_num / err_den,
                       "timings_s": {"restore": t_restore, "build": build_s, "nll": nll_s,
                                     **{f"build_{k}": v for k, v in res.timings.items()}}}
                if args.kind == "p3":
                    row.update(per_block={k: al[k] for k in ("sensitivity", "u", "raw", "block_sparsity_real",
                                                             "block_sparsity_achieved")},
                               shift=al["shift"], bisect_iterations=al["bisect_iterations"],
                               bisect_residual=al["bisect_residual"],
                               n_largest_remainder_added=al["n_largest_remainder_added"],
                               per_matrix_zeros=al["zeros"], collapsed_to_uniform=al["collapsed_to_uniform"],
                               totals={"zeros": al["total_zeros"], "p2_zeros": al["p2_total_zeros"],
                                       "eligible_weights": al["eligible_weights"]})
                elif args.kind == "q3v2":
                    nb = storage.packed_quant_bytes(shapes, list(al["per_matrix_bits"].values()))
                    assert nb == al["bytes"] <= al["budget"]
                    row.update(design=C.Q3V2_DESIGN,
                               per_unit={"ids": al["unit_ids"], "bits": al["bits"], "bytes": al["unit_bytes"],
                                         "sensitivity": al.get("sensitivity")},
                               bits_histogram=al["bits_histogram"], chunking=al["chunking"],
                               per_matrix_bits=al["per_matrix_bits"], collapsed_to_uniform=al["collapsed_to_uniform"],
                               objective=al.get("objective"), objective_exact=al.get("objective_exact"),
                               dp_states_per_unit=al.get("dp_states_per_unit"),
                               totals={"eligible_packed_bytes": nb, "budget_bytes": al["budget"],
                                       "cost_units": al["cost_units"], "budget_units": al["budget_units"],
                                       "ref_bits": al["ref_bits"], "non_eligible_params": counts["non_eligible"],
                                       "whole_model_bytes": storage.whole_model_bytes(nb, counts["non_eligible"]),
                                       "budget_whole_model_bytes": storage.whole_model_bytes(al["budget"],
                                                                                             counts["non_eligible"]),
                                       "bits_per_eligible_weight": storage.bits_per_eligible_weight(nb, shapes)})
                else:
                    nb = storage.packed_quant_bytes(shapes, list(al["per_matrix_bits"].values()))
                    assert nb == al["bytes"] <= al["budget"]
                    row.update(per_block={"bits": al["bits"], "bytes": al["block_bytes"],
                                          "sensitivity": al.get("sensitivity")},
                               per_matrix_bits=al["per_matrix_bits"], collapsed_to_uniform=al["collapsed_to_uniform"],
                               objective=al.get("objective"), objective_exact=al.get("objective_exact"),
                               dp_states_per_block=al.get("dp_states_per_block"),
                               totals={"eligible_packed_bytes": nb, "budget_bytes": al["budget"],
                                       "ref_bits": al["ref_bits"], "non_eligible_params": counts["non_eligible"],
                                       "whole_model_bytes": storage.whole_model_bytes(nb, counts["non_eligible"]),
                                       "budget_whole_model_bytes": storage.whole_model_bytes(al["budget"],
                                                                                             counts["non_eligible"]),
                                       "bits_per_eligible_weight": storage.bits_per_eligible_weight(nb, shapes)})
                rows.append(row)
                steps["progress"] = {"done": i + 1, "of": len(cands), "last": cid}
                rec.update(steps=steps, timings_s=timings)
            pristine.restore(model)
            steps.pop("progress", None)

            win = choose(rows, args.kind)
            for r in rows:
                r["selected"] = r["candidate_id"] == win["candidate_id"]
            write_rows(out / OUTPUT_FILES["candidates"], rows)
            write_rows(out / OUTPUT_FILES["candidate_nll"], nll_rows)
            meta = rec.record["meta"]
            if args.kind == "q3v2":
                sel = selected_v2(win, rows, cond, args, tgt, steps, info, lim, meta, mats)
            else:
                sel = None
            sel = sel or {"eval_id": C.EVAL_ID, "condition": cond, "kind": args.kind, "target": args.target,
                   "unit": C.ALLOC_UNIT, "n_blocks": A.n_blocks(mats), "rule": win["rule"],
                   "measure": win["measure"], "lambda": win["lambda"], "candidate_id": win["candidate_id"],
                   "per_block": win["per_block"], "totals": win["totals"],
                   "development": win["development"], "calibration": win["calibration"],
                   "selection_criterion": C.SELECTION_CRITERION, "n_candidates": len(rows),
                   "n_reference_candidates": sum(r["reference"] for r in rows),
                   "collapsed_to_uniform": win["collapsed_to_uniform"],
                   "all_candidates_collapsed": all(r["collapsed_to_uniform"] for r in rows if not r["reference"]),
                   "inputs": steps["inputs"], "model": info, "model_path": args.model_path, "limits": lim["limits"],
                   "code_commit": meta["git_commit"], "code_dirty": meta["git_dirty"],
                   "alloc_settings": C.alloc_settings()}
            if args.kind == "p3":
                sel.update(sparsity=tgt["sparsity"], per_matrix_zeros=win["per_matrix_zeros"], shift=win["shift"])
            elif args.kind == "q3v2":
                pass
            else:
                sel.update(ref_bits=tgt["bits"], per_matrix_bits=win["per_matrix_bits"],
                           objective=win["objective"])
            write_json_new(out / OUTPUT_FILES["selected"], sel)
            steps["selection"] = {"winner": win["candidate_id"], "development_mean_nll": win["development"]["mean_nll"],
                                  "file": OUTPUT_FILES["selected"], "sha256": sha256_file(out / OUTPUT_FILES["selected"]),
                                  "candidates": [{"candidate_id": r["candidate_id"], "reference": r["reference"],
                                                  "development_mean_nll": r["development"]["mean_nll"],
                                                  "calibration_mean_nll": r["calibration"]["mean_nll"]} for r in rows]}
            steps["selection"]["collapsed_to_uniform"] = [r["candidate_id"] for r in rows
                                                          if r["collapsed_to_uniform"] and not r["reference"]]
            steps["selection"]["winner_collapsed_to_uniform"] = win["collapsed_to_uniform"]
            steps["selection"]["all_candidates_collapsed"] = sel["all_candidates_collapsed"]
            steps["pristine_after"] = {"tensors_verified": pristine.verify(model)}
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                steps["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(dev)
        steps["held_out_guard"] = guard
        assert not guard["attempts"], guard
        rec.finish("ok", steps=steps, timings_s=timings)
    except BaseException as exc:
        rec.finish("failed", steps=steps, timings_s=timings, error=repr(exc), traceback=traceback.format_exc())
        raise
    return out


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
