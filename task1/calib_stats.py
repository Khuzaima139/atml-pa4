"""Dense calibration statistics and the AWQ scale search for b in {2, 3, 4, 8}, with the preservation check.

Usage: python -m task1.calib_stats --out DIR [--device cuda:0]
       [--model-path PATH --limit-calib N]   (tests only; refused for outputs under results/task1)
Reads only the 128-sequence C4 calibration file.
"""
import argparse
import json
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
    "dense_stats": "dense_stats.safetensors",
    "dense_stats_manifest": "dense_stats.manifest.json",
    "awq_scales": "awq_scales.safetensors",
    "awq_scales_manifest": "awq_scales.manifest.json",
    "awq_search": "awq_search.json",
}


def scale_name(layer: int, group: str, bits: int) -> str:
    return f"layer{layer}.{group}.b{bits}"


def load_run(calib_dir, args) -> dict:
    """run.json of a finished calib_stats run made with the same model source and calibration limit."""
    crun = json.loads((Path(calib_dir) / OUTPUT_FILES["run"]).read_text(encoding="utf-8"))
    if crun.get("status") != "ok":
        raise SystemExit(f"calib_stats run is not finished ok: {crun.get('status')}")
    if crun["config"]["model_path"] != args.model_path or crun["config"]["limits"]["calib"] != args.limit_calib:
        raise SystemExit("calib_stats run used a different model source or calibration limit")
    return crun


def provenance(calib_dir) -> dict:
    """sha256 of the calib_stats outputs that later steps read, checked against the run record."""
    d = Path(calib_dir)
    crun = json.loads((d / OUTPUT_FILES["run"]).read_text(encoding="utf-8"))
    out = {"dir": str(d), "run_sha256": sha256_file(d / OUTPUT_FILES["run"]),
           "calib_git_commit": crun["meta"]["git_commit"]}
    for k in ("dense_stats", "awq_scales"):
        got = sha256_file(d / OUTPUT_FILES[k])
        if got != crun["steps"]["files"][k]["sha256"]:
            raise SystemExit(f"calib_stats {k} sha256 differs from its run record")
        out[f"{k}_sha256"] = got
    out["awq_search_sha256"] = sha256_file(d / OUTPUT_FILES["awq_search"])
    return out


def load_dense_l2(calib_dir, args, device) -> tuple:
    """Dense per-channel input norms {qualname: x_l2} of every eligible projection, with provenance."""
    from .tensorfile import load_tensors

    load_run(calib_dir, args)
    d = Path(calib_dir)
    tensors, man = load_tensors(d / OUTPUT_FILES["dense_stats"], d / OUTPUT_FILES["dense_stats_manifest"], device=device)
    prov = provenance(d)
    assert man["file"]["sha256"] == prov["dense_stats_sha256"]
    return {k[:-len(".x_l2")]: v for k, v in tensors.items() if k.endswith(".x_l2")}, prov


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    runargs.add_test_args(ap)
    return ap.parse_args(argv)


def run(args):
    if args.limit_dev is not None:
        raise SystemExit("--limit-dev is not used by calib_stats")
    lim = runargs.check(args, [args.out])

    import torch

    from . import awq
    from .blocks import capture_block_inputs
    from .data import SEQUENCE_FILES, file_info, read_sequences
    from .model import Pristine, enumerate_eligible, load_model
    from .tensorfile import save_tensors

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(C.SEED)
    cfg = {"entry_point": "task1.calib_stats", "settings": C.settings_dict(), "device": args.device,
           "model_path": args.model_path, **lim, "output_files": OUTPUT_FILES,
           "data_files": [file_info(SEQUENCE_FILES["c4_calibration"])],
           "awq": {"alpha_grid": list(awq.ALPHA_GRID), "bits": list(awq.SEARCH_BITS), "groups": awq.GROUPS,
                   "convention": awq.AWQ_CONVENTION, "scale_floor": awq.SCALE_FLOOR, "token_chunk": awq.TOKEN_CHUNK},
           "batch_size": C.CALIB_BATCH_SIZE}
    rec = RunRecorder(out / OUTPUT_FILES["run"], cfg)
    steps, timings = {}, {}
    try:
        t = time.perf_counter()
        model, info = load_model(args.device, args.model_path)
        pristine = Pristine(model)
        steps["restore"] = {"tensors_verified": pristine.restore(model)}
        timings["load_s"] = time.perf_counter() - t
        rec.update(model=info, eval_id=C.EVAL_ID, steps=steps, timings_s=timings)

        seqs = read_sequences("c4_calibration", limit=args.limit_calib)
        if args.limit_calib is None:
            assert len(seqs) == C.N_CALIBRATION
        n_tokens = sum(len(s["token_ids"]) for s in seqs)
        t = time.perf_counter()
        stream = capture_block_inputs(model, seqs, C.CALIB_BATCH_SIZE)
        timings["capture_s"] = time.perf_counter() - t
        steps["calibration"] = {"split": "c4_calibration", "n_sequences": len(seqs), "n_tokens": n_tokens,
                                "source_rows_first_last": [seqs[0]["source_row"], seqs[-1]["source_row"]]}
        rec.update(steps=steps, timings_s=timings)

        def log(li, tb):
            steps.setdefault("block_timings", []).append(tb)
            rec.update(steps=steps, timings_s=timings)

        t = time.perf_counter()
        res = awq.search_model(model, stream, log=log)
        timings["search_total_s"] = time.perf_counter() - t
        del stream

        elig = enumerate_eligible(model)
        stats_t = {}
        for e in elig:
            st = res["stats"][(e.layer, e.name)]
            assert st["n_tokens"] == n_tokens
            stats_t[e.qualname + ".x_l2"] = st["l2"]
            stats_t[e.qualname + ".x_absmean"] = st["absmean"]
        t = time.perf_counter()
        meta = {"eval_id": C.EVAL_ID, "split": "c4_calibration", "n_sequences": len(seqs), "n_tokens": n_tokens,
                "model": info, "what": "per input channel of each eligible projection on the dense model: "
                "x_l2 = sqrt(sum of squares over calibration tokens) (FP32 accumulation), x_absmean = mean |x|"}
        dman = save_tensors(out / OUTPUT_FILES["dense_stats"], stats_t, meta, out / OUTPUT_FILES["dense_stats_manifest"])
        sc_t = {scale_name(li, g, b): s for (li, g, b), s in res["scales"].items()}
        smeta = {"eval_id": C.EVAL_ID, "n_sequences": len(seqs), "n_tokens": n_tokens, "model": info,
                 "geometry": res["geometry"], "what": "FP32 distinct AWQ scales at the chosen alpha per "
                 "(layer, group, bits); group o holds one value per (KV head, head dim)"}
        sman = save_tensors(out / OUTPUT_FILES["awq_scales"], sc_t, smeta, out / OUTPUT_FILES["awq_scales_manifest"])
        write_json_new(out / OUTPUT_FILES["awq_search"], {
            "eval_id": C.EVAL_ID, "alpha_grid": list(awq.ALPHA_GRID), "bits": list(awq.SEARCH_BITS),
            "groups": awq.GROUPS, "convention": awq.AWQ_CONVENTION, "geometry": res["geometry"],
            "search_matmul_dtype": res["search_matmul_dtype"], "n_sequences": len(seqs), "n_tokens": n_tokens,
            "scales_file": sman["file"], "entries": [{**e, "scale_tensor": scale_name(e["layer"], e["group"], e["bits"])}
                                                     for e in res["entries"]]})
        timings["save_s"] = time.perf_counter() - t

        pres = res["preservation"]
        steps["preservation"] = {
            "what": "block output with folded scales and no quantization versus the original block output, "
                    "calibration batch 0, model dtype",
            "per_block": pres,
            "max_over_blocks": {k: max(p[k] for p in pres) for k in
                                ("max_abs_err_over_max_abs_out", "rel_fro_out", "rel_fro_update")}}
        steps["search"] = {"search_matmul_dtype": res["search_matmul_dtype"], "geometry": res["geometry"],
                           "chosen_alpha": {str(b): {g: [e["alpha"] for e in res["entries"]
                                                         if e["bits"] == b and e["group"] == g]
                                                     for g in awq.GROUPS} for b in awq.SEARCH_BITS},
                           "file": OUTPUT_FILES["awq_search"]}
        steps["files"] = {"dense_stats": dman["file"], "awq_scales": sman["file"]}
        steps["pristine_after"] = {"tensors_verified": pristine.verify(model)}
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            steps["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
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
