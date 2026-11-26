"""End-to-end CPU run of Q3 design v2 with a tiny random 4-block model: calib_stats, probe quant_v2 in two block
ranges, select q3v2 from both parts, compress q3_int3 with the v2 allocation file."""
import json
import os
import subprocess
import sys

import pytest

from common.io import read_jsonl, sha256_file
from common.paths import REPO_ROOT
from task1 import compress, probe, select, storage
from task1 import config as C
from task1 import q3v2 as Q

LIM_C, LIM_D = 2, 2
N_LAYERS = 4
N_UNITS = N_LAYERS * len(C.Q3V2_UNITS)
SHAPES = {"q_proj": (256, 256), "k_proj": (128, 256), "v_proj": (128, 256), "o_proj": (256, 256),
          "gate_proj": (512, 256), "up_proj": (512, 256), "down_proj": (256, 512)}
MATS = [{"layer": li, "name": n, "qualname": f"model.layers.{li}.{'self_attn' if n in C.ATTN_PROJ else 'mlp'}.{n}",
         "shape": list(SHAPES[n])} for li in range(N_LAYERS) for n in C.ELIGIBLE]
PARTS = {"pa": (0, 1), "pb": (2, 3)}

RUN_TOP = {"config", "meta", "start_time", "status", "elapsed_wall_s", "updated_time", "end_time", "model",
           "eval_id", "timings_s", "steps"}
COMMON_CONFIG = {"entry_point", "settings", "device", "model_path", "limits", "limited", "test_only",
                 "output_files", "data_files", "batch_size"}
PROBE_CONFIG = COMMON_CONFIG | {"kind", "measures", "definitions", "alloc_settings", "calib_stats_dir", "design",
                                "q3v2_settings", "blocks"}
SELECT_CONFIG = COMMON_CONFIG | {"kind", "target", "target_spec", "condition", "alloc_settings", "probe_dir",
                                 "calib_stats_dir", "design", "q3v2_settings"}
COMPRESS_CONFIG = COMMON_CONFIG | {"condition", "condition_spec", "artifacts_dir", "calib_stats_dir", "allocation_file"}
SENS = {"eval_id", "kind", "unit", "design", "n_blocks", "block_range", "n_units_total", "units", "measures",
        "definitions", "values", "fp16_calib_nll", "effort", "calib_stats", "model", "limits", "alloc_settings",
        "q3v2_settings"}
UNIT_KEYS = {"index", "id", "layer", "unit", "scale_group", "projections", "groups"}
EFFORT = {"wall_s", "full_model_calibration_passes", "block_forward_passes", "calibration_sequences_per_pass"}
QS1_ROW = {"eval_id", "kind", "design", "measure", "level", "layer", "unit", "unit_id", "unit_index", "value",
           "calib_mean_nll", "calib_nll_sum", "n_scored", "fp16_calib_mean_nll", "wall_s", "bits", "scale_group",
           "n_guarded_groups", "n_clipped_codes"}
QS2_ROW = {"eval_id", "kind", "design", "measure", "level", "layer", "unit", "unit_id", "unit_index", "value",
           "projections"}
CAL_ROW = {"eval_id", "split", "seq_index", "source_row", "token_offset", "nll_sum", "n_scored"}
CAL_SUMMARY = {"eval_id", "split", "n_sequences", "N", "nll_sum", "mean_nll", "batch_size"}
CAND = {"eval_id", "candidate_id", "index", "condition", "kind", "target", "reference", "measure", "lambda", "rule",
        "development", "calibration", "output_error_pooled_rel", "timings_s", "selected", "collapsed_to_uniform",
        "design", "per_unit", "bits_histogram", "chunking", "per_matrix_bits", "objective", "objective_exact",
        "dp_states_per_unit", "totals"}
TOTALS = {"eligible_packed_bytes", "budget_bytes", "cost_units", "budget_units", "ref_bits", "non_eligible_params",
          "whole_model_bytes", "budget_whole_model_bytes", "bits_per_eligible_weight"}
CHUNKING = {"chunk_groups", "unit_chunks", "eighth_bytes_per_group", "cost_unit", "full_model_expected"}
SELECTED = {"eval_id", "design", "condition", "kind", "target", "unit", "n_units", "n_blocks", "rule", "measure",
            "lambda", "candidate_id", "per_unit", "bits_histogram", "chunking", "totals", "development",
            "calibration", "selection_criterion", "n_candidates", "n_reference_candidates", "collapsed_to_uniform",
            "all_candidates_collapsed", "inputs", "probe_union", "model", "model_path", "limits", "code_commit",
            "code_dirty", "q3v2_settings", "ref_bits", "per_matrix_bits", "objective", "objective_exact"}
MANIFEST = {"eval_id", "condition", "family", "kind", "model", "model_dtype", "conventions", "matrices",
            "extra_tensors", "logical_bytes", "tensors", "stored_tensor_bytes", "file", "allocation", "design"}
ROW_Q3 = {"eval_id", "layer", "name", "qualname", "shape", "err_num", "err_den", "rel_output_error", "nonzeros",
          "bits", "groups", "n_guarded_groups", "n_clipped_codes", "n_one_sided_groups"}


def _env():
    return dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")


def _run(args):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True,
                          timeout=1800)


def _t(model):
    return ["--model-path", str(model), "--device", "cpu", "--limit-calib", str(LIM_C)]


@pytest.fixture(scope="module")
def runs(tiny_compress4_dir, tmp_path_factory):
    b = tmp_path_factory.mktemp("q3v2_e2e")
    t = _t(tiny_compress4_dir)
    steps = [["task1.calib_stats", "--out", str(b / "calib"), *t]]
    for name, (lo, hi) in PARTS.items():
        steps.append(["task1.probe", "--kind", "quant_v2", "--blocks", f"{lo}-{hi}", "--calib-stats",
                      str(b / "calib"), "--out", str(b / name), *t])
    steps += [
        ["task1.select", "--kind", "q3v2", "--target", "int3", "--probe", str(b / "pa"), str(b / "pb"),
         "--calib-stats", str(b / "calib"), "--out", str(b / "sel"), *t, "--limit-dev", str(LIM_D)],
        ["task1.compress", "--condition", "q3_int3", "--allocation", str(b / "sel" / "selected_allocation.json"),
         "--calib-stats", str(b / "calib"), "--out", str(b / "q3_int3"), "--artifacts", str(b / "art"), *t,
         "--limit-dev", str(LIM_D)],
    ]
    for a in steps:
        p = _run(a)
        assert p.returncode == 0, (a[:4], p.stderr[-4000:])
    return b


def _load(p):
    return json.loads(p.read_text())


def _check_guard(steps):
    g = steps["held_out_guard"]
    assert g["attempts"] == [] and set(g["forbidden"]) == {
        "wikitext2_test_windows_locked.jsonl.gz", "c4_validation_windows_locked.jsonl.gz",
        "hellaswag_val_locked.jsonl.gz"}


@pytest.mark.parametrize("name", sorted(PARTS))
def test_probe_part_outputs(runs, name):
    lo, hi = PARTS[name]
    d = runs / name
    assert sorted(x.name for x in d.iterdir()) == sorted(probe.OUTPUT_FILES.values())
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == PROBE_CONFIG
    assert r["config"]["kind"] == "quant_v2" and r["config"]["design"] == C.Q3V2_DESIGN
    assert r["config"]["blocks"] == f"{lo}-{hi}" and r["config"]["q3v2_settings"] == C.q3v2_settings()
    assert set(r["timings_s"]) == {"load_s", "fp16_calib_nll_s"}
    s = r["steps"]
    assert set(s) == {"restore", "block_range", "calib_stats_source", "awq_alphas", "fp16_calib_nll",
                      "pristine_after", "sensitivity", "effort", "held_out_guard"}
    assert s["block_range"] == [lo, hi]
    _check_guard(s)
    sens = _load(d / "sensitivity.json")
    assert set(sens) == SENS and s["sensitivity"]["sha256"] == sha256_file(d / "sensitivity.json")
    assert sens["design"] == C.Q3V2_DESIGN and sens["unit"] == C.Q3V2_UNIT and sens["kind"] == "quant_v2"
    assert sens["n_blocks"] == N_LAYERS and sens["n_units_total"] == N_UNITS and sens["block_range"] == [lo, hi]
    units = [u for u in Q.units_of(MATS) if lo <= u["layer"] <= hi]
    assert [u["id"] for u in sens["units"]] == [u["id"] for u in units]
    assert all(set(u) == UNIT_KEYS for u in sens["units"])
    n = len(units)
    v = sens["values"]
    assert set(v) == set(C.Q3V2_MEASURES)
    for m in C.Q3V2_MEASURES:
        assert set(v[m]) == {"2", "3", "4", "8"} and all(len(x) == n for x in v[m].values())
    for j in range(n):
        assert v["QS2v2"]["2"][j] > v["QS2v2"]["3"][j] > v["QS2v2"]["4"][j] > v["QS2v2"]["8"][j] > 0
    ef = sens["effort"]
    assert set(ef) == {"QS1v2", "QS2v2", "fp16_reference"} and all(EFFORT <= set(e) for e in ef.values())
    assert ef["QS1v2"]["full_model_calibration_passes"] == 4 * n == ef["QS1v2"]["restores"]
    assert ef["QS2v2"]["block_forward_passes"] == lo + 4 * (hi - lo + 1)
    rows = read_jsonl(d / "probe_rows.jsonl.gz")
    qs1 = [x for x in rows if x["measure"] == "QS1v2"]
    qs2 = [x for x in rows if x["measure"] == "QS2v2"]
    assert len(qs1) == len(qs2) == 4 * n and len(rows) == 8 * n
    assert all(set(x) == QS1_ROW for x in qs1) and all(set(x) == QS2_ROW for x in qs2)
    for x in qs1:
        assert x["value"] == pytest.approx(x["calib_mean_nll"] - sens["fp16_calib_nll"]["mean_nll"], abs=1e-12)
        assert x["bits"] == int(x["level"]) and x["scale_group"] == Q.UNIT_GROUP[x["unit"]]
        assert x["design"] == C.Q3V2_DESIGN
    for x in qs2:
        assert set(x["projections"]) == set(Q.UNIT_PROJS[x["unit"]])
        assert x["value"] == pytest.approx(sum(p["err_num"] / p["err_den"] for p in x["projections"].values()))
    for m, rs in (("QS1v2", qs1), ("QS2v2", qs2)):
        for b in ("2", "3", "4", "8"):
            got = [x["value"] for x in rs if x["level"] == b]
            assert got == v[m][b] and [x["unit_id"] for x in rs if x["level"] == b] == [u["id"] for u in units]
    nll = read_jsonl(d / "probe_calib_nll.jsonl.gz")
    assert len(nll) == LIM_C * (1 + 4 * n)
    assert all(set(x) == CAL_ROW | {"measure", "level", "layer", "unit", "unit_id"} for x in nll)
    for p in d.glob("*.gz"):
        assert p.read_bytes()[4:8] == b"\x00\x00\x00\x00"


def test_probe_parts_share_fp16_reference(runs):
    a, b = (_load(runs / n / "sensitivity.json")["fp16_calib_nll"]["mean_nll"] for n in ("pa", "pb"))
    assert abs(a - b) <= C.Q3V2_FP16_REF_TOL


def test_select_outputs(runs):
    d = runs / "sel"
    assert sorted(x.name for x in d.iterdir()) == sorted(select.OUTPUT_FILES.values())
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == SELECT_CONFIG
    assert r["config"]["probe_dir"] == [str(runs / "pa"), str(runs / "pb")]
    assert r["config"]["design"] == C.Q3V2_DESIGN
    s = r["steps"]
    assert set(s) == {"restore", "inputs", "probe_union", "n_candidates", "selection", "pristine_after",
                      "held_out_guard"}
    _check_guard(s)
    assert s["n_candidates"] == {"total": 3, "reference": 1}
    assert [p["sensitivity_sha256"] for p in s["inputs"]["probe"]] == [
        sha256_file(runs / n / "sensitivity.json") for n in ("pa", "pb")]
    pu = s["probe_union"]
    assert set(pu) == {"n_units", "n_parts", "fp16_calib_mean_nll", "fp16_spread", "part_of_unit", "block_ranges"}
    assert pu["n_units"] == N_UNITS and pu["n_parts"] == 2 and pu["part_of_unit"] == [0] * 8 + [1] * 8
    assert pu["block_ranges"] == [[0, 1], [2, 3]] and pu["fp16_spread"] <= C.Q3V2_FP16_REF_TOL
    rows = read_jsonl(d / "candidates.jsonl.gz")
    assert len(rows) == 3 and sum(x["selected"] for x in rows) == 1
    assert all(set(x) == CAND for x in rows)
    assert [x["candidate_id"] for x in rows] == ["q3_int3.reference", "q3_int3.QS1v2", "q3_int3.QS2v2"]
    budget = storage.packed_quant_bytes([tuple(m["shape"]) for m in MATS], 3)
    sens = {}
    for n in ("pa", "pb"):
        sv = _load(runs / n / "sensitivity.json")["values"]
        for m in C.Q3V2_MEASURES:
            sens.setdefault(m, {b: [] for b in ("2", "3", "4", "8")})
            for b in sens[m]:
                sens[m][b] += sv[m][b]
    for x in rows:
        assert x["design"] == C.Q3V2_DESIGN and set(x["totals"]) == TOTALS and set(x["chunking"]) == CHUNKING
        t = x["totals"]
        assert t["eligible_packed_bytes"] <= t["budget_bytes"] == budget
        assert t["cost_units"] * x["chunking"]["chunk_groups"] == 8 * t["eligible_packed_bytes"]
        assert t["cost_units"] <= t["budget_units"]
        assert x["chunking"]["chunk_groups"] == 512 and x["chunking"]["unit_chunks"] == [2, 1, 4, 2] * N_LAYERS
        assert x["per_unit"]["ids"] == [u["id"] for u in Q.units_of(MATS)]
        assert len(x["per_unit"]["bits"]) == N_UNITS and set(x["per_unit"]["bits"]) <= set(C.Q3V2_BITS)
        assert x["collapsed_to_uniform"] == (x["per_unit"]["bits"] == [3] * N_UNITS)
        pm = {m["qualname"]: x["per_matrix_bits"][m["qualname"]] for m in MATS}
        assert Q.check_bits(MATS, pm, 3)["unit_bits"] == x["per_unit"]["bits"]
        assert x["bits_histogram"] == Q.bits_histogram(Q.units_of(MATS), x["per_unit"]["bits"])
        assert set(x["development"]) == CAL_SUMMARY and x["development"]["n_sequences"] == LIM_D
        assert set(x["calibration"]) == CAL_SUMMARY and x["calibration"]["n_sequences"] == LIM_C
        if not x["reference"]:
            # The candidate is the exact knapsack optimum of the joined sensitivities.
            per_unit = [{b: sens[x["measure"]][str(b)][i] for b in C.Q3V2_BITS} for i in range(N_UNITS)]
            assert Q.allocation(per_unit, MATS, 3)["bits"] == x["per_unit"]["bits"]
            assert x["per_unit"]["sensitivity"] == [{str(b): per_unit[i][b] for b in C.Q3V2_BITS}
                                                    for i in range(N_UNITS)]
    ref = rows[0]
    assert ref["reference"] and ref["per_unit"]["bits"] == [3] * N_UNITS and ref["collapsed_to_uniform"]
    pool = [x for x in rows if not x["reference"]]
    fresh = [x for x in pool if not x["collapsed_to_uniform"]]
    win = [x for x in rows if x["selected"]][0]
    assert not win["reference"]
    assert win["development"]["mean_nll"] == min(x["development"]["mean_nll"] for x in (fresh or pool))
    nll = read_jsonl(d / "candidate_nll.jsonl.gz")
    assert len(nll) == 3 * (LIM_C + LIM_D) and all(set(x) == CAL_ROW | {"candidate_id"} for x in nll)
    sel = _load(d / "selected_allocation.json")
    assert set(sel) == SELECTED
    assert sel["design"] == C.Q3V2_DESIGN and sel["unit"] == C.Q3V2_UNIT and sel["n_units"] == N_UNITS
    assert sel["candidate_id"] == win["candidate_id"] == s["selection"]["winner"]
    assert sel["collapsed_to_uniform"] == win["collapsed_to_uniform"]
    assert sel["all_candidates_collapsed"] == all(x["collapsed_to_uniform"] for x in pool)
    assert sel["per_matrix_bits"] == win["per_matrix_bits"] and sel["per_unit"] == win["per_unit"]
    assert sel["selection_criterion"] == C.Q3V2_SELECTION_CRITERION and sel["ref_bits"] == 3
    assert sel["inputs"]["calib_stats"]["awq_scales_sha256"] == sha256_file(runs / "calib" / "awq_scales.safetensors")
    assert sel["code_commit"] == r["meta"]["git_commit"] and len(sel["code_commit"]) == 40
    assert s["selection"]["sha256"] == sha256_file(d / "selected_allocation.json")


def test_compress_with_v2_allocation(runs):
    d = runs / "q3_int3"
    sel_file = runs / "sel" / "selected_allocation.json"
    sel = _load(sel_file)
    assert sorted(x.name for x in d.iterdir()) == sorted(compress.OUTPUT_FILES.values())
    r = _load(d / "run.json")
    assert set(r) == RUN_TOP and r["status"] == "ok" and set(r["config"]) == COMPRESS_CONFIG
    s = r["steps"]
    assert set(s) == {"restore", "allocation", "awq_source", "build", "storage", "calib_nll", "artifact",
                      "peak_gpu_allocated_bytes", "held_out_guard"}
    _check_guard(s)
    al = s["allocation"]
    assert set(al) == {"file", "sha256", "design", "candidate_id", "measure", "lambda", "selection_code_commit",
                       "totals", "inputs", "git"}
    assert al["design"] == C.Q3V2_DESIGN and al["sha256"] == sha256_file(sel_file)
    assert al["totals"]["unit_bits"] == sel["per_unit"]["bits"]
    assert set(s["awq_source"]) == {str(b) for b in set(sel["per_unit"]["bits"])}
    rows = read_jsonl(d / "projections.jsonl.gz")
    assert len(rows) == len(MATS) and all(set(x) == ROW_Q3 for x in rows)
    assert {x["qualname"]: x["bits"] for x in rows} == sel["per_matrix_bits"]
    pq = s["storage"]["packed_quant"]
    assert pq["total"] <= pq["budget_bytes"] == sel["totals"]["budget_bytes"]
    assert pq["total"] == s["artifact"]["logical_bytes"]["eligible"] == sel["totals"]["eligible_packed_bytes"]
    man = _load(runs / "art" / "manifest.json")
    assert set(man) == MANIFEST and man["design"] == C.Q3V2_DESIGN
    assert man["allocation"] == {"file": str(sel_file), "sha256": al["sha256"], "candidate_id": sel["candidate_id"],
                                 "design": C.Q3V2_DESIGN}
    assert [m["bits"] for m in man["matrices"]] == [sel["per_matrix_bits"][m["qualname"]] for m in MATS]
    assert s["artifact"]["reload_matches_executed"] is True and len(s["artifact"]["extra_tensors"]) == 3 * N_LAYERS
    # The saved condition reproduces the development and calibration NLL recorded at selection.
    assert s["calib_nll"]["c4_development"]["mean_nll"] == sel["development"]["mean_nll"]
    assert s["calib_nll"]["c4_calibration"]["mean_nll"] == sel["calibration"]["mean_nll"]


def test_v2_refusals(runs, tiny_compress4_dir, tmp_path):
    t = [*_t(tiny_compress4_dir), "--limit-dev", str(LIM_D)]
    base = ["task1.select", "--kind", "q3v2", "--target", "int3", "--calib-stats", str(runs / "calib")]
    p = _run([*base, "--probe", str(runs / "pa"), "--out", str(tmp_path / "a"), *t])
    assert p.returncode != 0 and "do not cover the 16 units exactly once" in p.stderr
    p = _run([*base, "--probe", str(runs / "pa"), str(runs / "pa"), str(runs / "pb"), "--out", str(tmp_path / "b"),
              *t])
    assert p.returncode != 0 and "is covered by probe parts" in p.stderr
    with pytest.raises(SystemExit, match="takes one probe directory"):
        select.main(["--kind", "q3", "--target", "int3", "--probe", "x", "y", "--calib-stats", "z",
                     "--out", str(tmp_path / "c")])
    with pytest.raises(SystemExit, match="only used by --kind quant_v2"):
        probe.main(["--kind", "quant", "--blocks", "0-1", "--calib-stats", "x", "--out", str(tmp_path / "d")])
    with pytest.raises(SystemExit, match="A-B"):
        probe.main(["--kind", "quant_v2", "--blocks", "1", "--calib-stats", "x", "--out", str(tmp_path / "e")])
    with pytest.raises(SystemExit, match="0 <= A <= B"):
        probe.main(["--kind", "quant_v2", "--blocks", "3-1", "--calib-stats", "x", "--out", str(tmp_path / "f")])
    p = _run(["task1.probe", "--kind", "quant_v2", "--blocks", "2-4", "--calib-stats", str(runs / "calib"),
              "--out", str(tmp_path / "g"), *_t(tiny_compress4_dir)])
    assert p.returncode != 0 and "exceeds the 4 blocks" in p.stderr
    # A v2 file is refused for a P3 condition and after a change to one unit's bits.
    sel = runs / "sel" / "selected_allocation.json"
    p = _run(["task1.compress", "--condition", "q3_int4", "--allocation", str(sel), "--calib-stats",
              str(runs / "calib"), "--out", str(tmp_path / "h"), "--artifacts", str(tmp_path / "hh"), *t])
    assert p.returncode != 0 and "is for condition q3_int3" in p.stderr
    bad = _load(sel)
    q = "model.layers.0.self_attn.k_proj"
    bad["per_matrix_bits"][q] = 8 if bad["per_matrix_bits"][q] != 8 else 2
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    p = _run(["task1.compress", "--condition", "q3_int3", "--allocation", str(tmp_path / "bad.json"),
              "--calib-stats", str(runs / "calib"), "--out", str(tmp_path / "i"), "--artifacts",
              str(tmp_path / "ii"), *t])
    assert p.returncode != 0 and "failed validation" in p.stderr
