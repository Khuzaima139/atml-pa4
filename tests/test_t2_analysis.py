"""task2.analysis on the committed Task 2 results: every output file and CSV column, recomputation assertions
(including tampered inputs that must be refused), bootstrap points, figure sizes and font sizes, --figures-only."""
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from common.paths import REPO_ROOT
from task2 import analysis as A
from task2 import analysis_figures as F

RESULTS = REPO_ROOT / "results" / "task2"
TABLES = {"A_errors.csv", "A_bench.csv", "A_backend.csv", "A_backend_compact.csv", "A_unsupported.csv", "A_storage.csv",
          "B_dev_curves.csv", "B_test_cells.csv", "B1_accuracy.csv", "B_bins.csv", "B2_mask_controls.csv",
          "B2_far_residue.csv", "B3_theoretical.csv", "B3_measured_training.csv", "B3_sweep.csv", "B3_items.csv",
          "C1_matched_budget.csv", "C_rows.csv", "C_cells.csv", "C2_bootstrap.csv", "C3_systems.csv", "C4_bench.csv",
          "C5_am4_controls.csv", "C5_am4_channels.csv", "P_predictions.csv"}
OUTPUTS = TABLES | set(F.FIGURE_FILES) | {"C_qualitative.jsonl", "C_qualitative.txt", "summary.json", "run.json"}
COLUMNS = {
    "C1_matched_budget.csv": {"method", "exact_match", "exact_match_conditional_fp16_correct", "n_fp16_correct",
                              "mean_reference_answer_nll", "mean_post_prefill_kl", "mean_teacher_forced_kl",
                              "mean_k_recon_rel_error", "mean_v_recon_rel_error", "mean_evidence_retained",
                              "realized_fraction_min", "realized_fraction_median", "realized_fraction_max",
                              "mean_context_bytes_payload", "mean_context_bytes_metadata",
                              "mean_context_bytes_fp16_retained", "mean_context_bytes_fp16_tail",
                              "mean_context_bytes_indices", "mean_context_bytes_total", "mean_shared_decode_time_bytes"},
    "C2_bootstrap.csv": {"metric", "a", "b", "point", "ci_low", "ci_high", "n_prompts", "n_resamples", "seed"},
    "P_predictions.csv": {"prediction", "protocol_text", "metric", "threshold", "measured", "comparison_holds", "notes"},
    "B1_accuracy.csv": {"configuration", "mask", "test_regime", "test_K", "mean", "min", "max", "chance_label_space",
                        "chance_1_over_K", "access_note", "own_cell"},
    "B3_theoretical.csv": {"mechanism", "attended_pairs_per_head_per_layer", "accessible_positions_per_query_at_last_position",
                           "state_bytes_per_layer_fp32"},
    "C4_bench.csv": {"config", "L", "batch", "pack_ms_median", "unpack_dequant_ms_median", "decode_step_ms_median",
                     "decode_step_ratio_to_fp16", "serialized_bytes_per_sequence"},
}


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("t2an") / "analysis"
    A.main(["--allow-uncommitted", "--out", str(d)])
    return d


def test_outputs_and_columns(out):
    assert {p.name for p in out.iterdir()} == OUTPUTS
    for name, cols in COLUMNS.items():
        assert cols <= set(rows(out / name)[0]), name
    s = json.loads((out / "summary.json").read_text())
    assert s["analysis_id"] == A.ANALYSIS_ID and s["round_rule"] == A.ROUND_RULE
    assert s["tables"]["C_rows.csv"] == 1080 and s["tables"]["C1_matched_budget.csv"] == 6
    assert s["checks"]["part_b"]["learned"] == 42 and s["checks"]["part_c"]["eval_rows"] == 1080
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "ok"


def test_tables_match_per_example_files(out):
    c1 = {r["method"]: r for r in rows(out / "C1_matched_budget.csv")}
    cr = rows(out / "C_rows.csv")
    for m in A.METHODS:
        x = [r for r in cr if r["method"] == m]
        assert len(x) == 180
        assert float(c1[m]["exact_match"]) == pytest.approx(np.mean([r["correct"] == "True" for r in x]), abs=1e-12)
    ids = {r["id"] for r in cr}
    assert len(ids) == 180
    for r in rows(out / "C2_bootstrap.csv"):
        key = "correct" if r["metric"] == "exact_match" else "answer_nll_mean"
        conv = (lambda v: float(v == "True")) if key == "correct" else float
        a = np.mean([conv(x[key]) for x in cr if x["method"] == r["a"]])
        b = np.mean([conv(x[key]) for x in cr if x["method"] == r["b"]])
        assert float(r["point"]) == pytest.approx(b - a, abs=1e-12)
        assert float(r["ci_low"]) <= float(r["point"]) <= float(r["ci_high"])
        assert int(r["n_resamples"]) == 10_000 and int(r["seed"]) == 6304
    preds = [r["prediction"] for r in rows(out / "P_predictions.csv")]
    for p in ("A-H1a", "A-H2", "A-H3", "A-H4a", "B-H1a", "B-H6a", "B3-H a", "C-H1", "C-H5 int4_channel_k",
              "C-H7 hybrid", "AM4 (c)", "AM4 rival"):
        assert p in preds
    assert len(rows(out / "C_cells.csv")) == 6 * (18 + 6 + 3)


def test_figure_sizes_and_fonts(out):
    for name in F.FIGURE_FILES:
        w, h = F.pdf_size_in(out / name)
        limit = F.HALF if name == "A2_gradient_error.pdf" else F.WIDTH
        assert w <= limit + 0.01 and h < 6, (name, w, h)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "small", fontsize=6)
    assert F.small_fonts(fig)
    with pytest.raises(F.FigureError):
        F._save(fig, out.parent / "small.pdf")


def test_figures_only(out, tmp_path):
    d = tmp_path / "figures"
    A.main(["--figures-only", "--allow-uncommitted", "--analysis", str(out), "--out", str(d)])
    assert {p.name for p in d.iterdir()} == set(F.FIGURE_FILES) | {"run.json"}


def mirror(tmp_path, tamper_rel, edit):
    root = tmp_path / "results"
    for p in RESULTS.rglob("*"):
        if p.is_file() and not p.name.endswith(".safetensors"):
            q = root / p.relative_to(RESULTS)
            q.parent.mkdir(parents=True, exist_ok=True)
            if str(p.relative_to(RESULTS)) == tamper_rel:
                shutil.copy(p, q)
                obj = json.loads(q.read_text())
                edit(obj)
                q.write_text(json.dumps(obj))
            else:
                q.symlink_to(p)
    return root


@pytest.mark.parametrize("rel,edit,msg", [
    ("part_c/eval/run.json", lambda o: o["summary"]["per_method"]["hybrid"].update(accuracy=1.0), "eval summary"),
    ("mqar/window_near_K16_s6305/run.json",
     lambda o: o["test_summary"]["native"]["far_K16"].update(accuracy=0.5), "test accuracy"),
    ("part_c/diag/run.json", lambda o: o["summary"]["channel"]["overall"].update(k_ratio=7.6), "channel overall"),
])
def test_tampered_stored_values_refused(tmp_path, rel, edit, msg):
    root = mirror(tmp_path, rel, edit)
    with pytest.raises(A.AnalysisError, match=msg):
        A.compute(A.Inputs(root))


def test_refuses_existing_output(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit):
        A.main(["--allow-uncommitted", "--out", str(tmp_path / "x")])


def test_overlap_rule_and_round_rule():
    assert A.overlap([0.1, 0.2], [0.15, 0.3]) and not A.overlap([0.1, 0.2], [0.21, 0.3])
    assert A.fmt(0.123456) == "0.1235" and A.fmt(None) == "n/a"
