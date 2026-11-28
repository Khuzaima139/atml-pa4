"""Task 2 analysis figures (PDF, 5.5 in text width or 2.7 in for a single side-by-side panel, fonts of at least
8 pt checked before saving, thinned ticks, tight bounding boxes). Every figure is drawn from the analysis tables."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator  # noqa: E402

from task1.analysis_figures import FONT, HALF, RC, WIDTH, FigureError, _save, _thin, pdf_size_in, small_fonts  # noqa: E402,F401

IMPL_STYLE = {"dense": dict(color="#9e3b2b", label="Dense"),
              "tiled": dict(color="#1f5fa8", label="Tiled (Python)"),
              "sdpa_default": dict(color="#2a8a4a", label="SDPA default")}
DTYPE_FILL = {"fp16": "full", "fp32": "none"}
DTYPE_LABEL = {"fp16": "FP16", "fp32": "FP32"}
PASS_LS = {"forward": "-", "forward_backward": "--"}
MECH = ("dense", "window", "local_dilated", "linear")
MECH_LABEL = {"dense": "Dense", "window": "Window 64", "local_dilated": "Local + dilated", "linear": "Linear"}
MECH_COLOR = {"dense": "#9e3b2b", "window": "#1f5fa8", "local_dilated": "#2a8a4a", "linear": "#7b4aa8"}
K_COLOR = {4: "#1b9e77", 8: "#d95f02", 16: "#7570b3"}
REGIME_LS = {"mixed": "-", "near": "--", "curriculum": "-."}
METHODS = ("fp16", "sink_recent", "question_aware", "int4_token", "int4_channel_k", "hybrid")
METHOD_STYLE = {"fp16": dict(color="black", marker="*", label="FP16 full cache"),
                "sink_recent": dict(color="#9e3b2b", marker="o", label="Sink + recent (FP16)"),
                "question_aware": dict(color="#1f5fa8", marker="s", label="Question-aware (FP16)"),
                "int4_token": dict(color="#d95f02", marker="v", label="INT4 token-wise K and V"),
                "int4_channel_k": dict(color="#2a8a4a", marker="^", label="INT4 channel-wise K"),
                "hybrid": dict(color="#7b4aa8", marker="D", label="FP16 + INT2 hybrid")}
BENCH_CONFIGS = ("int4_token", "int4_channel_k", "hybrid")
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
FIGURE_FILES = ("A1_error_latency_memory.pdf", "A2_gradient_error.pdf", "B1_dev_curves.pdf",
                "B2_distance_load_heatmap.pdf", "B3_sweep.pdf", "C1_accuracy_nll_breakdown.pdf", "C2_bench.pdf")


def _len_axis(ax, ticks=(256, 1024, 4096)):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v) // 1024}k" if v >= 1024 else f"{int(v)}"))
    ax.xaxis.set_minor_locator(NullLocator())


def _log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())


def _impl_of(label):
    return "tiled" if label.startswith("tiled") else label


def _a_lines(ax, series, ylog=True):
    """series: {(impl, dtype, pass): (xs, ys)}."""
    for (impl, dt, p), (xs, ys) in sorted(series.items()):
        st = IMPL_STYLE[impl]
        ax.plot(xs, ys, color=st["color"], linestyle=PASS_LS[p], marker="o", markersize=3.5,
                fillstyle=DTYPE_FILL[dt], markerfacecolor=st["color"] if dt == "fp16" else "white")
    _len_axis(ax)
    if ylog:
        _log_y(ax)


def _a_legend(fig, passes=True):
    h = [Line2D([], [], color=s["color"], label=s["label"]) for s in IMPL_STYLE.values()]
    h += [Line2D([], [], color="gray", marker="o", linestyle="none", markerfacecolor="gray", label="FP16"),
          Line2D([], [], color="gray", marker="o", linestyle="none", markerfacecolor="white", label="FP32")]
    if passes:
        h += [Line2D([], [], color="gray", linestyle="-", label="forward"),
              Line2D([], [], color="gray", linestyle="--", label="fwd + bwd")]
    fig.legend(handles=h, frameon=False, loc="outside lower center", ncol=4)


def _err_series(T, key):
    rows = [r for r in T["A_errors.csv"] if r["status"] == "ok" and _impl_of(r["implementation"]) in IMPL_STYLE]
    series = {}
    for impl in IMPL_STYLE:
        for dt in ("fp16", "fp32"):
            xs, ys = [], []
            for L in LENGTHS:
                v = [r[key] for r in rows if _impl_of(r["implementation"]) == impl and r["dtype"] == dt and r["L"] == L]
                if v:
                    xs.append(L)
                    ys.append(max(v))
            series[(impl, dt, "forward")] = (xs, ys)
    return series


def fig_part_a(T, out):
    bench = [r for r in T["A_bench.csv"] if r["status"] == "ok"
             and r["implementation"] in ("dense", "tiled_128x128", "sdpa_default")]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.75), layout="constrained")
        ax = axes[0]
        _a_lines(ax, _err_series(T, "out_max_abs_err"))
        ax.set_ylabel("Max abs output error")
        ax.set_title("(a) Output error")
        for j, (key, ylab, scale, title) in enumerate((("median_ms", "Median time, 10 runs (ms)", 1.0, "(b) Latency"),
                                                       ("peak_allocated_bytes", "Peak allocated (MiB)", 2.0 ** -20,
                                                        "(c) Peak memory"))):
            series = {}
            for r in bench:
                k = (_impl_of(r["implementation"]), r["dtype"], r["pass"])
                series.setdefault(k, ([], []))
                series[k][0].append(r["L"])
                series[k][1].append(r[key] * scale)
            ax = axes[j + 1]
            _a_lines(ax, series)
            ax.set_ylabel(ylab)
            ax.set_title(title)
        for ax in axes:
            ax.set_xlabel("Sequence length L")
        _a_legend(fig)
        return _save(fig, out / "A1_error_latency_memory.pdf")


def fig_grad(T, out):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(HALF, 2.6), layout="constrained")
        _a_lines(ax, _err_series(T, "grad_max_abs_err"))
        ax.set_ylabel("Max abs gradient error\n(max over dq, dk, dv)")
        ax.set_xlabel("Sequence length L")
        h = [Line2D([], [], color=s["color"], label=s["label"].split(" (")[0]) for s in IMPL_STYLE.values()]
        h += [Line2D([], [], color="gray", marker="o", linestyle="none", markerfacecolor=f, label=lab)
              for f, lab in (("gray", "FP16"), ("white", "FP32"))]
        fig.legend(handles=h, frameon=False, loc="outside lower center", ncol=2)
        return _save(fig, out / "A2_gradient_error.pdf")


def _parse(c):
    mech, regime, k = c.rsplit("_", 2)
    return mech, regime, int(k[1:])


def fig_dev_curves(T, out):
    rows = T["B_dev_curves.csv"]
    configs = list(dict.fromkeys(r["configuration"] for r in rows))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.4), sharex=True, sharey=True, layout="constrained")
        for ax, mech in zip(axes.flat, MECH):
            ks = set()
            for c in configs:
                m, regime, K = _parse(c)
                if m != mech:
                    continue
                ks.add(K)
                xs = sorted({r["update"] for r in rows if r["configuration"] == c})
                acc = np.array([[r["accuracy"] for r in rows if r["configuration"] == c and r["update"] == u] for u in xs])
                ax.fill_between(xs, acc.min(1), acc.max(1), color=K_COLOR[K], alpha=0.2, linewidth=0)
                ax.plot(xs, acc.mean(1), color=K_COLOR[K], linestyle=REGIME_LS[regime], marker="o", markersize=2.5)
            ax.axhline(1 / 64, color="gray", linestyle=":", linewidth=1)
            for K in sorted(ks):
                ax.axhline(1 / K, color=K_COLOR[K], linestyle=":", linewidth=0.8)
            ax.set_title(MECH_LABEL[mech])
            ax.set_ylim(0, 1.03)
            _thin(ax, 4, 5)
        for ax in axes[1]:
            ax.set_xlabel("Training update")
        for ax in axes[:, 0]:
            ax.set_ylabel("Development accuracy")
        h = [Line2D([], [], color=K_COLOR[K], label=f"trained K={K}") for K in (4, 8, 16)]
        h += [Line2D([], [], color="gray", linestyle=ls, label=f"{r} regime" if r != "curriculum" else "near-to-mixed curriculum")
              for r, ls in REGIME_LS.items()]
        h += [Line2D([], [], color="gray", linestyle=":", label="chance 1/64; coloured dotted: 1/K")]
        fig.legend(handles=h, frameon=False, loc="outside lower center", ncol=3)
        fig.suptitle("Own training cell, 256 sequences per checkpoint; line = mean of 3 seeds, band = seed min-max",
                     fontsize=FONT)
        return _save(fig, out / "B1_dev_curves.pdf")


def _row_label(c, mask):
    m, regime, K = _parse(c)
    reg = "curriculum" if regime == "curriculum" else regime
    lab = f"{MECH_LABEL[m]}, {reg} K{K}"
    return lab + " + W64 test mask" if mask == "window64" else lab


def fig_heatmap(T, out):
    rows = T["B_bins.csv"]
    keys = list(dict.fromkeys((r["configuration"], r["mask"]) for r in rows))
    keys = [k for k in keys if k[1] == "native"] + [k for k in keys if k[1] == "window64"]
    ks, bins = (4, 8, 16), ("[0,32)", "[32,64)", "[64,128)", "[128,256)")
    M = np.array([[next(r["mean"] for r in rows if (r["configuration"], r["mask"]) == k and r["test_K"] == K
                        and r["bin"] == b) for K in ks for b in bins] for k in keys])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(WIDTH, 0.21 * len(keys) + 1.3), layout="constrained")
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "1" if v >= 0.995 else f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=FONT,
                        color="black" if v > 0.6 else "white")
            K = _parse(keys[i][0])[2]
            j0 = ks.index(K) * len(bins)
            ax.add_patch(Rectangle((j0 - 0.5, i - 0.5), len(bins), 1, fill=False, edgecolor="red", linewidth=1.2))
        ax.set_yticks(range(len(keys)), [_row_label(*k) for k in keys])
        ax.set_xticks(range(M.shape[1]), [b for _ in ks for b in bins], rotation=90)
        for g in range(1, len(ks)):
            ax.axvline(g * len(bins) - 0.5, color="white", linewidth=2)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks([g * len(bins) + 1.5 for g in range(len(ks))], [f"test K={K}" for K in ks])
        sec.tick_params(length=0)
        ax.set_xlabel("Distance bin (tokens); red box: training K;\n1024 test sequences per regime and K")
        cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01, fraction=0.04)
        cb.set_label("Accuracy (3-seed mean)")
        cb.ax.yaxis.set_major_locator(MaxNLocator(5))
        return _save(fig, out / "B2_distance_load_heatmap.pdf")


def fig_sweep(T, out):
    rows = T["B3_sweep.csv"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), layout="constrained")
        for ax, key, scale, ylab in ((axes[0], "median_ms", 1.0, "Median forward time, 10 runs (ms)"),
                                     (axes[1], "peak_allocated_bytes", 2.0 ** -20, "Peak allocated (MiB)")):
            for m in MECH:
                rr = sorted([r for r in rows if r["mechanism"] == m], key=lambda r: r["T"])
                ax.plot([r["T"] for r in rr], [r[key] * scale for r in rr], marker="o", markersize=3.5,
                        color=MECH_COLOR[m], label=MECH_LABEL[m])
            _len_axis(ax, (256, 1024, 4096, 8192))
            _log_y(ax)
            ax.set_xlabel("Length T (B 2, FP32, forward)")
            ax.set_ylabel(ylab)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=4)
        return _save(fig, out / "B3_sweep.pdf")


def fig_c1(T, out):
    rows = T["C_cells.csv"]
    groups_l = [(lv, pos) for lv in (2048, 4096) for pos in ("start", "middle", "end")]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.2), sharey="row", layout="constrained",
                                 gridspec_kw={"width_ratios": [2, 1]})
        off = np.linspace(-0.25, 0.25, len(METHODS))
        for i, (key, ylab) in enumerate((("exact_match", "Exact match"), ("mean_reference_answer_nll", "Mean answer NLL (nats)"))):
            for j, (grouping, cats) in enumerate((("length x location", groups_l), ("distractors", (0, 2, 8)))):
                ax = axes[i, j]
                for o, m in zip(off, METHODS):
                    st = METHOD_STYLE[m]
                    if grouping == "distractors":
                        ys = [next(r[key] for r in rows if r["method"] == m and r["grouping"] == grouping
                                   and r["distractors"] == d) for d in cats]
                    else:
                        ys = [next(r[key] for r in rows if r["method"] == m and r["grouping"] == grouping
                                   and r["target_tokens"] == lv and r["position"] == pos) for lv, pos in cats]
                    ax.plot(np.arange(len(cats)) + o, ys, linestyle="none", marker=st["marker"], color=st["color"],
                            markersize=4, label=st["label"])
                if grouping == "distractors":
                    ax.set_xticks(range(3), [str(d) for d in cats])
                    if i == 1:
                        ax.set_xlabel("Distractors (60 each)")
                else:
                    ax.set_xticks(range(len(cats)), [f"{lv // 1024}k {pos}" for lv, pos in cats], rotation=30)
                    if i == 1:
                        ax.set_xlabel("Length and evidence location (30 each)")
                if key == "exact_match":
                    ax.set_ylim(-0.05, 1.05)
                    ax.yaxis.set_major_locator(MaxNLocator(5))
                else:
                    _log_y(ax)
                if j == 0:
                    ax.set_ylabel(ylab)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=3)
        fig.suptitle("Descriptive breakdown; each of the 18 grid cells holds 10 prompts", fontsize=FONT)
        return _save(fig, out / "C1_accuracy_nll_breakdown.pdf")


def fig_c2(T, out):
    rows = T["C4_bench.csv"]
    groups = [(L, b) for L in (2048, 4096) for b in (1, 8)]
    w = 0.26
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), layout="constrained")
        for ax, key, ylab in ((axes[0], "pack_ms_median", "Pack time (ms)"),
                              (axes[1], "unpack_dequant_ms_median", "Unpack + dequantize (ms)"),
                              (axes[2], "decode_step_ratio_to_fp16", "Decode step / FP16 decode step")):
            for k, c in enumerate(BENCH_CONFIGS):
                ys = [next(r[key] for r in rows if r["config"] == c and r["L"] == L and r["batch"] == b) for L, b in groups]
                ax.bar(np.arange(len(groups)) + (k - 1) * w, ys, w, color=METHOD_STYLE[c]["color"],
                       label=METHOD_STYLE[c]["label"])
            ax.set_xticks(range(len(groups)), [f"{L // 1024}k, b{b}" for L, b in groups], rotation=30)
            ax.set_ylabel(ylab)
            ax.set_xlabel("Context length, batch")
            if key == "decode_step_ratio_to_fp16":
                ax.axhline(1.0, color="black", linewidth=0.8)
                ax.set_ylim(0.9, max(r[key] for r in rows) * 1.05)
            ax.yaxis.set_major_locator(MaxNLocator(5))
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=3)
        fig.suptitle("Medians of 10 CUDA-synchronized runs after 3 warmups", fontsize=FONT)
        return _save(fig, out / "C2_bench.pdf")


def make_all(T, out, info=None):
    names = [fig_part_a(T, out), fig_grad(T, out), fig_dev_curves(T, out), fig_heatmap(T, out), fig_sweep(T, out),
             fig_c1(T, out), fig_c2(T, out)]
    if tuple(names) != FIGURE_FILES:
        raise FigureError(f"figure names {names}")
    if info is not None:
        info["size_inches"] = {n: [round(v, 3) for v in pdf_size_in(out / n)] for n in names}
    return names
