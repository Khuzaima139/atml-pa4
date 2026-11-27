"""Task 1 analysis figures (PDF, 5.5 in text width, fonts of at least 8 pt)."""
import matplotlib

matplotlib.use("Agg")
import re  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

from task1 import config as C  # noqa: E402

WIDTH = 5.5
HALF = 2.7
FONT = 8
PAD = 0.02  # inches around the tight bounding box
RC = {"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
      "ytick.labelsize": FONT, "legend.fontsize": FONT, "figure.titlesize": FONT, "pdf.fonttype": 42,
      "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "lines.markersize": 4}
METADATA = {"CreationDate": None, "Creator": None, "Producer": None}
FAM_STYLE = {
    "P1": dict(color="#9e3b2b", marker="o", label="Magnitude"),
    "P2": dict(color="#1f5fa8", marker="s", label="Wanda"),
    "P3": dict(color="#2a8a4a", marker="^", label="Wanda + OWL-style allocation"),
    "P2_2of4": dict(color="#7b4aa8", marker="D", label="Wanda 2:4"),
    "Q1": dict(color="#9e3b2b", marker="o", label="RTN"),
    "Q2": dict(color="#1f5fa8", marker="s", label="AWQ-style"),
    "Q3": dict(color="#2a8a4a", marker="^", label="Mixed precision"),
    "baseline": dict(color="black", marker="*", label="FP16"),
}
SUITE_LABEL = {"wikitext2_test": "WikiText-2 test", "c4_validation": "C4 validation"}
UNIT_LABEL = {"attn_in": "q, k, v", "attn_out": "o", "mlp_in": "gate, up", "mlp_out": "down"}


class FigureError(AssertionError):
    pass


def small_fonts(fig, minimum=FONT):
    """Visible text artists with a font size below the minimum (points)."""
    return [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < minimum - 1e-9]


def pdf_size_in(path):
    """(width, height) in inches of the first page of a PDF written by matplotlib."""
    m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", path.read_bytes())
    if m is None:
        raise FigureError(f"no MediaBox in {path}")
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


def _save(fig, path):
    small = small_fonts(fig)
    if small:
        plt.close(fig)
        raise FigureError(f"{path.name}: text below {FONT} pt: {small[:5]}")
    with open(path, "xb") as fh:
        fig.savefig(fh, format="pdf", bbox_inches="tight", pad_inches=PAD, metadata=METADATA)
    plt.close(fig)
    return path.name


def _plain_log(ax, axis="y"):
    """Plain-number tick labels on a log axis."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def _thin(ax, x=6, y=5):
    ax.xaxis.set_major_locator(MaxNLocator(x))
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(y))


def fig_quality(qrows, rrows, out):
    Q = {r["condition"]: r for r in qrows}
    R = {r["condition"]: r for r in rrows}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), constrained_layout=True)
        for j, s in enumerate(("wikitext2_test", "c4_validation")):
            ax = axes[0, j]
            for fam in ("P1", "P2", "P3", "P2_2of4"):
                cs = [c for c in Q if Q[c]["family"] == fam]
                xs = [100 * R[c]["achieved_sparsity"] for c in cs]
                ys = [Q[c][f"{s}_ppl"] for c in cs]
                if fam in ("P1", "P2"):
                    xs, ys = [100 * R["fp16"]["achieved_sparsity"]] + xs, [Q["fp16"][f"{s}_ppl"]] + ys
                st = FAM_STYLE[fam]
                ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=st["label"],
                        linestyle="-" if fam in ("P1", "P2") else "none")
            st = FAM_STYLE["baseline"]
            ax.plot([100 * R["fp16"]["achieved_sparsity"]], [Q["fp16"][f"{s}_ppl"]], linestyle="none",
                    marker=st["marker"], color=st["color"], markersize=7, label="FP16")
            ax.set_yscale("log")
            ax.set_xlabel("Zeroed eligible weights (%)")
            ax.set_ylabel(f"{SUITE_LABEL[s]} perplexity")
            ax.set_title(f"Pruning, {SUITE_LABEL[s]}")
            _thin(ax)
            ax = axes[1, j]
            for fam in ("Q1", "Q2", "Q3"):
                cs = [c for c in Q if Q[c]["family"] == fam]
                cs = sorted(cs, key=lambda c: R[c]["logical_eligible_bytes"])
                xs = [R[c]["logical_eligible_bytes"] / 1e9 for c in cs]
                ys = [Q[c][f"{s}_ppl"] for c in cs]
                stf = FAM_STYLE[fam]
                ax.plot(xs, ys, marker=stf["marker"], color=stf["color"], label=stf["label"],
                        linestyle="-" if fam != "Q3" else "none", fillstyle="none" if fam == "Q3" else "full")
            ax.plot([R["fp16"]["logical_eligible_bytes"] / 1e9], [Q["fp16"][f"{s}_ppl"]], linestyle="none",
                    marker=st["marker"], color=st["color"], markersize=7, label="FP16")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v) % 5 == 0 else ""))
            ax.set_xlabel("Packed eligible weight storage (GB)")
            ax.set_ylabel(f"{SUITE_LABEL[s]} perplexity")
            ax.set_title(f"Quantization, {SUITE_LABEL[s]}")
            _thin(ax)
        h, lab = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(h, lab, frameon=False, loc="upper left")
        h, lab = axes[1, 0].get_legend_handles_labels()
        axes[1, 0].legend(h, lab, frameon=False, loc="upper right")
        return _save(fig, out / "F1_quality_vs_resource.pdf")


def fig_p3(D, out):
    pp = D["probe_prune"]["values"]
    nb = len(pp["S1"])
    x = np.arange(nb)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.2), sharex=True, constrained_layout=True)
        axes[0].bar(x, pp["S1"], color="#2a8a4a")
        axes[0].set_ylabel("Outlier fraction")
        axes[0].set_title("Outlier sensitivity per block (Wanda scores above 5 times the block mean)")
        for k, col in zip(sorted(pp["S2"]), ("#1f5fa8", "#9e3b2b")):
            axes[1].plot(x, pp["S2"][k], marker="o", color=col, label=f"Block alone pruned at {float(k):.0%}")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Calibration NLL increase")
        axes[1].set_title("Isolated loss increase per block")
        axes[1].legend(frameon=False, loc="upper center", ncol=2)
        axes[1].set_ylim(top=axes[1].get_ylim()[1] * 4)
        for c, col in (("p3_s50", "#1f5fa8"), ("p3_s75", "#9e3b2b")):
            sel = D["select"][c]["selected"]
            target = C.CONDITIONS[c]["sparsity"]
            axes[2].plot(x, np.asarray(sel["per_block"]["block_sparsity_achieved"]) * 100, marker="o", color=col,
                         label=f"{target:.0%} target (measure {sel['measure']}, \u03bb = {sel['lambda']})")
            axes[2].axhline(target * 100, color=col, linestyle=":", linewidth=0.8)
        axes[2].set_ylabel("Block sparsity (%)")
        axes[2].set_xlabel("Transformer block")
        axes[2].set_title("Final block sparsity (dotted: uniform Wanda)")
        axes[2].set_ylim(30, 100)
        h, lab = axes[2].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=2)
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(14, integer=True))
        return _save(fig, out / "F2_p3_allocation.pdf")


def _bit_map(D, c, shape, v2_units):
    sel = D["select"][c]["selected"]
    m = np.full(shape, np.nan)
    for uid, b in zip(sel["per_unit"]["ids"], sel["per_unit"]["bits"]):
        layer, unit = uid.split(".")
        m[list(v2_units).index(unit), int(layer)] = b
    return m


BIT_CMAP = ListedColormap(["#c44e52", "#dd8452", "#8fbcd4", "#2a4d8f"])
BIT_NORM = BoundaryNorm([1.5, 2.5, 3.5, 6, 9], BIT_CMAP.N)


def fig_q3_main(D, out, unit_values, v2_units=("attn_in", "attn_out", "mlp_in", "mlp_out"), bits="3"):
    """Compact Q3 figure: isolated sensitivity at one bit width and the two final bit maps."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.5), sharex=True, constrained_layout=True)
        vmax = float(np.nanmax(np.abs(unit_values[bits])))
        norm = SymLogNorm(linthresh=1e-3, vmin=-vmax, vmax=vmax, base=10)
        im = axes[0].imshow(unit_values[bits], aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        axes[0].set_title(f"Isolated calibration NLL increase, one unit at {bits} bits")
        fig.colorbar(im, ax=axes[0], label="NLL increase", shrink=0.95)
        for ax, c in zip(axes[1:], ("q3_int4", "q3_int3")):
            im2 = ax.imshow(_bit_map(D, c, unit_values[bits].shape, v2_units), aspect="auto", cmap=BIT_CMAP,
                            norm=BIT_NORM, interpolation="nearest")
            ax.set_title(f"Final bit widths, {C.CONDITIONS[c]['budget'].replace('q2_b', 'INT')} budget")
        cb = fig.colorbar(im2, ax=axes[1:], ticks=[2, 3, 4, 8], shrink=0.95)
        cb.set_label("Bits")
        for ax in axes:
            ax.set_yticks(range(len(v2_units)), [UNIT_LABEL[u] for u in v2_units])
            ax.xaxis.set_major_locator(MaxNLocator(14, integer=True))
        axes[-1].set_xlabel("Transformer block")
        return _save(fig, out / "F3_q3_allocation.pdf")


def fig_q3(D, out, unit_values, v2_units=("attn_in", "attn_out", "mlp_in", "mlp_out"),
           name="F3_full_q3_allocation.pdf"):
    pq = D["probe_quant"]["values"]["QS1"]
    bits_show = [b for b in ("2", "3", "4") if b in unit_values]
    with plt.rc_context(RC):
        n = len(bits_show) + 3
        fig, axes = plt.subplots(n, 1, figsize=(WIDTH, 1.15 * n + 0.6), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1] * (n - 1) + [1.2]})
        vmax = max(np.nanmax(np.abs(unit_values[b])) for b in bits_show)
        norm = SymLogNorm(linthresh=1e-3, vmin=-vmax, vmax=vmax, base=10)
        for ax, b in zip(axes, bits_show):
            im = ax.imshow(unit_values[b], aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
            ax.set_yticks(range(len(v2_units)), [UNIT_LABEL[u] for u in v2_units])
            ax.set_title(f"Isolated calibration NLL increase, one unit at {b} bits")
            ax.xaxis.set_major_locator(MaxNLocator(14, integer=True))
        fig.colorbar(im, ax=axes[:len(bits_show)], label="NLL increase (nats)", shrink=0.9)
        cmap = ListedColormap(["#c44e52", "#dd8452", "#8fbcd4", "#2a4d8f"])
        bnorm = BoundaryNorm([1.5, 2.5, 3.5, 6, 9], cmap.N)
        for ax, c in zip(axes[len(bits_show):len(bits_show) + 2], ("q3_int4", "q3_int3")):
            sel = D["select"][c]["selected"]
            ids, bits = sel["per_unit"]["ids"], sel["per_unit"]["bits"]
            m = np.full(unit_values[bits_show[0]].shape, np.nan)
            for uid, b in zip(ids, bits):
                layer, unit = uid.split(".")
                m[list(v2_units).index(unit), int(layer)] = b
            im2 = ax.imshow(m, aspect="auto", cmap=cmap, norm=bnorm, interpolation="nearest")
            ax.set_yticks(range(len(v2_units)), [UNIT_LABEL[u] for u in v2_units])
            ax.set_title(f"Final bit widths, {C.CONDITIONS[c]['budget'].replace('q2_b', 'INT')} budget")
            ax.xaxis.set_major_locator(MaxNLocator(14, integer=True))
        cb = fig.colorbar(im2, ax=axes[len(bits_show):len(bits_show) + 2], ticks=[2, 3, 4, 8], shrink=0.9)
        cb.set_label("Bits")
        ax = axes[-1]
        x = np.arange(len(pq[next(iter(pq))]))
        for b, col in zip(sorted(pq, key=int), ("#c44e52", "#dd8452", "#8fbcd4", "#2a4d8f")):
            ax.plot(x, np.clip(pq[b], 1e-5, None), marker="o", markersize=2.5, color=col, label=f"{b} bits")
        ax.set_yscale("log")
        ax.set_title("Block-level design (not evaluated): whole block at one width")
        ax.set_ylabel("NLL increase")
        ax.set_xlabel("Transformer block")
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(14, integer=True))
        for a in axes[:-1]:
            a.set_xticklabels([])
        return _save(fig, out / name)


LABEL_OFFSETS = [(5, 2), (5, -10), (-5, 2), (-5, -10), (0, 6), (0, -14), (8, -4), (-8, -4)]


def place_labels(ax, xs, ys, labels, marker_pt):
    """Label points without overlap: each label is tried at fixed offsets and kept only if its box lies inside
    the axes and overlaps neither another label nor any marker; labels of crowded points are skipped.
    Points are labelled in order of distance to their nearest neighbour (display units), largest first.
    Returns {"shown": [...], "skipped": [...]}."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    px = fig.dpi / 72.0
    half = 0.5 * marker_pt * px + 1.0
    markers = [(p[0] - half, p[1] - half, p[0] + half, p[1] + half) for p in pts]
    frame = ax.get_window_extent(renderer)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)) + np.diag(np.full(len(pts), np.inf))
    order = sorted(range(len(pts)), key=lambda k: -d[k].min())

    def hits(bb, box):
        return bb.x0 < box[2] and bb.x1 > box[0] and bb.y0 < box[3] and bb.y1 > box[1]

    placed, shown, skipped = [], [], []
    for k in order:
        ok = None
        for dx, dy in LABEL_OFFSETS:
            t = ax.annotate(labels[k], (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset points",
                            ha="left" if dx > 0 else ("right" if dx < 0 else "center"), fontsize=FONT)
            bb = t.get_window_extent(renderer)
            inside = (bb.x0 >= frame.x0 + 1 and bb.x1 <= frame.x1 - 1 and bb.y0 >= frame.y0 + 1
                      and bb.y1 <= frame.y1 - 1)
            if inside and not any(bb.overlaps(p) for p in placed) and not any(hits(bb, m) for m in markers):
                ok = (t, bb)
                break
            t.remove()
        if ok is None:
            skipped.append(labels[k])
        else:
            placed.append(ok[1])
            shown.append(labels[k])
    return {"shown": shown, "skipped": skipped}


def _family_handle(fam, open_marker):
    st = FAM_STYLE[fam]
    return Line2D([], [], linestyle="none", marker=st["marker"], color=st["color"],
                  markersize=7 if fam == "baseline" else plt.rcParams["lines.markersize"],
                  fillstyle="none" if open_marker else "full", label=st["label"])


F4_FAMILIES = ("baseline", "P1", "P2", "P3", "P2_2of4", "Q1", "Q2", "Q3")


def fig_calibration(qrows, out, info=None):
    """Calibration vs held-out mean NLL, one panel per held-out suite, log-log; pruning families filled
    markers, quantization families open markers (colours and marker shapes as in F1)."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), layout="constrained")
        handles = [_family_handle(f, f.startswith("Q")) for f in F4_FAMILIES]
        xs = [r["calibration_mean_nll"] for r in qrows]
        for ax, s in zip(axes, ("wikitext2_test", "c4_validation")):
            ys = [r[f"{s}_mean_nll"] for r in qrows]
            for r, x, y in zip(qrows, xs, ys):
                h = handles[F4_FAMILIES.index(r["family"])]
                ax.plot([x], [y], linestyle="none", marker=h.get_marker(), color=h.get_color(),
                        markersize=h.get_markersize(), fillstyle=h.get_fillstyle(), zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo, hi = min(xs + ys), max(xs + ys)
            ax.set_xlim(lo / 1.08, hi * 1.25)
            ax.set_ylim(lo / 1.08, hi * 1.25)
            ax.plot([lo / 1.08, hi * 1.25], [lo / 1.08, hi * 1.25], color="0.8", linewidth=0.8, zorder=1)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("C4 calibration mean NLL (nats)")
            ax.set_ylabel(f"{SUITE_LABEL[s]} mean NLL (nats)")
            ax.set_title(SUITE_LABEL[s])
        fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4, handletextpad=0.2,
                   columnspacing=1.0)
        fig.canvas.draw()
        fig.set_layout_engine("none")
        placed = {}
        for ax, s in zip(axes, ("wikitext2_test", "c4_validation")):
            ys = [r[f"{s}_mean_nll"] for r in qrows]
            placed[s] = place_labels(ax, xs, ys, [r["label"] for r in qrows], marker_pt=7)
        if info is not None:
            info["F4_point_labels"] = placed
        return _save(fig, out / "F4_calibration_vs_heldout.pdf")


def fig_magnitude_wanda(D, out, proj_records):
    types = list(C.ELIGIBLE)
    short = [t.replace("_proj", "") for t in types]
    colors = plt.get_cmap("tab10")(np.arange(len(types)))
    sp = list(C.SPARSITIES)
    recs = {s: proj_records(D, f"p2_s{int(s * 100)}") for s in sp}
    levels = D["cond"]["p2_s50"]["run1"]["steps"]["build"]["score_distributions"]["quantile_levels"]
    i10, i90 = levels.index(0.1), levels.index(0.9)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), constrained_layout=True)
        ax = axes[0]
        for t, lab, col in zip(types, short, colors):
            ys = [100 * np.mean([r["disagreement_with_magnitude"] for r in recs[s] if r["name"] == t]) for s in sp]
            ax.plot([100 * s for s in sp], ys, marker="o", color=col, label=lab)
        ax.set_xlabel("Target sparsity (%)")
        ax.set_ylabel("Mask disagreement (%)")
        ax.set_title("(a) Mean over blocks")
        ax.set_xticks([100 * s for s in sp])
        _thin(ax, 4, 4)
        ax = axes[1]
        for t, col in zip(types, colors):
            rr = [r for r in recs[0.5] if r["name"] == t]
            ax.plot([r["x_norm_cv_dense"] for r in rr], [100 * r["disagreement_with_magnitude"] for r in rr],
                    linestyle="none", marker="o", markersize=2.5, color=col)
        ax.set_xscale("log")
        _plain_log(ax, "x")
        ax.set_xlabel("Input-norm coeff. of variation")
        ax.set_ylabel("Mask disagreement at 50% (%)")
        ax.set_title("(b) 196 projections, 50%")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax = axes[2]
        w = 0.38
        xm = np.arange(len(types))
        spread = {}
        for key in ("dist_magnitude", "dist_wanda"):
            spread[key] = [np.mean([np.log10(r[key]["quantiles"][i90] / r[key]["quantiles"][i10])
                                    for r in recs[0.5] if r["name"] == t]) for t in types]
        ax.bar(xm - w / 2, spread["dist_magnitude"], w, color="#9e3b2b", label="Magnitude |W|")
        ax.bar(xm + w / 2, spread["dist_wanda"], w, color="#1f5fa8", label="Wanda |W| x input norm")
        lo = min(min(v) for v in spread.values())
        hi = max(max(v) for v in spread.values())
        ax.set_ylim(lo - 0.15, hi + 0.4)
        ax.set_xticks(xm, short, rotation=45)
        ax.set_ylabel("log10(90th / 10th percentile)")
        ax.set_title("(c) Score spread, 50%")
        ax.legend(frameon=False, loc="upper left", fontsize=FONT)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=7)
        return _save(fig, out / "F5_magnitude_vs_wanda.pdf")


FIGURE_FILES = ("F1_quality_vs_resource.pdf", "F2_p3_allocation.pdf", "F3_q3_allocation.pdf",
                "F3_full_q3_allocation.pdf", "F4_calibration_vs_heldout.pdf", "F5_magnitude_vs_wanda.pdf")


def make_all(D, qrows, rrows, out, info=None):
    from task1.analysis import proj_records, q3v2_unit_values
    uv = q3v2_unit_values(D, "QS1v2")
    names = [fig_quality(qrows, rrows, out), fig_p3(D, out), fig_q3_main(D, out, uv), fig_q3(D, out, uv),
             fig_calibration(qrows, out, info), fig_magnitude_wanda(D, out, proj_records)]
    assert tuple(names) == FIGURE_FILES, names
    return names
