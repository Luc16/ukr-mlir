"""Plots for the tile-level ukr-mlir vs IREE comparison (scripts/iree_tile_bench.sh).

Reads every run_*.csv produced by test/iree_tile_bench.c, aggregates across runs
(median of the per-run medians, min/max across runs as error bars) and writes:

  1_headline_8x8.png      8x8 tile: IREE vs ukr-mlir (specialized, BLIS-style) and ukr-mlir 8x12
  2_ratio_heatmap.png     ukr-mlir / IREE throughput for every M0 x K x cache mode x accumulate
  3_narrow_m0.png         throughput vs M0 (narrow tiles), one panel per K
  4_hot_accumulate.png    the hot-mode accumulate artifact in IREE's narrow tiles
  5_reliability.png       per-run FMA peak and run-to-run spread
  6_before_after.png      IREE vs ukr-mlir v1 (C interface, no K split) vs current
                          (only when the runs contain "ukr-v1")

Usage:
  .venv/bin/python plots/plot_iree_tile.py [--runs tmp/iree_tile/runs] [--out plots/iree_tile]
"""

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Reference palette (dataviz skill, light mode). Categorical slots in fixed order.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#e34948"
GREY_V1 = "#a8a7a2"  # superseded version: de-emphasized, not a categorical slot
DIVERGING_MID = "#f0efec"

# Series identity is fixed per entity, never per rank.
SERIES = {
    ("iree", 8): ("IREE 8x8 (hand-written)", BLUE),
    ("ukr", 8): ("ukr-mlir 8x8", ORANGE),
    ("ukr-blis", 8): ("ukr-mlir 8x8 BLIS-style", AQUA),
    ("ukr", 12): ("ukr-mlir 8x12", YELLOW),
    ("ukr-v1", 8): ("ukr-mlir v1 (C interface, no K split)", GREY_V1),
}

KS = [64, 256, 1024]
MODES = ["hot", "warm", "cold"]
M0S = [1, 2, 4, 8]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.labelcolor": TEXT,
})


def load(runs_dir):
    files = sorted(glob.glob(os.path.join(runs_dir, "run_*.csv")))
    if not files:
        raise SystemExit(f"no run_*.csv in {runs_dir}")
    per_run = defaultdict(list)  # key -> [median_gflops per run]
    peaks = []
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        peaks.append(float(rows[0]["peak_gflops"]))
        for r in rows:
            key = (r["impl"], int(r["M0"]), int(r["N0"]), int(r["K"]),
                   int(r["accumulate"]), r["mode"])
            per_run[key].append(float(r["median_gflops"]))
    agg = {}
    for key, vals in per_run.items():
        med = statistics.median(vals)
        agg[key] = {"med": med, "min": min(vals), "max": max(vals),
                    "spread": 100 * (max(vals) - min(vals)) / med}
    return agg, peaks, len(files)


def get(agg, impl, m0, n0, k, acc, mode):
    return agg[(impl, m0, n0, k, acc, mode)]


def has_impl(agg, impl):
    return any(key[0] == impl for key in agg)


def errbar(vals):
    return [[v["med"] - v["min"] for v in vals], [v["max"] - v["med"] for v in vals]]


def footer(fig, text):
    fig.text(0.01, 0.01, text, color=TEXT_MUTED, fontsize=8, ha="left", va="bottom")


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. Headline: the 8x8 tile and the 8x12 shape IREE does not have.
# ---------------------------------------------------------------------------
def plot_headline(agg, peaks, nruns, out, mode="warm", acc=0):
    series = [("iree", 8, 8), ("ukr", 8, 8), ("ukr-blis", 8, 8), ("ukr", 8, 12)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.2
    gap = 0.02
    peak = statistics.median(peaks)
    for si, (impl, m0, n0) in enumerate(series):
        label, color = SERIES[(impl, n0)]
        vals = [get(agg, impl, m0, n0, k, acc, mode) for k in KS]
        xs = [i + (si - 1.5) * (width + gap) for i in range(len(KS))]
        ax.bar(xs, [v["med"] for v in vals], width, color=color, label=label,
               yerr=errbar(vals), error_kw={"ecolor": TEXT_2, "elinewidth": 0.8, "capsize": 2})
        for x, v in zip(xs, vals):
            ax.text(x, v["max"] + 1.5, f"{v['med']:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_2)
    # 8x12 gain over IREE 8x8, the headline number.
    for i, k in enumerate(KS):
        base = get(agg, "iree", 8, 8, k, acc, mode)["med"]
        gain = get(agg, "ukr", 8, 12, k, acc, mode)["med"] / base - 1
        bar = get(agg, "ukr", 8, 12, k, acc, mode)["med"]
        ax.text(i + 1.5 * (width + gap), bar - 4, f"{gain:+.0%}\nvs IREE", ha="center",
                va="top", fontsize=8.5, color=TEXT, fontweight="bold")
    ax.axhline(peak, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.5, peak + 1.5, f"NEON FMA peak {peak:.0f}", ha="left",
            va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xticks(range(len(KS)), [f"K = {k}" for k in KS])
    ax.set_ylabel("GFLOP/s (1 P-core)")
    ax.set_ylim(0, peak + 12)
    ax.set_title("8x8: ukr-mlir matches IREE's hand-written kernel; ukr-mlir 8x12 beats it",
                 loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, fontsize=8.5,
              handlelength=1.2, columnspacing=1.2)
    footer(fig, f"Apple M4, f32, {mode} cache, {'accumulate' if acc else 'no accumulate'}. "
                f"Median of {nruns} runs (each the median of 15 samples);\n"
                "error bars = min/max across runs. Bold: 8x12 gain over IREE 8x8.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "1_headline_8x8.png")


# ---------------------------------------------------------------------------
# 2. Ratio heatmap: every configuration at once.
# ---------------------------------------------------------------------------
def plot_ratio_heatmap(agg, nruns, out):
    cmap = LinearSegmentedColormap.from_list("div", [RED, DIVERGING_MID, BLUE])
    lim = math.log2(3)  # colors saturate at 3x either way
    cols = [(mode, k) for mode in MODES for k in KS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, acc in zip(axes, [0, 1]):
        grid = []
        for m0 in M0S:
            row = []
            for mode, k in cols:
                i = get(agg, "iree", m0, 8, k, acc, mode)
                u = get(agg, "ukr", m0, 8, k, acc, mode)
                ratio = u["med"] / i["med"]
                significant = abs(ratio - 1) > max(0.02, max(i["spread"], u["spread"]) / 100)
                # IREE's hot accumulate drop: much slower than its own hot, no-accumulate run.
                artifact = (mode == "hot" and acc == 1 and
                            i["med"] < 0.85 * get(agg, "iree", m0, 8, k, 0, "hot")["med"])
                row.append((ratio, significant, artifact))
            grid.append(row)
        ax.imshow([[min(max(math.log2(r), -lim), lim) for r, _, _ in row] for row in grid],
                  cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for y, row in enumerate(grid):
            for x, (ratio, sig, art) in enumerate(row):
                txt = (f"{ratio:.1f}x" if ratio >= 1.5 else f"{ratio:.2f}") + ("*" if art else "")
                ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
                        color=TEXT if sig else TEXT_MUTED,
                        fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(cols)), [f"{k}" for _, k in cols], fontsize=8.5)
        for j, mode in enumerate(MODES):
            ax.text(j * 3 + 1, -0.6, mode, ha="center", va="bottom",
                    fontsize=9, color=TEXT_2, fontweight="bold")
            if j:
                ax.axvline(j * 3 - 0.5, color=SURFACE, linewidth=3)
        ax.set_yticks(range(len(M0S)), [f"{m}x8" for m in M0S])
        ax.set_xlabel("K")
        ax.set_title("no accumulate (beta = 0)" if acc == 0 else "accumulate (beta = 1)",
                     loc="left", pad=20)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("tile M0 x N0")
    fig.suptitle("ukr-mlir / IREE throughput  (blue: ukr-mlir faster, red: IREE faster; log color scale)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    footer(fig, "Bold: difference larger than 2% and than the run-to-run spread; grey: tie. "
                "* hot-mode artifact: IREE slows down when the same tile is accumulated into\n"
                f"over and over (not a realistic case). Median of {nruns} runs.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, out, "2_ratio_heatmap.png")


# ---------------------------------------------------------------------------
# 3. Narrow tiles: throughput vs M0.
# ---------------------------------------------------------------------------
def plot_narrow(agg, peaks, nruns, out, mode="warm", acc=0):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    peak = statistics.median(peaks)
    for ax, k in zip(axes, KS):
        lines = [("iree", "IREE", "o"), ("ukr", "ukr-mlir", "s")]
        if has_impl(agg, "ukr-v1"):
            lines.insert(1, ("ukr-v1", "ukr-mlir v1", "D"))
        for impl, label, marker in lines:
            color = SERIES[(impl, 8)][1]
            vals = [get(agg, impl, m0, 8, k, acc, mode) for m0 in M0S]
            ax.errorbar(M0S, [v["med"] for v in vals], yerr=errbar(vals), color=color,
                        marker=marker, markersize=6, linewidth=2, capsize=2,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
                        linestyle=(0, (4, 2)) if impl == "ukr-v1" else "-")
        r1 = get(agg, "ukr", 1, 8, k, acc, mode)["med"] / get(agg, "iree", 1, 8, k, acc, mode)["med"]
        ax.annotate(f"1x8: {r1:.2f}x IREE", (1, get(agg, "ukr", 1, 8, k, acc, mode)["med"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=TEXT)
        ax.axhline(peak, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(M0S, [f"{m}" for m in M0S])
        ax.minorticks_off()
        ax.set_xlabel("M0 (tile = M0 x 8)")
        ax.set_title(f"K = {k}", loc="left")
        ax.grid(True, axis="both")
    axes[0].text(1, peak - 3, f"peak {peak:.0f}", fontsize=8, color=TEXT_2, va="top")
    axes[0].set_ylabel("GFLOP/s")
    axes[0].set_ylim(0, peak + 8)
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Narrow tiles: throughput vs M0 "
                 f"({mode}, {'accumulate' if acc else 'no accumulate'})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    footer(fig, f"Median of {nruns} runs; error bars = min/max across runs. "
                "Narrow tiles are what IREE uses when M is small (e.g. M0 = 1 for matrix-vector).")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, out, "3_narrow_m0.png")


# ---------------------------------------------------------------------------
# 4. Hot-mode accumulate artifact (explains the outliers in the heatmap).
# ---------------------------------------------------------------------------
def plot_hot_accumulate(agg, nruns, out, k=64):
    conds = [("hot", 0, "hot, no accumulate", BLUE),
             ("hot", 1, "hot, accumulate", ORANGE),
             ("warm", 1, "warm, accumulate", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    width, gap = 0.26, 0.02
    narrow = [1, 2, 4]
    for ax, impl, title in zip(axes, ["iree", "ukr"], ["IREE", "ukr-mlir"]):
        for ci, (mode, acc, label, color) in enumerate(conds):
            vals = [get(agg, impl, m0, 8, k, acc, mode) for m0 in narrow]
            xs = [i + (ci - 1) * (width + gap) for i in range(len(narrow))]
            ax.bar(xs, [v["med"] for v in vals], width, color=color, label=label,
                   yerr=errbar(vals), error_kw={"ecolor": TEXT_2, "elinewidth": 0.8, "capsize": 2})
            for x, v in zip(xs, vals):
                ax.text(x, v["max"] + 1.5, f"{v['med']:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color=TEXT_2)
        ax.set_xticks(range(len(narrow)), [f"{m}x8" for m in narrow])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(f"GFLOP/s (K = {k})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.1), ncol=3, fontsize=9)
    fig.suptitle("Hot-mode artifact: IREE slows down when one tile is accumulated into repeatedly",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    footer(fig, "Likely cause: IREE loads the output tile before its K loop, so each call waits on "
                "the previous call's stores;\nukr-mlir reads C after the K loop. Exclude "
                f"'hot, accumulate' from comparisons. Median of {nruns} runs.")
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    save(fig, out, "4_hot_accumulate.png")


# ---------------------------------------------------------------------------
# 5. Reliability: peak per run and run-to-run spread.
# ---------------------------------------------------------------------------
def plot_reliability(agg, peaks, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [1, 2.2]})
    ax1.bar(range(1, len(peaks) + 1), peaks, 0.6, color=BLUE)
    for i, p in enumerate(peaks, 1):
        ax1.text(i, p + 1, f"{p:.0f}", ha="center", va="bottom", fontsize=8, color=TEXT_2)
    ax1.set_xticks(range(1, len(peaks) + 1))
    ax1.set_xlabel("run")
    ax1.set_ylabel("measured FMA peak (GFLOP/s)")
    ax1.set_ylim(0, max(peaks) * 1.12)
    ax1.set_title("Peak per run", loc="left")

    impls = [("iree", BLUE, "IREE"), ("ukr", ORANGE, "ukr-mlir")]
    for xi, (mode, k) in enumerate([(m, k) for m in MODES for k in KS]):
        for oi, (impl, color, label) in enumerate(impls):
            spreads = [v["spread"] for (im, m0, n0, kk, acc, md), v in agg.items()
                       if im == impl and kk == k and md == mode and n0 == 8]
            xs = [xi + (oi - 0.5) * 0.3] * len(spreads)
            ax2.scatter(xs, spreads, s=22, color=color, edgecolors=SURFACE, linewidths=1,
                        label=label if xi == 0 else None, zorder=3)
    ax2.axhline(3, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax2.text(8.4, 3.1, "3%", fontsize=8, color=TEXT_2, va="bottom", ha="right")
    ax2.set_xticks(range(9), [f"{m}\nK={k}" for m in MODES for k in KS], fontsize=8)
    ax2.set_ylabel("run-to-run spread (%)")
    ax2.set_ylim(0, None)
    ax2.set_title("Spread across runs, per configuration", loc="left")
    ax2.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Measurement reliability", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=TEXT)
    footer(fig, "Spread = (max - min) / median of the per-run medians. "
                "Each dot is one (M0, accumulate) configuration.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, out, "5_reliability.png")


# ---------------------------------------------------------------------------
# 6. Before/after: v1 (C interface, no K split) vs current, against IREE.
# ---------------------------------------------------------------------------
def plot_before_after(agg, peaks, nruns, out, mode="warm", acc=0):
    series = [("iree", "IREE (hand-written)"), ("ukr-v1", "ukr-mlir v1"),
              ("ukr", "ukr-mlir: bare entry + K split")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    width, gap = 0.26, 0.02
    peak = statistics.median(peaks)
    for ax, k in zip(axes, KS):
        for si, (impl, label) in enumerate(series):
            color = SERIES[(impl, 8)][1]
            vals = [get(agg, impl, m0, 8, k, acc, mode) for m0 in M0S]
            xs = [i + (si - 1) * (width + gap) for i in range(len(M0S))]
            ax.bar(xs, [v["med"] for v in vals], width, color=color, label=label,
                   yerr=errbar(vals), error_kw={"ecolor": TEXT_2, "elinewidth": 0.8, "capsize": 2})
            if impl == "ukr":
                for x, m0, v in zip(xs, M0S, vals):
                    r = v["med"] / get(agg, "iree", m0, 8, k, acc, mode)["med"]
                    ax.text(x, v["max"] + 2, f"{r:.2f}x" if r < 1.5 else f"{r:.1f}x",
                            ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="bold")
        ax.axhline(peak, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks(range(len(M0S)), [f"{m}x8" for m in M0S])
        ax.set_title(f"K = {k}", loc="left")
    axes[0].text(-0.45, peak + 1.5, f"peak {peak:.0f}", fontsize=8, color=TEXT_2, va="bottom")
    axes[0].set_ylabel("GFLOP/s")
    axes[0].set_ylim(0, peak + 14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.075), ncol=3, fontsize=9)
    fig.suptitle(f"Splitting K over independent accumulators fixes the narrow tiles "
                 f"({mode}, {'accumulate' if acc else 'no accumulate'})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    footer(fig, "Bold: current ukr-mlir / IREE. Narrow tiles have too few accumulators to hide the "
                "4-cycle FMA latency (IREE and v1 alike);\nthe generator now alternates the K loop "
                f"between independent accumulator sets. Median of {nruns} runs; error bars = "
                "min/max across runs.")
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    save(fig, out, "6_before_after.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="tmp/iree_tile/runs")
    parser.add_argument("--out", default="plots/iree_tile")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    agg, peaks, nruns = load(args.runs)
    plot_headline(agg, peaks, nruns, args.out)
    plot_ratio_heatmap(agg, nruns, args.out)
    plot_narrow(agg, peaks, nruns, args.out)
    plot_hot_accumulate(agg, nruns, args.out)
    plot_reliability(agg, peaks, args.out)
    if has_impl(agg, "ukr-v1"):
        plot_before_after(agg, peaks, nruns, args.out)


if __name__ == "__main__":
    main()
