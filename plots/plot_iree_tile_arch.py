"""Cross-architecture plots for the tile-level ukr-mlir vs IREE comparison.

Each architecture is a directory with either run_*.csv files from
test/iree_tile_bench.c (aggregated: median of the per-run medians, min/max across
runs as error bars) or, as a fallback, a summary_medians.csv with
already-aggregated medians (columns impl,M0,N0,K,accumulate,mode,median_gflops,
peak_lo,peak_hi).

Writes:
  1_ksplit_narrow.png     narrow tiles, IREE vs ukr-mlir without K split (v1) vs
                          with K split: one row per architecture, one column per K
  2_larger_shapes.png     IREE's largest tile vs ukr-mlir at the same shape and at
                          a larger shape IREE does not have (warm and cold)
  3_speedup_heatmap.png   ukr-mlir / IREE for every tile, K and cache mode, all
                          architectures in one view

Usage:
  python3 plots/plot_iree_tile_arch.py [--out plots/iree_tile_arch]
  (architectures and their data dirs: DEFAULT_ARCHS below)
"""

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

# Reference palette (dataviz skill, light mode), validated: blue / orange / aqua
# pass every check; aqua is below 3:1 contrast, so its bars carry value labels.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY_V1 = "#a8a7a2"  # the version without K split: de-emphasized, not a slot
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#eb6834"

# Series colour follows the entity in every figure.
C_IREE, C_V1, C_UKR, C_BIG = BLUE, GREY_V1, ORANGE, AQUA

KS = [64, 256, 1024]

# Architecture -> data dir, IREE's tile shapes (narrow first, largest last), and
# the larger ukr-only shape. AVX2 uses sorgan-cpu6; cpu5 agrees within a few %.
DEFAULT_ARCHS = [
    ("Apple M4 (NEON)", "results/iree_tile_m4",
     [(1, 8), (2, 8), (4, 8), (8, 8)], (8, 12)),
    ("Xeon E-2224 (AVX2)", "results/iree_tile_x86/sorgan-cpu6",
     [(1, 8), (2, 8), (4, 8), (8, 8)], (6, 16)),
    ("Xeon Gold 6252 (AVX-512)", "results/iree_tile_x86/sorgan-cpu2",
     [(1, 16), (2, 16), (4, 16), (8, 16), (16, 16)], (12, 32)),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 10,
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


def load(path):
    """Returns (agg, (peak_lo, peak_hi), nruns). agg[(impl, M0, N0, K, acc, mode)]
    = {"med", "min", "max"}; min/max equal med for pre-aggregated data."""
    files = sorted(glob.glob(os.path.join(path, "run_*.csv")))
    if files:
        per_run, peaks = defaultdict(list), []
        for f in files:
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            peaks.append(float(rows[0]["peak_gflops"]))
            for r in rows:
                key = (r["impl"], int(r["M0"]), int(r["N0"]), int(r["K"]),
                       int(r["accumulate"]), r["mode"])
                per_run[key].append(float(r["median_gflops"]))
        agg = {k: {"med": statistics.median(v), "min": min(v), "max": max(v)}
               for k, v in per_run.items()}
        return agg, (min(peaks), max(peaks)), len(files)
    summary = os.path.join(path, "summary_medians.csv")
    with open(summary) as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    agg = {}
    for r in rows:
        v = float(r["median_gflops"])
        agg[(r["impl"], int(r["M0"]), int(r["N0"]), int(r["K"]),
             int(r["accumulate"]), r["mode"])] = {"med": v, "min": v, "max": v}
    return agg, (float(rows[0]["peak_lo"]), float(rows[0]["peak_hi"])), 0


def get(agg, impl, shape, k, mode, acc=0):
    return agg.get((impl, shape[0], shape[1], k, acc, mode))


def draw_peak(ax, peak, label=True):
    lo, hi = peak
    if hi - lo > 2:
        ax.axhspan(lo, hi, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax.axhline(hi, color=TEXT_2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    if label:
        txt = f"FMA peak {lo:.0f}-{hi:.0f}" if hi - lo > 2 else f"FMA peak {hi:.0f}"
        ax.text(0.99, hi, txt, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=7.5, color=TEXT_2)


def bars(ax, x, vals, width, color, label=None, value_labels=False, fmt=None):
    """Thin bars with min/max whiskers. vals: list of agg entries or None."""
    hs = [v["med"] if v else 0 for v in vals]
    xs = [xi for xi, v in zip(x, vals) if v]
    ax.bar(xs, [h for h, v in zip(hs, vals) if v], width, color=color, label=label,
           edgecolor=SURFACE, linewidth=1, zorder=2)
    err = [[v["med"] - v["min"] for v in vals if v], [v["max"] - v["med"] for v in vals if v]]
    if any(e > 0 for e in err[0] + err[1]):
        ax.errorbar(xs, [v["med"] for v in vals if v], yerr=err, fmt="none",
                    ecolor=TEXT_2, elinewidth=0.8, capsize=0, zorder=3)
    if value_labels:
        for xi, v in zip(x, vals):
            if v:
                ax.text(xi, v["max"] + 2, fmt(v) if fmt else f"{v['med']:.0f}",
                        ha="center", va="bottom", fontsize=7, color=TEXT)


def footer(fig, text):
    fig.text(0.01, 0.005, text, color=TEXT_MUTED, fontsize=7.5, ha="left", va="bottom")


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. K split on narrow tiles.
# ---------------------------------------------------------------------------
def plot_ksplit(archs, out, mode="warm"):
    fig, axes = plt.subplots(len(archs), len(KS), figsize=(12, 3.0 * len(archs) + 1.2),
                             squeeze=False)
    for r, a in enumerate(archs):
        shapes = a["shapes"]
        ymax = a["peak"][1] * 1.18
        for c, k in enumerate(KS):
            ax = axes[r][c]
            x = list(range(len(shapes)))
            w = 0.26
            iree = [get(a["agg"], "iree", s, k, mode) for s in shapes]
            v1 = [get(a["agg"], "ukr-v1", s, k, mode) for s in shapes]
            ukr = [get(a["agg"], "ukr", s, k, mode) for s in shapes]
            bars(ax, [i - w for i in x], iree, w, C_IREE, "IREE (hand-written)")
            bars(ax, x, v1, w, C_V1, "ukr-mlir v1 (no K split)")
            bars(ax, [i + w for i in x], ukr, w, C_UKR, "ukr-mlir (K split)")
            # Speedup over IREE above the K-split bar: the number the figure is about.
            for i, (u, b) in enumerate(zip(ukr, iree)):
                if u and b:
                    ax.text(i + w, u["max"] + 2, f"{u['med'] / b['med']:.1f}x",
                            ha="center", va="bottom", fontsize=7, color=TEXT,
                            fontweight="bold" if u["med"] / b["med"] >= 1.3 else "normal")
            draw_peak(ax, a["peak"], label=(c == len(KS) - 1))
            ax.set_xticks(x, [f"{m}x{n}" for m, n in shapes])
            ax.set_ylim(0, ymax)
            if r == 0:
                ax.set_title(f"K = {k}")
            if c == 0:
                ax.set_ylabel(f"{a['name']}\nGFLOP/s", color=TEXT, fontsize=9)
    handles = [Patch(color=C_IREE, label="IREE (hand-written)"),
               Patch(color=C_V1, label="ukr-mlir v1 (no K split)"),
               Patch(color=C_UKR, label="ukr-mlir (K split chosen by the generator)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(f"K split on narrow tiles: few accumulators cannot hide FMA latency "
                 f"({mode} cache, no accumulate)", fontsize=11, fontweight="bold",
                 color=TEXT, y=0.995)
    footer(fig, "Labels: ukr-mlir / IREE. Median of per-run medians, whiskers = min/max "
                "across runs (5 runs per machine). f32, 1 thread, IREE 8071457.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, out, "1_ksplit_narrow.png")


# ---------------------------------------------------------------------------
# 2. Larger shapes than IREE's largest tile.
# ---------------------------------------------------------------------------
def plot_larger(archs, out):
    modes = ["warm", "cold"]
    fig, axes = plt.subplots(len(modes), len(archs), figsize=(12, 7.2), squeeze=False)
    for c, a in enumerate(archs):
        top, big = a["shapes"][-1], a["big"]
        tl, bl = f"{top[0]}x{top[1]}", f"{big[0]}x{big[1]}"
        for r, mode in enumerate(modes):
            ax = axes[r][c]
            x = list(range(len(KS)))
            w = 0.26
            iree = [get(a["agg"], "iree", top, k, mode) for k in KS]
            same = [get(a["agg"], "ukr", top, k, mode) for k in KS]
            larger = [get(a["agg"], "ukr", big, k, mode) for k in KS]
            bars(ax, [i - w for i in x], iree, w, C_IREE, value_labels=True)
            bars(ax, x, same, w, C_UKR, value_labels=True)
            bars(ax, [i + w for i in x], larger, w, C_BIG, value_labels=True,
                 fmt=lambda v: f"{v['med']:.0f}")
            draw_peak(ax, a["peak"])
            # The ratio the figure is about goes under each K, clear of the bars.
            ticks = []
            for k, l, b in zip(KS, larger, iree):
                ratio = f"{l['med'] / b['med']:.2f}x" if l and b else "n/a"
                ticks.append(f"K={k}\n{ratio}")
            ax.set_xticks(x, ticks)
            for t in ax.get_xticklabels():
                t.set_fontsize(8)
            ax.set_ylim(0, a["peak"][1] * 1.25)
            if r == 0:
                ax.set_title(a["name"])
            if c == 0:
                ax.set_ylabel(f"{mode} cache\nGFLOP/s", color=TEXT, fontsize=9)
            ax.legend(handles=[Patch(color=C_IREE, label=f"IREE {tl}"),
                               Patch(color=C_UKR, label=f"ukr-mlir {tl}"),
                               Patch(color=C_BIG, label=f"ukr-mlir {bl}")],
                      loc="upper left", fontsize=7.5, ncol=3, handlelength=1,
                      columnspacing=0.8)
    fig.suptitle("Larger tiles use more of the register file: ukr-mlir shapes IREE does "
                 "not ship vs IREE's largest tile", fontsize=11, fontweight="bold",
                 color=TEXT)
    footer(fig, "Under each K: larger ukr-mlir tile (aqua) / IREE's largest tile (blue). No accumulate. Whiskers = "
                "min/max across 5 runs. f32, 1 thread, IREE 8071457.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, out, "2_larger_shapes.png")


# ---------------------------------------------------------------------------
# 3. Speedup heatmap: every tile x K x mode, all architectures.
# ---------------------------------------------------------------------------
def plot_heatmap(archs, out):
    modes = ["hot", "warm", "cold"]
    cols = [(m, k) for m in modes for k in KS]
    rows, labels, groups = [], [], []
    for a in archs:
        top = a["shapes"][-1]
        start = len(rows)
        for s in a["shapes"] + [a["big"]]:
            ref = s if s != a["big"] else top
            row = []
            for m, k in cols:
                u, b = get(a["agg"], "ukr", s, k, m), get(a["agg"], "iree", ref, k, m)
                row.append(u["med"] / b["med"] if u and b else float("nan"))
            rows.append(row)
            name = f"{s[0]}x{s[1]}" + (f" vs IREE {top[0]}x{top[1]}" if s == a["big"] else "")
            labels.append(name)
        groups.append((a["name"], start, len(rows)))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.8))
    cmap = LinearSegmentedColormap.from_list("div", [DIV_LO, DIV_MID, DIV_HI])
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.5)
    im = ax.imshow(rows, cmap=cmap, norm=norm, aspect="auto")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color=TEXT, fontweight="bold" if v >= 1.5 else "normal")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=TEXT_MUTED)
    ax.set_xticks(range(len(cols)), [f"{m}\nK={k}" for m, k in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.grid(False)
    for name, s, e in groups:
        if s:
            ax.axhline(s - 0.5, color=SURFACE, linewidth=4)
        ax.text(-0.19, (s + e - 1) / 2, name.replace(" (", "\n("),
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, fontweight="bold", color=TEXT)
    for j in (len(KS) - 0.5, 2 * len(KS) - 0.5):
        ax.axvline(j, color=SURFACE, linewidth=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("ukr-mlir / IREE (>1: ukr-mlir faster)", color=TEXT_2)
    cb.outline.set_visible(False)
    ax.set_title("ukr-mlir / IREE throughput, every tile, K and cache mode "
                 "(no accumulate)", loc="left")
    footer(fig, "Median of 5 runs per machine. Larger ukr-mlir shapes are compared "
                "with IREE's largest tile.")
    fig.tight_layout(rect=(0.08, 0.03, 1, 1))
    save(fig, out, "3_speedup_heatmap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/iree_tile_arch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    archs = []
    for name, path, shapes, big in DEFAULT_ARCHS:
        agg, peak, nruns = load(path)
        archs.append({"name": name, "agg": agg, "peak": peak, "shapes": shapes, "big": big})
        print(f"{name}: {path} ({nruns or 'pre-aggregated'} runs, peak {peak[0]:.0f}-{peak[1]:.0f})")
    plot_ksplit(archs, args.out)
    plot_larger(archs, args.out)
    plot_heatmap(archs, args.out)


if __name__ == "__main__":
    main()
