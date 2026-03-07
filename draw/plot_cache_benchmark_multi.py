#!/usr/bin/env python
"""
Plot Multi-Algorithm LRU Cache Benchmark results.

Generates normalized time plot (operation_time / unlimited_time).
Style matches plot_scaling_results.py exactly.
No in-plot legend — shares legend with scaling plot.

Usage:
    python plot_cache_benchmark_multi.py /path/to/results_multi.json
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── style (matching plot_scaling_results.py exactly) ─────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 30,
    "figure.dpi": 300,
})

# ── algorithm style config (matching plot_scaling_results.py STYLE) ──
ALGO_STYLE = {
    "CAL":          {"color": "#ff7f0e", "marker": "s", "linestyle": "--",              "label": "CAL"},
    "SemDeDup":     {"color": "#1f77b4", "marker": "o", "linestyle": "-",               "label": "SemDeDup"},
    "KNNPrompting": {"color": "#9467bd", "marker": "v", "linestyle": (0, (3, 1, 1, 1)), "label": "KNN Prompting"},
    "SCIP":         {"color": "#8c564b", "marker": "P", "linestyle": (0, (5, 2)),       "label": "SCIP"},
    "CRAIG":        {"color": "#17becf", "marker": "<", "linestyle": (0, (1, 1)),       "label": "CRAIG"},
}

OUTDIR = "/home/yichen/old/draw"


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def plot_cache_benchmark(all_results, outdir):
    """Plot Normalized Time vs Cache Max Size — no legend, matches scaling style."""

    plot_order = ["CAL", "SemDeDup", "KNNPrompting", "SCIP", "CRAIG"]
    figsize = (5.5, 3.8)  # matches scaling plots

    # ── Normalized Time Plot ──
    fig, ax = plt.subplots(figsize=figsize)

    for algo_name in plot_order:
        if algo_name not in all_results:
            continue
        results = all_results[algo_name]
        style = ALGO_STYLE[algo_name]

        sizes = []
        means = []
        stds = []
        for r in results:
            cs = r['cache_max_size']
            if cs is None:
                sizes.append(2000)
            else:
                sizes.append(cs)
            means.append(r['cache_time_mean'])
            stds.append(r['cache_time_std'])

        sizes = np.array(sizes, dtype=float)
        means = np.array(means)
        stds = np.array(stds)

        # Normalize by unlimited (last entry)
        unlimited_mean = means[-1] if means[-1] > 0 else 1.0
        stds = stds / unlimited_mean
        means = means / unlimited_mean

        ax.plot(sizes, means,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                markersize=12,
                markerfacecolor='none',
                markeredgewidth=2,
                linewidth=2,
                label=style['label'],
                zorder=3)

        ax.fill_between(sizes, means - stds, means + stds,
                        color=style['color'], alpha=0.18, zorder=2)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel("Cache Max Size")
    ax.set_ylabel("Time Ratio")
    ax.yaxis.set_label_coords(-0.12, 0.42)

    ax.set_xscale("log")
    finite_ticks = [10, 20, 50, 100, 200, 500, 1000]
    ax.set_xticks(finite_ticks + [2000])
    ax.set_xticklabels([str(t) for t in finite_ticks] + ["∞"], fontsize=16)
    ax.set_xlim(7, 3500)
    ax.grid(True, alpha=0.3, linestyle='--')

    # No legend — shares with scaling legend
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        outpath = os.path.join(outdir, f"cache_benchmark_multi_normalized.{ext}")
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_cache_benchmark_multi.py <results_multi.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    all_results = load_results(json_path)

    print(f"Loaded results for algorithms: {list(all_results.keys())}")
    for algo, results in all_results.items():
        print(f"  {algo}: {len(results)} cache sizes")

    plot_cache_benchmark(all_results, OUTDIR)


if __name__ == "__main__":
    main()
