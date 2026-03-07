#!/usr/bin/env python
"""
Plot Multi-Algorithm LRU Cache Benchmark (Ratio-Based).

X-axis: cache_size / working_set_size (shared across all algorithms).
Y-axis: normalized time (operation_time / unlimited_time).
Style matches plot_scaling_results.py exactly.
No in-plot legend — shares legend with scaling plot.

Usage:
    python plot_cache_benchmark_ratio.py /path/to/results_ratio.json
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
    """Plot Normalized Time vs Cache Size Ratio — no legend, matches scaling style."""

    plot_order = ["CAL", "SemDeDup", "KNNPrompting", "SCIP", "CRAIG"]
    figsize = (7, 3.8)

    fig, ax = plt.subplots(figsize=figsize)

    for algo_name in plot_order:
        if algo_name not in all_results:
            continue
        results = all_results[algo_name]
        style = ALGO_STYLE[algo_name]

        ratios = []
        means = []
        stds = []
        for r in results:
            cr = r.get('cache_ratio')
            if cr is None:
                continue  # skip unlimited — equivalent to 100%
            ratios.append(cr)
            means.append(r['cache_time_mean'])
            stds.append(r['cache_time_std'])

        ratios = np.array(ratios, dtype=float)
        means = np.array(means)
        stds = np.array(stds)

        # Normalize by ratio=1.0 (last finite entry, equivalent to unlimited)
        baseline_mean = means[-1] if means[-1] > 0 else 1.0
        stds = stds / baseline_mean
        means = means / baseline_mean

        ax.plot(ratios, means,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                markersize=12,
                markerfacecolor='none',
                markeredgewidth=2,
                linewidth=2,
                label=style['label'],
                zorder=3)

        ax.fill_between(ratios, means - stds, means + stds,
                        color=style['color'], alpha=0.18, zorder=2)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel("Cache Size Ratio")
    ax.set_ylabel("Time Ratio")
    ax.yaxis.set_label_coords(-0.12, 0.42)

    ax.set_xscale("log")
    finite_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_xticks(finite_ticks)
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%", "50%", "100%"], fontsize=16)
    ax.set_xlim(0.007, 1.5)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()

    for ext in ['pdf', 'png']:
        outpath = os.path.join(outdir, f"cache_benchmark_ratio.{ext}")
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_cache_benchmark_ratio.py <results_ratio.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    all_results = load_results(json_path)

    print(f"Loaded results for algorithms: {list(all_results.keys())}")
    for algo, results in all_results.items():
        ws = results[0].get('working_set_size', '?')
        print(f"  {algo}: {len(results)} ratios, working_set={ws}")

    plot_cache_benchmark(all_results, OUTDIR)


if __name__ == "__main__":
    main()
