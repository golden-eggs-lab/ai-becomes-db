#!/usr/bin/env python
"""
Unified Multi-Task ANN Hyperparameter Study Plot

Generates 4-panel figure: recall/time × nprobe/nlist, one line per task.
Each task's default anchor is marked with a unique open marker.

Usage:
    python plot_ann_hyperparam_multitask.py \
        --semdedup /path/to/semdedup.json \
        --cal      /path/to/cal.json \
        --ood      /path/to/ood.json \
        --code     /path/to/codepruning.json \
        --coreset  /path/to/coreset.json
"""

import json, argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# Distinct colors and markers per task (publication-quality)
TASK_STYLES = {
    "SemDeDup": {
        "color": "#1f77b4",  # blue
        "marker": "o",
        "default_marker": "o",
        "label": "SemDeDup",
    },
    "CAL": {
        "color": "#ff7f0e",  # orange
        "marker": "s",
        "default_marker": "s",
        "label": "CAL",
    },
    "KNN-OOD": {
        "color": "#2ca02c",  # green
        "marker": "^",
        "default_marker": "^",
        "label": "KNN-OOD",
    },
    "CodePruning": {
        "color": "#d62728",  # red
        "marker": "D",
        "default_marker": "D",
        "label": "CodePruning",
    },
    "Coreset": {
        "color": "#9467bd",  # purple
        "marker": "v",
        "default_marker": "v",
        "label": "Coreset",
    },
}


def load_results(path):
    """Load JSON results file."""
    with open(path) as f:
        return json.load(f)


def extract_sweep_data(data, sweep_key, param_key, anchor_nlist=100, anchor_nprobe=10):
    """Extract x, y_recall, y_time arrays from sweep data."""
    sweep = data[sweep_key]
    xs = [r[param_key] for r in sweep]
    recall_mean = [r["recall_mean"] for r in sweep]
    recall_std  = [r.get("recall_std", 0) for r in sweep]
    time_mean   = [r["search_time_mean"] for r in sweep]
    time_std    = [r.get("search_time_std", 0) for r in sweep]
    return xs, recall_mean, recall_std, time_mean, time_std


def find_default_point(xs, means, stds, default_val):
    """Find the (x, y) for the task's default parameter, or interpolate."""
    if default_val in xs:
        idx = xs.index(default_val)
        return default_val, means[idx]
    # Default not in sweep — find nearest
    closest_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - default_val))
    return xs[closest_idx], means[closest_idx]


def plot_panel(ax, tasks_data, sweep_key, param_key, y_key, y_label,
               x_label, default_param_field, mark_defaults=True):
    """Plot one panel of the 4-panel figure."""
    for task_name, data in tasks_data.items():
        if data is None:
            continue
        
        style = TASK_STYLES[task_name]
        sweep = data[sweep_key]
        xs = [r[param_key] for r in sweep]
        ys = [r[f"{y_key}_mean"] for r in sweep]
        yerr = [r.get(f"{y_key}_std", 0) for r in sweep]
        
        # Main line
        ax.errorbar(xs, ys, yerr=yerr,
                     color=style["color"],
                     marker=style["marker"],
                     markersize=5,
                     markerfacecolor=style["color"],
                     markeredgecolor=style["color"],
                     linewidth=1.5,
                     capsize=3,
                     label=style["label"],
                     zorder=3)
        
        # Mark default with hollow ring
        if mark_defaults:
            default_val = data.get(default_param_field)
            if default_val is not None and default_val in xs:
                idx = xs.index(default_val)
                ax.plot(xs[idx], ys[idx],
                        marker=style["default_marker"],
                        markersize=12,
                        markerfacecolor="none",
                        markeredgecolor=style["color"],
                        markeredgewidth=2.0,
                        linestyle="none",
                        zorder=5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    
    # Clean x-axis ticks
    all_xs = set()
    for task_name, data in tasks_data.items():
        if data is not None:
            for r in data[sweep_key]:
                all_xs.add(r[param_key])
    all_xs = sorted(all_xs)
    ax.set_xticks(all_xs)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.grid(True, alpha=0.3, linestyle="--")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semdedup", type=str, default=None)
    parser.add_argument("--cal",     type=str, default=None)
    parser.add_argument("--ood",     type=str, default=None)
    parser.add_argument("--code",    type=str, default=None)
    parser.add_argument("--coreset", type=str, default=None)
    parser.add_argument("--output",  type=str, default="ann_hyperparam_multitask.pdf")
    args = parser.parse_args()
    
    # Load all available results
    tasks = {}
    if args.semdedup:
        tasks["SemDeDup"] = load_results(args.semdedup)
    if args.cal:
        tasks["CAL"] = load_results(args.cal)
    if args.ood:
        tasks["KNN-OOD"] = load_results(args.ood)
    if args.code:
        tasks["CodePruning"] = load_results(args.code)
    if args.coreset:
        tasks["Coreset"] = load_results(args.coreset)
    
    if not tasks:
        print("ERROR: no result files provided. Use --semdedup, --cal, etc.")
        return
    
    print(f"Loaded {len(tasks)} tasks: {list(tasks.keys())}")
    
    # ── create 2×2 figure ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    
    # Top-left: Recall vs nprobe
    plot_panel(axes[0, 0], tasks, "nprobe_sweep", "nprobe", "recall",
               "Recall", r"$n_{\mathrm{probe}}$", "default_nprobe")
    axes[0, 0].set_title(r"Recall vs. $n_{\mathrm{probe}}$")
    
    # Top-right: Recall vs nlist
    plot_panel(axes[0, 1], tasks, "nlist_sweep", "nlist", "recall",
               "Recall", r"$n_{\mathrm{list}}$", "default_nlist")
    axes[0, 1].set_title(r"Recall vs. $n_{\mathrm{list}}$")
    
    # Bottom-left: Search Time vs nprobe
    plot_panel(axes[1, 0], tasks, "nprobe_sweep", "nprobe", "search_time",
               "Search Time (s)", r"$n_{\mathrm{probe}}$", "default_nprobe")
    axes[1, 0].set_title(r"Search Time vs. $n_{\mathrm{probe}}$")
    
    # Bottom-right: Search Time vs nlist
    plot_panel(axes[1, 1], tasks, "nlist_sweep", "nlist", "search_time",
               "Search Time (s)", r"$n_{\mathrm{list}}$", "default_nlist")
    axes[1, 1].set_title(r"Search Time vs. $n_{\mathrm{list}}$")
    
    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Add a "Default" marker to legend
    from matplotlib.lines import Line2D
    default_handle = Line2D([0], [0], marker='o', color='gray',
                            markerfacecolor='none', markeredgecolor='gray',
                            markeredgewidth=2.0, markersize=10,
                            linestyle='none', label='Default')
    handles.append(default_handle)
    labels.append('Default')
    
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=True, framealpha=0.9, edgecolor='lightgray',
               bbox_to_anchor=(0.5, -0.02))
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"✓ Saved to {args.output}")
    
    # Also save PNG
    png_path = args.output.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"✓ Saved to {png_path}")


if __name__ == "__main__":
    main()
