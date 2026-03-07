#!/usr/bin/env python
"""
ANN hyperparameter sweep – SEPARATE panels + standalone legend.
v3: KNN-OOD uses GPU sweep data (ann_hyperparam_ood_20260214_063124.json)
    to match E2E GPU baseline. All other tasks unchanged.

Outputs 5 files (v3):
  ann_sweep_recall_nprobe_v3.pdf    (a) Recall vs nprobe
  ann_sweep_time_nprobe_v3.pdf      (b) Normalized Time vs nprobe
  ann_sweep_recall_nlist_v3.pdf     (c) Recall vs nlist
  ann_sweep_time_nlist_v3.pdf       (d) Normalized Time vs nlist
  ann_sweep_legend_v3.pdf           Standalone shared legend

5 tasks: SemDeDup (CIFAR-10), CAL, KNN-OOD (GPU), CodePruning (K=1000), CRAIG
"""

import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── style ────────────────────────────────────────────────────────────
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

OUTDIR = "/home/yichen/old/draw"

# ── task configs ─────────────────────────────────────────────────────
TASKS = {
    "SemDeDup": {
        "json": "/home/yichen/old/SemDeDup/results/ann_hyperparam_multi_20260215_005833.json",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "label": "SemDeDup",
        "time_key": "search_time",
        "exact_time": None,   # nprobe=100 proxy
    },
    "CAL": {
        "json": "/home/yichen/old/contrastive-active-learning/results/ann_hyperparam_cal_20260214_005135.json",
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "label": "CAL",
        "time_key": "search_time",
        "exact_time": (12.3044 + 12.3584 + 12.8316) / 3,
    },
    "KNN-OOD": {
        # v3: GPU sweep (use_gpu=true, anchor_nlist=1000)
        "json": "/home/yichen/old/knn-ood/results/ann_hyperparam_ood_20260214_063124.json",
        "color": "#d62728",
        "marker": "D",
        "linestyle": ":",
        "label": "KNN-OOD",
        "time_key": "search_time",
        # GPU exact search for 50k queries: from sweep log Run 1 "Exact search (GPU): 30.84s"
        # Cross-validated: E2E 85k queries = 53.97s → 50k/85.64k * 53.97 ≈ 31.5s
        "exact_time": 30.84,
    },
    "CodePruning": {
        "json": "/home/yichen/old/codepruning/results/ann_hyperparam_codepruning_20260214_043759.json",
        "color": "#8c564b",
        "marker": "P",
        "linestyle": (0, (5, 2)),
        "label": "SCIP",
        "time_key": "search_time",
        "exact_time": (8.5777 + 8.6727 + 8.5699) / 3,
    },
    "CRAIG": {
        "json": None,  # will auto-find latest
        "color": "#17becf",
        "marker": "^",
        "linestyle": "-.",
        "label": "CRAIG",
        "time_key": "search_time",
        "exact_time": None,   # from JSON
    },
}


def find_latest_craig_json():
    """Find latest CRAIG e2e JSON."""
    pattern = "/home/yichen/old/craig/results/ann_hyperparam_craig_*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print("WARNING: No CRAIG JSON found, skipping CRAIG")
        return None
    return files[-1]


def load_sweep(data, sweep_key, time_key, exact_time):
    """Load a sweep from JSON data."""
    sweep = data[sweep_key]
    param_key = "nprobe" if "nprobe" in sweep_key else "nlist"

    params = [r[param_key] for r in sweep]
    recall_mean = [r["recall_mean"] for r in sweep]
    time_mean = [r[f"{time_key}_mean"] for r in sweep]
    time_std = [r.get(f"{time_key}_std", 0) for r in sweep]

    norm_mean = [t / exact_time for t in time_mean]
    norm_std = [s / exact_time for s in time_std]

    return params, recall_mean, time_mean, time_std, norm_mean, norm_std


def main():
    print("Loading data... (v3 – KNN-OOD GPU)")

    # Auto-find CRAIG JSON
    if TASKS["CRAIG"]["json"] is None:
        craig_json = find_latest_craig_json()
        if craig_json:
            TASKS["CRAIG"]["json"] = craig_json
            print(f"  CRAIG JSON: {craig_json}")

    all_data = {}
    for name, cfg in TASKS.items():
        if cfg["json"] is None:
            continue
        with open(cfg["json"]) as f:
            data = json.load(f)

        exact_time = cfg["exact_time"]
        if exact_time is None:
            if "exact_search_time_mean" in data:
                exact_time = data["exact_search_time_mean"]
                print(f"  {name}: exact from JSON = {exact_time:.2f}s")
            else:
                last = data["nprobe_sweep"][-1]
                exact_time = last[f"{cfg['time_key']}_mean"]
                print(f"  {name}: nprobe={last['nprobe']} proxy = {exact_time:.4f}s")

        np_data = load_sweep(data, "nprobe_sweep", cfg["time_key"], exact_time)
        nl_data = load_sweep(data, "nlist_sweep", cfg["time_key"], exact_time)

        all_data[name] = {
            "nprobe": np_data, "nlist": nl_data,
            "exact_time": exact_time, "style": cfg,
        }
        print(f"  {name}: exact={exact_time:.2f}s, gpu={data.get('use_gpu', 'N/A')}")

    # ── individual figures ───────────────────────────────────────────
    all_nprobes = sorted(set(p for d in all_data.values() for p in d["nprobe"][0]))
    all_nlists = sorted(set(p for d in all_data.values() for p in d["nlist"][0]))

    figsize = (5.5, 3.8)

    # (a) Recall vs nprobe  — broken y-axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(figsize[0], figsize[1] + 0.4),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08},
    )
    for name, d in all_data.items():
        s = d["style"]
        for ax in (ax_top, ax_bot):
            ax.plot(d["nprobe"][0], d["nprobe"][1], color=s["color"], marker=s["marker"],
                    markersize=12, linewidth=2, linestyle=s["linestyle"],
                    label=s["label"], zorder=3,
                    markerfacecolor='none', markeredgewidth=2)
    ax_top.set_ylim(0.55, 1.03)
    ax_bot.set_ylim(0.35, 0.45)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # break marks
    d_brk = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d_brk, +d_brk), (-d_brk, +d_brk), **kwargs)
    ax_top.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d_brk, +d_brk), (1 - d_brk, 1 + d_brk), **kwargs)
    ax_bot.plot((1 - d_brk, 1 + d_brk), (1 - d_brk, 1 + d_brk), **kwargs)
    ax_bot.set_xlabel(r"$n_{\mathrm{probe}}$")
    fig.text(0.02, 0.45, "Selection Recall", va="center", rotation="vertical", fontsize=24)
    ax_bot.set_xscale("log")
    ax_bot.set_xticks(all_nprobes)
    ax_bot.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_bot.get_xaxis().set_tick_params(which="minor", size=0)
    for ax in (ax_top, ax_bot):
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.subplots_adjust(left=0.24)
    fig.savefig(f"{OUTDIR}/ann_sweep_recall_nprobe_v3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/ann_sweep_recall_nprobe_v3.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  ✓ recall_nprobe")

    # (b) Normalized Time vs nprobe
    fig, ax = plt.subplots(figsize=figsize)
    for name, d in all_data.items():
        s = d["style"]
        params = d["nprobe"][0]
        _, _, _, _, nm, ns = d["nprobe"]
        nm_arr = np.array(nm)
        ns_arr = np.array(ns)
        ax.plot(params, nm, color=s["color"], marker=s["marker"],
                markersize=12, linewidth=2, linestyle=s["linestyle"],
                label=s["label"], zorder=3,
                markerfacecolor='none', markeredgewidth=2)
        ax.fill_between(params, nm_arr - ns_arr, nm_arr + ns_arr,
                        color=s["color"], alpha=0.18, zorder=2)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel(r"$n_{\mathrm{probe}}$")
    ax.set_ylabel("Search Time Ratio")
    ax.yaxis.set_label_coords(-0.22, 0.42)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_nprobes)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which="minor", size=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.subplots_adjust(left=0.24)
    fig.savefig(f"{OUTDIR}/ann_sweep_time_nprobe_v3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/ann_sweep_time_nprobe_v3.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  ✓ time_nprobe")

    # (c) Recall vs nlist  — broken y-axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(figsize[0], figsize[1] + 0.4),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08},
    )
    for name, d in all_data.items():
        s = d["style"]
        for ax in (ax_top, ax_bot):
            ax.plot(d["nlist"][0], d["nlist"][1], color=s["color"], marker=s["marker"],
                    markersize=12, linewidth=2, linestyle=s["linestyle"],
                    label=s["label"], zorder=3,
                    markerfacecolor='none', markeredgewidth=2)
    ax_top.set_ylim(0.55, 1.03)
    ax_bot.set_ylim(0.35, 0.45)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # break marks
    d_brk = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d_brk, +d_brk), (-d_brk, +d_brk), **kwargs)
    ax_top.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d_brk, +d_brk), (1 - d_brk, 1 + d_brk), **kwargs)
    ax_bot.plot((1 - d_brk, 1 + d_brk), (1 - d_brk, 1 + d_brk), **kwargs)
    ax_bot.set_xlabel(r"$n_{\mathrm{list}}$")
    fig.text(0.02, 0.45, "Selection Recall", va="center", rotation="vertical", fontsize=24)
    ax_bot.set_xscale("log")
    ax_bot.set_xticks(all_nlists)
    ax_bot.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_bot.get_xaxis().set_tick_params(which="minor", size=0)
    for ax in (ax_top, ax_bot):
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.subplots_adjust(left=0.24)
    fig.savefig(f"{OUTDIR}/ann_sweep_recall_nlist_v3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/ann_sweep_recall_nlist_v3.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  ✓ recall_nlist")

    # (d) Normalized Time vs nlist
    fig, ax = plt.subplots(figsize=figsize)
    for name, d in all_data.items():
        s = d["style"]
        params = d["nlist"][0]
        _, _, _, _, nm, ns = d["nlist"]
        nm_arr = np.array(nm)
        ns_arr = np.array(ns)
        ax.plot(params, nm, color=s["color"], marker=s["marker"],
                markersize=12, linewidth=2, linestyle=s["linestyle"],
                label=s["label"], zorder=3,
                markerfacecolor='none', markeredgewidth=2)
        ax.fill_between(params, nm_arr - ns_arr, nm_arr + ns_arr,
                        color=s["color"], alpha=0.18, zorder=2)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel(r"$n_{\mathrm{list}}$")
    ax.set_ylabel("Search Time Ratio")
    ax.yaxis.set_label_coords(-0.19, 0.42)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_nlists)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which="minor", size=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.subplots_adjust(left=0.22)
    fig.savefig(f"{OUTDIR}/ann_sweep_time_nlist_v3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/ann_sweep_time_nlist_v3.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  ✓ time_nlist")

    # (e) Standalone legend
    fig_leg = plt.figure(figsize=(14, 1.6))
    handles = []
    for name, d in all_data.items():
        s = d["style"]
        h, = fig_leg.gca().plot([], [], color=s["color"], marker=s["marker"],
                                markersize=16, linewidth=3.5, linestyle=s["linestyle"],
                                label=s["label"],
                                markerfacecolor='none', markeredgewidth=2)
        handles.append(h)
    fig_leg.gca().set_visible(False)
    # Row 1: first 3 items, Row 2: last 2 items (centered)
    leg_kw = dict(frameon=False, fontsize=22, columnspacing=1.5,
                  handletextpad=0.5, handlelength=2.5)
    fig_leg.legend(handles=handles[:3], loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 0.88), **leg_kw)
    fig_leg.legend(handles=handles[3:], loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 0.32), **leg_kw)
    fig_leg.savefig(f"{OUTDIR}/ann_sweep_legend_v3.pdf", bbox_inches="tight", dpi=300)
    fig_leg.savefig(f"{OUTDIR}/ann_sweep_legend_v3.png", bbox_inches="tight", dpi=300)
    plt.close(fig_leg)
    print("  ✓ legend")

    print(f"\n✓ All 5 files saved to {OUTDIR}/ (v3)")


if __name__ == "__main__":
    main()
