#!/usr/bin/env python
"""
Dataset-size scaling results – 4 separate panels:
  (a) Normalized Time    = opt_time / baseline_time   (with error shading)
  (b) Normalized Perf    = opt_metric / baseline_metric
  (c) Recall
  (d) Standalone legend

6 algorithms, x-axis = ratio (0.3, 0.5, 0.7, 1.0).

Style follows the ANN sweep separate panels (fill_between shading).
"""

import json, csv, os, sys
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
RATIOS = [0.3, 0.5, 0.7, 1.0]

# ── task visual config ───────────────────────────────────────────────
STYLE = {
    "CAL":          {"color": "#ff7f0e", "marker": "s",  "linestyle": "--",              "label": "CAL"},
    "DEFT-UCS":     {"color": "#2ca02c", "marker": "^",  "linestyle": "-.",              "label": "DEFT-UCS"},
    "SemDeDup":     {"color": "#1f77b4", "marker": "o",  "linestyle": "-",               "label": "SemDeDup"},
    "KNN-OOD":      {"color": "#d62728", "marker": "D",  "linestyle": ":",               "label": "KNN-OOD"},
    "KNN Prompting":{"color": "#9467bd", "marker": "v",  "linestyle": (0, (3, 1, 1, 1)), "label": "KNN Prompting"},
    "SCIP":         {"color": "#8c564b", "marker": "P",  "linestyle": (0, (5, 2)),       "label": "SCIP"},
    "CRAIG":        {"color": "#17becf", "marker": "<",  "linestyle": (0, (1, 1)),       "label": "CRAIG"},
}


# ====================================================================
# Data loading
# ====================================================================

def load_cal():
    """CAL: knn_total_time from separate experiment dirs."""
    base = "/home/yichen/old/contrastive-active-learning/experiments"
    # Mapping: ratio -> (baseline_knn_total, [opt_knn_totals], baseline_acc, opt_acc)
    data = {}
    import glob

    # ratio=1.0: separate experiment dirs
    runs_10 = {
        "baseline": f"{base}/al_imdb_bert_cal_imdb_full_baseline_run1/42_imdb_full_baseline_run1_cls/results_of_iteration.json",
        "opt1": f"{base}/al_imdb_bert_cal_imdb_full_optimized_run1/42_imdb_full_optimized_run1_cls/results_of_iteration.json",
        "opt2": f"{base}/al_imdb_bert_cal_imdb_full_optimized_run2/42_imdb_full_optimized_run2_cls/results_of_iteration.json",
    }

    def extract_cal_iter7(path):
        with open(path) as f:
            d = json.load(f)
        it7 = d["7"]
        acc = it7.get("test_acc") or it7.get("test_results", {}).get("acc", None)
        return it7["knn_total_time"], acc

    # ratio=1.0
    bt, ba = extract_cal_iter7(runs_10["baseline"])
    ot1, oa = extract_cal_iter7(runs_10["opt1"])
    ot2, _ = extract_cal_iter7(runs_10["opt2"])
    data[1.0] = {"b_time": bt, "o_times": [ot1, ot2], "b_perf": ba, "o_perf": oa, "recall": 1.0}

    # ratio=0.7, 0.5, 0.3: from scaling experiment dirs
    for ratio, cap in [(0.7, 15750), (0.5, 11250), (0.3, 6750)]:
        bp = f"{base}/al_imdb_bert_cal_scaling_cap{cap}_baseline_run1/42_scaling_cap{cap}_baseline_run1_cls/results_of_iteration.json"
        o1p = f"{base}/al_imdb_bert_cal_scaling_cap{cap}_optimized_run1/42_scaling_cap{cap}_optimized_run1_cls/results_of_iteration.json"
        o2p = f"{base}/al_imdb_bert_cal_scaling_cap{cap}_optimized_run2/42_scaling_cap{cap}_optimized_run2_cls/results_of_iteration.json"
        bt, ba = extract_cal_iter7(bp)
        ot1, oa = extract_cal_iter7(o1p)
        ot2, _ = extract_cal_iter7(o2p)
        data[ratio] = {"b_time": bt, "o_times": [ot1, ot2], "b_perf": ba, "o_perf": oa, "recall": 1.0}

    return data


def load_deft_ucs():
    """DEFT-UCS: selection_time, SARI (from single SARI JSON which has timing too)."""
    with open("/home/yichen/old/coreset/experiments/artifacts/wikilarge_scaling_sari/scaling_results.json") as f:
        raw = json.load(f)

    data = {}
    for r_str, v in raw.items():
        ratio = float(r_str)
        data[ratio] = {
            "b_time": v["baseline_time"],
            "o_times": v["optimized_times"],
            "b_perf": v.get("baseline_sari", None),
            "o_perf": v.get("optimized_sari", None),
            "recall": v.get("recall", None),
        }

    # ratio=1.0 from wikilarge_comparison: 3 runs each
    # baseline_selection_timing_run{1,2,3}.json → total_time
    # optimized_selection_timing_run{1,2,3}.json → total_time
    # eval_results.json → SARI
    comp_dir = "/home/yichen/old/coreset/experiments/artifacts/wikilarge_comparison"
    b_times_10 = []
    for i in range(1, 4):
        p = f"{comp_dir}/baseline_selection_timing_run{i}.json"
        if os.path.exists(p):
            with open(p) as f:
                b_times_10.append(json.load(f)["total_time"])
    o_times_10 = []
    for i in range(1, 4):
        p = f"{comp_dir}/optimized_selection_timing_run{i}.json"
        if os.path.exists(p):
            with open(p) as f:
                o_times_10.append(json.load(f)["total_time"])
    eval_p = f"{comp_dir}/eval_results.json"
    b_sari, o_sari = None, None
    if os.path.exists(eval_p):
        with open(eval_p) as f:
            ev = json.load(f)
        b_sari = ev.get("baseline", {}).get("sari", None)
        o_sari = ev.get("optimized", {}).get("sari", None)
    # recall: baseline selected 48265, optimized selected 47381 → recall = 47381/48265
    recall_10 = 47381 / 48265
    data[1.0] = {
        "b_time": np.mean(b_times_10),
        "o_times": o_times_10,
        "b_perf": b_sari,
        "o_perf": o_sari,
        "recall": recall_10,
    }
    return data


def load_semdedup():
    """SemDeDup: dedup time, acc@0.05."""
    data = {}

    # ratio=0.3, 0.5 from v2
    with open("/home/yichen/old/SemDeDup/output_scaling_v2/scaling_results.json") as f:
        v2 = json.load(f)
    for r_str, v in v2.items():
        ratio = float(r_str)
        data[ratio] = {
            "b_time": v["baseline_time"],
            "o_times": v["optimized_times"],
            "b_perf": None, "o_perf": None,
            "recall": v.get("recall", None),
        }

    # ratio=0.7 from new run
    with open("/home/yichen/old/SemDeDup/output_scaling_v2_ratio07/scaling_results.json") as f:
        v07 = json.load(f)
    v = v07["0.7"]
    data[0.7] = {
        "b_time": v["baseline_time"],
        "o_times": v["optimized_times"],
        "b_perf": None, "o_perf": None,
        "recall": v.get("recall", None),
    }

    # ratio=1.0 from cifar10_new_run{1,2,3}.log:
    # Exact_NoCache (baseline): [86.76, 86.44, 86.33]
    # ANN_Cache (optimized):    [0.37, 0.40, 0.35]
    # Recall = 100% (same indices kept)
    data[1.0] = {
        "b_time": np.mean([86.76, 86.44, 86.33]),
        "o_times": [0.37, 0.40, 0.35],
        "b_perf": None, "o_perf": None,
        "recall": 1.0,
    }

    # Add acc@0.05 data (trained separately)
    acc_data = {
        1.0: (84.25, 84.55),   # from paper table
        0.7: (89.80, 89.87),
        0.5: (87.18, 87.20),
        0.3: (81.16, 79.55),
    }
    for ratio, (ba, oa) in acc_data.items():
        if ratio in data:
            data[ratio]["b_perf"] = ba
            data[ratio]["o_perf"] = oa

    return data


def load_knn_ood():
    """KNN-OOD: total time, places50 AUROC. Recall = 100% (ANN doesn't affect selection)."""
    with open("/home/yichen/old/knn-ood/results_scaling/scaling_results.json") as f:
        raw = json.load(f)

    # Performance: places50 AUROC (baseline, optimized) from scaling logs
    auroc_data = {
        0.7: (81.41, 79.56),
        0.5: (79.43, 79.27),
        0.3: (77.49, 76.36),
    }

    data = {}
    for r_str, v in raw.items():
        ratio = float(r_str)
        ba, oa = auroc_data.get(ratio, (None, None))
        data[ratio] = {
            "b_time": v["baseline_time"],
            "o_times": v["optimized_times"],
            "b_perf": ba,
            "o_perf": oa,
            "recall": 1.0,  # ANN doesn't change final OOD scores
        }

    # ratio=1.0 from knn_ood_benchmark_3runs.log:
    # Baseline (Exact): total_time = [64.07, 63.61, 64.45]
    # Optimized (ANN-IVF): total_time = [35.81, 36.47, 36.33]
    # places50 AUROC: baseline=85.24, optimized=84.81
    data[1.0] = {
        "b_time": np.mean([64.07, 63.61, 64.45]),
        "o_times": [35.81, 36.47, 36.33],
        "b_perf": 85.24,
        "o_perf": 84.81,
        "recall": 1.0,
    }
    return data


def load_knn_prompting():
    """KNN Prompting: infer_time from CSVs, accuracy."""
    data = {}

    # ratio=1.0 from output_agnews
    base = "/home/yichen/old/KNNPrompting/output_agnews"
    b_infer = []
    for i in range(1, 4):
        p = f"{base}/results_baseline_run{i}.csv"
        if os.path.exists(p):
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    b_infer.append(float(row["infer_time"]))
    o_infer = []
    for i in range(1, 4):
        p = f"{base}/results_optimized_run{i}.csv"
        if os.path.exists(p):
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    o_infer.append(float(row["infer_time"]))

    # For ratio=1.0, use mean of baselines as "baseline"
    data[1.0] = {
        "b_time": np.mean(b_infer),
        "o_times": o_infer,
        "b_perf": 0.890625,
        "o_perf": 0.890625,
        "recall": 1.0,  # ANN doesn't change prompting results
    }

    # ratio=0.7, 0.5, 0.3 from output_agnews_scaling
    base_s = "/home/yichen/old/KNNPrompting/output_agnews_scaling"
    shot_map = {0.7: 717, 0.5: 512, 0.3: 307}
    for ratio, shot in shot_map.items():
        bp = f"{base_s}/results_baseline_shot{shot}.csv"
        with open(bp) as f:
            reader = csv.DictReader(f)
            for row in reader:
                b_t = float(row["infer_time"])
                b_acc = float(row["acc"])

        o_ts = []
        o_acc = None
        for i in range(1, 3):
            op = f"{base_s}/results_optimized_shot{shot}_run{i}.csv"
            if os.path.exists(op):
                with open(op) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        o_ts.append(float(row["infer_time"]))
                        o_acc = float(row["acc"])

        data[ratio] = {
            "b_time": b_t,
            "o_times": o_ts,
            "b_perf": b_acc,
            "o_perf": o_acc,
            "recall": 1.0,  # ANN doesn't change prompting results
        }

    return data


def load_scip():
    """SCIP: algorithm time (Python v3), MRR."""
    with open("/home/yichen/old/codepruning/scip_scaling_results_v3.json") as f:
        timing = json.load(f)
    with open("/home/yichen/old/codepruning/scip_scaling_mrr.json") as f:
        mrr = json.load(f)

    data = {}
    for r_str in timing:
        ratio = float(r_str)
        t = timing[r_str]
        m = mrr.get(r_str, {})
        data[ratio] = {
            "b_time": t["baseline_time"],
            "o_times": t["optimized_times"],
            "b_perf": m.get("baseline_mrr", None),
            "o_perf": m.get("optimized_mrr", None),
            "recall": t.get("recall", None),
        }
    return data


# ====================================================================
# Main
# ====================================================================

def main():
    print("Loading data...")
    all_data = {
        "CAL": load_cal(),
        "DEFT-UCS": load_deft_ucs(),
        "SemDeDup": load_semdedup(),
        "KNN-OOD": load_knn_ood(),
        "KNN Prompting": load_knn_prompting(),
        "SCIP": load_scip(),
    }

    # Debug print
    for name, dmap in all_data.items():
        print(f"\n  {name}:")
        for ratio in sorted(dmap):
            d = dmap[ratio]
            o_mean = np.mean(d["o_times"])
            speedup = d["b_time"] / o_mean if o_mean > 0 else 0
            print(f"    ratio={ratio}: B_t={d['b_time']:.2f}, O_t={o_mean:.2f} "
                  f"(n={len(d['o_times'])}), speedup={speedup:.2f}x, "
                  f"B_perf={d['b_perf']}, O_perf={d['o_perf']}, recall={d.get('recall')}")

    figsize = (5.5, 3.8)

    # ── (a) Normalized Time ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    for name in STYLE:
        if name not in all_data:
            continue
        s = STYLE[name]
        dmap = all_data[name]
        ratios_avail = sorted(r for r in RATIOS if r in dmap)
        norm_mean = []
        norm_lo = []
        norm_hi = []
        for r in ratios_avail:
            d = dmap[r]
            bt = d["b_time"]
            ots = np.array(d["o_times"])
            nm = np.mean(ots) / bt
            if len(ots) > 1:
                ns = np.std(ots, ddof=1) / bt
            else:
                ns = 0
            norm_mean.append(nm)
            norm_lo.append(nm - ns)
            norm_hi.append(nm + ns)

        nm_arr = np.array(norm_mean)
        ax.plot(ratios_avail, norm_mean, color=s["color"], marker=s["marker"],
                markersize=12, linewidth=2, linestyle=s["linestyle"],
                label=s["label"], zorder=3,
                markerfacecolor='none', markeredgewidth=2)
        ax.fill_between(ratios_avail, norm_lo, norm_hi,
                        color=s["color"], alpha=0.18, zorder=2)

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel("Dataset Size Ratio")
    ax.set_ylabel("Time Ratio")
    ax.yaxis.set_label_coords(-0.12, 0.42)
    ax.set_xticks(RATIOS)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/scaling_time.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/scaling_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("\n  ✓ scaling_time")

    # ── (b) Normalized Performance ───────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    for name in STYLE:
        if name not in all_data:
            continue
        s = STYLE[name]
        dmap = all_data[name]
        ratios_avail = sorted(r for r in RATIOS if r in dmap and
                              dmap[r]["b_perf"] is not None and dmap[r]["o_perf"] is not None)
        if not ratios_avail:
            continue
        norm_perf = []
        for r in ratios_avail:
            d = dmap[r]
            np_ = d["o_perf"] / d["b_perf"] if d["b_perf"] != 0 else 1.0
            norm_perf.append(np_)

        ax.plot(ratios_avail, norm_perf, color=s["color"], marker=s["marker"],
                markersize=12, linewidth=2, linestyle=s["linestyle"],
                label=s["label"], zorder=3,
                markerfacecolor='none', markeredgewidth=2)

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel("Dataset Size Ratio")
    ax.set_ylabel("Performance Ratio")
    ax.yaxis.set_label_coords(-0.12, 0.42)
    ax.set_xticks(RATIOS)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/scaling_performance.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/scaling_performance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  ✓ scaling_performance")

    # ── (c) Recall ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    for name in STYLE:
        if name not in all_data:
            continue
        s = STYLE[name]
        dmap = all_data[name]
        ratios_avail = sorted(r for r in RATIOS if r in dmap and
                              dmap[r].get("recall") is not None)
        if not ratios_avail:
            continue
        recalls = [dmap[r]["recall"] for r in ratios_avail]

        ax.plot(ratios_avail, recalls, color=s["color"], marker=s["marker"],
                markersize=12, linewidth=2.5, linestyle=s["linestyle"],
                label=s["label"], zorder=3,
                markerfacecolor='none', markeredgewidth=2)

    ax.set_xlabel("Dataset Size Ratio")
    ax.set_ylabel("Selection Recall")
    ax.set_xticks(RATIOS)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0.55, top=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/scaling_recall.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUTDIR}/scaling_recall.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  ✓ scaling_recall")

    # ── (d) Standalone legend (two centered rows) ──────────────────
    fig_leg = plt.figure(figsize=(14, 1.3))
    handles = []
    for name in STYLE:
        s = STYLE[name]
        h, = fig_leg.gca().plot([], [], color=s["color"], marker=s["marker"],
                                markersize=16, linewidth=3.5, linestyle=s["linestyle"],
                                label=s["label"],
                                markerfacecolor='none', markeredgewidth=2)
        handles.append(h)
    fig_leg.gca().set_visible(False)
    leg_kw = dict(frameon=False, fontsize=22, columnspacing=1.5,
                  handletextpad=0.5, handlelength=2.5)
    fig_leg.legend(handles=handles[:4], loc='upper center', ncol=4,
                   bbox_to_anchor=(0.5, 0.82), **leg_kw)
    fig_leg.legend(handles=handles[4:], loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 0.38), **leg_kw)
    fig_leg.savefig(f"{OUTDIR}/scaling_legend.pdf", bbox_inches="tight", dpi=300)
    fig_leg.savefig(f"{OUTDIR}/scaling_legend.png", bbox_inches="tight", dpi=300)
    plt.close(fig_leg)
    print("  ✓ scaling_legend")

    print(f"\n✓ All 4 files saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
