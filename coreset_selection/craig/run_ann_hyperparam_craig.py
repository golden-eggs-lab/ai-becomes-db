#!/usr/bin/env python
"""
CRAIG ANN Hyperparameter Study – End-to-End Pipeline

Runs the ACTUAL greedy facility-location pipeline (get_orders_and_weights)
with varying nlist/nprobe.  Recall = subset overlap with exact baseline.
Time = ordering_time (greedy loop with FAISS search inside).

Multi-run support for mean/std.

Usage:
    conda run -n craig python -u run_ann_hyperparam_craig.py --runs 3
"""

import os, sys, json, time, argparse
import numpy as np
from datetime import datetime

# ── hyperparameters ──────────────────────────────────────────────────
ANCHOR_NLIST  = 100
ANCHOR_NPROBE = 10

NPROBE_VALUES = [1, 5, 10, 25, 50, 100]
NLIST_VALUES  = [25, 50, 100, 200, 400, 1000]

ANN_K = 20
SUBSET_SIZE = 0.4   # select 40% of data

# ── helpers ──────────────────────────────────────────────────────────

def prepare_data():
    """Load MNIST, build model, compute gradients (once).
    
    Gradients = preds - Y_train (softmax output gradient), matching the
    original CRAIG pipeline's compute_gradients().
    """
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.regularizers import l2

    print("Loading MNIST data...")
    (X_train, Y_train_single), _ = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255
    Y_train = to_categorical(Y_train_single, 10)
    Y_train_nocat = Y_train_single

    print("Building model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,), kernel_regularizer=l2(0.0001)),
        Activation('relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

    print("Computing gradients...")
    preds = model.predict(X_train, verbose=0)
    gradients = (preds - Y_train).astype(np.float32)  # shape [N, 10]

    print(f"Gradients computed: {gradients.shape}")
    return gradients, Y_train_nocat


def run_exact(gradients, Y_train_nocat, B):
    """Run exact baseline through end-to-end pipeline."""
    import util_faiss as util
    subset, _, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
        B, gradients, 'euclidean', smtk=0, no=0, y=Y_train_nocat,
        stoch_greedy=0, equal_num=True,
        use_ann=False, ann_k=ANN_K, ann_backend='faiss',
        force_backend=False
    )
    return subset, ordering_time, similarity_time


def run_ann(gradients, Y_train_nocat, B, nlist, nprobe):
    """Run ANN through end-to-end pipeline with custom nlist/nprobe."""
    import util_faiss as util
    import lazy_greedy_faiss
    import faiss

    # Monkey-patch to use custom nlist/nprobe
    original_init = lazy_greedy_v2.FacilityLocationANN._init_faiss_engine

    def custom_init(self):
        gradients_float32 = self.gradients.astype(np.float32)
        if self.use_ann:
            quantizer = faiss.IndexFlatL2(self.d)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.d, nlist, faiss.METRIC_L2)
            self.faiss_index.train(gradients_float32)
            self.faiss_index.nprobe = nprobe
        else:
            self.faiss_index = faiss.IndexFlatL2(self.d)
        self.faiss_index.add(gradients_float32)

    lazy_greedy_v2.FacilityLocationANN._init_faiss_engine = custom_init
    try:
        subset, _, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
            B, gradients, 'euclidean', smtk=0, no=0, y=Y_train_nocat,
            stoch_greedy=0, equal_num=True,
            use_ann=True, ann_k=ANN_K, ann_backend='faiss',
            force_backend=True
        )
        return subset, ordering_time, similarity_time
    finally:
        lazy_greedy_v2.FacilityLocationANN._init_faiss_engine = original_init


def recall_subset(exact_subset, ann_subset):
    """Recall = |ANN ∩ Exact| / |Exact|."""
    return len(set(exact_subset) & set(ann_subset)) / len(exact_subset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"results/ann_hyperparam_craig_{timestamp}.json"
    os.makedirs("results", exist_ok=True)

    # Prepare data once
    gradients, Y_train_nocat = prepare_data()
    B = int(SUBSET_SIZE * len(gradients))
    print(f"\nCRAIG ANN Hyperparameter Study (End-to-End)")
    print(f"  Data: {gradients.shape[0]}×{gradients.shape[1]}")
    print(f"  Budget: {B}, K={ANN_K}, runs={args.runs}")
    print(f"  nprobe sweep: {NPROBE_VALUES} (fixed nlist={ANCHOR_NLIST})")
    print(f"  nlist sweep:  {NLIST_VALUES} (fixed nprobe={ANCHOR_NPROBE})")

    # Collect per-run data
    all_runs = []
    for run_idx in range(args.runs):
        print(f"\n{'='*60}\nRun {run_idx+1}\n{'='*60}")

        # Exact baseline
        exact_sub, exact_ord_t, exact_sim_t = run_exact(gradients, Y_train_nocat, B)
        print(f"  Exact: ordering={exact_ord_t:.2f}s, similarity={exact_sim_t:.2f}s")

        run_data = {
            "exact_ordering_time": exact_ord_t,
            "exact_similarity_time": exact_sim_t,
            "nprobe": [],
            "nlist": [],
        }

        # nprobe sweep
        for nprobe in NPROBE_VALUES:
            ann_sub, ann_ord_t, ann_sim_t = run_ann(gradients, Y_train_nocat, B, ANCHOR_NLIST, nprobe)
            rec = recall_subset(exact_sub, ann_sub)
            run_data["nprobe"].append({
                "nprobe": nprobe, "recall": rec,
                "ordering_time": ann_ord_t, "similarity_time": ann_sim_t,
            })
            print(f"  nprobe={nprobe:4d}  recall={rec:.4f}  ord_t={ann_ord_t:.2f}s")

        # nlist sweep
        for nlist in NLIST_VALUES:
            ann_sub, ann_ord_t, ann_sim_t = run_ann(gradients, Y_train_nocat, B, nlist, ANCHOR_NPROBE)
            rec = recall_subset(exact_sub, ann_sub)
            run_data["nlist"].append({
                "nlist": nlist, "recall": rec,
                "ordering_time": ann_ord_t, "similarity_time": ann_sim_t,
            })
            print(f"  nlist={nlist:4d}   recall={rec:.4f}  ord_t={ann_ord_t:.2f}s")

        all_runs.append(run_data)

    # Aggregate
    def aggregate(sweep_key, param_key):
        results = []
        param_vals = [r[param_key] for r in all_runs[0][sweep_key]]
        for i, pval in enumerate(param_vals):
            recalls = [run[sweep_key][i]["recall"] for run in all_runs]
            ord_times = [run[sweep_key][i]["ordering_time"] for run in all_runs]
            sim_times = [run[sweep_key][i]["similarity_time"] for run in all_runs]
            results.append({
                param_key: pval,
                "recall_mean": float(np.mean(recalls)),
                "recall_std": float(np.std(recalls)),
                "search_time_mean": float(np.mean(ord_times)),
                "search_time_std": float(np.std(ord_times)),
                "similarity_time_mean": float(np.mean(sim_times)),
                "similarity_time_std": float(np.std(sim_times)),
                "all_recalls": recalls,
            })
        return results

    exact_ord_times = [r["exact_ordering_time"] for r in all_runs]
    exact_sim_times = [r["exact_similarity_time"] for r in all_runs]

    output = {
        "experiment": "craig_ann_hyperparam_e2e",
        "dataset": "MNIST (60000×10 gradients)",
        "budget": B,
        "ann_k": ANN_K,
        "runs": args.runs,
        "timestamp": timestamp,
        "exact_search_time_mean": float(np.mean(exact_ord_times)),
        "exact_search_time_std": float(np.std(exact_ord_times)),
        "exact_similarity_time_mean": float(np.mean(exact_sim_times)),
        "exact_similarity_time_std": float(np.std(exact_sim_times)),
        "nprobe_sweep": aggregate("nprobe", "nprobe"),
        "nlist_sweep": aggregate("nlist", "nlist"),
    }

    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {out_file}")


if __name__ == "__main__":
    main()
