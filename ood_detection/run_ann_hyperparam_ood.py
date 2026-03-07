#!/usr/bin/env python
"""
KNN-OOD ANN Hyperparameter Study

Measures recall and search time for FAISS IVF vs Flat on KNN search
over ImageNet-1k ResNet features (2048D, 1.28M samples).

Uses faiss-gpu to align with the main experiment.

Recall = fraction of exact K-th NN distances also found by ANN.

Usage:
    conda run -n ood python run_ann_hyperparam_ood.py --runs 3
"""

import os, sys, json, time, argparse
import numpy as np
import faiss
from datetime import datetime

# ── hyperparameters ──────────────────────────────────────────────────
ANCHOR_NLIST  = 1000  # matches end-to-end run_imagenet.py --nlist 1000
ANCHOR_NPROBE = 10

NPROBE_VALUES = [1, 5, 10, 25, 50]  # 100 causes GPU OOM (nprobe=100 on 1000 cells ≈ brute force)
NLIST_VALUES  = [50, 100, 200, 400, 1000]  # nlist=25 with nprobe=10 causes OOM

KNN_K = 1000   # matches main experiment

# OOD experiment defaults (for marking on plot)
DEFAULT_NLIST  = 1000
DEFAULT_NPROBE = 5

# ── data loading ─────────────────────────────────────────────────────
def load_imagenet_features(args):
    """Load precomputed ImageNet-1k features."""
    id_train_size = 1281167
    
    cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
    feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r',
                         shape=(id_train_size, 2048))
    
    # Normalize
    norm_cache = f"{cache_dir}/feat_norm.mmap"
    if os.path.exists(norm_cache):
        feat_norm = np.memmap(norm_cache, dtype=float, mode='r',
                              shape=(id_train_size, 2048))
    else:
        print("Computing normalized features...")
        feat_norm = np.memmap(norm_cache, dtype=float, mode='w+',
                              shape=(id_train_size, 2048))
        norms = np.linalg.norm(feat_log, axis=-1, keepdims=True) + 1e-10
        feat_norm[:] = feat_log / norms
    
    ftrain = np.ascontiguousarray(feat_norm.astype(np.float32))
    
    # Also load test features for recall on test queries
    id_val_size = 50000
    cache_dir_val = f"cache/{args.in_dataset}_val_{args.name}_in"
    feat_val = np.memmap(f"{cache_dir_val}/feat.mmap", dtype=float, mode='r',
                         shape=(id_val_size, 2048))
    norm_cache_val = f"{cache_dir_val}/feat_norm.mmap"
    if os.path.exists(norm_cache_val):
        feat_val_norm = np.memmap(norm_cache_val, dtype=float, mode='r',
                                  shape=(id_val_size, 2048))
    else:
        feat_val_norm = np.memmap(norm_cache_val, dtype=float, mode='w+',
                                  shape=(id_val_size, 2048))
        norms = np.linalg.norm(feat_val, axis=-1, keepdims=True) + 1e-10
        feat_val_norm[:] = feat_val / norms
    
    ftest = np.ascontiguousarray(feat_val_norm.astype(np.float32))
    
    print(f"Loaded ImageNet features: train={ftrain.shape}, test={ftest.shape}")
    return ftrain, ftest

def maybe_gpu(index, use_gpu, gpu_id=0):
    """Move index to a SINGLE GPU (matching end-to-end run_imagenet.py)."""
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            return index, True, res
        except Exception as e:
            print(f"  GPU failed ({e}), using CPU")
    return index, False, None


def batched_search(index, queries, k, batch_size=10000):
    """Search in batches to avoid GPU OOM for large k.
    
    End-to-end run_imagenet.py does index.search(ftest, k) in one shot,
    but under conda run the GPU memory layout differs and causes OOM.
    batch_size=10000 keeps overhead minimal while avoiding OOM.
    """
    n = queries.shape[0]
    if n <= batch_size:
        return index.search(queries, k)
    D = np.empty((n, k), dtype=np.float32)
    I = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        D[start:end], I[start:end] = index.search(queries[start:end], k)
    return D, I


# ── search functions ─────────────────────────────────────────────────
def search_exact(ftrain, ftest, k, use_gpu=True):
    """Exact KNN using Flat index. Returns (D_exact, I_exact, search_time)."""
    d = ftrain.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(ftrain)
    index, on_gpu, _res = maybe_gpu(index, use_gpu)
    
    t0 = time.time()
    D, I = batched_search(index, ftest, k)
    search_time = time.time() - t0
    
    print(f"  Exact search ({'GPU' if on_gpu else 'CPU'}): {search_time:.2f}s")
    return D, I, search_time


def search_ann(ftrain, ftest, k, nlist, nprobe, use_gpu=True):
    """ANN KNN using IVF index. Returns (D_ann, I_ann, search_time)."""
    n, d = ftrain.shape
    actual_nlist = max(1, min(nlist, n // 39))  # FAISS requires n >= 39*nlist for training
    
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, actual_nlist, faiss.METRIC_L2)
    index.train(ftrain)
    index.add(ftrain)
    index.nprobe = min(nprobe, actual_nlist)
    
    index, on_gpu, _res = maybe_gpu(index, use_gpu)
    
    t0 = time.time()
    D, I = batched_search(index, ftest, k)
    search_time = time.time() - t0
    
    return D, I, search_time


OOD_PERCENTILE = 95  # top 5% flagged as OOD

def compute_recall(exact_D, ann_D, k, percentile=OOD_PERCENTILE):
    """Task-level recall: overlap of OOD-detected samples.
    
    Uses k-th NN distance as OOD score.
    Flags top (100-percentile)% of test points as OOD.
    Recall = |exact_ood ∩ ann_ood| / |exact_ood|
    Consistent with SemDeDup's overlap-based task recall.
    """
    exact_kth = exact_D[:, k-1]
    ann_kth   = ann_D[:, k-1]
    
    threshold = np.percentile(exact_kth, percentile)
    exact_ood = set(np.where(exact_kth >= threshold)[0])
    ann_ood   = set(np.where(ann_kth >= threshold)[0])
    
    if len(exact_ood) == 0:
        return 1.0
    recall = len(exact_ood & ann_ood) / len(exact_ood)
    return float(recall)


# ── main ─────────────────────────────────────────────────────────────
def run_single(ftrain, ftest, k, nprobe_values, nlist_values, run_id, use_gpu):
    print(f"\n{'='*60}\nRun {run_id}\n{'='*60}")
    
    D_exact, I_exact, exact_time = search_exact(ftrain, ftest, k, use_gpu)
    
    nprobe_results = []
    for nprobe in nprobe_values:
        D_ann, I_ann, t = search_ann(ftrain, ftest, k, ANCHOR_NLIST, nprobe, use_gpu)
        recall = compute_recall(D_exact, D_ann, k)
        nprobe_results.append({"nprobe": nprobe, "recall": recall, "search_time": t})
        print(f"  nprobe={nprobe:4d}  recall={recall:.6f}  time={t:.2f}s")
    
    nlist_results = []
    for nlist in nlist_values:
        D_ann, I_ann, t = search_ann(ftrain, ftest, k, nlist, ANCHOR_NPROBE, use_gpu)
        recall = compute_recall(D_exact, D_ann, k)
        nlist_results.append({"nlist": nlist, "recall": recall, "search_time": t})
        print(f"  nlist={nlist:4d}   recall={recall:.6f}  time={t:.2f}s")
    
    return nprobe_results, nlist_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--name", type=str, default="resnet50-supcon")
    parser.add_argument("--in-dataset", type=str, default="imagenet")
    parser.add_argument("--no-gpu", action="store_true", help="Disable faiss-gpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results/ann_hyperparam_ood_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file) or "results", exist_ok=True)
    
    ftrain, ftest = load_imagenet_features(args)
    n, d = ftrain.shape
    
    print(f"\nKNN-OOD ANN Hyperparameter Study")
    print(f"  Train: {n}×{d}, Test: {ftest.shape[0]}×{d}")
    print(f"  K={KNN_K}, runs={args.runs}, GPU={'yes' if use_gpu else 'no'}")
    print(f"  Default: nlist={DEFAULT_NLIST}, nprobe={DEFAULT_NPROBE}")
    print(f"  nprobe sweep: {NPROBE_VALUES} (fixed nlist={ANCHOR_NLIST})")
    print(f"  nlist sweep:  {NLIST_VALUES} (fixed nprobe={ANCHOR_NPROBE})")
    
    all_nprobe = {p: [] for p in NPROBE_VALUES}
    all_nlist  = {nl: [] for nl in NLIST_VALUES}
    
    for run_id in range(1, args.runs + 1):
        np_res, nl_res = run_single(ftrain, ftest, KNN_K, NPROBE_VALUES, NLIST_VALUES, run_id, use_gpu)
        for r in np_res:
            all_nprobe[r["nprobe"]].append(r)
        for r in nl_res:
            all_nlist[r["nlist"]].append(r)
    
    # Aggregate
    nprobe_agg = []
    for p in NPROBE_VALUES:
        runs = all_nprobe[p]
        nprobe_agg.append({
            "nprobe": p,
            "recall_mean": float(np.mean([r["recall"] for r in runs])),
            "recall_std":  float(np.std([r["recall"] for r in runs])),
            "search_time_mean": float(np.mean([r["search_time"] for r in runs])),
            "search_time_std":  float(np.std([r["search_time"] for r in runs])),
            "all_recalls": [r["recall"] for r in runs],
        })
    
    nlist_agg = []
    for nl in NLIST_VALUES:
        runs = all_nlist[nl]
        nlist_agg.append({
            "nlist": nl,
            "recall_mean": float(np.mean([r["recall"] for r in runs])),
            "recall_std":  float(np.std([r["recall"] for r in runs])),
            "search_time_mean": float(np.mean([r["search_time"] for r in runs])),
            "search_time_std":  float(np.std([r["search_time"] for r in runs])),
            "all_recalls": [r["recall"] for r in runs],
        })
    
    results = {
        "experiment": "ood_ann_hyperparameter_study",
        "task": "KNN-OOD",
        "dataset": "ImageNet-1k",
        "num_train": int(n),
        "num_test": int(ftest.shape[0]),
        "embedding_dim": int(d),
        "knn_k": KNN_K,
        "num_runs": args.runs,
        "use_gpu": use_gpu,
        "timestamp": timestamp,
        "anchor_nlist": ANCHOR_NLIST,
        "anchor_nprobe": ANCHOR_NPROBE,
        "default_nlist": DEFAULT_NLIST,
        "default_nprobe": DEFAULT_NPROBE,
        "nprobe_sweep": nprobe_agg,
        "nlist_sweep": nlist_agg,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
