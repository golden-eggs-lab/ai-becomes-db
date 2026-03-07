#!/usr/bin/env python3
"""
Quick peak memory measurement for SemDeDup.
Runs only the dedup step (no downstream eval) using pre-clustered data.

Measures:
1. Peak memory during dedup processing (tracemalloc)
2. Cache size (the similarity vector M stored per cluster)

Uses the existing clustered data from a previous run.
"""

import os
import sys
import time
import tracemalloc
import numpy as np
import faiss
import pickle
import pandas as pd

# ---------- Config (same as run_cifar10_experiment.py) ----------
CONFIG = {
    'dataset_size': 50000,
    'emb_size': 512,
    'num_clusters': 10,
    'eps_list': [0.05, 0.1, 0.15],
    'embs_memory_loc': '/home/yichen/old/SemDeDup/cifar10_embeddings.npy',
    # Use pre-clustered data from a previous run
    'save_folder': '/home/yichen/old/SemDeDup/output_cifar10_run_new_run1',
}

def init_memmap_embs(path, size, dim):
    return np.memmap(path, dtype='float32', mode='r', shape=(size, dim))


def semdedup_exact_faiss(cluster_embs):
    """Exact FAISS search. Returns (M, build_time, search_time)."""
    n, d = cluster_embs.shape
    t0 = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(cluster_embs)
    build_time = time.time() - t0

    t0 = time.time()
    similarities, indices = index.search(cluster_embs, n)
    search_time = time.time() - t0

    self_mask = (indices == np.arange(n)[:, None])
    similarities_masked = np.where(self_mask, -np.inf, similarities)
    M = np.max(similarities_masked, axis=1)
    return M, build_time, search_time


def semdedup_ann_faiss(cluster_embs, ann_k=100, nprobe=10):
    """ANN FAISS search. Returns (M, build_time, search_time)."""
    n, d = cluster_embs.shape
    if n <= ann_k:
        return semdedup_exact_faiss(cluster_embs)

    t0 = time.time()
    nlist = max(1, min(n // 10, 100))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = min(nprobe, nlist)
    index.train(cluster_embs)
    index.add(cluster_embs)
    build_time = time.time() - t0

    t0 = time.time()
    k = min(ann_k, n)
    similarities, indices = index.search(cluster_embs, k)
    search_time = time.time() - t0

    self_mask = (indices == np.arange(n)[:, None])
    similarities_masked = np.where(self_mask, -np.inf, similarities)
    M = np.max(similarities_masked, axis=1)
    return M, build_time, search_time


def measure_dedup_memory(use_ann=True, use_cache=True, label=""):
    """Run SemDeDup dedup and measure peak memory."""
    sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
    embs = init_memmap_embs(CONFIG['embs_memory_loc'], CONFIG['dataset_size'], CONFIG['emb_size'])

    tracemalloc.start()

    total_build = 0.0
    total_search = 0.0
    total_cache_bytes = 0  # Track cumulative cache size
    total_time = 0.0

    for cluster_id in range(CONFIG['num_clusters']):
        cluster_data = np.load(
            os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_size = cluster_data.shape[0]
        if cluster_size <= 1:
            continue

        cluster_indices = cluster_data[:, 1].astype(int)
        cluster_embs = np.array(embs[cluster_indices], dtype=np.float32)
        faiss.normalize_L2(cluster_embs)

        if use_cache:
            # Compute M once, reuse for all eps
            if use_ann:
                M, bt, st = semdedup_ann_faiss(cluster_embs)
            else:
                M, bt, st = semdedup_exact_faiss(cluster_embs)
            total_build += bt
            total_search += st

            # Cache size: M is float32, shape (cluster_size,)
            total_cache_bytes += M.nbytes

            for eps in CONFIG['eps_list']:
                _ = M > (1 - eps)
        else:
            # No cache: compute for each eps separately
            for eps in CONFIG['eps_list']:
                embs_copy = cluster_embs.copy()
                if use_ann:
                    M, bt, st = semdedup_ann_faiss(embs_copy)
                else:
                    M, bt, st = semdedup_exact_faiss(embs_copy)
                total_build += bt
                total_search += st
                _ = M > (1 - eps)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  ANN: {use_ann}, Cache: {use_cache}")
    print(f"  Total build time:  {total_build:.4f}s")
    print(f"  Total search time: {total_search:.4f}s")
    print(f"  Peak memory:       {peak_mem/1024/1024:.2f} MB")
    print(f"  Current memory:    {current_mem/1024/1024:.2f} MB")
    if use_cache:
        print(f"  Cache size (M vectors): {total_cache_bytes/1024:.2f} KB ({total_cache_bytes/1024/1024:.4f} MB)")
        print(f"    (sum of M arrays across {CONFIG['num_clusters']} clusters)")
    print(f"{'='*60}")

    return {
        'build_time': total_build,
        'search_time': total_search,
        'peak_memory_mb': peak_mem / 1024 / 1024,
        'current_memory_mb': current_mem / 1024 / 1024,
        'cache_bytes': total_cache_bytes if use_cache else 0,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SemDeDup Peak Memory Measurement")
    print(f"Dataset: CIFAR-10 ({CONFIG['dataset_size']:,} samples, {CONFIG['emb_size']}D)")
    print(f"Clusters: {CONFIG['num_clusters']}")
    print("=" * 60)

    # Check that pre-clustered data exists
    sorted_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
    if not os.path.exists(sorted_path):
        print(f"ERROR: Pre-clustered data not found at {sorted_path}")
        print("Please run run_cifar10_experiment.py first.")
        sys.exit(1)

    configs = [
        ("SemDeDup_Exact_NoCache", False, False),
        ("SemDeDup_Exact_Cache",   False, True),
        ("SemDeDup_ANN_NoCache",   True,  False),
        ("SemDeDup_ANN_Cache",     True,  True),
    ]

    results = {}
    for label, use_ann, use_cache in configs:
        r = measure_dedup_memory(use_ann=use_ann, use_cache=use_cache, label=label)
        results[label] = r

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'Build(s)':>10} {'Search(s)':>10} {'Peak Mem(MB)':>14} {'Cache(KB)':>12}")
    print("-" * 76)
    for label, r in results.items():
        cache_kb = r['cache_bytes'] / 1024 if r['cache_bytes'] > 0 else 0
        print(f"{label:<30} {r['build_time']:>10.4f} {r['search_time']:>10.4f} {r['peak_memory_mb']:>14.2f} {cache_kb:>12.2f}")
