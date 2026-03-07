#!/usr/bin/env python3
"""
SCIP: Dataset Size Scaling Experiment (AWS, CPU-only)
Uses FAISS KMeans (same as run_scip_only.py) to align with e2e measurements.
Vary embeddings ratio: 1.0, 0.7, 0.5, 0.3
Measure: SCIP time (baseline vs optimized), recall
Baseline × 1 run, Optimized × 2 runs per ratio

Run on AWS: python run_scip_scaling.py --embeddings /path/to/embeddings.npy
"""

import os
import json
import time
import argparse
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TimingResult:
    step_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    def add_step(self, name: str, elapsed: float):
        self.step_times[name] = elapsed
        self.total_time += elapsed


def scip_baseline(embeddings, n_clusters=100, p=0.2, alpha=0.8):
    """SCIP baseline: FAISS KMeans + exact centroid search + numpy distances.
    Matches run_scip_only.py baseline implementation."""
    timing = TimingResult()
    N, D = embeddings.shape
    K = n_clusters

    # 1. KMeans clustering (sklearn MiniBatchKMeans, same as run_scip_only.py)
    t0 = time.perf_counter()
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)

    # 2. Centroid search (FAISS exact)
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(D)
    index.add(centroids.astype(np.float32))
    sims, labels = index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)

    # 3. Per-point distances
    t0 = time.perf_counter()
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)

    # 4. Sort by cluster size → prune smallest clusters
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    idx_by_size = np.argsort(size_per_point)
    total_prune = int(round(p * N))
    size_quota = int(round(alpha * total_prune))
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("SortBySize", time.perf_counter() - t0)

    # 5. Sort by distance → prune farthest from centroid
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    dist_remaining = np.zeros(len(remaining), dtype=np.float32)
    for idx, i in enumerate(remaining):
        dist_remaining[idx] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    order = np.argsort(-dist_remaining)
    dist_quota = total_prune - size_quota
    prune_by_dist = remaining[order[:dist_quota]]
    timing.add_step("SortByDist", time.perf_counter() - t0)

    # 6. Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)

    return keep_indices, timing


def scip_optimized(embeddings, n_clusters=100, p=0.2, alpha=0.8):
    """SCIP optimized: FAISS KMeans + IVF ANN search + TopK partition.
    Matches run_scip_only.py optimized implementation."""
    timing = TimingResult()
    N, D = embeddings.shape
    K = n_clusters

    # 1. KMeans (same MiniBatchKMeans as baseline for fair comparison)
    t0 = time.perf_counter()
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)

    # 2. Centroid search (IVF ANN)
    t0 = time.perf_counter()

    nlist = min(int(np.sqrt(K)), max(1, K // 4))
    quantizer = faiss.IndexFlatIP(D)
    ivf_index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf_index.train(centroids_norm.astype(np.float32))
    ivf_index.add(centroids_norm.astype(np.float32))
    ivf_index.nprobe = max(1, nlist // 2)

    sims, labels = ivf_index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)

    # 3. Vectorized distances
    t0 = time.perf_counter()
    distances = 1.0 - np.sum(embeddings * centroids_norm[labels], axis=1).astype(np.float32)
    timing.add_step("Distances", time.perf_counter() - t0)

    # 4. TopK by cluster size (argpartition instead of argsort)
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    total_prune = int(round(p * N))
    size_quota = int(round(alpha * total_prune))
    if size_quota < N:
        part_idx = np.argpartition(size_per_point, size_quota)[:size_quota]
    else:
        part_idx = np.arange(N)
    prune_by_size = part_idx
    timing.add_step("TopKBySize", time.perf_counter() - t0)

    # 5. TopK by distance
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    dist_remaining = 1.0 - np.sum(embeddings[remaining] * centroids_norm[labels[remaining]], axis=1).astype(np.float32)
    dist_quota = total_prune - size_quota
    if dist_quota > 0 and dist_quota < len(remaining):
        part_idx = np.argpartition(-dist_remaining, dist_quota)[:dist_quota]
        prune_by_dist = remaining[part_idx]
    elif dist_quota >= len(remaining):
        prune_by_dist = remaining
    else:
        prune_by_dist = np.array([], dtype=np.int64)
    timing.add_step("TopKByDist", time.perf_counter() - t0)

    # 6. Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)

    return keep_indices, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings")
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--ratios", nargs='+', type=float, default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument("--output", type=str, default="scip_scaling_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    full_embeddings = np.load(args.embeddings).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
    full_embeddings = full_embeddings / (norms + 1e-12)
    print(f"Full shape: {full_embeddings.shape}")

    np.random.seed(args.seed)
    results = {}

    for ratio in args.ratios:
        n_sub = int(len(full_embeddings) * ratio)
        print(f"\n{'='*60}")
        print(f"RATIO={ratio} (n_samples={n_sub})")
        print(f"{'='*60}")

        # Subsample
        if ratio < 1.0:
            idx = np.sort(np.random.choice(len(full_embeddings), n_sub, replace=False))
            sub_embs = full_embeddings[idx]
        else:
            idx = np.arange(len(full_embeddings))
            sub_embs = full_embeddings

        # Baseline × 1
        print("\n[Baseline]")
        keep_b, tb = scip_baseline(sub_embs, args.n_clusters, args.p, args.alpha)
        print(f"  Time: {tb.total_time:.2f}s, Kept: {len(keep_b)}")
        for s, t in tb.step_times.items():
            print(f"    {s}: {t:.4f}s")

        # Optimized × 2
        opt_times = []
        keep_o_list = []
        for run in range(2):
            print(f"\n[Optimized run{run+1}]")
            keep_o, to = scip_optimized(sub_embs, args.n_clusters, args.p, args.alpha)
            opt_times.append(to.total_time)
            keep_o_list.append(set(keep_o))
            print(f"  Time: {to.total_time:.2f}s, Kept: {len(keep_o)}")
            for s, t in to.step_times.items():
                print(f"    {s}: {t:.4f}s")

        # Recall
        recall = len(set(keep_b) & keep_o_list[0]) / len(keep_b) if keep_b.size > 0 else 1.0
        print(f"\n  Recall: {recall*100:.1f}%")

        results[str(ratio)] = {
            'ratio': ratio,
            'n_samples': n_sub,
            'baseline_time': tb.total_time,
            'baseline_kept': len(keep_b),
            'optimized_times': opt_times,
            'optimized_kept': len(keep_o_list[0]),
            'recall': recall,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SCIP SCALING SUMMARY")
    print(f"{'='*60}")
    for r, d in sorted(results.items(), key=lambda x: -float(x[0])):
        opt_mean = np.mean(d['optimized_times'])
        speedup = d['baseline_time'] / opt_mean if opt_mean > 0 else 0
        print(f"  ratio={r}: baseline={d['baseline_time']:.2f}s, opt={opt_mean:.2f}s, speedup={speedup:.2f}x, recall={d['recall']*100:.1f}%")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
