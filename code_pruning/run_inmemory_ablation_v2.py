"""
In-Memory SCIP Ablation Experiments - Enhanced Version

Enhancements:
1. Individual run times for each breakdown step
2. Standard deviation for each step
3. ANN recall (label overlap with baseline)
4. Reuse savings rate (computation saved)

Usage:
    python run_inmemory_ablation_v2.py --mode original --n_clusters 1000 --n_runs 5
"""

import os
import json
import time
import argparse
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class Config:
    max_samples: int = 50000
    n_clusters: int = 1000
    p: float = 0.2
    alpha: float = 0.8
    n_runs: int = 5
    output_dir: str = "./experiments/ablation"


@dataclass
class TimingResult:
    step_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    
    def add_step(self, name: str, elapsed: float):
        self.step_times[name] = elapsed
        self.total_time += elapsed


def load_embeddings(cfg: Config) -> np.ndarray:
    """Load pre-computed embeddings."""
    emb_path = "./experiments/aligned/embeddings.npy"
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found at {emb_path}")
    
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings


def scip_baseline(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult, np.ndarray, int]:
    """Baseline: Faiss Flat + No Reuse + Full Sort
    
    Returns:
        keep_indices, timing, labels, reuse_count (0 for baseline)
    """
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss Exact search
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(D)
    index.add(centroids.astype(np.float32))
    sims, labels = index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances - no reuse
    t0 = time.perf_counter()
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Sort by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    idx_by_size = np.argsort(size_per_point)
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("SortBySize", time.perf_counter() - t0)
    
    # Sort by distance
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
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing, labels, 0


def scip_ann_only(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult, np.ndarray, int]:
    """ANN only: Faiss IVF + No Reuse + Full Sort"""
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss IVF search (ANN)
    t0 = time.perf_counter()
    centroids_f32 = centroids.astype(np.float32)
    nlist = max(4, int(np.sqrt(K)))
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(centroids_f32)
    index.add(centroids_f32)
    index.nprobe = max(1, nlist // 4)
    
    sims, labels = index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances - no reuse
    t0 = time.perf_counter()
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Sort by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    idx_by_size = np.argsort(size_per_point)
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("SortBySize", time.perf_counter() - t0)
    
    # Sort by distance
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
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing, labels, 0


def scip_reuse_only(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult, np.ndarray, int]:
    """Reuse only: Faiss Flat + Reuse + Full Sort"""
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss Exact search
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(D)
    index.add(centroids.astype(np.float32))
    sims, labels = index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    sims = sims.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances - REUSE from search
    t0 = time.perf_counter()
    distances = 1.0 - sims
    reuse_count = N  # All N distances reused
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Sort by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    idx_by_size = np.argsort(size_per_point)
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("SortBySize", time.perf_counter() - t0)
    
    # Sort by distance - REUSE
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    
    dist_remaining = distances[remaining]  # REUSE!
    
    order = np.argsort(-dist_remaining)
    dist_quota = total_prune - size_quota
    prune_by_dist = remaining[order[:dist_quota]]
    timing.add_step("SortByDist", time.perf_counter() - t0)
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing, labels, reuse_count


def scip_topk_only(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult, np.ndarray, int]:
    """TopK only: Faiss Flat + No Reuse + TopK"""
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss Exact search
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(D)
    index.add(centroids.astype(np.float32))
    sims, labels = index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances - no reuse
    t0 = time.perf_counter()
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # TopK by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    
    if size_quota > 0:
        idx_by_size = np.argpartition(size_per_point, size_quota)
    else:
        idx_by_size = np.argsort(size_per_point)
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("TopKBySize", time.perf_counter() - t0)
    
    # TopK by distance
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    
    dist_remaining = np.zeros(len(remaining), dtype=np.float32)
    for idx, i in enumerate(remaining):
        dist_remaining[idx] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    
    dist_quota = total_prune - size_quota
    if dist_quota > 0 and len(remaining) > 0:
        part_idx = np.argpartition(-dist_remaining, dist_quota)
        prune_by_dist = remaining[part_idx[:dist_quota]]
    else:
        prune_by_dist = np.array([], dtype=np.int64)
    timing.add_step("TopKByDist", time.perf_counter() - t0)
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing, labels, 0


def scip_optimized(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult, np.ndarray, int]:
    """Fully Optimized: Faiss IVF + Reuse + TopK"""
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss IVF search (ANN)
    t0 = time.perf_counter()
    centroids_f32 = centroids.astype(np.float32)
    nlist = max(4, int(np.sqrt(K)))
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(centroids_f32)
    index.add(centroids_f32)
    index.nprobe = max(1, nlist // 4)
    
    sims, labels = index.search(embeddings.astype(np.float32), k=1)
    labels = labels.flatten()
    sims = sims.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances - REUSE from search
    t0 = time.perf_counter()
    distances = 1.0 - sims
    reuse_count = N
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # TopK by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    
    if size_quota > 0:
        idx_by_size = np.argpartition(size_per_point, size_quota)
    else:
        idx_by_size = np.argsort(size_per_point)
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("TopKBySize", time.perf_counter() - t0)
    
    # TopK by distance - REUSE
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    
    dist_remaining = distances[remaining]  # REUSE!
    
    dist_quota = total_prune - size_quota
    if dist_quota > 0 and len(remaining) > 0:
        part_idx = np.argpartition(-dist_remaining, dist_quota)
        prune_by_dist = remaining[part_idx[:dist_quota]]
    else:
        prune_by_dist = np.array([], dtype=np.int64)
    timing.add_step("TopKByDist", time.perf_counter() - t0)
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing, labels, reuse_count


def run_experiment(mode: str, cfg: Config):
    """Run ablation experiment for the specified mode."""
    print(f"\n{'='*60}")
    print(f"Running Ablation Experiment: {mode.upper()}")
    print(f"{'='*60}")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Load embeddings
    embeddings = load_embeddings(cfg)
    
    algo_map = {
        "original": scip_baseline,
        "iv-aligned": scip_optimized,
        "iv1": scip_ann_only,
        "iv2": scip_reuse_only,
        "iv3": scip_topk_only,
    }
    
    if mode not in algo_map:
        raise ValueError(f"Unknown mode: {mode}")
    
    algo_func = algo_map[mode]
    
    # Run multiple times and collect all data
    all_timings = []
    all_labels = []
    all_keep_indices = []
    all_reuse_counts = []
    
    for run in range(cfg.n_runs):
        print(f"  Run {run+1}/{cfg.n_runs}...")
        keep, timing, labels, reuse_count = algo_func(embeddings, cfg)
        all_timings.append(timing)
        all_labels.append(labels)
        all_keep_indices.append(keep)
        all_reuse_counts.append(reuse_count)
    
    # Calculate statistics
    individual_times = [t.total_time for t in all_timings]
    
    # Breakdown stats (mean and std for each step)
    all_steps = list(all_timings[0].step_times.keys())
    breakdown_mean = {}
    breakdown_std = {}
    breakdown_individual = {step: [] for step in all_steps}
    
    for step in all_steps:
        times = [t.step_times.get(step, 0.0) for t in all_timings]
        breakdown_mean[step] = float(np.mean(times))
        breakdown_std[step] = float(np.std(times))
        breakdown_individual[step] = [float(t) for t in times]
    
    # Calculate ANN recall (if not original)
    ann_recall = None
    if mode == "iv1":
        # Load original labels for comparison
        original_labels_path = os.path.join(cfg.output_dir, "original_labels.npy")
        if os.path.exists(original_labels_path):
            original_labels = np.load(original_labels_path)
            # Calculate label agreement
            ann_recall = float(np.mean(all_labels[0] == original_labels))
    
    # Save original labels for ANN recall calculation
    if mode == "original":
        np.save(os.path.join(cfg.output_dir, "original_labels.npy"), all_labels[0])
    
    # Calculate reuse savings rate
    reuse_savings_rate = None
    if mode == "iv2" and all_reuse_counts[0] > 0:
        N = len(embeddings)
        # Baseline computes distances for all N samples twice (initial + remaining)
        # Reuse only computes once and reuses
        total_computations_baseline = N + len(all_keep_indices[0])  # Approximate
        reuse_savings_rate = float(all_reuse_counts[0] / N)  # Simplified: reused / total
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {mode.upper()}")
    print(f"{'='*60}")
    
    print(f"\nTotal Time:")
    for i, t in enumerate(individual_times):
        print(f"  Run {i+1}: {t:.4f}s")
    print(f"  Mean: {np.mean(individual_times):.4f}s ± {np.std(individual_times):.4f}s")
    
    print(f"\nBreakdown (mean ± std):")
    for step in all_steps:
        mean_val = breakdown_mean[step]
        std_val = breakdown_std[step]
        print(f"  {step}: {mean_val:.4f}s ± {std_val:.4f}s")
        # Show individual runs
        for i, val in enumerate(breakdown_individual[step]):
            print(f"    Run {i+1}: {val:.4f}s")
    
    if ann_recall is not None:
        print(f"\nANN Recall (label agreement with original): {ann_recall*100:.2f}%")
    
    if reuse_savings_rate is not None:
        print(f"\nReuse Savings Rate: {reuse_savings_rate*100:.2f}%")
    
    # Save results
    results = {
        "mode": mode,
        "n_samples": len(embeddings),
        "n_clusters": cfg.n_clusters,
        "n_runs": cfg.n_runs,
        "total_time": {
            "individual": individual_times,
            "mean": float(np.mean(individual_times)),
            "std": float(np.std(individual_times)),
            "min": float(np.min(individual_times)),
            "max": float(np.max(individual_times)),
        },
        "breakdown": {
            "mean": breakdown_mean,
            "std": breakdown_std,
            "individual": breakdown_individual,
        },
        "ann_recall": ann_recall,
        "reuse_savings_rate": reuse_savings_rate,
        "keep_count": len(all_keep_indices[0]),
    }
    
    output_file = os.path.join(cfg.output_dir, f"{mode}_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="In-Memory SCIP Ablation Experiments v2")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["original", "iv-aligned", "iv1", "iv2", "iv3"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--n_clusters", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./experiments/ablation")
    args = parser.parse_args()
    
    cfg = Config(
        n_runs=args.n_runs,
        n_clusters=args.n_clusters,
        output_dir=args.output_dir,
    )
    
    run_experiment(args.mode, cfg)


if __name__ == "__main__":
    main()
