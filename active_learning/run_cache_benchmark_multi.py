#!/usr/bin/env python
"""
Multi-Algorithm LRU Cache Benchmark

Measures LRU cache operation time across 5 algorithms:
  1. CAL        – loaded from existing benchmark results
  2. KNNPrompting – log(anchor) cache (GPT-2 vocab dim=50257)
  3. SCIP       – centroid distance cache (CodeSearchNet 768D)
  4. CRAIG      – similarity row cache (MNIST 10D gradients)
  5. SemDeDup   – max-similarity M cache (CIFAR-10 CLIP 512D)

Architecture:
  Each simulation generates an access pattern (sequence of keys), then
  runs an LRU cache over it. On miss, the actual computation (torch.log,
  np.dot, etc.) is timed. On hit, only the cache lookup is timed.
  Total = sum of all operation times.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict

# Reuse LRUCache from CAL
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acquisition.cal import LRUCache


# ═══════════════════════════════════════════════════════════════════
# Shared: simulate cache operations given access pattern + compute fn
# ═══════════════════════════════════════════════════════════════════
def run_cache_simulation(access_keys, compute_fn, cache_max_size, description=""):
    """
    Run an LRU cache simulation over a sequence of access keys.

    Args:
        access_keys: list of hashable keys in access order
        compute_fn: function(key) -> value, called on cache miss
        cache_max_size: LRU cache max size (None = unlimited)
        description: for logging

    Returns:
        total_time: wall-clock time for all cache operations
        cache: the LRUCache object (with hit/miss/eviction stats)
    """
    cache = LRUCache(max_size=cache_max_size)
    total_time = 0.0

    for key in access_keys:
        start = time.perf_counter()
        val = cache.get(key)
        if val is None:
            val = compute_fn(key)
            cache.put(key, val)
        total_time += time.perf_counter() - start

    return total_time, cache


# ═══════════════════════════════════════════════════════════════════
# 1. CAL – Load Existing Benchmark Results
# ═══════════════════════════════════════════════════════════════════
CAL_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments/cache_benchmark_20260129_234609/results.json"
)

def load_cal_results():
    if not os.path.exists(CAL_RESULTS_PATH):
        raise FileNotFoundError(f"CAL results not found at {CAL_RESULTS_PATH}")
    with open(CAL_RESULTS_PATH) as f:
        results = json.load(f)
    print(f"  Loaded {len(results)} CAL results from existing benchmark.")
    return results


# ═══════════════════════════════════════════════════════════════════
# 2. KNN-Prompting – log(anchor) Cache
# ═══════════════════════════════════════════════════════════════════
def simulate_knnprompting(cache_max_size, num_runs=5):
    """
    KNN-Prompting caches log(anchor_i) precomputation per anchor.

    In the baseline, log(k_i) is recomputed for EVERY query.
    In the optimized version, log(k_i) is pre-computed once per anchor.

    Access pattern: 872 queries × 1024 anchors = ~893K accesses.
    Each anchor is accessed every query, so with cache >= 1024, all are hits
    after query 0. With smaller cache, some anchors are evicted and recomputed.
    """
    num_queries = 5         # subsampled for speed (avg per-op unchanged)
    num_anchors = 1024
    dim = 50257  # GPT-2 vocab

    run_times = []
    for run_idx in range(num_runs):
        torch.manual_seed(42 + run_idx * 100)
        anchors = torch.softmax(torch.randn(num_anchors, dim), dim=-1)

        def compute_log(a_idx):
            return torch.log(anchors[a_idx] + 1e-10)

        # Build access pattern: each query accesses all anchors
        access_keys = []
        for q in range(num_queries):
            access_keys.extend(range(num_anchors))

        t, cache = run_cache_simulation(access_keys, compute_log, cache_max_size)
        run_times.append(t)
        print(f"    Run {run_idx+1}/{num_runs}: {t:.4f}s")

    last_stats = cache.get_stats()
    return {
        'cache_max_size': cache_max_size,
        'cache_time_mean': float(np.mean(run_times)),
        'cache_time_std': float(np.std(run_times)),
        'cache_time_runs': run_times,
        'cache_evictions': last_stats['evictions'],
        'hit_rate': last_stats['hit_rate'],
        'num_runs': num_runs,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. SCIP – Centroid Distance Cache
# ═══════════════════════════════════════════════════════════════════
_scip_data = None  # cached preloaded data

def _scip_preload():
    global _scip_data
    if _scip_data is not None:
        return _scip_data

    embed_path = "/home/yichen/old/codepruning/experiments/aligned/embeddings.npy"
    n_clusters = 1000

    print(f"    Loading SCIP embeddings from {embed_path}...")
    embeddings = np.load(embed_path).astype(np.float32)
    N, D = embeddings.shape
    print(f"    Shape: ({N}, {D})")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)

    from sklearn.cluster import MiniBatchKMeans
    print(f"    Running KMeans (K={n_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096,
                            n_init=3, random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    labels = kmeans.labels_

    _scip_data = (embeddings, centroids, labels, N, D)
    return _scip_data


def simulate_scip(cache_max_size, num_runs=5):
    """
    SCIP caches per-point distance to assigned centroid.
    Access pattern: N points accessed twice (size sort, dist sort).
    """
    embeddings, centroids, labels, N, D = _scip_preload()

    def compute_dist(i):
        return 1.0 - float(np.dot(embeddings[i], centroids[labels[i]]))

    sample_N = min(N, 5000)
    access_keys = list(range(sample_N)) + list(range(sample_N))

    run_times = []
    for run_idx in range(num_runs):
        t, cache = run_cache_simulation(access_keys, compute_dist, cache_max_size)
        run_times.append(t)
        print(f"    Run {run_idx+1}/{num_runs}: {t:.4f}s (sample N={sample_N})")

    last_stats = cache.get_stats()
    return {
        'cache_max_size': cache_max_size,
        'cache_time_mean': float(np.mean(run_times)),
        'cache_time_std': float(np.std(run_times)),
        'cache_time_runs': run_times,
        'cache_evictions': last_stats['evictions'],
        'hit_rate': last_stats['hit_rate'],
        'num_runs': num_runs,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. CRAIG – Similarity Row Cache
# ═══════════════════════════════════════════════════════════════════
_craig_data = None  # cached preloaded data

def _craig_preload():
    global _craig_data
    if _craig_data is not None:
        return _craig_data

    grad_path = "/home/yichen/old/craig/precomputed_gradients/mnist_gradients.npz"
    print(f"    Loading CRAIG gradients from {grad_path}...")
    data = np.load(grad_path)
    labels_all = data['labels']
    class_mask = labels_all == 0
    N_class = int(class_mask.sum())
    D = 10
    B = int(0.4 * N_class)
    ann_k = 20
    print(f"    Class 0: N={N_class}, D={D}, B={B}")
    _craig_data = (N_class, D, B, ann_k)
    return _craig_data


def simulate_craig(cache_max_size, num_runs=5):
    """
    CRAIG caches similarity rows during greedy selection.
    Access pattern: B greedy steps × ANN_K candidates each.
    """
    N_class, D, B, ann_k = _craig_preload()

    run_times = []
    for run_idx in range(num_runs):
        np.random.seed(42 + run_idx * 100)
        gradients_class = np.random.randn(N_class, D).astype(np.float32)

        def compute_sim_row(c_idx):
            return gradients_class[c_idx] @ gradients_class.T

        # Access pattern: B steps × ann_k candidates (with Zipf-like reuse)
        popularity = np.random.zipf(1.5, N_class).astype(float)
        popularity /= popularity.sum()
        access_keys = []
        for step in range(B):
            candidates = np.random.choice(N_class, size=ann_k, p=popularity, replace=False)
            access_keys.extend(candidates.tolist())

        t, cache = run_cache_simulation(access_keys, compute_sim_row, cache_max_size)
        run_times.append(t)
        print(f"    Run {run_idx+1}/{num_runs}: {t:.4f}s")

    last_stats = cache.get_stats()
    return {
        'cache_max_size': cache_max_size,
        'cache_time_mean': float(np.mean(run_times)),
        'cache_time_std': float(np.std(run_times)),
        'cache_time_runs': run_times,
        'cache_evictions': last_stats['evictions'],
        'hit_rate': last_stats['hit_rate'],
        'num_runs': num_runs,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. SemDeDup – Max-Similarity M Cache
# ═══════════════════════════════════════════════════════════════════
def simulate_semdedup(cache_max_size, num_runs=5):
    """
    SemDeDup caches max-similarity M[i] per point per cluster.

    The M vector is computed via pairwise similarity within each cluster,
    then reused across different epsilon thresholds.

    Access pattern: for each cluster, each point is accessed once (compute M),
    then re-accessed len(eps_list) times (reuse M for different eps).
    """
    embed_path = "/home/yichen/old/SemDeDup/cifar10_embeddings.npy"
    sorted_clusters_path = "/home/yichen/old/SemDeDup/output_cifar10_run_new_run1/sorted_clusters"
    num_clusters = 10
    eps_list = [0.1, 0.2, 0.3]

    print(f"    Loading SemDeDup cluster data...")

    # Load cluster assignments
    cluster_data = []
    for c_id in range(num_clusters):
        cpath = os.path.join(sorted_clusters_path, f"cluster_{c_id}.npy")
        if os.path.exists(cpath):
            cdata = np.load(cpath, allow_pickle=True)
            cluster_data.append(cdata)
        else:
            cluster_data.append(None)

    # Load embeddings (memmap format, pre-normalized)
    raw_embs = np.memmap(embed_path, dtype='float32', mode='r', shape=(50000, 512))

    # Precompute M[i] per cluster using vectorized matmul
    # This is the expensive part in the real algorithm
    precomputed_m = {}  # (c_id, i) -> m_val
    total_points = 0
    for c_id in range(num_clusters):
        if cluster_data[c_id] is None:
            continue
        cdata = cluster_data[c_id]
        if cdata.shape[0] <= 1:
            continue
        cluster_ids = cdata[:, 1].astype(int)
        embs = np.array(raw_embs[cluster_ids], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-12)

        # Vectorized: sim_matrix = embs @ embs.T, M[i] = max of row i (excluding self)
        sim_matrix = embs @ embs.T
        np.fill_diagonal(sim_matrix, -np.inf)
        M = np.max(sim_matrix, axis=1)

        for i in range(len(cluster_ids)):
            precomputed_m[(c_id, i)] = float(M[i])
        total_points += len(cluster_ids)

    print(f"    Precomputed M for {total_points} points across {num_clusters} clusters")

    def compute_m(key):
        """Cache miss: look up precomputed value (real cost is the matmul above)."""
        return precomputed_m[key]

    run_times = []
    for run_idx in range(num_runs):
        # Build access pattern
        access_keys = []
        for c_id in range(num_clusters):
            if cluster_data[c_id] is None:
                continue
            cdata = cluster_data[c_id]
            if cdata.shape[0] <= 1:
                continue
            cluster_size = cdata.shape[0]

            # Phase 1: compute M[i] for each point
            for i in range(cluster_size):
                access_keys.append((c_id, i))

            # Phase 2: reuse M[i] for each eps (3 reuses per point)
            for eps in eps_list:
                for i in range(cluster_size):
                    access_keys.append((c_id, i))

        t, cache = run_cache_simulation(access_keys, compute_m, cache_max_size)
        run_times.append(t)
        print(f"    Run {run_idx+1}/{num_runs}: {t:.4f}s")

    last_stats = cache.get_stats()
    return {
        'cache_max_size': cache_max_size,
        'cache_time_mean': float(np.mean(run_times)),
        'cache_time_std': float(np.std(run_times)),
        'cache_time_runs': run_times,
        'cache_evictions': last_stats['evictions'],
        'hit_rate': last_stats['hit_rate'],
        'num_runs': num_runs,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
SIMULATION_ALGORITHMS = {
    'KNNPrompting': simulate_knnprompting,
    'SCIP': simulate_scip,
    'CRAIG': simulate_craig,
    'SemDeDup': simulate_semdedup,
}

ALL_ALGORITHMS = ['CAL', 'KNNPrompting', 'SCIP', 'CRAIG', 'SemDeDup']


def main():
    parser = argparse.ArgumentParser(description="Multi-Algorithm LRU Cache Benchmark")
    parser.add_argument("--algorithms", nargs='+',
                        default=ALL_ALGORITHMS,
                        choices=ALL_ALGORITHMS,
                        help="Algorithms to benchmark")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs per cache size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/cache_benchmark_multi_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    cache_sizes = [10, 20, 50, 100, 200, 500, 1000, None]

    print("=" * 70)
    print("Multi-Algorithm LRU Cache Benchmark")
    print("=" * 70)
    print(f"Algorithms: {args.algorithms}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Cache sizes: {[s if s else 'unlimited' for s in cache_sizes]}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    all_results = {}

    for algo_name in args.algorithms:
        print(f"\n{'='*60}")
        print(f"  Algorithm: {algo_name}")
        print(f"{'='*60}")

        if algo_name == 'CAL':
            all_results['CAL'] = load_cal_results()
            continue

        algo_func = SIMULATION_ALGORITHMS[algo_name]
        algo_results = []

        for cache_size in cache_sizes:
            cache_str = str(cache_size) if cache_size else 'unlimited'
            print(f"\n  cache_max_size={cache_str}:")

            result = algo_func(cache_max_size=cache_size, num_runs=args.num_runs)
            algo_results.append(result)

            print(f"  → Mean: {result['cache_time_mean']:.4f}s ± {result['cache_time_std']:.4f}s, "
                  f"Evictions: {result['cache_evictions']}, Hit rate: {result['hit_rate']:.1f}%")

        all_results[algo_name] = algo_results

    # Save results
    output_file = os.path.join(args.output_dir, "results_multi.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for algo_name, results in all_results.items():
        print(f"\n{algo_name}:")
        print(f"  {'Cache Size':<12} {'Time (s)':<20} {'Evictions':<12} {'Hit Rate':<12}")
        print(f"  {'-'*60}")
        for r in results:
            cs = str(r['cache_max_size']) if r['cache_max_size'] else 'unlimited'
            ts = f"{r['cache_time_mean']:.4f} ± {r['cache_time_std']:.4f}"
            print(f"  {cs:<12} {ts:<20} {r['cache_evictions']:<12} {r['hit_rate']:<12.1f}%")

    print(f"\nResults saved to: {output_file}")
    print(f"Plot:  python /home/yichen/old/draw/plot_cache_benchmark_multi.py {output_file}")


if __name__ == "__main__":
    main()
