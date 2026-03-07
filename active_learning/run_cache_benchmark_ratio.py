#!/usr/bin/env python
"""
Multi-Algorithm LRU Cache Benchmark (Ratio-Based)

Uses cache_size_ratio (cache_size / working_set_size) as the x-axis instead
of absolute cache sizes, so that all algorithms are compared fairly.

Working set sizes:
  - CAL:          ~200  unique KL-divergence keys
  - KNNPrompting: 1024  unique anchor indices
  - SCIP:         5000  unique point indices (accessed twice)
  - CRAIG:        ~6000 unique candidate indices (Zipf access)
  - SemDeDup:     ~50000 unique (cluster, point) keys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acquisition.cal import LRUCache


# ═══════════════════════════════════════════════════════════════════
# Shared: simulate cache operations given access pattern + compute fn
# ═══════════════════════════════════════════════════════════════════
def run_cache_simulation(access_keys, compute_fn, cache_max_size, description=""):
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
# Working set sizes per algorithm
# ═══════════════════════════════════════════════════════════════════
WORKING_SET_SIZES = {
    'CAL':          200,
    'KNNPrompting': 1024,
    'SCIP':         5000,
    'CRAIG':        None,  # computed at runtime from data
    'SemDeDup':     None,  # computed at runtime from data
}

# Ratios to test
CACHE_RATIOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, None]  # None = unlimited


def ratio_to_cache_size(ratio, working_set_size):
    """Convert a ratio to an absolute cache size."""
    if ratio is None:
        return None  # unlimited
    return max(1, int(ratio * working_set_size))


# ═══════════════════════════════════════════════════════════════════
# 1. CAL – Simulate with ratio-based cache sizes
# ═══════════════════════════════════════════════════════════════════
CAL_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments/cache_benchmark_20260129_234609/results.json"
)


def simulate_cal(cache_max_size, num_runs=3):
    """
    CAL caches KL-divergence computations.
    Access pattern: 200 unique pairs, accessed in AL query batches.
    Simulated: 10 batches of 20 queries, each accessing all cached pairs.
    """
    working_set = 200
    num_batches = 10
    queries_per_batch = 20
    dim = 768  # BERT-like embedding dim

    run_times = []
    for run_idx in range(num_runs):
        torch.manual_seed(42 + run_idx * 100)
        # Pre-generate data
        keys = list(range(working_set))
        log_data = [torch.softmax(torch.randn(dim), dim=-1) for _ in range(working_set)]

        def compute_kl(k):
            return torch.log(log_data[k] + 1e-10)

        # Access pattern: each batch accesses all keys
        access_keys = []
        for b in range(num_batches):
            access_keys.extend(keys)

        t, cache = run_cache_simulation(access_keys, compute_kl, cache_max_size)
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
# 2. KNN-Prompting – log(anchor) Cache
# ═══════════════════════════════════════════════════════════════════
def simulate_knnprompting(cache_max_size, num_runs=3):
    num_queries = 5
    num_anchors = 1024
    dim = 50257

    run_times = []
    for run_idx in range(num_runs):
        torch.manual_seed(42 + run_idx * 100)
        anchors = torch.softmax(torch.randn(num_anchors, dim), dim=-1)

        def compute_log(a_idx):
            return torch.log(anchors[a_idx] + 1e-10)

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
_scip_data = None

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


def simulate_scip(cache_max_size, num_runs=3):
    embeddings, centroids, labels, N, D = _scip_preload()

    def compute_dist(i):
        return 1.0 - float(np.dot(embeddings[i], centroids[labels[i]]))

    sample_N = min(N, 5000)
    # Accessed twice: once for size sort, once for dist sort
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
_craig_data = None

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


def simulate_craig(cache_max_size, num_runs=3):
    N_class, D, B, ann_k = _craig_preload()

    run_times = []
    unique_keys_set = set()
    for run_idx in range(num_runs):
        np.random.seed(42 + run_idx * 100)
        gradients_class = np.random.randn(N_class, D).astype(np.float32)

        def compute_sim_row(c_idx):
            return gradients_class[c_idx] @ gradients_class.T

        popularity = np.random.zipf(1.5, N_class).astype(float)
        popularity /= popularity.sum()
        access_keys = []
        for step in range(B):
            candidates = np.random.choice(N_class, size=ann_k, p=popularity, replace=False)
            access_keys.extend(candidates.tolist())

        if run_idx == 0:
            unique_keys_set = set(access_keys)

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
    }, len(unique_keys_set)


# ═══════════════════════════════════════════════════════════════════
# 5. SemDeDup – Max-Similarity M Cache
# ═══════════════════════════════════════════════════════════════════
def simulate_semdedup(cache_max_size, num_runs=3):
    embed_path = "/home/yichen/old/SemDeDup/cifar10_embeddings.npy"
    sorted_clusters_path = "/home/yichen/old/SemDeDup/output_cifar10_run_new_run1/sorted_clusters"
    num_clusters = 10
    eps_list = [0.1, 0.2, 0.3]

    print(f"    Loading SemDeDup cluster data...")
    cluster_data = []
    for c_id in range(num_clusters):
        cpath = os.path.join(sorted_clusters_path, f"cluster_{c_id}.npy")
        if os.path.exists(cpath):
            cdata = np.load(cpath, allow_pickle=True)
            cluster_data.append(cdata)
        else:
            cluster_data.append(None)

    raw_embs = np.memmap(embed_path, dtype='float32', mode='r', shape=(50000, 512))

    precomputed_m = {}
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
        sim_matrix = embs @ embs.T
        np.fill_diagonal(sim_matrix, -np.inf)
        M = np.max(sim_matrix, axis=1)
        for i in range(len(cluster_ids)):
            precomputed_m[(c_id, i)] = float(M[i])
        total_points += len(cluster_ids)

    print(f"    Precomputed M for {total_points} points across {num_clusters} clusters")

    def compute_m(key):
        return precomputed_m[key]

    # Count unique keys in access pattern
    access_keys_template = []
    for c_id in range(num_clusters):
        if cluster_data[c_id] is None:
            continue
        cdata = cluster_data[c_id]
        if cdata.shape[0] <= 1:
            continue
        cluster_size = cdata.shape[0]
        for i in range(cluster_size):
            access_keys_template.append((c_id, i))
        for eps in eps_list:
            for i in range(cluster_size):
                access_keys_template.append((c_id, i))

    unique_keys = len(set(access_keys_template))

    run_times = []
    for run_idx in range(num_runs):
        t, cache = run_cache_simulation(access_keys_template, compute_m, cache_max_size)
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
    }, unique_keys


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
ALL_ALGORITHMS = ['CAL', 'KNNPrompting', 'SCIP', 'CRAIG', 'SemDeDup']

SIMULATION_FNS = {
    'CAL': simulate_cal,
    'KNNPrompting': simulate_knnprompting,
    'SCIP': simulate_scip,
    'CRAIG': simulate_craig,
    'SemDeDup': simulate_semdedup,
}

# Algorithms that return (result, unique_keys) tuple
RETURNS_UNIQUE_KEYS = {'CRAIG', 'SemDeDup'}


def main():
    parser = argparse.ArgumentParser(description="Cache Benchmark (Ratio-Based)")
    parser.add_argument("--algorithms", nargs='+', default=ALL_ALGORITHMS,
                        choices=ALL_ALGORITHMS)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/cache_benchmark_ratio_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    cache_ratios = CACHE_RATIOS

    print("=" * 70)
    print("Multi-Algorithm LRU Cache Benchmark (Ratio-Based)")
    print("=" * 70)
    print(f"Algorithms: {args.algorithms}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Cache ratios: {[r if r else 'unlimited' for r in cache_ratios]}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    all_results = {}

    for algo_name in args.algorithms:
        print(f"\n{'='*60}")
        print(f"  Algorithm: {algo_name}")
        print(f"{'='*60}")

        algo_func = SIMULATION_FNS[algo_name]

        # Determine working set size
        # For CRAIG and SemDeDup, run a probe first to get unique key count
        ws = WORKING_SET_SIZES.get(algo_name)
        if ws is None:
            print(f"  Probing working set size...")
            if algo_name == 'CRAIG':
                probe_result, ws = algo_func(cache_max_size=None, num_runs=1)
            elif algo_name == 'SemDeDup':
                probe_result, ws = algo_func(cache_max_size=None, num_runs=1)
            print(f"  Working set size: {ws}")
        else:
            print(f"  Working set size: {ws}")

        algo_results = []
        for ratio in cache_ratios:
            cache_size = ratio_to_cache_size(ratio, ws) if ratio is not None else None
            ratio_str = f"{ratio}" if ratio else "unlimited"
            cache_str = str(cache_size) if cache_size else "unlimited"
            print(f"\n  ratio={ratio_str} (cache_size={cache_str}):")

            if algo_name in RETURNS_UNIQUE_KEYS:
                result, _ = algo_func(cache_max_size=cache_size, num_runs=args.num_runs)
            else:
                result = algo_func(cache_max_size=cache_size, num_runs=args.num_runs)

            # Add ratio metadata
            result['cache_ratio'] = ratio
            result['working_set_size'] = ws
            algo_results.append(result)

            print(f"  → Mean: {result['cache_time_mean']:.4f}s ± {result['cache_time_std']:.4f}s, "
                  f"Evictions: {result['cache_evictions']}, Hit rate: {result['hit_rate']:.1f}%")

        all_results[algo_name] = algo_results

    # Save results
    output_file = os.path.join(args.output_dir, "results_ratio.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for algo_name, results in all_results.items():
        ws = results[0].get('working_set_size', '?')
        print(f"\n{algo_name} (working_set={ws}):")
        print(f"  {'Ratio':<10} {'Cache Size':<12} {'Time (s)':<20} {'Evictions':<12} {'Hit Rate':<12}")
        print(f"  {'-'*70}")
        for r in results:
            rs = str(r['cache_ratio']) if r['cache_ratio'] else 'unlimited'
            cs = str(r['cache_max_size']) if r['cache_max_size'] else 'unlimited'
            ts = f"{r['cache_time_mean']:.4f} ± {r['cache_time_std']:.4f}"
            print(f"  {rs:<10} {cs:<12} {ts:<20} {r['cache_evictions']:<12} {r['hit_rate']:<12.1f}%")

    print(f"\nResults saved to: {output_file}")
    print(f"Plot:  python /home/yichen/old/draw/plot_cache_benchmark_ratio.py {output_file}")


if __name__ == "__main__":
    main()
