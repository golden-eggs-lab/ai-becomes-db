"""
Measure build time, query time, and peak cache memory for CRAIG.
Uses ACTUAL FacilityLocation / FacilityLocationANN + lazy_greedy_heap.

Baseline:  FacilityLocation + lazy_greedy_heap        (exact, no ANN, no reuse)
Optimized: FacilityLocationANN + lazy_greedy_heap_ann  (NearPy ANN + residual reuse)
"""

import time
import tracemalloc
import numpy as np
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from lazy_greedy import FacilityLocation, lazy_greedy_heap, FacilityLocationANN, lazy_greedy_heap_ann

# Config
N_RUNS = 3
SUBSET_SIZE = 0.4
ANN_K = 20
GRADIENT_PATH = "./precomputed_gradients/mnist_gradients.npz"


def load_data():
    data = np.load(GRADIENT_PATH)
    gradients = data['gradients']
    labels = data['labels']
    print(f"Loaded gradients: {gradients.shape}, labels: {labels.shape}")
    return gradients, labels


def measure_baseline(S, N, B, n_runs):
    """Baseline: FacilityLocation + lazy_greedy_heap (no ANN, no index)."""
    build_times = []
    query_times = []

    for run in range(n_runs):
        # BUILD: No index for baseline
        build_times.append(0.0)

        # QUERY: lazy_greedy_heap with exact similarity matrix
        V = list(range(N))
        t0 = time.perf_counter()
        F = FacilityLocation(S, V)
        sset, vals = lazy_greedy_heap(F, V, B)
        query_time = time.perf_counter() - t0
        query_times.append(query_time)
        print(f"    Baseline run {run+1}/{n_runs}: query={query_time:.2f}s, selected={len(sset)}")

    return build_times, query_times


def measure_optimized(S, gradients_class, N, B, n_runs):
    """Optimized: FacilityLocationANN + lazy_greedy_heap_ann (NearPy ANN + reuse)."""
    build_times = []
    query_times = []

    for run in range(n_runs):
        V = list(range(N))

        # BUILD: NearPy engine (measured inside FacilityLocationANN.__init__)
        t0 = time.perf_counter()
        F = FacilityLocationANN(S, V, gradients_class, ann_k=ANN_K, use_ann=True,
                                ann_backend='nearpy', use_reuse=True)
        build_time = time.perf_counter() - t0
        build_times.append(build_time)

        # QUERY: lazy_greedy_heap_ann
        t0 = time.perf_counter()
        sset, vals = lazy_greedy_heap_ann(F, V, B, use_ann=True)
        query_time = time.perf_counter() - t0
        query_times.append(query_time)
        print(f"    Optimized run {run+1}/{n_runs}: build={build_time:.4f}s, query={query_time:.2f}s, selected={len(sset)}")

    return build_times, query_times


def measure_cache_memory(gradients_class, B):
    """Measure peak memory of the cache (residual vector)."""
    N, d = gradients_class.shape

    tracemalloc.start()
    current_residual = np.sum(gradients_class, axis=0)
    for step in range(B):
        current_residual = current_residual - gradients_class[step % N]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cache_bytes = d * 4  # float32
    return peak, cache_bytes


def main():
    print("=" * 60)
    print("CRAIG Build/Query/Cache Measurement (NearPy)")
    print("=" * 60)

    gradients, labels = load_data()
    N_total, d = gradients.shape
    classes = np.unique(labels)

    print(f"Dataset: MNIST, N={N_total}, d={d}, classes={len(classes)}")
    print(f"Subset size: {SUBSET_SIZE}, ANN k: {ANN_K}, Runs: {N_RUNS}")

    # Use class 0
    class_idx = 0
    class_mask = labels == class_idx
    gradients_class = gradients[class_mask]
    N_class = gradients_class.shape[0]
    B_class = int(SUBSET_SIZE * N_class)

    print(f"\nClass {class_idx}: N={N_class}, B={B_class}, d={d}")

    # Similarity matrix
    print(f"\nComputing similarity matrix ({N_class}x{N_class})...")
    t0 = time.perf_counter()
    dists = sklearn_pairwise_distances(gradients_class, metric='euclidean', n_jobs=1)
    m = np.max(dists)
    S = m - dists
    sim_time = time.perf_counter() - t0
    print(f"  Similarity matrix time: {sim_time:.2f}s")

    # Baseline
    print(f"\n{'='*60}")
    print(f"Baseline (FacilityLocation + lazy_greedy_heap)")
    print(f"{'='*60}")
    bl_build, bl_query = measure_baseline(S, N_class, B_class, N_RUNS)

    # Optimized
    print(f"\n{'='*60}")
    print(f"Optimized (FacilityLocationANN + lazy_greedy_heap_ann)")
    print(f"{'='*60}")
    opt_build, opt_query = measure_optimized(S, gradients_class, N_class, B_class, N_RUNS)

    # Cache memory
    print(f"\n{'='*60}")
    print(f"Cache Memory (residual vector)")
    print(f"{'='*60}")
    peak_mem, cache_bytes = measure_cache_memory(gradients_class, B_class)
    print(f"  Peak tracemalloc: {peak_mem / 1024:.2f} KB")
    print(f"  Cache (residual vector): {cache_bytes} bytes = {cache_bytes / 1024:.2f} KB")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (class 0: N={N_class}, B={B_class}, d={d})")
    print(f"{'='*60}")
    print(f"{'':>25s} {'Baseline':>20s} {'Optimized':>20s}")
    print(f"{'Build (s)':>25s} {'0 (no index)':>20s} {np.mean(opt_build):>15.4f}±{np.std(opt_build):.4f}")
    print(f"{'Query (s)':>25s} {np.mean(bl_query):>15.4f}±{np.std(bl_query):.4f} {np.mean(opt_query):>15.4f}±{np.std(opt_query):.4f}")
    if np.mean(opt_query) > 0:
        print(f"{'Query Speedup':>25s} {'':>20s} {np.mean(bl_query)/np.mean(opt_query):>19.1f}x")
    print(f"{'Cache Peak Mem':>25s} {'0 (no cache)':>20s} {peak_mem/1024:>17.2f} KB")


if __name__ == "__main__":
    main()
