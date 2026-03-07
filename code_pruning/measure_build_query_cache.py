"""
Measure build time, query time, and peak cache memory for CodePruning (SCIP).

Build = FAISS index construction on centroids
Query = FAISS index.search(embeddings, k=1) for all N points  
Cache = the reused sims array (distances = 1.0 - sims)
"""

import os
import time
import tracemalloc
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans

# Config
N_CLUSTERS = 1000
N_RUNS = 3
EMBED_PATH = "./experiments/aligned/embeddings.npy"

def load_embeddings():
    """Load embeddings."""
    emb = np.load(EMBED_PATH)
    print(f"Loaded embeddings: {emb.shape}")
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-12)
    return emb.astype(np.float32)

def run_kmeans(embeddings, n_clusters):
    """Run KMeans and return normalized centroids."""
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return centroids.astype(np.float32)

def measure_baseline(embeddings, centroids, n_runs):
    """Baseline: Flat index + No Reuse."""
    N, D = embeddings.shape
    K = centroids.shape[0]
    
    build_times = []
    query_times = []
    
    for run in range(n_runs):
        # Build: index construction
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(D)
        index.add(centroids)
        build_time = time.perf_counter() - t0
        build_times.append(build_time)
        
        # Query: search all embeddings
        t0 = time.perf_counter()
        sims, labels = index.search(embeddings, k=1)
        query_time = time.perf_counter() - t0
        query_times.append(query_time)
    
    return build_times, query_times

def measure_ann(embeddings, centroids, n_runs):
    """ANN: IVF index + No Reuse."""
    N, D = embeddings.shape
    K = centroids.shape[0]
    nlist = max(1, int(np.sqrt(K)))
    
    build_times = []
    query_times = []
    
    for run in range(n_runs):
        # Build: index construction + training
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(centroids)
        index.add(centroids)
        index.nprobe = max(1, nlist // 2)
        build_time = time.perf_counter() - t0
        build_times.append(build_time)
        
        # Query: search all embeddings
        t0 = time.perf_counter()
        sims, labels = index.search(embeddings, k=1)
        query_time = time.perf_counter() - t0
        query_times.append(query_time)
    
    return build_times, query_times

def measure_cache_memory(embeddings, centroids):
    """Measure peak memory of the cache (reused sims array)."""
    N, D = embeddings.shape
    
    # Build index
    index = faiss.IndexFlatIP(D)
    index.add(centroids)
    
    # Measure memory of the cache operation
    tracemalloc.start()
    
    # The "cache" is: search returns sims, which are reused as distances
    sims, labels = index.search(embeddings, k=1)
    sims_flat = sims.flatten()
    labels_flat = labels.flatten()
    
    # In the optimized version, distances = 1.0 - sims (reuse)
    distances = 1.0 - sims_flat  # This IS the cache
    
    # Also need cluster_sizes and size_per_point
    cluster_sizes = np.bincount(labels_flat, minlength=centroids.shape[0])
    size_per_point = cluster_sizes[labels_flat]
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return peak, distances.nbytes

def main():
    print("=" * 60)
    print("CodePruning (SCIP) Build/Query/Cache Measurement")
    print("=" * 60)
    
    embeddings = load_embeddings()
    N, D = embeddings.shape
    
    print(f"\nRunning KMeans (K={N_CLUSTERS})...")
    t0 = time.perf_counter()
    centroids = run_kmeans(embeddings, N_CLUSTERS)
    kmeans_time = time.perf_counter() - t0
    print(f"KMeans time: {kmeans_time:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"Baseline (Flat Index)")
    print(f"{'='*60}")
    build_times, query_times = measure_baseline(embeddings, centroids, N_RUNS)
    print(f"  Build times: {build_times}")
    print(f"  Query times: {query_times}")
    print(f"  Build: {np.mean(build_times):.6f} ± {np.std(build_times):.6f}s")
    print(f"  Query: {np.mean(query_times):.4f} ± {np.std(query_times):.4f}s")
    
    print(f"\n{'='*60}")
    print(f"Optimized (IVF Index)")
    print(f"{'='*60}")
    ann_build_times, ann_query_times = measure_ann(embeddings, centroids, N_RUNS)
    print(f"  Build times: {ann_build_times}")
    print(f"  Query times: {ann_query_times}")
    print(f"  Build: {np.mean(ann_build_times):.6f} ± {np.std(ann_build_times):.6f}s")
    print(f"  Query: {np.mean(ann_query_times):.4f} ± {np.std(ann_query_times):.4f}s")
    
    print(f"\n{'='*60}")
    print(f"Cache Memory (distance reuse)")
    print(f"{'='*60}")
    peak_mem, cache_bytes = measure_cache_memory(embeddings, centroids)
    print(f"  Peak tracemalloc memory: {peak_mem / 1024 / 1024:.2f} MB")
    print(f"  Cache (sims array) size: {cache_bytes / 1024 / 1024:.2f} MB")
    print(f"  Cache shape: [{N}] float32")
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  N={N:,}, D={D}, K={N_CLUSTERS}")
    print(f"  Baseline Build: {np.mean(build_times):.6f} ± {np.std(build_times):.6f}s")
    print(f"  Baseline Query: {np.mean(query_times):.4f} ± {np.std(query_times):.4f}s")
    print(f"  ANN Build:      {np.mean(ann_build_times):.6f} ± {np.std(ann_build_times):.6f}s")
    print(f"  ANN Query:      {np.mean(ann_query_times):.4f} ± {np.std(ann_query_times):.4f}s")
    print(f"  Cache Peak Mem: {peak_mem / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
