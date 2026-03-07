"""
Simplified Milvus SCIP Experiment - Based on Working Test Pattern

Uses the exact pattern that successfully worked with 453K embeddings.
"""

import os, json, time, argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict
from pymilvus import MilvusClient
from pymilvus.milvus_client import IndexParams
from sklearn.cluster import MiniBatchKMeans


@dataclass
class TimingResult:
    step_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    def add_step(self, name: str, elapsed: float):
        self.step_times[name] = elapsed
        self.total_time += elapsed


@dataclass
class Config:
    n_clusters: int = 1000
    p: float = 0.2
    alpha: float = 0.8


def run_milvus_scip(embeddings: np.ndarray, cfg: Config, db_path: str, version: str) -> Tuple[np.ndarray, TimingResult]:
    """Run SCIP with Milvus - using proven pattern."""
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
    
    
    # Milvus setup with explicit index type
    t0 = time.perf_counter()
    client = MilvusClient(uri=db_path)
    
    if version == "baseline":
        # Baseline: FLAT index (exact search)
        client.create_collection('centroids', dimension=D, metric_type='IP')
    else:
        # Optimized: IVF index (ANN search)
        # For small centroid set (100), use IVF with appropriate parameters
        client.create_collection(
            collection_name='centroids',
            dimension=D,
            metric_type='IP',
        )
    
    time.sleep(1)
    
    # Insert centroids
    centroid_data = [{'id': i, 'vector': centroids[i].astype(np.float32).tolist()} for i in range(K)]
    client.insert('centroids', centroid_data)
    
    # Create index after insertion - explicit for both versions
    index_params = IndexParams()
    
    if version == "optimized":
        # Use IVF_FLAT for ANN - nlist based on K
        nlist = max(4, min(K // 2, 16))  # Between 4-16 for K=100
        index_params.add_index(
            field_name='vector',
            index_type='IVF_FLAT',
            metric_type='IP',
            params={'nlist': nlist}
        )
    else:
        # Baseline: Explicitly create FLAT index (exact search)
        index_params.add_index(
            field_name='vector',
            index_type='FLAT',
            metric_type='IP',
        )
    
    client.create_index(collection_name='centroids', index_params=index_params)
    
    # Load collection to memory
    client.load_collection('centroids')
    
    timing.add_step("MilvusSetup", time.perf_counter() - t0)
    
    # Centroid search
    t0 = time.perf_counter()
    batch_size = 1000
    all_labels, all_sims = [], []
    
    search_params = {}
    if version == "optimized":
        # For IVF, set nprobe
        search_params = {'nprobe': max(2, nlist // 2)}
    
    for i in range(0, N, batch_size):
        batch = embeddings[i:i+batch_size]
        if version == "optimized" and search_params:
            results = client.search(
                'centroids', 
                data=batch.tolist(), 
                limit=1, 
                output_fields=["id"],
                search_params=search_params
            )
        else:
            results = client.search('centroids', data=batch.tolist(), limit=1, output_fields=["id"])
        for r in results:
            all_labels.append(r[0]["id"])
            all_sims.append(r[0]["distance"])
    labels = np.array(all_labels)
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances (reuse for optimized)
    t0 = time.perf_counter()
    distances = 1.0 - np.array(all_sims) if version == "optimized" else np.zeros(N, dtype=np.float32)
    if version == "baseline":
        for i in range(N):
            distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Sort/TopK by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    
    if version == "optimized" and size_quota > 0:
        idx_by_size = np.argpartition(size_per_point, size_quota)
    else:
        idx_by_size = np.argsort(size_per_point)
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("SortBySize" if version == "baseline" else "TopKBySize", time.perf_counter() - t0)
    
    # Sort/TopK by distance
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    dist_remaining = distances[remaining]
    dist_quota = total_prune - size_quota
    
    if version == "optimized" and dist_quota > 0 and len(remaining) > 0:
        part_idx = np.argpartition(-dist_remaining, dist_quota)
        prune_by_dist = remaining[part_idx[:dist_quota]]
    else:
        order = np.argsort(-dist_remaining)
        prune_by_dist = remaining[order[:dist_quota]]
    timing.add_step("SortByDist" if version == "baseline" else "TopKByDist", time.perf_counter() - t0)
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    client.close()
    return keep_indices, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, default="./experiments/aligned")
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./experiments/milvus")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config()
    
    # Load embeddings
    emb_path = os.path.join(args.embeddings_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        emb_path = "./embeddings/codesearchnet/embeddings.npy"
    print(f"Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    print(f"  Shape: {embeddings.shape}")
    
    print(f"\n{'='*60}")
    print(f"Milvus SCIP Comparison: N={len(embeddings)}")
    print(f"{'='*60}\n")
    
    baseline_times, optimized_times = [], []
    keep_baseline = keep_optimized = None
    
    for run in range(args.n_runs):
        print(f"Run {run+1}/{args.n_runs}...")
        
        db_b = os.path.abspath(os.path.join(args.output_dir, f"baseline_r{run}.db"))
        if os.path.exists(db_b): os.remove(db_b)
        keep_b, timing_b = run_milvus_scip(embeddings, cfg, db_b, "baseline")
        baseline_times.append(timing_b.total_time)
        print(f"  Baseline: {timing_b.total_time:.2f}s")
        
        db_o = os.path.abspath(os.path.join(args.output_dir, f"optimized_r{run}.db"))
        if os.path.exists(db_o): os.remove(db_o)
        keep_o, timing_o = run_milvus_scip(embeddings, cfg, db_o, "optimized")
        optimized_times.append(timing_o.total_time)
        print(f"  Optimized: {timing_o.total_time:.2f}s")
        
        if run == 0:
            keep_baseline, keep_optimized = keep_b, keep_o
    
    baseline_avg = np.mean(baseline_times)
    optimized_avg = np.mean(optimized_times)
    speedup = baseline_avg / optimized_avg
    overlap = len(set(keep_baseline) & set(keep_optimized)) / len(keep_baseline)
    
    print(f"\n{'='*60}")
    print("Milvus Results:")
    print(f"{'='*60}")
    print(f"Baseline:  {baseline_avg:.4f}s")
    print(f"Optimized: {optimized_avg:.4f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Overlap:   {overlap*100:.1f}%")
    
    results = {
        "baseline_time": baseline_avg,
        "optimized_time": optimized_avg,
        "speedup": speedup,
        "overlap": overlap,
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
