"""
SCIP Experiment - Spark with Custom Exact/Approximate KMeans

Implements 3/3 optimizations using pure Spark:
1. Exact vs Approximate KMeans (custom implementation)
2. Distance reuse
3. TopK selection
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, LongType

# Import custom KMeans
sys.path.insert(0, os.path.dirname(__file__))
from spark_custom_kmeans_fixed import run_exact_kmeans_scip, run_approximate_kmeans_scip


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


def create_spark_session(app_name: str = "SCIP") -> SparkSession:
    """Create Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "64g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.maxResultSize", "8g") \
        .config("spark.local.dir", "/tmp/spark-tmp") \
        .getOrCreate()


def load_embeddings(embeddings_dir: str) -> np.ndarray:
    """Load precomputed embeddings."""
    path = os.path.join(embeddings_dir, "embeddings.npy")
    print(f"Loading embeddings from {path}...", flush=True)
    embeddings = np.load(path)
    print(f"  Shape: {embeddings.shape}", flush=True)
    return embeddings


def numpy_to_spark_df(spark: SparkSession, embeddings: np.ndarray):
    """Convert numpy embeddings to Spark DataFrame."""
    N, D = embeddings.shape
    data = [(int(i), Vectors.dense(embeddings[i].tolist())) for i in range(N)]
    
    schema = StructType([
        StructField("id", LongType(), False),
        StructField("features", VectorUDT(), False),
    ])
    
    return spark.createDataFrame(data, schema)


def scip_spark_baseline_exact(
    spark: SparkSession,
    embeddings: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, TimingResult]:
    """
    SCIP Baseline - Exact KMeans + recompute distances + full sort
    """
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    print("Baseline (Exact KMeans):", flush=True)
    
    # Step 1: Create DataFrame
    t0 = time.perf_counter()
    df = numpy_to_spark_df(spark, embeddings)
    df.cache()
    df.count()
    timing.add_step("CreateDF", time.perf_counter() - t0)
    print(f"  CreateDF: {time.perf_counter() - t0:.2f}s", flush=True)
    
    # Step 2: Exact KMeans
    t0 = time.perf_counter()
    centroids_np, results = run_exact_kmeans_scip(spark, df, K)
    timing.add_step("ExactKMeans", time.perf_counter() - t0)
    print(f"  ExactKMeans: {time.perf_counter() - t0:.2f}s", flush=True)
    
    # Extract results
    ids = np.array([r[0] for r in results])
    labels = np.array([r[1] for r in results])
    cluster_sizes = np.bincount(labels, minlength=K)
    
    # Step 3: Recompute distances (no reuse)
    t0 = time.perf_counter()
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        vec = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-12)
        distances[i] = 1.0 - np.dot(vec, centroids_np[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Step 4: Full sort by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    idx_by_size = np.argsort(size_per_point)
    prune_by_size = ids[idx_by_size[:size_quota]]
    timing.add_step("SortBySize", time.perf_counter() - t0)
    
    # Step 5: Full sort by distance
    t0 = time.perf_counter()
    prune_set = set(prune_by_size)
    remaining_mask = np.array([ids[i] not in prune_set for i in range(N)])
    remaining_ids = ids[remaining_mask]
    remaining_dist = distances[remaining_mask]
    
    order = np.argsort(-remaining_dist)
    dist_quota = total_prune - size_quota
    prune_by_dist = remaining_ids[order[:dist_quota]]
    timing.add_step("SortByDist", time.perf_counter() - t0)
    
    # Combine
    prune_all = set(prune_by_size) | set(prune_by_dist)
    keep_indices = np.array([i for i in ids if i not in prune_all])
    
    df.unpersist()
    
    return keep_indices, timing


def scip_spark_optimized_approx(
    spark: SparkSession,
    embeddings: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, TimingResult]:
    """
    SCIP Optimized - Approximate KMeans + distance reuse + TopK
    """
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    print("Optimized (Approximate KMeans):", flush=True)
    
    # Step 1: Create DataFrame
    t0 = time.perf_counter()
    df = numpy_to_spark_df(spark, embeddings)
    df.cache()
    df.count()
    timing.add_step("CreateDF", time.perf_counter() - t0)
    print(f"  CreateDF: {time.perf_counter() - t0:.2f}s", flush=True)
    
    # Step 2: Approximate KMeans
    t0 = time.perf_counter()
    centroids_np, results = run_approximate_kmeans_scip(spark, df, K, sample_ratio=0.5)
    timing.add_step("ApproxKMeans", time.perf_counter() - t0)
    print(f"  ApproxKMeans: {time.perf_counter() - t0:.2f}s", flush=True)
    
    # Extract results
    ids = np.array([r[0] for r in results])
    labels = np.array([r[1] for r in results])
    sims = np.array([r[2] for r in results])
    cluster_sizes = np.bincount(labels, minlength=K)
    
    # Step 3: Distance REUSE (from KMeans search)
    t0 = time.perf_counter()
    distances = 1.0 - sims
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Step 4: TopK by size
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    
    if size_quota > 0:
        idx_by_size = np.argpartition(size_per_point, size_quota)
    else:
        idx_by_size = np.argsort(size_per_point)
    prune_by_size = ids[idx_by_size[:size_quota]]
    timing.add_step("TopKBySize", time.perf_counter() - t0)
    
    # Step 5: TopK by distance
    t0 = time.perf_counter()
    prune_set = set(prune_by_size)
    remaining_mask = np.array([ids[i] not in prune_set for i in range(N)])
    remaining_ids = ids[remaining_mask]
    remaining_dist = distances[remaining_mask]
    
    dist_quota = total_prune - size_quota
    if dist_quota > 0 and len(remaining_ids) > 0:
        part_idx = np.argpartition(-remaining_dist, dist_quota)
        prune_by_dist = remaining_ids[part_idx[:dist_quota]]
    else:
        prune_by_dist = np.array([], dtype=np.int64)
    timing.add_step("TopKByDist", time.perf_counter() - t0)
    
    # Combine
    prune_all = set(prune_by_size) | set(prune_by_dist)
    keep_indices = np.array([i for i in ids if i not in prune_all])
    
    df.unpersist()
    
    return keep_indices, timing


def run_comparison(spark: SparkSession, embeddings: np.ndarray, n_runs: int = 1):
    """Run comparison."""
    cfg = Config()
    
    print(f"\n{'='*60}", flush=True)
    print(f"Spark Custom KMeans SCIP: N={len(embeddings)}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    baseline_times = []
    optimized_times = []
    keep_baseline = None
    keep_optimized = None
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...", flush=True)
        
        keep_b, timing_b = scip_spark_baseline_exact(spark, embeddings, cfg)
        baseline_times.append(timing_b.total_time)
        print(f"  Baseline total: {timing_b.total_time:.2f}s\n", flush=True)
        
        keep_o, timing_o = scip_spark_optimized_approx(spark, embeddings, cfg)
        optimized_times.append(timing_o.total_time)
        print(f"  Optimized total: {timing_o.total_time:.2f}s\n", flush=True)
        
        if run == 0:
            keep_baseline = keep_b
            keep_optimized = keep_o
    
    baseline_avg = np.mean(baseline_times)
    optimized_avg = np.mean(optimized_times)
    speedup = baseline_avg / optimized_avg
    overlap = len(set(keep_baseline) & set(keep_optimized)) / len(keep_baseline)
    
    print(f"\n{'='*60}", flush=True)
    print("Spark Custom KMeans Results:", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline:  {baseline_avg:.4f}s", flush=True)
    print(f"Optimized: {optimized_avg:.4f}s", flush=True)
    print(f"Speedup:   {speedup:.2f}x", flush=True)
    print(f"Overlap:   {overlap*100:.1f}%", flush=True)
    
    return {
        "baseline_time": baseline_avg,
        "optimized_time": optimized_avg,
        "speedup": speedup,
        "overlap": overlap,
        "n_samples": len(embeddings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, default="./experiments/aligned")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./experiments/spark")
    parser.add_argument("--max_samples", type=int, default=500000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    embeddings = load_embeddings(args.embeddings_dir)
    
    if args.max_samples and args.max_samples < len(embeddings):
        embeddings = embeddings[:args.max_samples]
        print(f"Limited to {len(embeddings)} samples", flush=True)
    
    spark = create_spark_session("SCIP_Custom_KMeans")
    
    try:
        results = run_comparison(spark, embeddings, args.n_runs)
        
        with open(os.path.join(args.output_dir, "results_custom_kmeans.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {args.output_dir}/results_custom_kmeans.json", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
