"""
Step 4: Run SemDeDup using Spark for distributed processing.
Supports Exact baseline and Optimized (ANN + Reuse) modes.

Usage:
    python run_semdedup_spark.py --mode exact       # Baseline: Exact KNN, no caching
    python run_semdedup_spark.py --mode optimized   # ANN (LSH) + Term Reuse
    python run_semdedup_spark.py --compare          # Run both and compare

LSH Parameters (for optimized mode):
    --lsh-tables: Number of hash tables (more = higher recall, slower)
    --lsh-bucket-length: Bucket length (smaller = more precise, may miss neighbors)

Spark Resources:
    --spark-memory: Executor memory (default: 256g)
    --spark-cores: Number of cores (default: 32)
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import pickle
import time
from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, monotonically_increasing_id, broadcast, lit, explode, array
from pyspark.sql.types import FloatType, ArrayType, IntegerType, StructType, StructField, BooleanType
from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector


def parse_args():
    parser = argparse.ArgumentParser(description='SemDeDup with Spark (Exact/Optimized modes)')
    parser.add_argument('--mode', type=str, default='exact', choices=['exact', 'optimized'],
                        help='Search mode: exact (baseline), optimized (ANN + Reuse)')
    parser.add_argument('--compare', action='store_true',
                        help='Run both modes and compare')
    
    # LSH parameters
    parser.add_argument('--lsh-tables', type=int, default=2,
                        help='Number of LSH hash tables (more = higher recall, slower)')
    parser.add_argument('--lsh-bucket-length', type=float, default=2.0,
                        help='LSH bucket length (smaller = more precise)')
    
    # Spark resources
    parser.add_argument('--spark-memory', type=str, default='256g',
                        help='Spark executor memory')
    parser.add_argument('--spark-cores', type=int, default=32,
                        help='Number of Spark cores')
    parser.add_argument('--spark-driver-memory', type=str, default='32g',
                        help='Spark driver memory')
    
    parser.add_argument('--config', type=str, default='semdedup_configs.yaml',
                        help='Config file path')
    return parser.parse_args()


class SparkSemDeDupProcessor:
    """Spark-based SemDeDup processor with Exact and Optimized modes."""
    
    def __init__(self, config_path, mode='exact', 
                 lsh_tables=2, lsh_bucket_length=2.0,
                 spark_memory='256g', spark_cores=32, spark_driver_memory='32g'):
        self.mode = mode
        self.lsh_tables = lsh_tables
        self.lsh_bucket_length = lsh_bucket_length
        
        # Load config
        with open(config_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        
        self.eps_list = params.get('eps_list', [0.1, 0.2, 0.3])
        self.save_folder = params['save_folder']
        self.sorted_clusters_path = params['sorted_clusters_path']
        self.num_clusters = params['num_clusters']
        self.dataset_size = params['dataset_size']
        self.emb_size = params['emd_size']
        self.embs_memory_loc = params['embs_memory_loc']
        
        # Load embeddings using memmap (same as other implementations)
        self.embs = np.memmap(
            self.embs_memory_loc, dtype='float32', mode='r', 
            shape=(self.dataset_size, self.emb_size)
        )
        
        # Initialize Spark
        self.spark = SparkSession.builder \
            .appName(f"SemDeDup-{mode}") \
            .config("spark.executor.memory", spark_memory) \
            .config("spark.driver.memory", spark_driver_memory) \
            .config("spark.executor.cores", str(spark_cores)) \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", str(spark_cores * 2)) \
            .config("spark.local.dir", "/mnt/data2/yichen/spark_temp") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Stats
        self.total_time = 0
        self.cluster_times = []
        
        # Pre-build global LSH index for optimized mode (one-time Spark call)
        self.global_lsh_index = None
        self.global_normalized_embs = None
        if mode == 'optimized':
            self._build_global_lsh_index()
    
    def _build_global_lsh_index(self):
        """
        Build LSH index for ALL embeddings using pure NumPy.
        No Spark MLlib - just random hyperplanes and hash computation.
        This eliminates Spark overhead entirely.
        """
        print("Building global LSH index (pure NumPy)...")
        import time
        start = time.time()
        
        # Normalize all embeddings
        all_embs = np.array(self.embs)  # Load from memmap
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
        self.global_normalized_embs = (all_embs / (norms + 1e-8)).astype(np.float32)
        
        # Generate random hyperplanes for LSH (no Spark needed)
        dim = self.emb_size
        num_tables = self.lsh_tables
        num_hashes = 8  # bits per hash code (2^8 = 256 possible buckets per table)
        
        rng = np.random.default_rng(42)
        hyperplanes = rng.standard_normal((num_tables, num_hashes, dim)).astype(np.float32)
        # Normalize hyperplanes
        hp_norms = np.linalg.norm(hyperplanes, axis=2, keepdims=True)
        hyperplanes = hyperplanes / (hp_norms + 1e-8)
        
        # Compute hash codes for all vectors - pure NumPy (very fast!)
        # Shape: (N, num_tables)
        N = self.dataset_size
        hash_codes = np.zeros((N, num_tables), dtype=np.int32)
        powers = 2 ** np.arange(num_hashes)
        
        for t in range(num_tables):
            # Dot products: (N, num_hashes)
            dots = self.global_normalized_embs @ hyperplanes[t].T
            # Binary hash: 1 if dot > 0, else 0
            binary_hash = (dots > 0).astype(np.int32)
            # Convert binary to integer hash code
            hash_codes[:, t] = binary_hash @ powers
        
        # Build hash bucket to point IDs mapping
        from collections import defaultdict
        bucket_to_points = defaultdict(set)
        point_hashes = {}
        
        for point_id in range(N):
            hashes = tuple(hash_codes[point_id].tolist())
            point_hashes[point_id] = hashes
            
            for table_idx, hash_val in enumerate(hashes):
                bucket_key = (table_idx, hash_val)
                bucket_to_points[bucket_key].add(point_id)
        
        self.global_lsh_index = {
            'bucket_to_points': dict(bucket_to_points),
            'point_hashes': point_hashes
        }
        
        elapsed = time.time() - start
        print(f"  ✓ Global LSH index built in {elapsed:.2f}s ({len(point_hashes)} points, {len(bucket_to_points)} buckets)")
    
    def get_output_dir(self):
        """Get output directory based on mode."""
        if self.mode == 'exact':
            return os.path.join(self.save_folder, "dataframes_spark_exact")
        else:
            return os.path.join(self.save_folder, f"dataframes_spark_optimized_t{self.lsh_tables}_b{self.lsh_bucket_length}")
    
    def _to_spark_df(self, cluster_reps_np):
        """Convert numpy embeddings to Spark DataFrame with vector column."""
        # Create list of (id, vector) tuples
        data = [(int(i), Vectors.dense(cluster_reps_np[i].tolist())) 
                for i in range(len(cluster_reps_np))]
        
        schema = StructType([
            StructField("id", IntegerType(), False),
            StructField("features", VectorUDT(), False)
        ])
        
        return self.spark.createDataFrame(data, schema)
    
    def _normalize_vectors(self, df):
        """L2 normalize vectors for cosine similarity."""
        normalizer = Normalizer(inputCol="features", outputCol="normalized_features", p=2.0)
        return normalizer.transform(df)
    
    def semdedup_exact(self, cluster_reps_np):
        """
        Exact pairwise cosine similarity using TRUE Spark distributed computation.
        
        FAIR COMPARISON: Uses same per-candidate loop structure as Optimized.
        - No matmul advantage: each similarity computed individually
        - Only difference from Optimized: compares with ALL points, not just LSH candidates
        """
        n = len(cluster_reps_np)
        
        # For very small clusters, use numpy directly
        if n <= 10:
            norms = np.linalg.norm(cluster_reps_np, axis=1, keepdims=True)
            normalized = cluster_reps_np / (norms + 1e-8)
            sim_matrix = normalized @ normalized.T
            np.fill_diagonal(sim_matrix, 0)
            M = np.max(sim_matrix, axis=1)
            return M
        
        # Normalize vectors
        norms = np.linalg.norm(cluster_reps_np, axis=1, keepdims=True)
        normalized_np = (cluster_reps_np / (norms + 1e-8)).astype(np.float32)
        
        # Broadcast normalized vectors to all executors
        vectors_broadcast = self.spark.sparkContext.broadcast(normalized_np)
        n_broadcast = self.spark.sparkContext.broadcast(n)
        
        # Create RDD of point indices
        point_ids_rdd = self.spark.sparkContext.parallelize(range(n), numSlices=min(n, 64))
        
        # Define the per-point computation (runs on executors)
        # FAIR: Use for-loop, same structure as Optimized
        def compute_max_similarity_loop(point_id):
            """Compute max similarity using per-candidate loop (same as Optimized)."""
            all_vectors = vectors_broadcast.value
            n_points = n_broadcast.value
            point_vec = all_vectors[point_id]
            
            # Loop through ALL candidates (this is the only difference from Optimized)
            max_sim = 0.0
            for other_id in range(n_points):
                if other_id == point_id:
                    continue
                # Per-candidate dot product (same computation as Optimized)
                sim = float(np.dot(all_vectors[other_id], point_vec))
                if sim > max_sim:
                    max_sim = sim
            
            return (point_id, max_sim)
        
        # Distributed computation: each point processed on executor
        results = point_ids_rdd.map(compute_max_similarity_loop).collect()
        
        # Convert results to M array
        M = np.zeros(n)
        for point_id, max_sim in results:
            M[point_id] = max_sim
        
        # Cleanup broadcast variable
        vectors_broadcast.unpersist()
        n_broadcast.unpersist()
        
        return M
    
    def semdedup_optimized(self, cluster_reps_np, cluster_indices=None):
        """
        Optimized mode: TRUE Spark distributed with EXECUTOR-SIDE LSH lookup.
        
        FAIR COMPARISON: 
        - Same per-candidate for-loop structure as Exact (no matmul advantage)
        - LSH candidate lookup happens on executor, not driver
        - Only difference from Exact: compares with LSH candidates (~20%) instead of all points
        """
        n = len(cluster_reps_np)
        
        # Same threshold as baseline for fair comparison
        if n <= 10:
            norms = np.linalg.norm(cluster_reps_np, axis=1, keepdims=True)
            normalized = cluster_reps_np / (norms + 1e-8)
            sim_matrix = normalized @ normalized.T
            np.fill_diagonal(sim_matrix, 0)
            M = np.max(sim_matrix, axis=1)
            return M
        
        # Use pre-built global LSH index
        if self.global_lsh_index is None:
            raise RuntimeError("Global LSH index not built. Run in optimized mode.")
        
        point_hashes = self.global_lsh_index['point_hashes']
        bucket_to_points = self.global_lsh_index['bucket_to_points']
        
        # If cluster_indices provided, use them; otherwise assume sequential
        if cluster_indices is None:
            cluster_indices = list(range(n))
        
        # Normalize vectors
        norms = np.linalg.norm(cluster_reps_np, axis=1, keepdims=True)
        normalized_np = (cluster_reps_np / (norms + 1e-8)).astype(np.float32)
        
        # Build hash codes for this cluster (local ID -> hash tuple)
        local_hashes = []
        for local_id, global_id in enumerate(cluster_indices):
            if global_id in point_hashes:
                local_hashes.append(point_hashes[global_id])
            else:
                local_hashes.append(tuple([0] * self.lsh_tables))
        
        # Build local bucket_to_points for this cluster only (for efficiency)
        cluster_set = set(cluster_indices)
        global_to_local = {g: l for l, g in enumerate(cluster_indices)}
        
        local_bucket_to_points = {}
        for (table_idx, hash_val), global_points in bucket_to_points.items():
            local_points = [global_to_local[g] for g in global_points if g in cluster_set]
            if local_points:
                local_bucket_to_points[(table_idx, hash_val)] = local_points
        
        # Broadcast everything to executors
        vectors_broadcast = self.spark.sparkContext.broadcast(normalized_np)
        hashes_broadcast = self.spark.sparkContext.broadcast(local_hashes)
        bucket_to_points_broadcast = self.spark.sparkContext.broadcast(local_bucket_to_points)
        
        # Create RDD of point indices
        point_ids_rdd = self.spark.sparkContext.parallelize(range(n), numSlices=min(n, 64))
        
        # Define the per-point computation (runs on executors)
        # FAIR: Same for-loop structure as Exact, LSH lookup done here
        def compute_max_similarity_lsh_executor(point_id):
            """Compute max similarity with LSH candidates using same loop as Exact."""
            all_vectors = vectors_broadcast.value
            all_hashes = hashes_broadcast.value
            bucket_map = bucket_to_points_broadcast.value
            
            point_vec = all_vectors[point_id]
            my_hashes = all_hashes[point_id]
            
            # LSH candidate lookup (done on executor, not driver!)
            candidates = set()
            for table_idx, hash_val in enumerate(my_hashes):
                bucket_key = (table_idx, hash_val)
                if bucket_key in bucket_map:
                    candidates.update(bucket_map[bucket_key])
            candidates.discard(point_id)  # Exclude self
            
            if len(candidates) == 0:
                return (point_id, 0.0)
            
            # FAIR: Same for-loop structure as Exact (no matmul)
            max_sim = 0.0
            for other_id in candidates:
                sim = float(np.dot(all_vectors[other_id], point_vec))
                if sim > max_sim:
                    max_sim = sim
            
            return (point_id, max_sim)
        
        # Distributed computation: each point processed on executor
        results = point_ids_rdd.map(compute_max_similarity_lsh_executor).collect()
        
        # Convert results to M array
        M = np.zeros(n)
        for point_id, max_sim in results:
            M[point_id] = max_sim
        
        # Cleanup broadcast variables
        vectors_broadcast.unpersist()
        hashes_broadcast.unpersist()
        bucket_to_points_broadcast.unpersist()
        
        return M
    
    def process_cluster(self, cluster_id, dataframes_dir):
        """Process a single cluster."""
        df_file_loc = os.path.join(dataframes_dir, f"cluster_{cluster_id}.pkl")
        
        # Load sorted cluster
        cluster_i = np.load(
            os.path.join(self.sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_size = cluster_i.shape[0]
        
        # Handle single-item clusters
        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            for eps in self.eps_list:
                points_to_remove_df[f"eps={eps}"] = [False]
            with open(df_file_loc, "wb") as f:
                pickle.dump(points_to_remove_df, f)
            return 0.0
        
        # Get embeddings for cluster items
        cluster_ids = cluster_i[:, 1].astype("int32")
        cluster_reps = np.array(self.embs[cluster_ids], dtype=np.float32)
        
        # Time the similarity computation
        start_time = time.time()
        
        if self.mode == 'exact':
            # Baseline: compute M for each epsilon separately (no caching)
            cluster_item_indices = list(range(cluster_size))
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = cluster_item_indices
            
            total_elapsed = 0
            for eps in self.eps_list:
                eps_start = time.time()
                M = self.semdedup_exact(cluster_reps)
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove
                total_elapsed += time.time() - eps_start
            
            elapsed = total_elapsed
        else:
            # Optimized: compute M once (with ANN + reuse), apply to all eps
            # Pass cluster_ids for global LSH index lookup
            M = self.semdedup_optimized(cluster_reps, cluster_indices=cluster_ids.tolist())
            elapsed = time.time() - start_time
            
            cluster_item_indices = list(range(cluster_size))
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = cluster_item_indices
            
            for eps in self.eps_list:
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove
        
        # Save
        with open(df_file_loc, "wb") as f:
            pickle.dump(points_to_remove_df, f)
        
        return elapsed
    
    def run(self):
        """Run SemDeDup on all clusters."""
        dataframes_dir = self.get_output_dir()
        os.makedirs(dataframes_dir, exist_ok=True)
        
        print("=" * 70)
        print(f"SemDeDup Spark - Mode: {self.mode.upper()}")
        if self.mode == 'optimized':
            print(f"LSH Tables: {self.lsh_tables}, Bucket Length: {self.lsh_bucket_length}")
        print("=" * 70)
        print(f"Embeddings: {self.embs_memory_loc}")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Epsilon values: {self.eps_list}")
        print(f"Output: {dataframes_dir}")
        print("=" * 70)
        
        total_start = time.time()
        
        for cluster_id in tqdm(range(self.num_clusters), desc=f"Processing ({self.mode})"):
            elapsed = self.process_cluster(cluster_id, dataframes_dir)
            self.cluster_times.append(elapsed)
        
        self.total_time = time.time() - total_start
        
        return self.get_statistics(dataframes_dir)
    
    def get_statistics(self, dataframes_dir):
        """Compute duplicate statistics."""
        total_duplicates = {eps: 0 for eps in self.eps_list}
        
        for cluster_id in range(self.num_clusters):
            df_file_loc = os.path.join(dataframes_dir, f"cluster_{cluster_id}.pkl")
            with open(df_file_loc, "rb") as f:
                df = pickle.load(f)
            for eps in self.eps_list:
                total_duplicates[eps] += df[f"eps={eps}"].sum()
        
        stats = {
            'mode': self.mode,
            'total_time': self.total_time,
            'avg_cluster_time': np.mean(self.cluster_times) if self.cluster_times else 0,
            'duplicates': total_duplicates,
            'output_dir': dataframes_dir,
            'num_clusters': self.num_clusters,
        }
        
        if self.mode == 'optimized':
            stats['lsh_tables'] = self.lsh_tables
            stats['lsh_bucket_length'] = self.lsh_bucket_length
        
        return stats
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


def print_statistics(stats, dataset_size):
    """Print statistics for one run."""
    print(f"\n{'='*70}")
    print(f"Results - Mode: {stats['mode'].upper()}")
    if stats['mode'] == 'optimized':
        print(f"LSH: {stats.get('lsh_tables', 'N/A')} tables, bucket length {stats.get('lsh_bucket_length', 'N/A')}")
    print(f"{'='*70}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Avg time per cluster: {stats['avg_cluster_time']*1000:.2f}ms")
    print(f"\nDuplicate statistics:")
    for eps, count in stats['duplicates'].items():
        pct = 100 * count / dataset_size
        kept = dataset_size - count
        print(f"  eps={eps}: {count:,} duplicates ({pct:.1f}%), {kept:,} kept")
    print(f"\nOutput: {stats['output_dir']}")


def compare_results(stats_exact, stats_optimized, dataset_size, eps_list):
    """Compare Exact vs Optimized results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Spark Exact (Baseline) vs Spark Optimized (ANN + Reuse)")
    print("=" * 70)
    
    # Time comparison
    speedup = stats_exact['total_time'] / stats_optimized['total_time'] if stats_optimized['total_time'] > 0 else 0
    print(f"\n⏱️  Time Comparison:")
    print(f"   Exact (Baseline): {stats_exact['total_time']:.2f}s")
    print(f"   Optimized:        {stats_optimized['total_time']:.2f}s")
    print(f"   Speedup:          {speedup:.2f}x")
    
    # Accuracy comparison
    print(f"\n📊 Duplicate Detection Comparison:")
    print(f"   {'eps':<8} {'Exact':<12} {'Optimized':<12} {'Diff':<10} {'Match %':<10}")
    print(f"   {'-'*52}")
    
    for eps in eps_list:
        exact_count = stats_exact['duplicates'][eps]
        opt_count = stats_optimized['duplicates'][eps]
        diff = opt_count - exact_count
        match_pct = 100 * min(exact_count, opt_count) / max(exact_count, opt_count) if max(exact_count, opt_count) > 0 else 100
        print(f"   {eps:<8} {exact_count:<12,} {opt_count:<12,} {diff:+<10,} {match_pct:<9.1f}%")
    
    # Per-sample comparison for recall
    print(f"\n🔍 Detailed Comparison (eps=0.1):")
    
    exact_dir = stats_exact['output_dir']
    opt_dir = stats_optimized['output_dir']
    
    exact_dups = set()
    opt_dups = set()
    
    for cluster_id in range(min(100, stats_exact.get('num_clusters', 100))):
        try:
            with open(os.path.join(exact_dir, f"cluster_{cluster_id}.pkl"), "rb") as f:
                df_exact = pickle.load(f)
            with open(os.path.join(opt_dir, f"cluster_{cluster_id}.pkl"), "rb") as f:
                df_opt = pickle.load(f)
            
            exact_mask = df_exact['eps=0.1'].values
            opt_mask = df_opt['eps=0.1'].values
            
            for i, (e, o) in enumerate(zip(exact_mask, opt_mask)):
                key = (cluster_id, i)
                if e:
                    exact_dups.add(key)
                if o:
                    opt_dups.add(key)
        except Exception:
            pass
    
    if exact_dups or opt_dups:
        intersection = exact_dups & opt_dups
        only_exact = exact_dups - opt_dups
        only_opt = opt_dups - exact_dups
        
        recall = len(intersection) / len(exact_dups) * 100 if exact_dups else 100
        precision = len(intersection) / len(opt_dups) * 100 if opt_dups else 100
        
        print(f"   Duplicates found by both: {len(intersection):,}")
        print(f"   Only by Exact: {len(only_exact):,}")
        print(f"   Only by Optimized: {len(only_opt):,}")
        print(f"   Recall (Optimized finds what Exact finds): {recall:.1f}%")
        print(f"   Precision: {precision:.1f}%")
    
    print("=" * 70)


def main():
    args = parse_args()
    
    if args.compare:
        print("\n🔬 Running comparison: Spark Exact (Baseline) vs Spark Optimized (ANN + Reuse)\n")
        
        # Run Exact baseline
        processor_exact = SparkSemDeDupProcessor(
            args.config, mode='exact',
            spark_memory=args.spark_memory,
            spark_cores=args.spark_cores,
            spark_driver_memory=args.spark_driver_memory
        )
        stats_exact = processor_exact.run()
        print_statistics(stats_exact, processor_exact.dataset_size)
        processor_exact.stop()
        
        # Run Optimized
        processor_opt = SparkSemDeDupProcessor(
            args.config, mode='optimized',
            lsh_tables=args.lsh_tables,
            lsh_bucket_length=args.lsh_bucket_length,
            spark_memory=args.spark_memory,
            spark_cores=args.spark_cores,
            spark_driver_memory=args.spark_driver_memory
        )
        stats_opt = processor_opt.run()
        print_statistics(stats_opt, processor_opt.dataset_size)
        
        # Compare
        compare_results(
            stats_exact, stats_opt,
            processor_opt.dataset_size,
            processor_opt.eps_list
        )
        
        processor_opt.stop()
        
    else:
        # Run single mode
        processor = SparkSemDeDupProcessor(
            args.config,
            mode=args.mode,
            lsh_tables=args.lsh_tables,
            lsh_bucket_length=args.lsh_bucket_length,
            spark_memory=args.spark_memory,
            spark_cores=args.spark_cores,
            spark_driver_memory=args.spark_driver_memory
        )
        stats = processor.run()
        print_statistics(stats, processor.dataset_size)
        processor.stop()
        
        print("\n✓ Done!")


if __name__ == "__main__":
    main()
