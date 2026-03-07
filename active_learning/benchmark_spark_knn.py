#!/usr/bin/env python
"""
Single Iteration CAL Benchmark with Real GPU Operations

Simulates only the LAST iteration of AL loop (the 137.25s vs 65.26s measurement):
- Uses real BERT embeddings dimension (768)
- Real GPU tensor operations (softmax, KL divergence)
- Same data sizes as final iteration (~51K unlabeled, ~9K labeled)
- Matches the exact CAL loop timing from cal.py

No model training required - just CAL selection loop timing.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pyspark.sql import SparkSession
import os


class LRUCache:
    """LRU Cache for probability caching"""
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif self.max_size is not None and len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.access_order.append(key)


def run_spark_knn(spark, xb, xq, k, method='exact', num_tables=5, bucket_width=2.0, num_partitions=64):
    """Run Spark KNN search (same as before)"""
    build_start = time.time()
    
    if method == "lsh":
        d = xb.shape[1]
        np.random.seed(42)
        hash_matrices = []
        num_projections = max(1, int(d / 16))
        for _ in range(num_tables):
            random_matrix = np.random.randn(num_projections, d).astype('float32')
            hash_matrices.append(random_matrix)
        
        broadcast_hash_matrices = spark.sparkContext.broadcast(hash_matrices)
        broadcast_xb = spark.sparkContext.broadcast(xb)
        broadcast_k = spark.sparkContext.broadcast(k)
        broadcast_bucket_width = spark.sparkContext.broadcast(bucket_width)
        
        build_time = time.time() - build_start
        search_start = time.time()
        
        query_rdd = spark.sparkContext.parallelize(
            [(int(i), xq[i].tolist()) for i in range(len(xq))],
            numSlices=num_partitions
        )
        
        def lsh_search_partition(iterator):
            import numpy as _np
            train_data = broadcast_xb.value
            hash_mats = broadcast_hash_matrices.value
            num_neighbors = broadcast_k.value
            bw = broadcast_bucket_width.value
            
            train_hash_codes = []
            for hash_mat in hash_mats:
                projections = train_data @ hash_mat.T
                hash_codes = _np.floor(projections / bw).astype('int32')
                train_hash_codes.append(hash_codes)
            
            results = []
            for query_id, query_vec in iterator:
                query = _np.array(query_vec, dtype='float32')
                candidate_set = set()
                for table_idx, hash_mat in enumerate(hash_mats):
                    query_proj = query @ hash_mat.T
                    query_hash = _np.floor(query_proj / bw).astype('int32')
                    train_hashes = train_hash_codes[table_idx]
                    matches = _np.all(train_hashes == query_hash, axis=1)
                    candidate_indices = _np.where(matches)[0]
                    candidate_set.update(candidate_indices.tolist())
                
                if len(candidate_set) == 0:
                    candidate_set = set(_np.random.choice(len(train_data), 
                                                         min(num_neighbors * 10, len(train_data)), 
                                                         replace=False))
                
                candidates = list(candidate_set)
                candidate_vectors = train_data[candidates]
                distances = _np.linalg.norm(candidate_vectors - query, axis=1)
                
                if len(distances) <= num_neighbors:
                    top_k_idx = _np.arange(len(distances))
                else:
                    top_k_idx = _np.argpartition(distances, num_neighbors)[:num_neighbors]
                top_k_idx = top_k_idx[_np.argsort(distances[top_k_idx])]
                top_k_indices = [candidates[i] for i in top_k_idx]
                
                if len(top_k_indices) < num_neighbors:
                    top_k_indices = top_k_indices + [-1] * (num_neighbors - len(top_k_indices))
                
                results.append((query_id, top_k_indices[:num_neighbors]))
            return iter(results)
        
        result_rdd = query_rdd.mapPartitions(lsh_search_partition)
        results = result_rdd.collect()
        broadcast_hash_matrices.unpersist()
        broadcast_xb.unpersist()
        broadcast_k.unpersist()
        broadcast_bucket_width.unpersist()
    else:
        broadcast_xb = spark.sparkContext.broadcast(xb)
        broadcast_k = spark.sparkContext.broadcast(k)
        build_time = time.time() - build_start
        search_start = time.time()
        
        query_rdd = spark.sparkContext.parallelize(
            [(int(i), xq[i].tolist()) for i in range(len(xq))],
            numSlices=num_partitions
        )
        
        def compute_knn_partition(iterator):
            import numpy as _np
            train_data = broadcast_xb.value
            num_neighbors = broadcast_k.value
            results = []
            for query_id, query_vec in iterator:
                query = _np.array(query_vec, dtype='float32')
                distances = _np.linalg.norm(train_data - query, axis=1)
                if len(distances) <= num_neighbors:
                    top_k_indices = _np.arange(len(distances))
                else:
                    top_k_indices = _np.argpartition(distances, num_neighbors)[:num_neighbors]
                    top_k_indices = top_k_indices[_np.argsort(distances[top_k_indices])]
                results.append((query_id, top_k_indices.tolist()))
            return iter(results)
        
        result_rdd = query_rdd.mapPartitions(compute_knn_partition)
        results = result_rdd.collect()
        broadcast_xb.unpersist()
        broadcast_k.unpersist()
    
    search_time = time.time() - search_start
    results.sort(key=lambda x: x[0])
    neighbours_all = np.array([r[1] for r in results], dtype='int64')
    
    return {'build_time': build_time, 'search_time': search_time, 'neighbours': neighbours_all}


def run_cal_loop(neighbours_all, train_logits, pool_logits, num_classes=2, use_cache=True, device='cuda'):
    """
    Run the actual CAL selection loop with real GPU tensor operations.
    This matches the timing from cal.py.
    """
    num_queries = len(neighbours_all)
    num_labeled = train_logits.shape[0]
    k = neighbours_all.shape[1]
    
    # Move to GPU
    train_logits = train_logits.to(device)
    pool_logits = pool_logits.to(device)
    
    # Pre-compute log probs for pool (always done)
    pool_log_probs = F.log_softmax(pool_logits, dim=-1)
    
    # Setup cache
    if use_cache:
        train_probs_cache = LRUCache(max_size=None)
    else:
        train_probs_cache = None
    
    # Timing breakdown
    time_mapping = 0
    time_data_prep = 0
    time_list_comprehension = 0
    time_prob_cache = 0
    time_kl_compute = 0
    time_kl_aggregate = 0
    time_pred_stats = 0
    
    kl_scores = []
    num_adv = 0
    
    # Create fake train_dataset.tensors[3] (labels)
    fake_labels = torch.randint(0, num_classes, (num_labeled,))
    
    loop_start = time.time()
    
    for unlab_i in tqdm(range(num_queries), desc="Finding neighbours for every unlabeled data point", mininterval=0.5):
        # Step 1: Index mapping
        step_start = time.time()
        neighbours = neighbours_all[unlab_i:unlab_i+1]
        time_mapping += time.time() - step_start
        
        # Step 2: Data preparation (simulate tensor indexing)
        step_start = time.time()
        valid_neighbors = [n for n in neighbours[0] if 0 <= n < num_labeled]
        if len(valid_neighbors) == 0:
            valid_neighbors = [0]
        labeled_neighbours_labels = fake_labels[valid_neighbors]
        # Simulate labeled_inds indexing
        labeled_neighbours_inds = np.array(valid_neighbors)
        time_data_prep += time.time() - step_start
        
        # Step 3: List comprehensions for logits
        step_start = time.time()
        logits_neigh = [train_logits[n] for n in valid_neighbors]
        preds_neigh = [torch.argmax(train_logits[n]).item() for n in valid_neighbors]
        time_list_comprehension += time.time() - step_start
        
        # Step 4: Probability cache/compute
        step_start = time.time()
        if use_cache and train_probs_cache is not None:
            neigh_prob_list = []
            for n_idx in valid_neighbors:
                cached_val = train_probs_cache.get(n_idx)
                if cached_val is None:
                    prob = F.softmax(train_logits[n_idx:n_idx+1], dim=-1)
                    train_probs_cache.put(n_idx, prob)
                    neigh_prob_list.append(prob)
                else:
                    neigh_prob_list.append(cached_val)
            neigh_prob = torch.cat(neigh_prob_list, dim=0)
        else:
            neigh_prob = F.softmax(train_logits[valid_neighbors], dim=-1)
        time_prob_cache += time.time() - step_start
        
        # Step 5: Prediction stats
        step_start = time.time()
        pred_candidate = torch.argmax(pool_logits[unlab_i]).item()
        if pred_candidate in preds_neigh:
            num_adv += 1
        time_pred_stats += time.time() - step_start
        
        # Step 6: KL divergence computation
        step_start = time.time()
        candidate_log_prob = pool_log_probs[unlab_i:unlab_i+1].expand(len(neigh_prob), -1)
        kl = torch.sum(F.kl_div(candidate_log_prob, neigh_prob, reduction='none'), dim=-1)
        time_kl_compute += time.time() - step_start
        
        # Step 7: KL aggregation
        step_start = time.time()
        kl_scores.append(kl.mean().item())
        time_kl_aggregate += time.time() - step_start
    
    loop_time = time.time() - loop_start
    
    cache_stats = {
        'hits': train_probs_cache.hits if train_probs_cache else 0,
        'misses': train_probs_cache.misses if train_probs_cache else num_queries * k
    }
    
    timing = {
        'loop_time': loop_time,
        'time_mapping': time_mapping,
        'time_data_prep': time_data_prep,
        'time_list_comprehension': time_list_comprehension,
        'time_prob_cache': time_prob_cache,
        'time_kl_compute': time_kl_compute,
        'time_kl_aggregate': time_kl_aggregate,
        'time_pred_stats': time_pred_stats
    }
    
    return timing, cache_stats


def main():
    parser = argparse.ArgumentParser(description='Single Iteration CAL Benchmark')
    parser.add_argument('--num_queries', type=int, default=51523, help='Unlabeled pool size')
    parser.add_argument('--num_candidates', type=int, default=9090, help='Labeled pool size')
    parser.add_argument('--dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--num_tables', type=int, default=5, help='LSH hash tables')
    parser.add_argument('--bucket_width', type=float, default=2.0, help='LSH bucket width')
    parser.add_argument('--partitions', type=int, default=64, help='Spark partitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = parser.parse_args()
    
    # Setup
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("Single Iteration CAL Benchmark (Real GPU Operations)")
    print("=" * 70)
    print(f"Queries: {args.num_queries}, Labeled: {args.num_candidates}")
    print(f"Dim: {args.dim}, Classes: {args.num_classes}, K: {args.k}")
    print(f"Device: {device}, Runs: {args.runs}")
    print("=" * 70)
    
    # Generate simulated data
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Embeddings (normalized like real BERT embeddings)
    xb = np.random.randn(args.num_candidates, args.dim).astype('float32')
    xq = np.random.randn(args.num_queries, args.dim).astype('float32')
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    xq = xq / np.linalg.norm(xq, axis=1, keepdims=True)
    
    # Logits (simulating model output)
    train_logits = torch.randn(args.num_candidates, args.num_classes)
    pool_logits = torch.randn(args.num_queries, args.num_classes)
    
    print(f"Data: xb={xb.shape}, xq={xq.shape}, logits={train_logits.shape}")
    
    exact_times = []
    lsh_times = []
    
    for run in range(args.runs):
        print(f"\n{'='*70}")
        print(f"Run {run + 1}/{args.runs}")
        print("=" * 70)
        
        # ========== Spark Exact + No Cache ==========
        print("\n[1/2] Spark Exact + No Cache (baseline)...")
        knn_total_start = time.time()
        
        spark = SparkSession.builder.appName("CAL_Exact").config("spark.driver.memory", "8g").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        knn_result = run_spark_knn(spark, xb, xq, args.k, method='exact', num_partitions=args.partitions)
        spark.stop()
        
        cal_timing, cal_stats = run_cal_loop(
            knn_result['neighbours'], train_logits, pool_logits, 
            args.num_classes, use_cache=False, device=device
        )
        
        knn_total_time = time.time() - knn_total_start
        exact_times.append(knn_total_time)
        
        print(f"  Spark Build: {knn_result['build_time']:.2f}s")
        print(f"  Spark Search: {knn_result['search_time']:.2f}s")
        print(f"  CAL Loop: {cal_timing['loop_time']:.2f}s")
        print(f"    - Data prep: {cal_timing['time_data_prep']:.2f}s")
        print(f"    - KL compute: {cal_timing['time_kl_compute']:.2f}s")
        print(f"  KNN Total Time: {knn_total_time:.2f}s")
        
        # ========== Spark LSH + Cache ==========
        print("\n[2/2] Spark LSH + Cache (optimized)...")
        knn_total_start = time.time()
        
        spark = SparkSession.builder.appName("CAL_LSH").config("spark.driver.memory", "8g").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        knn_result = run_spark_knn(spark, xb, xq, args.k, method='lsh',
                                   num_tables=args.num_tables, bucket_width=args.bucket_width,
                                   num_partitions=args.partitions)
        spark.stop()
        
        cal_timing, cal_stats = run_cal_loop(
            knn_result['neighbours'], train_logits, pool_logits,
            args.num_classes, use_cache=True, device=device
        )
        
        knn_total_time = time.time() - knn_total_start
        lsh_times.append(knn_total_time)
        
        print(f"  Spark Build: {knn_result['build_time']:.2f}s")
        print(f"  Spark Search: {knn_result['search_time']:.2f}s")
        print(f"  CAL Loop: {cal_timing['loop_time']:.2f}s")
        print(f"    - Data prep: {cal_timing['time_data_prep']:.2f}s")
        print(f"    - KL compute: {cal_timing['time_kl_compute']:.2f}s")
        print(f"  Cache: hits={cal_stats['hits']}, misses={cal_stats['misses']}")
        print(f"  KNN Total Time: {knn_total_time:.2f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (KNN Total Time)")
    print("=" * 70)
    
    exact_mean, exact_std = np.mean(exact_times), np.std(exact_times)
    lsh_mean, lsh_std = np.mean(lsh_times), np.std(lsh_times)
    
    print(f"Spark Exact (no cache): {exact_mean:.2f}s ± {exact_std:.2f}s")
    print(f"Spark LSH (cache):      {lsh_mean:.2f}s ± {lsh_std:.2f}s")
    print(f"Speedup:                {exact_mean/lsh_mean:.2f}x")
    
    # Combined with previous run
    prev_exact = 137.25
    prev_lsh = 65.26
    print("\n" + "-" * 70)
    print("Combined with previous full experiment (137.25 vs 65.26):")
    all_exact = [prev_exact] + exact_times
    all_lsh = [prev_lsh] + lsh_times
    print(f"Spark Exact: {np.mean(all_exact):.2f}s ± {np.std(all_exact):.2f}s")
    print(f"Spark LSH:   {np.mean(all_lsh):.2f}s ± {np.std(all_lsh):.2f}s")
    print(f"Speedup:     {np.mean(all_exact)/np.mean(all_lsh):.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
