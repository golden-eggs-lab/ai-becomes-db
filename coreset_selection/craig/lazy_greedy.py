print(__doc__)
import matplotlib
#matplotlib.use('TkAgg')

import heapq
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy import spatial
import matplotlib.pyplot as plt
import time

# ANN libraries
try:
    from nearpy import Engine
    from nearpy.hashes import RandomBinaryProjections
    from nearpy.distances import CosineDistance
    NEARPY_AVAILABLE = True
except ImportError:
    NEARPY_AVAILABLE = False
    print("Warning: NearPy not available. ANN optimization disabled.")

try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Warning: Milvus not available. Milvus backend disabled.")

try:
    from pyspark.sql import SparkSession
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.functions import udf, col, row_number, broadcast
    from pyspark.sql.types import FloatType, IntegerType
    from pyspark.ml.feature import BucketedRandomProjectionLSH
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Warning: PySpark not available. Spark backend disabled.")


class FacilityLocation:

    def __init__(self, D, V, alpha=1.):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        
        # Timing instrumentation for bottleneck analysis
        self.inc_time = 0  # Time for inc() - candidate evaluation (ANN target)
        self.add_time = 0  # Time for add() - state update (Reuse target)
        self.inc_calls = 0
        self.add_calls = 0
        
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        start = time.time()
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                result = math.log(1 + self.alpha * 1)
            else:
                result = self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            result = self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal
        self.inc_time += time.time() - start
        self.inc_calls += 1
        return result

    def add(self, sset, ndx):
        start = time.time()
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        self.add_time += time.time() - start
        self.add_calls += 1
        return self.curVal
    
    def print_bottleneck_stats(self):
        """Print bottleneck analysis for baseline"""
        total = self.inc_time + self.add_time
        if total > 0:
            print(f"\n=== BASELINE BOTTLENECK ANALYSIS ===")
            print(f"  inc() time (candidate eval): {self.inc_time:.2f}s ({100*self.inc_time/total:.1f}%) - ANN target")
            print(f"  add() time (state update):   {self.add_time:.2f}s ({100*self.add_time/total:.1f}%) - Reuse target")
            print(f"  TOTAL:                       {total:.2f}s (100%)")
            print(f"  inc() calls: {self.inc_calls}, add() calls: {self.add_calls}")


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv >= 0:
            if not order:
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    # Print bottleneck analysis for baseline
    if hasattr(F, 'print_bottleneck_stats'):
        F.print_bottleneck_stats()

    return sset, vals


class FacilityLocationANN:
    """
    Facility Location with ANN optimization for argmax search.
    Supports multiple backends: NearPy (LSH) or Milvus (vector database).
    """
    
    def __init__(self, D, V, gradients, alpha=1., ann_k=10, use_ann=True, ann_backend='nearpy', force_backend=False, use_reuse=True):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - gradients: np.array, shape [N, d], gradient vectors for each element
        - alpha: float
        - ann_k: int, number of candidates to retrieve from ANN (default 10)
        - use_ann: bool, whether to use ANN approximation (if False but force_backend=True, still uses backend but with exact search)
        - ann_backend: str, 'nearpy' or 'milvus' (default 'nearpy')
        - force_backend: bool, force using the backend even for exact search (for fair comparison)
        - use_reuse: bool, whether to use incremental residual update R_{i+1} = R_i - g_j (default True)
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])
        
        # ANN-specific attributes
        self.ann_backend = ann_backend.lower()
        self.force_backend = force_backend  # Force using backend infrastructure even for exact
        self.use_backend = (use_ann or force_backend) and self._check_backend_available()
        self.use_ann = use_ann  # Whether to use approximation (limit=k) or exact (limit=all)
        self.ann_k = ann_k
        self.gradients = gradients
        self.N, self.d = gradients.shape
        self.use_reuse = use_reuse  # Whether to use incremental residual update
        self.selected_set = []  # Track selected elements for non-reuse mode
        
        # Statistics
        self.ann_time = 0
        self.exact_time = 0
        self.ann_queries = 0
        
        # Incremental residual: R_{i+1} = R_i - g_j
        # Initialize R_0 = sum of all gradients
        self.current_residual = np.sum(self.gradients, axis=0)  # O(N×d) only once!
        
        if not self.use_reuse:
            print(f"  Note: Residual reuse DISABLED - will recompute R from scratch each time")
        
        if self.use_backend:
            mode = "ANN" if use_ann else "Exact"
            print(f"Initializing {mode} engine ({self.ann_backend}) with dimension {self.d}, k={ann_k if use_ann else 'all'}")
            if self.ann_backend == 'milvus':
                self._init_milvus_engine()
            elif self.ann_backend == 'spark':
                self._init_spark_engine()
            else:
                self._init_nearpy_engine()
    
    def _check_backend_available(self):
        """Check if requested backend is available"""
        if self.ann_backend == 'milvus':
            if not MILVUS_AVAILABLE:
                print(f"Warning: Milvus not available, falling back to exact search")
                return False
            return True
        elif self.ann_backend == 'spark':
            if not SPARK_AVAILABLE:
                print(f"Warning: PySpark not available, falling back to exact search")
                return False
            return True
        elif self.ann_backend == 'nearpy':
            if not NEARPY_AVAILABLE:
                print(f"Warning: NearPy not available, falling back to exact search")
                return False
            return True
        else:
            print(f"Warning: Unknown backend '{self.ann_backend}', falling back to exact search")
            return False
    
    def _init_nearpy_engine(self):
        """Initialize NearPy ANN engine"""
        # Use fewer hash tables for faster indexing (trade recall for speed)
        # For small datasets like MNIST, overhead of many tables is too high
        num_bits = min(16, self.d // 2)  # Reduce bits for faster hashing
        num_tables = 3  # Reduce from 10 to 3 for faster indexing
        
        print(f"  Building NearPy index: {num_tables} tables, {num_bits} bits...")
        start = time.time()
        
        self.engine = Engine(self.d, lshashes=[
            RandomBinaryProjections(f'rbp_{i}', num_bits) 
            for i in range(num_tables)
        ])
        
        # Index all gradient vectors
        for idx in range(self.N):
            self.engine.store_vector(self.gradients[idx], data=idx)
        
        elapsed = time.time() - start
        print(f"  ✓ NearPy index built in {elapsed:.2f}s")
        
        print(f"NearPy engine initialized with {self.N} vectors")
    
    def _init_milvus_engine(self):
        """Initialize Milvus Lite engine"""
        print(f"  Building Milvus index...")
        start = time.time()
        
        # Initialize Milvus Lite (embedded version) with proper file path
        # Use absolute path for Milvus Lite - pymilvus 2.3.0 requires this format
        import os
        db_path = "/mnt/data2/yichen/milvus_craig.db"
        self.milvus_client = MilvusClient(db_path)
        
        # Drop collection if exists
        if self.milvus_client.has_collection("gradients"):
            self.milvus_client.drop_collection("gradients")
        
        # Create collection with proper schema
        self.milvus_client.create_collection(
            collection_name="gradients",
            dimension=self.d,
            metric_type="L2",  # L2 distance for nearest neighbor
            auto_id=False
        )
        
        # Prepare data for batch insertion
        data = [
            {"id": idx, "vector": self.gradients[idx].tolist()}
            for idx in range(self.N)
        ]
        
        # Batch insert all gradient vectors
        self.milvus_client.insert(
            collection_name="gradients",
            data=data
        )
        
        elapsed = time.time() - start
        print(f"  ✓ Milvus index built in {elapsed:.2f}s")
        print(f"Milvus engine initialized with {self.N} vectors")
    
    def _init_spark_engine(self):
        """Initialize Spark engine for BATCH query strategy
        
        BATCH STRATEGY:
        - Spark is used to narrow down search space (6000 → 100 candidates)
        - Python does exact search within the small candidate set
        - R is updated incrementally after each selection
        
        FAIR COMPARISON:
        - ANN mode: Spark uses LSH approxNearestNeighbors (approximate)
        - Exact mode: Spark uses full distance computation (precise)
        - Both return ~100 candidates, then Python does the same exact search
        """
        print(f"  Building Spark engine (BATCH query mode)...")
        start = time.time()
        
        # Initialize SparkSession
        self.spark = SparkSession.builder \
            .appName("CRAIG-ANN-BATCH") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.default.parallelism", "8") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.local.dir", "/mnt/data2/spark_temp") \
            .config("spark.worker.dir", "/mnt/data2/spark_temp") \
            .getOrCreate()
        
        # Suppress Spark logging
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Convert gradients to Spark DataFrame
        print(f"  Creating Spark DataFrame...")
        data = [(int(i), Vectors.dense(self.gradients[i].tolist())) 
                for i in range(self.N)]
        self.spark_df = self.spark.createDataFrame(data, ["id", "features"])
        self.spark_df.cache()
        self.spark_df.count()  # Force materialization
        
        # Build LSH model (used for ANN mode)
        print(f"  Building LSH model...")
        self.lsh = BucketedRandomProjectionLSH(
            inputCol="features",
            outputCol="hashes",
            bucketLength=2.0,   # Smaller = more accurate, larger = faster
            numHashTables=3     # More tables = better recall
        )
        self.lsh_model = self.lsh.fit(self.spark_df)
        
        # Pre-hash all vectors for faster ANN queries
        self.hashed_df = self.lsh_model.transform(self.spark_df).cache()
        self.hashed_df.count()
        
        elapsed = time.time() - start
        print(f"  ✓ Spark engine ready in {elapsed:.2f}s")
        print(f"Spark engine initialized with {self.N} vectors")
    
    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()
    
    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal
    
    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal
    
    def get_residual(self, sset=None):
        """
        Get current residual vector.
        
        If use_reuse=True: Uses cached value (updated incrementally via update_residual).
        If use_reuse=False: Recomputes R = Σg_u - Σg_s from scratch (O(N×d)).
        """
        if self.use_reuse:
            return self.current_residual.copy()
        else:
            # Recompute from scratch: R = Σg_u - Σg_s
            total_sum = np.sum(self.gradients, axis=0)  # O(N×d)
            selected_sum = np.sum(self.gradients[self.selected_set], axis=0) if self.selected_set else np.zeros(self.d)
            return total_sum - selected_sum
    
    def update_residual(self, selected_idx):
        """
        Update residual after selecting an element.
        
        If use_reuse=True: Incremental update R_{i+1} = R_i - g_j (O(d))
        If use_reuse=False: Just track selected elements (recompute happens in get_residual)
        
        Args:
            selected_idx: index of the newly selected gradient
        """
        if self.use_reuse:
            # Incremental: O(d)
            self.current_residual -= self.gradients[selected_idx]
        else:
            # Track for recomputation: O(1)
            self.selected_set.append(selected_idx)
    
    def reset_residual(self):
        """
        Reset residual to initial state (sum of all gradients).
        Call this at the start of each epoch.
        """
        self.current_residual = np.sum(self.gradients, axis=0)
        self.selected_set = []  # Also reset selected set for non-reuse mode
    
    def find_best_candidate_ann(self, sset, candidates):
        """
        Use ANN to find top-k candidates, then exact search within them
        """
        start = time.time()
        
        # Compute residual as query vector
        R = self.get_residual(sset)
        
        # ANN search based on backend
        if self.ann_backend == 'milvus':
            ann_candidates = self._milvus_search(R, candidates)
        elif self.ann_backend == 'spark':
            ann_candidates = self._spark_search(R, candidates)
        else:  # nearpy
            ann_candidates = self._nearpy_search(R, candidates)
        
        self.ann_time += time.time() - start
        self.ann_queries += 1
        return ann_candidates
    
    def _nearpy_search(self, query_vector, candidates):
        """
        NearPy-based ANN search
        """
        # NearPy returns (vector, data, distance) tuples
        results = self.engine.neighbours(query_vector)
        
        # Extract candidate indices from ANN results
        ann_candidates = []
        for vec, data, dist in results[:self.ann_k]:
            if data in candidates:  # only consider remaining candidates
                ann_candidates.append(data)
        
        # Fallback: if ANN doesn't return enough candidates, add random ones
        if len(ann_candidates) < min(self.ann_k, len(candidates)):
            remaining = list(set(candidates) - set(ann_candidates))
            np.random.shuffle(remaining)
            needed = min(self.ann_k, len(candidates)) - len(ann_candidates)
            ann_candidates.extend(remaining[:needed])
        
        return ann_candidates
    
    def _milvus_search(self, query_vector, candidates):
        """
        Milvus-based search (exact or ANN depending on self.use_ann)
        """
        # Determine search limit: all candidates for exact, top-k for ANN
        search_limit = len(candidates) if not self.use_ann else self.ann_k * 2
        
        # Search for nearest neighbors
        results = self.milvus_client.search(
            collection_name="gradients",
            data=[query_vector.tolist()],
            limit=search_limit,  # All for exact, limited for ANN
            output_fields=["id"]
        )
        
        # Extract candidate indices from Milvus results
        ann_candidates = []
        candidates_set = set(candidates)
        target_count = len(candidates) if not self.use_ann else self.ann_k
        
        for hit in results[0]:
            idx = hit['id']
            if idx in candidates_set:
                ann_candidates.append(idx)
                if len(ann_candidates) >= target_count:
                    break
        
        # Fallback: if not enough candidates found
        if len(ann_candidates) < min(target_count, len(candidates)):
            remaining = list(candidates_set - set(ann_candidates))
            np.random.shuffle(remaining)
            needed = min(target_count, len(candidates)) - len(ann_candidates)
            ann_candidates.extend(remaining[:needed])
        
        return ann_candidates
    
    def _spark_search(self, query_vector, candidates):
        """
        Spark-based search with DYNAMIC residual query.
        
        BATCH STRATEGY:
        - Spark narrows search space: 6000 candidates → ~100 candidates
        - Returns candidates to Python for exact search within batch
        
        FAIR COMPARISON:
        - ANN mode: Spark uses LSH approxNearestNeighbors (fast, approximate)
        - Exact mode: Spark uses full distance computation (slower, precise)
        - Both return ~ann_k candidates for Python to do exact search
        
        Args:
            query_vector: current residual R (dynamically updated)
            candidates: remaining candidate indices
            
        Returns:
            list of candidate indices (narrowed down by Spark)
        """
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        
        candidates_set = set(candidates)
        candidates_list = list(candidates_set)
        
        # Filter DataFrame to only remaining candidates
        candidates_df = self.spark_df.filter(col("id").isin(candidates_list))
        
        # Convert query to Spark Vector
        query_vec = Vectors.dense(query_vector.tolist())
        
        if self.use_ann:
            # ANN mode: Use LSH approximate nearest neighbors
            # This is fast because it only searches within LSH buckets
            try:
                result_df = self.lsh_model.approxNearestNeighbors(
                    candidates_df, 
                    query_vec, 
                    self.ann_k,
                    distCol="distance"
                )
                ann_candidates = [row.id for row in result_df.select("id").collect()]
            except Exception as e:
                # Fallback if LSH fails (e.g., empty buckets)
                print(f"  LSH fallback: {e}")
                ann_candidates = candidates_list[:self.ann_k]
        else:
            # Exact mode: Compute distance to ALL candidates, return ALL (no limit)
            # This matches the euclidean similarity used in F.inc(): S[i,j] = max - ||g_i - g_j||
            # So argmax S[i,R] = argmin ||g_i - R||
            # IMPORTANT: No .limit() here - we want ALL candidates for exact search
            query_broadcast = self.spark.sparkContext.broadcast(query_vector)
            
            def compute_distance(features):
                vec = np.array(features.toArray())
                return float(np.linalg.norm(vec - query_broadcast.value))  # Euclidean distance
            
            distance_udf = udf(compute_distance, FloatType())
            
            result_df = candidates_df \
                .withColumn("distance", distance_udf(col("features"))) \
                .orderBy(col("distance").asc())
                # Exact mode: no .limit() - return all candidates sorted by distance
            
            ann_candidates = [row.id for row in result_df.select("id").collect()]
        
        # Fallback: ensure we have enough candidates
        if len(ann_candidates) < min(self.ann_k, len(candidates)):
            remaining = list(candidates_set - set(ann_candidates))
            np.random.shuffle(remaining)
            needed = min(self.ann_k, len(candidates)) - len(ann_candidates)
            ann_candidates.extend(remaining[:needed])
        
        return ann_candidates
    
    def print_stats(self):
        """Print ANN performance statistics"""
        if self.use_ann and self.ann_queries > 0:
            print(f"\n=== ANN Statistics ===")
            print(f"Total ANN queries: {self.ann_queries}")
            print(f"Avg ANN time per query: {self.ann_time/self.ann_queries*1000:.2f} ms")
            print(f"Total ANN time: {self.ann_time:.3f} s")
            print(f"Total exact time: {self.exact_time:.3f} s")
            print(f"Speedup: {self.exact_time/self.ann_time:.2f}x" if self.ann_time > 0 else "N/A")


def lazy_greedy_heap_ann_batch(F, V, B, batch_size=20, use_ann=True):
    """
    Lazy greedy with BATCH ANN optimization.
    
    BATCH STRATEGY:
    - Every batch_size selections, call Spark to get ~ann_k candidates
    - Within batch, use pure Python for exact search in the candidate set
    - R is updated incrementally after each selection: R_{i+1} = R_i - g_j
    
    FAIR COMPARISON (ANN vs Exact):
    - Both use same batch_size and ann_k
    - Both call Spark same number of times (B / batch_size)
    - Only difference: Spark uses LSH (ANN) vs full distance (Exact)
    
    Args:
    - F: FacilityLocationANN object
    - V: list of candidate indices
    - B: budget (number of elements to select)
    - batch_size: how many selections per Spark call (default 20)
    - use_ann: whether to use ANN for Spark search
    
    Returns:
    - sset: selected subset
    - vals: objective values at each iteration
    """
    sset = []
    vals = []
    remaining = set(V)
    curVal = 0
    
    # Reset residual to initial state (R_0 = sum of all gradients)
    if hasattr(F, 'reset_residual'):
        F.reset_residual()
    
    print(f"Starting BATCH greedy selection: B={B}, batch_size={batch_size}, ann_k={F.ann_k}")
    print(f"Mode: {'ANN' if use_ann else 'Exact'}")
    print(f"Expected Spark calls: {(B + batch_size - 1) // batch_size}")
    
    batch_num = 0
    spark_time = 0
    python_time = 0
    
    while len(sset) < B and remaining:
        batch_num += 1
        batch_start = time.time()
        
        # ========== Step 1: Spark query to get candidates ==========
        spark_start = time.time()
        R = F.get_residual()  # O(d) - just returns cached residual
        
        # Spark narrows down: remaining (e.g., 6000) → candidates (e.g., 100)
        if F.ann_backend == 'spark' and F.use_backend:
            candidates = F._spark_search(R, remaining)
        elif F.ann_backend == 'milvus' and F.use_backend:
            candidates = F._milvus_search(R, remaining)
        elif F.ann_backend == 'nearpy' and F.use_backend:
            candidates = F._nearpy_search(R, remaining)
        else:
            # No backend, use all remaining as candidates
            candidates = list(remaining)
        
        F.ann_queries += 1
        spark_elapsed = time.time() - spark_start
        spark_time += spark_elapsed
        
        # ========== Step 2: Python exact search within candidates ==========
        python_start = time.time()
        candidates_set = set(candidates) & remaining  # Ensure all are still valid
        
        # Select batch_size elements from candidates using exact search
        batch_selected = 0
        while batch_selected < batch_size and len(sset) < B and candidates_set:
            # Find best candidate using true marginal gain (Facility Location)
            # This is the correct two-stage approach:
            #   1. Spark coarse selection: N → ann_k candidates
            #   2. Python fine selection: use F.inc() for true gain
            best_idx = None
            best_gain = float('-inf')
            
            for idx in candidates_set:
                gain = F.inc(sset, idx)  # True marginal gain (Facility Location objective)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx is None:
                break
            
            # Add to selected set
            curVal = F.add(sset, best_idx)
            sset.append(best_idx)
            remaining.remove(best_idx)
            candidates_set.remove(best_idx)
            vals.append(curVal)
            
            # Incremental residual update: R_{i+1} = R_i - g_j
            if hasattr(F, 'update_residual'):
                F.update_residual(best_idx)
                # Note: R is updated for next Spark call, not needed within same batch
                # since we're using F.inc() which uses the similarity matrix directly
            
            batch_selected += 1
        
        python_elapsed = time.time() - python_start
        python_time += python_elapsed
        
        # Progress report
        batch_elapsed = time.time() - batch_start
        print(f"Batch {batch_num}: selected {batch_selected} elements, "
              f"|S|={len(sset)}/{B}, remaining={len(remaining)}, "
              f"spark={spark_elapsed:.2f}s, python={python_elapsed:.3f}s")
    
    # Final statistics
    print(f"\n=== BATCH Statistics ===")
    print(f"Total Spark queries: {F.ann_queries}")
    print(f"Total Spark time: {spark_time:.2f}s")
    print(f"Total Python time: {python_time:.2f}s")
    print(f"Avg Spark time per query: {spark_time/F.ann_queries*1000:.1f}ms" if F.ann_queries > 0 else "N/A")
    
    return sset, vals


def lazy_greedy_heap_ann(F, V, B, use_ann=True):
    """
    Lazy greedy with optional ANN optimization.
    
    Uses incremental residual update: R_{i+1} = R_i - g_j
    This reduces residual computation from O(N×d) to O(d) per iteration!
    
    Args:
    - F: FacilityLocationANN object
    - V: list of candidate indices
    - B: budget (number of elements to select)
    - use_ann: whether to use ANN for candidate search
    
    Returns:
    - sset: selected subset
    - vals: objective values at each iteration
    """
    curVal = 0
    sset = []
    vals = []
    remaining = set(V)
    
    # Reset residual to initial state (R_0 = sum of all gradients)
    if hasattr(F, 'reset_residual'):
        F.reset_residual()
    
    order = []
    heapq._heapify_max(order)
    
    # Initial evaluation
    print(f"Initial evaluation of {len(V)} candidates...")
    for index in V:
        gain = F.inc(sset, index)
        _heappush_max(order, (gain, index))
    
    iteration = 0
    while order and len(sset) < B:
        iteration += 1
        
        # Use ANN every 10 iterations to balance speed and accuracy
        # Early iterations: use ANN for fast candidate filtering
        # This reduces the number of exact evaluations needed
        use_ann_this_iter = (iteration % 10 == 0) and use_ann
        
        if use_ann_this_iter and F.use_ann and len(remaining) > F.ann_k * 2:
            # Get top-k candidates from ANN
            ann_start = time.time()
            top_candidates = F.find_best_candidate_ann(sset, remaining)
            
            # Re-evaluate only these candidates
            new_order = []
            heapq._heapify_max(new_order)
            for idx in top_candidates:
                gain = F.inc(sset, idx)
                _heappush_max(new_order, (gain, idx))
            
            # Also keep some random candidates for diversity
            other_candidates = list(remaining - set(top_candidates))
            np.random.shuffle(other_candidates)
            for idx in other_candidates[:min(10, len(other_candidates))]:
                gain = F.inc(sset, idx)
                _heappush_max(new_order, (gain, idx))
            
            order = new_order
            F.ann_time += time.time() - ann_start
        
        # Standard lazy greedy evaluation
        el = _heappop_max(order)
        
        exact_start = time.time()
        improv = F.inc(sset, el[1])
        F.exact_time += time.time() - exact_start
        
        # check for valid elements
        if improv >= 0 and el[1] in remaining:
            if not order:
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                remaining.remove(el[1])
                vals.append(curVal)
                # Incremental residual update: R_{i+1} = R_i - g_j
                if hasattr(F, 'update_residual'):
                    F.update_residual(el[1])
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    remaining.remove(el[1])
                    vals.append(curVal)
                    # Incremental residual update: R_{i+1} = R_i - g_j
                    if hasattr(F, 'update_residual'):
                        F.update_residual(el[1])
                    
                    if iteration % 10 == 0:
                        print(f"Iter {iteration}: Selected element {el[1]}, "
                              f"gain={improv:.4f}, |S|={len(sset)}, remaining={len(remaining)}")
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)
    
    if F.use_ann:
        F.print_stats()
    
    return sset, vals


def test():
    n = 10
    X = np.random.rand(n, n)
    D = X * np.transpose(X)
    F = FacilityLocation(D, range(0, n))
    sset = lazy_greedy(F, range(0, n), 15)
    print(sset)

