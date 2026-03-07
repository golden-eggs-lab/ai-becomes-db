"""
Spark-based AnchorStore - V3 Batch Processing

Key Optimizations:
1. ✅ Only store log(k) - recover k via exp() when needed
2. ✅ Batch process ALL queries at once (not one by one)
3. ✅ Use same KL divergence formula as baseline

This should be faster than baseline because:
- Same data size (single Vector column)
- Amortize Spark overhead across all queries
- Pre-computed log(k) reused for all queries
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.types import FloatType, IntegerType, ArrayType, DoubleType, StructType, StructField, LongType
    from pyspark.ml.linalg import Vectors, VectorUDT
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark not installed. Install with: pip install pyspark")


class AnchorStoreSparkV3(nn.Module):
    """
    Spark-based AnchorStore - V3 with Batch Processing.
    
    Key improvements over V2:
    1. Only store log(k) - recover k = exp(log_k) when needed
    2. Batch process all queries at once
    3. Same KL divergence formula as baseline
    
    Expected speedup: 
    - Spark overhead amortized across all queries
    - Same storage as baseline (single Vector column)
    """

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2, 
                 spark_master="local[*]", app_name="KNN_Prompting_V3"):
        super(AnchorStoreSparkV3, self).__init__()
        
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark not installed. Install with: pip install pyspark")

        self.K = K
        self.dim = dim
        self.knn = knn
        self.n_class = n_class
        self.spark_master = spark_master
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .master(spark_master) \
            .appName(app_name) \
            .config("spark.driver.memory", "32g") \
            .config("spark.executor.memory", "32g") \
            .config("spark.driver.maxResultSize", "8g") \
            .config("spark.kryoserializer.buffer.max", "512m") \
            .getOrCreate()
        
        print(f"✅ Spark session created (V3 - Batch): {spark_master}")
        print(f"   Executors: {self.spark.sparkContext.defaultParallelism}")
        
        # Only store log(k) - recover k via exp() when needed
        self.register_buffer("queue_anchor_log", torch.zeros(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.df_anchors = None
        self.spark_built = False
        
        # Cache log(k) as numpy for batch processing
        self.log_k_np = None
        self.labels_np = None

    def enqueue(self, anchors, labels):
        """
        ✅ Optimization: Pre-compute and store log(k) only
        """
        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        # Only store log(k)
        self.queue_anchor_log[ptr:ptr + bs, :] = torch.log(anchors + 1e-10)
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

    def build_index(self):
        """Build Spark DataFrame with log(k) only."""
        ptr = int(self.queue_ptr)
        if ptr == 0:
            raise ValueError("No anchors in the store.")
        
        print(f"Building Spark DataFrame (V3 - Batch processing)...")
        
        # Store log(k) only
        self.log_k_np = self.queue_anchor_log[:ptr].cpu().numpy()
        self.labels_np = self.queue_label[:ptr].cpu().numpy()
        
        # Create DataFrame with log(k) only
        data = [
            (int(i), Vectors.dense(self.log_k_np[i]), int(self.labels_np[i]))
            for i in range(ptr)
        ]
        
        self.df_anchors = self.spark.createDataFrame(
            data, 
            schema=["anchor_id", "log_anchor", "label"]
        )
        
        self.df_anchors.cache()
        count = self.df_anchors.count()
        
        self.spark_built = True
        print(f"✅ Spark DataFrame built: {count} anchors")
        print(f"   Stores: log(k) only (single column)")
        print(f"   ✅ Batch processing enabled")

    def knn_infer(self, query):
        """
        KNN inference with BATCH processing.
        
        Instead of processing one query at a time (256 Spark DAGs),
        we process ALL queries in ONE Spark job.
        
        Steps:
        1. Broadcast all queries
        2. Cross-join queries with anchors
        3. Compute KL scores for all pairs
        4. Group by query, get top-k per query
        5. Collect and vote
        """
        if not self.spark_built:
            raise RuntimeError("Spark DataFrame not built.")
        
        batch_size = query.shape[0]
        
        # Convert queries to numpy
        queries_np = query.cpu().numpy()  # [batch_size, dim]
        
        # Compute log(q) for all queries
        log_queries = np.log(queries_np + 1e-10)  # [batch_size, dim]
        
        # Broadcast all queries at once
        log_queries_broadcast = self.spark.sparkContext.broadcast(log_queries)
        
        # UDF to compute KL score
        # Recovers k = exp(log_k) to compute k * (log_k - log_q)
        def compute_kl_scores_batch(log_anchor_vec, anchor_id):
            log_k = log_anchor_vec.toArray()
            k = np.exp(log_k)  # Recover k from log(k)
            
            all_log_q = log_queries_broadcast.value  # [batch_size, dim]
            
            # Compute KL for each query
            scores = []
            for q_idx in range(all_log_q.shape[0]):
                log_q = all_log_q[q_idx]
                kl = np.sum(k * (log_k - log_q))
                scores.append((q_idx, -float(kl)))  # Negative for sorting
            return scores
        
        # This approach still has issues with Spark UDF...
        # Let's try a different approach: process in Python with Spark for parallelism
        
        # Actually, for batch processing to work well in Spark, 
        # we should use native Spark operations, not UDFs
        
        # Alternative: Use Spark's map function on RDD
        log_k_broadcast = self.spark.sparkContext.broadcast(self.log_k_np)
        labels_broadcast = self.spark.sparkContext.broadcast(self.labels_np)
        
        # Create RDD of query indices
        query_rdd = self.spark.sparkContext.parallelize(range(batch_size), numSlices=64)
        
        def process_query(q_idx):
            log_q = log_queries_broadcast.value[q_idx]  # [dim]
            log_k_all = log_k_broadcast.value  # [N, dim]
            labels = labels_broadcast.value  # [N]
            
            # Recover k and compute KL scores
            k_all = np.exp(log_k_all)  # [N, dim]
            
            # KL = sum(k * (log_k - log_q), axis=1)
            kl_scores = np.sum(k_all * (log_k_all - log_q), axis=1)  # [N]
            
            # Top-k (minimize KL = maximize -KL)
            top_k_indices = np.argsort(kl_scores)[:3]  # knn=3
            top_k_labels = labels[top_k_indices]
            
            # Vote
            counts = np.bincount(top_k_labels, minlength=2)
            prediction = int(np.argmax(counts))
            
            return (q_idx, prediction)
        
        # Process all queries in parallel
        results_rdd = query_rdd.map(process_query)
        results = results_rdd.collect()
        
        # Sort by query index and extract predictions
        results.sort(key=lambda x: x[0])
        predictions = [r[1] for r in results]
        
        # Cleanup
        log_queries_broadcast.unpersist()
        log_k_broadcast.unpersist()
        labels_broadcast.unpersist()
        
        return predictions

    def __del__(self):
        """Clean up Spark session."""
        if hasattr(self, 'spark'):
            try:
                self.spark.stop()
            except:
                pass
