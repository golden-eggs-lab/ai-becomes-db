import os
import time
from util.args_loader import get_args
from util import metrics
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Try to import torch (optional, only needed for GPU features and random seed)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. GPU features disabled, using numpy for random seed.")

# Milvus imports (optional)
if '--use-milvus' in os.sys.argv or '--use_milvus' in os.sys.argv:
    try:
        from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
        MILVUS_AVAILABLE = True
    except ImportError:
        print("Warning: pymilvus not installed. Install with: pip install pymilvus")
        MILVUS_AVAILABLE = False
else:
    MILVUS_AVAILABLE = False

# Spark imports (optional)
if '--use-spark' in os.sys.argv or '--use_spark' in os.sys.argv:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import udf, col, posexplode
        from pyspark.sql.types import ArrayType, FloatType, DoubleType, StructType, StructField
        from pyspark.ml.feature import BucketedRandomProjectionLSH
        from pyspark.ml.linalg import Vectors, VectorUDT
        SPARK_AVAILABLE = True
    except ImportError:
        print("Warning: pyspark not installed. Install with: pip install pyspark")
        SPARK_AVAILABLE = False
else:
    SPARK_AVAILABLE = False

args = get_args()

seed = args.seed
print(seed)
if TORCH_AVAILABLE:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Only check for CUDA if torch is available
device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class_num = 1000
id_train_size = 1281167
id_val_size = 50000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 2048))
score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))
label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))


cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 2048))
score_log_val = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))
label_log_val = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
FORCE_RUN = False
norm_cache = f"cache/{args.in_dataset}_train_{args.name}_in/feat_norm.mmap"
if not FORCE_RUN and os.path.exists(norm_cache):
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_train_size, 2048))
else:
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_train_size, 2048))
    feat_log_norm[:] = normalizer(feat_log)

norm_cache = f"cache/{args.in_dataset}_val_{args.name}_in/feat_norm.mmap"
if not FORCE_RUN and os.path.exists(norm_cache):
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_val_size, 2048))
else:
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_val_size, 2048))
    feat_log_val_norm[:] = normalizer(feat_log_val)


prepos_feat = lambda x: np.ascontiguousarray(normalizer(x).astype(np.float32))
ftrain = np.ascontiguousarray(feat_log_norm.astype(np.float32))
ftest = np.ascontiguousarray(feat_log_val_norm.astype(np.float32))


ood_feat_log_all = {}
food_all = {}
sood_all = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}

for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 2048))
    ood_score_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], class_num))
    ood_feat_log_all[ood_dataset] = ood_feat_log
    food_all[ood_dataset] = prepos_feat(ood_feat_log).astype(np.float32)

#################### KNN/ANN score OOD detection #################

ALPHA = 1.00
for K in [1000]:
    rand_ind = np.random.choice(id_train_size, int(id_train_size * ALPHA), replace=False)
    ftrain_sample = ftrain[rand_ind]
    
    # Apply general subsample ratio (for scaling experiments)
    if hasattr(args, 'subsample_ratio') and args.subsample_ratio < 1.0:
        subsample_size = int(len(ftrain_sample) * args.subsample_ratio)
        subsample_idx = np.random.choice(len(ftrain_sample), subsample_size, replace=False)
        ftrain_sample = ftrain_sample[subsample_idx]
        print(f"Subsampling training data: {args.subsample_ratio * 100:.1f}% -> {len(ftrain_sample)} samples")
    
    # Check if using Milvus, Spark, or Faiss
    if args.use_spark:
        if not SPARK_AVAILABLE:
            print("ERROR: Spark requested but pyspark not installed!")
            print("Install with: pip install pyspark")
            exit(1)
        
        # ============= Spark Implementation =============
        # Exact KNN: Cross Join + distance computation (true exact search)
        # ANN: MLlib BucketedRandomProjectionLSH (approximate search)
        print(f"Using Spark for KNN search")
        print(f"Spark master: {args.spark_master}, Partitions: {args.spark_partitions}")
        print(f"Subsampling training data: {args.spark_subsample * 100:.1f}%")
        
        if args.spark_use_lsh:
            method_name = 'Spark-LSH'
            print(f"Mode: ANN (MLlib LSH, bucket_length={args.spark_lsh_bucket_length}, tables={args.spark_lsh_tables})")
        else:
            method_name = 'Spark-ExactKNN'
            print(f"Mode: Exact KNN (Cross Join + distance computation)")
        
        start_time = time.time()
        
        # Subsample training data for Spark (to reduce computation)
        spark_subsample_size = int(len(ftrain_sample) * args.spark_subsample)
        spark_subsample_indices = np.random.choice(len(ftrain_sample), spark_subsample_size, replace=False)
        ftrain_spark = ftrain_sample[spark_subsample_indices]
        print(f"Using {len(ftrain_spark)} training samples (subsampled from {len(ftrain_sample)})")
        
        # Initialize Spark with custom temp directory
        spark_temp_dir = "/mnt/data2/yichen/spark_temp"
        os.makedirs(spark_temp_dir, exist_ok=True)
        
        spark = SparkSession.builder \
            .appName("KNN-OOD") \
            .master(args.spark_master) \
            .config("spark.driver.memory", "32g") \
            .config("spark.executor.memory", "16g") \
            .config("spark.sql.shuffle.partitions", str(args.spark_partitions)) \
            .config("spark.default.parallelism", str(args.spark_partitions)) \
            .config("spark.driver.maxResultSize", "8g") \
            .config("spark.local.dir", spark_temp_dir) \
            .config("spark.worker.dir", spark_temp_dir) \
            .getOrCreate()
        
        sc = spark.sparkContext
        print(f"Spark session started with {sc.defaultParallelism} cores")
        
        # ========== Implementation based on the approach ==========
        # Exact KNN: DataFrame + UDF 暴力算距离
        # ANN: MLlib BucketedRandomProjectionLSH
        #
        # ========== 批量计算优化 ==========
        # 原来的实现是逐个查询串行处理，每个查询一个 job/stage，overhead 巨大
        # 现在改成批量计算：
        # - Exact KNN: Cross Join 一次性计算所有 query-train 距离，Window 取 top-K
        # - LSH: 使用 approxSimilarityJoin 批量处理
        
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, ArrayType, FloatType
        from pyspark.sql.window import Window
        
        build_time = time.time() - start_time
        
        # Build training DataFrame (both modes need this)
        print(f"Building training DataFrame with {len(ftrain_spark)} vectors...")
        train_data = [(int(i), Vectors.dense(vec.tolist())) for i, vec in enumerate(ftrain_spark)]
        df_train = spark.createDataFrame(train_data, ["train_id", "train_features"])
        df_train = df_train.repartition(args.spark_partitions).cache()
        df_train.count()  # Force cache
        print(f"Training DataFrame cached: {len(ftrain_spark)} vectors")
        
        if args.spark_use_lsh:
            # ========== ANN: Random Projection + Two-stage filtering (Spark-optimized) ==========
            print(f"Building Random Projection ANN...")
            
            # Stage 1: Random projection to low dimension
            low_dim = 64  # 2048 -> 64
            np.random.seed(42)
            random_matrix = np.random.randn(ftrain_spark.shape[1], low_dim).astype(np.float32)
            random_matrix /= np.sqrt(low_dim)  # Normalize
            
            print(f"  Random projection: {ftrain_spark.shape[1]}D -> {low_dim}D")
            ftrain_low = ftrain_spark @ random_matrix
            
            # Add low-dim features to training DataFrame
            train_data_low = [(int(i), Vectors.dense(vec.tolist()), Vectors.dense(vec_low.tolist())) 
                              for i, (vec, vec_low) in enumerate(zip(ftrain_spark, ftrain_low))]
            df_train = spark.createDataFrame(
                train_data_low, 
                ["train_id", "train_features", "train_features_low"]
            )
            df_train = df_train.repartition(args.spark_partitions).cache()
            df_train.count()
            
            lsh_build_time = time.time() - start_time - build_time
            print(f"Random projection time: {lsh_build_time:.3f}s")
            
            search_start = time.time()
            
            # Broadcast full-dim and low-dim training data
            ftrain_bc = sc.broadcast(ftrain_spark.astype(np.float32))
            ftrain_low_bc = sc.broadcast(ftrain_low.astype(np.float32))
            random_matrix_bc = sc.broadcast(random_matrix)
            
            batch_size = args.spark_batch_size
            print(f"Query batch size: {batch_size}")
            print(f"Two-stage filtering: low-dim top-{K*2} -> high-dim top-{K}")
            
            def compute_knn_ann_batch(test_vectors, dataset_name):
                """Two-stage ANN: Random projection coarse filter + exact refinement"""
                n_queries = len(test_vectors)
                n_batches = (n_queries + batch_size - 1) // batch_size
                print(f"  Processing {n_queries} {dataset_name} vectors in {n_batches} batches...")
                
                all_scores = np.zeros(n_queries)
                
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_queries)
                    batch_vectors = test_vectors[batch_start:batch_end]
                    
                    if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
                        print(f"    Batch {batch_idx + 1}/{n_batches} (queries {batch_start}-{batch_end})")
                    
                    # Project queries to low dimension
                    batch_low = batch_vectors @ random_matrix_bc.value
                    
                    # Build query DataFrame
                    query_data = [(int(i),) for i in range(len(batch_vectors))]
                    df_query = spark.createDataFrame(query_data, ["query_id"])
                    df_query = df_query.repartition(min(args.spark_partitions, len(batch_vectors))).cache()
                    
                    # Broadcast batch data
                    batch_bc = sc.broadcast(batch_vectors.astype(np.float32))
                    batch_low_bc = sc.broadcast(batch_low.astype(np.float32))
                    
                    # Stage 1: Cross join and compute low-dim distances (fast coarse filter)
                    df_cross = df_query.crossJoin(df_train.select("train_id"))
                    
                    @udf(returnType=DoubleType())
                    def compute_dist_low(query_id, train_id):
                        import numpy as np
                        q = batch_low_bc.value[query_id]
                        t = ftrain_low_bc.value[train_id]
                        return float(np.sum((q - t) ** 2))
                    
                    df_with_dist_low = df_cross.withColumn(
                        "dist_low", compute_dist_low(col("query_id"), col("train_id"))
                    )
                    
                    # Keep top-2K candidates per query based on low-dim distance
                    window_coarse = Window.partitionBy("query_id").orderBy("dist_low")
                    df_candidates = df_with_dist_low.withColumn("rank_low", F.row_number().over(window_coarse)) \
                        .filter(F.col("rank_low") <= K * 2) \
                        .select("query_id", "train_id")
                    
                    # Stage 2: Refine with high-dim exact distance (only for candidates)
                    @udf(returnType=DoubleType())
                    def compute_dist_high(query_id, train_id):
                        import numpy as np
                        q = batch_bc.value[query_id]
                        t = ftrain_bc.value[train_id]
                        return float(np.sum((q - t) ** 2))
                    
                    df_refined = df_candidates.withColumn(
                        "dist", compute_dist_high(col("query_id"), col("train_id"))
                    )
                    
                    # Get K-th nearest neighbor distance
                    window_fine = Window.partitionBy("query_id").orderBy("dist")
                    df_ranked = df_refined.withColumn("rank", F.row_number().over(window_fine))
                    df_kth = df_ranked.filter(F.col("rank") == K).select("query_id", "dist")
                    
                    results = df_kth.collect()
                    
                    # Map to global indices
                    for row in results:
                        global_idx = batch_start + row.query_id
                        all_scores[global_idx] = -row.dist
                    
                    # Cleanup
                    batch_bc.unpersist()
                    batch_low_bc.unpersist()
                    df_query.unpersist()
                
                return all_scores
            
            # Process all datasets
            scores_in = compute_knn_ann_batch(ftest, "ID test")
            
            all_results = []
            for ood_dataset, food in food_all.items():
                scores_ood = compute_knn_ann_batch(food, ood_dataset)
                result_metrics = metrics.cal_metric(scores_in, scores_ood)
                all_results.append(result_metrics)
            
            # Cleanup
            ftrain_bc.unpersist()
            ftrain_low_bc.unpersist()
            random_matrix_bc.unpersist()
            df_train.unpersist()
            
            method_name = 'Spark-RandomProjection-ANN'
        
        else:
            # ========== Exact KNN: Cross Join + Window (批量) ==========
            print(f"Using Cross Join + Window for batch exact KNN...")
            print(f"Training data build time: {time.time() - start_time:.3f}s")
            
            search_start = time.time()
            lsh_build_time = 0
            
            # Broadcast training vectors for efficient distance computation
            ftrain_bc = sc.broadcast(ftrain_spark.astype(np.float32))
            
            # Batch size for query processing (avoid cross join explosion)
            batch_size = args.spark_batch_size
            print(f"Query batch size: {batch_size}")
            
            def compute_knn_exact_batch(test_vectors, dataset_name):
                """批量计算 K-th NN distance using Cross Join + Window (with batching)"""
                n_queries = len(test_vectors)
                n_batches = (n_queries + batch_size - 1) // batch_size
                print(f"  Processing {n_queries} {dataset_name} vectors in {n_batches} batches...")
                
                all_scores = np.zeros(n_queries)
                
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_queries)
                    batch_vectors = test_vectors[batch_start:batch_end]
                    
                    if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
                        print(f"    Batch {batch_idx + 1}/{n_batches} (queries {batch_start}-{batch_end})")
                    
                    # Build query DataFrame with local batch IDs
                    query_data = [(int(i),) for i in range(len(batch_vectors))]
                    df_query = spark.createDataFrame(query_data, ["query_id"])
                    df_query = df_query.repartition(min(args.spark_partitions, len(batch_vectors))).cache()
                    
                    # Broadcast batch query vectors
                    batch_bc = sc.broadcast(batch_vectors.astype(np.float32))
                    
                    # Cross join: each query in batch paired with each training sample
                    df_cross = df_query.crossJoin(df_train.select("train_id"))
                    
                    # UDF to compute squared L2 distance
                    @udf(returnType=DoubleType())
                    def compute_dist(query_id, train_id):
                        import numpy as np
                        q = batch_bc.value[query_id]
                        t = ftrain_bc.value[train_id]
                        return float(np.sum((q - t) ** 2))
                    
                    # Compute all pairwise distances
                    df_with_dist = df_cross.withColumn(
                        "dist", compute_dist(col("query_id"), col("train_id"))
                    )
                    
                    # Use Window function to rank distances per query and get K-th
                    window_spec = Window.partitionBy("query_id").orderBy("dist")
                    df_ranked = df_with_dist.withColumn("rank", F.row_number().over(window_spec))
                    df_kth = df_ranked.filter(F.col("rank") == K).select("query_id", "dist")
                    
                    # Collect results
                    results = df_kth.collect()
                    
                    # Map back to global indices
                    for row in results:
                        global_idx = batch_start + row.query_id
                        all_scores[global_idx] = -row.dist
                    
                    # Cleanup batch resources
                    batch_bc.unpersist()
                    df_query.unpersist()
                
                return all_scores
            
            # Process all datasets
            scores_in = compute_knn_exact_batch(ftest, "ID test")
            
            all_results = []
            for ood_dataset, food in food_all.items():
                scores_ood = compute_knn_exact_batch(food, ood_dataset)
                result_metrics = metrics.cal_metric(scores_in, scores_ood)
                all_results.append(result_metrics)
            
            # Cleanup
            ftrain_bc.unpersist()
            df_train.unpersist()
        
        search_time = time.time() - search_start
        print(f"Search time: {search_time:.3f}s")
        print(f"Total time: {build_time + lsh_build_time + search_time:.3f}s")
        
        metrics.print_all_results(all_results, args.out_datasets, method_name)
        
        # Cleanup
        spark.stop()
        print()
        
    elif args.use_milvus:
        if not MILVUS_AVAILABLE:
            print("ERROR: Milvus requested but pymilvus not installed!")
            print("Install with: pip install pymilvus")
            exit(1)
        
        # ============= Milvus Implementation =============
        if args.milvus_lite:
            print(f"Using Milvus Lite (local embedded mode)")
        else:
            print(f"Using Milvus server (host={args.milvus_host}:{args.milvus_port})")
        print(f"Index type: {args.milvus_index_type}, Metric: {args.milvus_metric}")
        
        start_time = time.time()
        
        # Connect to Milvus
        if args.milvus_lite:
            # Milvus Lite uses a local file path
            connections.connect(uri="./milvus_knn_ood.db")
        else:
            # Milvus server uses host:port
            connections.connect(host=args.milvus_host, port=args.milvus_port)
        
        # Set timeout for Milvus Lite to avoid RPC timeout
        import pymilvus
        pymilvus.connections.get_connection_addr("default")
        
        # Create collection schema
        # Collection name can only contain letters, numbers and underscores (no hyphens)
        collection_name = f"knn_ood_{args.in_dataset}_{args.name}".replace("-", "_")
        dim = ftrain.shape[1]
        
        # Check if collection already exists
        collection_exists = utility.has_collection(collection_name)
        
        if collection_exists:
            collection = Collection(name=collection_name)
            collection.load()
            
            # Check if data count matches expected
            expected_count = len(ftrain_sample)
            actual_count = collection.num_entities
            
            if actual_count == expected_count:
                print(f"Collection '{collection_name}' already exists with {actual_count} vectors, reusing it...")
                build_time = 0  # No build time since we're reusing
            else:
                print(f"Collection exists but has wrong count ({actual_count} vs {expected_count}), recreating...")
                collection.release()
                utility.drop_collection(collection_name)
                collection_exists = False
        
        if not collection_exists:
            print(f"Creating new collection '{collection_name}'...")
            # Drop collection if exists (redundant but safe)
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields=fields, description="KNN-OOD training features")
            collection = Collection(name=collection_name, schema=schema)
            
            # Insert training data with multi-threading (small batches for Milvus Lite)
            print(f"Inserting {len(ftrain_sample)} training vectors with 2 threads...")
            batch_size = 1000  # Small batch to avoid RPC timeout in Milvus Lite
            
            # Prepare all batches
            batches = []
            for i in range(0, len(ftrain_sample), batch_size):
                batch_end = min(i + batch_size, len(ftrain_sample))
                batch = ftrain_sample[i:batch_end].tolist()
                batches.append((i, batch_end, batch))
            
            # Insert batches in parallel
            inserted_count = 0
            def insert_batch(args):
                idx, batch_end, batch = args
                try:
                    collection.insert([batch], timeout=60)
                    return batch_end
                except Exception as e:
                    print(f"\n  Error at batch {idx}: {e}", flush=True)
                    raise
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = {executor.submit(insert_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    batch_end = future.result()
                    inserted_count += 1
                    # Print progress every 10 batches
                    if inserted_count % 10 == 0 or batch_end == len(ftrain_sample):
                        pct = 100.0 * batch_end / len(ftrain_sample)
                        print(f"  Progress: ~{batch_end}/{len(ftrain_sample)} (~{pct:.1f}%)", flush=True)
            
            collection.flush()
            print("Data insertion completed")
        
        # Create index based on config
        index_params = {
            "metric_type": args.milvus_metric,
            "index_type": args.milvus_index_type,
        }
        
        if args.milvus_index_type == "IVF_FLAT":
            index_params["params"] = {"nlist": args.nlist}
        elif args.milvus_index_type == "IVF_SQ8":
            index_params["params"] = {"nlist": args.nlist}
        elif args.milvus_index_type == "HNSW":
            index_params["params"] = {"M": args.hnsw_M, "efConstruction": 40}
        
        # Check if index already exists AND matches requested type
        index_info = collection.indexes
        existing_index = next((idx for idx in index_info if idx.field_name == "embedding"), None)
        
        need_rebuild = False
        if existing_index is None:
            need_rebuild = True
            print(f"No index exists, building new index...")
        else:
            # Check if index type matches
            existing_type = existing_index.params.get('index_type', 'UNKNOWN')
            if existing_type != args.milvus_index_type:
                need_rebuild = True
                print(f"Index type mismatch: existing={existing_type}, requested={args.milvus_index_type}")
                print(f"Dropping old index and rebuilding...")
                collection.release()
                collection.drop_index()
            else:
                print(f"Index already exists with correct type ({existing_type}), reusing...")
        
        if need_rebuild:
            print(f"Building index with params: {index_params}")
            collection.create_index(field_name="embedding", index_params=index_params)
        
        collection.load()
        
        build_time = time.time() - start_time
        print(f"Milvus setup time: {build_time:.3f}s")
        
        # Search parameters
        search_params = {}
        if args.milvus_index_type in ["IVF_FLAT", "IVF_SQ8"]:
            search_params = {"metric_type": args.milvus_metric, "params": {"nprobe": args.nprobe}}
        elif args.milvus_index_type == "HNSW":
            search_params = {"metric_type": args.milvus_metric, "params": {"ef": args.hnsw_efSearch}}
        else:  # FLAT
            search_params = {"metric_type": args.milvus_metric, "params": {}}
        
        method_name = f'Milvus-{args.milvus_index_type}'
        
        # Perform search in batches to avoid gRPC message size limit
        search_start = time.time()
        
        # Search ID test data in batches
        # gRPC limit is 268MB. With K=1000, each result ~8KB, so 10000 queries = ~80MB (safe)
        print(f"Searching {len(ftest)} ID test vectors in batches...")
        search_batch_size = 10000  # Balance between speed and gRPC message size limit
        scores_in_list = []
        
        for i in range(0, len(ftest), search_batch_size):
            batch_end = min(i + search_batch_size, len(ftest))
            batch = ftest[i:batch_end].tolist()
            
            print(f"  ID search: processing batch {i}-{batch_end}...", flush=True)
            results = collection.search(
                data=batch,
                anns_field="embedding",
                param=search_params,
                limit=K,
                output_fields=[]
            )
            
            # Extract K-th nearest neighbor distance
            batch_scores = np.array([-r[K-1].distance for r in results])
            scores_in_list.append(batch_scores)
            
            print(f"  ID search progress: {batch_end}/{len(ftest)} done", flush=True)
        
        scores_in = np.concatenate(scores_in_list)
        
        all_results = []
        for ood_dataset, food in food_all.items():
            print(f"Searching {len(food)} {ood_dataset} vectors in batches...")
            scores_ood_list = []
            
            for i in range(0, len(food), search_batch_size):
                batch_end = min(i + search_batch_size, len(food))
                batch = food[i:batch_end].tolist()
                
                print(f"  {ood_dataset} search: processing batch {i}-{batch_end}...", flush=True)
                results = collection.search(
                    data=batch,
                    anns_field="embedding",
                    param=search_params,
                    limit=K,
                    output_fields=[]
                )
                batch_scores = np.array([-r[K-1].distance for r in results])
                scores_ood_list.append(batch_scores)
                print(f"  {ood_dataset} search: {batch_end}/{len(food)} done", flush=True)
            
            scores_ood_test = np.concatenate(scores_ood_list)
            result_metrics = metrics.cal_metric(scores_in, scores_ood_test)
            all_results.append(result_metrics)
        
        search_time = time.time() - search_start
        print(f"Search time: {search_time:.3f}s")
        print(f"Total time: {build_time + search_time:.3f}s")
        
        metrics.print_all_results(all_results, args.out_datasets, method_name)
        
        # Cleanup: release but don't drop collection (reuse for next run)
        collection.release()
        connections.disconnect(alias="default")
        print()
        
    else:
        # ============= Faiss Implementation =============
        # Build index based on args
        start_time = time.time()
        if args.use_ann:
            if args.ann_method == 'ivf':
                print(f"Using ANN method: IVF (nlist={args.nlist}, nprobe={args.nprobe})")
                quantizer = faiss.IndexFlatL2(ftrain.shape[1])
                index = faiss.IndexIVFFlat(quantizer, ftrain.shape[1], args.nlist)
                index.train(ftrain_sample)
                index.add(ftrain_sample)
                index.nprobe = args.nprobe
                method_name = f'ANN-IVF(nprobe={args.nprobe})'
            elif args.ann_method == 'hnsw':
                print(f"Using ANN method: HNSW (M={args.hnsw_M}, efSearch={args.hnsw_efSearch})")
                index = faiss.IndexHNSWFlat(ftrain.shape[1], args.hnsw_M)
                index.hnsw.efConstruction = 40
                index.add(ftrain_sample)
                index.hnsw.efSearch = args.hnsw_efSearch
                method_name = f'ANN-HNSW(efSearch={args.hnsw_efSearch})'
        else:
            print("Using exact KNN (IndexFlatL2)")
            index = faiss.IndexFlatL2(ftrain.shape[1])
            index.add(ftrain_sample)
            method_name = 'KNN'
        
        # Move to GPU if requested
        if args.use_faiss_gpu:
            if not TORCH_AVAILABLE:
                print("Warning: --use-faiss-gpu requested but torch not available. Using CPU.")
            else:
                try:
                    import faiss.contrib.torch_utils
                    if torch.cuda.is_available():
                        print(f"Moving {method_name} index to GPU...")
                        # Try both possible API styles
                        try:
                            res = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(res, 0, index)
                        except AttributeError:
                            # Alternative import for older versions
                            import faiss
                            res = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(res, 0, index)
                        print(f"Index successfully moved to GPU")
                        method_name = f'{method_name}-GPU'
                    else:
                        print("GPU not available, using CPU")
                except Exception as e:
                    print(f"Failed to move to GPU: {e}, using CPU")
        
        build_time = time.time() - start_time
        print(f"Index build time: {build_time:.3f}s")

        ################### Using KNN/ANN distance Directly ###################
        if True:
            search_start = time.time()
            D, _ = index.search(ftest, K, )
            scores_in = -D[:,-1]
            all_results = []
            for ood_dataset, food in food_all.items():
                D, _ = index.search(food, K)
                scores_ood_test = -D[:,-1]
                results = metrics.cal_metric(scores_in, scores_ood_test)
                all_results.append(results)
            
            search_time = time.time() - search_start
            print(f"Search time: {search_time:.3f}s")
            print(f"Total time: {build_time + search_time:.3f}s")

            metrics.print_all_results(all_results, args.out_datasets, method_name)
            print()
