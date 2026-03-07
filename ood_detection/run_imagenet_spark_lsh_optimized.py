# 优化版本：避免使用 approxSimilarityJoin，改用更高效的方式
# 核心思想：
# 1. LSH hash 在 Python 端预计算并 collect
# 2. 对每个 query，在 Python 端快速找到候选 bucket
# 3. 只对候选集做 Spark 计算（大幅减少数据量）

import numpy as np
from collections import defaultdict

def compute_knn_lsh_optimized(spark, lsh_model, df_train_hashed, ftrain_spark, test_vectors, K, batch_size):
    """
    优化的 LSH KNN：避免 approxSimilarityJoin 的 overhead
    """
    # Step 1: Collect LSH hashes to driver (一次性)
    # 这个可以接受因为 training data 只有 64k
    print("Collecting LSH hashes to driver...")
    train_hashes = df_train_hashed.select("train_id", "hashes").collect()
    
    # Build hash table: hash_signature -> [train_ids]
    hash_table = defaultdict(list)
    for row in train_hashes:
        train_id = row.train_id
        hashes = row.hashes  # Vector of hash values
        # For each hash table
        for table_idx, hash_val in enumerate(hashes):
            # Convert to hashable key
            key = (table_idx, int(hash_val))
            hash_table[key].append(train_id)
    
    print(f"Hash table built with {len(hash_table)} buckets")
    
    # Step 2: Process queries in batches
    n_queries = len(test_vectors)
    n_batches = (n_queries + batch_size - 1) // batch_size
    all_scores = np.zeros(n_queries)
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_queries)
        batch_vectors = test_vectors[batch_start:batch_end]
        
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx + 1}/{n_batches} (queries {batch_start}-{batch_end})")
        
        # Step 3: For each query, find候选 using hash table
        for local_idx, query_vec in enumerate(batch_vectors):
            global_idx = batch_start + local_idx
            
            # Hash the query
            query_df = spark.createDataFrame([(0, Vectors.dense(query_vec.tolist()))], 
                                            ["query_id", "train_features"])
            query_hashed = lsh_model.transform(query_df)
            query_hash_row = query_hashed.select("hashes").first()
            query_hashes = query_hash_row.hashes
            
            # Find candidate train IDs
            candidate_ids = set()
            for table_idx, hash_val in enumerate(query_hashes):
                key = (table_idx, int(hash_val))
                if key in hash_table:
                    candidate_ids.update(hash_table[key])
            
            # If not enough candidates, use brute force
            if len(candidate_ids) < K:
                candidate_ids = set(range(len(ftrain_spark)))
            
            # Step 4: Compute exact distances only for candidates (in numpy, much faster!)
            candidate_ids = list(candidate_ids)
            candidate_vectors = ftrain_spark[candidate_ids]
            
            # Vectorized distance computation
            dists = np.sum((candidate_vectors - query_vec) ** 2, axis=1)
            
            # Get K-th distance
            if len(dists) >= K:
                kth_dist = np.partition(dists, K-1)[K-1]
            else:
                kth_dist = 1e10
            
            all_scores[global_idx] = -kth_dist
    
    return all_scores
