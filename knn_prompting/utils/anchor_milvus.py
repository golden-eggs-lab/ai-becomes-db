import torch
from torch import nn
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)


class AnchorStoreMilvus(nn.Module):
    """
    Milvus-based AnchorStore for comparing Exact KNN vs ANN search.
    
    Note: Milvus has a max dimension limit of 32768, but vocab_size can be 50257.
    Solution: Use random projection to reduce dimensionality while preserving
    inner product similarities (Johnson-Lindenstrauss theorem).
    
    Setup:
    - Both versions pre-compute log(k_i) offline (same preprocessing)
    - Apply random projection to reduce dim to 16384 (safe limit)
    - Store projected log(k_i) in Milvus
    - Search with projected query q
    
    Comparison:
    - use_ann=False: Exact KNN search (FLAT index, brute force)
    - use_ann=True:  Approximate NN search (IVF index, faster but approximate)
    
    This isolates the benefit of ANN vs exact search in Milvus.
    """

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2, 
                 use_ann=False, collection_name="knn_anchors", 
                 use_projection=False, projected_dim=16384,
                 milvus_uri="./milvus_knn.db"):
        super(AnchorStoreMilvus, self).__init__()

        self.K = K
        self.dim = dim
        self.knn = knn
        self.n_class = n_class
        self.use_ann = use_ann  # False: FLAT (exact), True: IVF (ANN)
        self.collection_name = collection_name
        self.milvus_uri = milvus_uri  # Local .db file for Milvus Lite
        
        # Dimension reduction option
        # Note: For >32768 dim, index creation will be skipped (brute force search used)
        # This is slower but maintains accuracy for small datasets
        self.use_projection = use_projection
        self.projected_dim = min(projected_dim, dim) if use_projection else dim
        
        if self.use_projection:
            print(f"Using random projection: {dim} -> {self.projected_dim}")
            print(f"⚠️  This may cause accuracy degradation")
            # Random projection matrix (fixed, for reproducibility)
            self.register_buffer("projection_matrix", 
                                self._create_projection_matrix(dim, self.projected_dim))
        else:
            self.projection_matrix = None
            if dim > 32768:
                print(f"Using full dimension: {dim} (exceeds index limit, will use brute force)")
            else:
                print(f"Using full dimension: {dim}")
        
        # Always store both k_i and pre-computed log(k_i)
        self.register_buffer("queue_anchor", torch.zeros(K, dim))
        self.register_buffer("queue_anchor_log", torch.zeros(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Milvus collection
        self.collection = None
        self.milvus_initialized = False

    def _create_projection_matrix(self, input_dim, output_dim):
        """
        Create random projection matrix using Gaussian random projection.
        Preserves inner products approximately with high probability.
        Uses FIXED seed for reproducibility and fair comparison with baseline.
        """
        # Use FIXED seed to match baseline version
        torch.manual_seed(42)
        # Use Gaussian random projection: P_{ij} ~ N(0, 1/sqrt(output_dim))
        projection = torch.randn(input_dim, output_dim) / np.sqrt(output_dim)
        return projection

    def _init_milvus(self):
        """Initialize Milvus collection."""
        if self.milvus_initialized:
            return
        
        # Connect to Milvus Lite (local .db file)
        try:
            # For Milvus Lite, use local URI
            # This avoids connecting to Docker standalone and uses embedded Milvus
            connections.connect(
                alias="default",
                uri=self.milvus_uri,
                # Milvus Lite supports configuring maxDimension via environment
                # export MILVUS_PROXY_MAX_DIMENSION=65536
            )
            print(f"✅ Connected to Milvus Lite: {self.milvus_uri}")
        except Exception as e:
            print(f"Warning: Could not connect to Milvus: {e}")
            print("For Milvus Lite with high dimensions:")
            print("  export MILVUS_PROXY_MAX_DIMENSION=65536")
            raise
        
        # Drop existing collection if exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="anchor_vector", dtype=DataType.FLOAT_VECTOR, dim=self.projected_dim),
        ]
        desc = f"KNN Prompting Anchors (dim={self.projected_dim}"
        desc += ", projected)" if self.use_projection else ", full)"
        schema = CollectionSchema(fields, description=desc)
        
        # Create collection
        self.collection = Collection(self.collection_name, schema)
        
        # Create index based on use_ann flag and dimension
        # Note: Milvus Lite has 32768 dimension limit for index creation
        # For higher dimensions, we skip index creation and use brute force search
        if self.projected_dim > 32768:
            print(f"⚠️  Dimension {self.projected_dim} exceeds Milvus index limit (32768)")
            print(f"⚠️  Skipping index creation, will use brute force search")
            self.index_created = False
        elif self.use_ann:
            # ANN: Use IVF_FLAT for approximate search (faster)
            index_params = {
                "metric_type": "IP",  # Inner Product
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}  # Number of clusters
            }
            self.collection.create_index("anchor_vector", index_params)
            print(f"Using ANN index (IVF_FLAT) for approximate search")
            self.index_created = True
        else:
            # Exact: Use FLAT for brute-force exact search
            index_params = {
                "metric_type": "IP",  # Inner Product
                "index_type": "FLAT",  # Brute force exact search
                "params": {}
            }
            self.collection.create_index("anchor_vector", index_params)
            print(f"Using FLAT index for exact KNN search")
            self.index_created = True
        
        self.milvus_initialized = True
        print(f"Milvus collection '{self.collection_name}' initialized")
        if self.use_projection:
            print(f"Dimension: {self.dim} -> {self.projected_dim} (random projection)")
        else:
            print(f"Dimension: {self.projected_dim} (full, no projection)")

    def enqueue(self, anchors, labels):
        """
        Enqueue anchor distributions with pre-computed log.
        
        Args:
            anchors: probability distributions [batch_size, vocab_size]
            labels: corresponding labels [batch_size]
        """
        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        # Store both k_i and log(k_i)
        self.queue_anchor[ptr:ptr + bs, :] = anchors
        self.queue_anchor_log[ptr:ptr + bs, :] = torch.log(anchors + 1e-10)
        self.queue_label[ptr:ptr + bs] = labels
        
        self.queue_ptr[0] = ptr + bs

    def build_index(self):
        """Build Milvus index with pre-computed log(k_i)."""
        if not self.milvus_initialized:
            self._init_milvus()
        
        ptr = int(self.queue_ptr)
        if ptr == 0:
            raise ValueError("No anchors in the store. Call enqueue() first.")
        
        # Apply projection if enabled
        if self.use_projection:
            # Project log(k_i) to lower dimension
            # Shape: [N, dim] @ [dim, projected_dim] = [N, projected_dim]
            vectors = torch.matmul(
                self.queue_anchor_log[:ptr], 
                self.projection_matrix
            ).cpu().numpy().astype('float32')
        else:
            # Use full dimension (no projection)
            vectors = self.queue_anchor_log[:ptr].cpu().numpy().astype('float32')
        
        # Insert data in batches to avoid gRPC message size limit (64MB)
        # Each vector is ~65KB for 16384-dim or ~200KB for 50257-dim
        # Safe batch size depends on dimension
        batch_size = 100 if self.use_projection else 50
        for start_idx in range(0, ptr, batch_size):
            end_idx = min(start_idx + batch_size, ptr)
            batch_vectors = vectors[start_idx:end_idx]
            
            entities = [
                list(range(start_idx, end_idx)),  # IDs
                batch_vectors.tolist()  # Vectors
            ]
            
            self.collection.insert(entities)
            print(f"  Inserted batch {start_idx}-{end_idx} ({end_idx-start_idx} vectors)")
        
        self.collection.flush()
        self.collection.load()
        
        index_type = "IVF_FLAT (ANN)" if self.use_ann else "FLAT (Exact KNN)"
        proj_str = f", projected {self.dim}->{self.projected_dim}" if self.use_projection else ", full dim"
        print(f"Milvus index built: {ptr} anchors, index={index_type}{proj_str}")

    def knn_infer(self, query):
        """
        KNN inference using Milvus search.
        
        Args:
            query: test probability distribution [batch_size, vocab_size]
            
        Returns:
            predicted labels [batch_size]
        """
        if not self.milvus_initialized:
            raise ValueError("Milvus not initialized. Call build_index() first.")
        
        ptr = int(self.queue_ptr)
        batch_size = query.shape[0]
        
        # Apply projection to query if enabled
        if self.use_projection:
            # Project query to match stored vectors
            # Inner product is preserved approximately: (P*log(k)) · (P*q) ≈ log(k) · q
            query_vectors = torch.matmul(query, self.projection_matrix)  # [batch, projected_dim]
        else:
            # Use full dimension query directly
            query_vectors = query  # [batch, dim]
        
        query_list = query_vectors.cpu().numpy().astype('float32').tolist()
        
        # Search parameters depend on index type
        if self.use_ann:
            # ANN: Need to specify nprobe (number of clusters to search)
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        else:
            # Exact: FLAT index doesn't need extra params
            search_params = {"metric_type": "IP", "params": {}}
        
        k = min(self.knn, ptr)
        
        results = self.collection.search(
            data=query_list,
            anns_field="anchor_vector",
            param=search_params,
            limit=k,
            output_fields=[]
        )
        
        # Process results
        predictions = []
        for i in range(batch_size):
            if self.knn == 1:
                # Direct nearest neighbor
                idx = results[i][0].id
                predictions.append(self.queue_label[idx].item())
            else:
                # Vote among k nearest neighbors
                indices = [hit.id for hit in results[i]]
                knn_cnt = torch.zeros(self.n_class)
                for idx in indices:
                    label = self.queue_label[idx].item()
                    knn_cnt[label] += 1
                predictions.append(knn_cnt.argmax().item())
        
        return predictions

    def cleanup(self):
        """Clean up Milvus resources."""
        if self.collection is not None:
            self.collection.release()
            utility.drop_collection(self.collection_name)
        connections.disconnect("default")


class AnchorStoreMilvusExact(AnchorStoreMilvus):
    """
    Exact KNN search using Milvus FLAT index (brute force).
    Pre-computes log(k_i), uses exact nearest neighbor search.
    """
    def __init__(self, K=1024, dim=50257, knn=1, n_class=2, collection_name="knn_anchors_exact"):
        super().__init__(K, dim, knn, n_class, use_ann=False, collection_name=collection_name)


class AnchorStoreMilvusANN(AnchorStoreMilvus):
    """
    Approximate NN search using Milvus IVF index (faster but approximate).
    Pre-computes log(k_i), uses ANN search for speed.
    """
    def __init__(self, K=1024, dim=50257, knn=1, n_class=2, collection_name="knn_anchors_ann"):
        super().__init__(K, dim, knn, n_class, use_ann=True, collection_name=collection_name)
