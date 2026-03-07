# SCIP — Code Pruning

Scalable Code data Importance-based Pruning, optimized with **ANN + Distance Reuse + TopK**.

## Optimization

| Component            | Baseline                | Optimized                 |
| -------------------- | ----------------------- | ------------------------- |
| Centroid Search      | Exact (FAISS Flat)      | ANN (FAISS IVF)           |
| Distance Computation | Recompute all distances | Reuse search similarities |
| Sorting              | Full argsort            | TopK via argpartition     |

## Quick Start

```bash
pip install -r requirements.txt

# In-Memory ablation (1000 clusters, 4 experiments)
python run_inmemory_ablation_v2.py --mode all --n_clusters 1000 --n_runs 5

# Milvus
python run_milvus_simple.py --n_clusters 1000

# Spark
python run_spark_fixed.py --n_clusters 1000
```

## Data

**The Stack v1.1** (Python, Java) — auto-downloaded via HuggingFace `datasets` library.

```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack", data_dir="data/python")
```

## Key Files

| File                          | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `main.py`                     | Core SCIP algorithm                                  |
| `embeddings.py`               | Code embedding generation                            |
| `run_inmemory_ablation_v2.py` | In-memory ablation: Baseline / +ANN / +Reuse / +TopK |
| `run_milvus_simple.py`        | Milvus backend                                       |
| `run_spark_fixed.py`          | Spark backend                                        |
| `data_loading.py`             | Dataset loading utilities                            |
