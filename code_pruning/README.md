# SCIP — Code Data Pruning

**Task**: Code Data Pruning (Table 1)  
**Invariants**: IV1 (Approximation → ANN), IV2 (Reuse → Distance Reuse), IV3 (Partial Order → Top-k)  
**Bottleneck Profile**: B1: 40.0±18.0%, B2: 1.0±0.0%, B3: 1.0±0.0%, Total: 42.0±18.0%

## IV-Aligned Implementation

| Invariant | Original Execution                                   | IV-Aligned Execution                      |
| --------- | ---------------------------------------------------- | ----------------------------------------- |
| IV1       | Exact centroid search (FAISS Flat)                   | ANN (FAISS IVF, nlist=√K, nprobe=nlist/2) |
| IV2       | Recompute sample–centroid distances in ranking stage | Reuse distances from assignment stage     |
| IV3       | Full argsort over all samples                        | Top-k selection via `argpartition`        |

## Datasets

| Dataset              | Samples | Dim | Acquisition                        |
| -------------------- | ------- | --- | ---------------------------------- |
| CodeSearchNet-Python | 453,499 | 768 | Auto-download (`datasets` library) |
| CodeSearchNet-Java   | 491,506 | 768 | Auto-download (`datasets` library) |

```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack", data_dir="data/python")
```

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
```

### Experiment 1: End-to-End (Table 3)

```bash
# Run all modes (original + IV-aligned), 1000 clusters, 5 runs
python run_inmemory_ablation_v2.py --mode all --n_clusters 1000 --n_runs 5

# Paper results: Python 50.05s → 23.57s (-52.91%), Java 63.54s → 29.83s (-53.05%)
```

### Experiment 2: Ablation (Table 4)

```bash
# Individual invariants
python run_inmemory_ablation_v2.py --mode baseline --n_clusters 1000 --n_runs 5
python run_inmemory_ablation_v2.py --mode ann --n_clusters 1000 --n_runs 5
python run_inmemory_ablation_v2.py --mode reuse --n_clusters 1000 --n_runs 5
python run_inmemory_ablation_v2.py --mode topk --n_clusters 1000 --n_runs 5
```

### Experiment 5: Cross-Setup (Table 7)

```bash
# Milvus
python run_milvus_simple.py --n_clusters 1000

# Spark
python run_spark_fixed.py --n_clusters 1000
```

## Key Files

| File                          | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| `main.py`                     | Core SCIP algorithm                               |
| `embeddings.py`               | Code embedding generation                         |
| `run_inmemory_ablation_v2.py` | In-memory ablation: Original / +IV1 / +IV2 / +IV3 |
| `run_milvus_simple.py`        | Milvus vector-database setup                      |
| `run_spark_fixed.py`          | Spark distributed setup                           |
| `data_loading.py`             | Dataset loading utilities                         |
