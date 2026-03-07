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

| Dataset              | Samples | Dim | Metric | Acquisition                        |
| -------------------- | ------- | --- | ------ | ---------------------------------- |
| CodeSearchNet-Python | 453,499 | 768 | MRR    | Auto-download (`datasets` library) |
| CodeSearchNet-Java   | 491,506 | 768 | MRR    | Auto-download (`datasets` library) |

```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack", data_dir="data/python")
```

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
```

### Exp 1: End-to-End (Table 3)

```bash
python run_inmemory_ablation_v2.py --mode all --n_clusters 1000 --n_runs 5
# Paper: Python 50.05s → 23.57s (-52.91%), Java 63.54s → 29.83s (-53.05%)
```

### Exp 2: Ablation (Table 4)

```bash
python run_inmemory_ablation_v2.py --mode baseline --n_clusters 1000 --n_runs 5
python run_inmemory_ablation_v2.py --mode ann --n_clusters 1000 --n_runs 5
python run_inmemory_ablation_v2.py --mode reuse --n_clusters 1000 --n_runs 5
python run_inmemory_ablation_v2.py --mode topk --n_clusters 1000 --n_runs 5
```

### Exp 4: Overhead (Table 6)

```bash
python measure_build_query_cache.py
# Reports: build time, query time, peak cache size
```

### Exp 5: Cross-Setup (Table 7)

```bash
python run_milvus_simple.py --n_clusters 1000      # Milvus
python run_spark_fixed.py --n_clusters 1000        # Spark
```

### Exp 7: Dataset Size (Figure 4 top)

```bash
python run_scip_scaling.py
python eval_scaling_mrr.py
# Varies dataset size ratio, reports recall + task performance + runtime ratio
```

### Exp 8: Cache Capacity (Figure 4 bottom-right)

SCIP is covered by the multi-algorithm cache benchmark:

```bash
# Run from active_learning/ directory
python run_cache_benchmark_multi.py --algorithms SCIP
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparam_codepruning.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

## Key Files

| File                                | Description                  |
| ----------------------------------- | ---------------------------- |
| `main.py`                           | Core SCIP algorithm          |
| `run_inmemory_ablation_v2.py`       | E2E + ablation (Exp 1, 2)    |
| `measure_build_query_cache.py`      | Overhead measurement (Exp 4) |
| `run_milvus_simple.py`              | Milvus (Exp 5)               |
| `run_spark_fixed.py`                | Spark (Exp 5)                |
| `run_scip_scaling.py`               | Dataset size ratio (Exp 7)   |
| `run_ann_hyperparam_codepruning.py` | ANN sweep (Exp 9)            |
| `embeddings.py`                     | Code embedding generation    |
| `data_loading.py`                   | Dataset loading              |
