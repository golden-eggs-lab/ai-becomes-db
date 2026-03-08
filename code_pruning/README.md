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

### Data Preparation & Embeddings

The Python benchmark scripts (like `run_inmemory_ablation_v2.py`) assume you have pre-computed the code vectors and saved them to `./experiments/aligned/embeddings.npy`. If you do not have these, you must generate them by running the embedding utility functions in `embeddings.py` on the `CodeSearchNet-Python` dataset.

_(Note: The Java End-to-End script, `run_java_experiment.py`, automatically handles downloading the dataset and generating the embeddings on the fly if they are missing)._

### Exp 1: End-to-End (Table 3)

The default scripts run on the **CodeSearchNet-Python** dataset using pre-computed embeddings:

```bash
bash run_e2e_ablation.sh
# Paper: Python 50.05s → 23.57s (-52.91%)
```

To reproduce the full end-to-end pipeline (downloading texts, embedding generation, SCIP, and downstream CodeBERT evaluation) on **CodeSearchNet-Java**:

```bash
python run_java_experiment.py --max_samples 0 --n_gpus 8
# Paper: Java 63.54s → 29.83s (-53.05%)
```

### Exp 2: Ablation (Table 4)

```bash
bash run_e2e_ablation.sh
# Tests: baseline, ann, reuse, and topk optimizations
```

### Exp 3: Component-Level Breakdown (Table 5)

Component-level breakdown timings (e.g., KMeans, CentroidSearch, Distances, Top-K) are natively reported in the console output during the execution of the ablation script (`run_e2e_ablation.sh`).

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
python run_cache_benchmark_ratio.py --algorithms SCIP
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparam_codepruning.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

## Key Files

| File                                | Description                               |
| ----------------------------------- | ----------------------------------------- |
| `run_e2e_ablation.sh`               | E2E + ablation (Exp 1, 2) on Python       |
| `run_java_experiment.py`            | Full E2E pipeline (Exp 1) on Java dataset |
| `run_inmemory_ablation_v2.py`       | Ablation internal Python backend          |
| `measure_build_query_cache.py`      | Overhead measurement (Exp 4)              |
| `run_milvus_simple.py`              | Milvus (Exp 5)                            |
| `run_spark_fixed.py`                | Spark (Exp 5)                             |
| `run_scip_scaling.py`               | Dataset size ratio (Exp 7)                |
| `run_ann_hyperparam_codepruning.py` | ANN sweep (Exp 9)                         |
| `embeddings.py`                     | Code embedding generation                 |
| `data_loading.py`                   | Dataset loading                           |
