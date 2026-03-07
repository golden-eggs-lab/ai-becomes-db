# CAL — Contrastive Active Learning

**Task**: Active Learning (Table 1)  
**Invariants**: IV1 (Approximation → ANN), IV2 (Reuse → Probability Reuse)  
**Bottleneck Profile**: B1: 77.0±10.0%, B2: 10.5±6.5%, Total: 87.5±3.5%

## IV-Aligned Implementation

| Invariant | Original Execution                                                 | IV-Aligned Execution                         |
| --------- | ------------------------------------------------------------------ | -------------------------------------------- |
| IV1       | Exact KNN (brute-force)                                            | ANN (sklearn BallTree, leaf_size=40)         |
| IV2       | Recompute softmax probabilities for labeled samples at every query | Cache and reuse probabilities across queries |

## Datasets

| Dataset | Samples | Dim | Metric   | Acquisition        |
| ------- | ------- | --- | -------- | ------------------ |
| SST-2   | 67,349  | 768 | Accuracy | `bash get_data.sh` |
| IMDB    | 25,000  | 768 | Accuracy | `bash get_data.sh` |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
bash get_data.sh
```

### Exp 1: End-to-End (Table 3)

```bash
bash run_optimization_comparison.sh
# Paper: SST-2 1472.87s → 283.21s (-80.77%), IMDB 53.75s → 42.48s (-20.97%)
```

### Exp 2: Ablation (Table 4)

```bash
bash run_ablation.sh
# Tests: Original / +IV1 / +IV2 / +All
```

### Exp 4: Overhead (Table 6)

```bash
python measure_cache_memory.py
# Reports: peak cache size for probability reuse
```

### Exp 5: Cross-Setup (Table 7)

```bash
bash run_milvus_comparison.sh       # Milvus vector-database
python benchmark_spark_knn.py       # Spark distributed
```

### Exp 7: Dataset Size (Figure 4 top)

```bash
bash run_imdb_scaling.sh
# Varies dataset size ratio, reports recall + task performance + runtime ratio
```

### Exp 8: Cache Capacity (Figure 4 bottom-right)

```bash
python run_cache_benchmark_multi.py --algorithms CAL
python run_cache_benchmark_ratio.py
# Multi-algorithm cache benchmark covering CAL, KNNPrompting, SCIP, CRAIG, SemDeDup
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparam_cal.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

### View Results

```bash
python show_acc.py
python show_acc.py --ablation
```

## Key Files

| File                             | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `run_al.py`                      | Main active learning experiment              |
| `acquisition/cal.py`             | CAL with IV1 (ANN) + IV2 (probability reuse) |
| `run_optimization_comparison.sh` | E2E comparison (Exp 1)                       |
| `run_ablation.sh`                | Ablation study (Exp 2)                       |
| `measure_cache_memory.py`        | Overhead measurement (Exp 4)                 |
| `benchmark_spark_knn.py`         | Spark distributed (Exp 5)                    |
| `run_milvus_comparison.sh`       | Milvus (Exp 5)                               |
| `run_imdb_scaling.sh`            | Dataset size ratio (Exp 7)                   |
| `run_cache_benchmark_multi.py`   | Cache capacity sweep (Exp 8)                 |
| `run_ann_hyperparam_cal.py`      | ANN sweep (Exp 9)                            |
