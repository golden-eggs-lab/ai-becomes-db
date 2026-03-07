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

| Dataset | Samples | Dim | Acquisition        |
| ------- | ------- | --- | ------------------ |
| SST-2   | 67,349  | 768 | `bash get_data.sh` |
| IMDB    | 25,000  | 768 | `bash get_data.sh` |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
bash get_data.sh
```

### Experiment 1: End-to-End (Table 3)

```bash
# Original vs IV-aligned comparison
bash run_optimization_comparison.sh

# Paper results: SST-2 1472.87s → 283.21s (-80.77%), IMDB 53.75s → 42.48s (-20.97%)
```

### Experiment 2: Ablation (Table 4)

```bash
bash run_ablation.sh
```

### Experiment 5: Cross-Setup (Table 7)

```bash
# Milvus
bash run_milvus_comparison.sh

# Spark
python benchmark_spark_knn.py
```

### View Results

```bash
python show_acc.py
python show_acc.py --ablation
```

## Key Arguments

| Argument                | Description                                                  | Default |
| ----------------------- | ------------------------------------------------------------ | ------- |
| `--use_sklearn_ann`     | Enable IV1 (ANN via BallTree)                                | False   |
| `--cache_probabilities` | Enable IV2 (probability reuse)                               | False   |
| `--use_milvus`          | Use Milvus vector-database setup                             | False   |
| `--milvus_index_type`   | Milvus index: FLAT (original) / IVF_FLAT (IV-aligned) / HNSW | FLAT    |

### Example Commands

```bash
# Original
python run_al.py --dataset_name sst-2 --acquisition cal \
    --use_sklearn_ann False --cache_probabilities False

# IV-Aligned (IV1 + IV2)
python run_al.py --dataset_name sst-2 --acquisition cal \
    --use_sklearn_ann True --cache_probabilities True

# Milvus (IV-aligned)
python run_al.py --dataset_name sst-2 --acquisition cal \
    --use_milvus True --milvus_index_type IVF_FLAT --cache_probabilities True
```

## Key Files

| File                             | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `run_al.py`                      | Main active learning experiment              |
| `acquisition/cal.py`             | CAL with IV1 (ANN) + IV2 (probability reuse) |
| `benchmark_spark_knn.py`         | Spark distributed setup                      |
| `run_optimization_comparison.sh` | In-memory: original vs IV-aligned            |
| `run_ablation.sh`                | Ablation study                               |
| `run_milvus_comparison.sh`       | Milvus vector-database setup                 |
