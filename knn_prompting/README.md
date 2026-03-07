# KNN Prompting — In-Context Learning

**Task**: In-Context Learning (Table 1)  
**Invariants**: IV2 (Reuse → Intermediate Results Reuse)  
**Bottleneck Profile**: B2: 75.0±24.0%, Total: 75.0±24.0%

## IV-Aligned Implementation

| Invariant | Original Execution                          | IV-Aligned Execution                                |
| --------- | ------------------------------------------- | --------------------------------------------------- |
| IV2       | Compute log(k_i) at every query (redundant) | Pre-compute log(k_i) once, reuse across all queries |

In the KL divergence D_KL(p_test ∥ k_i), the term log(k_i) depends only on the fixed anchor distribution and remains unchanged across queries. The IV-aligned execution materializes log(k_i) during anchor construction and reuses it during inference.

## Datasets

| Dataset | Samples | Dim            | Acquisition                        |
| ------- | ------- | -------------- | ---------------------------------- |
| SST-2   | 67,349  | 50,257 (vocab) | Included in `data/` or HuggingFace |
| AGNews  | 120,000 | 50,257 (vocab) | Included in `data/` or HuggingFace |

**LLM**: GPT-2 XL — download via HuggingFace (`gpt2-xl`) or provide local path.

## Reproducing Results

### Setup

```bash
pip install torch transformers tqdm
```

### Experiment 1: End-to-End (Table 3)

```bash
# Original (baseline)
python knn_prompting.py \
  --llm_dir /path/to/gpt2-xl --data_dir ./data --dataset sst2 \
  --seed 13 --n_train_shot 32 --n_demo_shot 4 --knn 3 \
  --optimization baseline --output_dir ./output

# IV-Aligned (IV2: pre-computed log reuse)
python knn_prompting.py \
  --llm_dir /path/to/gpt2-xl --data_dir ./data --dataset sst2 \
  --seed 13 --n_train_shot 32 --n_demo_shot 4 --knn 3 \
  --optimization opt_v2 --output_dir ./output

# Paper results: SST-2 111.03s → 70.74s (-36.29%), AGNews 144.75s → 71.01s (-50.94%)
```

### Experiment 5: Cross-Setup (Table 7)

```bash
# Milvus
bash run_milvus_comparison.sh

# Spark
bash run_spark_comparison.sh
```

### Batch comparison

```bash
bash run_comparison.sh    # In-memory: original vs IV-aligned
```

## Key Files

| File                     | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `knn_prompting.py`       | Main entry point (all setups)                       |
| `utils/anchor.py`        | Original AnchorStore (computes log(k_i) at runtime) |
| `utils/anchor_reuse.py`  | IV-aligned: pre-computed log(k_i) reuse             |
| `utils/anchor_milvus.py` | Milvus vector-database setup                        |
| `utils/anchor_spark.py`  | Spark distributed setup (batch RDD)                 |
