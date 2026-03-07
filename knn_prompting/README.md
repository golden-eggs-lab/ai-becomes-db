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

| Dataset | Samples | Dim            | Metric   | Acquisition                        |
| ------- | ------- | -------------- | -------- | ---------------------------------- |
| SST-2   | 67,349  | 50,257 (vocab) | Accuracy | Included in `data/` or HuggingFace |
| AGNews  | 120,000 | 50,257 (vocab) | Accuracy | Included in `data/` or HuggingFace |

**LLM**: GPT-2 XL — download via HuggingFace (`gpt2-xl`) or provide local path.

## Reproducing Results

### Setup

```bash
pip install torch transformers tqdm
```

### Exp 1: End-to-End (Table 3)

```bash
# Original
python knn_prompting.py \
  --llm_dir /path/to/gpt2-xl --data_dir ./data --dataset sst2 \
  --seed 13 --n_train_shot 32 --n_demo_shot 4 --knn 3 \
  --optimization baseline --output_dir ./output

# IV-Aligned (IV2: pre-computed log reuse)
python knn_prompting.py \
  --llm_dir /path/to/gpt2-xl --data_dir ./data --dataset sst2 \
  --seed 13 --n_train_shot 32 --n_demo_shot 4 --knn 3 \
  --optimization opt_v2 --output_dir ./output

# Paper: SST-2 111.03s → 70.74s (-36.29%), AGNews 144.75s → 71.01s (-50.94%)
```

### Exp 5: Cross-Setup (Table 7)

```bash
bash run_milvus_comparison.sh       # Milvus
bash run_spark_comparison.sh        # Spark
```

### Exp 8: Cache Capacity (Figure 4 bottom-right)

KNN Prompting is covered by the multi-algorithm cache benchmark:

```bash
# Run from active_learning/ directory
python run_cache_benchmark_multi.py --algorithms KNNPrompting
```

### Batch Comparison

```bash
bash run_comparison.sh              # In-memory: original vs IV-aligned
```

## Key Files

| File                       | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `knn_prompting.py`         | Main entry point (Exp 1, 5)                         |
| `utils/anchor.py`          | Original AnchorStore (computes log(k_i) at runtime) |
| `utils/anchor_reuse.py`    | IV-aligned: pre-computed log(k_i) reuse             |
| `utils/anchor_milvus.py`   | Milvus (Exp 5)                                      |
| `utils/anchor_spark.py`    | Spark (Exp 5)                                       |
| `run_comparison.sh`        | In-memory comparison                                |
| `run_milvus_comparison.sh` | Milvus comparison                                   |
| `run_spark_comparison.sh`  | Spark comparison                                    |
