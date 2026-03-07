# KNN Prompting

KNN-based in-context learning for LLMs, optimized with **Intermediate Results Reuse** (pre-computed log(k)).

## Optimization

| Component        | Baseline                        | Optimized                        |
| ---------------- | ------------------------------- | -------------------------------- |
| KL Divergence    | Compute log(k_i) at every query | Pre-compute log(k_i) once, reuse |
| Spark Processing | Per-query sequential            | Batch RDD parallel               |

## Quick Start

```bash
pip install torch transformers tqdm

# In-Memory: baseline vs optimized
bash run_comparison.sh

# Milvus
bash run_milvus_comparison.sh

# Spark
bash run_spark_comparison.sh
```

### Example command

```bash
python knn_prompting.py \
  --llm_dir /path/to/gpt2-xl \
  --data_dir ./data \
  --dataset sst2 \
  --seed 13 \
  --n_train_shot 32 \
  --n_demo_shot 4 \
  --knn 3 \
  --optimization opt_v2 \
  --output_dir ./output
```

## Data

- **Datasets**: SST-2, AGNews, SUBJ, CR, etc. — included in `data/` or auto-downloaded
- **LLM**: GPT-2 XL — download via HuggingFace (`gpt2-xl`) or provide local path

## Key Files

| File                     | Description                                    |
| ------------------------ | ---------------------------------------------- |
| `knn_prompting.py`       | Main entry point                               |
| `utils/anchor.py`        | Baseline AnchorStore (runtime log computation) |
| `utils/anchor_reuse.py`  | Optimized: pre-computed log(k) reuse           |
| `utils/anchor_milvus.py` | Milvus backend                                 |
| `utils/anchor_spark.py`  | Spark backend (batch RDD processing)           |
