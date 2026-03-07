# DEFT-UCS — Coreset Selection

**Task**: Coreset Selection (Table 1)  
**Invariants**: IV1 (Approximation → ANN), IV2 (Reuse → Distance Reuse), IV3 (Partial Order → Top-k)  
**Bottleneck Profile**: B1: 98.7±0.9%, B2: 0.7±0.3%, B3: 0.2±0.2%, Total: 99.5±0.5%

## IV-Aligned Implementation

| Invariant | Original Execution                          | IV-Aligned Execution                             |
| --------- | ------------------------------------------- | ------------------------------------------------ |
| IV1       | Exact KMeans (FAISS Flat)                   | ANN KMeans (FAISS IVF, nlist=√K, nprobe=nlist/2) |
| IV2       | Recompute cosine distances in ranking stage | Reuse L2 distances from KMeans assignment        |
| IV3       | Full argsort over all samples               | Top-k selection via `argpartition`               |

## Datasets

| Dataset   | Samples | Dim | Acquisition                              |
| --------- | ------- | --- | ---------------------------------------- |
| CoEDIT    | 69,071  | 768 | `python prepare_data.py` (auto-download) |
| WikiLarge | 148,843 | 768 | `python prepare_data.py` (auto-download) |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
python prepare_data.py
```

### Experiment 1: End-to-End (Table 3)

```bash
# Selection + Fine-tune + Evaluation: original vs IV-aligned
python run_finetune_comparison.py --setting both

# Paper results: CoEDIT 24.98s → 14.51s (-41.91%), WikiLarge 135.64s → 79.43s (-41.44%)
```

### Experiment 2: Ablation (Table 4)

```bash
python run_ablation.py
# Tests: Original / +IV1 / +IV2 / +IV3 / +All
```

### Experiment 5: Vector-Database Setup (Table 7)

```bash
python run_milvus_comparison.py
```

### Selection-Only Benchmark

```bash
python run_comparison.py
```

## Key Files

| File                         | Description                                             |
| ---------------------------- | ------------------------------------------------------- |
| `coreset_selection.py`       | Core selection: original and IV-aligned implementations |
| `run_finetune_comparison.py` | E2E: selection → fine-tune → evaluation                 |
| `run_comparison.py`          | Selection-only time benchmark                           |
| `run_ablation.py`            | Ablation: IV1/IV2/IV3 individually and combined         |
| `run_milvus_comparison.py`   | Milvus vector-database setup                            |
| `evaluate.py`                | SARI, BLEU, ROUGE-L, FKGL metrics                       |
| `prepare_data.py`            | Data preparation                                        |

## Key Arguments

| Argument      | Description                                                | Default               |
| ------------- | ---------------------------------------------------------- | --------------------- |
| `--setting`   | `baseline` (original), `optimized` (IV-aligned), or `both` | `both`                |
| `--eval-only` | Only run evaluation (skip training)                        | False                 |
| `--model`     | Model name                                                 | `google/flan-t5-base` |
