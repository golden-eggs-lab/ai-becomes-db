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

| Dataset   | Samples | Dim | Metric                    | Acquisition              |
| --------- | ------- | --- | ------------------------- | ------------------------ |
| CoEDIT    | 69,071  | 768 | SARI, BLEU, ROUGE-L, FKGL | `python prepare_data.py` |
| WikiLarge | 148,843 | 768 | SARI, BLEU, ROUGE-L, FKGL | `python prepare_data.py` |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
python prepare_data.py
```

### Exp 1: End-to-End (Table 3)

```bash
python run_finetune_comparison.py --setting both
# Paper: CoEDIT 24.98s → 14.51s (-41.91%), WikiLarge 135.64s → 79.43s (-41.44%)
```

### Exp 2: Ablation (Table 4)

```bash
python run_ablation.py
# Tests: Original / +IV1 / +IV2 / +IV3 / +All
```

### Exp 5: Cross-Setup (Table 7)

```bash
python run_milvus_comparison.py
```

### Exp 7: Dataset Size (Figure 4 top)

```bash
python run_scaling_experiment.py
# Varies dataset size ratio, reports recall + task performance + runtime ratio
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparam_coreset.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

## Key Files

| File                            | Description                                     |
| ------------------------------- | ----------------------------------------------- |
| `coreset_selection.py`          | Core selection: original and IV-aligned         |
| `run_finetune_comparison.py`    | E2E: selection → fine-tune → evaluation (Exp 1) |
| `run_comparison.py`             | Selection-only time benchmark                   |
| `run_ablation.py`               | Ablation study (Exp 2)                          |
| `run_milvus_comparison.py`      | Milvus vector-database (Exp 5)                  |
| `run_scaling_experiment.py`     | Dataset size ratio (Exp 7)                      |
| `run_ann_hyperparam_coreset.py` | ANN sweep (Exp 9)                               |
| `evaluate.py`                   | SARI, BLEU, ROUGE-L, FKGL metrics               |
