# CRAIG — Coreset Selection

**Task**: Coreset Selection (Table 1)  
**Invariants**: IV1 (Approximation → ANN), IV2 (Reuse → Residual Reuse)  
**Bottleneck Profile**: B1: 80.0±9.0%, B2: 1.5±0.5%, Total: 81.5±8.5%

## IV-Aligned Implementation

| Invariant | Original Execution                                      | IV-Aligned Execution                              |
| --------- | ------------------------------------------------------- | ------------------------------------------------- |
| IV1       | Exact pairwise similarity over gradients                | ANN indexing (NearPy LSH / FAISS IVF)             |
| IV2       | Recompute residual gradient from scratch each iteration | Incremental residual update: R\_{i+1} = R_i - g_j |

## Datasets

| Dataset       | Samples | Dim | Metric   | Acquisition                                 |
| ------------- | ------- | --- | -------- | ------------------------------------------- |
| MNIST         | 60,000  | 10  | Accuracy | Auto-download (`tensorflow.keras.datasets`) |
| Fashion-MNIST | 60,000  | 10  | Accuracy | Auto-download (`tensorflow.keras.datasets`) |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
```

### Exp 1: End-to-End (Table 3)

```bash
python run_craig_benchmark.py --backend memory --mode both --n_runs 3
# Paper: MNIST 53.60s → 48.17s (-10.13%), Fashion-MNIST 57.72s → 49.30s (-14.59%)
```

### Exp 4: Overhead (Table 6)

```bash
python measure_build_query_cache.py
# Reports: build time, query time, peak cache size
```

### Exp 5: Cross-Setup (Table 7)

```bash
python run_craig_benchmark.py --backend milvus --mode both --n_runs 3
python run_craig_benchmark.py --backend spark --mode both --n_runs 3
```

### Exp 6: Memory Constraint (Figure 3)

```bash
python run_memory_experiment_1class.py        # Per-class runtime under memory limits
python run_faiss_memory_experiment.py          # FAISS-based ANN variant
python run_memory_experiment_v2.py             # Extended memory sweep
```

### Exp 8: Cache Capacity (Figure 4 bottom-right)

CRAIG is covered by the multi-algorithm cache benchmark:

```bash
# Run from active_learning/ directory
python run_cache_benchmark_multi.py --algorithms CRAIG
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparam_craig.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

## Key Files

| File                           | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| `lazy_greedy.py`               | Core greedy selection with IV1 (ANN) + IV2 (residual reuse) |
| `run_craig_benchmark.py`       | E2E benchmark (Exp 1, 5)                                    |
| `compare_fashion_mnist_ann.py` | Original vs IV-aligned comparison                           |
| `measure_build_query_cache.py` | Overhead measurement (Exp 4)                                |
| `run_memory_experiment*.py`    | Memory constraint (Exp 6)                                   |
| `run_ann_hyperparam_craig.py`  | ANN sweep (Exp 9)                                           |
