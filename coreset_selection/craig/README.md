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

| Dataset       | Samples | Dim | Acquisition                                 |
| ------------- | ------- | --- | ------------------------------------------- |
| MNIST         | 60,000  | 10  | Auto-download (`tensorflow.keras.datasets`) |
| Fashion-MNIST | 60,000  | 10  | Auto-download (`tensorflow.keras.datasets`) |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
```

### Experiment 1: End-to-End (Table 3)

```bash
# In-memory: original vs IV-aligned, 3 runs
python run_craig_benchmark.py --backend memory --mode both --n_runs 3

# Expected output: runtime and accuracy for MNIST/Fashion-MNIST
# Paper results: MNIST 53.60s → 48.17s (-10.13%), Fashion-MNIST 57.72s → 49.30s (-14.59%)
```

### Experiment 5: Cross-Setup (Table 7)

```bash
# Milvus
python run_craig_benchmark.py --backend milvus --mode both --n_runs 3

# Spark
python run_craig_benchmark.py --backend spark --mode both --n_runs 3
```

## Key Files

| File                                          | Description                                                 |
| --------------------------------------------- | ----------------------------------------------------------- |
| `lazy_greedy.py`                              | Core greedy selection with IV1 (ANN) + IV2 (residual reuse) |
| `run_craig_benchmark.py`                      | Benchmark across in-memory / Milvus / Spark                 |
| `compare_fashion_mnist_ann.py`                | Original vs IV-aligned comparison                           |
| `mnist.py`                                    | Data loading                                                |
| `logistic.py`, `resnet.py`, `train_resnet.py` | Model training                                              |
