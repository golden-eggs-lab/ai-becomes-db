# CRAIG — Coreset Selection

Coresets for Accelerating Incremental Gradient descent (CRAIG), optimized with **ANN + Residual Reuse**.

## Optimization

| Component               | Baseline                | Optimized                         |
| ----------------------- | ----------------------- | --------------------------------- |
| Nearest Neighbor Search | Exact pairwise distance | ANN (FAISS IVF / NearPy)          |
| Greedy Selection        | Recompute all residuals | Reuse residuals across iterations |

## Quick Start

```bash
pip install -r requirements.txt

# Run benchmark (In-Memory, Milvus, Spark)
python run_craig_benchmark.py --backend memory --mode both --n_runs 3

# Compare baseline vs optimized on Fashion MNIST
python compare_fashion_mnist_ann.py
```

## Data

**MNIST / Fashion MNIST** — auto-downloaded via `tensorflow.keras.datasets`.

## Key Files

| File                           | Description                                     |
| ------------------------------ | ----------------------------------------------- |
| `lazy_greedy.py`               | Core greedy selection with ANN + Residual Reuse |
| `run_craig_benchmark.py`       | Benchmark across memory / Milvus / Spark        |
| `compare_fashion_mnist_ann.py` | Baseline vs ANN comparison                      |
| `mnist.py`, `logistic.py`      | Data loading and model training                 |
