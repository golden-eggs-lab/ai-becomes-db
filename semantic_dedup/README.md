# SemDeDup / FairDeDup — Semantic Deduplication

**Task**: Semantic Deduplication (Table 1)  
**Invariants**: IV1 (Approximation → ANN), IV2 (Reuse → Search Results Reuse)  
**Bottleneck Profile**:

- SemDeDup: B2: 82.5±12.5%, Total: 82.5±12.5%
- FairDeDup: B2: 89.5±9.5%, Total: 89.5±9.5%

## IV-Aligned Implementation

| Invariant | Original Execution                            | IV-Aligned Execution                            |
| --------- | --------------------------------------------- | ----------------------------------------------- |
| IV1       | Exact pairwise similarity (FAISS Flat, O(n²)) | ANN (FAISS IVF, nlist=100, nprobe=10)           |
| IV2       | Recompute similarity for each ε threshold     | Compute once, reuse search results across all ε |

## Methods

| Method             | Description                                 |
| ------------------ | ------------------------------------------- |
| **SemDeDup** [1]   | Greedy sequential deduplication             |
| **FairDeDup** [56] | Fairness-aware deduplication via Union-Find |

## Datasets

| Dataset   | Samples | Dim | Acquisition                   |
| --------- | ------- | --- | ----------------------------- |
| CIFAR-10  | 50,000  | 512 | Auto-download (`torchvision`) |
| CIFAR-100 | 50,000  | 512 | Auto-download (`torchvision`) |

Embeddings are pre-computed using CLIP:

```bash
python experiments/compute_embeddings.py --dataset cifar10 --output embeddings/cifar10_embeddings.npy
```

## Reproducing Results

### Setup

```bash
conda env create -f environment.yml && conda activate semdedup
# Or: pip install numpy pandas torch torchvision faiss-cpu tqdm
```

### Experiment 1: End-to-End (Table 3)

```bash
python experiments/run_cifar10_experiment.py --seed 42 --run-id run1

# Paper results:
# SemDeDup CIFAR-10: 86.51s → 0.37s (-99.57%)
# FairDeDup CIFAR-10: 543.81s → 14.74s (-97.29%)
```

### Experiment 2: Ablation (Table 4)

```bash
bash scripts/run_ablation.sh
# Tests 4 configurations: Original / +IV1 / +IV2 / +IV1+IV2
# SemDeDup: 86.51s → 1.13s (+IV1) → 28.91s (+IV2) → 0.37s (+All)
```

### Experiment 5: Cross-Setup (Table 7)

```bash
# Milvus
python experiments/run_cifar10_milvus_experiment.py --seed 42

# Spark
python run_semdedup_spark.py
```

## Experiment Configurations

8 configurations (2 methods × 4 settings):

| Configuration   | IV1 (ANN) | IV2 (Reuse) | Paper Label |
| --------------- | --------- | ----------- | ----------- |
| `Exact_NoCache` | ❌        | ❌          | Original    |
| `Exact_Cache`   | ❌        | ✅          | +IV2        |
| `ANN_NoCache`   | ✅        | ❌          | +IV1        |
| `ANN_Cache`     | ✅        | ✅          | IV-Aligned  |

## Key Files

| File                                           | Description                           |
| ---------------------------------------------- | ------------------------------------- |
| `experiments/run_cifar10_experiment.py`        | FAISS in-memory: all 8 configurations |
| `experiments/run_cifar10_milvus_experiment.py` | Milvus vector-database setup          |
| `experiments/compute_embeddings.py`            | CLIP embedding generation             |
| `run_semdedup_spark.py`                        | Spark distributed setup               |
| `scripts/run_ablation.sh`                      | Ablation study                        |
