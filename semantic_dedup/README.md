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

## Datasets

| Dataset   | Samples | Dim | Metric                      | Acquisition                   |
| --------- | ------- | --- | --------------------------- | ----------------------------- |
| CIFAR-10  | 50,000  | 512 | Accuracy@ε∈{0.05,0.10,0.15} | Auto-download (`torchvision`) |
| CIFAR-100 | 50,000  | 512 | Accuracy@ε∈{0.05,0.10,0.15} | Auto-download (`torchvision`) |

Pre-compute CLIP embeddings:

```bash
python experiments/compute_embeddings.py --dataset cifar10 --output embeddings/cifar10_embeddings.npy
```

## Reproducing Results

### Setup

```bash
conda env create -f environment.yml && conda activate semdedup
# Or: pip install numpy pandas torch torchvision faiss-cpu tqdm
```

### Exp 1: End-to-End (Table 3)

```bash
python experiments/run_cifar10_experiment.py --seed 42 --run-id run1
# Paper: SemDeDup CIFAR-10 86.51s → 0.37s (-99.57%), FairDeDup 543.81s → 14.74s (-97.29%)
```

### Exp 2: Ablation (Table 4)

```bash
bash scripts/run_ablation.sh
# SemDeDup: 86.51s → 1.13s (+IV1) → 28.91s (+IV2) → 0.37s (+All)
```

### Exp 4: Overhead (Table 6)

```bash
python measure_semdedup_memory.py
# Reports: build time, query time, peak cache size
```

### Exp 5: Cross-Setup (Table 7)

```bash
python experiments/run_cifar10_milvus_experiment.py --seed 42   # Milvus
python run_semdedup_spark.py                                     # Spark
```

### Exp 7: Dataset Size (Figure 4 top)

```bash
python run_scaling_experiment.py
python eval_scaling_acc.py
# Varies dataset size ratio, reports recall + task performance + runtime ratio
```

### Exp 8: Cache Capacity (Figure 4 bottom-right)

SemDeDup is covered by the multi-algorithm cache benchmark:

```bash
# Run from active_learning/ directory
python run_cache_benchmark_multi.py --algorithms SemDeDup
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparameter_study.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

## Key Files

| File                                           | Description                  |
| ---------------------------------------------- | ---------------------------- |
| `experiments/run_cifar10_experiment.py`        | E2E: all 8 configs (Exp 1)   |
| `scripts/run_ablation.sh`                      | Ablation study (Exp 2)       |
| `measure_semdedup_memory.py`                   | Overhead measurement (Exp 4) |
| `experiments/run_cifar10_milvus_experiment.py` | Milvus (Exp 5)               |
| `run_semdedup_spark.py`                        | Spark (Exp 5)                |
| `run_scaling_experiment.py`                    | Dataset size ratio (Exp 7)   |
| `run_ann_hyperparameter_study.py`              | ANN sweep (Exp 9)            |
