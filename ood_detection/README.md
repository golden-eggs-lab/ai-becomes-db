# KNN-OOD — Out-of-Distribution Detection

**Task**: OOD Detection (Table 1)  
**Invariants**: IV1 (Approximation → ANN with GPU)  
**Bottleneck Profile**: B1: 86.5±8.5%, Total: 86.5±8.5%

## IV-Aligned Implementation

| Invariant | Original Execution          | IV-Aligned Execution                       |
| --------- | --------------------------- | ------------------------------------------ |
| IV1       | Exact KNN (FAISS Flat, GPU) | ANN (FAISS IVF, nlist=1000, nprobe=5, GPU) |

## Datasets

| Dataset     | Samples   | Dim | Metric                  | Acquisition                                                   |
| ----------- | --------- | --- | ----------------------- | ------------------------------------------------------------- |
| ImageNet-1K | 1,281,167 | 512 | FPR@95%TPR, AUROC, AUIN | Manual download ([image-net.org](https://www.image-net.org/)) |
| CIFAR-10    | 50,000    | 512 | FPR@95%TPR, AUROC, AUIN | Auto-download (`torchvision`)                                 |
| OOD sets    | —         | —   | —                       | Places50, iNaturalist, SUN, DTD                               |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
# Download pre-trained ResNet-50 (SupCon) — see shell scripts for URLs
```

### Step 1: Feature Extraction

```bash
python feat_extract.py --in-dataset cifar10
python feat_extract_largescale.py --in-dataset imagenet
```

### Exp 1: End-to-End (Table 3)

```bash
# ImageNet — Original vs IV-Aligned
python run_imagenet.py --name resnet50-supcon --in-dataset imagenet \
  --out-datasets inat sun50 places50 dtd --use-faiss-gpu --seed 1

python run_imagenet.py --name resnet50-supcon --in-dataset imagenet \
  --out-datasets inat sun50 places50 dtd --use-faiss-gpu \
  --use-ann --ann-method ivf --nlist 1000 --nprobe 5 --seed 1
# Paper: ImageNet 64.07s → 36.21s (-43.48%)

# CIFAR-10
python run_cifar10_knnood.py --use-faiss-gpu
# Paper: 1.758s → 0.914s (-48.01%)
```

### Exp 4: Overhead (Table 6)

Build/query time is logged within `run_imagenet.py` output.

### Exp 5: Cross-Setup (Table 7)

```bash
python run_milvus_benchmark.py                      # Milvus
bash run_spark_comparison.sh                        # Spark (ImageNet, 5% sample)
```

### Exp 7: Dataset Size (Figure 4 top)

```bash
bash run_scaling.sh
# Varies dataset size ratio, reports recall + task performance + runtime ratio
```

### Exp 9: ANN Sweep (Figure 5)

```bash
python run_ann_hyperparam_ood.py
# Sweeps nlist/nprobe, reports selection recall + search time ratio
```

## Key Files

| File                                  | Description                                      |
| ------------------------------------- | ------------------------------------------------ |
| `run_imagenet.py`                     | Main KNN-OOD with IV1 (ANN IVF + GPU) (Exp 1, 4) |
| `run_cifar10_knnood.py`               | CIFAR-10 benchmark (Exp 1)                       |
| `feat_extract.py`                     | Feature extraction (CIFAR-10)                    |
| `feat_extract_largescale.py`          | Feature extraction (ImageNet, mmap)              |
| `run_milvus_benchmark.py`             | Milvus (Exp 5)                                   |
| `run_spark_comparison.sh`             | Spark comparison (Exp 5)                         |
| `run_imagenet_spark_lsh_optimized.py` | Standalone Spark LSH logic reference function    |
| `run_scaling.sh`                      | Dataset size ratio (Exp 7)                       |
| `run_ann_hyperparam_ood.py`           | ANN sweep (Exp 9)                                |
