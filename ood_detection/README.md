# KNN-OOD — Out-of-Distribution Detection

**Task**: OOD Detection (Table 1)  
**Invariants**: IV1 (Approximation → ANN with GPU)  
**Bottleneck Profile**: B1: 86.5±8.5%, Total: 86.5±8.5%

## IV-Aligned Implementation

| Invariant | Original Execution          | IV-Aligned Execution                       |
| --------- | --------------------------- | ------------------------------------------ |
| IV1       | Exact KNN (FAISS Flat, GPU) | ANN (FAISS IVF, nlist=1000, nprobe=5, GPU) |

## Datasets

| Dataset      | Samples   | Dim | Acquisition                                                                             |
| ------------ | --------- | --- | --------------------------------------------------------------------------------------- |
| ImageNet-1K  | 1,281,167 | 512 | Manual download from [image-net.org](https://www.image-net.org/)                        |
| CIFAR-10     | 50,000    | 512 | Auto-download (`torchvision`)                                                           |
| OOD datasets | —         | —   | Places50, iNaturalist, SUN, DTD (see [KNN-OOD paper](https://arxiv.org/abs/2204.06507)) |

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
# Download pre-trained model (ResNet-50 SupCon) — see shell scripts for URLs
```

### Step 1: Feature Extraction

```bash
# CIFAR-10
python feat_extract.py --in-dataset cifar10

# ImageNet (large-scale, memory-mapped)
python feat_extract_largescale.py --in-dataset imagenet
```

### Experiment 1: End-to-End (Table 3)

```bash
# ImageNet — Original
python run_imagenet.py --name resnet50-supcon --in-dataset imagenet \
  --out-datasets inat sun50 places50 dtd --use-faiss-gpu --seed 1

# ImageNet — IV-Aligned (IV1: ANN)
python run_imagenet.py --name resnet50-supcon --in-dataset imagenet \
  --out-datasets inat sun50 places50 dtd --use-faiss-gpu \
  --use-ann --ann-method ivf --nlist 1000 --nprobe 5 --seed 1

# Paper results: 64.07s → 36.21s (-43.48%)

# CIFAR-10
python run_cifar10_knnood.py --use-faiss-gpu
# Paper results: 1.758s → 0.914s (-48.01%)
```

### Experiment 5: Cross-Setup (Table 7)

```bash
# Milvus
python run_milvus_benchmark.py

# Spark
python run_imagenet_spark_lsh_optimized.py
```

## Key Files

| File                                  | Description                                |
| ------------------------------------- | ------------------------------------------ |
| `run_imagenet.py`                     | Main KNN-OOD with IV1 (ANN IVF/HNSW + GPU) |
| `run_cifar10_knnood.py`               | CIFAR-10 benchmark                         |
| `feat_extract.py`                     | Feature extraction (CIFAR-10)              |
| `feat_extract_largescale.py`          | Feature extraction (ImageNet, mmap)        |
| `run_milvus_benchmark.py`             | Milvus vector-database setup               |
| `run_imagenet_spark_lsh_optimized.py` | Spark distributed setup                    |
