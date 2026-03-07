# KNN-OOD — Out-of-Distribution Detection

KNN-based OOD detection, optimized with **ANN (IVF/HNSW) + GPU acceleration**.

## Optimization

| Component   | Baseline           | Optimized          |
| ----------- | ------------------ | ------------------ |
| KNN Search  | Exact (FAISS Flat) | ANN (IVF with GPU) |
| Computation | CPU-based          | GPU-accelerated    |

## Quick Start

```bash
pip install -r requirements.txt

# 1. Extract features (CIFAR-10)
python feat_extract.py --in-dataset cifar10

# 2. Run KNN-OOD benchmark (In-Memory, GPU)
python run_cifar10_knnood.py --use-faiss-gpu

# 3. Run ImageNet benchmark
bash demo_imagenet_ann.sh

# 4. Milvus benchmark
python run_milvus_benchmark.py

# 5. Spark benchmark
python run_imagenet_spark_lsh_optimized.py
```

## Data

- **CIFAR-10**: Auto-downloaded via torchvision
- **ImageNet**: Download from [ImageNet website](https://www.image-net.org/)
- **OOD datasets** (iNaturalist, SUN, Places, DTD): See [KNN-OOD paper](https://arxiv.org/abs/2204.06507) for download links
- **Pre-trained models**: ResNet-50 (SupCon), download link in shell scripts

## Key Files

| File                                  | Description                                  |
| ------------------------------------- | -------------------------------------------- |
| `run_imagenet.py`                     | Main KNN-OOD with ANN support (IVF/HNSW/GPU) |
| `run_cifar10_knnood.py`               | CIFAR-10 KNN-OOD benchmark                   |
| `feat_extract.py`                     | Feature extraction (CIFAR-10)                |
| `feat_extract_largescale.py`          | Feature extraction (ImageNet, memory-mapped) |
| `run_milvus_benchmark.py`             | Milvus backend benchmark                     |
| `run_imagenet_spark_lsh_optimized.py` | Spark LSH backend                            |
