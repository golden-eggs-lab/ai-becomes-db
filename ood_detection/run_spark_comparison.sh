#!/bin/bash
# Spark KNN-OOD Benchmark Script
# Runs both Baseline (ExactKNN) and Optimized (RandomProjection-ANN) experiments
# using 5% subsampled training data for distributed processing evaluation.

echo "============================================================"
echo "Starting Spark KNN-OOD Benchmark (5% Sampling)"
echo "Start time: $(date)"
echo "============================================================"

# Ensure we're in the right directory
cd "$(dirname "$0")"

SUBSAMPLE=0.05  # 5% = ~64k training vectors
BATCH_SIZE=500  # Query batch size

# Baseline: Spark ExactKNN
echo ""
echo "============================================================"
echo "Running: Baseline (Spark-ExactKNN)"
echo "============================================================"
python -u run_imagenet.py \
    --name resnet50-supcon \
    --in-dataset imagenet \
    --out-datasets inat sun50 places50 dtd \
    --use-spark \
    --spark-master "local[*]" \
    --spark-partitions 100 \
    --spark-subsample $SUBSAMPLE \
    --spark-batch-size $BATCH_SIZE \
    --seed 1

echo ""
echo "============================================================"
echo "Baseline complete at: $(date)"
echo "============================================================"

# Optimized: Spark RandomProjection-ANN (LSH)
echo ""
echo "============================================================"
echo "Running: Optimized (Spark-RandomProjection-ANN)"
echo "============================================================"
python -u run_imagenet.py \
    --name resnet50-supcon \
    --in-dataset imagenet \
    --out-datasets inat sun50 places50 dtd \
    --use-spark \
    --spark-master "local[*]" \
    --spark-partitions 100 \
    --spark-subsample $SUBSAMPLE \
    --spark-batch-size $BATCH_SIZE \
    --spark-use-lsh \
    --seed 1

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "End time: $(date)"
echo "============================================================"
