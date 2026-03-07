#!/bin/bash

# Demo script for ImageNet with ANN (Approximate Nearest Neighbors)
# This script runs KNN-OOD detection using ANN for faster inference

echo "=================================================="
echo "Running KNN-OOD with ANN (HNSW method) on ImageNet"
echo "=================================================="

# Run with HNSW method (Hierarchical Navigable Small World)
# Increased efSearch for better accuracy
python run_imagenet.py \
    --in-dataset imagenet \
    --name resnet50-supcon \
    --out-datasets inat sun50 places50 dtd \
    --use-ann \
    --ann-method hnsw \
    --hnsw-M 64 \
    --hnsw-efSearch 256
