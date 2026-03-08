#!/bin/bash
# Run ablation study on CIFAR-10

set -e

echo "=============================================="
echo "FairDeDup & SemDeDup Ablation Study"
echo "=============================================="

SEED=42

# Instead of just calling run_experiment.py three times,
# we need to be running it in the right mode for the ablation configurations.
# We will use the run_cifar10_experiment.py

echo ""
echo "Running Ablation experiments..."
python experiments/run_cifar10_experiment.py

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="

