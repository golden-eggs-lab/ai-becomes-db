#!/bin/bash
# Complete ablation experiments (no MRR)
# Runs 5 experiments total:
#   1. Baseline
#   2. Fully Optimized (IVF + Reuse + TopK)
#   3-5. Three ablations: +ANN, +Reuse, +TopK

set -e

PYTHON=python
SCRIPT=run_inmemory_ablation_v2.py
N_CLUSTERS=1000
N_RUNS=5
OUTPUT_DIR=./experiments/ablation_complete

echo "=========================================="
echo "Complete SCIP Ablation Experiments"
echo "=========================================="
echo ""
echo "Settings: K=${N_CLUSTERS}, runs=${N_RUNS}"
echo ""
echo "Experiments:"
echo "  1. Baseline (Flat + No Reuse + Full Sort)"
echo "  2. Fully Optimized (IVF + Reuse + TopK)"
echo "  3. +ANN only (IVF + No Reuse + Full Sort)"
echo "  4. +Reuse only (Flat + Reuse + Full Sort)"
echo "  5. +TopK only (Flat + No Reuse + TopK)"
echo ""
echo "Metrics for each:"
echo "  - Individual run times (total + breakdown)"
echo "  - Mean ± std for all steps"
echo "  - ANN recall (for ann/optimized modes)"
echo "  - Reuse savings rate (for reuse/optimized modes)"
echo ""

# Experiment 1: Baseline
echo "=========================================="
echo "Experiment 1/5: BASELINE"
echo "  ANN: OFF | Reuse: OFF | TopK: OFF"
echo "=========================================="
$PYTHON $SCRIPT --mode baseline --n_clusters $N_CLUSTERS --n_runs $N_RUNS --output_dir $OUTPUT_DIR

# Experiment 2: Fully Optimized
echo ""
echo "=========================================="
echo "Experiment 2/5: FULLY OPTIMIZED"
echo "  ANN: ON | Reuse: ON | TopK: ON"
echo "=========================================="
$PYTHON $SCRIPT --mode optimized --n_clusters $N_CLUSTERS --n_runs $N_RUNS --output_dir $OUTPUT_DIR

# Experiment 3: +ANN only
echo ""
echo "=========================================="
echo "Experiment 3/5: +ANN ONLY"
echo "  ANN: ON | Reuse: OFF | TopK: OFF"
echo "=========================================="
$PYTHON $SCRIPT --mode ann --n_clusters $N_CLUSTERS --n_runs $N_RUNS --output_dir $OUTPUT_DIR

# Experiment 4: +Reuse only
echo ""
echo "=========================================="
echo "Experiment 4/5: +REUSE ONLY"
echo "  ANN: OFF | Reuse: ON | TopK: OFF"
echo "=========================================="
$PYTHON $SCRIPT --mode reuse --n_clusters $N_CLUSTERS --n_runs $N_RUNS --output_dir $OUTPUT_DIR

# Experiment 5: +TopK only
echo ""
echo "=========================================="
echo "Experiment 5/5: +TOPK ONLY"
echo "  ANN: OFF | Reuse: OFF | TopK: ON"
echo "=========================================="
$PYTHON $SCRIPT --mode topk --n_clusters $N_CLUSTERS --n_runs $N_RUNS --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo "  - baseline_results.json"
echo "  - optimized_results.json"
echo "  - ann_results.json"
echo "  - reuse_results.json"
echo "  - topk_results.json"
echo ""
echo "Each result includes:"
echo "  - Total time: individual runs + mean ± std"
echo "  - Breakdown: individual runs + mean ± std for each step"
echo "  - ANN recall (ann/optimized modes)"
echo "  - Reuse savings rate (reuse/optimized modes)"
