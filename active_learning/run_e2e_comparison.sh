#!/bin/bash
# =============================================================================
# CAL End-to-End Comparison: Original vs IV-Aligned
# 
# Reproduces Experiment 1 (Table 3) for CAL on both SST-2 and IMDB.
#   Original:   exact KNN, no probability reuse
#   IV-Aligned: ANN (BallTree) + probability reuse (IV1 + IV2)
#
# Usage:
#   bash run_e2e_comparison.sh                  # Run all (SST-2 + IMDB)
#   bash run_e2e_comparison.sh --dataset sst-2  # SST-2 only
#   bash run_e2e_comparison.sh --dataset imdb   # IMDB only
#   bash run_e2e_comparison.sh --num_runs 5     # 5 runs instead of 3
# =============================================================================

set -e

# Defaults
GPU=0
SEED=42
DATASETS=("sst-2" "imdb")
NUM_RUNS=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASETS=("$2"); shift 2 ;;
        --num_runs) NUM_RUNS=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="e2e_comparison_${TIMESTAMP}.log"

echo "=========================================" | tee -a "${LOG_FILE}"
echo "CAL E2E: Original vs IV-Aligned"          | tee -a "${LOG_FILE}"
echo "Datasets: ${DATASETS[*]}"                  | tee -a "${LOG_FILE}"
echo "Runs: ${NUM_RUNS}, Seed: ${SEED}, GPU: ${GPU}" | tee -a "${LOG_FILE}"
echo "Started at: $(date)"                       | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"

run_experiment() {
    local dataset=$1
    local exp_name=$2
    local use_ann=$3
    local cache_prob=$4
    local run_num=$5

    local indicator="${exp_name}_run${run_num}"
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> [${dataset}] Run ${run_num}/${NUM_RUNS}: ${exp_name} (ANN=${use_ann}, Cache=${cache_prob})" | tee -a "${LOG_FILE}"
    echo "    Started at: $(date)" | tee -a "${LOG_FILE}"

    local start_time=$(date +%s)

    CUDA_VISIBLE_DEVICES=${GPU} python -u run_al.py \
        --dataset_name ${dataset} \
        --acquisition cal \
        --model_type bert \
        --model_name_or_path bert-base-cased \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 256 \
        --num_train_epochs 3 \
        --seed ${SEED} \
        --init_train_data 1% \
        --acquisition_size 2% \
        --budget 15% \
        --init random \
        --indicator ${indicator} \
        --use_sklearn_ann ${use_ann} \
        --cache_probabilities ${cache_prob} \
        2>&1 | tee -a "${LOG_FILE}"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "[${dataset}] Run ${run_num} ${exp_name} completed in ${duration}s" | tee -a "${LOG_FILE}"
}

for dataset in "${DATASETS[@]}"; do
    echo "" | tee -a "${LOG_FILE}"
    echo "###############################################" | tee -a "${LOG_FILE}"
    echo "Dataset: ${dataset}" | tee -a "${LOG_FILE}"
    echo "###############################################" | tee -a "${LOG_FILE}"

    for run in $(seq 1 ${NUM_RUNS}); do
        # Original: exact KNN, no probability reuse
        run_experiment "${dataset}" "original" "False" "False" ${run}

        # IV-Aligned: ANN (BallTree) + probability reuse
        run_experiment "${dataset}" "iv_aligned" "True" "True" ${run}
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "All runs completed at: $(date)" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"

# Show results
python show_acc.py 2>/dev/null || true
