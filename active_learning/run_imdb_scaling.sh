#!/bin/bash
# CAL IMDB: Dataset Size Scaling Experiment
# Vary training pool via --cap_training_pool: 22500 (1.0), 15750 (0.7), 11250 (0.5), 6750 (0.3)
# Baseline (exact KNN, no cache) × 1 run, Optimized (ANN + cache) × 2 runs per ratio
# Ratio=1.0: reuse existing results from al_imdb_bert_cal_imdb_full_*
# Env: cal
# WARNING: ~1 hour per run. Run last.

set -e

eval "$(conda shell.bash hook)"
conda activate cal

export CUDA_VISIBLE_DEVICES=0

GPU=0
SEED=42
DATASET="imdb"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="imdb_scaling_${TIMESTAMP}.log"

echo "=========================================" | tee -a "${LOG_FILE}"
echo "CAL IMDB: Dataset Size Scaling"           | tee -a "${LOG_FILE}"
echo "Started at: $(date)"                      | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"

run_experiment() {
    local cap=$1
    local ratio=$2
    local use_ann=$3
    local cache_prob=$4
    local run_num=$5
    local mode=$6  # baseline or optimized

    local indicator="scaling_cap${cap}_${mode}_run${run_num}"
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> [ratio=${ratio}, cap=${cap}] ${mode} run${run_num} (ANN=${use_ann}, Cache=${cache_prob})" | tee -a "${LOG_FILE}"
    echo "    Started at: $(date)" | tee -a "${LOG_FILE}"

    local start_time=$(date +%s)

    CUDA_VISIBLE_DEVICES=${GPU} python -u run_al.py \
        --dataset_name ${DATASET} \
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
        --cap_training_pool ${cap} \
        --use_sklearn_ann ${use_ann} \
        --cache_probabilities ${cache_prob} \
        2>&1 | tee -a "${LOG_FILE}"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "[ratio=${ratio}] ${mode} run${run_num} completed in ${duration}s" | tee -a "${LOG_FILE}"
}

# Ratios: 0.7 (15750), 0.5 (11250), 0.3 (6750)
# Ratio=1.0 (22500) already exists as imdb_full_baseline/optimized

CAPS=(15750 11250 6750)
RATIOS=(0.7 0.5 0.3)

for i in "${!CAPS[@]}"; do
    cap=${CAPS[$i]}
    ratio=${RATIOS[$i]}
    
    echo "" | tee -a "${LOG_FILE}"
    echo "===== ratio=${ratio} (cap=${cap}) =====" | tee -a "${LOG_FILE}"
    
    # Skip ratio=0.7 baseline (already completed)
    if [ "${cap}" != "15750" ]; then
        # Baseline × 1
        run_experiment ${cap} ${ratio} "False" "False" 1 "baseline"
    else
        echo "  [SKIP] ratio=0.7 baseline already completed" | tee -a "${LOG_FILE}"
    fi
    
    # Optimized × 2
    run_experiment ${cap} ${ratio} "True" "True" 1 "optimized"
    run_experiment ${cap} ${ratio} "True" "True" 2 "optimized"
done

echo "" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "All CAL IMDB scaling runs completed at: $(date)" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
