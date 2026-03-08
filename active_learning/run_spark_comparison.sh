#!/bin/bash
# =============================================================================
# Spark KNN Comparison Experiment Script
# Compare: Spark Exact (no cache) vs Spark LSH (with cache)
# 
# Dataset: SST-2
# Settings (aligned with optimization_comparison):
#   - Initial labeled data: 1%
#   - Acquisition per iteration: 2%
#   - Total budget: 15%
#   - Model: BERT-BASE-cased (fine-tuned)
#   - Seed: 42
# =============================================================================

set -e

# Set JAVA_HOME for Spark
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Experiment settings
DATASET="sst-2"
SEED=42
INIT_PERCENT="1%"           # 1% initial labeled data
ACQUISITION_PERCENT="2%"    # 2% per iteration
TOTAL_BUDGET="15%"          # 15% total budget

# GPU to use (single GPU for sequential execution)
GPU=0

# Base output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="experiments/spark_comparison_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT_DIR}"

# Log file
LOG_FILE="${BASE_OUTPUT_DIR}/experiment_log.txt"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Spark KNN Comparison Experiment" | tee -a "${LOG_FILE}"
echo "Dataset: ${DATASET}" | tee -a "${LOG_FILE}"
echo "Seed: ${SEED}" | tee -a "${LOG_FILE}"
echo "Settings: init=${INIT_PERCENT}, acq=${ACQUISITION_PERCENT}, budget=${TOTAL_BUDGET}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU}" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo ""

# Common arguments
BASE_ARGS="--dataset_name ${DATASET} \
  --acquisition cal \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 256 \
  --num_train_epochs 3 \
  --seed ${SEED} \
  --init_train_data ${INIT_PERCENT} \
  --acquisition_size ${ACQUISITION_PERCENT} \
  --budget ${TOTAL_BUDGET} \
  --init random \
  --use_spark True"

# Function to run single experiment
run_experiment() {
    local indicator=$1
    local spark_method=$2
    local cache_prob=$3
    local num_tables=${4:-5}
    local bucket_length=${5:-2.0}
    
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Running: ${indicator}" | tee -a "${LOG_FILE}"
    echo "  Dataset: ${DATASET}, Seed: ${SEED}" | tee -a "${LOG_FILE}"
    echo "  Spark Method: ${spark_method}" | tee -a "${LOG_FILE}"
    echo "  Cache Probabilities: ${cache_prob}" | tee -a "${LOG_FILE}"
    if [ "${spark_method}" == "lsh" ]; then
        echo "  LSH Tables: ${num_tables}, Bucket Length: ${bucket_length}" | tee -a "${LOG_FILE}"
    fi
    echo "  Started at: $(date)" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    # Output directory for this experiment
    local output_dir="${BASE_OUTPUT_DIR}/${indicator}/seed_${SEED}"
    mkdir -p "${output_dir}"
    
    # Log file for this experiment
    local exp_log="${output_dir}/run.log"
    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=${GPU} python -u run_al.py \
        ${BASE_ARGS} \
        --spark_knn_method ${spark_method} \
        --cache_probabilities ${cache_prob} \
        --indicator ${indicator}"
    
    # Add LSH parameters if using LSH
    if [ "${spark_method}" == "lsh" ]; then
        cmd="${cmd} \
        --spark_lsh_num_tables ${num_tables} \
        --spark_lsh_bucket_length ${bucket_length}"
    fi
    
    # Log command
    echo "Command: ${cmd}" | tee -a "${LOG_FILE}"
    
    # Run experiment and capture output
    start_time=$(date +%s)
    
    if eval ${cmd} 2>&1 | tee "${exp_log}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "SUCCESS: ${indicator} (seed ${SEED}) completed in ${duration}s" | tee -a "${LOG_FILE}"
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "FAILED: ${indicator} (seed ${SEED}) after ${duration}s" | tee -a "${LOG_FILE}"
    fi
    
    # Copy results to output directory
    local exp_result_dir="experiments/al_${DATASET}_bert_cal_${indicator}/${SEED}_${indicator}_cls"
    if [ -d "${exp_result_dir}" ]; then
        echo "Copying results from ${exp_result_dir}" | tee -a "${LOG_FILE}"
        cp -r "${exp_result_dir}"/* "${output_dir}/" 2>/dev/null || true
    else
        # Also check without _cls suffix
        exp_result_dir="experiments/al_${DATASET}_bert_cal_${indicator}/${SEED}_${indicator}"
        if [ -d "${exp_result_dir}" ]; then
            echo "Copying results from ${exp_result_dir}" | tee -a "${LOG_FILE}"
            cp -r "${exp_result_dir}"/* "${output_dir}/" 2>/dev/null || true
        else
            echo "Warning: Result directory not found: ${exp_result_dir}" | tee -a "${LOG_FILE}"
        fi
    fi
}

# =============================================================================
# Run experiments SEQUENTIALLY (not in parallel)
# =============================================================================
echo "" | tee -a "${LOG_FILE}"
echo "Running experiments sequentially..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# =====================================================
# Experiment 1: Baseline (Spark Exact + No Cache)
# - Exact brute-force KNN search
# - No probability caching (to match baseline)
# =====================================================
run_experiment "spark_exact_nocache" "exact" "False"

# =====================================================
# Experiment 2: Optimized (Spark LSH + Cache)
# - LSH approximate KNN search
# - Probability caching enabled
# =====================================================
run_experiment "spark_lsh_cache" "lsh" "True" "5" "2.0"

# =============================================================================
# Collect and summarize results
# =============================================================================
echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "All experiments completed at $(date)" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Extracting timing information from logs..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Extract timing info from logs
for config in "spark_exact_nocache" "spark_lsh_cache"; do
    exp_log="${BASE_OUTPUT_DIR}/${config}/seed_${SEED}/run.log"
    
    if [ -f "${exp_log}" ]; then
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        echo "Configuration: ${config}" | tee -a "${LOG_FILE}"
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        
        # Extract selection times
        echo "Selection times per iteration:" | tee -a "${LOG_FILE}"
        grep "Selection time:" "${exp_log}" | tee -a "${LOG_FILE}"
        
        # Extract KNN times
        echo "" | tee -a "${LOG_FILE}"
        echo "KNN times per iteration:" | tee -a "${LOG_FILE}"
        grep "KNN total time:" "${exp_log}" | tee -a "${LOG_FILE}"
        
        # Extract final accuracy
        echo "" | tee -a "${LOG_FILE}"
        echo "Final test accuracy:" | tee -a "${LOG_FILE}"
        grep "test_acc" "${exp_log}" | tail -1 | tee -a "${LOG_FILE}"
        
        echo "" | tee -a "${LOG_FILE}"
    else
        echo "Warning: Log file not found: ${exp_log}" | tee -a "${LOG_FILE}"
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "Summary" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "Dataset: SST-2 (~67k samples)" | tee -a "${LOG_FILE}"
echo "Seed: ${SEED}" | tee -a "${LOG_FILE}"
echo "Configurations tested (sequentially):" | tee -a "${LOG_FILE}"
echo "  1. Spark Exact + No Cache (baseline)" | tee -a "${LOG_FILE}"
echo "  2. Spark LSH + Cache (optimized)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Results directory: ${BASE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Experiment completed successfully!" | tee -a "${LOG_FILE}"
