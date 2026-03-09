#!/bin/bash
# E2E Benchmark Pipeline for DEFT-UCS (CoEDIT and WikiLarge)
# Decouples Selection, Fine-tuning, and Evaluation

echo "=========================================================="
echo "          DEFT-UCS E2E Benchmarking Pipeline              "
echo "=========================================================="

DATASETS=("coedit" "wikilarge")
SETTINGS=("baseline" "optimized")
RUNS=3

# Output directory for decoupled artifacts
ARTIFACTS_DIR="artifacts/benchmark_e2e"
mkdir -p "$ARTIFACTS_DIR"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================================="
    echo "          Dataset: ${DATASET^^}"
    echo "=========================================================="

    # Prepare Data if necessary
    if [ "$DATASET" == "coedit" ]; then
        if [ ! -f "cache/coedit_embeddings.npy" ]; then
            echo "Running coedit data prep..."
            python prepare_data.py
        fi
    elif [ "$DATASET" == "wikilarge" ]; then
        if [ ! -f "cache/wikilarge_embeddings.npy" ]; then
            echo "Running wikilarge data prep..."
            python prepare_wikilarge.py
        fi
    fi

    for SETTING in "${SETTINGS[@]}"; do
        echo ""
        echo "----------------------------------------------------------"
        echo "          Setting: ${SETTING^^}"
        echo "----------------------------------------------------------"

        for ((RUN=1; RUN<=RUNS; RUN++)); do
            echo "[${DATASET^^} - ${SETTING^^}] Run ${RUN}/${RUNS}"
            
            # Step 1: Selection
            SAMPLES_PATH="${ARTIFACTS_DIR}/${DATASET}_samples_${SETTING}_run${RUN}.json"
            TIMING_PATH="${ARTIFACTS_DIR}/${DATASET}_timing_${SETTING}_run${RUN}.json"
            
            if [ ! -f "$SAMPLES_PATH" ]; then
                echo "1. Running Coreset Selection..."
                python benchmark_selection.py \
                    --dataset "$DATASET" \
                    --setting "$SETTING" \
                    --output "$SAMPLES_PATH" \
                    --timing-output "$TIMING_PATH" \
                    --seed 42  # Seed stays the same assuming fixed subsets are comparable
            else
                echo "1. Coreset Selection already completed."
            fi

            # Step 2: Fine-Tuning
            MODEL_DIR="${ARTIFACTS_DIR}/${DATASET}_model_${SETTING}_run${RUN}"
            
            if [ ! -d "${MODEL_DIR}/final" ]; then
                echo "2. Running Fine-tuning..."
                python benchmark_finetune.py \
                    --samples "$SAMPLES_PATH" \
                    --output "$MODEL_DIR"
            else
                echo "2. Fine-tuning already completed."
            fi

            # Step 3: Evaluation
            EVAL_RESULT_PATH="${MODEL_DIR}/eval_results.json"
            
            if [ ! -f "$EVAL_RESULT_PATH" ]; then
                echo "3. Running Evaluation..."
                python benchmark_evaluate.py --model-dir "$MODEL_DIR"
            else
                echo "3. Evaluation already completed."
            fi
        done
        
        echo "----------------------------------------------------------"
        echo " ${DATASET^^} - ${SETTING^^} : $RUNS RUNS COMPLETE."
        echo "----------------------------------------------------------"
    done
done

echo ""
echo "=========================================================="
echo "                   PIPELINE FINISHED                      "
echo "=========================================================="
