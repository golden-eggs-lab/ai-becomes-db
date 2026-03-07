#!/bin/bash

# Comparison script for Baseline KL vs ANN-optimized KNN Prompting
# This script runs both methods with the same seeds and compares:
# 1. Accuracy (should be identical or very similar)
# 2. Build time (anchor store construction)
# 3. Inference time
# 4. Total time

export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
OUTPUT_DIR=./output

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Set max demonstration shot w.r.t. context length
if [[ "${LLM}" == "gpt2-xl" ]] || [[ "${LLM}" == "gpt2-large" ]]; then
# max context length = 1024
array1=(mpqa) # maxshot = 32
array2=(sst2) # maxshot = 16
array3=(subj cr mr trec) # maxshot = 8
array4=(rte) # maxshot = 4
array5=(agnews cb) # maxshot = 2
array6=(dbpedia) # maxshot = 1
else
# max context length = 2048
array1=(sst2 mpqa)
array2=(subj cr mr trec)
array3=(rte)
array4=(agnews cb)
array5=(none)
array6=(dbpedia)
fi

# You can change this to test different datasets
# Options: sst2 subj mpqa agnews cb cr dbpedia mr rte trec
DATASET=sst2

if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=32
elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=16
elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=8
elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=4
elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=2
else
N_DEMO_SHOT=1
fi

N_TRAIN_SHOT=1024
KNN=3

echo "=========================================="
echo "KNN Prompting Comparison Experiment"
echo "Dataset: ${DATASET}"
echo "LLM: ${LLM}"
echo "N_TRAIN_SHOT: ${N_TRAIN_SHOT}"
echo "N_DEMO_SHOT: ${N_DEMO_SHOT}"
echo "KNN: ${KNN}"
echo "=========================================="
echo ""

# Run experiments for seed 1 only (quick test)
# To run all seeds 1-5, change to: for SEED in 1 2 3 4 5; do
for SEED in 1; do
    echo "==================== SEED ${SEED} ===================="
    
    # Run Baseline (original KL method)
    echo ""
    echo "[Baseline] Running with seed ${SEED}..."
    python3 knn_prompting.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        --knn ${KNN} \
        --optimization baseline
    
    echo ""
    echo "[Opt-v1] Running with seed ${SEED}..."
    python3 knn_prompting.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        --knn ${KNN} \
        --optimization opt_v1
    
    echo ""
    echo "[Opt-v2] Running with seed ${SEED}..."
    python3 knn_prompting.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        --knn ${KNN} \
        --optimization opt_v2
    
    echo ""
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Baseline: ${OUTPUT_DIR}/results_knnprompting_baseline_seperated.csv"
echo "  Opt-v1:   ${OUTPUT_DIR}/results_knnprompting_opt_v1.csv"
echo "  Opt-v2:   ${OUTPUT_DIR}/results_knnprompting_opt_v2.csv"
echo ""
echo "Generating comparison report..."

# Generate comparison report using Python
python3 << 'EOF'
import csv
import os

output_dir = os.environ.get('OUTPUT_DIR', './output')
baseline_file = f'{output_dir}/results_knnprompting_baseline_seperated.csv'
opt_v1_file = f'{output_dir}/results_knnprompting_opt_v1.csv'
opt_v2_file = f'{output_dir}/results_knnprompting_opt_v2.csv'

def read_results(filename):
    results = []
    if not os.path.exists(filename):
        return results
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

baseline_results = read_results(baseline_file)
opt_v1_results = read_results(opt_v1_file)
opt_v2_results = read_results(opt_v2_file)

if baseline_results and opt_v1_results and opt_v2_results:
    print("\n" + "="*100)
    print("OPTIMIZATION COMPARISON REPORT")
    print("="*100)
    print(f"{'Seed':<6} {'Method':<12} {'Acc':<8} {'Build(s)':<10} {'Infer(s)':<10} {'Total(s)':<10}")
    print("-"*100)
    
    for i in range(min(len(baseline_results), len(opt_v1_results), len(opt_v2_results))):
        b = baseline_results[i]
        v1 = opt_v1_results[i]
        v2 = opt_v2_results[i]
        
        print(f"{b['seed']:<6} {'Baseline':<12} {float(b['acc']):<8.4f} {b['build_time']:<10} {b['infer_time']:<10} {b['total_time']:<10}")
        print(f"{'':6} {'Opt-v1':<12} {float(v1['acc']):<8.4f} {v1['build_time']:<10} {v1['infer_time']:<10} {v1['total_time']:<10}")
        print(f"{'':6} {'Opt-v2':<12} {float(v2['acc']):<8.4f} {v2['build_time']:<10} {v2['infer_time']:<10} {v2['total_time']:<10}")
        print()
    
    print("-"*100)
    
    # Calculate speedup
    baseline_infer = sum(float(r['infer_time']) for r in baseline_results) / len(baseline_results)
    opt_v1_infer = sum(float(r['infer_time']) for r in opt_v1_results) / len(opt_v1_results)
    opt_v2_infer = sum(float(r['infer_time']) for r in opt_v2_results) / len(opt_v2_results)
    
    baseline_total = sum(float(r['total_time']) for r in baseline_results) / len(baseline_results)
    opt_v1_total = sum(float(r['total_time']) for r in opt_v1_results) / len(opt_v1_results)
    opt_v2_total = sum(float(r['total_time']) for r in opt_v2_results) / len(opt_v2_results)
    
    print("\nInference Time Speedup:")
    print(f"  Opt-v1 vs Baseline: {baseline_infer / opt_v1_infer:.2f}x")
    print(f"  Opt-v2 vs Baseline: {baseline_infer / opt_v2_infer:.2f}x")
    print(f"  Opt-v2 vs Opt-v1:   {opt_v1_infer / opt_v2_infer:.2f}x")
    
    print("\nTotal Time Speedup:")
    print(f"  Opt-v1 vs Baseline: {baseline_total / opt_v1_total:.2f}x")
    print(f"  Opt-v2 vs Baseline: {baseline_total / opt_v2_total:.2f}x")
    print(f"  Opt-v2 vs Opt-v1:   {opt_v1_total / opt_v2_total:.2f}x")
    
    # Calculate average accuracy
    baseline_acc = sum(float(r['acc']) for r in baseline_results) / len(baseline_results)
    opt_v1_acc = sum(float(r['acc']) for r in opt_v1_results) / len(opt_v1_results)
    opt_v2_acc = sum(float(r['acc']) for r in opt_v2_results) / len(opt_v2_results)
    
    print(f"\nAverage Accuracy:")
    print(f"  Baseline: {baseline_acc:.4f}")
    print(f"  Opt-v1:   {opt_v1_acc:.4f} (diff: {abs(baseline_acc - opt_v1_acc):.6f})")
    print(f"  Opt-v2:   {opt_v2_acc:.4f} (diff: {abs(baseline_acc - opt_v2_acc):.6f})")
    print("="*100)
else:
    print("\nWarning: Could not find all results files for comparison.")
    print(f"Expected files:\n  {baseline_file}\n  {opt_v1_file}\n  {opt_v2_file}")

EOF

echo ""
echo "Done!"
