#!/bin/bash

# Comparison script for Milvus: Exact KNN vs ANN search
# Both use pre-computed log(k_i), comparing search strategies:
# - Exact KNN: FLAT index (brute force, 100% accurate)
# - ANN: IVF index (approximate, faster but may sacrifice some accuracy)

export CUDA_VISIBLE_DEVICES=2

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
OUTPUT_DIR=./output_milvus

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

# Dataset to test
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
echo "Milvus: Exact KNN vs ANN Comparison"
echo "Dataset: ${DATASET}"
echo "LLM: ${LLM}"
echo "N_TRAIN_SHOT: ${N_TRAIN_SHOT}"
echo "N_DEMO_SHOT: ${N_DEMO_SHOT}"
echo "KNN: ${KNN}"
echo "=========================================="
echo ""
echo "Setup: Both versions pre-compute log(k_i)"
echo "Comparison: Exact (FLAT) vs ANN (IVF)"
echo ""
echo "Dimension strategy:"
echo "  - Both use 16384 dim projection (for fair comparison)"
echo "  - Comparison focuses on: Index type (FLAT vs IVF)"
echo "  - FLAT: Brute force exact search"
echo "  - IVF: Approximate nearest neighbor search"
echo ""
echo "NOTE: Using Milvus Lite (.db files, no Docker required)"
echo ""

# Run experiments for seed 1 only (quick test)
# To run all seeds 1-5, change to: for SEED in 1 2 3 4 5; do
for SEED in 1; do
    echo "==================== SEED ${SEED} ===================="
    
    # Run Milvus with Exact KNN (FLAT index)
    echo ""
    echo "[Milvus Exact KNN - FLAT, 16384 dim] Running with seed ${SEED}..."
    python3 knn_prompting.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        --knn ${KNN} \
        --use_milvus
    
    echo ""
    echo "[Milvus ANN - IVF, 16384 dim] Running with seed ${SEED}..."
    # Run Milvus with ANN (IVF index)
    python3 knn_prompting.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        --knn ${KNN} \
        --use_milvus \
        --milvus_use_ann
    
    echo ""
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Milvus Exact KNN: ${OUTPUT_DIR}/results_knnprompting_milvus_exact.csv"
echo "  Milvus ANN:       ${OUTPUT_DIR}/results_knnprompting_milvus_ann.csv"
echo ""
echo "Generating comparison report..."

# Generate comparison report using Python
python3 << 'EOF'
import csv
import os

output_dir = os.environ.get('OUTPUT_DIR', './output_milvus')
exact_file = f'{output_dir}/results_knnprompting_milvus_exact.csv'
ann_file = f'{output_dir}/results_knnprompting_milvus_ann.csv'

def read_results(filename):
    results = []
    if not os.path.exists(filename):
        return results
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

exact_results = read_results(exact_file)
ann_results = read_results(ann_file)

if exact_results and ann_results:
    print("\n" + "="*100)
    print("MILVUS: EXACT KNN vs ANN COMPARISON")
    print("="*100)
    print(f"{'Seed':<6} {'Method':<18} {'Acc':<8} {'Build(s)':<10} {'Infer(s)':<10} {'Total(s)':<10}")
    print("-"*100)
    
    for i in range(min(len(exact_results), len(ann_results))):
        e = exact_results[i]
        a = ann_results[i]
        
        print(f"{e['seed']:<6} {'Exact KNN (FLAT)':<18} {float(e['acc']):<8.4f} {e['build_time']:<10} {e['infer_time']:<10} {e['total_time']:<10}")
        print(f"{'':6} {'ANN (IVF)':<18} {float(a['acc']):<8.4f} {a['build_time']:<10} {a['infer_time']:<10} {a['total_time']:<10}")
        print()
    
    print("-"*100)
    
    # Calculate speedup and accuracy
    exact_infer = sum(float(r['infer_time']) for r in exact_results) / len(exact_results)
    ann_infer = sum(float(r['infer_time']) for r in ann_results) / len(ann_results)
    
    exact_total = sum(float(r['total_time']) for r in exact_results) / len(exact_results)
    ann_total = sum(float(r['total_time']) for r in ann_results) / len(ann_results)
    
    exact_acc = sum(float(r['acc']) for r in exact_results) / len(exact_results)
    ann_acc = sum(float(r['acc']) for r in ann_results) / len(ann_results)
    
    print("\nPerformance:")
    print(f"  Inference Time Speedup: {exact_infer / ann_infer:.2f}x (ANN faster)")
    print(f"  Total Time Speedup:     {exact_total / ann_total:.2f}x (ANN faster)")
    
    print(f"\nAccuracy:")
    print(f"  Exact KNN: {exact_acc:.4f}")
    print(f"  ANN:       {ann_acc:.4f}")
    print(f"  Difference: {abs(exact_acc - ann_acc):.6f} ({(ann_acc - exact_acc) * 100:+.2f}%)")
    
    print("\n" + "="*100)
    print("KEY INSIGHT:")
    print("="*100)
    print("Both methods pre-compute log(k_i) offline. The comparison shows:")
    print("1. Speed: How much faster is ANN (IVF) compared to exact search (FLAT)")
    print("2. Accuracy: Whether ANN sacrifices accuracy for speed")
    print("3. Trade-off: ANN is worthwhile if speedup > accuracy loss")
    print("="*100)
else:
    print("\nWarning: Could not find results files for comparison.")
    print(f"Expected files:\n  {exact_file}\n  {ann_file}")

EOF

echo ""
echo "Done!"
