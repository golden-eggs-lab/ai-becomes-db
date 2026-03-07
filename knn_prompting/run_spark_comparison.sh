#!/bin/bash

# Spark Comparison Script for KNN Prompting
# Compares Spark BASELINE vs Spark OPTIMIZED (internal comparison)

echo "========================================="
echo "Spark KNN Prompting Comparison"
echo "Baseline vs Optimized (Spark internal)"
echo "========================================="
echo ""

# Configuration
LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
DATASET="sst2"
N_TRAIN_SHOT=1024
N_DEMO_SHOT=16
KNN=3
SEED=1
SPARK_MASTER="local[4]"  # 4 cores, adjust based on your machine

# Create output directory
OUTPUT_DIR="output_spark"
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  LLM: $LLM"
echo "  LLM_DIR: $LLM_DIR"
echo "  Dataset: $DATASET"
echo "  Train shots: $N_TRAIN_SHOT"
echo "  Demo shots: $N_DEMO_SHOT"
echo "  KNN: $KNN"
echo "  Seed: $SEED"
echo "  Spark master: $SPARK_MASTER"
echo ""

# 1. Spark BASELINE (no optimizations)
echo "========================================="
echo "1. Spark BASELINE (no optimizations)"
echo "   ❌ Computes log(k_i) every inference"
echo "   ❌ Python loop over queries"
echo "========================================="
python knn_prompting.py \
    --llm_dir $LLM_DIR \
    --data_dir $DATA_DIR \
    --dataset $DATASET \
    --n_train_shot $N_TRAIN_SHOT \
    --n_demo_shot $N_DEMO_SHOT \
    --knn $KNN \
    --seed $SEED \
    --use_spark \
    --spark_master $SPARK_MASTER \
    --output_dir $OUTPUT_DIR \
    --csv_name results_knnprompting_spark_baseline.csv

# 2. Spark OPTIMIZED (with two optimizations)
echo ""
echo "========================================="
echo "2. Spark OPTIMIZED (with optimizations)"
echo "   ✅ Pre-computed log(k_i)"
echo "   ✅ Batch processing + Window functions"
echo "========================================="
python knn_prompting.py \
    --llm_dir $LLM_DIR \
    --data_dir $DATA_DIR \
    --dataset $DATASET \
    --n_train_shot $N_TRAIN_SHOT \
    --n_demo_shot $N_DEMO_SHOT \
    --knn $KNN \
    --seed $SEED \
    --use_spark \
    --spark_optimized \
    --spark_master $SPARK_MASTER \
    --output_dir $OUTPUT_DIR \
    --csv_name results_knnprompting_spark_optimized.csv

echo ""
echo "========================================="
echo "Comparison complete!"
echo "========================================="
echo ""
echo "Results saved in $OUTPUT_DIR/:"
ls -lh $OUTPUT_DIR/results_knnprompting_spark_*.csv

echo ""
echo "Generating summary report..."

# Generate summary using Python
python3 << 'EOF'
import pandas as pd
import os

output_dir = "output_spark"

# Read results
results = []
for filename in [
    "results_knnprompting_spark_baseline.csv",
    "results_knnprompting_spark_optimized.csv"
]:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if not df.empty:
            results.append(df.iloc[-1].to_dict())

if results and len(results) == 2:
    df_summary = pd.DataFrame(results)
    
    # Extract relevant columns
    cols_to_show = ['method', 'setup', 'acc', 'build_time', 'infer_time', 'total_time']
    cols_to_show = [c for c in cols_to_show if c in df_summary.columns]
    df_summary = df_summary[cols_to_show]
    
    print("\n" + "="*80)
    print("SPARK PERFORMANCE COMPARISON (Baseline vs Optimized)")
    print("="*80)
    print(df_summary.to_string(index=False))
    print()
    
    # Calculate speedup
    baseline_time = results[0]['infer_time']
    opt_time = results[1]['infer_time']
    speedup = baseline_time / opt_time
    
    baseline_build = results[0]['build_time']
    opt_build = results[1]['build_time']
    
    baseline_acc = results[0]['acc']
    opt_acc = results[1]['acc']
    
    print("="*80)
    print("OPTIMIZATION IMPACT")
    print("="*80)
    print(f"Build Time:")
    print(f"  Baseline: {baseline_build:.2f}s (no log pre-computation)")
    print(f"  Optimized: {opt_build:.2f}s (with log pre-computation)")
    print(f"  Build overhead: {opt_build - baseline_build:.2f}s")
    print()
    print(f"Inference Time:")
    print(f"  Baseline: {baseline_time:.2f}s (Python loop + runtime log)")
    print(f"  Optimized: {opt_time:.2f}s (batch + pre-computed log)")
    print(f"  Speedup: {speedup:.2f}x faster")
    print()
    print(f"Accuracy:")
    print(f"  Baseline: {baseline_acc:.2f}%")
    print(f"  Optimized: {opt_acc:.2f}%")
    print(f"  Difference: {opt_acc - baseline_acc:.2f}%")
    print()
    
    print("Two key optimizations:")
    print("  ✅ Opt 1: Pre-compute log(k_i) - saved in build phase")
    print("  ✅ Opt 2: Batch processing - eliminates Python loop")
    print()
    
    if speedup > 1.5:
        print(f"✅ Optimizations effective: {speedup:.2f}x speedup!")
    elif speedup > 1.1:
        print(f"⚠️  Moderate improvement: {speedup:.2f}x speedup")
    else:
        print(f"⚠️  Limited improvement: {speedup:.2f}x speedup")
        print("   (Spark overhead may dominate for small datasets)")
else:
    print("Error: Could not find both result files!")
    print("Expected files:")
    print("  - results_knnprompting_spark_baseline.csv")
    print("  - results_knnprompting_spark_optimized.csv")

EOF

echo ""
echo "Done!"
