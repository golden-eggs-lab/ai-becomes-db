#!/bin/bash
# KNN-OOD ImageNet-1K: Dataset Size Scaling Experiment
# Vary training data subsample ratio: 0.7, 0.5, 0.3
# Baseline (Exact KNN GPU) × 1 run, Optimized (ANN-IVF GPU) × 2 runs per ratio
# Ratio=1.0 available from existing knn_ood_benchmark_3runs.log
# Env: ood

set -e

eval "$(conda shell.bash hook)"
conda activate ood

export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR=./results_scaling
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p ${LOG_DIR}

RATIOS="0.7 0.5 0.3"

BASE_ARGS=(
    --name resnet50-supcon
    --in-dataset imagenet
    --out-datasets inat sun50 places50 dtd
    --use-faiss-gpu
    --seed 1
)

echo "========================================="
echo "KNN-OOD: Dataset Size Scaling"
echo "Date: $(date)"
echo "Ratios: ${RATIOS}"
echo "========================================="

for ratio in ${RATIOS}; do
    echo ""
    echo "===== ratio=${ratio} ====="

    # Baseline: Exact KNN GPU × 1
    echo "  [Baseline] Running..."
    python -u run_imagenet.py \
        "${BASE_ARGS[@]}" \
        --subsample-ratio ${ratio} \
        > "${LOG_DIR}/baseline_ratio${ratio}.log" 2>&1
    echo "  [Baseline] Done."

    # Optimized: ANN-IVF GPU × 2
    for run in 1 2; do
        echo "  [Optimized run${run}] Running..."
        python -u run_imagenet.py \
            "${BASE_ARGS[@]}" \
            --use-ann --ann-method ivf \
            --nlist 1000 --nprobe 5 \
            --subsample-ratio ${ratio} \
            > "${LOG_DIR}/optimized_ratio${ratio}_run${run}.log" 2>&1
        echo "  [Optimized run${run}] Done."
    done
done

echo ""
echo "========================================="
echo "All runs completed! $(date)"
echo "========================================="

# Summary extraction
python3 << 'PYEOF'
import re, json, os
import numpy as np
from pathlib import Path

log_dir = Path("./results_scaling/logs")
results = {}

for ratio in [0.7, 0.5, 0.3]:
    d = {'ratio': ratio}
    
    # Baseline
    f = log_dir / f"baseline_ratio{ratio}.log"
    if f.exists():
        text = f.read_text()
        m = re.search(r'Total time:\s*([\d.]+)s', text)
        if m: d['baseline_time'] = float(m.group(1))
        m = re.search(r'AUROC.*?:\s*([\d.]+)', text)
        if m: d['baseline_auroc'] = float(m.group(1))
    
    # Optimized
    opt_times, opt_aurocs = [], []
    for run in [1, 2]:
        f = log_dir / f"optimized_ratio{ratio}_run{run}.log"
        if f.exists():
            text = f.read_text()
            m = re.search(r'Total time:\s*([\d.]+)s', text)
            if m: opt_times.append(float(m.group(1)))
            m = re.search(r'AUROC.*?:\s*([\d.]+)', text)
            if m: opt_aurocs.append(float(m.group(1)))
    
    d['optimized_times'] = opt_times
    if opt_aurocs: d['optimized_auroc'] = np.mean(opt_aurocs)
    
    results[str(ratio)] = d

with open("./results_scaling/scaling_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\nKNN-OOD Scaling Results:")
for r, d in sorted(results.items(), key=lambda x: -float(x[0])):
    if 'baseline_time' in d and d.get('optimized_times'):
        opt_mean = np.mean(d['optimized_times'])
        print(f"  ratio={r}: baseline={d['baseline_time']:.1f}s, opt={opt_mean:.1f}s, speedup={d['baseline_time']/opt_mean:.2f}x")
PYEOF
