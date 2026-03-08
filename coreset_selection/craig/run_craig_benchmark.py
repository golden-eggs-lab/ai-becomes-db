#!/usr/bin/env python
"""
CRAIG Benchmark Script - Multi-run experiments across backends

Runs experiments on 3 backends (memory, milvus, spark) with:
- 3 repetitions per configuration for mean ± std
- Bottleneck analysis for memory baseline (optimization-related component timing)
- Individual run results saved to JSON

Usage:
    # Quick timing test (1 epoch)
    python run_craig_benchmark.py --backends memory --runs 1 --epochs 1
    
    # Standard benchmark (15 epochs, 3 runs)
    python run_craig_benchmark.py --backends memory milvus spark --runs 3 --epochs 15
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import numpy as np
import time
import json
import argparse
from datetime import datetime
import util_nearpy as util

# Parse arguments
parser = argparse.ArgumentParser(description='CRAIG Multi-Backend Benchmark')
parser.add_argument('--backends', nargs='+', default=['memory'], 
                    choices=['memory', 'milvus', 'spark'],
                    help='Backends to benchmark')
parser.add_argument('--runs', type=int, default=3, help='Number of runs per config')
parser.add_argument('--epochs', type=int, default=15, help='Training epochs (standard=15, use 1 for timing only)')
parser.add_argument('--subset_size', type=float, default=0.4, help='Subset size')
parser.add_argument('--ann_k', type=int, default=20, help='ANN candidate count')
parser.add_argument('--spark_batch_size', type=int, default=50, help='Spark batch size')
args = parser.parse_args()

# Load MNIST data (NO normalization - matches original compare_baseline_vs_optimized.py)
print("Loading MNIST dataset...")
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
Y_train_nocat = Y_train.copy()
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/benchmark_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*70}")
print(f"CRAIG BENCHMARK")
print(f"{'='*70}")
print(f"Backends: {args.backends}")
print(f"Runs per config: {args.runs}")
print(f"Epochs: {args.epochs}")
print(f"Subset size: {args.subset_size}")
print(f"Output: {output_dir}")
print()


def get_backend_config(backend, mode):
    """Get configuration for each backend/mode combination
    
    Baseline: use_ann=False, use_reuse=False (no optimizations)
    Optimized: use_ann=True, use_reuse=True (both optimizations)
    """
    if backend == 'memory':
        return {
            'ann_backend': 'nearpy',
            'use_ann': mode == 'optimized',
            'use_reuse': mode == 'optimized',  # Key: baseline has NO reuse
            'force_backend': mode == 'optimized',
        }
    elif backend == 'milvus':
        return {
            'ann_backend': 'milvus',
            'use_ann': mode == 'optimized',
            'use_reuse': mode == 'optimized',  # Key: baseline has NO reuse
            'force_backend': True,  # Always use backend for fair comparison
        }
    elif backend == 'spark':
        return {
            'ann_backend': 'spark',
            'use_ann': mode == 'optimized',
            'use_reuse': mode == 'optimized',  # Key: baseline has NO reuse
            'force_backend': True,
        }


def run_single_experiment(backend, mode, run_id):
    """Run a single experiment and return results"""
    config = get_backend_config(backend, mode)
    
    print(f"\n{'='*70}")
    print(f"[{backend.upper()}] {mode.upper()} - Run {run_id}")
    print(f"Config: use_ann={config['use_ann']}, use_reuse={config['use_reuse']}, backend={config['ann_backend']}")
    print(f"{'='*70}")
    
    # Build model
    model = Sequential([
        Dense(100, input_dim=784, kernel_regularizer=l2(1e-4), activation='sigmoid'),
        Dense(10, kernel_regularizer=l2(1e-4), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
    
    # Tracking
    epoch_results = []
    total_ordering_time = 0
    total_similarity_time = 0
    total_weight_time = 0  # Algorithm line 8 - weight assignment
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"  Epoch {epoch+1}/{args.epochs}", end="", flush=True)
        
        # Get gradients (predictions - labels)
        preds = model.predict(X_train, verbose=0)
        preds = preds - Y_train
        
        # Greedy selection
        B = int(args.subset_size * len(X_train))
        greedy_start = time.time()
        
        subset, subset_weight, _, _, ordering_time, similarity_time, weight_time = util.get_orders_and_weights(
            B, preds, 'euclidean', smtk=0, no=0, y=Y_train_nocat,
            stoch_greedy=0, equal_num=True,
            use_ann=config['use_ann'],
            ann_k=args.ann_k,
            ann_backend=config['ann_backend'],
            force_backend=config['force_backend'],
            batch_size=args.spark_batch_size,
            use_reuse=config['use_reuse']  # Key: baseline has NO reuse
        )
        
        greedy_time = time.time() - greedy_start
        total_ordering_time += ordering_time
        total_similarity_time += similarity_time
        total_weight_time += weight_time
        
        # Train
        weights = np.zeros(len(X_train))
        subset_weight_normalized = subset_weight / np.sum(subset_weight) * len(subset_weight)
        weights[subset] = subset_weight_normalized
        model.fit(X_train, Y_train, sample_weight=weights, batch_size=32, epochs=1, verbose=0)
        
        # Evaluate
        _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        
        epoch_time = time.time() - epoch_start
        print(f" - greedy: {greedy_time:.2f}s, acc: {test_acc:.4f}")
        
        epoch_results.append({
            'epoch': epoch + 1,
            'greedy_time': greedy_time,
            'ordering_time': ordering_time,
            'similarity_time': similarity_time,
            'test_acc': test_acc,
            'epoch_time': epoch_time
        })
    
    total_time = time.time() - total_start
    
    # Compile results
    greedy_times = [e['greedy_time'] for e in epoch_results]
    test_accs = [e['test_acc'] for e in epoch_results]
    
    result = {
        'backend': backend,
        'mode': mode,
        'run_id': run_id,
        'config': {
            'epochs': args.epochs,
            'subset_size': args.subset_size,
            'ann_k': args.ann_k,
            **config
        },
        'metrics': {
            'total_time': total_time,
            'total_greedy_time': sum(greedy_times),
            'avg_greedy_time': np.mean(greedy_times),
            'std_greedy_time': np.std(greedy_times),
            'total_ordering_time': total_ordering_time,
            'total_similarity_time': total_similarity_time,
            'total_weight_time': total_weight_time,
            'final_acc': test_accs[-1],
            'best_acc': max(test_accs)
        },
        'epochs': epoch_results
    }
    
    # Bottleneck analysis for memory baseline
    if backend == 'memory' and mode == 'baseline':
        total_algorithm_time = result['metrics']['total_greedy_time']
        # Optimization targets (ANN + Reuse) apply to ordering stage only
        result['bottleneck_analysis'] = {
            'total_algorithm_time': total_algorithm_time,
            'ordering_time': total_ordering_time,
            'weight_time': total_weight_time,
            'similarity_time': total_similarity_time,
            'ordering_percentage': (total_ordering_time / total_algorithm_time * 100) if total_algorithm_time > 0 else 0,
            'weight_percentage': (total_weight_time / total_algorithm_time * 100) if total_algorithm_time > 0 else 0,
            'similarity_percentage': (total_similarity_time / total_algorithm_time * 100) if total_algorithm_time > 0 else 0,
            'optimization_target_percentage': (total_ordering_time / total_algorithm_time * 100) if total_algorithm_time > 0 else 0,
            'note': 'Optimization targets (ANN + Reuse) apply to ordering stage. ordering_time = optimization target.'
        }
        print(f"\n  📊 Bottleneck Analysis:")
        print(f"     Total Algorithm Time: {total_algorithm_time:.2f}s")
        print(f"     Ordering (ANN+Reuse target): {total_ordering_time:.2f}s ({result['bottleneck_analysis']['ordering_percentage']:.1f}%)")
        print(f"     Similarity (matrix precompute): {total_similarity_time:.2f}s ({result['bottleneck_analysis']['similarity_percentage']:.1f}%)")
    
    # Save individual run
    run_file = os.path.join(output_dir, f"{backend}_{mode}_run{run_id}.json")
    with open(run_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {run_file}")
    
    return result


def compute_summary(results):
    """Compute mean and std across runs"""
    summary = {}
    
    for key in ['total_time', 'total_greedy_time', 'avg_greedy_time', 'final_acc', 'best_acc']:
        values = [r['metrics'][key] for r in results]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return summary


# Run all experiments
all_results = {}
bottleneck_results = {}

for backend in args.backends:
    all_results[backend] = {'baseline': [], 'optimized': []}
    
    for mode in ['baseline', 'optimized']:
        for run_id in range(1, args.runs + 1):
            result = run_single_experiment(backend, mode, run_id)
            all_results[backend][mode].append(result)
            
            # Collect bottleneck data for memory baseline
            if backend == 'memory' and mode == 'baseline' and 'bottleneck_analysis' in result:
                bottleneck_results[f'run{run_id}'] = result['bottleneck_analysis']

# Compute summaries
summary = {}
for backend in args.backends:
    summary[backend] = {}
    for mode in ['baseline', 'optimized']:
        if all_results[backend][mode]:
            summary[backend][mode] = compute_summary(all_results[backend][mode])
    
    # Compute speedup
    if summary[backend].get('baseline') and summary[backend].get('optimized'):
        baseline_time = summary[backend]['baseline']['avg_greedy_time']['mean']
        optimized_time = summary[backend]['optimized']['avg_greedy_time']['mean']
        summary[backend]['speedup'] = {
            'greedy_speedup': baseline_time / optimized_time if optimized_time > 0 else 0
        }

# Bottleneck summary for memory baseline
if bottleneck_results:
    ordering_pcts = [b['ordering_percentage'] for b in bottleneck_results.values()]
    bottleneck_summary = {
        'runs': bottleneck_results,
        'summary': {
            'ordering_percentage_mean': np.mean(ordering_pcts),
            'ordering_percentage_std': np.std(ordering_pcts),
            'optimization_target_percentage_mean': np.mean(ordering_pcts),
            'optimization_target_percentage_std': np.std(ordering_pcts),
            'note': 'Optimization targets (ANN + Reuse) apply to ordering stage only'
        }
    }
    
    bottleneck_file = os.path.join(output_dir, "bottleneck_analysis.json")
    with open(bottleneck_file, 'w') as f:
        json.dump(bottleneck_summary, f, indent=2)
    print(f"\n📊 Bottleneck analysis saved: {bottleneck_file}")

# Save summary
summary_file = os.path.join(output_dir, "summary.json")
with open(summary_file, 'w') as f:
    json.dump({
        'config': {
            'backends': args.backends,
            'runs': args.runs,
            'epochs': args.epochs,
            'subset_size': args.subset_size,
            'ann_k': args.ann_k
        },
        'summary': summary
    }, f, indent=2)

# Print final summary
print(f"\n{'='*70}")
print("BENCHMARK SUMMARY")
print(f"{'='*70}")

for backend in args.backends:
    print(f"\n📦 {backend.upper()}:")
    if 'baseline' in summary[backend]:
        s = summary[backend]['baseline']
        print(f"   Baseline  : {s['avg_greedy_time']['mean']:.3f} ± {s['avg_greedy_time']['std']:.3f}s "
              f"(acc: {s['final_acc']['mean']:.4f})")
    if 'optimized' in summary[backend]:
        s = summary[backend]['optimized']
        print(f"   Optimized : {s['avg_greedy_time']['mean']:.3f} ± {s['avg_greedy_time']['std']:.3f}s "
              f"(acc: {s['final_acc']['mean']:.4f})")
    if 'speedup' in summary[backend]:
        print(f"   Speedup   : {summary[backend]['speedup']['greedy_speedup']:.2f}x")

# Print bottleneck summary for memory baseline
if bottleneck_results:
    print(f"\n📊 MEMORY BASELINE BOTTLENECK ANALYSIS:")
    bs = bottleneck_summary['summary']
    print(f"   Ordering (ANN+Reuse target): {bs['ordering_percentage_mean']:.1f} ± {bs['ordering_percentage_std']:.1f}%")

print(f"\n📁 Results saved to: {output_dir}")
print("✓ BENCHMARK COMPLETED!")
