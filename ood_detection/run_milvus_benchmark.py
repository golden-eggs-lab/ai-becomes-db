#!/usr/bin/env python
"""
Milvus KNN-OOD Benchmark Script
Run experiments to get mean and std of search times.
"""

import subprocess
import re
import numpy as np
import sys

NUM_RUNS = 1  # Run once, combine with previous results

def run_experiment(cmd, name):
    """Run a single experiment and extract timing info."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)
    
    # Extract timing info
    search_time = None
    total_time = None
    setup_time = None
    
    for line in output.split('\n'):
        if 'Search time:' in line:
            match = re.search(r'Search time:\s*([\d.]+)s', line)
            if match:
                search_time = float(match.group(1))
        elif 'Total time:' in line:
            match = re.search(r'Total time:\s*([\d.]+)s', line)
            if match:
                total_time = float(match.group(1))
        elif 'Milvus setup time:' in line:
            match = re.search(r'Milvus setup time:\s*([\d.]+)s', line)
            if match:
                setup_time = float(match.group(1))
    
    return {
        'search_time': search_time,
        'total_time': total_time,
        'setup_time': setup_time,
        'output': output
    }

def main():
    # Base command for Milvus
    base_cmd = [
        'python', 'run_imagenet.py',
        '--name', 'resnet50-supcon',
        '--in-dataset', 'imagenet',
        '--out-datasets', 'inat', 'sun50', 'places50', 'dtd',
        '--use-milvus', '--milvus-lite',
        '--seed', '1'
    ]
    
    # Baseline: Milvus FLAT (exact search)
    baseline_cmd = base_cmd.copy() + [
        '--milvus-index-type', 'FLAT',
        '--milvus-metric', 'L2'
    ]
    
    # Optimized: Milvus IVF_FLAT (ANN)
    optimized_cmd = base_cmd.copy() + [
        '--milvus-index-type', 'IVF_FLAT',
        '--milvus-metric', 'L2',
        '--nlist', '1000',
        '--nprobe', '5'
    ]
    
    # Results storage
    baseline_results = {'search_time': [], 'total_time': [], 'setup_time': []}
    optimized_results = {'search_time': [], 'total_time': [], 'setup_time': []}
    
    print(f"\n{'#'*60}")
    print(f"# Milvus KNN-OOD Benchmark: {NUM_RUNS} runs per method")
    print(f"{'#'*60}")
    
    # Run experiments
    for run_idx in range(NUM_RUNS):
        print(f"\n\n{'*'*60}")
        print(f"* RUN {run_idx + 1}/{NUM_RUNS}")
        print(f"{'*'*60}")
        
        # Baseline
        result = run_experiment(baseline_cmd, f"Baseline (Milvus-FLAT) - Run {run_idx + 1}")
        if result['search_time'] is not None:
            baseline_results['search_time'].append(result['search_time'])
        if result['total_time'] is not None:
            baseline_results['total_time'].append(result['total_time'])
        if result['setup_time'] is not None:
            baseline_results['setup_time'].append(result['setup_time'])
        
        # Optimized
        result = run_experiment(optimized_cmd, f"Optimized (Milvus-IVF_FLAT) - Run {run_idx + 1}")
        if result['search_time'] is not None:
            optimized_results['search_time'].append(result['search_time'])
        if result['total_time'] is not None:
            optimized_results['total_time'].append(result['total_time'])
        if result['setup_time'] is not None:
            optimized_results['setup_time'].append(result['setup_time'])
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY: Mean ± Std")
    print(f"{'='*60}")
    
    def print_stats(name, results):
        print(f"\n{name}:")
        for metric, values in results.items():
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                print(f"  {metric}: {mean:.3f} ± {std:.3f}s (n={len(values)})")
    
    print_stats("Baseline (Milvus-FLAT)", baseline_results)
    print_stats("Optimized (Milvus-IVF_FLAT)", optimized_results)
    
    # Calculate speedup
    if baseline_results['search_time'] and optimized_results['search_time']:
        baseline_mean = np.mean(baseline_results['search_time'])
        optimized_mean = np.mean(optimized_results['search_time'])
        speedup = baseline_mean / optimized_mean
        print(f"\n{'='*60}")
        print(f"SPEEDUP (Search Time): {speedup:.2f}x")
        print(f"  Baseline: {baseline_mean:.3f}s")
        print(f"  Optimized: {optimized_mean:.3f}s")
        print(f"{'='*60}")
    
    # Previous results for reference
    print(f"\n\n{'='*60}")
    print("PREVIOUS RESULTS (from screenshot):")
    print("  Milvus-FLAT: 5546.779s")
    print("  Milvus-IVF_FLAT: 4560.137s")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
