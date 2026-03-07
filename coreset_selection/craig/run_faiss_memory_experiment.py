#!/usr/bin/env python
"""
FAISS Memory Constraint Experiment

Compare FAISS-based baseline (exact) vs optimized (ANN/IVF) under memory pressure.
This demonstrates that FAISS has lower memory overhead than NearPy (hash-based ANN).

Memory limiting should be done externally via Docker: --memory=200m

Usage:
  python run_faiss_memory_experiment.py --setup faiss_baseline --memory-tag 200m
  python run_faiss_memory_experiment.py --setup faiss_optimized --memory-tag 200m
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import json
import argparse
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Training config
EPOCHS = 1  # Only 1 epoch needed to measure greedy selection time
SUBSET_SIZE = 0.4
ANN_K = 20

# FAISS setups
# baseline: exact search (FLAT index, no approximation)
# optimized: ANN search (IVF index with approximate search)
SETUPS = {
    'faiss_baseline': {'use_ann': False},    # Exact search via FAISS FLAT
    'faiss_optimized': {'use_ann': True},    # ANN via FAISS IVF
}


def run_experiment(setup_name: str, setup_config: dict) -> dict:
    """Run a single experiment configuration."""
    use_ann = setup_config['use_ann']
    
    print(f"\n{'='*60}")
    print(f"Running: {setup_name}")
    print(f"  use_ann={use_ann} (FAISS backend)")
    print(f"{'='*60}")
    
    # Import util_v2 which uses lazy_greedy_v2 (FAISS-based)
    import util_v2 as util
    
    # Prepare data
    (X_train_full, Y_train_full), (X_test_full, Y_test_full) = mnist.load_data()
    
    # **MEMORY EXPERIMENT OPTIMIZATION**: Only use class 0 to speed up tests
    # This reduces test time from ~7 hours to ~40 min per experiment while
    # still showing memory pressure trends across different memory limits
    print("Filtering to class 0 only for faster testing...")
    class_0_train_mask = (Y_train_full == 0)
    class_0_test_mask = (Y_test_full == 0)
    
    X_train = X_train_full[class_0_train_mask]
    Y_train_single = Y_train_full[class_0_train_mask]
    X_test = X_test_full[class_0_test_mask]
    Y_test_single = Y_test_full[class_0_test_mask]
    
    print(f"  Original: {len(X_train_full)} train, {len(X_test_full)} test")
    print(f"  Class 0:  {len(X_train)} train, {len(X_test)} test")
    
    X_train = X_train.reshape(-1, 784).astype('float32') / 255
    X_test = X_test.reshape(-1, 784).astype('float32') / 255
    
    # Binary classification for class 0 vs rest (but we only have class 0)
    Y_train = to_categorical(Y_train_single, 2)
    Y_test = to_categorical(Y_test_single, 2)
    Y_train_nocat = Y_train_single
    
    # Build binary classifier (class 0 only)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,), kernel_regularizer=l2(0.0001)),
        Activation('relu'),
        Dense(2, activation='softmax')  # Binary: class 0 vs rest (though we only have 0)
    ])
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
    
    greedy_times = []
    test_accs = []
    total_start = time.time()
    
    for epoch in range(EPOCHS):
        preds = model.predict(X_train, verbose=0)
        preds = preds - Y_train
        
        B = int(SUBSET_SIZE * len(X_train))
        greedy_start = time.time()
        
        # Force FAISS backend
        subset, subset_weight, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
            B, preds, 'euclidean', smtk=0, no=0, y=Y_train_nocat,
            stoch_greedy=0, equal_num=True,
            use_ann=use_ann, ann_k=ANN_K, ann_backend='faiss',
            force_backend=True
        )
        
        greedy_time = time.time() - greedy_start
        greedy_times.append(greedy_time)
        
        weights = np.zeros(len(X_train))
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        weights[subset] = subset_weight
        
        model.fit(X_train, Y_train, sample_weight=weights, batch_size=32, epochs=1, verbose=0)
        _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        test_accs.append(test_acc)
        
        print(f"  Epoch {epoch+1}/{EPOCHS}: greedy={greedy_time:.2f}s, acc={test_acc:.4f}")
    
    total_time = time.time() - total_start
    
    return {
        'setup': setup_name,
        'avg_greedy_time': float(np.mean(greedy_times)),
        'total_time': total_time,
        'greedy_times': greedy_times,
        'test_accs': test_accs,
        'final_acc': test_accs[-1] if test_accs else 0
    }


def main():
    parser = argparse.ArgumentParser(description='FAISS Memory Experiment')
    parser.add_argument('--setup', type=str, default=None, 
                        choices=list(SETUPS.keys()),
                        help='Run specific setup only')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--memory-tag', type=str, default='unlimited',
                        help='Tag for memory constraint (for record keeping)')
    parser.add_argument('--no-ann', action='store_true',
                        help='Force exact search (override setup config)')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results/memory_exp_faiss_{args.memory_tag}_{timestamp}.json"
    
    print("="*70)
    print("FAISS Memory Constraint Experiment")
    print("="*70)
    print(f"Memory tag: {args.memory_tag}")
    print(f"Output: {output_file}")
    print()
    
    # Select setups to run
    if args.setup:
        setups_to_run = {args.setup: SETUPS[args.setup].copy()}
        # Override with --no-ann if specified
        if args.no_ann:
            setups_to_run[args.setup]['use_ann'] = False
    else:
        setups_to_run = SETUPS
    
    results = []
    for setup_name, setup_config in setups_to_run.items():
        try:
            result = run_experiment(setup_name, setup_config)
            result['memory_tag'] = args.memory_tag
            result['error'] = None
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'setup': setup_name,
                'memory_tag': args.memory_tag,
                'error': str(e),
                'avg_greedy_time': float('nan'),
                'total_time': float('nan')
            })
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else 'results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        status = "✓" if not r.get('error') else "❌"
        time_str = f"{r['avg_greedy_time']:.2f}s" if not r.get('error') else r['error'][:30]
        print(f"  {status} {r['setup']:<24} | {time_str}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
