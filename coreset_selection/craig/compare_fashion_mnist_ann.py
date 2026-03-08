#!/usr/bin/env python
"""
Fashion-MNIST comparison: Exact vs ANN CRAIG
Identical to compare_mnist_ann.py except dataset is Fashion-MNIST.
All settings (model, epochs, batch_size, subset_size) unchanged.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# TensorFlow 2.x compatible imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

import numpy as np
import time
import util_nearpy as util
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare Fashion-MNIST CRAIG: Exact vs ANN')
parser.add_argument('--backend', type=str, default='nearpy', choices=['nearpy', 'milvus', 'spark'],
                    help='ANN backend: nearpy, milvus, or spark (default: nearpy)')
parser.add_argument('--ann_k', type=int, default=20,
                    help='Number of ANN candidates (default: 20)')
parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs (default: 15)')
parser.add_argument('--subset_size', type=float, default=0.4,
                    help='Subset size ratio (default: 0.4)')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size for Spark backend (default: 20)')
args = parser.parse_args()

print("="*70)
print(f"Fashion-MNIST CRAIG: Exact vs ANN Comparison (Backend: {args.backend})")
print("="*70)

# Load data — ONLY CHANGE from MNIST version
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(60000, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(10000, 784).astype(np.float32) / 255.0

num_classes = 10
Y_train_nocat = Y_train
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# Configuration — IDENTICAL to MNIST
batch_size = 32
subset_size = args.subset_size
epochs = args.epochs
reg = 1e-4
runs = 1  # Use 1 run for quick comparison
smtk = 0
ann_backend = args.backend
ann_k = args.ann_k
spark_batch_size = args.batch_size  # For Spark BATCH strategy

print(f"\nConfiguration:")
print(f"  Backend: {ann_backend}")
print(f"  Subset size: {subset_size} ({int(60000*subset_size)} samples)")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  ANN k: {ann_k}")
print(f"  Spark batch size: {spark_batch_size}")
print(f"  Runs: {runs}")

def run_training(use_ann=False, ann_k=20, ann_backend='nearpy', force_backend=False):
    """Run one training experiment"""
    if force_backend and not use_ann:
        method = f"Exact-{ann_backend}"
    elif use_ann:
        method = f"ANN-{ann_backend}(k={ann_k})"
    else:
        method = "Exact"
    
    print(f"\n{'='*70}")
    print(f"Running {method} CRAIG")
    print(f"{'='*70}")
    
    # Create model — IDENTICAL to MNIST
    model = Sequential()
    model.add(Dense(100, input_dim=784, kernel_regularizer=l2(reg)))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, kernel_regularizer=l2(reg)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
    
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    train_acc = np.zeros(epochs)
    test_acc = np.zeros(epochs)
    greedy_times = np.zeros(epochs)
    
    total_start = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Predict and select subset
        pred_start = time.time()
        preds = model.predict(X_train, verbose=0)
        pred_time = time.time() - pred_start
        
        # Subtract ground truth (residual)
        preds = preds - Y_train
        
        # CRAIG subset selection
        greedy_start = time.time()
        B = int(subset_size * len(X_train))
        subset, subset_weight, _, _, ordering_time, similarity_time, weight_time = util.get_orders_and_weights(
            B, preds, 'euclidean', smtk=smtk, no=0, y=Y_train_nocat, 
            stoch_greedy=0, equal_num=True,
            use_ann=use_ann, ann_k=ann_k, ann_backend=ann_backend, force_backend=force_backend,
            batch_size=spark_batch_size
        )
        greedy_times[epoch] = time.time() - greedy_start
        
        print(f"  Greedy time: {greedy_times[epoch]:.3f}s (ordering: {ordering_time:.3f}s, sim: {similarity_time:.3f}s)")
        
        # Train on subset
        weights = np.zeros(len(X_train))
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        weights[subset] = subset_weight
        
        train_start = time.time()
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, 
                           sample_weight=weights, verbose=0)
        train_time = time.time() - train_start
        
        train_loss[epoch] = history.history['loss'][0]
        train_acc[epoch] = history.history['accuracy'][0]
        
        # Evaluate
        test_result = model.evaluate(X_test, Y_test, verbose=0)
        test_loss[epoch] = test_result[0]
        test_acc[epoch] = test_result[1]
        
        print(f"  Train acc: {train_acc[epoch]:.4f}, Test acc: {test_acc[epoch]:.4f}, Train time: {train_time:.2f}s")
    
    total_time = time.time() - total_start
    
    print(f"\n✓ {method} completed in {total_time:.1f}s")
    print(f"  Avg greedy time: {np.mean(greedy_times):.3f}s")
    print(f"  Total greedy time: {np.sum(greedy_times):.3f}s")
    print(f"  Final test accuracy: {test_acc[-1]:.4f}")
    print(f"  Best test accuracy: {np.max(test_acc):.4f}")
    
    return {
        'method': method,
        'use_ann': use_ann,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'greedy_times': greedy_times,
        'total_time': total_time,
        'avg_greedy_time': np.mean(greedy_times),
        'total_greedy_time': np.sum(greedy_times),
        'final_acc': test_acc[-1],
        'best_acc': np.max(test_acc)
    }

# Run Exact version
force_backend = (ann_backend in ['milvus', 'spark'])
exact_result = run_training(use_ann=False, ann_backend=ann_backend, force_backend=force_backend)

# Run ANN version
ann_result = run_training(use_ann=True, ann_k=ann_k, ann_backend=ann_backend, force_backend=force_backend)

# Compare results
print(f"\n{'='*70}")
print("COMPARISON RESULTS")
print(f"{'='*70}")

print(f"\n⏱️  GREEDY SELECTION TIME:")
print(f"   Exact (avg):      {exact_result['avg_greedy_time']:8.3f}s")
print(f"   ANN (avg):        {ann_result['avg_greedy_time']:8.3f}s")
speedup = exact_result['avg_greedy_time'] / ann_result['avg_greedy_time']
print(f"   Speedup:          {speedup:8.2f}x  {'✓' if speedup > 1 else '✗'}")

print(f"\n   Exact (total):    {exact_result['total_greedy_time']:8.3f}s")
print(f"   ANN (total):      {ann_result['total_greedy_time']:8.3f}s")
total_speedup = exact_result['total_greedy_time'] / ann_result['total_greedy_time']
print(f"   Total speedup:    {total_speedup:8.2f}x")

print(f"\n⏱️  TOTAL TRAINING TIME:")
print(f"   Exact:            {exact_result['total_time']:8.1f}s ({exact_result['total_time']/60:.1f} min)")
print(f"   ANN:              {ann_result['total_time']:8.1f}s ({ann_result['total_time']/60:.1f} min)")
overall_speedup = exact_result['total_time'] / ann_result['total_time']
print(f"   Overall speedup:  {overall_speedup:8.2f}x")

print(f"\n🎯 TEST ACCURACY:")
print(f"   Exact (final):    {exact_result['final_acc']:8.4f}")
print(f"   ANN (final):      {ann_result['final_acc']:8.4f}")
acc_diff = ann_result['final_acc'] - exact_result['final_acc']
print(f"   Difference:       {acc_diff:+8.4f}")

print(f"\n   Exact (best):     {exact_result['best_acc']:8.4f}")
print(f"   ANN (best):       {ann_result['best_acc']:8.4f}")
best_diff = ann_result['best_acc'] - exact_result['best_acc']
print(f"   Difference:       {best_diff:+8.4f}")

# Save results
os.makedirs('results', exist_ok=True)

# Save CSV summary
summary = pd.DataFrame([{
    'dataset': 'Fashion-MNIST',
    'backend': ann_backend,
    'subset_size': subset_size,
    'epochs': epochs,
    'ann_k': ann_k,
    'exact_avg_greedy_time': exact_result['avg_greedy_time'],
    'ann_avg_greedy_time': ann_result['avg_greedy_time'],
    'greedy_speedup': speedup,
    'exact_total_time': exact_result['total_time'],
    'ann_total_time': ann_result['total_time'],
    'overall_speedup': overall_speedup,
    'exact_final_acc': exact_result['final_acc'],
    'ann_final_acc': ann_result['final_acc'],
    'acc_diff': acc_diff,
    'exact_best_acc': exact_result['best_acc'],
    'ann_best_acc': ann_result['best_acc'],
    'best_acc_diff': best_diff
}])

csv_filename = f'results/fashion_mnist_comparison_summary_{ann_backend}.csv'
summary.to_csv(csv_filename, index=False)
print(f"📄 Summary saved to: {csv_filename}")

print(f"\n{'='*70}")
print("✓ Fashion-MNIST COMPARISON COMPLETED!")
print(f"{'='*70}\n")
