#!/usr/bin/env python3
"""
SemDeDup CIFAR-10: Dataset Size Scaling Experiment
Uses the ORIGINAL SemDeDupProcessor from run_cifar10_experiment.py
to ensure measurements match the baseline (86.8s for ratio=1.0).

Vary training data ratio: 0.7, 0.5, 0.3
Measure: e2e dedup time (clustering + dedup), recall (overlap of kept indices)
Baseline (Exact, NoCache) × 1 run, Optimized (ANN, Cache) × 2 runs per ratio
"""

import os
import sys
import time
import json
import pickle
import argparse
import shutil
import numpy as np
import faiss
from tqdm import tqdm
from datetime import datetime

# Import the original classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cifar10_experiment import (
    CONFIG, SemDeDupProcessor, init_memmap_embs
)

import torchvision.datasets as datasets


def run_clustering_on_subset(subset_embs, num_clusters, save_folder):
    """
    Run KMeans clustering on subset embeddings and save sorted clusters.
    Returns clustering_time.
    """
    embs_array = subset_embs.copy().astype(np.float32)
    faiss.normalize_L2(embs_array)
    
    sorted_clusters_path = os.path.join(save_folder, 'sorted_clusters')
    os.makedirs(sorted_clusters_path, exist_ok=True)
    
    start_time = time.time()
    
    kmeans = faiss.Kmeans(
        embs_array.shape[1], num_clusters,
        niter=20, verbose=False, spherical=True, gpu=False
    )
    kmeans.train(embs_array)
    
    _, assignments = kmeans.index.search(embs_array, 1)
    assignments = assignments.flatten()
    
    centroids = kmeans.centroids
    
    for cid in range(num_clusters):
        mask = assignments == cid
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            cluster_data = np.empty((0, 2))
            np.save(os.path.join(sorted_clusters_path, f"cluster_{cid}.npy"), cluster_data)
            continue
        
        cluster_embs = embs_array[indices]
        centroid = centroids[cid:cid+1]
        distances = np.dot(cluster_embs, centroid.T).flatten()
        
        sorted_order = np.argsort(-distances)
        sorted_indices = indices[sorted_order]
        sorted_distances = distances[sorted_order]
        
        cluster_data = np.column_stack([sorted_distances, sorted_indices])
        np.save(os.path.join(sorted_clusters_path, f"cluster_{cid}.npy"), cluster_data)
    
    clustering_time = time.time() - start_time
    return clustering_time


def run_dedup(embs_subset, num_clusters, save_folder, use_ann, use_cache, eps_list, nlist=50, nprobe=5):
    """
    Run SemDeDup dedup on the subset using the ORIGINAL SemDeDupProcessor.
    Returns (kept_indices, total_time_with_clustering).
    """
    # Step 1: cluster
    clustering_time = run_clustering_on_subset(embs_subset, num_clusters, save_folder)
    
    # Step 2: create temp memmap for the subset embeddings (processor loads by index)
    embs_path = os.path.join(save_folder, 'subset_embs.dat')
    n, d = embs_subset.shape
    mm = np.memmap(embs_path, dtype='float32', mode='w+', shape=(n, d))
    mm[:] = embs_subset[:]
    mm.flush()
    
    # Step 3: create processor using original class
    processor = SemDeDupProcessor(use_ann=use_ann, use_cache=use_cache)
    # Override paths to point to our subset data
    processor.sorted_clusters_path = os.path.join(save_folder, 'sorted_clusters')
    processor.embs = np.memmap(embs_path, dtype='float32', mode='r', shape=(n, d))
    # Override ANN params if needed
    if use_ann:
        processor.nlist = nlist
        processor.nprobe = nprobe
    
    # Step 4: run dedup per cluster
    output_dir = os.path.join(save_folder, 'dataframes')
    os.makedirs(output_dir, exist_ok=True)
    
    dedup_time = 0
    for cid in tqdm(range(num_clusters), desc="Dedup", leave=False):
        elapsed = processor.process_cluster(cid, output_dir)
        dedup_time += elapsed
    
    total_time = clustering_time + dedup_time
    
    # Step 5: extract kept indices (using eps=0.1)
    eps = eps_list[0]
    kept_indices = []
    for cid in range(num_clusters):
        cluster_npy = np.load(
            os.path.join(save_folder, 'sorted_clusters', f"cluster_{cid}.npy"),
            allow_pickle=True
        )
        if len(cluster_npy) == 0:
            continue
        cluster_orig_indices = cluster_npy[:, 1].astype(int)
        
        df_file = os.path.join(output_dir, f"cluster_{cid}.pkl")
        with open(df_file, 'rb') as f:
            df = pickle.load(f)
        
        for i, orig_idx in enumerate(cluster_orig_indices):
            if not df[f'eps={eps}'].iloc[i]:
                kept_indices.append(int(orig_idx))
    
    timing_breakdown = processor.get_timing_breakdown()
    
    return sorted(kept_indices), total_time, clustering_time, dedup_time, timing_breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.7, 0.5, 0.3])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output_scaling')
    parser.add_argument('--skip-downstream', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    
    # Load full embeddings (same as original)
    print("Loading embeddings...")
    embs_mmap = init_memmap_embs(
        CONFIG['embs_memory_loc'],
        CONFIG['dataset_size'],
        CONFIG['emb_size']
    )
    full_embeddings = np.array(embs_mmap, dtype=np.float32)
    print(f"Full embeddings: {full_embeddings.shape}")
    
    # Load labels
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    full_labels = np.array(train_dataset.targets)
    
    num_clusters = CONFIG['num_clusters']
    eps_list = CONFIG['eps_list']
    
    results = {}
    
    for ratio in args.ratios:
        n_samples = int(CONFIG['dataset_size'] * ratio)
        print(f"\n{'='*60}")
        print(f"RATIO={ratio} (n_samples={n_samples})")
        print(f"{'='*60}")
        
        # Stratified subsample
        np.random.seed(args.seed)
        subset_indices = []
        for cls in range(10):
            cls_indices = np.where(full_labels == cls)[0]
            n_per_class = int(len(cls_indices) * ratio)
            chosen = np.random.choice(cls_indices, n_per_class, replace=False)
            subset_indices.extend(chosen.tolist())
        subset_indices = sorted(subset_indices)
        
        subset_embs = full_embeddings[subset_indices]
        print(f"Subset: {len(subset_indices)} samples")
        
        ratio_results = {
            'ratio': ratio,
            'n_samples': len(subset_indices),
        }
        
        # --- Baseline (Exact, NoCache) × 1 ---
        print("\n[Baseline: Exact, NoCache]")
        save_folder_b = os.path.join(args.output_dir, f'ratio_{ratio}_baseline')
        if os.path.exists(save_folder_b):
            shutil.rmtree(save_folder_b)
        
        kept_b, time_b, clust_b, dedup_b, tb_b = run_dedup(
            subset_embs, num_clusters, save_folder_b,
            use_ann=False, use_cache=False, eps_list=eps_list
        )
        print(f"  Time: {time_b:.2f}s (clustering: {clust_b:.2f}s, dedup: {dedup_b:.2f}s), Kept: {len(kept_b)}")
        ratio_results['baseline_time'] = time_b
        ratio_results['baseline_clustering'] = clust_b
        ratio_results['baseline_dedup'] = dedup_b
        ratio_results['baseline_kept'] = len(kept_b)
        
        # --- Optimized (ANN, Cache) × 2 ---
        opt_times = []
        opt_kept_list = []
        for run in range(2):
            print(f"\n[Optimized run{run+1}: ANN, Cache]")
            save_folder_o = os.path.join(args.output_dir, f'ratio_{ratio}_opt_run{run+1}')
            if os.path.exists(save_folder_o):
                shutil.rmtree(save_folder_o)
            
            kept_o, time_o, clust_o, dedup_o, tb_o = run_dedup(
                subset_embs, num_clusters, save_folder_o,
                use_ann=True, use_cache=True, eps_list=eps_list,
                nlist=CONFIG.get('nlist', 50), nprobe=CONFIG.get('nprobe', 5)
            )
            opt_times.append(time_o)
            opt_kept_list.append(set(kept_o))
            print(f"  Time: {time_o:.2f}s (clustering: {clust_o:.2f}s, dedup: {dedup_o:.2f}s), Kept: {len(kept_o)}")
        
        ratio_results['optimized_times'] = opt_times
        ratio_results['optimized_kept'] = len(opt_kept_list[0])
        
        # Recall
        recall = len(set(kept_b) & opt_kept_list[0]) / len(kept_b) if kept_b else 1.0
        ratio_results['recall'] = recall
        print(f"\n  Recall: {recall*100:.1f}%")
        
        results[str(ratio)] = ratio_results
        
        # Save intermediate
        with open(os.path.join(args.output_dir, 'scaling_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r_str, d in sorted(results.items(), key=lambda x: -float(x[0])):
        opt_mean = np.mean(d['optimized_times'])
        speedup = d['baseline_time'] / opt_mean if opt_mean > 0 else 0
        print(f"\nRatio={r_str} (n={d['n_samples']}):")
        print(f"  Baseline:  {d['baseline_time']:.2f}s  (kept={d['baseline_kept']})")
        print(f"  Optimized: {opt_mean:.2f}s  (speedup: {speedup:.2f}x)")
        print(f"  Recall: {d['recall']*100:.1f}%")
    
    print(f"\nResults saved to {args.output_dir}/scaling_results.json")


if __name__ == "__main__":
    main()
