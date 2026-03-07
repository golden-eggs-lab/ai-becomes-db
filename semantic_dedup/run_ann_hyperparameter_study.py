#!/usr/bin/env python
"""
SemDeDup ANN Hyperparameter Study

Test FAISS IVF hyperparameters (nprobe and nlist) impact on recall.
Recall = overlap between ANN duplicate detection and exact duplicate detection.

Anchor values (current defaults): nlist=100, nprobe=10
Experiments:
  1. Vary nprobe [1, 5, 10, 25, 50, 100] with fixed nlist=100
  2. Vary nlist [25, 50, 100, 200, 400] with fixed nprobe=10

Usage:
    python run_ann_hyperparameter_study.py
    python run_ann_hyperparameter_study.py --quick  # Fewer configurations
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import pickle
import faiss
import time
import json
from datetime import datetime
from tqdm import tqdm
from semdedup import init_memmap_embs


# Anchor values (current defaults)
ANCHOR_NLIST = 100
ANCHOR_NPROBE = 10

# Sweep ranges
NPROBE_VALUES = [1, 5, 10, 25, 50, 100]
NLIST_VALUES = [25, 50, 100, 200, 400]

# Quick test
NPROBE_VALUES_QUICK = [5, 10, 50]
NLIST_VALUES_QUICK = [50, 100, 200]

# Default k for ANN
ANN_K = 100


class SemDeDupHyperparamStudy:
    """Study nprobe and nlist impact on recall for SemDeDup."""
    
    def __init__(self, config_path, ann_k=ANN_K):
        self.ann_k = ann_k
        
        # Load config
        with open(config_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        
        self.save_folder = params['save_folder']
        self.sorted_clusters_path = params['sorted_clusters_path']
        self.num_clusters = params['num_clusters']
        self.dataset_size = params['dataset_size']
        self.emb_size = params['emd_size']
        self.embs_memory_loc = params['embs_memory_loc']
        
        # Load embeddings
        self.embs = init_memmap_embs(self.embs_memory_loc, self.dataset_size, self.emb_size)
        
        # Test parameters
        self.eps = 0.1  # Use eps=0.1 for comparison
        
        print(f"Loaded config: {self.num_clusters} clusters, {self.dataset_size} samples, {self.emb_size}D embeddings")
    
    def _get_cluster_data(self, cluster_id):
        """Load cluster data and embeddings."""
        cluster_i = np.load(
            os.path.join(self.sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_ids = cluster_i[:, 1].astype("int32")
        cluster_reps = np.array(self.embs[cluster_ids], dtype=np.float32)
        return cluster_reps
    
    def _semdedup_exact(self, cluster_reps):
        """Exact duplicate detection using FAISS Flat."""
        n, d = cluster_reps.shape
        if n <= 1:
            return set()
        
        # Normalize for cosine similarity
        cluster_reps = cluster_reps.copy()
        faiss.normalize_L2(cluster_reps)
        
        # Exact search
        index = faiss.IndexFlatIP(d)
        index.add(cluster_reps)
        
        k = min(n, n)
        similarities, indices = index.search(cluster_reps, k)
        
        # Find max similarity excluding self
        M = np.zeros(n, dtype=np.float32)
        for i in range(n):
            for j in range(k):
                if indices[i, j] != i:
                    M[i] = similarities[i, j]
                    break
        
        # Duplicates: samples with max_sim > 1 - eps
        duplicates = set(np.where(M > 1 - self.eps)[0])
        return duplicates
    
    def _semdedup_ann(self, cluster_reps, nlist, nprobe):
        """ANN duplicate detection with specific nlist and nprobe."""
        n, d = cluster_reps.shape
        if n <= self.ann_k:
            return self._semdedup_exact(cluster_reps)
        
        # Normalize for cosine similarity
        cluster_reps = cluster_reps.copy()
        faiss.normalize_L2(cluster_reps)
        
        # Build IVF index with custom params
        actual_nlist = max(1, min(nlist, n // 10))  # Can't have more clusters than points/10
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        
        index.train(cluster_reps)
        index.nprobe = min(nprobe, actual_nlist)
        index.add(cluster_reps)
        
        # Search
        k = min(self.ann_k, n)
        similarities, indices = index.search(cluster_reps, k)
        
        # Find max similarity excluding self
        M = np.zeros(n, dtype=np.float32)
        for i in range(n):
            for j in range(k):
                if indices[i, j] != i:
                    M[i] = max(M[i], similarities[i, j])
        
        # Duplicates
        duplicates = set(np.where(M > 1 - self.eps)[0])
        return duplicates
    
    def run_exact_baseline(self):
        """Run exact search on all clusters to get baseline."""
        print("\n" + "="*60)
        print("Running EXACT baseline (FAISS Flat)")
        print("="*60)
        
        all_duplicates = {}  # {cluster_id: set of duplicate indices}
        total_exact_dups = 0
        
        start = time.time()
        for cluster_id in tqdm(range(self.num_clusters), desc="Exact"):
            cluster_reps = self._get_cluster_data(cluster_id)
            duplicates = self._semdedup_exact(cluster_reps)
            all_duplicates[cluster_id] = duplicates
            total_exact_dups += len(duplicates)
        
        elapsed = time.time() - start
        print(f"  Total duplicates (eps={self.eps}): {total_exact_dups}")
        print(f"  Time: {elapsed:.2f}s")
        
        return all_duplicates, elapsed
    
    def run_ann(self, nlist, nprobe):
        """Run ANN search with specific config."""
        all_duplicates = {}
        total_ann_dups = 0
        
        start = time.time()
        for cluster_id in range(self.num_clusters):
            cluster_reps = self._get_cluster_data(cluster_id)
            duplicates = self._semdedup_ann(cluster_reps, nlist, nprobe)
            all_duplicates[cluster_id] = duplicates
            total_ann_dups += len(duplicates)
        
        elapsed = time.time() - start
        return all_duplicates, total_ann_dups, elapsed
    
    def calculate_recall(self, exact_dups, ann_dups):
        """Calculate recall = |ANN ∩ Exact| / |Exact|"""
        total_exact = 0
        total_overlap = 0
        
        for cluster_id in exact_dups:
            exact_set = exact_dups[cluster_id]
            ann_set = ann_dups.get(cluster_id, set())
            
            total_exact += len(exact_set)
            total_overlap += len(exact_set & ann_set)
        
        recall = total_overlap / total_exact if total_exact > 0 else 1.0
        return recall, total_overlap, total_exact
    
    def run_study(self, nprobe_values, nlist_values):
        """Run the full hyperparameter study."""
        
        # Get exact baseline
        exact_dups, exact_time = self.run_exact_baseline()
        total_exact_dups = sum(len(d) for d in exact_dups.values())
        
        # Experiment 1: Vary nprobe (fixed nlist=ANCHOR_NLIST)
        print(f"\n{'='*60}")
        print(f"Experiment 1: Varying nprobe (fixed nlist={ANCHOR_NLIST})")
        print("="*60)
        
        nprobe_results = []
        for nprobe in nprobe_values:
            print(f"  Testing nprobe={nprobe}...", end=" ", flush=True)
            ann_dups, total_ann_dups, ann_time = self.run_ann(ANCHOR_NLIST, nprobe)
            recall, overlap, total = self.calculate_recall(exact_dups, ann_dups)
            print(f"Recall: {recall:.4f} ({overlap}/{total}), Time: {ann_time:.2f}s")
            
            nprobe_results.append({
                'nlist': ANCHOR_NLIST,
                'nprobe': nprobe,
                'recall': float(recall),
                'overlap': int(overlap),
                'total_exact': int(total),
                'ann_duplicates': int(total_ann_dups),
                'time': float(ann_time)
            })
        
        # Experiment 2: Vary nlist (fixed nprobe=ANCHOR_NPROBE)
        print(f"\n{'='*60}")
        print(f"Experiment 2: Varying nlist (fixed nprobe={ANCHOR_NPROBE})")
        print("="*60)
        
        nlist_results = []
        for nlist in nlist_values:
            print(f"  Testing nlist={nlist}...", end=" ", flush=True)
            ann_dups, total_ann_dups, ann_time = self.run_ann(nlist, ANCHOR_NPROBE)
            recall, overlap, total = self.calculate_recall(exact_dups, ann_dups)
            print(f"Recall: {recall:.4f} ({overlap}/{total}), Time: {ann_time:.2f}s")
            
            nlist_results.append({
                'nlist': nlist,
                'nprobe': ANCHOR_NPROBE,
                'recall': float(recall),
                'overlap': int(overlap),
                'total_exact': int(total),
                'ann_duplicates': int(total_ann_dups),
                'time': float(ann_time)
            })
        
        return {
            'exact': {
                'total_duplicates': int(total_exact_dups),
                'time': float(exact_time)
            },
            'nprobe_sweep': {
                'anchor_nlist': ANCHOR_NLIST,
                'results': nprobe_results
            },
            'nlist_sweep': {
                'anchor_nprobe': ANCHOR_NPROBE,
                'results': nlist_results
            }
        }


def main():
    parser = argparse.ArgumentParser(description='SemDeDup ANN Hyperparameter Study')
    parser.add_argument('--config', type=str, default='semdedup_configs.yaml')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer configs')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()
    
    # Select hyperparameter ranges
    if args.quick:
        nprobe_values = NPROBE_VALUES_QUICK
        nlist_values = NLIST_VALUES_QUICK
        print("🚀 QUICK TEST MODE")
    else:
        nprobe_values = NPROBE_VALUES
        nlist_values = NLIST_VALUES
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results/ann_hyperparam_study_{timestamp}.json"
    
    print("="*70)
    print("SemDeDup ANN Hyperparameter Study")
    print("="*70)
    print(f"Anchor values: nlist={ANCHOR_NLIST}, nprobe={ANCHOR_NPROBE}")
    print(f"nprobe sweep: {nprobe_values} (fixed nlist={ANCHOR_NLIST})")
    print(f"nlist sweep: {nlist_values} (fixed nprobe={ANCHOR_NPROBE})")
    print(f"Output: {output_file}")
    print()
    
    # Run study
    study = SemDeDupHyperparamStudy(args.config)
    results = study.run_study(nprobe_values, nlist_values)
    
    # Add metadata
    results['experiment'] = 'semdedup_ann_hyperparameter_study'
    results['num_clusters'] = study.num_clusters
    results['dataset_size'] = study.dataset_size
    results['embedding_dim'] = study.emb_size
    results['eps'] = study.eps
    results['ann_k'] = study.ann_k
    results['timestamp'] = timestamp
    
    # Save
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else 'results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Exact baseline: {results['exact']['total_duplicates']} duplicates in {results['exact']['time']:.2f}s")
    
    print(f"\nNprobe sweep (fixed nlist={ANCHOR_NLIST}):")
    for r in results['nprobe_sweep']['results']:
        print(f"  nprobe={r['nprobe']:3d} → recall={r['recall']:.4f}, time={r['time']:.2f}s")
    
    print(f"\nNlist sweep (fixed nprobe={ANCHOR_NPROBE}):")
    for r in results['nlist_sweep']['results']:
        print(f"  nlist={r['nlist']:3d} → recall={r['recall']:.4f}, time={r['time']:.2f}s")
    
    print(f"\n📊 Results saved to: {output_file}")
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
