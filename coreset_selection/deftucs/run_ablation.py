"""
Ablation study with timing breakdown (DEFT-UCS).

Tests combinations of optimizations to isolate contributions:
1. Baseline: Exact KMeans + Recompute cosine + Full Sort
2. +IV1: ANN KMeans + Recompute cosine + Full Sort
3. +IV2: Exact KMeans + Cache L2 (Cosine Reuse) + Full Sort
4. +IV3: Exact KMeans + Recompute cosine + TopK (Argpartition)
5. Optimized: All three (ANN + Reuse L2 + TopK)

Usage:
    python run_ablation.py
"""

import json
import time
import math
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path

from config import CACHE_DIR, ARTIFACTS_DIR, SELECTION_CONFIG
from coreset_selection import baseline_selection, optimized_selection

def run_config(name, kwargs, vectors, K, A, baseline_indices=None):
    """Run a single ablation configuration using optimized_selection toggles."""
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"Params: {kwargs}")
    print(f"{'='*60}")
    
    # We use optimized_selection providing the correct boolean kwargs
    selected_indices, timing = optimized_selection(vectors, K, A, seed=42, verbose=True, **kwargs)
    
    # Validation against baseline
    recall = 1.0
    if baseline_indices is not None:
        overlap = len(set(selected_indices) & baseline_indices)
        recall = overlap / len(baseline_indices)
        timing["recall"] = recall
        print(f"Recall vs baseline: {recall*100:.1f}%")
        
    return timing, set(selected_indices)

def main():
    print("="*60)
    print("DEFT-UCS ABLATION STUDY WITH TIMING BREAKDOWN")
    print("="*60)
    
    # Load embeddings
    embeddings_file = CACHE_DIR / "coedit_embeddings.npy"
    if not embeddings_file.exists():
        print(f"Error: {embeddings_file} not found. Run prepare_data.py first.")
        return
        
    vectors = np.load(embeddings_file).astype(np.float32)
    print(f"Loaded embeddings: {vectors.shape}")
    
    N = vectors.shape[0]
    K = SELECTION_CONFIG["K"]
    A = SELECTION_CONFIG["A"]
    
    print(f"\nConfig: N={N}, K={K}, A={A}")
    
    # Define configurations
    configs = {
        "Baseline": {"is_ann": False, "is_reuse_l2": False, "is_topk": False},
        "+IV1 (ANN)": {"is_ann": True, "is_reuse_l2": False, "is_topk": False},
        "+IV2 (L2 Reuse)": {"is_ann": False, "is_reuse_l2": True, "is_topk": False},
        "+IV3 (Top-k)": {"is_ann": False, "is_reuse_l2": False, "is_topk": True},
        "Optimized (All)": {"is_ann": True, "is_reuse_l2": True, "is_topk": True},
    }
    
    results = {}
    baseline_indices = None
    
    # Run Baseline first to get the recall indices
    timing, indices = run_config("Baseline", configs["Baseline"], vectors, K, A)
    results["Baseline"] = timing
    baseline_indices = indices
    
    # Run variants
    for name, kwargs in configs.items():
        if name == "Baseline": continue
        timing, indices = run_config(name, kwargs, vectors, K, A, baseline_indices=baseline_indices)
        results[name] = timing
    
    # Summary table
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'KMeans(s)':<12} {'Select(s)':<12} {'Total(s)':<12} {'Speedup':<12} {'Recall':<12}")
    print("-" * 80)
    
    baseline_time = results["Baseline"]["total"]
    
    for name, r in results.items():
        speedup = baseline_time / r["total"]
        recall_str = f"{r.get('recall', 1.0)*100:.1f}%"
        print(f"{name:<20} {r['kmeans']:<12.2f} {r['selection']:<12.2f} {r['total']:<12.2f} {speedup:<12.2f}x {recall_str:<12}")
    
    # Save results
    output_dir = ARTIFACTS_DIR / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
