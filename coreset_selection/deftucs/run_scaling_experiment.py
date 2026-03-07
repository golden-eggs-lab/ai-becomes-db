#!/usr/bin/env python3
"""
DEFT-UCS WikiLarge: Dataset Size Scaling Experiment
Vary training data ratio: 1.0, 0.7, 0.5, 0.3
Measure: e2e time (selection only), downstream SARI, recall
Baseline (numpy) × 1 run, Optimized (FAISS) × 2 runs per ratio
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "in_memory_and_milvus"))

from config import ARTIFACTS_DIR, CACHE_DIR
from prepare_wikilarge import prepare_wikilarge
from demo_memory import run_core_set_selection

# Same params as run_wikilarge_comparison.py
SELECTION_RATIO = 0.325
K = 7


def do_selection(setting, embeddings, n_samples):
    """Run coreset selection. Returns (selected_indices, elapsed)."""
    A = max(1, int(n_samples * SELECTION_RATIO / K))
    
    if setting == "baseline":
        selected_indices, elapsed, km_stats = run_core_set_selection(
            embeddings, K=K, A=A, alpha=0.5, beta=0.5,
            is_ann=False, is_topk=False, is_reuse_l2=False,
            use_vecdb=False, rng_seed=42,
            use_faiss_exact=False, use_blocked_knn=False,
        )
    else:
        selected_indices, elapsed, km_stats = run_core_set_selection(
            embeddings, K=K, A=A, alpha=0.5, beta=0.5,
            is_ann=True, is_topk=True, is_reuse_l2=True,
            use_vecdb=False, rng_seed=42,
            use_faiss_exact=False, use_hnsw=False,
            use_blocked_knn=False, probe_frac=0.5,
        )
    
    return np.unique(np.array(selected_indices)), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratios', nargs='+', type=float, default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument('--output-dir', type=str, default=str(ARTIFACTS_DIR / 'wikilarge_scaling'))
    parser.add_argument('--skip-downstream', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load full data
    print("Loading WikiLarge data + embeddings...")
    samples, embeddings = prepare_wikilarge(device="cuda")
    N = len(samples)
    print(f"Full dataset: N={N}, embeddings={embeddings.shape}")
    
    results = {}
    
    for ratio in args.ratios:
        n_sub = int(N * ratio)
        print(f"\n{'='*60}")
        print(f"RATIO={ratio} (n_samples={n_sub})")
        print(f"{'='*60}")
        
        # Subsample
        np.random.seed(42)
        if ratio < 1.0:
            subset_idx = np.sort(np.random.choice(N, n_sub, replace=False))
            sub_embeddings = embeddings[subset_idx]
            sub_samples = [samples[i] for i in subset_idx]
        else:
            subset_idx = np.arange(N)
            sub_embeddings = embeddings
            sub_samples = samples
        
        ratio_results = {'ratio': ratio, 'n_samples': n_sub}
        
        # Baseline × 1
        print("\n[Baseline: numpy]")
        base_idx, base_time = do_selection("baseline", sub_embeddings, n_sub)
        # Map back to original indices
        base_orig_idx = set(subset_idx[base_idx] if ratio < 1.0 else base_idx)
        ratio_results['baseline_time'] = base_time
        ratio_results['baseline_selected'] = len(base_idx)
        print(f"  Time: {base_time:.2f}s, Selected: {len(base_idx)}")
        
        # Optimized × 2
        opt_times = []
        opt_orig_idx_list = []
        for run in range(2):
            print(f"\n[Optimized run{run+1}: FAISS]")
            opt_idx, opt_time = do_selection("optimized", sub_embeddings, n_sub)
            opt_orig_idx = set(subset_idx[opt_idx] if ratio < 1.0 else opt_idx)
            opt_times.append(opt_time)
            opt_orig_idx_list.append(opt_orig_idx)
            print(f"  Time: {opt_time:.2f}s, Selected: {len(opt_idx)}")
        
        ratio_results['optimized_times'] = opt_times
        ratio_results['optimized_selected'] = len(opt_orig_idx_list[0])
        
        # Recall
        recall = len(base_orig_idx & opt_orig_idx_list[0]) / len(base_orig_idx) if base_orig_idx else 1.0
        ratio_results['recall'] = recall
        print(f"\n  Recall: {recall*100:.1f}%")
        
        # Downstream (SARI) — skip if requested
        if not args.skip_downstream:
            from run_wikilarge_comparison import finetune, run_evaluation, OUTPUT_DIR
            
            # Finetune + eval for baseline
            base_samples = [sub_samples[i] for i in base_idx]
            print(f"\n[Downstream: Baseline] {len(base_samples)} samples")
            finetune("baseline", base_samples)
            
            # Finetune + eval for optimized (first run)
            # Map global indices back to sub_samples indices
            opt_sub_idx = [i for i in range(len(subset_idx)) if subset_idx[i] in opt_orig_idx_list[0]]
            opt_samples = [sub_samples[i] for i in opt_sub_idx]
            print(f"[Downstream: Optimized] {len(opt_samples)} samples")
            finetune("optimized", opt_samples)
            
            # Eval both
            eval_results = run_evaluation(["baseline", "optimized"])
            ratio_results['baseline_sari'] = eval_results.get('baseline', {}).get('sari')
            ratio_results['optimized_sari'] = eval_results.get('optimized', {}).get('sari')
        
        results[str(ratio)] = ratio_results
        
        # Save intermediate
        with open(os.path.join(args.output_dir, 'scaling_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("DEFT-UCS SCALING SUMMARY")
    print(f"{'='*60}")
    for ratio_str, d in sorted(results.items(), key=lambda x: -float(x[0])):
        opt_mean = np.mean(d['optimized_times'])
        speedup = d['baseline_time'] / opt_mean if opt_mean > 0 else 0
        print(f"\nRatio={ratio_str} (n={d['n_samples']}):")
        print(f"  Time:  baseline={d['baseline_time']:.2f}s, opt={opt_mean:.2f}s, speedup={speedup:.2f}x")
        print(f"  Recall: {d['recall']*100:.1f}%")
        if 'baseline_sari' in d:
            print(f"  SARI: baseline={d['baseline_sari']:.2f}, opt={d['optimized_sari']:.2f}")
    
    print(f"\nSaved to {args.output_dir}/scaling_results.json")


if __name__ == "__main__":
    main()
