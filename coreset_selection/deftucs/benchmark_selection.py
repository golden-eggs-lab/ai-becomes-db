"""
benchmark_selection.py
Selects Coreset samples using Exact (Baseline) or FAISS / NearPy + Reuse (Optimized).
Outputs selected samples as a JSON file for decoupled fine-tuning.
"""
import os
import json
import argparse
import time
from pathlib import Path
import numpy as np

from config import CACHE_DIR, SELECTION_CONFIG, ARTIFACTS_DIR
from coreset_selection import baseline_selection, optimized_selection

def selection_coedit(args):
    print("Loading CoEDIT dataset...")
    embeddings_path = CACHE_DIR / "coedit_embeddings.npy"
    samples_path = CACHE_DIR / "coedit_dataset.json"

    if not embeddings_path.exists() or not samples_path.exists():
        raise FileNotFoundError(f"CoEDIT data not found in cache. Run `prepare_data.py` first.")

    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(samples_path, 'r') as f:
        all_samples = json.load(f)

    K = SELECTION_CONFIG["K"]
    A = SELECTION_CONFIG["A"]
    seed = args.seed

    print(f"Selection params: K={K}, A={A}, seed={seed}")

    if args.setting == "baseline":
        selected_indices, timing = baseline_selection(embeddings, K, A, seed=seed)
    else:
        selected_indices, timing = optimized_selection(embeddings, K, A, seed=seed)

    samples = [all_samples[i] for i in selected_indices]
    return samples, timing

def selection_wikilarge(args):
    from prepare_wikilarge import prepare_wikilarge

    print("Preparing/Loading WikiLarge dataset...")
    all_samples, embeddings = prepare_wikilarge(device="cuda")

    N = len(all_samples)
    SELECTION_RATIO = 0.325
    K = 7
    A = max(1, int(N * SELECTION_RATIO / K))

    print(f"Selection params: N={N}, K={K}, A={A}, target={int(N*SELECTION_RATIO)}")

    if args.setting == "baseline":
        # Baseline: Exact KMeans + Full Sort + Cosine Recompute
        selected_indices, timing = baseline_selection(embeddings, K, A, seed=args.seed, verbose=False)
    else:
        # Optimized: FAISS IVF (ANN) + TopK + L2 Reuse
        selected_indices, timing = optimized_selection(embeddings, K, A, seed=args.seed, verbose=False)

    selected_indices = np.unique(np.array(selected_indices))
    samples = [all_samples[i] for i in selected_indices]

    return samples, timing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--dataset", choices=["coedit", "wikilarge"], required=True)
    parser.add_argument("--output", type=str, required=True, help="JSON path to save selected samples")
    parser.add_argument("--timing-output", type=str, required=True, help="JSON path to save timing info")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"CORESET SELECTION: {args.setting.upper()} on {args.dataset.upper()}")
    print(f"{'='*60}")

    if args.dataset == "coedit":
        samples, timing = selection_coedit(args)
    else:
        samples, timing = selection_wikilarge(args)

    print(f"\nSelected {len(samples)} samples. Total time: {timing['total']:.2f}s")

    # Ensure directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.timing_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(samples, f)
        
    with open(args.timing_output, 'w') as f:
        json.dump(timing, f, indent=2)

if __name__ == "__main__":
    main()
