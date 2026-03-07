#!/usr/bin/env python
"""
Coreset (DEFT-UCS) ANN Hyperparameter Study

Measures recall and search time for FAISS IVF vs Flat on the KMeans
centroid-assignment step used by the DEFT-UCS coreset algorithm.

Architecture (matching demo_memory.py exactly):
  - Index is built on K CENTROIDS (not N vectors)
  - N vectors are the QUERIES (each finds its nearest centroid)
  - Default: nlist = min(ivf_nlist, sqrt(K)), nprobe = min(ivf_nprobe, nlist, ...)
  - With K=7: nlist_effective = 2, nprobe_effective = 1

Recall = fraction of vectors assigned to the same centroid as exact search,
averaged over multiple KMeans iterations.

Usage:
    conda run -n vecdb python run_ann_hyperparam_coreset.py --runs 3
"""

import os, sys, json, time, argparse, math
import numpy as np
import faiss
from datetime import datetime
from tqdm import tqdm

# ── hyperparameters ──────────────────────────────────────────────────
# These are the "requested" ivf_nlist/ivf_nprobe passed to the KMeans function.
# The actual effective values are min(ivf_nlist, sqrt(K)).
# For K=7, sqrt(K)=2, so nlist is always capped to 2.

NPROBE_VALUES = [1, 5, 10, 25, 50, 100]
NLIST_VALUES  = [25, 50, 100, 200, 400, 1000]

# For sweeping, we fix one and vary the other (matching all other scripts)
ANCHOR_NLIST  = 100   # requested nlist when sweeping nprobe
ANCHOR_NPROBE = 10    # requested nprobe when sweeping nlist

# Coreset defaults (from config.py FAISS_CONFIG)
DEFAULT_NLIST  = 256
DEFAULT_NPROBE = 128

# Following the paper
K_CLUSTERS = 7
MAX_KMEANS_ITER = 50
KMEANS_TOL = 1e-4
SEED = 42


# ── data loading ─────────────────────────────────────────────────────
def load_coreset_embeddings():
    """Load CoEDIT embeddings (Sentence-T5-base, 768D)."""
    # Try cached embeddings first
    cache_path = os.path.join(os.path.dirname(__file__), "cache_coreset_embeddings.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"Loaded cached embeddings: {data['embeddings'].shape}")
        return data['embeddings'].astype(np.float32)
    
    # Try coreset_release cache
    for candidate in [
        os.path.join(os.path.dirname(__file__), "experiments", "cache", "coedit_embeddings.npy"),
        os.path.join(os.path.dirname(__file__), "coreset_release", "cache", "embeddings.npy"),
        os.path.join(os.path.dirname(__file__), "coreset_release", "artifacts", "embeddings.npy"),
    ]:
        if os.path.exists(candidate):
            embeddings = np.load(candidate).astype(np.float32)
            print(f"Loaded embeddings from {candidate}: {embeddings.shape}")
            return embeddings
    
    print("Computing CoEDIT embeddings (first run only)...")
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: Need datasets, sentence-transformers. Install them first.")
        sys.exit(1)
    
    dataset = load_dataset("grammarly/coedit", split="train")
    texts = [item["src"] for item in dataset]
    
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=False).astype(np.float32)
    
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Computed and cached embeddings: {embeddings.shape}")
    return embeddings


# ── KMeans with exact assignment (baseline) ──────────────────────────
def kmeans_exact(vectors, K, max_iter=MAX_KMEANS_ITER, tol=KMEANS_TOL, seed=SEED):
    """
    Run exact KMeans. Returns (labels, centroids, total_search_time).
    Mirrors demo_memory.py's baseline path (numpy broadcast distance).
    """
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    selected = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected].copy()
    
    total_search_time = 0.0
    
    for it in range(max_iter):
        # Assign: exact L2 using numpy broadcast (matching actual pipeline baseline)
        t0 = time.time()
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        x_sq = np.sum(vectors ** 2, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        dists = x_sq + c_sq - 2.0 * vectors @ centroids.T  # (N, K)
        labels = np.argmin(dists, axis=1).astype(np.int32)
        total_search_time += time.time() - t0
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= np.maximum(counts[non_empty], 1e-8)[:, None]
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
        
        diff = np.linalg.norm(new_centroids - centroids)
        norm = np.linalg.norm(centroids)
        rel_change = diff / norm if norm > 1e-8 else diff
        centroids = new_centroids
        
        if rel_change < tol:
            break
    
    return labels, centroids, total_search_time


# ── KMeans with ANN assignment ───────────────────────────────────────
def kmeans_ann(vectors, K, ivf_nlist, ivf_nprobe,
               max_iter=MAX_KMEANS_ITER, tol=KMEANS_TOL, seed=SEED):
    """
    Run ANN KMeans (matching demo_memory.py's IVF path).
    The IVF index is built on centroids, queries are all N vectors.
    nlist is capped to min(ivf_nlist, sqrt(K)), matching the original code.
    Returns (labels, centroids, total_search_time).
    """
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    selected = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected].copy()
    
    # Effective nlist (matching demo_memory.py line 121-122)
    sqrtK = max(1, int(round(math.sqrt(K))))
    actual_nlist = min(ivf_nlist, sqrtK)
    
    # Pre-create IVF index (matching demo_memory.py line 127-136)
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, actual_nlist)
    
    # Train once on a random sample of vectors
    min_train = 39 * actual_nlist
    train_size = min(N, max(min_train, 10000))
    train_idx = rng.choice(N, size=train_size, replace=False)
    train_sample = vectors[train_idx].astype(np.float32)
    index_ivf.train(train_sample)
    
    # Effective nprobe (matching demo_memory.py line 134-136)
    probe_frac = 0.5
    target_probe = max(1, int(probe_frac * actual_nlist))
    nprobe_cap = 64
    effective_nprobe = min(ivf_nprobe, actual_nlist, nprobe_cap, target_probe)
    index_ivf.nprobe = effective_nprobe
    
    total_search_time = 0.0
    
    for it in range(max_iter):
        # Reset + add centroids (matching demo_memory.py line 249-251)
        index_ivf.reset()
        index_ivf.add(centroids.astype(np.float32))
        
        t0 = time.time()
        D, I = index_ivf.search(vectors.astype(np.float32), 1)
        total_search_time += time.time() - t0
        
        labels = I.ravel().astype(np.int32)
        
        # Update centroids (same as exact)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= np.maximum(counts[non_empty], 1e-8)[:, None]
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
        
        diff = np.linalg.norm(new_centroids - centroids)
        norm = np.linalg.norm(centroids)
        rel_change = diff / norm if norm > 1e-8 else diff
        centroids = new_centroids
        
        if rel_change < tol:
            break
    
    return labels, centroids, total_search_time


# ── KMeans with ANN (NO nlist cap, for meaningful sweep) ────────────
def kmeans_ann_uncapped(vectors, K, ivf_nlist, ivf_nprobe,
                        max_iter=MAX_KMEANS_ITER, tol=KMEANS_TOL, seed=SEED):
    """
    Same as kmeans_ann but WITHOUT the sqrt(K) cap on nlist.
    This allows the hyperparameter sweep to show meaningful variation.
    The index is still built on centroids (K vectors).
    
    Note: nlist is capped to max(1, K//2) to ensure FAISS can train.
    """
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    selected = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected].copy()
    
    # Uncapped nlist (but still bounded by K)
    actual_nlist = max(1, min(ivf_nlist, K // 2))
    actual_nprobe = min(ivf_nprobe, actual_nlist)
    
    # Build IVF
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, actual_nlist)
    
    # Train on vectors
    min_train = 39 * actual_nlist
    train_size = min(N, max(min_train, 10000))
    train_idx = rng.choice(N, size=train_size, replace=False)
    index_ivf.train(vectors[train_idx].astype(np.float32))
    
    index_ivf.nprobe = actual_nprobe
    
    total_search_time = 0.0
    
    for it in range(max_iter):
        index_ivf.reset()
        index_ivf.add(centroids.astype(np.float32))
        
        t0 = time.time()
        D, I = index_ivf.search(vectors.astype(np.float32), 1)
        total_search_time += time.time() - t0
        
        labels = I.ravel().astype(np.int32)
        
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= np.maximum(counts[non_empty], 1e-8)[:, None]
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
        
        diff = np.linalg.norm(new_centroids - centroids)
        norm = np.linalg.norm(centroids)
        rel_change = diff / norm if norm > 1e-8 else diff
        centroids = new_centroids
        
        if rel_change < tol:
            break
    
    return labels, centroids, total_search_time


def compute_recall(exact_labels, ann_labels):
    """Recall = fraction of vectors assigned to same centroid."""
    return float(np.mean(exact_labels == ann_labels))


# ── main ─────────────────────────────────────────────────────────────
def run_single(vectors, K, nprobe_values, nlist_values, run_id):
    print(f"\n{'='*60}\nRun {run_id}\n{'='*60}")
    
    # Exact KMeans (baseline)
    exact_labels, exact_centroids, exact_time = kmeans_exact(vectors, K)
    print(f"  Exact KMeans: search_time={exact_time:.4f}s")
    
    # nprobe sweep (fix nlist=ANCHOR_NLIST, vary nprobe)
    nprobe_results = []
    for nprobe in nprobe_values:
        ann_labels, _, ann_time = kmeans_ann(vectors, K, ANCHOR_NLIST, nprobe)
        recall = compute_recall(exact_labels, ann_labels)
        nprobe_results.append({
            "nprobe": nprobe, "recall": recall, "search_time": ann_time
        })
        print(f"  nprobe={nprobe:4d}  recall={recall:.6f}  time={ann_time:.4f}s")
    
    # nlist sweep (fix nprobe=ANCHOR_NPROBE, vary nlist)
    nlist_results = []
    for nlist in nlist_values:
        ann_labels, _, ann_time = kmeans_ann(vectors, K, nlist, ANCHOR_NPROBE)
        recall = compute_recall(exact_labels, ann_labels)
        nlist_results.append({
            "nlist": nlist, "recall": recall, "search_time": ann_time
        })
        print(f"  nlist={nlist:4d}   recall={recall:.6f}  time={ann_time:.4f}s")
    
    return nprobe_results, nlist_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--k-clusters", type=int, default=K_CLUSTERS,
                        help=f"Number of KMeans clusters (default: {K_CLUSTERS} from paper)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    K = args.k_clusters
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results/ann_hyperparam_coreset_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file) or "results", exist_ok=True)
    
    embeddings = load_coreset_embeddings()
    n, d = embeddings.shape
    
    # Show effective parameters
    sqrtK = max(1, int(round(math.sqrt(K))))
    eff_nlist = min(ANCHOR_NLIST, sqrtK)
    eff_nprobe = min(ANCHOR_NPROBE, eff_nlist, 64, max(1, int(0.5 * eff_nlist)))
    
    print(f"\nCoreset (DEFT-UCS) ANN Hyperparameter Study")
    print(f"  Embeddings: {n}×{d}, K={K} clusters")
    print(f"  sqrt(K)={sqrtK}, so effective nlist is always capped to {sqrtK}")
    print(f"  Default: ivf_nlist={DEFAULT_NLIST}, ivf_nprobe={DEFAULT_NPROBE}")
    print(f"  Effective: nlist={eff_nlist}, nprobe={eff_nprobe}")
    print(f"  nprobe sweep: {NPROBE_VALUES} (all will be capped)")
    print(f"  nlist sweep:  {NLIST_VALUES} (all will be capped to sqrt({K})={sqrtK})")
    
    all_nprobe = {p: [] for p in NPROBE_VALUES}
    all_nlist  = {nl: [] for nl in NLIST_VALUES}
    
    for run_id in range(1, args.runs + 1):
        np_res, nl_res = run_single(embeddings, K, NPROBE_VALUES, NLIST_VALUES, run_id)
        for r in np_res:
            all_nprobe[r["nprobe"]].append(r)
        for r in nl_res:
            all_nlist[r["nlist"]].append(r)
    
    # Aggregate
    nprobe_agg = []
    for p in NPROBE_VALUES:
        runs = all_nprobe[p]
        nprobe_agg.append({
            "nprobe": p,
            "recall_mean": float(np.mean([r["recall"] for r in runs])),
            "recall_std":  float(np.std([r["recall"] for r in runs])),
            "search_time_mean": float(np.mean([r["search_time"] for r in runs])),
            "search_time_std":  float(np.std([r["search_time"] for r in runs])),
            "all_recalls": [r["recall"] for r in runs],
        })
    
    nlist_agg = []
    for nl in NLIST_VALUES:
        runs = all_nlist[nl]
        nlist_agg.append({
            "nlist": nl,
            "recall_mean": float(np.mean([r["recall"] for r in runs])),
            "recall_std":  float(np.std([r["recall"] for r in runs])),
            "search_time_mean": float(np.mean([r["search_time"] for r in runs])),
            "search_time_std":  float(np.std([r["search_time"] for r in runs])),
            "all_recalls": [r["recall"] for r in runs],
        })
    
    results = {
        "experiment": "coreset_ann_hyperparameter_study",
        "task": "Coreset",
        "dataset": "CoEDIT",
        "num_samples": int(n),
        "embedding_dim": int(d),
        "k_clusters": int(K),
        "num_runs": args.runs,
        "timestamp": timestamp,
        "anchor_nlist": ANCHOR_NLIST,
        "anchor_nprobe": ANCHOR_NPROBE,
        "default_nlist": DEFAULT_NLIST,
        "default_nprobe": DEFAULT_NPROBE,
        "effective_nlist_note": f"All nlist values capped to sqrt(K)={sqrtK} in the original code",
        "nprobe_sweep": nprobe_agg,
        "nlist_sweep": nlist_agg,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
