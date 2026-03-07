#!/usr/bin/env python
"""
CodePruning (SCIP) ANN Hyperparameter Study

Measures recall and search time for FAISS IVF vs Flat on centroid
assignment search (N embeddings → K centroids).

Recall = fraction of points assigned to same centroid as exact search.

Usage:
    conda run -n code python run_ann_hyperparam_codepruning.py --runs 3
"""

import os, sys, json, time, argparse
import numpy as np
import faiss
from datetime import datetime
from tqdm import tqdm

# ── hyperparameters ──────────────────────────────────────────────────
ANCHOR_NLIST  = 100
ANCHOR_NPROBE = 10

NPROBE_VALUES = [1, 5, 10, 25, 50, 100]
NLIST_VALUES  = [25, 50, 100, 200, 400, 1000]

# CodePruning defaults (dynamic, for marking on plot)
# default_nlist = sqrt(K) ≈ 31 for K=1000, default_nprobe = nlist//4
# K=1000 matches the ablation experiments (measure_build_query_cache.py, run_inmemory_ablation_v2.py)
DEFAULT_N_CLUSTERS = 1000


# ── data loading ─────────────────────────────────────────────────────
def load_codepruning_data(max_samples=50000):
    """Load or compute SCIP embeddings."""
    # Try to load cached embeddings
    cache_path = os.path.join(os.path.dirname(__file__), "cache_scip_embeddings.npz")
    
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        embeddings = data["embeddings"].astype(np.float32)
        print(f"Loaded cached SCIP embeddings: {embeddings.shape}")
        return embeddings
    
    # Try the aligned experiment embeddings
    aligned_path = os.path.join(os.path.dirname(__file__), "experiments/aligned/embeddings.npy")
    if os.path.exists(aligned_path):
        embeddings = np.load(aligned_path).astype(np.float32)
        print(f"Loaded aligned experiment embeddings: {embeddings.shape}")
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        np.savez_compressed(cache_path, embeddings=embeddings)
        return embeddings
    
    print("Computing SCIP embeddings (first run only)...")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("ERROR: Need datasets, transformers, torch. Install them first.")
        sys.exit(1)
    
    dataset = load_dataset('Nan-Do/code-search-net-python', split='train')
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starencoder")
    model = AutoModel.from_pretrained("bigcode/starencoder")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    all_embs = []
    batch_size = 32
    codes = [item.get("code", "") for item in dataset if item.get("code", "") and len(item.get("code", "")) > 50]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Embedding"):
            batch = codes[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embs.append(emb.cpu().numpy())
    
    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Computed and cached embeddings: {embeddings.shape}")
    return embeddings


def compute_centroids(embeddings, n_clusters=100):
    """Compute cluster centroids using MiniBatchKMeans."""
    from sklearn.cluster import MiniBatchKMeans
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096,
                            n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    # L2 normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    centroids = centroids / norms
    return centroids


# ── search functions ─────────────────────────────────────────────────
def search_exact(embeddings, centroids):
    """Exact centroid assignment. Returns (labels, search_time)."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(centroids)
    
    t0 = time.time()
    _, labels = index.search(embeddings, 1)
    search_time = time.time() - t0
    
    return labels.flatten(), search_time


def search_ann(embeddings, centroids, nlist, nprobe):
    """ANN centroid assignment with IVF. Returns (labels, search_time)."""
    K, d = centroids.shape
    actual_nlist = max(1, min(nlist, K // 2))  # nlist can't exceed num centroids / 2 for training
    
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, actual_nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(centroids)
    index.add(centroids)
    index.nprobe = min(nprobe, actual_nlist)
    
    t0 = time.time()
    _, labels = index.search(embeddings, 1)
    search_time = time.time() - t0
    
    return labels.flatten(), search_time


def compute_recall(exact_labels, ann_labels):
    """Recall = fraction of points assigned to same centroid."""
    return float(np.mean(exact_labels == ann_labels))


# ── main ─────────────────────────────────────────────────────────────
def run_single(embeddings, centroids, nprobe_values, nlist_values, run_id):
    print(f"\n{'='*60}\nRun {run_id}\n{'='*60}")
    
    exact_labels, exact_time = search_exact(embeddings, centroids)
    print(f"  Exact search: {exact_time:.4f}s")
    
    nprobe_results = []
    for nprobe in nprobe_values:
        ann_labels, t = search_ann(embeddings, centroids, ANCHOR_NLIST, nprobe)
        recall = compute_recall(exact_labels, ann_labels)
        nprobe_results.append({"nprobe": nprobe, "recall": recall, "search_time": t})
        print(f"  nprobe={nprobe:4d}  recall={recall:.6f}  time={t:.4f}s")
    
    nlist_results = []
    for nlist in nlist_values:
        ann_labels, t = search_ann(embeddings, centroids, nlist, ANCHOR_NPROBE)
        recall = compute_recall(exact_labels, ann_labels)
        nlist_results.append({"nlist": nlist, "recall": recall, "search_time": t})
        print(f"  nlist={nlist:4d}   recall={recall:.6f}  time={t:.4f}s")
    
    return nprobe_results, nlist_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results/ann_hyperparam_codepruning_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file) or "results", exist_ok=True)
    
    embeddings = load_codepruning_data(args.max_samples)
    centroids = compute_centroids(embeddings, args.n_clusters)
    
    K = len(centroids)
    default_nlist = max(1, int(np.sqrt(K)))
    default_nprobe = max(1, default_nlist // 4)
    
    n, d = embeddings.shape
    print(f"\nCodePruning (SCIP) ANN Hyperparameter Study")
    print(f"  Embeddings: {n}×{d}, Centroids: {K}")
    print(f"  Default: nlist={default_nlist}, nprobe={default_nprobe}")
    print(f"  nprobe sweep: {NPROBE_VALUES} (fixed nlist={ANCHOR_NLIST})")
    print(f"  nlist sweep:  {NLIST_VALUES} (fixed nprobe={ANCHOR_NPROBE})")
    
    all_nprobe = {p: [] for p in NPROBE_VALUES}
    all_nlist  = {nl: [] for nl in NLIST_VALUES}
    
    for run_id in range(1, args.runs + 1):
        np_res, nl_res = run_single(embeddings, centroids, NPROBE_VALUES, NLIST_VALUES, run_id)
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
        "experiment": "codepruning_ann_hyperparameter_study",
        "task": "CodePruning",
        "dataset": "CodeSearchNet-Python",
        "num_samples": int(n),
        "embedding_dim": int(d),
        "n_clusters": int(K),
        "num_runs": args.runs,
        "timestamp": timestamp,
        "anchor_nlist": ANCHOR_NLIST,
        "anchor_nprobe": ANCHOR_NPROBE,
        "default_nlist": int(default_nlist),
        "default_nprobe": int(default_nprobe),
        "nprobe_sweep": nprobe_agg,
        "nlist_sweep": nlist_agg,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
