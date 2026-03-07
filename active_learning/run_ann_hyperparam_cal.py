#!/usr/bin/env python
"""
CAL (Contrastive Active Learning) ANN Hyperparameter Study

Measures recall and search time for FAISS IVF vs Flat on KNN search
matching the real CAL pipeline: index on labeled pool, query all data.

Uses real labeled pool indices from a previous baseline_exact CAL run.

Recall = task-level (acquisition batch overlap between exact and ANN).

Usage:
    conda run -n rag python run_ann_hyperparam_cal.py --runs 3
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

KNN_K = 10  # number of neighbors (matches CAL's default num_nei)

# Path to labeled pool indices from baseline_exact run (seed 42)
LABELED_IDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments/al_sst-2_bert_cal_baseline_exact/42_baseline_exact_cls/selected_ids_per_iteration.json"
)

# ── helpers ──────────────────────────────────────────────────────────
def load_sst2_embeddings():
    """Load or compute BERT CLS embeddings for SST-2 training data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "cache_sst2_bert_embeddings.npz")
    
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"Loaded cached SST-2 embeddings: {data['embeddings'].shape}")
        return data['embeddings'].astype(np.float32)
    
    print("Computing BERT embeddings for SST-2 (first run only)...")
    try:
        from datasets import load_dataset
        from transformers import BertTokenizer, BertModel
        import torch
    except ImportError:
        print("ERROR: Need datasets, transformers, torch. Install them first.")
        sys.exit(1)
    
    dataset = load_dataset("glue", "sst2", split="train")
    
    local_model_path = os.path.join(script_dir, "cache", "bert-base-cased-local")
    if os.path.exists(local_model_path):
        print(f"Using local BERT model: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
        model = BertModel.from_pretrained(local_model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertModel.from_pretrained("bert-base-cased")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    all_embs = []
    batch_size = 64
    
    sentences = dataset["sentence"] if hasattr(dataset, "__getitem__") else [ex["sentence"] for ex in dataset]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding SST-2"):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_emb = outputs[0][:, 0, :]
            all_embs.append(cls_emb.cpu().numpy())
    
    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Computed and cached SST-2 embeddings: {embeddings.shape}")
    return embeddings


def load_labeled_indices():
    """Load real labeled pool indices from baseline_exact run."""
    with open(LABELED_IDS_PATH, 'r') as f:
        ids_per_it = json.load(f)
    
    all_labeled = []
    for key in sorted(ids_per_it.keys(), key=int):
        all_labeled += ids_per_it[key]
    
    print(f"Loaded {len(all_labeled)} labeled pool indices from baseline_exact (seed 42)")
    return all_labeled


# ── Task-level recall matching SemDeDup approach ─────────────────────
# CAL proxy: mean distance to K neighbors as acquisition score.
# Select top SELECTION_PCT% by score, compare exact vs ANN selected sets.
SELECTION_PCT = 0.10  # top 10% = acquisition batch proxy

def search_exact(labeled_embs, query_embs, k):
    """Exact KNN: index on labeled pool, query all data. Returns (D, I, search_time)."""
    d = labeled_embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(labeled_embs)
    
    t0 = time.time()
    D, I = index.search(query_embs, k)
    search_time = time.time() - t0
    
    return D, I, search_time


def search_ann(labeled_embs, query_embs, k, nlist, nprobe):
    """ANN KNN: index on labeled pool, query all data using IVF. Returns (D, I, search_time)."""
    n_labeled = labeled_embs.shape[0]
    d = labeled_embs.shape[1]
    actual_nlist = max(1, min(nlist, n_labeled // 10))
    
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, actual_nlist, faiss.METRIC_L2)
    index.train(labeled_embs)
    index.add(labeled_embs)
    index.nprobe = min(nprobe, actual_nlist)
    
    t0 = time.time()
    D, I = index.search(query_embs, k)
    search_time = time.time() - t0
    
    return D, I, search_time


def compute_recall(exact_D, ann_D, k, selection_pct=SELECTION_PCT):
    """Task-level recall: overlap of selected acquisition batch.
    
    Uses mean distance to K neighbors as acquisition score proxy.
    Selects top-B% points, compares exact vs ANN selections.
    Recall = |exact_selected ∩ ann_selected| / |exact_selected|
    Consistent with SemDeDup's overlap-based task recall.
    """
    n = len(exact_D)
    B = max(1, int(n * selection_pct))
    
    # Mean distance to K neighbors (no self-exclusion: queries != index)
    exact_scores = np.mean(exact_D[:, :k], axis=1)
    ann_scores   = np.mean(ann_D[:, :k], axis=1)
    
    # Select top-B points by score (higher distance = more uncertain = selected)
    exact_selected = set(np.argsort(exact_scores)[-B:])
    ann_selected   = set(np.argsort(ann_scores)[-B:])
    
    recall = len(exact_selected & ann_selected) / len(exact_selected)
    return float(recall)


# ── main ─────────────────────────────────────────────────────────────
def run_single(labeled_embs, query_embs, k, nprobe_values, nlist_values, run_id):
    print(f"\n{'='*60}\nRun {run_id}\n{'='*60}")
    
    exact_D, exact_I, exact_search_time = search_exact(labeled_embs, query_embs, k)
    print(f"  Exact search: {exact_search_time:.4f}s")
    
    nprobe_results = []
    for nprobe in nprobe_values:
        ann_D, ann_I, ann_search_time = search_ann(labeled_embs, query_embs, k, ANCHOR_NLIST, nprobe)
        recall = compute_recall(exact_D, ann_D, k)
        nprobe_results.append({"nprobe": nprobe, "recall": recall, "search_time": ann_search_time})
        print(f"  nprobe={nprobe:4d}  recall={recall:.6f}  search_time={ann_search_time:.4f}s")
    
    nlist_results = []
    for nlist in nlist_values:
        ann_D, ann_I, ann_search_time = search_ann(labeled_embs, query_embs, k, nlist, ANCHOR_NPROBE)
        recall = compute_recall(exact_D, ann_D, k)
        nlist_results.append({"nlist": nlist, "recall": recall, "search_time": ann_search_time})
        print(f"  nlist={nlist:4d}   recall={recall:.6f}  search_time={ann_search_time:.4f}s")
    
    return nprobe_results, nlist_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results/ann_hyperparam_cal_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file) or "results", exist_ok=True)
    
    all_embeddings = load_sst2_embeddings()
    labeled_inds = load_labeled_indices()
    
    labeled_embs = all_embeddings[labeled_inds]
    query_embs = all_embeddings  # query ALL data against labeled pool
    
    n_labeled = labeled_embs.shape[0]
    n_query = query_embs.shape[0]
    d = all_embeddings.shape[1]
    
    print(f"\nCAL ANN Hyperparameter Study (aligned with pipeline)")
    print(f"  Labeled pool (index): {n_labeled} × {d}")
    print(f"  Query (all data):     {n_query} × {d}")
    print(f"  K={KNN_K}, runs={args.runs}")
    print(f"  nprobe sweep: {NPROBE_VALUES} (fixed nlist={ANCHOR_NLIST})")
    print(f"  nlist sweep:  {NLIST_VALUES} (fixed nprobe={ANCHOR_NPROBE})")
    
    all_nprobe = {p: [] for p in NPROBE_VALUES}
    all_nlist  = {n: [] for n in NLIST_VALUES}
    
    for run_id in range(1, args.runs + 1):
        nprobe_res, nlist_res = run_single(labeled_embs, query_embs, KNN_K, NPROBE_VALUES, NLIST_VALUES, run_id)
        for r in nprobe_res:
            all_nprobe[r["nprobe"]].append(r)
        for r in nlist_res:
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
        "experiment": "cal_ann_hyperparameter_study",
        "task": "CAL",
        "dataset": "SST-2",
        "search_setup": "labeled_pool_to_all (aligned with pipeline)",
        "num_labeled": int(n_labeled),
        "num_queries": int(n_query),
        "embedding_dim": int(d),
        "knn_k": KNN_K,
        "num_runs": args.runs,
        "timestamp": timestamp,
        "anchor_nlist": ANCHOR_NLIST,
        "anchor_nprobe": ANCHOR_NPROBE,
        "default_nlist": 100,
        "default_nprobe": 10,
        "nprobe_sweep": nprobe_agg,
        "nlist_sweep": nlist_agg,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
