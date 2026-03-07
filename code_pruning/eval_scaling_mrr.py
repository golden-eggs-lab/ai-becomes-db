"""
SCIP Scaling - MRR Evaluation

Matches run_aligned_experiment.py approach:
- Load Nan-Do/code-search-net-python (Parquet, works with modern datasets)
- Run SCIP baseline/optimized for each ratio to get kept indices
- Finetune CodeBERT+LoRA on pruned data
- Evaluate MRR on held-out 1000 samples

Usage:
    python eval_scaling_mrr.py --ratios 1.0 0.7 0.5 0.3
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from tqdm import tqdm
from dataclasses import dataclass

import faiss
from sklearn.cluster import MiniBatchKMeans


@dataclass
class Config:
    n_clusters: int = 1000
    p: float = 0.2
    alpha: float = 0.8
    train_steps: int = 300
    train_batch_size: int = 16


def load_data_with_docstrings(language="python", max_samples=None):
    """Load code and docstring pairs from Nan-Do/code-search-net-{language}."""
    from datasets import load_dataset
    
    ds_name = f'Nan-Do/code-search-net-{language}'
    print(f"Loading {ds_name}...")
    dataset = load_dataset(ds_name, split='train')
    print(f"Total available: {len(dataset)} samples")
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    codes = []
    docstrings = []
    for item in tqdm(dataset, desc="Processing"):
        code = item.get("code", "")
        docstring = item.get("docstring", "")
        if code and docstring and len(code) > 50 and len(docstring) > 10:
            codes.append(code)
            docstrings.append(docstring)
    
    print(f"Loaded {len(codes)} valid code-docstring pairs")
    return codes, docstrings


def scip_baseline(embeddings, cfg):
    """Baseline SCIP: Flat search + loop distances + argsort."""
    N, D = embeddings.shape
    K = min(cfg.n_clusters, N // 2)
    
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    
    # Flat search
    index = faiss.IndexFlatIP(D)
    index.add(centroids.astype(np.float32))
    sims, labels = index.search(embeddings, k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    
    # Loop distances
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    
    # Argsort selection
    size_per_point = cluster_sizes[labels]
    idx_by_size = np.argsort(size_per_point)
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    prune_by_size = idx_by_size[:size_quota]
    
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    dist_remaining = distances[remaining]
    order = np.argsort(-dist_remaining)
    dist_quota = total_prune - size_quota
    prune_by_dist = remaining[order[:dist_quota]]
    
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    return np.nonzero(keep_mask)[0]


def scip_optimized(embeddings, cfg):
    """Optimized SCIP: IVF search + vectorized distances + TopK."""
    N, D = embeddings.shape
    K = min(cfg.n_clusters, N // 2)
    
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    
    # IVF search
    centroids_f32 = centroids.astype(np.float32)
    if K >= 16:
        nlist = max(1, int(np.sqrt(K)))
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(centroids_f32)
        index.add(centroids_f32)
        index.nprobe = max(1, nlist // 2)
    else:
        index = faiss.IndexFlatIP(D)
        index.add(centroids_f32)
    
    sims, labels = index.search(embeddings, k=1)
    labels = labels.flatten()
    sims = sims.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    
    # Vectorized distances (reuse sims)
    distances = 1.0 - sims
    size_per_point = cluster_sizes[labels]
    
    # TopK selection
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    if size_quota > 0:
        idx_by_size = np.argpartition(size_per_point, size_quota)
    else:
        idx_by_size = np.argsort(size_per_point)
    prune_by_size = idx_by_size[:size_quota]
    
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    dist_remaining = distances[remaining]
    dist_quota = total_prune - size_quota
    
    if dist_quota > 0 and len(remaining) > 0:
        part_idx = np.argpartition(-dist_remaining, dist_quota)
        prune_by_dist = remaining[part_idx[:dist_quota]]
    else:
        prune_by_dist = np.array([], dtype=np.int64)
    
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    return np.nonzero(keep_mask)[0]


def compute_mrr(query_embs, code_embs):
    """Compute MRR."""
    query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12)
    code_embs = code_embs / (np.linalg.norm(code_embs, axis=1, keepdims=True) + 1e-12)
    sim = query_embs @ code_embs.T
    N = len(query_embs)
    ranks = []
    for i in range(N):
        rank = (sim[i] > sim[i, i]).sum() + 1
        ranks.append(rank)
    return np.mean(1.0 / np.array(ranks))


def train_and_eval_mrr(codes, docstrings, keep_indices, name, cfg):
    """Train CodeBERT+LoRA on pruned data and eval MRR. Matches run_aligned_experiment.py."""
    from transformers import AutoTokenizer, AutoModel
    from peft import get_peft_model, LoraConfig, TaskType
    
    pruned_codes = [codes[i] for i in keep_indices if i < len(codes)]
    pruned_docs = [docstrings[i] for i in keep_indices if i < len(docstrings)]
    
    print(f"\n  [{name}] Training on {len(pruned_codes)} samples, {cfg.train_steps} steps")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    n_samples = len(pruned_codes)
    batch_size = cfg.train_batch_size
    
    for step in tqdm(range(cfg.train_steps), desc=f"  Training {name}"):
        indices = np.random.choice(n_samples, batch_size, replace=False)
        batch_codes = [pruned_codes[i] for i in indices]
        batch_docs = [pruned_docs[i] for i in indices]
        
        code_inputs = tokenizer(batch_codes, padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(device)
        doc_inputs = tokenizer(batch_docs, padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(device)
        
        code_embs = model(**code_inputs).last_hidden_state[:, 0, :]
        doc_embs = model(**doc_inputs).last_hidden_state[:, 0, :]
        
        code_embs = F.normalize(code_embs, dim=1)
        doc_embs = F.normalize(doc_embs, dim=1)
        
        sim = doc_embs @ code_embs.T
        labels = torch.arange(batch_size).to(device)
        loss = F.cross_entropy(sim * 20.0, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Eval on held-out (last 1000)
    print(f"  [{name}] Evaluating...")
    model.eval()
    
    eval_codes = codes[-1000:]
    eval_docs = docstrings[-1000:]
    
    all_code_embs = []
    all_doc_embs = []
    
    with torch.no_grad():
        for i in range(0, 1000, 32):
            batch_codes = eval_codes[i:i+32]
            batch_docs = eval_docs[i:i+32]
            
            code_inputs = tokenizer(batch_codes, padding=True, truncation=True,
                                    max_length=256, return_tensors="pt").to(device)
            doc_inputs = tokenizer(batch_docs, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(device)
            
            code_embs = model(**code_inputs).last_hidden_state[:, 0, :]
            doc_embs = model(**doc_inputs).last_hidden_state[:, 0, :]
            
            all_code_embs.append(code_embs.cpu().numpy())
            all_doc_embs.append(doc_embs.cpu().numpy())
    
    code_embs_np = np.concatenate(all_code_embs, axis=0)
    doc_embs_np = np.concatenate(all_doc_embs, axis=0)
    
    mrr = compute_mrr(doc_embs_np, code_embs_np)
    print(f"  [{name}] MRR: {mrr:.4f}")
    
    del model
    torch.cuda.empty_cache()
    
    return mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, 
                        default="./experiments/aligned/embeddings.npy")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument("--n_clusters", type=int, default=1000)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--output", type=str, default="scip_scaling_mrr.json")
    parser.add_argument("--language", type=str, default="python", choices=["python", "java"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    cfg = Config(n_clusters=args.n_clusters, train_steps=args.train_steps)
    
    print(f"GPUs available: {torch.cuda.device_count()}")
    
    # Load embeddings (453k from aligned experiment)
    embeddings = np.load(args.embeddings).astype(np.float32)
    N_full = len(embeddings)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Load code-docstring pairs
    codes, docstrings = load_data_with_docstrings(language=args.language)
    N_data = len(codes)
    print(f"Data samples: {N_data}, Embedding samples: {N_full}")
    
    # Use min(N_full, N_data) to align
    N = min(N_full, N_data)
    embeddings = embeddings[:N]
    codes = codes[:N]
    docstrings = docstrings[:N]
    print(f"Aligned to {N} samples")
    
    results = {}
    
    for ratio in args.ratios:
        print(f"\n{'='*60}")
        print(f"RATIO={ratio}")
        print(f"{'='*60}")
        
        n_sub = int(N * ratio)
        sub_idx = np.arange(n_sub)  # deterministic subset (first n_sub)
        sub_emb = embeddings[:n_sub]
        
        print(f"  Subset: {n_sub} samples")
        
        # Run SCIP
        print("  Running SCIP baseline...")
        base_keep = scip_baseline(sub_emb, cfg)
        
        print("  Running SCIP optimized...")
        opt_keep = scip_optimized(sub_emb, cfg)
        
        recall = len(set(base_keep.tolist()) & set(opt_keep.tolist())) / len(base_keep)
        print(f"  Baseline kept: {len(base_keep)}, Opt kept: {len(opt_keep)}, Recall: {recall:.1%}")
        
        # MRR eval
        base_mrr = train_and_eval_mrr(codes, docstrings, base_keep, 
                                       f"baseline_r{ratio}", cfg)
        opt_mrr = train_and_eval_mrr(codes, docstrings, opt_keep,
                                      f"optimized_r{ratio}", cfg)
        
        results[str(ratio)] = {
            "ratio": ratio,
            "n_samples": n_sub,
            "baseline_mrr": base_mrr,
            "optimized_mrr": opt_mrr,
            "baseline_kept": int(len(base_keep)),
            "optimized_kept": int(len(opt_keep)),
            "recall": recall,
        }
        
        print(f"\n  MRR: baseline={base_mrr:.4f}, optimized={opt_mrr:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SCIP SCALING MRR SUMMARY")
    print(f"{'='*60}")
    for r, v in results.items():
        print(f"  ratio={r}: B_MRR={v['baseline_mrr']:.4f}, O_MRR={v['optimized_mrr']:.4f}")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
