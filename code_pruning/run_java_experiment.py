"""
SCIP Aligned End-to-End Experiment — CodeSearchNet Java

Same pipeline as run_aligned_experiment.py but for Java:
1. Load data (code + Javadoc) from CodeSearchNet Java
2. Compute StarEncoder embeddings (multi-GPU)
3. Run SCIP Baseline vs Optimized
4. Finetune CodeBERT (LoRA) on pruned data
5. Evaluate with MRR

Usage:
    python run_java_experiment.py --max_samples 0  # 0 = all samples
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
from dataclasses import dataclass, field
import faiss

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Data
    max_samples: int = 0  # 0 = use all samples
    
    # Embedding
    embed_model: str = "bigcode/starencoder"
    embed_dim: int = 768
    embed_batch_size: int = 64  # per GPU
    
    # SCIP
    n_clusters: int = 100
    p: float = 0.2
    alpha: float = 0.8
    
    # Training
    train_steps: int = 300
    train_batch_size: int = 16
    
    # Output
    output_dir: str = "./experiments/aligned_java"
    
    # Multi-GPU
    n_gpus: int = 8


@dataclass
class TimingResult:
    step_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    def add_step(self, name: str, elapsed: float):
        self.step_times[name] = elapsed
        self.total_time += elapsed


# ============================================================================
# Data Loading
# ============================================================================

def load_data_with_docstrings(max_samples: int) -> Tuple[List[str], List[str]]:
    """Load code and docstring pairs from CodeSearchNet Java."""
    print(f"\n{'='*60}")
    print("Phase 1: Loading CodeSearchNet Java")
    print(f"{'='*60}")
    
    from datasets import load_dataset
    
    dataset = load_dataset('Nan-Do/code-search-net-java', split='train')
    
    print(f"Total available: {len(dataset)} samples")
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    codes = []
    docstrings = []
    
    for item in tqdm(dataset, desc="Processing"):
        code = item.get("code", "")
        docstring = item.get("docstring", "")
        
        # Filter valid pairs
        if code and docstring and len(code) > 50 and len(docstring) > 10:
            codes.append(code)
            docstrings.append(docstring)
    
    print(f"Loaded {len(codes)} valid code-docstring pairs")
    return codes, docstrings


# ============================================================================
# Multi-GPU Embedding
# ============================================================================

def compute_embeddings_multi_gpu(texts: List[str], cfg: Config) -> np.ndarray:
    """Compute StarEncoder embeddings using multiple GPUs."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Computing embeddings (multi-GPU, {cfg.n_gpus} GPUs)")
    print(f"{'='*60}")
    
    from transformers import AutoTokenizer, AutoModel
    import copy
    
    n_available = torch.cuda.device_count()
    n_gpus = min(cfg.n_gpus, n_available)
    print(f"Using {n_gpus} GPUs (available: {n_available})")
    
    if n_gpus <= 1:
        return compute_embeddings_single(texts, cfg, device="cuda:0")
    
    # Pre-load model on CPU once (avoids meta tensor issues)
    print("Loading StarEncoder model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModel.from_pretrained(cfg.embed_model)
    base_model.eval()
    print("Model loaded. Copying to GPUs...")
    
    # Split texts across GPUs
    chunk_size = (len(texts) + n_gpus - 1) // n_gpus
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    all_embeddings = [None] * len(chunks)
    
    def process_chunk(gpu_id, chunk_texts, result_list, idx, model_copy):
        device = f"cuda:{gpu_id}"
        model_copy = model_copy.to(device)
        model_copy.eval()
        
        n_batches = (len(chunk_texts) + cfg.embed_batch_size - 1) // cfg.embed_batch_size
        chunk_embs = []
        with torch.no_grad():
            for i in range(0, len(chunk_texts), cfg.embed_batch_size):
                batch = chunk_texts[i:i + cfg.embed_batch_size]
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(device)
                
                outputs = model_copy(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs.attention_mask.unsqueeze(-1)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                chunk_embs.append(emb.cpu().numpy())
                
                batch_idx = i // cfg.embed_batch_size + 1
                if batch_idx % 50 == 0 or batch_idx == n_batches:
                    print(f"  GPU {gpu_id}: {batch_idx}/{n_batches} batches", flush=True)
        
        result_list[idx] = np.concatenate(chunk_embs, axis=0)
        del model_copy
        torch.cuda.empty_cache()
    
    # Deep-copy model for each GPU, then dispatch threads
    import threading
    threads = []
    
    gpu_models = []
    for gpu_id in range(len(chunks)):
        gpu_models.append(copy.deepcopy(base_model))
    del base_model  # free CPU memory
    
    print(f"Dispatching {len(chunks)} chunks to {n_gpus} GPUs...")
    for gpu_id in range(len(chunks)):
        print(f"  GPU {gpu_id}: {len(chunks[gpu_id])} texts")
        t = threading.Thread(
            target=process_chunk, 
            args=(gpu_id, chunks[gpu_id], all_embeddings, gpu_id, gpu_models[gpu_id])
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def compute_embeddings_single(texts: List[str], cfg: Config, device: str = "cuda:0") -> np.ndarray:
    """Compute StarEncoder embeddings on single GPU."""
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model)
    model = AutoModel.from_pretrained(cfg.embed_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    all_embs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), cfg.embed_batch_size), desc="Embedding"):
            batch = texts[i:i + cfg.embed_batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embs.append(emb.cpu().numpy())
    
    embeddings = np.concatenate(all_embs, axis=0)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ============================================================================
# SCIP Algorithm
# ============================================================================

def scip_baseline(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult]:
    """Baseline SCIP: Faiss Flat + No Reuse + Full Sort"""
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means
    t0 = time.perf_counter()
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss Exact search
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(D)
    index.add(centroids.astype(np.float32))
    sims, labels = index.search(embeddings, k=1)
    labels = labels.flatten()
    cluster_sizes = np.bincount(labels, minlength=K)
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances (no reuse)
    t0 = time.perf_counter()
    distances = np.zeros(N, dtype=np.float32)
    for i in range(N):
        distances[i] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # Sort by size (full)
    t0 = time.perf_counter()
    size_per_point = cluster_sizes[labels]
    idx_by_size = np.argsort(size_per_point)
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("SortBySize", time.perf_counter() - t0)
    
    # Sort remaining by distance (no reuse, full)
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    
    dist_remaining = np.zeros(len(remaining), dtype=np.float32)
    for idx, i in enumerate(remaining):
        dist_remaining[idx] = 1.0 - np.dot(embeddings[i], centroids[labels[i]])
    
    order = np.argsort(-dist_remaining)
    dist_quota = total_prune - size_quota
    prune_by_dist = remaining[order[:dist_quota]]
    timing.add_step("SortByDist", time.perf_counter() - t0)
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing


def scip_optimized(embeddings: np.ndarray, cfg: Config) -> Tuple[np.ndarray, TimingResult]:
    """Optimized SCIP: Faiss IVF + Reuse + TopK"""
    timing = TimingResult()
    N, D = embeddings.shape
    K = cfg.n_clusters
    
    # K-means (same)
    t0 = time.perf_counter()
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init='auto', random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    timing.add_step("KMeans", time.perf_counter() - t0)
    
    # Faiss ANN search
    t0 = time.perf_counter()
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
    timing.add_step("CentroidSearch", time.perf_counter() - t0)
    
    # Distances (REUSE)
    t0 = time.perf_counter()
    distances = 1.0 - sims
    size_per_point = cluster_sizes[labels]
    timing.add_step("Distances", time.perf_counter() - t0)
    
    # TopK by size
    t0 = time.perf_counter()
    total_prune = int(round(cfg.p * N))
    size_quota = int(round(cfg.alpha * total_prune))
    if size_quota > 0:
        idx_by_size = np.argpartition(size_per_point, size_quota)
    else:
        idx_by_size = np.argsort(size_per_point)
    prune_by_size = idx_by_size[:size_quota]
    timing.add_step("TopKBySize", time.perf_counter() - t0)
    
    # TopK by distance (REUSE)
    t0 = time.perf_counter()
    mask = np.ones(N, dtype=bool)
    mask[prune_by_size] = False
    remaining = np.nonzero(mask)[0]
    
    dist_remaining = distances[remaining]  # REUSE!
    dist_quota = total_prune - size_quota
    
    if dist_quota > 0 and len(remaining) > 0:
        part_idx = np.argpartition(-dist_remaining, dist_quota)
        prune_by_dist = remaining[part_idx[:dist_quota]]
    else:
        prune_by_dist = np.array([], dtype=np.int64)
    timing.add_step("TopKByDist", time.perf_counter() - t0)
    
    # Combine
    t0 = time.perf_counter()
    prune_all = np.unique(np.concatenate([prune_by_size, prune_by_dist]))
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[prune_all] = False
    keep_indices = np.nonzero(keep_mask)[0]
    timing.add_step("Combine", time.perf_counter() - t0)
    
    return keep_indices, timing


def run_scip_comparison(embeddings: np.ndarray, cfg: Config, n_runs: int = 3):
    """Run SCIP baseline vs optimized comparison."""
    print(f"\n{'='*60}")
    print(f"Phase 3: SCIP Algorithm Comparison ({n_runs} runs)")
    print(f"{'='*60}")
    
    baseline_times = []
    optimized_times = []
    keep_baseline = None
    keep_optimized = None
    
    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")
        
        keep_b, timing_b = scip_baseline(embeddings, cfg)
        baseline_times.append(timing_b.total_time)
        print(f"    Baseline:  {timing_b.total_time:.4f}s")
        for step, t in timing_b.step_times.items():
            print(f"      {step}: {t:.4f}s")
        
        keep_o, timing_o = scip_optimized(embeddings, cfg)
        optimized_times.append(timing_o.total_time)
        print(f"    Optimized: {timing_o.total_time:.4f}s")
        for step, t in timing_o.step_times.items():
            print(f"      {step}: {t:.4f}s")
        
        if run == 0:
            keep_baseline = keep_b
            keep_optimized = keep_o
    
    baseline_avg = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    optimized_avg = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    speedup = baseline_avg / optimized_avg
    
    overlap = len(set(keep_baseline) & set(keep_optimized)) / len(keep_baseline)
    
    print(f"\n  Summary:")
    print(f"  Baseline:  {baseline_avg:.4f}s ± {baseline_std:.4f}s")
    print(f"  Optimized: {optimized_avg:.4f}s ± {optimized_std:.4f}s")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"  Overlap:   {overlap*100:.1f}%")
    
    return keep_baseline, keep_optimized, {
        "baseline_time": baseline_avg,
        "baseline_std": baseline_std,
        "optimized_time": optimized_avg,
        "optimized_std": optimized_std,
        "speedup": speedup,
        "overlap": overlap,
        "baseline_runs": baseline_times,
        "optimized_runs": optimized_times,
    }


# ============================================================================
# Code Search Training & Evaluation
# ============================================================================

def compute_mrr(query_embs: np.ndarray, code_embs: np.ndarray) -> float:
    """Compute MRR."""
    query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12)
    code_embs = code_embs / (np.linalg.norm(code_embs, axis=1, keepdims=True) + 1e-12)
    
    sims = query_embs @ code_embs.T  # [n_query, n_code]
    
    mrr = 0.0
    for i in range(len(query_embs)):
        ranked = np.argsort(-sims[i])
        rank = np.where(ranked == i)[0][0] + 1
        mrr += 1.0 / rank
    
    return mrr / len(query_embs)


def train_and_eval_codesearch(
    codes: List[str],
    docstrings: List[str],
    keep_indices: np.ndarray,
    version: str,
    cfg: Config,
) -> Dict[str, float]:
    """Train CodeBERT on pruned data and evaluate with MRR."""
    from transformers import AutoTokenizer, AutoModel
    from peft import get_peft_model, LoraConfig, TaskType
    
    # Get pruned data
    pruned_codes = [codes[i] for i in keep_indices]
    pruned_docs = [docstrings[i] for i in keep_indices]
    
    print(f"\n  Training on {len(pruned_codes)} samples...")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    
    # LoRA
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
    
    for step in tqdm(range(cfg.train_steps), desc=f"  Training {version}"):
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
    
    # Evaluate on held-out samples (last 1000)
    print(f"  Evaluating {version}...")
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
    
    code_embs = np.concatenate(all_code_embs, axis=0)
    doc_embs = np.concatenate(all_doc_embs, axis=0)
    
    mrr = compute_mrr(doc_embs, code_embs)
    print(f"  {version} MRR: {mrr:.4f}")
    
    del model
    torch.cuda.empty_cache()
    
    return {"mrr": mrr}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="./experiments/aligned_java")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip embedding computation; load from cache")
    args = parser.parse_args()
    
    cfg = Config(
        max_samples=args.max_samples,
        train_steps=args.train_steps,
        output_dir=args.output_dir,
        n_gpus=args.n_gpus,
    )
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Phase 1: Load data
    codes, docstrings = load_data_with_docstrings(cfg.max_samples)
    
    # Phase 2: Compute or load embeddings
    embeddings_path = os.path.join(cfg.output_dir, "embeddings.npy")
    
    if args.skip_embeddings and os.path.exists(embeddings_path):
        print(f"\nLoading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        t_embed_start = time.time()
        embeddings = compute_embeddings_multi_gpu(codes, cfg)
        t_embed = time.time() - t_embed_start
        print(f"Embedding time: {t_embed:.1f}s ({t_embed/60:.1f}min)")
        
        # Save embeddings for reuse
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
    
    # Phase 3: SCIP comparison
    keep_baseline, keep_optimized, algo_results = run_scip_comparison(embeddings, cfg)
    
    # Phase 4: Downstream evaluation
    print(f"\n{'='*60}")
    print("Phase 4: Downstream Evaluation (Code Search)")
    print(f"{'='*60}")
    
    baseline_results = train_and_eval_codesearch(
        codes, docstrings, keep_baseline, "baseline", cfg
    )
    
    optimized_results = train_and_eval_codesearch(
        codes, docstrings, keep_optimized, "optimized", cfg
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\nDataset: CodeSearchNet Java ({len(codes)} samples)")
    print(f"\nAlgorithm:")
    print(f"  Baseline:  {algo_results['baseline_time']:.4f}s ± {algo_results['baseline_std']:.4f}s")
    print(f"  Optimized: {algo_results['optimized_time']:.4f}s ± {algo_results['optimized_std']:.4f}s")
    print(f"  Speedup:   {algo_results['speedup']:.2f}x")
    print(f"  Overlap:   {algo_results['overlap']*100:.1f}%")
    print(f"\nCode Search (MRR):")
    print(f"  Baseline:  {baseline_results['mrr']:.4f}")
    print(f"  Optimized: {optimized_results['mrr']:.4f}")
    
    mrr_diff = abs(baseline_results['mrr'] - optimized_results['mrr'])
    print(f"  Diff:      {mrr_diff:.4f}")
    
    # Save results
    results = {
        "config": {
            "dataset": "CodeSearchNet Java",
            "n_samples": len(codes),
            "n_clusters": cfg.n_clusters,
            "p": cfg.p,
            "alpha": cfg.alpha,
        },
        "algorithm": algo_results,
        "downstream": {
            "baseline": baseline_results,
            "optimized": optimized_results,
        }
    }
    
    with open(os.path.join(cfg.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {cfg.output_dir}/results.json")


if __name__ == "__main__":
    main()
