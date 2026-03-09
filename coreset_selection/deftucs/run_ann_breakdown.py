import numpy as np
import time
from tqdm import tqdm
import json
import faiss
import math
from sklearn.metrics.pairwise import cosine_distances

from config import CACHE_DIR

def run_experiment(vectors, is_opt, K=7, A=3206, nlist_cap=256):
    """Run one pass of coreset selection (baseline or opt) and return detailed timing."""
    N, dim = vectors.shape
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    timing = {"kmeans": 0, "dist": 0, "sort": 0, "total": 0}
    
    # --- PHASE 1: KMEANS ---
    t_kmeans_start = time.time()
    
    selected_init = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected_init].copy()
    
    if is_opt:
        sqrtK = max(1, int(round(math.sqrt(K))))
        effective_nlist = min(nlist_cap, sqrtK)
        quantizer = faiss.IndexFlatL2(dim)
        index_ivf = faiss.IndexIVFFlat(quantizer, dim, effective_nlist)
        
        min_train = 39 * effective_nlist
        train_size = min(N, max(min_train, 10000))
        train_idx = rng.choice(N, size=train_size, replace=False)
        index_ivf.train(vectors[train_idx].astype(np.float32))
        index_ivf.nprobe = min(64, effective_nlist, max(1, int(0.5 * effective_nlist)))
        
    tol = 1e-4
    max_iter = 50
    labels = None
    l2_all = None
    
    for iter_idx in range(max_iter):
        if is_opt:
            index_ivf.reset()
            index_ivf.add(centroids.astype(np.float32))
            D, I = index_ivf.search(vectors.astype(np.float32), 1)
            labels = I.ravel().astype(np.int32)
            l2_all = D.ravel()
        else:
            dists = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1).astype(np.int32)
            l2_all = dists[np.arange(N), labels]
            
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= counts[non_empty][:, None]
            
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
            
        diff = np.linalg.norm(new_centroids - centroids)
        centroid_norm = np.linalg.norm(centroids)
        relative_change = diff / max(centroid_norm, 1e-8)
        
        if relative_change < tol:
            centroids = new_centroids
            if is_opt:
                index_ivf.reset()
                index_ivf.add(centroids.astype(np.float32))
                D, I = index_ivf.search(vectors.astype(np.float32), 1)
                labels = I.ravel().astype(np.int32)
                l2_all = D.ravel()
            else:
                dists = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1).astype(np.int32)
                l2_all = dists[np.arange(N), labels]
            break
        centroids = new_centroids
        
    timing["kmeans"] = time.time() - t_kmeans_start
    
    # --- PHASE 2: SELECTION ---
    Dc_indices = []
    
    centroid_norms_sq = np.sum(centroids ** 2, axis=1)
    vector_norms_sq = np.sum(vectors ** 2, axis=1)
    
    for cluster_id in range(K):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0: continue
        
        t_dist_start = time.time()
        if is_opt:
            l2_dist_sq = l2_all[cluster_indices]
            v_norms_sq = vector_norms_sq[cluster_indices]
            c_norm_sq = centroid_norms_sq[cluster_id]
            
            dot_products = 0.5 * (c_norm_sq + v_norms_sq - l2_dist_sq)
            v_norms = np.sqrt(v_norms_sq)
            c_norm = np.sqrt(c_norm_sq)
            
            denominator = v_norms * c_norm
            safe_denominator = np.maximum(denominator, 1e-12)
            cosine_sim = dot_products / safe_denominator
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
            cosine_dists = 1 - cosine_sim
        else:
            cluster_vectors = vectors[cluster_indices]
            centroid = centroids[cluster_id].reshape(1, -1)
            cosine_dists = cosine_distances(cluster_vectors, centroid).reshape(-1)
        timing["dist"] += time.time() - t_dist_start
        
        num_easy = min(int(0.5 * A), len(cosine_dists))
        num_hard = min(int(0.5 * A), len(cosine_dists) - num_easy)
        
        t_sort_start = time.time()
        if num_easy + num_hard > 0:
            if is_opt:
                if num_easy == 0:
                    easy_indices = np.empty(0, dtype=int)
                elif num_easy >= len(cosine_dists):
                    easy_indices = np.arange(len(cosine_dists))
                else:
                    easy_indices = np.argpartition(cosine_dists, num_easy - 1)[:num_easy]
                    
                if num_hard == 0:
                    hard_indices = np.empty(0, dtype=int)
                elif num_hard >= len(cosine_dists):
                    hard_indices = np.arange(len(cosine_dists))
                else:
                    hard_indices = np.argpartition(-cosine_dists, num_hard - 1)[:num_hard]
                
                selected = list(easy_indices) + list(hard_indices)
            else:
                sorted_indices = np.argsort(cosine_dists)
                selected = list(sorted_indices[:num_easy]) + list(sorted_indices[-num_hard:])
                
            Dc_indices.extend(cluster_indices[selected])
        timing["sort"] += time.time() - t_sort_start
        
    timing["total"] = timing["kmeans"] + timing["dist"] + timing["sort"]
    
    return timing, Dc_indices

def format_cell(data_list):
    mean = np.mean(data_list)
    std = np.std(data_list)
    return f"{mean:.2f}±{std:.2f}"

def format_cell_pct(recall_list):
    mean = np.mean(recall_list) * 100
    std = np.std(recall_list) * 100
    return f"{mean:.2f}%±{std:.2f}"
    
def main():
    print("Loading data...")
    embeddings_path = CACHE_DIR / "coedit_embeddings.npy"
    vectors = np.load(embeddings_path).astype(np.float32)
    
    N, dim = vectors.shape
    selection_ratio = 0.325
    target_n = int(N * selection_ratio)
    K = 7
    A = max(1, int(target_n / K))
    
    print(f"Loaded {N} samples. Target A={A}, K={K}")
    
    n_runs = 3
    baseline_stats = {"kmeans": [], "dist": [], "sort": [], "total": []}
    opt_stats = {"kmeans": [], "dist": [], "sort": [], "total": [], "recall": []}
    
    print(f"\nRunning Baseline {n_runs} times...")
    baseline_indices_master = None
    for i in range(n_runs):
        timing, indices = run_experiment(vectors, is_opt=False, K=K, A=A)
        for k, v in timing.items(): baseline_stats[k].append(v)
        if i == 0: baseline_indices_master = set(indices)
        
    print(f"Running IV-Aligned (All) {n_runs} times...")
    for i in range(n_runs):
        timing, indices = run_experiment(vectors, is_opt=True, K=K, A=A)
        for k, v in timing.items(): opt_stats[k].append(v)
        recall = len(set(indices) & baseline_indices_master) / len(baseline_indices_master)
        opt_stats["recall"].append(recall)
        
    # Format and print the table mimicking the user screenshot
    print("\n" + "="*120)
    print("TABLE 5: TIMING BREAKDOWN (s) - CORESET SELECTION")
    print("="*120)
    
    headers = ["Task Phase", "-", "B1/IV1", "-", "B2/IV2", "B3/IV3"]
    header_row = f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]:<15}"
    
    headers2 = ["", "Baseline Exec.", "IV1-Aligned", "Recall", "Baseline Exec.", "IV2-Aligned", "Baseline Exec.", "IV3-Aligned"]
    header_row2 = f"{headers2[0]:<20} {headers2[1]:<16} {headers2[2]:<16} {headers2[3]:<16} {headers2[4]:<16} {headers2[5]:<16} {headers2[6]:<16} {headers2[7]:<16}"
    
    print(header_row2)
    print("-" * 130)
    
    # Values
    b_kmeans = format_cell(baseline_stats['kmeans'])
    o_kmeans = format_cell(opt_stats['kmeans'])
    recall_str = format_cell_pct(opt_stats['recall'])
    
    b_dist = format_cell(baseline_stats['dist'])
    o_dist = format_cell(opt_stats['dist'])
    
    b_sort = format_cell(baseline_stats['sort'])
    o_sort = format_cell(opt_stats['sort'])
    
    row = f"{'Coreset Selection':<20} {b_kmeans:<16} {o_kmeans:<16} {recall_str:<16} {b_dist:<16} {o_dist:<16} {b_sort:<16} {o_sort:<16}"
    print(row)
    print("="*130)

if __name__ == '__main__':
    main()
