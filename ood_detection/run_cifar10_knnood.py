"""
KNN-OOD with CIFAR-10 as In-Distribution Dataset (FAISS GPU)
Compares Exact KNN GPU vs ANN-IVF GPU search performance.

Usage:
  # Step 1: Extract features (only needed once)
  python feat_extract.py --in-dataset CIFAR-10 --out-datasets inat sun50 places50 dtd \
      --name resnet18-supcon --model-arch resnet18-supcon

  # Step 2: Run experiment
  python run_cifar10_knnood.py --seed 1
  python run_cifar10_knnood.py --seed 1 --use-ann --ann-method ivf --nlist 20 --nprobe 5
"""

import os
import time
from util.args_loader import get_args
from util import metrics
import faiss
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. GPU features disabled.")

args = get_args()

seed = args.seed
print(f"Seed: {seed}")
if TORCH_AVAILABLE:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ============================================================================
# CIFAR-10 Parameters
# ============================================================================
class_num = 10
id_train_size = 50000
id_val_size = 10000
K = 50  # Standard KNN-OOD value for CIFAR-10

ood_dataset_size = {
    'inat': 10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640,
}

# ============================================================================
# Load Features (from .npy alllayers format)
# ============================================================================
print(f"\n{'='*60}")
print(f"Loading CIFAR-10 features (resnet18-supcon)")
print(f"{'='*60}")

# ID train features
cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.npy"
print(f"Loading ID train: {cache_name}")
feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
print(f"  Shape: feat={feat_log.shape}, score={score_log.shape}")

# ID val features
cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.npy"
print(f"Loading ID val: {cache_name}")
feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)
print(f"  Shape: feat={feat_log_val.shape}, score={score_log_val.shape}")

# Preprocessing: normalize + last layer only
# ResNet18 feature dims: [64, 128, 256, 512] = 960 total
# Last layer (512-dim): indices [448, 960)
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448, 960)]))  # Last Layer only

ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
print(f"Preprocessed: ftrain={ftrain.shape}, ftest={ftest.shape}")

# OOD features
ood_feat_log_all = {}
food_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    print(f"Loading OOD ({ood_dataset}): {cache_name}")
    ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
    ood_feat_log_all[ood_dataset] = ood_feat_log
    food_all[ood_dataset] = prepos_feat(ood_feat_log)
    print(f"  Shape: {ood_feat_log.shape} -> preprocessed: {food_all[ood_dataset].shape}")

# ============================================================================
# KNN/ANN OOD Detection (FAISS GPU)
# ============================================================================
print(f"\n{'='*60}")
print(f"KNN-OOD Detection (K={K})")
print(f"{'='*60}")

ALPHA = 1.00
rand_ind = np.random.choice(id_train_size, int(id_train_size * ALPHA), replace=False)
ftrain_sample = ftrain[rand_ind]

# Build FAISS index
start_time = time.time()
if args.use_ann:
    if args.ann_method == 'ivf':
        print(f"Using ANN method: IVF (nlist={args.nlist}, nprobe={args.nprobe})")
        quantizer = faiss.IndexFlatL2(ftrain.shape[1])
        index = faiss.IndexIVFFlat(quantizer, ftrain.shape[1], args.nlist)
        index.train(ftrain_sample)
        index.add(ftrain_sample)
        index.nprobe = args.nprobe
        method_name = f'ANN-IVF(nprobe={args.nprobe})'
    elif args.ann_method == 'hnsw':
        print(f"Using ANN method: HNSW (M={args.hnsw_M}, efSearch={args.hnsw_efSearch})")
        index = faiss.IndexHNSWFlat(ftrain.shape[1], args.hnsw_M)
        index.hnsw.efConstruction = 40
        index.add(ftrain_sample)
        index.hnsw.efSearch = args.hnsw_efSearch
        method_name = f'ANN-HNSW(efSearch={args.hnsw_efSearch})'
else:
    print("Using exact KNN (IndexFlatL2)")
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain_sample)
    method_name = 'KNN'

# Move to GPU if requested
if args.use_faiss_gpu:
    if not TORCH_AVAILABLE:
        print("Warning: --use-faiss-gpu requested but torch not available. Using CPU.")
    else:
        try:
            import faiss.contrib.torch_utils
            if torch.cuda.is_available():
                print(f"Moving {method_name} index to GPU...")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f"Index successfully moved to GPU")
                method_name = f'{method_name}-GPU'
            else:
                print("GPU not available, using CPU")
        except Exception as e:
            print(f"Failed to move to GPU: {e}, using CPU")

build_time = time.time() - start_time
print(f"Index build time: {build_time:.3f}s")

# Search
search_start = time.time()
D, _ = index.search(ftest, K)
scores_in = -D[:, -1]
all_results = []
for ood_dataset, food in food_all.items():
    D, _ = index.search(food, K)
    scores_ood_test = -D[:, -1]
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

search_time = time.time() - search_start
print(f"Search time: {search_time:.3f}s")
print(f"Total time: {build_time + search_time:.3f}s")

metrics.print_all_results(all_results, args.out_datasets, method_name)
print()
