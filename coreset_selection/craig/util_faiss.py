import itertools
import os
import subprocess
import time
import gc

import matplotlib.pyplot as plt
import numpy as np
from lazy_greedy_faiss import FacilityLocation, lazy_greedy_heap, FacilityLocationANN, lazy_greedy_heap_ann, lazy_greedy_heap_ann_batch
import scipy.spatial

from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances


SEED = 100
EPS = 1E-8
PLOT_NAMES = ['lr', 'data_loss', 'epoch_loss', 'test_loss']


def load_dataset(dataset, dataset_dir):
    '''
    Args
    - dataset: str, one of ['cifar10', 'covtype'] or filename in `data/`
    - dataset_dir: str, path to `data` folder

    Returns
    - X: np.array, shape [N, d]
      - exception: shape [N, 32, 32, 3] for cifar10
    - y: np.array, shape [N]
    '''
    if dataset == 'cifar10':
        path = os.path.join(dataset_dir, 'cifar10', 'cifar10.npz')
        with np.load(path) as npz:
            X = npz['x']  # shape [60000, 32, 32, 3], type uint8
            y = npz['y']  # shape [60000], type uint8
        # convert to float in (0, 1), center at mean 0
        X = X.astype(np.float32) / 255
    elif dataset == 'cifar10_features':
        path = os.path.join(dataset_dir, 'cifar10', 'train_features.npz')
        with np.load(path) as npz:
            X = npz['features']  # shape [50000, 64], type float32
            y = npz['labels']  # shape [50000], type int64
    elif dataset == 'cifar10_grads':
        # labels
        path = os.path.join(dataset_dir, 'cifar10', 'train_features.npz')
        with np.load(path) as npz:
            y = npz['labels']  # shape [50000], type int64
        # feautres
        path = os.path.join('grad_features.npy')
        X = np.load(path)  # shape [50000, 1000], type float16
    elif dataset == 'mnist':
        # Use TensorFlow 2.x Keras API
        from tensorflow.keras.datasets import mnist as mnist_dataset
        (X_train, y_train), (X_test, y_test) = mnist_dataset.load_data()
        
        # Reshape and normalize
        X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
        X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
        
        return X_train, y_train, X_test, y_test

    else:
        num, dim, name = 0, 0, ''
        if dataset == 'covtype':
            num, dim = 581012, 54
            name = 'covtype.libsvm.binary.scale'
        elif dataset == 'ijcnn1.t' or dataset == 'ijcnn1.tr':
            num, dim = 49990 if 'tr' in dataset else 91701, 22
            name = dataset
        elif dataset == 'combined_scale' or dataset == 'combined_scale.t':
            num, dim = 19705 if '.t' in dataset else 78823, 100
            name = dataset

        X = np.zeros((num, dim), dtype=np.float32)
        y = np.zeros(num, dtype=np.int32)
        path = os.path.join(dataset_dir, name)

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                y[i] = float(line.split()[0])
                for e in line.split()[1:]:
                    cur = e.split(':')
                    X[i][int(cur[0]) - 1] = float(cur[1])
                i += 1
        y = np.array(y, dtype=np.int32)
        if name in ['ijcnn1.t', 'ijcnn1.tr']:
            y[y == -1] = 0
        else:
            y = y - np.ones(len(y), dtype=np.int32)

    return X, y


def similarity(X, metric):
    '''Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]
    '''
    start = time.time()
    dists = sklearn_pairwise_distances(X, metric=metric, n_jobs=1)
    elapsed = time.time() - start

    if metric == 'cosine':
        S = 1 - dists
    elif metric == 'euclidean' or metric == 'l1':
        m = np.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')

    return S, elapsed


def get_facility_location_submodular_order(S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None, 
                                           use_ann=False, gradients=None, ann_k=20, ann_backend='faiss', 
                                           force_backend=False, batch_size=20, use_reuse=True):
    '''
    Args
    - S: np.array, shape [N, N], similarity matrix
    - B: int, number of points to select
    - use_ann: bool, whether to use ANN acceleration (default False for backward compatibility)
    - gradients: np.array, shape [N, d], gradient vectors (required if use_ann=True or force_backend=True)
    - ann_k: int, number of ANN candidates to consider (default 20)
    - ann_backend: str, 'faiss', 'milvus', or 'spark' (default 'faiss')
    - force_backend: bool, force using backend even for exact search (for fair comparison on same infrastructure)
    - batch_size: int, for Spark backend: how many selections per Spark call (default 20)

    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    N = S.shape[0]
    no = smtk if no == 0 else no

    if smtk > 0:
        print(f'Calculating ordering with SMTK... part size: {len(S)}, B: {B}', flush=True)
        np.save(f'/tmp/{no}/{smtk}-{c}', S)
        if stoch_greedy > 0:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                 -stochastic-greedy -sg-epsilon {stoch_greedy} -flnpy /tmp/{no}/{smtk}-{c}.'
                f'npy -pnpv -porder -ptime'.split())
        else:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                             -flnpy /tmp/{no}/{smtk}-{c}.npy -pnpv -porder -ptime'.split())
        s = p.decode("utf-8")
        str, end = ['([', ',])']
        order = s[s.find(str) + len(str):s.rfind(end)].split(',')
        greedy_time = float(s[s.find('CPU') + 4 : s.find('s (User')])
        str = 'f(Solution) = '
        F_val = float(s[s.find(str) + len(str) : s.find('Summary Solution') - 1])
    else:
        V = list(range(N))
        start = time.time()
        
        # Choose backend and mode
        if (use_ann or force_backend) and gradients is not None:
            mode = "ANN" if use_ann else "Exact"
            print(f'Using {mode} greedy (backend={ann_backend}, k={ann_k if use_ann else "all"})...', flush=True)
            F = FacilityLocationANN(S, V, gradients, ann_k=ann_k, use_ann=use_ann, ann_backend=ann_backend, force_backend=force_backend, use_reuse=use_reuse)
            
            # Use batch strategy for backends
            if ann_backend in ['spark', 'faiss', 'milvus']:
                print(f'  Using BATCH strategy: batch_size={batch_size}', flush=True)
                order, _ = lazy_greedy_heap_ann_batch(F, V, B, batch_size=batch_size, use_ann=use_ann)
            else:
                order, _ = lazy_greedy_heap_ann(F, V, B, use_ann=use_ann)
        else:
            F = FacilityLocation(S, V)
            order, _ = lazy_greedy_heap(F, V, B)
        
        greedy_time = time.time() - start
        F_val = 0

    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    collected = gc.collect()
    return order, sz, greedy_time, F_val


def faciliy_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None, 
                          use_ann=False, ann_k=20, ann_backend='faiss', force_backend=False, batch_size=20, use_reuse=True):
    class_indices = np.where(y == c)[0]
    print(c)
    print(class_indices)
    print(len(class_indices))
    S, S_time = similarity(X[class_indices], metric=metric)
    
    # Pass gradients for backend if requested
    gradients = X[class_indices] if (use_ann or force_backend) else None
    
    order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
        S, num_per_class, c, smtk, no, stoch_greedy, weights, 
        use_ann=use_ann, gradients=gradients, ann_k=ann_k, ann_backend=ann_backend, 
        force_backend=force_backend, batch_size=batch_size, use_reuse=use_reuse)
    return class_indices[order], cluster_sz, greedy_time, S_time


def get_orders_and_weights(B, X, metric, smtk, no=0, stoch_greedy=0, y=None, weights=None, equal_num=False, 
                          outdir='.', use_ann=False, ann_k=20, ann_backend='faiss', force_backend=False, batch_size=20, use_reuse=True):
    '''
    Args
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist
    - use_ann: bool, whether to use ANN acceleration (default False)
    - ann_k: int, number of ANN candidates (default 20)
    - ann_backend: str, 'faiss', 'milvus', or 'spark' (default 'faiss')
    - force_backend: bool, force using backend infrastructure for exact search (for fair comparison)
    - batch_size: int, for backend: how many selections per call (default 20)

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        print('not equal_num')

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights, 
                                        use_ann=use_ann, ann_k=ann_k, ann_backend=ann_backend, 
                                        force_backend=force_backend, batch_size=batch_size, use_reuse=use_reuse), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))
        print(f'Selecting with ratios {np.array(class_ratios)}')
        print(f'Class proportions {np.array(props)}')

    order_mg_all = np.array(order_mg_all)
    cluster_sizes_all = np.array(cluster_sizes_all)
    for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))):
        for c in classes:
            ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(weights_mg, dtype=np.float32)
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals
