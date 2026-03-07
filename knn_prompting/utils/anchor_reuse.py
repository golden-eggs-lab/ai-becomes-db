import torch
from torch import nn
import torch.nn.functional as F


class AnchorStoreOptV1(nn.Module):
    """
    Optimization v1: Pre-compute log(k_i) offline only.
    
    Key optimization:
    - Pre-compute log(k_i) offline to avoid redundant log computation at inference time
    - Still use the original KL formula: sum(k * (log(k) - log(q)))
    
    This isolates the benefit of pre-computing log(k_i).
    """

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2):
        super(AnchorStoreOptV1, self).__init__()

        # Store both original anchor and pre-computed log
        self.register_buffer("queue_anchor", torch.zeros(K, dim))
        self.register_buffer("queue_anchor_log", torch.zeros(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.knn = knn
        self.n_class = n_class

    def enqueue(self, anchors, labels):
        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        self.queue_anchor[ptr:ptr + bs, :] = anchors
        # Pre-compute log(k_i) offline
        self.queue_anchor_log[ptr:ptr + bs, :] = torch.log(anchors + 1e-10)
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

    def knn_infer(self, query):
        ptr = int(self.queue_ptr)
        
        # Use pre-computed log(k_i) instead of computing it every time
        # KL(k||q) = sum(k * (log(k) - log(q)))
        valid_anchor = self.queue_anchor[:ptr]
        valid_anchor_log = self.queue_anchor_log[:ptr]
        valid_labels = self.queue_label[:ptr]
        
        # Original KL formula but with pre-computed log
        query_log = torch.log(query + 1e-10)
        kl_distance = torch.mean(
            valid_anchor[None, :, :] * (valid_anchor_log[None, :, :] - query_log[:, None, :]),
            dim=2
        )  # [batch, N]
        
        if self.knn == 1:
            return valid_labels[kl_distance.argmin(dim=1)].tolist()
        else:
            values, indices = torch.topk(kl_distance, self.knn, dim=1, largest=False)
            knn_cnt = torch.zeros((query.shape[0], self.n_class), device=query.device)
            for i in range(self.n_class):
                knn_cnt[:, i] = (valid_labels[indices] == i).sum(dim=1)
            return knn_cnt.argmax(dim=1).tolist()


class AnchorStoreOptV2(nn.Module):
    """
    Optimization v2: Pre-compute log(k_i) + Matrix multiplication (Inner Product).
    
    Key optimizations:
    1. Pre-compute log(k_i) offline to avoid redundant log computation
    2. Use matrix multiplication for Inner Product search instead of KL formula
    
    Mathematical equivalence:
    KL(k||q) = sum(k * log(k)) - sum(k * log(q))
    Since sum(k * log(k)) is constant for each anchor, minimizing KL is equivalent to
    maximizing sum(k * log(q)) = k · log(q), which is an Inner Product.
    
    This uses efficient BLAS operations (matmul) instead of element-wise operations.
    """

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2):
        super(AnchorStoreOptV2, self).__init__()

        # Store anchor k_i, log(k_i), and per-anchor entropy sum(k*log(k))
        self.register_buffer("queue_anchor", torch.zeros(K, dim))
        self.register_buffer("queue_anchor_log", torch.zeros(K, dim))
        self.register_buffer("queue_anchor_entropy", torch.zeros(K))  # sum(k*log(k)) per anchor
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.knn = knn
        self.n_class = n_class

    def enqueue(self, anchors, labels):
        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        self.queue_anchor[ptr:ptr + bs, :] = anchors
        # Pre-compute log(k_i) offline
        anchor_log = torch.log(anchors + 1e-10)
        self.queue_anchor_log[ptr:ptr + bs, :] = anchor_log
        # Pre-compute per-anchor entropy: sum(k * log(k))
        self.queue_anchor_entropy[ptr:ptr + bs] = torch.sum(anchors * anchor_log, dim=1)
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

    def knn_infer(self, query):
        ptr = int(self.queue_ptr)
        batch_size = query.shape[0]
        
        # Get valid anchors
        valid_anchor = self.queue_anchor[:ptr]  # [N, dim]
        valid_entropy = self.queue_anchor_entropy[:ptr]  # [N]
        valid_labels = self.queue_label[:ptr]  # [N]
        
        # Compute query log once
        query_log = torch.log(query + 1e-10)  # [batch, dim]
        
        # KL(k||q) = sum(k*log(k)) - sum(k*log(q))
        # ip_scores[i, j] = sum(anchor[j] * log(query[i]))
        ip_scores = torch.matmul(valid_anchor, query_log.T).T  # [batch, N]
        
        # Correct KL: subtract entropy term so we can use argmin
        # neg_kl[i,j] = sum(k_j*log(q_i)) - sum(k_j*log(k_j)) = -KL(k_j||q_i)
        neg_kl = ip_scores - valid_entropy[None, :]  # [batch, N]
        
        # Higher neg_kl means lower KL divergence (more similar)
        k = min(self.knn, ptr)
        
        if self.knn == 1:
            indices = neg_kl.argmax(dim=1)
            return valid_labels[indices].tolist()
        else:
            _, indices = torch.topk(neg_kl, k, dim=1, largest=True)
            knn_cnt = torch.zeros((batch_size, self.n_class), device=query.device)
            for i in range(self.n_class):
                knn_cnt[:, i] = (valid_labels[indices] == i).sum(dim=1)
            return knn_cnt.argmax(dim=1).tolist()


# Alias for backward compatibility
AnchorStoreANN = AnchorStoreOptV2
