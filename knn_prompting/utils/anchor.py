import torch
from torch import nn
import torch.nn.functional as F
import time

class AnchorStore(nn.Module):
    """Baseline AnchorStore with bottleneck timing instrumentation."""

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2):
        super(AnchorStore, self).__init__()

        self.register_buffer("queue_anchor", torch.randn(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.knn = knn
        self.n_class = n_class
        
        # Bottleneck timing stats
        self.total_infer_time = 0.0
        self.log_k_time = 0.0  # Time spent on log(k) - the optimized component
        self.other_time = 0.0

    def enqueue(self, anchors, labels):

        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        self.queue_anchor[ptr:ptr + bs, :] = anchors
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

    def knn_infer(self, query):
        """KNN inference with bottleneck timing."""
        start_total = time.time()
        
        # === OPTIMIZED COMPONENT: log(k) + distance calculation ===
        # This is the part that gets optimized in opt_v2:
        # - log(k) is pre-computed (reuse)
        # - distance uses matmul instead of broadcasting
        start_optimizable = time.time()
        anchor_log = self.queue_anchor[:, None, :].log()  # Redundantly computed!
        query_log = query.log()
        kl_distance = torch.mean(
            self.queue_anchor[:, None, :] * (anchor_log - query_log), 
            dim=2
        ).transpose(1, 0)
        end_optimizable = time.time()
        
        # === TopK selection (not optimized) ===
        start_topk = time.time()
        if self.knn == 1:
            result = self.queue_label[kl_distance.argmin(dim=1)].tolist()
        else:
            values, indices = torch.topk(kl_distance, self.knn, dim=1, largest=False)
            knn_cnt = torch.zeros((query.shape[0], self.n_class))
            for i in range(self.n_class):
                knn_cnt[:, i] = (self.queue_label[indices] == i).sum(dim=1)
            result = knn_cnt.argmax(dim=1).tolist()
        end_topk = time.time()
        
        # Record timing
        self.log_k_time += (end_optimizable - start_optimizable)  # Now includes log + distance
        self.other_time += (end_topk - start_topk)
        self.total_infer_time += (time.time() - start_total)
        
        return result
    
    def get_bottleneck_stats(self):
        """Return bottleneck statistics."""
        if self.total_infer_time > 0:
            optimized_pct = (self.log_k_time / self.total_infer_time) * 100
            topk_pct = (self.other_time / self.total_infer_time) * 100
        else:
            optimized_pct = 0.0
            topk_pct = 0.0
        
        return {
            'total_infer_time': self.total_infer_time,
            'log_k_time': self.log_k_time,  # Now: log + distance calculation
            'log_k_percentage': optimized_pct,
            'other_time': self.other_time,  # Now: topk only
            'other_percentage': topk_pct,
        }
