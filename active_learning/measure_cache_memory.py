"""
Quick script to measure peak memory of the CAL probability cache.
Simulates the cache at iteration 8 without running the full AL loop.

From run_inmemory_run23.log iteration 8:
- labeled set size: 606 (1% init) + 8 * 1212 (2% per iter) = 10,302
- num_classes = 2 (SST-2)
- cache_max_size = None (unlimited)
- Each entry: F.softmax(train_logits[n_idx:n_idx+1], dim=-1) -> tensor(1, num_classes)
"""

import sys
import tracemalloc
import torch
import torch.nn.functional as F
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.eviction_count = 0
        self.hit_count = 0
        self.miss_count = 0
    
    def put(self, key, value):
        if key in self.cache:
            if self.max_size is not None:
                self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if self.max_size is not None and len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self.eviction_count += 1
            self.cache[key] = value
    
    def __len__(self):
        return len(self.cache)


def measure_cache_memory(num_labeled, num_classes, cache_max_size=None):
    """Simulate the probability cache and measure peak memory."""
    
    # Simulate train_logits (what the model would produce)
    train_logits = torch.randn(num_labeled, num_classes)
    
    tracemalloc.start()
    
    # Create cache (same as in cal.py)
    cache = LRUCache(max_size=cache_max_size)
    
    # Fill cache with softmax probabilities (simulating full cache at end of iteration)
    for idx in range(num_labeled):
        prob = F.softmax(train_logits[idx:idx+1], dim=-1)
        cache.put(idx, prob)
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Also compute with sys.getsizeof for cross-validation
    # Size of a single entry
    single_prob = F.softmax(train_logits[0:0+1], dim=-1)
    single_tensor_size = sys.getsizeof(single_prob.storage()) + sys.getsizeof(single_prob)
    
    print(f"=== CAL Probability Cache Memory Measurement ===")
    print(f"Configuration:")
    print(f"  Labeled set size:  {num_labeled}")
    print(f"  num_classes:       {num_classes}")
    print(f"  cache_max_size:    {cache_max_size or 'unlimited'}")
    print(f"  Cache entries:     {len(cache)}")
    print(f"  Entry shape:       (1, {num_classes})")
    print(f"")
    print(f"Memory (tracemalloc):")
    print(f"  Current:           {current_mem / 1024:.2f} KB  ({current_mem / (1024*1024):.4f} MB)")
    print(f"  Peak:              {peak_mem / 1024:.2f} KB  ({peak_mem / (1024*1024):.4f} MB)")
    print(f"")
    print(f"Per-entry estimate:")
    print(f"  Single tensor:     {single_tensor_size} bytes")
    print(f"  Pure data:         {num_classes * 4} bytes (float32)")
    print(f"  Estimated total:   {single_tensor_size * num_labeled / 1024:.2f} KB  ({single_tensor_size * num_labeled / (1024*1024):.4f} MB)")
    
    return peak_mem


if __name__ == "__main__":
    # SST-2 dataset parameters from run_inmemory_run23.log
    INIT_SIZE = 606        # 1% of ~60k training set
    ACQ_PER_ITER = 1212    # 2% per iteration
    NUM_CLASSES = 2        # SST-2 binary
    
    print("=" * 70)
    print("Measuring CAL probability cache memory at each iteration")
    print("=" * 70)
    
    for iteration in range(1, 9):
        num_labeled = INIT_SIZE + iteration * ACQ_PER_ITER
        print(f"\n--- Iteration {iteration} (labeled={num_labeled}) ---")
        peak = measure_cache_memory(num_labeled, NUM_CLASSES)
    
    # Also measure for larger datasets (if scaling)
    print(f"\n\n{'=' * 70}")
    print("Scaling analysis: Cache memory vs labeled set size")
    print("=" * 70)
    for n in [1000, 5000, 10000, 50000, 100000]:
        print(f"\n--- N={n:,}, classes=2 ---")
        measure_cache_memory(n, NUM_CLASSES)
