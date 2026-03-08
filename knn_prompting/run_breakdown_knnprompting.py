#!/usr/bin/env python
"""
KNN-Prompting Component Breakdown for Table 5.

Measures the Intermediate Results Reuse component:
  - Original: log(k) computation + broadcasting KL distance  (anchor.py)
  - +All (IV-Aligned): pre-computed log(k) lookup + BLAS matmul       (anchor_ann.py opt_v2)

Both share the same LLM inference / anchor building phase;
only the knn_infer() component is timed differently.

Usage:
    conda run --no-capture-output -n knnseq python -u run_breakdown_knnprompting.py --runs 3
"""

import os, sys, time, json, argparse, logging
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from utils.dataset import *
from utils.template import *
from utils.anchor import AnchorStore
from utils.anchor_ann import AnchorStoreOptV2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


# ── Instrument OptV2 to record component time ─────────────────────────
class AnchorStoreOptV2Instrumented(AnchorStoreOptV2):
    """AnchorStoreOptV2 with timing for the optimized component
    (query_log + matmul + entropy subtraction)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_time = 0.0   # matmul-based distance
        self.topk_time = 0.0        # topk selection
        self.total_infer_time = 0.0

    def knn_infer(self, query):
        start_total = time.time()
        ptr = int(self.queue_ptr)
        batch_size = query.shape[0]

        valid_anchor = self.queue_anchor[:ptr]
        valid_entropy = self.queue_anchor_entropy[:ptr]
        valid_labels = self.queue_label[:ptr]

        # ─── Optimized component: query_log + matmul + entropy ───
        start_component = time.time()
        query_log = torch.log(query + 1e-10)
        ip_scores = torch.matmul(valid_anchor, query_log.T).T
        neg_kl = ip_scores - valid_entropy[None, :]
        end_component = time.time()

        # ─── TopK selection ───
        start_topk = time.time()
        k = min(self.knn, ptr)
        if self.knn == 1:
            indices = neg_kl.argmax(dim=1)
            result = valid_labels[indices].tolist()
        else:
            _, indices = torch.topk(neg_kl, k, dim=1, largest=True)
            knn_cnt = torch.zeros((batch_size, self.n_class), device=query.device)
            for i in range(self.n_class):
                knn_cnt[:, i] = (valid_labels[indices] == i).sum(dim=1)
            result = knn_cnt.argmax(dim=1).tolist()
        end_topk = time.time()

        self.component_time += (end_component - start_component)
        self.topk_time += (end_topk - start_topk)
        self.total_infer_time += (time.time() - start_total)
        return result

    def get_component_stats(self):
        return {
            'component_time': self.component_time,
            'topk_time': self.topk_time,
            'total_infer_time': self.total_infer_time,
        }


# ── LLM generation (same as knn_prompting.py) ─────────────────────────
def llm_gen(model, prompt, tokenizer, max_context_len):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    with torch.no_grad():
        logits = model.forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        ).logits.detach().cpu()
    return logits[:, -1, :]


# ── Build anchor store and cache softmax outputs ──────────────────────
def build_anchors_and_cache(model, tokenizer, anchor_data, prompt_prefix,
                            dataset_name, label2id, max_context_len):
    """Build anchors with LLM and cache softmax outputs for reuse."""
    cached_logits = []
    cached_labels = []
    for ins in tqdm(anchor_data.data, total=anchor_data.__len__(), desc="Caching anchor logits"):
        labels = label2id[ins['label']]
        prompt = prompt_prefix + make_prompt(ins, dataset_name, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        softmax_out = torch.softmax(gen_logits, dim=-1)
        cached_logits.append(softmax_out)
        cached_labels.append(torch.tensor(labels))
    return cached_logits, cached_labels


def build_anchor_store_from_cache(anchor_store, cached_logits, cached_labels):
    """Fill anchor store from cached logits (no LLM call)."""
    for logit, label in zip(cached_logits, cached_labels):
        anchor_store.enqueue(logit, label)


# ── Run inference with a given anchor store ───────────────────────────
def run_inference(anchor_store, model, tokenizer, dev_data, prompt_prefix,
                  dataset_name, label2id, max_context_len):
    """Run inference on dev set, return acc and timing."""
    dev_labels = []
    dev_pred = []
    for ins in tqdm(dev_data.data, total=dev_data.__len__(), desc="Inference"):
        dev_labels.append(label2id[ins['label']])
        prompt = prompt_prefix + make_prompt(ins, dataset_name, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        dev_pred.extend(anchor_store.knn_infer(torch.softmax(gen_logits, dim=-1)))

    correct = sum(1 for a, b in zip(dev_labels, dev_pred) if a == b)
    acc = correct / len(dev_labels)
    return acc


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KNN-Prompting Table 5 Component Breakdown")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--llm_dir", type=str, default="./llm/gpt2-xl")
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--n_train_shot", type=int, default=1024)
    parser.add_argument("--n_demo_shot", type=int, default=2)
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_anchor_shot = args.n_train_shot - args.n_demo_shot
    args.knn = min(args.knn, n_anchor_shot)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.llm_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.llm_dir)
    model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
    model.to(device)
    model.eval()

    max_context_len = 1024 if 'gpt2' in args.llm_dir else 2048

    # Prepare dataset
    dataset_map = {
        'sst2': SST2Dataset, 'subj': SUBJDataset, 'agnews': AGNEWSDataset,
        'cb': CBDataset, 'cr': CRDataset, 'dbpedia': DBPEDIADataset,
        'mpqa': MPQADataset, 'mr': MRDataset, 'rte': RTEDataset, 'trec': TRECDataset,
    }
    AutoDataset = dataset_map[args.dataset]
    datadir = os.path.join(args.data_dir, args.dataset)
    train_data = AutoDataset(datadir, mode='train')
    dev_data = AutoDataset(datadir, mode='dev')
    anchor_data = AutoDataset(datadir, mode='train')

    train_data.subsamplebyshot(args.n_demo_shot, args.seed)
    prompt_prefix = make_prompt(train_data, args.dataset, mode='train')
    anchor_data.subsamplebyshot(n_anchor_shot, args.seed, exclude=train_data.data)
    label2id = dev_data.label2id

    vocab_size = model_config.vocab_size
    n_class = len(label2id)

    print(f"\n{'='*70}")
    print(f"KNN-Prompting Component Breakdown")
    print(f"  Dataset: {args.dataset}, LLM: {args.llm_dir}")
    print(f"  n_train={args.n_train_shot}, n_demo={args.n_demo_shot}, "
          f"n_anchor={n_anchor_shot}, knn={args.knn}")
    print(f"  Vocab size: {vocab_size}, Num classes: {n_class}")
    print(f"  Runs: {args.runs}")
    print(f"{'='*70}\n")

    # Cache anchor logits once (LLM inference is shared)
    print("Building and caching anchor softmax outputs (one-time LLM cost)...")
    cached_logits, cached_labels = build_anchors_and_cache(
        model, tokenizer, anchor_data, prompt_prefix,
        args.dataset, label2id, max_context_len
    )
    print(f"Cached {len(cached_logits)} anchor softmax outputs.\n")

    # Also cache dev softmax outputs to avoid repeated LLM calls
    print("Caching dev set softmax outputs (one-time LLM cost)...")
    dev_cached_logits = []
    dev_cached_labels = []
    for ins in tqdm(dev_data.data, total=dev_data.__len__(), desc="Caching dev logits"):
        dev_cached_labels.append(label2id[ins['label']])
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        dev_cached_logits.append(torch.softmax(gen_logits, dim=-1))
    print(f"Cached {len(dev_cached_logits)} dev softmax outputs.\n")

    # Run breakdown experiments
    baseline_results = []
    optimized_results = []

    for run_id in range(1, args.runs + 1):
        print(f"\n{'='*60}")
        print(f"Run {run_id}/{args.runs}")
        print(f"{'='*60}")

        # ─── Original: raw `knn_infer()` execution (includes log(k) + PRKL + TopK) ───
        print(f"\n  [Original] Creating anchor store and running inference...")
        baseline_store = AnchorStore(
            K=n_anchor_shot, dim=vocab_size, knn=args.knn, n_class=n_class
        )
        build_anchor_store_from_cache(baseline_store, cached_logits, cached_labels)

        # We will time the *entire* `knn_infer()` call, which encapsulates the true 
        # original component cost (including the very expensive repeated `torch.log`)
        dev_pred_baseline = []
        baseline_infer_start = time.time()
        for logit in tqdm(dev_cached_logits, desc="  Original inference"):
            dev_pred_baseline.extend(baseline_store.knn_infer(logit))
        baseline_total_infer_time = time.time() - baseline_infer_start

        baseline_stats = baseline_store.get_bottleneck_stats()
        baseline_acc = sum(1 for a, b in zip(dev_cached_labels, dev_pred_baseline) if a == b) / len(dev_cached_labels)

        # The actual component time = Total Infer Time - TopK Time (to isolate "Intermediate Results Computation")
        actual_baseline_component_time = baseline_total_infer_time - baseline_stats['other_time']

        print(f"  [Original] Acc: {baseline_acc:.4f}")
        print(f"  [Original] Component (Full log+KL): {actual_baseline_component_time:.4f}s")
        print(f"  [Original] TopK:                    {baseline_stats['other_time']:.4f}s")
        print(f"  [Original] Total knn_infer:         {baseline_total_infer_time:.4f}s")

        baseline_results.append({
            'acc': baseline_acc,
            'component_time': actual_baseline_component_time,
            'topk_time': baseline_stats['other_time'],
            'total_infer_time': baseline_total_infer_time,
        })

        # ─── +All (IV-Aligned) (anchor_ann.py opt_v2): pre-computed log + matmul ───
        print(f"\n  [+All] Creating anchor store and running inference...")
        opt_store = AnchorStoreOptV2Instrumented(
            K=n_anchor_shot, dim=vocab_size, knn=args.knn, n_class=n_class
        )
        build_anchor_store_from_cache(opt_store, cached_logits, cached_labels)

        dev_pred_opt = []
        for logit in tqdm(dev_cached_logits, desc="  +All inference"):
            dev_pred_opt.extend(opt_store.knn_infer(logit))

        opt_stats = opt_store.get_component_stats()
        opt_acc = sum(1 for a, b in zip(dev_cached_labels, dev_pred_opt) if a == b) / len(dev_cached_labels)

        print(f"  [+All] Acc: {opt_acc:.4f}")
        print(f"  [+All] Component (matmul):  {opt_stats['component_time']:.4f}s")
        print(f"  [+All] TopK:                {opt_stats['topk_time']:.4f}s")
        print(f"  [+All] Total knn_infer:     {opt_stats['total_infer_time']:.4f}s")

        # Compute reduction using the actual full baseline component time
        if actual_baseline_component_time > 0:
            reduction = (1 - opt_stats['component_time'] / actual_baseline_component_time) * 100
        else:
            reduction = 0.0
        print(f"\n  Computation Reduced: {reduction:.2f}%")

        optimized_results.append({
            'acc': opt_acc,
            'component_time': opt_stats['component_time'],
            'topk_time': opt_stats['topk_time'],
            'total_infer_time': opt_stats['total_infer_time'],
        })

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"SUMMARY: KNN-Prompting Component Breakdown (mean ± std, {args.runs} runs)")
    print(f"{'='*70}")

    bl_comp = [r['component_time'] for r in baseline_results]
    bl_topk = [r['topk_time'] for r in baseline_results]
    bl_total = [r['total_infer_time'] for r in baseline_results]
    bl_acc = [r['acc'] for r in baseline_results]

    op_comp = [r['component_time'] for r in optimized_results]
    op_topk = [r['topk_time'] for r in optimized_results]
    op_total = [r['total_infer_time'] for r in optimized_results]
    op_acc = [r['acc'] for r in optimized_results]

    reductions = [(1 - o / b) * 100 for b, o in zip(bl_comp, op_comp)]

    print(f"\nOriginal (Full log(k) computation + broadcasting KL):")
    print(f"  Accuracy:        {np.mean(bl_acc)*100:.2f} ± {np.std(bl_acc)*100:.2f}%")
    print(f"  Component time:  {np.mean(bl_comp):.4f} ± {np.std(bl_comp):.4f}s")
    print(f"  TopK time:       {np.mean(bl_topk):.4f} ± {np.std(bl_topk):.4f}s")
    print(f"  Total knn_infer: {np.mean(bl_total):.4f} ± {np.std(bl_total):.4f}s")

    print(f"\n+All (IV-Aligned) (pre-computed log + BLAS matmul):")
    print(f"  Accuracy:        {np.mean(op_acc)*100:.2f} ± {np.std(op_acc)*100:.2f}%")
    print(f"  Component time:  {np.mean(op_comp):.4f} ± {np.std(op_comp):.4f}s")
    print(f"  TopK time:       {np.mean(op_topk):.4f} ± {np.std(op_topk):.4f}s")
    print(f"  Total knn_infer: {np.mean(op_total):.4f} ± {np.std(op_total):.4f}s")

    print(f"\n--- Table 5 Row (Intermediate Results Reuse) ---")
    print(f"  Original Time:        {np.mean(bl_comp):.2f} ± {np.std(bl_comp):.2f}s")
    print(f"  +Reuse Time:          {np.mean(op_comp):.4f} ± {np.std(op_comp):.4f}s")
    print(f"  Computation Reduced:  {np.mean(reductions):.2f} ± {np.std(reductions):.2f}%")
    print(f"  Speedup:              {np.mean(bl_comp) / np.mean(op_comp):.1f}x")
    print(f"{'='*70}")

    # Save results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"results_breakdown_{timestamp}.json"
    results = {
        "experiment": "knn_prompting_component_breakdown",
        "dataset": args.dataset,
        "llm": args.llm_dir,
        "n_train_shot": args.n_train_shot,
        "n_demo_shot": args.n_demo_shot,
        "n_anchor_shot": n_anchor_shot,
        "knn": args.knn,
        "seed": args.seed,
        "num_runs": args.runs,
        "timestamp": timestamp,
        "original": {
            "method": "log(k) + broadcasting KL (anchor.py)",
            "component_time_mean": float(np.mean(bl_comp)),
            "component_time_std": float(np.std(bl_comp)),
            "topk_time_mean": float(np.mean(bl_topk)),
            "topk_time_std": float(np.std(bl_topk)),
            "total_infer_time_mean": float(np.mean(bl_total)),
            "total_infer_time_std": float(np.std(bl_total)),
            "accuracy_mean": float(np.mean(bl_acc)),
            "accuracy_std": float(np.std(bl_acc)),
            "all_runs": baseline_results,
        },
        "+all": {
            "method": "pre-computed log(k) + BLAS matmul (anchor_ann.py opt_v2)",
            "component_time_mean": float(np.mean(op_comp)),
            "component_time_std": float(np.std(op_comp)),
            "topk_time_mean": float(np.mean(op_topk)),
            "topk_time_std": float(np.std(op_topk)),
            "total_infer_time_mean": float(np.mean(op_total)),
            "total_infer_time_std": float(np.std(op_total)),
            "accuracy_mean": float(np.mean(op_acc)),
            "accuracy_std": float(np.std(op_acc)),
            "all_runs": optimized_results,
        },
        "table5_reuse": {
            "baseline_time": f"{np.mean(bl_comp):.2f} ± {np.std(bl_comp):.2f}s",
            "reuse_time": f"{np.mean(op_comp):.4f} ± {np.std(op_comp):.4f}s",
            "computation_reduced": f"{np.mean(reductions):.2f} ± {np.std(reductions):.2f}%",
            "speedup": f"{np.mean(bl_comp) / np.mean(op_comp):.1f}x",
        },
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
