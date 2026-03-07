"""
SCIP: Synthetic Corruption Informed Pruning
------------------------------------------

This file implements the core pruning algorithm from the paper:

    "Decoding Data Quality via Synthetic Corruptions:
     Embedding-guided Pruning of Code Data"
     (Yu Yang et al., ENLSP @ NeurIPS 2023)

High-level idea (what this file is supposed to do):

1. We start from a large code dataset (e.g., The Stack v1.1, Python subset).
2. We obtain code embeddings (e.g., using StarEncoder from the BigCode project).
3. We cluster embeddings with K-Means, then for each sample we know:
   - Which cluster it belongs to
   - The size of that cluster
   - The distance from the sample to the closest cluster centroid
4. We prune (remove) a fraction p of the dataset according to Algorithm 1:
   - First prune examples from *small clusters* (clusters with few members)
   - Then prune examples that are *far from centroids* (large distance)
   - The ratio between these two pruning criteria is controlled by alpha
5. We then fine-tune a code LLM (e.g., Code LLaMA 1.5B) on:
   - Original dataset (no pruning)
   - Various pruned datasets (Random, SSL-Prototypes, SemDeDup, D4, SCIP)
6. Finally, we evaluate the fine-tuned model on HumanEval and MBPP
   using pass@k (especially pass@1) as in the paper.

This file focuses on Step 3–4 (the SCIP pruning itself) and provides
TODO hooks for:

- Loading a dataset (e.g., The Stack Python subset)
- Computing embeddings using StarEncoder
- Fine-tuning a model (e.g., LLaMA / Code LLaMA)
- Evaluating on HumanEval / MBPP

So that GitHub Copilot (or you) can fill in the missing pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

@dataclass
class PruningConfig:
    """
    Configuration for SCIP-style pruning.

    Attributes
    ----------
    p : float
        Fraction of data to prune in total. In the paper they set p = 0.2
        (i.e., prune 20% of the data).
    alpha : float
        Weight between [0, 1] controlling the fraction of pruning driven by
        cluster size vs. distance:
            - alpha * p     fraction pruned by "small clusters"
            - (1 - alpha)*p fraction pruned by "far from centroid"
        In the paper they found alpha = 0.8 works best on HumanEval.
    n_clusters : int
        Number of K-Means clusters. The paper uses K = 100.
    batch_size : int
        MiniBatchKMeans batch size.
    random_state : int
        Random seed for reproducibility.
    """
    p: float = 0.2
    alpha: float = 0.8
    n_clusters: int = 100
    batch_size: int = 4096
    random_state: int = 42


# -----------------------------------------------------------------------------
# Core SCIP algorithm (Algorithm 1)
# -----------------------------------------------------------------------------

def scip_prune(
    embeddings: np.ndarray,
    cfg: PruningConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Embedding-guided Weighted Pruning of Code Data (Algorithm 1 in the paper).

    Parameters
    ----------
    embeddings : np.ndarray
        A 2D array of shape (N, D).
        Each row is an embedding vector for a code example. In the paper,
        these are obtained from StarEncoder (BigCode) and l2-normalized.
    cfg : PruningConfig
        Configuration parameters (p, alpha, n_clusters, ...).

    Returns
    -------
    keep_indices : np.ndarray
        Indices of examples to KEEP (high-quality data) after pruning.
    prune_indices : np.ndarray
        Indices of examples to PRUNE (low-quality data).
    extra_info : Dict[str, np.ndarray]
        Extra diagnostics:
            - "labels": cluster id per example (shape = [N])
            - "cluster_sizes": size of each cluster (shape = [K])
            - "distances": distance to closest centroid per example (shape = [N])

    Notes
    -----
    Distance metric
    ---------------
    The paper uses cosine distance in the embedding space. If we normalize
    embeddings and centroids to unit norm, then:

        cosine_similarity(x, c) = x · c
        cosine_distance(x, c)   = 1 - cosine_similarity(x, c)

    After l2-normalization, we can safely use dot products to compute cosines.
    """

    p = cfg.p
    alpha = cfg.alpha
    K = cfg.n_clusters

    assert 0.0 < p < 1.0, "p must be in (0, 1)"
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"

    N, D = embeddings.shape

    # -------------------------------------------------------------------------
    # Step 0: Normalize embeddings to unit norm (so cosine distance is valid)
    # -------------------------------------------------------------------------
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings_norm = embeddings / norms

    # -------------------------------------------------------------------------
    # Step 1: Cluster D into K clusters (K-Means with cosine-ish distance).
    #
    # In the paper they say: "We used k = 100 clusters, identified using the
    # K-means algorithm with cosine similarity as the metric."
    #
    # MiniBatchKMeans uses Euclidean distance internally, but with normalized
    # embeddings, minimizing Euclidean distance is equivalent to maximizing
    # cosine similarity up to a constant factor.
    #
    # TODO: if you want EXACT cosine K-means, you could implement spherical
    # k-means or use a library that supports cosine distance directly.
    # -------------------------------------------------------------------------
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=cfg.batch_size,
        random_state=cfg.random_state,
        verbose=False,
    )
    kmeans.fit(embeddings_norm)

    labels = kmeans.labels_                 # shape (N,)
    centroids = kmeans.cluster_centers_     # shape (K, D)

    # Normalize centroids as well
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    # -------------------------------------------------------------------------
    # Step 2: Calculate cluster sizes s(c_i) for each centroid
    # -------------------------------------------------------------------------
    cluster_sizes = np.bincount(labels, minlength=K)  # shape (K,)

    # -------------------------------------------------------------------------
    # Step 3–5: For each x in D:
    #   c_min(x) <- argmin_c d_C(e(x), c)
    #   d(x)     <- d_C(e(x), c_min(x))
    #
    # Here we already know c_min(x) from labels. We only need to compute
    # distances d(x) = 1 - cosine_similarity(e(x), c_min(x)).
    # -------------------------------------------------------------------------
    assigned_centroids = centroids_norm[labels]            # (N, D)
    cos_sim = np.sum(embeddings_norm * assigned_centroids, axis=1)  # (N,)
    distances = 1.0 - cos_sim                              # d(x)

    # -------------------------------------------------------------------------
    # Step 7: Rank D based on s(c_min(x)) in ascending order
    #         (i.e., samples in smaller clusters are ranked earlier)
    # -------------------------------------------------------------------------
    size_per_point = cluster_sizes[labels]  # (N,)
    idx_by_size = np.argsort(size_per_point)  # indices from smallest cluster to largest

    # Total number of samples to prune
    total_to_prune = int(round(p * N))
    total_to_prune = min(max(total_to_prune, 0), N)

    # Number of samples to prune by cluster size
    prune_by_size_quota = int(round(alpha * total_to_prune))
    prune_by_size_quota = min(prune_by_size_quota, total_to_prune)

    # Step 8: D_prune_by_size <- top alpha * p % of D based on cluster size ranking
    prune_by_size_indices = idx_by_size[:prune_by_size_quota]

    # -------------------------------------------------------------------------
    # Step 9: Rank remaining D \ D_prune_by_size based on d(x) in descending order
    # -------------------------------------------------------------------------
    mask_remaining = np.ones(N, dtype=bool)
    mask_remaining[prune_by_size_indices] = False
    remaining_indices = np.nonzero(mask_remaining)[0]

    distances_remaining = distances[remaining_indices]
    order_by_dist_desc = np.argsort(-distances_remaining)  # larger distance first
    remaining_sorted_by_dist = remaining_indices[order_by_dist_desc]

    # Number of samples to prune by distance
    prune_by_distance_quota = total_to_prune - prune_by_size_quota
    prune_by_distance_quota = min(prune_by_distance_quota, remaining_sorted_by_dist.size)

    # Step 10: D_prune_by_distance <- top (1 - alpha) * p % based on distance
    prune_by_distance_indices = remaining_sorted_by_dist[:prune_by_distance_quota]

    # -------------------------------------------------------------------------
    # Step 11: D_pruned <- D \ (D_prune_by_size ∪ D_prune_by_distance)
    # -------------------------------------------------------------------------
    prune_indices = np.concatenate([prune_by_size_indices, prune_by_distance_indices])
    prune_indices = np.unique(prune_indices)  # safety: remove possible duplicates

    mask_keep = np.ones(N, dtype=bool)
    mask_keep[prune_indices] = False
    keep_indices = np.nonzero(mask_keep)[0]

    extra_info = {
        "labels": labels,
        "cluster_sizes": cluster_sizes,
        "distances": distances,
    }

    return keep_indices, prune_indices, extra_info


# -----------------------------------------------------------------------------
# Import implementations from separate modules
# -----------------------------------------------------------------------------

from data_loading import (
    load_code_dataset as load_code_dataset_impl,
    preprocess_code_for_training,
    create_pruned_dataset,
    DataConfig,
)
from embeddings import compute_embeddings_for_dataset as compute_embeddings_impl
from training import (
    finetune_model_on_dataset as finetune_model_impl,
    setup_model_and_tokenizer,
    ModelConfig,
    TrainConfig,
    compute_training_metrics,
)
from evaluation import (
    evaluate_model_on_code_benchmarks as evaluate_model_impl,
    EvalConfig,
    save_evaluation_results,
)
from utils import (
    ExperimentConfig,
    ExperimentTracker,
    analyze_pruning_distribution,
    compare_pruning_methods,
)

# Import torch again here to ensure it's available in this scope
import torch as torch_module


# -----------------------------------------------------------------------------
# Complete end-to-end pipeline
# -----------------------------------------------------------------------------

def run_scip_experiment(
    experiment_config: ExperimentConfig,
    run_baseline: bool = True,
    run_pruned: bool = True,
) -> Dict[str, Any]:
    """
    Run complete SCIP experiment: prune dataset, train model, evaluate.
    
    This implements the full pipeline from the paper:
    1. Load The Stack dataset (Python subset)
    2. Compute embeddings with StarEncoder
    3. Run SCIP pruning algorithm
    4. Fine-tune models on original and pruned datasets
    5. Evaluate on HumanEval and MBPP
    6. Compare results
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Complete experiment configuration.
    run_baseline : bool
        Whether to train on unpruned baseline.
    run_pruned : bool
        Whether to train on SCIP-pruned dataset.
    
    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing all experiment results.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_config)
    
    logger.info("="*80)
    logger.info("Starting SCIP Experiment")
    logger.info(f"Experiment: {experiment_config.experiment_name}")
    logger.info(f"Pruning ratio: {experiment_config.pruning_ratio}")
    logger.info(f"Alpha: {experiment_config.alpha}")
    logger.info("="*80)
    
    # -------------------------------------------------------------------------
    # Step 1: Load dataset
    # -------------------------------------------------------------------------
    logger.info("\n[Step 1] Loading dataset...")
    data_config = DataConfig(
        dataset_name=experiment_config.dataset_name,
        language=experiment_config.language,
        max_samples=experiment_config.max_samples,
        tokenizer_name=experiment_config.model_name,
    )
    
    dataset = load_code_dataset_impl(data_config)
    logger.info(f"Loaded {len(dataset)} code samples")
    
    # -------------------------------------------------------------------------
    # Step 2: Compute embeddings
    # -------------------------------------------------------------------------
    logger.info("\n[Step 2] Computing embeddings...")
    
    # Extract code content from dataset
    code_texts = [item.get("content", item.get("code", "")) for item in dataset]
    
    # Compute embeddings using StarEncoder
    embeddings = compute_embeddings_impl(
        code_texts,
        model_name="bigcode/starencoder",
        device="cuda" if torch_module.cuda.is_available() else "cpu",
        batch_size=16,
        max_length=512,
    )
    
    logger.info(f"Computed embeddings: {embeddings.shape}")
    
    # -------------------------------------------------------------------------
    # Step 3: Run SCIP pruning
    # -------------------------------------------------------------------------
    logger.info("\n[Step 3] Running SCIP pruning...")
    pruning_config = PruningConfig(
        p=experiment_config.pruning_ratio,
        alpha=experiment_config.alpha,
        n_clusters=experiment_config.n_clusters,
        random_state=experiment_config.seed,
    )
    
    keep_idx, prune_idx, extra_info = scip_prune(embeddings, pruning_config)
    
    logger.info(f"Pruning complete:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Kept samples: {len(keep_idx)} ({100*len(keep_idx)/len(dataset):.1f}%)")
    logger.info(f"  Pruned samples: {len(prune_idx)} ({100*len(prune_idx)/len(dataset):.1f}%)")
    
    # Analyze pruning distribution
    pruning_stats = analyze_pruning_distribution(
        keep_idx,
        prune_idx,
        extra_info,
        output_dir=str(tracker.experiment_dir / "pruning_analysis"),
    )
    tracker.log_pruning_stats(pruning_stats)
    
    # Create pruned dataset
    pruned_dataset = create_pruned_dataset(dataset, keep_idx.tolist())
    
    # -------------------------------------------------------------------------
    # Step 4: Tokenize datasets
    # -------------------------------------------------------------------------
    logger.info("\n[Step 4] Tokenizing datasets...")
    
    max_seq_length = getattr(experiment_config, 'max_length', 512)
    logger.info(f"Using max sequence length: {max_seq_length}")
    
    tokenized_full = preprocess_code_for_training(
        dataset,
        experiment_config.model_name,
        max_length=max_seq_length,
    )
    
    tokenized_pruned = preprocess_code_for_training(
        pruned_dataset,
        experiment_config.model_name,
        max_length=max_seq_length,
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Fine-tune models
    # -------------------------------------------------------------------------
    results = {}
    
    # Setup model and training configs
    model_config = ModelConfig(
        model_name=experiment_config.model_name,
        use_flash_attention=False,  # Disable for compatibility
        use_lora=getattr(experiment_config, 'use_lora', True),
        lora_r=getattr(experiment_config, 'lora_r', 8),  # Smaller for speed
        lora_alpha=getattr(experiment_config, 'lora_alpha', 16),
        load_in_8bit=False,  # Disable 8-bit, use bfloat16 instead
        device_map="auto",  # Multi-GPU distribution
    )
    
    train_config = TrainConfig(
        output_dir=str(tracker.experiment_dir / "checkpoints"),
        max_steps=experiment_config.max_steps,
        learning_rate=experiment_config.learning_rate,
        per_device_train_batch_size=getattr(experiment_config, 'batch_size', 8),
        gradient_accumulation_steps=getattr(experiment_config, 'gradient_accumulation_steps', 2),
        use_small_config=experiment_config.max_samples is not None and experiment_config.max_samples < 10000,
    )
    
    # Train on baseline (no pruning)
    if run_baseline:
        logger.info("\n[Step 5a] Fine-tuning on full dataset (baseline)...")
        train_config.output_dir = str(tracker.experiment_dir / "checkpoints_baseline")
        
        model_baseline, trainer_baseline = finetune_model_impl(
            tokenized_full,
            model_config,
            train_config,
            run_name=f"{experiment_config.experiment_name}_baseline",
        )
        
        training_metrics_baseline = compute_training_metrics(trainer_baseline)
        
        # Evaluate baseline
        logger.info("\n[Step 6a] Evaluating baseline model...")
        eval_config = EvalConfig(
            temperature=experiment_config.eval_temperature,
            num_samples=experiment_config.num_samples,
        )
        
        tokenizer_baseline = trainer_baseline.tokenizer
        eval_results_baseline = evaluate_model_impl(
            model_baseline,
            tokenizer_baseline,
            eval_config,
        )
        
        results["baseline"] = {
            "training_stats": training_metrics_baseline,
            "evaluation_results": eval_results_baseline,
        }
        
        # Save baseline results
        save_evaluation_results(
            eval_results_baseline,
            str(tracker.experiment_dir / "baseline_eval.json"),
        )
        
        # IMPORTANT: Free baseline model from GPU memory before loading pruned model
        logger.info("Releasing baseline model from GPU memory...")
        del model_baseline
        del trainer_baseline
        import torch
        torch.cuda.empty_cache()
        logger.info("✓ Baseline model released")
    
    # Train on pruned dataset
    if run_pruned:
        logger.info("\n[Step 5b] Fine-tuning on SCIP-pruned dataset...")
        train_config.output_dir = str(tracker.experiment_dir / "checkpoints_pruned")
        
        model_pruned, trainer_pruned = finetune_model_impl(
            tokenized_pruned,
            model_config,
            train_config,
            run_name=f"{experiment_config.experiment_name}_pruned",
        )
        
        training_metrics_pruned = compute_training_metrics(trainer_pruned)
        
        # Evaluate pruned
        logger.info("\n[Step 6b] Evaluating SCIP-pruned model...")
        eval_config = EvalConfig(
            temperature=experiment_config.eval_temperature,
            num_samples=experiment_config.num_samples,
        )
        
        tokenizer_pruned = trainer_pruned.tokenizer
        eval_results_pruned = evaluate_model_impl(
            model_pruned,
            tokenizer_pruned,
            eval_config,
        )
        
        results["scip_pruned"] = {
            "training_stats": training_metrics_pruned,
            "evaluation_results": eval_results_pruned,
        }
        
        tracker.log_training_stats(training_metrics_pruned)
        tracker.log_evaluation_results(eval_results_pruned)
        
        # Save pruned results
        save_evaluation_results(
            eval_results_pruned,
            str(tracker.experiment_dir / "scip_pruned_eval.json"),
        )
    
    # -------------------------------------------------------------------------
    # Step 7: Compare results
    # -------------------------------------------------------------------------
    if run_baseline and run_pruned:
        logger.info("\n[Step 7] Comparing results...")
        compare_pruning_methods(
            results,
            output_dir=str(tracker.experiment_dir / "comparison"),
        )
    
    # Save all results
    tracker.save_results()
    
    logger.info("\n" + "="*80)
    logger.info("SCIP Experiment Complete!")
    logger.info(f"Results saved to: {tracker.experiment_dir}")
    logger.info("="*80)
    
    return results


def main():
    """
    Main entry point for SCIP code pruning experiments.
    
    This reproduces the experimental setup from:
        "Decoding Data Quality via Synthetic Corruptions:
         Embedding-guided Pruning of Code Data"
         (Yu Yang et al., ENLSP @ NeurIPS 2023)
    
    Usage:
        python main.py
    
    For testing on smaller hardware, set max_samples to a small value (e.g., 10000).
    For full reproduction, set max_samples=None to use the entire Stack dataset.
    """
    import argparse
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="SCIP Code Pruning Experiment")
    parser.add_argument("--experiment-name", type=str, default="scip_default",
                       help="Name for this experiment")
    parser.add_argument("--pruning-ratio", type=float, default=0.2,
                       help="Fraction of data to prune (default: 0.2)")
    parser.add_argument("--alpha", type=float, default=0.8,
                       help="Weight for cluster size pruning (default: 0.8)")
    parser.add_argument("--n-clusters", type=int, default=100,
                       help="Number of K-means clusters (default: 100)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples for testing (default: None = full dataset)")
    parser.add_argument("--max-steps", type=int, default=56000,
                       help="Training steps (default: 56000)")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Base model name")
    parser.add_argument("--no-baseline", action="store_true",
                       help="Skip baseline training")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                       help="Output directory for experiments")
    
    args = parser.parse_args()
    
    # Create experiment config
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        pruning_method="scip",
        pruning_ratio=args.pruning_ratio,
        alpha=args.alpha,
        n_clusters=args.n_clusters,
        max_samples=args.max_samples,
        model_name=args.model_name,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    
    # Run experiment
    results = run_scip_experiment(
        config,
        run_baseline=not args.no_baseline,
        run_pruned=True,
    )
    
    print("\nExperiment completed successfully!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()