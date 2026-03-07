"""
Utility functions and helper classes for SCIP experiments.
Includes experiment tracking, result visualization, and comparison utilities.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Complete configuration for a SCIP experiment."""
    # Pruning settings
    pruning_method: str = "scip"  # "scip", "random", "ssl_prototypes", "semdedup", "d4"
    pruning_ratio: float = 0.2
    alpha: float = 0.8
    n_clusters: int = 100
    
    # Dataset settings
    dataset_name: str = "bigcode/the-stack-dedup"
    language: str = "python"
    max_samples: Optional[int] = None
    streaming: bool = True  # Use streaming to avoid downloading full dataset
    max_length: int = 512  # Sequence length (reduced from 2048 for memory efficiency)
    
    # Model settings
    model_name: str = "meta-llama/CodeLlama-7b-Python-hf"  # CodeLlama specialized for Python
    use_lora: bool = True  # Use LoRA for efficient fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Training settings
    learning_rate: float = 3e-4
    batch_size: int = 8  # Per-device batch size (adjust based on GPU memory)
    gradient_accumulation_steps: int = 2  # Effective batch = batch_size * gradient_accumulation_steps * num_gpus
    max_steps: int = 56000
    
    # Evaluation settings
    eval_temperature: float = 0.8
    num_samples: int = 1
    
    # Experiment metadata
    experiment_name: str = "scip_default"
    output_dir: str = "./experiments"
    seed: int = 42


class ExperimentTracker:
    """Track and save experiment results."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize experiment tracker.
        
        Parameters
        ----------
        experiment_config : ExperimentConfig
            Configuration for this experiment.
        """
        self.config = experiment_config
        self.experiment_dir = Path(experiment_config.output_dir) / experiment_config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "config": asdict(experiment_config),
            "pruning_stats": {},
            "training_stats": {},
            "evaluation_results": {},
        }
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def log_pruning_stats(self, stats: Dict[str, Any]):
        """Log statistics from pruning phase."""
        self.results["pruning_stats"] = stats
        logger.info(f"Pruning stats: {stats}")
    
    def log_training_stats(self, stats: Dict[str, Any]):
        """Log statistics from training phase."""
        self.results["training_stats"] = stats
        logger.info(f"Training stats: {stats}")
    
    def log_evaluation_results(self, results: Dict[str, Any]):
        """Log evaluation results."""
        self.results["evaluation_results"] = results
        logger.info(f"Evaluation results: {results}")
    
    def save_results(self):
        """Save all results to JSON file."""
        output_file = self.experiment_dir / "results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get all tracked results."""
        return self.results


def compare_pruning_methods(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: str = "./comparisons",
):
    """
    Compare results from different pruning methods.
    
    Creates comparison tables and visualizations similar to Table 1 in the paper.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict[str, Any]]
        Dictionary mapping method name to its results.
        Example: {
            "no_pruning": {...},
            "random": {...},
            "scip": {...},
        }
    output_dir : str
        Directory to save comparison results.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    comparison_data = []
    for method, results in results_dict.items():
        eval_results = results.get("evaluation_results", {})
        comparison_data.append({
            "Method": method,
            "HumanEval pass@1": eval_results.get("humaneval_pass@1", 0) * 100,
            "MBPP pass@1": eval_results.get("mbpp_pass@1", 0) * 100,
            "Training Steps": results.get("training_stats", {}).get("training_steps", 0),
            "Tokens Seen": results.get("training_stats", {}).get("tokens_seen", 0),
        })
    
    # Create comparison table
    logger.info("\n" + "="*80)
    logger.info("Comparison of Pruning Methods (Table 1 style)")
    logger.info("="*80)
    logger.info(f"{'Method':<20} {'HumanEval':<15} {'MBPP':<15} {'Steps':<15}")
    logger.info("-"*80)
    
    for row in comparison_data:
        logger.info(
            f"{row['Method']:<20} "
            f"{row['HumanEval pass@1']:>12.2f}%  "
            f"{row['MBPP pass@1']:>12.2f}%  "
            f"{row['Training Steps']:>12}"
        )
    logger.info("="*80)
    
    # Save to JSON
    output_file = os.path.join(output_dir, "comparison.json")
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"Comparison saved to {output_file}")
    
    # Create visualization if matplotlib available
    try:
        visualize_comparison(comparison_data, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")


def visualize_comparison(
    comparison_data: List[Dict[str, Any]],
    output_dir: str,
):
    """
    Create bar charts comparing different pruning methods.
    
    Parameters
    ----------
    comparison_data : List[Dict[str, Any]]
        List of comparison data dictionaries.
    output_dir : str
        Directory to save plots.
    """
    methods = [d["Method"] for d in comparison_data]
    humaneval_scores = [d["HumanEval pass@1"] for d in comparison_data]
    mbpp_scores = [d["MBPP pass@1"] for d in comparison_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # HumanEval comparison
    ax1.bar(methods, humaneval_scores, color='skyblue')
    ax1.set_ylabel('Pass@1 (%)')
    ax1.set_title('HumanEval Performance')
    ax1.set_ylim([0, 100])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # MBPP comparison
    ax2.bar(methods, mbpp_scores, color='lightcoral')
    ax2.set_ylabel('Pass@1 (%)')
    ax2.set_title('MBPP Performance')
    ax2.set_ylim([0, 100])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_file}")
    plt.close()


def analyze_pruning_distribution(
    keep_indices: np.ndarray,
    prune_indices: np.ndarray,
    extra_info: Dict[str, np.ndarray],
    output_dir: str,
):
    """
    Analyze and visualize the distribution of pruned vs kept samples.
    
    Parameters
    ----------
    keep_indices : np.ndarray
        Indices of samples kept after pruning.
    prune_indices : np.ndarray
        Indices of samples pruned.
    extra_info : Dict[str, np.ndarray]
        Extra information from SCIP pruning (labels, distances, etc.).
    output_dir : str
        Directory to save analysis results.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    labels = extra_info["labels"]
    cluster_sizes = extra_info["cluster_sizes"]
    distances = extra_info["distances"]
    
    # Compute statistics
    stats = {
        "total_samples": len(labels),
        "kept_samples": len(keep_indices),
        "pruned_samples": len(prune_indices),
        "pruning_ratio": len(prune_indices) / len(labels),
        "num_clusters": len(cluster_sizes),
        "avg_cluster_size": float(np.mean(cluster_sizes)),
        "kept_avg_distance": float(np.mean(distances[keep_indices])),
        "pruned_avg_distance": float(np.mean(distances[prune_indices])),
    }
    
    logger.info("Pruning distribution analysis:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Save statistics
    with open(os.path.join(output_dir, "pruning_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Cluster size distribution
        axes[0, 0].hist(cluster_sizes, bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Cluster Size')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cluster Size Distribution')
        
        # Plot 2: Distance distribution for kept vs pruned
        axes[0, 1].hist(distances[keep_indices], bins=50, alpha=0.5, label='Kept', color='green')
        axes[0, 1].hist(distances[prune_indices], bins=50, alpha=0.5, label='Pruned', color='red')
        axes[0, 1].set_xlabel('Distance to Centroid')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distance Distribution')
        axes[0, 1].legend()
        
        # Plot 3: Pruning ratio per cluster
        keep_per_cluster = np.bincount(labels[keep_indices], minlength=len(cluster_sizes))
        prune_ratio_per_cluster = 1 - (keep_per_cluster / (cluster_sizes + 1e-10))
        axes[1, 0].scatter(cluster_sizes, prune_ratio_per_cluster, alpha=0.5)
        axes[1, 0].set_xlabel('Cluster Size')
        axes[1, 0].set_ylabel('Pruning Ratio')
        axes[1, 0].set_title('Pruning Ratio vs Cluster Size')
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        summary_text = "\n".join([
            "Summary Statistics",
            "-" * 40,
            f"Total Samples: {stats['total_samples']:,}",
            f"Kept: {stats['kept_samples']:,} ({100*(1-stats['pruning_ratio']):.1f}%)",
            f"Pruned: {stats['pruned_samples']:,} ({100*stats['pruning_ratio']:.1f}%)",
            f"Clusters: {stats['num_clusters']}",
            f"Avg Cluster Size: {stats['avg_cluster_size']:.1f}",
            "",
            "Average Distance to Centroid:",
            f"  Kept: {stats['kept_avg_distance']:.4f}",
            f"  Pruned: {stats['pruned_avg_distance']:.4f}",
        ])
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "pruning_analysis.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Pruning analysis plot saved to {output_file}")
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
    
    return stats


def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """
    Load experiment results from directory.
    
    Parameters
    ----------
    experiment_dir : str
        Path to experiment directory.
    
    Returns
    -------
    results : Dict[str, Any]
        Loaded experiment results.
    """
    results_file = Path(experiment_dir) / "results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


# Example usage
if __name__ == "__main__":
    # Create example experiment config
    config = ExperimentConfig(
        experiment_name="test_scip",
        pruning_ratio=0.2,
        alpha=0.8,
        max_samples=10000,
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(config)
    
    # Log some fake stats
    tracker.log_pruning_stats({
        "total_samples": 10000,
        "pruned_samples": 2000,
        "kept_samples": 8000,
    })
    
    tracker.log_training_stats({
        "training_steps": 56000,
        "tokens_seen": 67e9,
        "final_loss": 1.23,
    })
    
    tracker.log_evaluation_results({
        "humaneval_pass@1": 0.45,
        "mbpp_pass@1": 0.52,
    })
    
    # Save results
    tracker.save_results()
    
    print(f"Example results saved to {tracker.experiment_dir}")
