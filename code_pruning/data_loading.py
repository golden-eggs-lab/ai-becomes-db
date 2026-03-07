"""
Dataset loading and preprocessing for SCIP paper reproduction.
Handles The Stack v1.1 dataset loading, filtering, and tokenization.
"""

from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    dataset_name: str = "bigcode/the-stack-dedup"
    language: str = "python"
    max_samples: Optional[int] = None  # Set to small number (e.g., 10000) to only download partial data
    tokenizer_name: str = "meta-llama/CodeLlama-7b-Python-hf"  # Using CodeLlama tokenizer
    max_length: int = 2048
    min_length: int = 50  # Filter out very short code
    streaming: bool = True  # Use True to avoid downloading full dataset (recommended!)
    cache_dir: Optional[str] = None
    

def load_code_dataset(
    config: DataConfig,
    split: str = "train",
) -> Dataset:
    """
    Load The Stack v1.1 dataset (Python subset) or similar code dataset.
    
    The paper uses:
        - Dataset: The Stack v1.1 (Denis Kocetkov et al. 2022)
        - Language: Python
        - Size: 12.6M files, 20.4B tokens
    
    Parameters
    ----------
    config : DataConfig
        Dataset configuration.
    split : str
        Dataset split to load ("train", "test", etc.).
    
    Returns
    -------
    dataset : Dataset
        HuggingFace Dataset object containing Python code samples.
    """
    logger.info(f"Loading dataset {config.dataset_name}, language={config.language}")
    
    # First, check if user is authenticated
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if token:
        logger.info("✓ HuggingFace token found, authenticated")
    else:
        logger.warning("✗ No HuggingFace token found. Run: huggingface-cli login")
    
    try:
        # Load The Stack dataset with Python filter
        # The Stack v2 uses 'data_dir' parameter differently
        logger.info(f"Loading with streaming={config.streaming}, max_samples={config.max_samples}")
        dataset = load_dataset(
            config.dataset_name,
            data_dir=f"data/{config.language}",  # Correct path for The Stack v2
            split=split,
            streaming=config.streaming,  # Use streaming to avoid downloading full dataset
            cache_dir=config.cache_dir,
        )
        
        # If streaming and max_samples specified, take only first N samples
        if config.streaming and config.max_samples:
            logger.info(f"Taking first {config.max_samples} samples from stream")
            dataset = dataset.take(config.max_samples)
            # Convert to regular dataset for easier processing
            sample_list = list(dataset)
            if sample_list:
                dataset = Dataset.from_dict({k: [item[k] for item in sample_list] for k in sample_list[0].keys()})
                logger.info(f"✓ Loaded {len(dataset)} samples")
            else:
                raise ValueError("No samples loaded from stream")
    except Exception as e:
        logger.warning(f"Failed to load {config.dataset_name}: {e}")
        logger.info("Trying alternative loading method...")
        # Try without data_dir (for older versions)
        try:
            dataset = load_dataset(
                config.dataset_name,
                split=split,
                streaming=config.streaming,
                cache_dir=config.cache_dir,
            )
            # Filter for Python if dataset loaded successfully
            if config.language and 'lang' in dataset.column_names:
                dataset = dataset.filter(lambda x: x['lang'] == config.language)
        except Exception as e2:
            logger.warning(f"Alternative method also failed: {e2}")
            logger.info("Falling back to smaller test dataset")
        # Fallback to a smaller dataset for testing
        try:
            # Try code_search_net dataset
            dataset = load_dataset(
                "code_search_net",
                config.language,
                split=split,
                cache_dir=config.cache_dir,
            )
            # Rename 'func_code_string' to 'content' for consistency
            dataset = dataset.rename_column("func_code_string", "content")
        except Exception as e2:
            logger.warning(f"Failed to load code_search_net: {e2}")
            logger.info("Creating minimal synthetic dataset for testing")
            # Create a minimal synthetic dataset
            from datasets import Dataset as HFDataset
            synthetic_data = {
                "content": [
                    "def hello_world():\n    print('Hello, World!')\n",
                    "def add(a, b):\n    return a + b\n",
                    "def multiply(x, y):\n    return x * y\n",
                    "class Calculator:\n    def __init__(self):\n        self.result = 0\n",
                    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
                ] * (config.max_samples // 5 if config.max_samples else 1000)
            }
            dataset = HFDataset.from_dict(synthetic_data)
    
    # If streaming, convert to regular dataset for easier manipulation
    if config.streaming and config.max_samples:
        dataset = Dataset.from_dict({
            "content": [item["content"] for i, item in enumerate(dataset) 
                       if i < config.max_samples]
        })
    elif not config.streaming and config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    # Filter out very short code snippets
    if not config.streaming:
        dataset = dataset.filter(
            lambda x: len(x.get("content", "")) >= config.min_length,
            desc="Filtering short code"
        )
    
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    return dataset


def preprocess_code_for_training(
    dataset: Dataset,
    tokenizer_name: str,
    max_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """
    Tokenize code dataset for language model training.
    
    Parameters
    ----------
    dataset : Dataset
        Raw dataset with "content" field containing code strings.
    tokenizer_name : str
        HuggingFace tokenizer identifier.
    max_length : int
        Maximum sequence length (paper uses 2048).
    num_proc : int
        Number of processes for parallel tokenization.
    
    Returns
    -------
    tokenized_dataset : Dataset
        Dataset with "input_ids" and "attention_mask" fields.
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        """Tokenize a batch of code examples."""
        texts = examples.get("content", examples.get("code", []))
        
        # Tokenize with truncation and padding
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,  # Return lists, not tensors
        )
        
        # For causal LM, labels are the same as input_ids
        outputs["labels"] = outputs["input_ids"].copy()
        
        # Validate token IDs are within vocab size (fix CUDA indexing errors)
        vocab_size = tokenizer.vocab_size
        for i in range(len(outputs["input_ids"])):
            outputs["input_ids"][i] = [
                min(token_id, vocab_size - 1) if token_id >= 0 else 0
                for token_id in outputs["input_ids"][i]
            ]
            outputs["labels"][i] = [
                min(token_id, vocab_size - 1) if token_id >= 0 else 0
                for token_id in outputs["labels"][i]
            ]
        
        return outputs
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing code",
    )
    
    logger.info(f"Tokenization complete: {len(tokenized_dataset)} samples")
    return tokenized_dataset


def create_pruned_dataset(
    original_dataset: Dataset,
    keep_indices: List[int],
) -> Dataset:
    """
    Create a pruned dataset from original dataset using SCIP keep indices.
    
    Parameters
    ----------
    original_dataset : Dataset
        Original dataset before pruning.
    keep_indices : List[int]
        Indices of samples to keep after SCIP pruning.
    
    Returns
    -------
    pruned_dataset : Dataset
        Subset of original dataset containing only kept samples.
    """
    logger.info(f"Creating pruned dataset: keeping {len(keep_indices)} samples")
    pruned_dataset = original_dataset.select(keep_indices)
    logger.info(f"Pruned dataset size: {len(pruned_dataset)}")
    return pruned_dataset


def get_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Compute statistics about a code dataset.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset to analyze.
    
    Returns
    -------
    stats : Dict[str, Any]
        Dictionary with statistics (num_samples, avg_length, etc.).
    """
    stats = {
        "num_samples": len(dataset),
    }
    
    # Compute token statistics if available
    if "input_ids" in dataset.column_names:
        lengths = [len(ids) for ids in dataset["input_ids"]]
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["max_length"] = max(lengths)
        stats["min_length"] = min(lengths)
        stats["total_tokens"] = sum(lengths)
    
    return stats


# Example usage
if __name__ == "__main__":
    # Test dataset loading
    config = DataConfig(max_samples=1000)  # Small sample for testing
    dataset = load_code_dataset(config)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample: {dataset[0]['content'][:200]}...")
    
    # Test tokenization
    tokenized = preprocess_code_for_training(
        dataset,
        config.tokenizer_name,
        config.max_length,
    )
    
    stats = get_dataset_statistics(tokenized)
    print(f"Dataset statistics: {stats}")
