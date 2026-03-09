"""
WikiLarge Data Preparation
- Load WikiLarge dataset (~296K sentence pairs for text simplification)
- Convert to CoEDIT-compatible format (src, tgt, task)
- Generate embeddings using Sentence-T5 (same as CoEDIT pipeline)
- Cache for reuse
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import CACHE_DIR


# WikiLarge cache paths
WIKILARGE_DATASET_CACHE = CACHE_DIR / "wikilarge_dataset.json"
WIKILARGE_EMBEDDINGS_CACHE = CACHE_DIR / "wikilarge_embeddings.npy"

# Embedding model (same as CoEDIT pipeline)
EMBEDDING_MODEL = "sentence-transformers/sentence-t5-base"
EMBEDDING_BATCH_SIZE = 64


def load_wikilarge_dataset() -> Tuple[List[Dict], List[str]]:
    """
    Load WikiLarge dataset and convert to CoEDIT-compatible format.
    
    WikiLarge format: complex → simple sentence pairs
    CoEDIT format: src (with instruction prefix), tgt, task
    
    Returns:
        samples: List of {src, tgt, task} dicts
        texts_for_embedding: List of source texts for embedding
    """
    from datasets import load_dataset
    
    print("Loading WikiLarge dataset from HuggingFace...")
    # Try primary source
    try:
        ds = load_dataset("waboucay/wikilarge", split="train")
    except Exception as e:
        print(f"waboucay/wikilarge failed: {e}")
        print("Trying alternative source...")
        ds = load_dataset("bogdancazan/wikilarge-text-simplification", split="train")
    
    print(f"Loaded {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    
    # Show a sample to understand format
    print(f"\nSample entry:")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {str(v)[:100]}")
    
    # Convert to CoEDIT format
    samples = []
    texts_for_embedding = []
    
    # Detect column names (different HF versions use different names)
    src_col = None
    tgt_col = None
    for col in ds.column_names:
        col_lower = col.lower()
        if col_lower in ("original", "complex", "source", "src", "original_text"):
            src_col = col
        elif col_lower in ("simple", "simplified", "target", "tgt", "simple_text"):
            tgt_col = col
    
    if src_col is None or tgt_col is None:
        # Fallback: assume first two columns
        src_col, tgt_col = ds.column_names[0], ds.column_names[1]
        print(f"\nWarning: Auto-detected columns: src='{src_col}', tgt='{tgt_col}'")
    else:
        print(f"\nDetected columns: src='{src_col}', tgt='{tgt_col}'")
    
    for item in tqdm(ds, desc="Converting to CoEDIT format"):
        complex_text = item[src_col].strip()
        simple_text = item[tgt_col].strip()
        
        # Skip empty pairs
        if not complex_text or not simple_text:
            continue
        
        # CoEDIT-style instruction prefix for simplification
        src = f"Simplify this sentence: {complex_text}"
        
        sample = {
            "src": src,
            "tgt": simple_text,
            "task": "simplification",
        }
        samples.append(sample)
        texts_for_embedding.append(src)
    
    print(f"\nConverted {len(samples)} valid samples")
    print(f"Task: simplification (all samples)")
    
    return samples, texts_for_embedding


def generate_embeddings(
    texts: List[str],
    device: str = "cuda",
) -> np.ndarray:
    """Generate embeddings using Sentence-T5 (same model as CoEDIT pipeline)."""
    from sentence_transformers import SentenceTransformer
    
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    print(f"Batch size: {EMBEDDING_BATCH_SIZE}")
    
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def prepare_wikilarge(
    device: str = "cuda",
    force_regenerate: bool = False,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Full data preparation: load WikiLarge + generate/load embeddings.
    
    Returns:
        samples: List of {src, tgt, task} dicts
        embeddings: np.ndarray of shape (N, 768)
    """
    # Check cache
    if (not force_regenerate 
        and WIKILARGE_DATASET_CACHE.exists() 
        and WIKILARGE_EMBEDDINGS_CACHE.exists()):
        print("Loading cached WikiLarge data...")
        with open(WIKILARGE_DATASET_CACHE, 'r') as f:
            samples = json.load(f)
        embeddings = np.load(WIKILARGE_EMBEDDINGS_CACHE)
        print(f"Loaded {len(samples)} samples, embeddings shape: {embeddings.shape}")
        return samples, embeddings
    
    # Load and convert
    samples, texts = load_wikilarge_dataset()
    
    # Generate embeddings
    embeddings = generate_embeddings(texts, device=device)
    
    # Cache
    print("\nCaching WikiLarge data...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(WIKILARGE_DATASET_CACHE, 'w') as f:
        json.dump(samples, f)
    np.save(WIKILARGE_EMBEDDINGS_CACHE, embeddings)
    
    print(f"Cached to {WIKILARGE_DATASET_CACHE} and {WIKILARGE_EMBEDDINGS_CACHE}")
    
    return samples, embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare WikiLarge dataset")
    parser.add_argument("--device", default="cuda", help="Device for embedding generation")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    
    samples, embeddings = prepare_wikilarge(
        device=args.device,
        force_regenerate=args.force,
    )
    
    print(f"\n=== WikiLarge Data Summary ===")
    print(f"Total samples: {len(samples)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"\nSample entry:")
    print(f"  src: {samples[0]['src'][:120]}...")
    print(f"  tgt: {samples[0]['tgt'][:120]}...")
    print(f"  task: {samples[0]['task']}")
