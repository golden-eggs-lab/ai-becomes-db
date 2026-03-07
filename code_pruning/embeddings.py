from typing import Sequence, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def compute_embeddings_for_dataset(
    dataset: Sequence[str],
    model_name: str = "bigcode/starencoder",
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute code embeddings for a dataset using a code encoder (e.g., StarEncoder).

    This is the "compute" step the paper assumes:
        D  --(StarEncoder)-->  embeddings of shape (N, D)
        embeddings then go into SCIP pruning.

    In the paper:
        - They use StarEncoder [Li et al. 2023] as the code embedding model.
        - Dataset is the Stack v1.1 (Python subset), but this function is generic:
          as long as `dataset` is a list/sequence of code strings, it works.
        - Embeddings are l2-normalized before clustering (we'll normalize in
          `scip_prune`, so here we just output raw vectors).

    If you want to be *very close* to the paper setup:
        TODO:
        - Upstream: load "bigcode/the-stack" dataset from HuggingFace.
        - Filter language == "python".
        - Pass the list of Python files' contents into this function.

    Parameters
    ----------
    dataset : Sequence[str]
        A sequence (list, HF Dataset column, etc.) of code strings. Length N.
    model_name : str
        HuggingFace model id for the encoder. In the paper this should be
        something like "bigcode/starencoder".
    device : str
        "cuda" or "cpu".
    batch_size : int
        Batch size for embedding computation.
    max_length : int
        Maximum sequence length for the encoder. Long code will be truncated.

    Returns
    -------
    embeddings : np.ndarray
        A 2D numpy array of shape (N, D), where D is the encoder hidden size.
    """
    # Load tokenizer & model once
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Warning: Failed to load {model_name}: {e}")
        print("Falling back to lighter model: microsoft/codebert-base")
        # Fallback to a lighter code model
        model_name = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()


    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch_texts = dataset[start:start + batch_size]

            # Tokenize a batch of code snippets
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)

            # ----- Pooling strategy -----
            # The paper does not specify pooling details; here we use simple
            # mean pooling over the sequence dimension. You can switch to
            # CLS token, last hidden state, etc. if StarEncoder has a convention.
            #
            # TODO: if StarEncoder has a recommended pooling method, replace
            # this with that.
            hidden_states = outputs.last_hidden_state  # (B, T, H)
            mask = inputs.attention_mask.unsqueeze(-1)  # (B, T, 1)
            masked_hidden = hidden_states * mask
            sum_hidden = masked_hidden.sum(dim=1)            # (B, H)
            lengths = mask.sum(dim=1).clamp(min=1)           # (B, 1)
            batch_emb = (sum_hidden / lengths).cpu().numpy()  # (B, H)

            all_embeddings.append(batch_emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    assert embeddings.shape[0] == len(dataset), "Embedding count mismatch"

    return embeddings