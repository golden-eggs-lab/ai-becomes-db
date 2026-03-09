"""
Re-evaluate saved baseline and optimized models with proper metrics.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from config import ARTIFACTS_DIR


def compute_sari_simple(sources, predictions, references):
    """
    Simple SARI implementation without easse dependency.
    SARI = (keep_score + add_score + delete_score) / 3
    Uses 1-4 grams like the proper SARI metric.
    """
    from collections import Counter
    import sacrebleu
    
    def get_ngrams(text, n):
        words = text.lower().split()
        return Counter([tuple(words[i:i+n]) for i in range(len(words)-n+1)])
    
    def precision(sys_ngrams, ref_ngrams):
        if len(sys_ngrams) == 0:
            return 0
        overlap = sum((sys_ngrams & ref_ngrams).values())
        return overlap / sum(sys_ngrams.values())
    
    def recall(sys_ngrams, ref_ngrams):
        if len(ref_ngrams) == 0:
            return 0
        overlap = sum((sys_ngrams & ref_ngrams).values())
        return overlap / sum(ref_ngrams.values())
    
    def f1(p, r):
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)
    
    sari_scores = []
    
    for src, pred, refs in zip(sources, predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        
        keep_scores = []
        add_scores = []
        del_scores = []
        
        for n in range(1, 5):  # 1-4 grams
            src_ngrams = get_ngrams(src, n)
            pred_ngrams = get_ngrams(pred, n)
            
            # Average over all references
            ref_keep_scores = []
            ref_add_scores = []
            ref_del_scores = []
            
            for ref in refs:
                ref_ngrams = get_ngrams(ref, n)
                
                # Keep: ngrams in both src and pred that are also in ref
                src_pred_common = src_ngrams & pred_ngrams
                keep_p = precision(src_pred_common, ref_ngrams)
                keep_r = recall(src_pred_common, src_ngrams & ref_ngrams)
                ref_keep_scores.append(f1(keep_p, keep_r))
                
                # Add: ngrams in pred but not in src, that are in ref
                pred_only = pred_ngrams - src_ngrams
                ref_only = ref_ngrams - src_ngrams
                add_p = precision(pred_only, ref_only)
                ref_add_scores.append(add_p)
                
                # Delete: ngrams in src but not in pred, that are not in ref
                src_only = src_ngrams - pred_ngrams
                not_in_ref = src_ngrams - ref_ngrams
                del_p = precision(src_only, not_in_ref)
                ref_del_scores.append(del_p)
            
            keep_scores.append(np.mean(ref_keep_scores))
            add_scores.append(np.mean(ref_add_scores))
            del_scores.append(np.mean(ref_del_scores))
        
        # Average over n-grams
        keep = np.mean(keep_scores)
        add = np.mean(add_scores)
        delete = np.mean(del_scores)
        
        sari = (keep + add + delete) / 3 * 100
        sari_scores.append(sari)
    
    return np.mean(sari_scores)


def compute_rouge_l(predictions, references):
    """Compute ROUGE-L score (LCS-based)."""
    def lcs_length(x, y):
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    def rouge_l_sentence(pred, ref):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)
    
    scores = []
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        # Take max ROUGE-L across references
        ref_scores = [rouge_l_sentence(pred, ref) for ref in refs]
        scores.append(max(ref_scores))
    
    return np.mean(scores) * 100


def compute_bleu(predictions, references):
    """Compute BLEU using sacrebleu."""
    import sacrebleu
    
    # Transpose references for sacrebleu format
    max_refs = max(len(refs) for refs in references)
    refs_transposed = []
    for ref_idx in range(max_refs):
        ref_list = []
        for refs in references:
            if ref_idx < len(refs):
                ref_list.append(refs[ref_idx])
            else:
                ref_list.append(refs[0])
        refs_transposed.append(ref_list)
    
    bleu = sacrebleu.corpus_bleu(predictions, refs_transposed, force=True)
    return bleu.score


def load_test_data():
    """Load TurkCorpus test data."""
    print("Loading TurkCorpus test set...")
    turk = load_dataset("turk", split="test")
    
    test_data = []
    for item in turk:
        test_data.append({
            "source": item["original"],
            "references": item["simplifications"]
        })
    
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def evaluate_model(model_path, test_data, setting_name):
    """Evaluate a saved model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {setting_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
        batch_size = 8 * n_gpus
        print(f"Using {n_gpus} GPUs, batch_size={batch_size}")
    else:
        model.to(device)
        batch_size = 8
        print(f"Using 1 GPU, batch_size={batch_size}")
    
    model.eval()
    
    predictions = []
    sources = [item["source"] for item in test_data]
    
    gen_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sources), batch_size), desc="Generating"):
            batch_sources = [f"Simplify: {s}" for s in sources[i:i+batch_size]]
            inputs = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            
            outputs = gen_model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )
            
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)
    
    # Compute metrics
    references = [item["references"] for item in test_data]
    
    # BLEU (using sacrebleu)
    bleu_score = compute_bleu(predictions, references)
    
    # SARI
    sari_score = compute_sari_simple(sources, predictions, references)
    
    # ROUGE-L (multi-ref, take max)
    rouge_l = compute_rouge_l(predictions, references)
    
    # FKGL
    try:
        import textstat
        fkgl_scores = [textstat.flesch_kincaid_grade(pred) for pred in predictions]
        fkgl = np.mean(fkgl_scores)
    except:
        print("Warning: textstat not available, skipping FKGL")
        fkgl = 0.0
    
    results = {
        "setting": setting_name,
        "bleu": bleu_score,
        "sari": sari_score,
        "rouge_l": rouge_l,
        "fkgl": fkgl,
    }
    
    print(f"\n--- Results for {setting_name} ---")
    print(f"BLEU:    {bleu_score:.2f}")
    print(f"SARI:    {sari_score:.2f}")
    print(f"ROUGE-L: {rouge_l:.2f}")
    print(f"FKGL:    {fkgl:.2f}")
    
    return results


    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the directory containing the model (e.g., artifacts/benchmark_e2e/coedit_model_baseline_run1)")
    
    args = parser.parse_args()
    
    experiment_path = Path(args.model_dir)
    
    if not experiment_path.exists():
        print(f"Error: {experiment_path} not found")
        return
    
    final_model = experiment_path / "final"
    
    if not final_model.exists():
        print(f"Error: Final model not found at {final_model}")
        return
    
    # Load test data
    test_data = load_test_data()
    
    # Evaluate model
    results = evaluate_model(str(final_model), test_data, "evaluated_model")
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for metric in ["bleu", "sari", "rouge_l", "fkgl"]:
        val = results[metric]
        print(f"{metric.upper():<15} {val:>12.2f}")
    
    # Save results
    results_path = experiment_path / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
