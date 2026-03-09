"""
benchmark_finetune.py
Fine-tunes Flan-T5-base on a given dataset split (Coreset).
Outputs saved models for evaluation.
"""
import os
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from config import TRAINING_CONFIG, MODEL_NAME

def prepare_dataset(samples, tokenizer, max_input_length, max_target_length):
    # Support both "src" / "tgt" (WikiLarge) and "source" / "target" (CoEDIT)
    inputs = [s.get("src", s.get("source")) for s in samples]
    targets = [s.get("tgt", s.get("target")) for s in samples]
    
    dataset = Dataset.from_dict({
        "input_text": inputs,
        "target_text": targets,
    })
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        
        labels = tokenizer(
            examples["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        
        # Replace padding token id with -100
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True, help="Path to selected samples JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model (will save out_dir/final)")
    args = parser.parse_args()
    
    with open(args.samples, 'r') as f:
        samples = json.load(f)
        
    print(f"\n{'='*60}")
    print(f"FINE-TUNING")
    print(f"{'='*60}")
    
    # Split into train/val (90/10) with random permutation
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split = int(0.9 * len(samples))
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    train_dataset = prepare_dataset(
        train_samples, tokenizer,
        TRAINING_CONFIG["max_input_length"],
        TRAINING_CONFIG["max_target_length"],
    )
    val_dataset = prepare_dataset(
        val_samples, tokenizer,
        TRAINING_CONFIG["max_input_length"],
        TRAINING_CONFIG["max_target_length"],
    )
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        fp16=TRAINING_CONFIG["fp16"] and torch.cuda.is_available(),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        save_total_limit=1,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=TRAINING_CONFIG["max_target_length"],
        seed=42,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\nTraining complete in {train_time/60:.1f} minutes")
    print(f"Model saved to: {final_path}")

    timing = {
        "train_time": train_time,
        "n_samples": len(samples)
    }
    with open(output_dir / "finetune_timing.json", 'w') as f:
        json.dump(timing, f, indent=2)

if __name__ == "__main__":
    main()
