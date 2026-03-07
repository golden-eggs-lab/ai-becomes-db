"""
Fine-tuning pipeline for code LLM on pruned datasets.
Implements training setup from SCIP paper with FSDP support.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset
import os

# Optional wandb import - can be disabled via environment variable
try:
    if os.environ.get("WANDB_DISABLED", "false").lower() != "true":
        import wandb
    else:
        wandb = None
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the code LLM."""
    model_name: str = "meta-llama/CodeLlama-7b-Python-hf"  # CodeLlama specialized for Python
    # Paper uses 1.5B LLaMA with 48 layers, 24 heads, hidden_size=1536
    # Note: CodeLlama-7b-Python-hf requires HuggingFace login and Meta license agreement
    use_flash_attention: bool = False  # Disable flash attention for compatibility
    load_in_8bit: bool = False  # Disable 8-bit if bitsandbytes upgrade fails
    device_map: str = "auto"  # Auto-distribute across GPUs
    torch_dtype: str = "bfloat16"  # Use bfloat16 for better training stability
    
    # LoRA configuration (for efficient fine-tuning)
    use_lora: bool = True  # Enable LoRA for faster training
    lora_r: int = 8  # Smaller rank for faster training
    lora_alpha: int = 16  # Smaller alpha
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # None = auto-detect, or ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainConfig:
    """
    Training configuration matching SCIP paper setup.
    
    Paper settings:
        - Learning rate: 3e-4
        - Batch size: 576 (distributed across 32 A100 GPUs)
        - Sequence length: 2048
        - Training steps: 56,000 (~67B tokens)
        - Optimizer: AdamW
        - Strategy: Fully Sharded Data Parallel (FSDP)
    """
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 1
    max_steps: int = 56000  # Paper uses 56k steps
    per_device_train_batch_size: int = 2  # Adjust based on your GPU memory
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16  # To simulate larger batch size
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    logging_steps: int = 100
    save_steps: int = 5000
    eval_steps: int = 5000
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 for A100/H100
    gradient_checkpointing: bool = False  # Disable for speed with LoRA
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "none"  # or "tensorboard", "wandb"
    
    # FSDP settings (for multi-GPU training)
    fsdp: str = ""  # "full_shard auto_wrap" for FSDP
    fsdp_config: Optional[Dict] = None
    
    # For testing on smaller hardware
    use_small_config: bool = False  # Set True for quick testing


def setup_model_and_tokenizer(config: ModelConfig):
    """
    Load pre-trained code LLM and tokenizer.
    
    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    
    Returns
    -------
    model : AutoModelForCausalLM
        Pre-trained code language model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    """
    logger.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine torch dtype
    torch_dtype = getattr(torch, config.torch_dtype, torch.float32)
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": config.device_map,
    }
    
    if config.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    if config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
    except Exception as e:
        logger.warning(f"Failed to load with flash attention: {e}")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
    
    # Apply LoRA if enabled
    if config.use_lora:
        logger.info("Applying LoRA for efficient fine-tuning...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            
            # Prepare model for training (important for gradient checkpointing compatibility)
            model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,  # None = auto-detect
                bias="none",
            )
            
            model = get_peft_model(model, lora_config)
            
            # Enable gradient checkpointing for LoRA layers
            model.enable_input_require_grads()
            
            model.print_trainable_parameters()
            logger.info("✓ LoRA applied successfully")
        except ImportError:
            logger.warning("peft library not found. Install with: pip install peft")
            logger.warning("Falling back to full fine-tuning (slower)")
    
    logger.info(f"Model loaded: {model.config}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params / 1e9:.2f}B / {total_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def finetune_model_on_dataset(
    dataset: Dataset,
    model_config: ModelConfig,
    train_config: TrainConfig,
    eval_dataset: Optional[Dataset] = None,
    run_name: Optional[str] = None,
) -> tuple:
    """
    Fine-tune a code LLM on the given dataset.
    
    This implements the training setup from the SCIP paper:
    - Base model: Code LLaMA or similar
    - Training: 56k steps with lr=3e-4, batch_size=576
    - Parallelism: FSDP for multi-GPU training
    
    Parameters
    ----------
    dataset : Dataset
        Training dataset (tokenized).
    model_config : ModelConfig
        Model configuration.
    train_config : TrainConfig
        Training hyperparameters.
    eval_dataset : Optional[Dataset]
        Validation dataset.
    run_name : Optional[str]
        Name for the training run (for logging).
    
    Returns
    -------
    model : AutoModelForCausalLM
        Fine-tuned model.
    trainer : Trainer
        HuggingFace Trainer object (for saving, evaluation).
    """
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        max_steps=train_config.max_steps,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=train_config.warmup_steps,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",  # Updated parameter name
        eval_steps=train_config.eval_steps if eval_dataset else None,
        save_total_limit=train_config.save_total_limit,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        dataloader_num_workers=train_config.dataloader_num_workers,
        remove_unused_columns=train_config.remove_unused_columns,
        report_to=train_config.report_to,
        run_name=run_name or "scip_finetuning",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        fsdp=train_config.fsdp,
        ddp_find_unused_parameters=False,
    )
    
    # Override for small testing config
    if train_config.use_small_config:
        training_args.max_steps = 1000
        training_args.save_steps = 500
        training_args.eval_steps = 500
        logger.info("Using small config for testing")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Log training info
    effective_batch_size = (
        train_config.per_device_train_batch_size 
        * train_config.gradient_accumulation_steps
        * torch.cuda.device_count()
    )
    logger.info(f"Starting training:")
    logger.info(f"  Num examples: {len(dataset)}")
    logger.info(f"  Num epochs: {train_config.num_train_epochs}")
    logger.info(f"  Max steps: {train_config.max_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Learning rate: {train_config.learning_rate}")
    
    # Train
    trainer.train()
    
    logger.info("Training completed!")
    
    # Save final model
    final_output_dir = os.path.join(train_config.output_dir, "final_model")
    trainer.save_model(final_output_dir)
    logger.info(f"Model saved to {final_output_dir}")
    
    return model, trainer


def compute_training_metrics(trainer: Trainer) -> Dict[str, Any]:
    """
    Extract training metrics from trainer.
    
    Parameters
    ----------
    trainer : Trainer
        HuggingFace trainer object.
    
    Returns
    -------
    metrics : Dict[str, Any]
        Dictionary of training metrics.
    """
    metrics = {
        "training_steps": trainer.state.global_step,
        "training_loss": trainer.state.log_history[-1].get("loss", None),
    }
    
    # Estimate tokens seen
    if hasattr(trainer.args, "per_device_train_batch_size"):
        tokens_per_batch = (
            trainer.args.per_device_train_batch_size 
            * 2048  # sequence length
            * trainer.args.gradient_accumulation_steps
            * torch.cuda.device_count()
        )
        metrics["tokens_seen"] = tokens_per_batch * trainer.state.global_step
    
    return metrics


# Example usage
if __name__ == "__main__":
    from data_loading import load_code_dataset, preprocess_code_for_training, DataConfig
    
    # Load small dataset for testing
    data_config = DataConfig(max_samples=1000)
    dataset = load_code_dataset(data_config)
    tokenized_dataset = preprocess_code_for_training(
        dataset,
        data_config.tokenizer_name,
        data_config.max_length,
    )
    
    # Setup configs
    model_config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        use_flash_attention=False,  # Disable for testing
    )
    
    train_config = TrainConfig(
        output_dir="./test_checkpoints",
        use_small_config=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
    )
    
    # Train
    model, trainer = finetune_model_on_dataset(
        tokenized_dataset,
        model_config,
        train_config,
        run_name="test_run",
    )
    
    metrics = compute_training_metrics(trainer)
    print(f"Training metrics: {metrics}")
