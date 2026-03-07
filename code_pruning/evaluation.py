"""
Evaluation on HumanEval and MBPP benchmarks.
Implements pass@k metric computation as described in the SCIP paper.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import tempfile
import subprocess
import signal
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for code benchmark evaluation."""
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512
    num_samples: int = 1  # k for pass@k
    batch_size: int = 1
    timeout: int = 10  # seconds for test execution
    

class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timing out code execution."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def load_humaneval_dataset() -> List[Dict[str, Any]]:
    """
    Load HumanEval benchmark dataset.
    
    HumanEval consists of 164 hand-crafted programming problems.
    Each problem has:
    - prompt: Function signature + docstring
    - canonical_solution: Reference implementation
    - test: Unit tests to verify correctness
    
    Returns
    -------
    problems : List[Dict]
        List of HumanEval problems.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")
        problems = [
            {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "entry_point": item["entry_point"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
            }
            for item in dataset
        ]
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems
    except Exception as e:
        logger.error(f"Failed to load HumanEval: {e}")
        logger.info("Please install: pip install datasets")
        return []


def load_mbpp_dataset(split: str = "test") -> List[Dict[str, Any]]:
    """
    Load MBPP (Mostly Basic Programming Problems) benchmark.
    
    MBPP contains ~1000 crowd-sourced Python programming problems.
    The paper uses 3-shot evaluation for MBPP.
    
    Parameters
    ----------
    split : str
        Dataset split ("test", "train", "validation").
    
    Returns
    -------
    problems : List[Dict]
        List of MBPP problems.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("mbpp", split=split)
        problems = [
            {
                "task_id": item["task_id"],
                "text": item["text"],
                "code": item["code"],
                "test_list": item["test_list"],
                "test_setup_code": item.get("test_setup_code", ""),
                "challenge_test_list": item.get("challenge_test_list", []),
            }
            for item in dataset
        ]
        logger.info(f"Loaded {len(problems)} MBPP problems ({split})")
        return problems
    except Exception as e:
        logger.error(f"Failed to load MBPP: {e}")
        logger.info("Please install: pip install datasets")
        return []


def generate_code_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    config: EvalConfig,
) -> List[str]:
    """
    Generate code completions for a given prompt.
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        Fine-tuned code model.
    tokenizer : AutoTokenizer
        Model tokenizer.
    prompt : str
        Input prompt (function signature + docstring).
    config : EvalConfig
        Generation configuration.
    
    Returns
    -------
    completions : List[str]
        List of generated code completions.
    """
    device = model.device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate multiple samples
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            num_return_sequences=config.num_samples,
            do_sample=True if config.num_samples > 1 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode outputs
    completions = []
    for output in outputs:
        # Remove prompt from generated text
        generated_ids = output[inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)
    
    return completions


def test_code_execution(
    code: str,
    test_cases: str,
    timeout: int = 10,
) -> bool:
    """
    Execute code with test cases in a sandboxed environment.
    
    Parameters
    ----------
    code : str
        Generated code to test.
    test_cases : str
        Test cases to run.
    timeout : int
        Maximum execution time in seconds.
    
    Returns
    -------
    passed : bool
        True if all tests passed, False otherwise.
    """
    # Combine code and tests
    full_code = code + "\n\n" + test_cases
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        # Run code with timeout
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        
        # Check if execution succeeded
        passed = result.returncode == 0
        
        if not passed and result.stderr:
            logger.debug(f"Test failed: {result.stderr[:200]}")
        
        return passed
        
    except subprocess.TimeoutExpired:
        logger.debug("Test execution timed out")
        return False
    except Exception as e:
        logger.debug(f"Test execution error: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def evaluate_humaneval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: EvalConfig,
) -> Dict[str, float]:
    """
    Evaluate model on HumanEval benchmark (zero-shot).
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        Fine-tuned code model.
    tokenizer : AutoTokenizer
        Model tokenizer.
    config : EvalConfig
        Evaluation configuration.
    
    Returns
    -------
    metrics : Dict[str, float]
        Evaluation metrics including pass@k.
    """
    logger.info("Evaluating on HumanEval (zero-shot)...")
    
    problems = load_humaneval_dataset()
    if not problems:
        return {"humaneval_pass@1": 0.0}
    
    results = []
    
    for problem in tqdm(problems, desc="HumanEval"):
        prompt = problem["prompt"]
        test_cases = problem["test"]
        entry_point = problem["entry_point"]
        
        # Generate completions
        completions = generate_code_completion(model, tokenizer, prompt, config)
        
        # Test each completion
        passed_any = False
        for completion in completions:
            # Combine prompt + completion
            full_code = prompt + completion
            
            # Check function exists
            if entry_point not in full_code:
                continue
            
            # Run tests
            passed = test_code_execution(
                full_code,
                test_cases,
                timeout=config.timeout,
            )
            
            if passed:
                passed_any = True
                break  # At least one solution passed
        
        results.append(passed_any)
    
    # Compute pass@k
    pass_at_k = sum(results) / len(results) if results else 0.0
    
    metrics = {
        "humaneval_pass@1": pass_at_k,
        "humaneval_total": len(results),
        "humaneval_passed": sum(results),
    }
    
    logger.info(f"HumanEval Results: {metrics}")
    return metrics


def evaluate_mbpp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: EvalConfig,
    num_shots: int = 3,
) -> Dict[str, float]:
    """
    Evaluate model on MBPP benchmark (few-shot).
    
    The paper uses 3-shot evaluation for MBPP.
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        Fine-tuned code model.
    tokenizer : AutoTokenizer
        Model tokenizer.
    config : EvalConfig
        Evaluation configuration.
    num_shots : int
        Number of examples in prompt (paper uses 3).
    
    Returns
    -------
    metrics : Dict[str, float]
        Evaluation metrics including pass@k.
    """
    logger.info(f"Evaluating on MBPP ({num_shots}-shot)...")
    
    # Load train examples for few-shot prompting
    train_problems = load_mbpp_dataset("train")
    test_problems = load_mbpp_dataset("test")
    
    if not test_problems:
        return {"mbpp_pass@1": 0.0}
    
    # Select few-shot examples
    few_shot_examples = train_problems[:num_shots] if train_problems else []
    
    results = []
    
    for problem in tqdm(test_problems, desc="MBPP"):
        # Construct few-shot prompt
        prompt_parts = []
        
        # Add few-shot examples
        for ex in few_shot_examples:
            prompt_parts.append(f"# Problem: {ex['text']}")
            prompt_parts.append(ex['code'])
            prompt_parts.append("")
        
        # Add current problem
        prompt_parts.append(f"# Problem: {problem['text']}")
        prompt = "\n".join(prompt_parts)
        
        # Generate completions
        completions = generate_code_completion(model, tokenizer, prompt, config)
        
        # Test each completion
        passed_any = False
        for completion in completions:
            # Combine setup code + completion + tests
            test_code_parts = [problem.get("test_setup_code", "")]
            test_code_parts.append(completion)
            test_code_parts.extend(problem["test_list"])
            
            full_test_code = "\n".join(test_code_parts)
            
            # Run tests
            passed = test_code_execution(
                completion,
                full_test_code,
                timeout=config.timeout,
            )
            
            if passed:
                passed_any = True
                break
        
        results.append(passed_any)
    
    # Compute pass@k
    pass_at_k = sum(results) / len(results) if results else 0.0
    
    metrics = {
        "mbpp_pass@1": pass_at_k,
        "mbpp_total": len(results),
        "mbpp_passed": sum(results),
    }
    
    logger.info(f"MBPP Results: {metrics}")
    return metrics


def evaluate_model_on_code_benchmarks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate fine-tuned model on HumanEval and MBPP benchmarks.
    
    This is the main evaluation function matching the paper's setup:
    - HumanEval: zero-shot
    - MBPP: 3-shot
    - Metric: pass@1 (and optionally pass@k for k > 1)
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        Fine-tuned code model.
    tokenizer : AutoTokenizer
        Model tokenizer.
    config : Optional[EvalConfig]
        Evaluation configuration.
    
    Returns
    -------
    metrics : Dict[str, Any]
        Combined evaluation metrics from both benchmarks.
    """
    if config is None:
        config = EvalConfig()
    
    model.eval()
    
    # Evaluate on HumanEval
    humaneval_metrics = evaluate_humaneval(model, tokenizer, config)
    
    # Evaluate on MBPP
    mbpp_metrics = evaluate_mbpp(model, tokenizer, config, num_shots=3)
    
    # Combine metrics
    all_metrics = {**humaneval_metrics, **mbpp_metrics}
    
    logger.info("=" * 60)
    logger.info("Final Evaluation Results:")
    logger.info(f"  HumanEval pass@1: {all_metrics.get('humaneval_pass@1', 0):.2%}")
    logger.info(f"  MBPP pass@1: {all_metrics.get('mbpp_pass@1', 0):.2%}")
    logger.info("=" * 60)
    
    return all_metrics


def save_evaluation_results(
    metrics: Dict[str, Any],
    output_file: str,
):
    """
    Save evaluation results to JSON file.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Evaluation metrics.
    output_file : str
        Output file path.
    """
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    from training import setup_model_and_tokenizer, ModelConfig
    
    # Load model
    model_config = ModelConfig(model_name="meta-llama/CodeLlama-7b-Python-hf")
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # Evaluate
    eval_config = EvalConfig(num_samples=1)
    metrics = evaluate_model_on_code_benchmarks(model, tokenizer, eval_config)
    
    # Save results
    save_evaluation_results(metrics, "evaluation_results.json")
