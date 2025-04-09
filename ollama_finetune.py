"""
Fine-tune a model with Ollama using TD-inspired reward.

This script implements a fine-tuning approach for Ollama models, using the
TD-inspired reward function to guide the training process.
"""

import os
import argparse
import json
import torch
import numpy as np
import requests
import time
from tqdm import tqdm
import subprocess
import logging

from td_reward import TDRewardModel


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama_finetune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an Ollama model with TD-inspired reward")
    parser.add_argument("--base_model", type=str, default="llama2:7b", help="Base Ollama model to fine-tune")
    parser.add_argument("--output_model", type=str, required=True, help="Name for the fine-tuned Ollama model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to fine-tuning dataset in JSONL format")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for fine-tuning")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for retrospective penalty")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for exploration bonus")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--reward_model", type=str, default="gpt2", help="Model to use for TD reward calculation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for fine-tuning")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_ollama_model(model_name, ollama_host):
    """
    Check if an Ollama model exists.
    
    Args:
        model_name: Name of the Ollama model
        ollama_host: Ollama API host
        
    Returns:
        True if the model exists, False otherwise
    """
    try:
        response = requests.get(f"{ollama_host}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == model_name:
                    return True
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama model: {e}")
        return False


def prepare_finetune_dataset(dataset_path, reward_model, ollama_host):
    """
    Prepare dataset for fine-tuning with TD rewards.
    
    Args:
        dataset_path: Path to dataset in JSONL format
        reward_model: TD reward model for calculating rewards
        ollama_host: Ollama API host
        
    Returns:
        Prepared dataset with TD rewards
    """
    logger.info(f"Preparing fine-tuning dataset from {dataset_path}")
    
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(dataset)} examples from dataset")
    
    # Calculate TD rewards for each example
    for i, example in enumerate(tqdm(dataset, desc="Calculating TD rewards")):
        prompt = example["prompt"]
        completion = example["completion"]
        
        # Tokenize prompt and first token of completion
        context_ids = reward_model.tokenizer.encode(prompt, return_tensors="pt")
        
        # Get the first token of the completion
        first_completion_token = completion[:1]
        first_completion_token_id = reward_model.tokenizer.encode(
            first_completion_token, add_special_tokens=False
        )[0]
        first_completion_token_id = torch.tensor([[first_completion_token_id]])
        
        # Generate prediction from reward model
        with torch.no_grad():
            outputs = reward_model.language_model(context_ids, return_dict=True)
            logits = outputs.logits[:, -1]
            pred_token_id = torch.argmax(logits, dim=-1)
        
        # Calculate TD reward
        reward = reward_model.compute_td_reward(
            context=context_ids,
            prediction=pred_token_id.unsqueeze(0),
            actual_next_token=first_completion_token_id
        )
        
        # Add reward to example
        example["reward"] = reward
    
    # Sort dataset by reward (higher reward first)
    dataset.sort(key=lambda x: x.get("reward", 0), reverse=True)
    
    logger.info(f"Dataset prepared with TD rewards (max: {dataset[0]['reward']:.4f}, min: {dataset[-1]['reward']:.4f})")
    
    return dataset


def create_ollama_modelfile(base_model, output_model):
    """
    Create Modelfile for fine-tuning with Ollama.
    
    Args:
        base_model: Base Ollama model name
        output_model: Name for the fine-tuned model
        
    Returns:
        Path to created Modelfile
    """
    logger.info(f"Creating Modelfile for {output_model} based on {base_model}")
    
    modelfile_content = f"""
FROM {base_model}

# Model parameters for fine-tuning
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 128

# System prompt emphasizing prospective prediction
SYSTEM """

    system_prompt = """
You are an AI assistant trained with a special focus on prospective prediction - accurately anticipating what comes next based on current context, rather than relying on retrospective patterns.

Your training emphasized temporal difference learning, which rewards accurate anticipation of future events based on current state. This helps you develop a strong understanding of causality and temporal relationships.

When answering questions or responding to prompts, try to anticipate the user's needs and the logical progression of the conversation.
"""

    modelfile_content += system_prompt.strip()
    
    # Create Modelfile
    modelfile_path = "./Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    logger.info(f"Created Modelfile at {modelfile_path}")
    return modelfile_path


def create_finetuning_jsonl(dataset, output_path):
    """
    Create JSONL file for Ollama fine-tuning.
    
    Args:
        dataset: Prepared dataset with TD rewards
        output_path: Path to save the JSONL file
        
    Returns:
        Path to created JSONL file
    """
    logger.info(f"Creating fine-tuning JSONL file at {output_path}")
    
    # Convert dataset to Ollama fine-tuning format
    with open(output_path, "w") as f:
        for example in dataset:
            entry = {
                "prompt": example["prompt"],
                "response": example["completion"]
            }
            f.write(json.dumps(entry) + "\n")
    
    logger.info(f"Created fine-tuning JSONL file with {len(dataset)} examples")
    return output_path


def finetune_ollama_model(base_model, output_model, dataset_path, modelfile_path):
    """
    Fine-tune an Ollama model.
    
    Args:
        base_model: Base Ollama model name
        output_model: Name for the fine-tuned model
        dataset_path: Path to fine-tuning dataset in JSONL format
        modelfile_path: Path to Modelfile
        
    Returns:
        True if fine-tuning was successful, False otherwise
    """
    logger.info(f"Starting fine-tuning of {base_model} to create {output_model}")
    
    try:
        # Create the model first
        subprocess.run(
            ["ollama", "create", output_model, "-f", modelfile_path],
            check=True
        )
        
        # Run fine-tuning
        subprocess.run(
            ["ollama", "train", output_model, "--file", dataset_path],
            check=True
        )
        
        logger.info(f"Successfully fine-tuned {output_model}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fine-tuning Ollama model: {e}")
        return False


def evaluate_finetuned_model(output_model, base_model, dataset, reward_model, ollama_host):
    """
    Evaluate the fine-tuned model by comparing TD rewards with the base model.
    
    Args:
        output_model: Name of the fine-tuned model
        base_model: Name of the base model
        dataset: Evaluation dataset
        reward_model: TD reward model for evaluation
        ollama_host: Ollama API host
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Evaluating fine-tuned model {output_model} versus base model {base_model}")
    
    # Sample a subset of the dataset for evaluation
    eval_dataset = dataset[:min(50, len(dataset))]
    
    base_rewards = []
    finetuned_rewards = []
    
    for example in tqdm(eval_dataset, desc="Evaluating models"):
        prompt = example["prompt"]
        
        # Get completions from both models
        try:
            # Base model completion
            base_response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": base_model,
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": 1  # Just predict the first token for TD reward comparison
                }
            )
            
            if base_response.status_code != 200:
                logger.warning(f"Error getting base model completion: {base_response.status_code}")
                continue
                
            base_completion = base_response.json().get("response", "")
            
            # Fine-tuned model completion
            finetuned_response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": output_model,
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": 1  # Just predict the first token for TD reward comparison
                }
            )
            
            if finetuned_response.status_code != 200:
                logger.warning(f"Error getting fine-tuned model completion: {finetuned_response.status_code}")
                continue
                
            finetuned_completion = finetuned_response.json().get("response", "")
            
            # Calculate TD rewards
            context_ids = reward_model.tokenizer.encode(prompt, return_tensors="pt")
            
            # "Actual" next token is the first token of the expected completion
            expected_completion = example["completion"]
            expected_first_token_id = reward_model.tokenizer.encode(
                expected_completion[:1], add_special_tokens=False
            )[0]
            expected_first_token_id = torch.tensor([[expected_first_token_id]])
            
            # Base model reward
            base_pred_id = reward_model.tokenizer.encode(
                base_completion[:1], add_special_tokens=False
            )[0]
            base_pred_id = torch.tensor([[base_pred_id]])
            
            base_reward = reward_model.compute_td_reward(
                context=context_ids,
                prediction=base_pred_id,
                actual_next_token=expected_first_token_id
            )
            
            # Fine-tuned model reward
            finetuned_pred_id = reward_model.tokenizer.encode(
                finetuned_completion[:1], add_special_tokens=False
            )[0]
            finetuned_pred_id = torch.tensor([[finetuned_pred_id]])
            
            finetuned_reward = reward_model.compute_td_reward(
                context=context_ids,
                prediction=finetuned_pred_id,
                actual_next_token=expected_first_token_id
            )
            
            base_rewards.append(base_reward)
            finetuned_rewards.append(finetuned_reward)
            
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            continue
    
    # Calculate evaluation metrics
    results = {
        "base_model": {
            "name": base_model,
            "mean_reward": float(np.mean(base_rewards)),
            "median_reward": float(np.median(base_rewards)),
            "std_reward": float(np.std(base_rewards)),
            "min_reward": float(np.min(base_rewards)),
            "max_reward": float(np.max(base_rewards))
        },
        "finetuned_model": {
            "name": output_model,
            "mean_reward": float(np.mean(finetuned_rewards)),
            "median_reward": float(np.median(finetuned_rewards)),
            "std_reward": float(np.std(finetuned_rewards)),
            "min_reward": float(np.min(finetuned_rewards)),
            "max_reward": float(np.max(finetuned_rewards))
        },
        "improvement": {
            "mean_reward": float(np.mean(finetuned_rewards) - np.mean(base_rewards)),
            "median_reward": float(np.median(finetuned_rewards) - np.median(base_rewards)),
            "percent_improvement": float((np.mean(finetuned_rewards) - np.mean(base_rewards)) / abs(np.mean(base_rewards)) * 100)
        }
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"Base model ({base_model}) mean TD reward: {results['base_model']['mean_reward']:.4f}")
    logger.info(f"Fine-tuned model ({output_model}) mean TD reward: {results['finetuned_model']['mean_reward']:.4f}")
    logger.info(f"Improvement: {results['improvement']['mean_reward']:.4f} ({results['improvement']['percent_improvement']:.2f}%)")
    
    # Save results to JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation results saved to evaluation_results.json")
    
    return results


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize reward model
    logger.info(f"Initializing TD reward model with {args.reward_model}")
    reward_model = TDRewardModel(
        model_name=args.reward_model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Check if base model exists
    if not check_ollama_model(args.base_model, args.ollama_host):
        logger.error(f"Base model {args.base_model} not found. Please pull it first with 'ollama pull {args.base_model}'")
        return
    
    # Prepare dataset with TD rewards
    dataset = prepare_finetune_dataset(args.dataset, reward_model, args.ollama_host)
    
    # Create Modelfile
    modelfile_path = create_ollama_modelfile(args.base_model, args.output_model)
    
    # Create fine-tuning JSONL file
    finetune_jsonl_path = "finetune_data.jsonl"
    create_finetuning_jsonl(dataset, finetune_jsonl_path)
    
    # Fine-tune the model
    success = finetune_ollama_model(args.base_model, args.output_model, finetune_jsonl_path, modelfile_path)
    
    if success:
        # Evaluate fine-tuned model
        evaluation_results = evaluate_finetuned_model(
            args.output_model, args.base_model, dataset, reward_model, args.ollama_host
        )
        
        # Print final summary
        logger.info("\nFine-tuning summary:")
        logger.info(f"Base model: {args.base_model}")
        logger.info(f"Fine-tuned model: {args.output_model}")
        logger.info(f"Dataset: {args.dataset} ({len(dataset)} examples)")
        logger.info(f"TD reward parameters: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
        logger.info(f"Mean TD reward improvement: {evaluation_results['improvement']['mean_reward']:.4f} ({evaluation_results['improvement']['percent_improvement']:.2f}%)")
    else:
        logger.error("Fine-tuning failed.")


if __name__ == "__main__":
    main()
