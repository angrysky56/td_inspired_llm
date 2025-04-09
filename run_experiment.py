"""
Run a small experiment to test the TD-inspired reward function.

This script runs a small experiment to test the TD-inspired reward function
on a subset of a dataset. It compares the TD rewards for different LLM models
and prints the results.
"""

import os
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from td_reward import TDRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small experiment with TD-inspired reward")
    parser.add_argument("--models", type=str, nargs="+", default=["gpt2", "distilgpt2"], 
                        help="Models to compare")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./experiment_results", help="Directory to save results")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for retrospective penalty")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for exploration bonus")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(dataset_name, dataset_config, split, num_samples, seed):
    """
    Prepare dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        split: Dataset split to use
        num_samples: Number of samples to evaluate
        seed: Random seed
        
    Returns:
        List of text samples
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Sample a subset of the dataset
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    # Convert numpy.int64 to standard Python int
    indices = [int(i) for i in indices]
    samples = [dataset[i]["text"] for i in indices]
    
    # Filter out very short samples
    samples = [s for s in samples if len(s) > 50]
    
    # Truncate long samples
    samples = [s[:500] for s in samples]
    
    return samples


def evaluate_model_td_rewards(model_name, samples, reward_model, device):
    """
    Evaluate TD rewards for a model on samples.
    
    Args:
        model_name: Name of the model to evaluate
        samples: Text samples to evaluate
        reward_model: TD reward model for evaluation
        device: Device to run evaluation on
        
    Returns:
        List of TD rewards for each sample
    """
    print(f"Evaluating model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    rewards = []
    
    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        # Skip very short samples
        if len(sample) < 10:
            continue
        
        # Use first 80% as context and last 20% as target
        split_idx = int(len(sample) * 0.8)
        context = sample[:split_idx]
        target = sample[split_idx:]
        
        # Skip samples where target is empty or too short
        if not target or len(target.strip()) < 1:
            continue
            
        # Tokenize context
        context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(context_ids, return_dict=True)
            logits = outputs.logits[:, -1]  # Last token predictions
            pred_token_id = torch.argmax(logits, dim=-1)
        
        # Tokenize the first token of the target
        try:
            target_token_id = tokenizer.encode(target[:1], add_special_tokens=False)[0]
            target_token_id = torch.tensor([[target_token_id]]).to(device)
            
            # Make sure reward model is on the same device
            if reward_model.language_model.device != device:
                reward_model.language_model = reward_model.language_model.to(device)
                reward_model.belief_state_model = reward_model.belief_state_model.to(device)
                reward_model.value_function = reward_model.value_function.to(device)
            
            # Compute TD reward
            reward = reward_model.compute_td_reward(
                context=context_ids,
                prediction=pred_token_id.unsqueeze(0),
                actual_next_token=target_token_id
            )
            
            rewards.append(reward)
        except IndexError:
            # Skip samples that cause tokenization issues
            continue
    
    return rewards


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize reward model
    reward_model = TDRewardModel(
        model_name="gpt2",  # Base model for reward calculation
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Prepare dataset
    print(f"Loading dataset: {args.dataset}")
    samples = prepare_dataset(
        args.dataset, args.dataset_config, args.split, args.num_samples, args.seed
    )
    print(f"Prepared {len(samples)} samples for evaluation")
    
    # Evaluate models
    results = {}
    
    for model_name in args.models:
        rewards = evaluate_model_td_rewards(model_name, samples, reward_model, device)
        if not rewards:
            print(f"Warning: No valid rewards for {model_name}. Skipping...")
            continue
            
        results[model_name] = {
            "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "all_rewards": [float(r) for r in rewards]
        }
        print(f"{model_name}: Mean TD reward = {results[model_name]['mean_reward']:.4f}")
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    if len(results) > 0:
        # Bar plot for mean rewards
        plt.subplot(1, 2, 1)
        model_names = list(results.keys())
        mean_rewards = [results[model]["mean_reward"] for model in model_names]
        std_rewards = [results[model]["std_reward"] for model in model_names]
        
        bars = plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=10)
        plt.ylabel("Mean TD Reward")
        plt.title("Comparison of Models by Mean TD Reward")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{mean_rewards[i]:.4f}", ha="center")
        
        # Box plot for reward distributions
        plt.subplot(1, 2, 2)
        all_rewards = [results[model]["all_rewards"] for model in model_names]
        plt.boxplot(all_rewards, labels=model_names)
        plt.ylabel("TD Reward")
        plt.title("Distribution of TD Rewards by Model")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(args.output_dir, "rewards_comparison.png")
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
    else:
        print("No valid results to plot")
    
    # Print summary
    print("\nSummary of results:")
    for model_name in results:
        print(f"{model_name}:")
        print(f"  Mean TD reward: {results[model_name]['mean_reward']:.4f}")
        print(f"  Median TD reward: {results[model_name]['median_reward']:.4f}")
        print(f"  Std TD reward: {results[model_name]['std_reward']:.4f}")
        print(f"  Min TD reward: {results[model_name]['min_reward']:.4f}")
        print(f"  Max TD reward: {results[model_name]['max_reward']:.4f}")
        print()


if __name__ == "__main__":
    main()
