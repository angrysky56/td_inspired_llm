"""
Test script for evaluating a LoRA-adapted model using TD-inspired reward.

This script loads a base model with a LoRA adapter and evaluates it
using our TD-inspired reward function.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from td_reward import TDRewardModel
from lora_trainer import test_lora_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA-adapted model with TD reward")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--lora_adapter", type=str, required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--compare_models", action="store_true", help="Compare with base model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test prompts to evaluate")
    parser.add_argument("--output_dir", type=str, default="./lora_results", help="Directory to save results")
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


def generate_test_prompts(num_samples):
    """Generate a set of test prompts for evaluation."""
    base_prompts = [
        "The quick brown fox jumps over the",
        "In the beginning, there was",
        "Once upon a time, a brave knight",
        "The scientist discovered a new",
        "When I looked outside, I saw a",
        "The most important thing to remember is",
        "If you want to succeed, you must",
        "The future of artificial intelligence depends on",
        "The main difference between humans and machines is",
        "To solve this problem, we need to",
        "The role of dopamine in learning is to",
        "Temporal difference learning works by",
        "The belief state representation helps to",
        "Prospective prediction is better than retrospective learning because",
        "The key to effective contingency learning is",
        "When making decisions under uncertainty, we should",
        "The value of a state depends on",
        "To predict future rewards accurately, an agent must",
        "The difference between TD errors and prediction errors is",
        "A good reinforcement learning algorithm will"
    ]
    
    # If num_samples <= len(base_prompts), return a subset
    if num_samples <= len(base_prompts):
        return base_prompts[:num_samples]
    
    # Otherwise, return all base prompts plus some variations
    return base_prompts


def load_models(args):
    """Load base model and LoRA-adapted model."""
    print(f"Loading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter: {args.lora_adapter}")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    
    return base_model, lora_model, base_tokenizer


def evaluate_models(base_model, lora_model, tokenizer, reward_model, prompts):
    """
    Evaluate both base model and LoRA model using TD reward.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)
    lora_model = lora_model.to(device)
    
    # Ensure all reward model components are on the same device
    reward_model.language_model = reward_model.language_model.to(device)
    reward_model.belief_state_model = reward_model.belief_state_model.to(device)
    reward_model.value_function = reward_model.value_function.to(device)
    
    results = {
        "base_model": [],
        "lora_model": [],
    }
    
    for prompt in tqdm(prompts, desc="Evaluating models"):
        try:
            # Tokenize input
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Get expected next token from reward model
            with torch.no_grad():
                reward_outputs = reward_model.language_model(input_ids, return_dict=True)
                expected_token_id = torch.argmax(reward_outputs.logits[:, -1], dim=-1)
                expected_token = tokenizer.decode(expected_token_id)
            
            # Generate from base model
            with torch.no_grad():
                base_outputs = base_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
                base_prediction = tokenizer.decode(base_outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
                base_first_token = base_prediction.split()[0] if base_prediction.split() else base_prediction[:1]
            
            # Generate from LoRA model
            with torch.no_grad():
                lora_outputs = lora_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
                lora_prediction = tokenizer.decode(lora_outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
                lora_first_token = lora_prediction.split()[0] if lora_prediction.split() else lora_prediction[:1]
            
            # Calculate TD rewards
            base_token_ids = tokenizer.encode(base_first_token, add_special_tokens=False)
            lora_token_ids = tokenizer.encode(lora_first_token, add_special_tokens=False)
            
            if base_token_ids and lora_token_ids:
                base_prediction_id = torch.tensor([[base_token_ids[0]]]).to(device)
                lora_prediction_id = torch.tensor([[lora_token_ids[0]]]).to(device)
                
                base_reward = reward_model.compute_td_reward(
                    context=input_ids,
                    prediction=base_prediction_id,
                    actual_next_token=expected_token_id.unsqueeze(0)
                )
                
                lora_reward = reward_model.compute_td_reward(
                    context=input_ids,
                    prediction=lora_prediction_id,
                    actual_next_token=expected_token_id.unsqueeze(0)
                )
                
                # Save results
                base_result = {
                    "prompt": prompt,
                    "prediction": base_prediction,
                    "first_token": base_first_token,
                    "expected_token": expected_token,
                    "td_reward": base_reward
                }
                
                lora_result = {
                    "prompt": prompt,
                    "prediction": lora_prediction,
                    "first_token": lora_first_token,
                    "expected_token": expected_token,
                    "td_reward": lora_reward
                }
                
                results["base_model"].append(base_result)
                results["lora_model"].append(lora_result)
                
                # Print results
                print(f"Prompt: {prompt}")
                print(f"Expected next token: {expected_token}")
                print(f"Base model prediction: {base_prediction}")
                print(f"Base model TD reward: {base_reward:.4f}")
                print(f"LoRA model prediction: {lora_prediction}")
                print(f"LoRA model TD reward: {lora_reward:.4f}")
                print("-" * 40)
            
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            continue
    
    return results


def analyze_results(results, output_dir):
    """Analyze and visualize the results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics
    summary = {}
    
    for model_name, model_results in results.items():
        if not model_results:
            print(f"No valid results for {model_name}. Skipping analysis.")
            continue
            
        rewards = [r["td_reward"] for r in model_results]
        summary[model_name] = {
            "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "count": len(rewards)
        }
        
        print(f"\nResults for {model_name}:")
        print(f"  Mean TD reward: {summary[model_name]['mean_reward']:.4f}")
        print(f"  Median TD reward: {summary[model_name]['median_reward']:.4f}")
        print(f"  Standard deviation: {summary[model_name]['std_reward']:.4f}")
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "lora_comparison_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "summary": summary
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Create visualizations
    if len(summary) > 0:
        # Bar chart comparison
        plt.figure(figsize=(12, 8))
        
        # Mean rewards
        model_names = list(summary.keys())
        mean_rewards = [summary[m]["mean_reward"] for m in model_names]
        std_rewards = [summary[m]["std_reward"] for m in model_names]
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=10)
        plt.xlabel('Model')
        plt.ylabel('Mean TD Reward')
        plt.title('Comparison of Models by Mean TD Reward')
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add values on top of bars
        for bar, value in zip(bars, mean_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{value:.4f}", ha="center")
        
        # Box plot comparison
        plt.subplot(2, 1, 2)
        data = [[r["td_reward"] for r in results[model]] for model in model_names]
        plt.boxplot(data, labels=model_names)
        plt.xlabel('Model')
        plt.ylabel('TD Reward')
        plt.title('Distribution of TD Rewards by Model')
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, "lora_comparison.png")
        plt.savefig(plot_file)
        print(f"Visualization saved to {plot_file}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Generate test prompts
    prompts = generate_test_prompts(args.num_samples)
    
    # Initialize reward model
    print(f"Initializing TD reward model with alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    reward_model = TDRewardModel(
        model_name=args.base_model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    if args.compare_models:
        # Load models
        base_model, lora_model, tokenizer = load_models(args)
        
        # Evaluate models
        results = evaluate_models(base_model, lora_model, tokenizer, reward_model, prompts)
        
        # Analyze results
        analyze_results(results, args.output_dir)
    else:
        # Just test the LoRA model on prompts
        print("Testing LoRA model generations:")
        for prompt in prompts:
            generated_text = test_lora_model(
                args.base_model, 
                args.lora_adapter, 
                prompt,
                max_length=100
            )
            
            print(f"Prompt: {prompt}")
            print(f"Generation: {generated_text}")
            print("-" * 40)


if __name__ == "__main__":
    main()
