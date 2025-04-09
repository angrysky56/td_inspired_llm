"""
Test script for evaluating Hudson/gpt2-instruct:345m-q8_0 with TD-inspired reward.

This script pulls the Hudson/gpt2-instruct:345m-q8_0 model using Ollama and evaluates
its performance using our TD-inspired reward function.
"""

import os
import argparse
import torch
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from td_reward import TDRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Test GPT-2 Instruct with TD reward")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model for reward calculation")
    parser.add_argument("--ollama_model", type=str, default="Hudson/gpt2-instruct:345m-q8_0", 
                        help="Ollama model to evaluate")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test prompts to evaluate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for retrospective penalty")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for exploration bonus")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama API host")
    
    return parser.parse_args()


def check_ollama_model(model_name, ollama_host):
    """Check if the Ollama model exists, pull it if not."""
    try:
        response = requests.get(f"{ollama_host}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == model_name:
                    print(f"Model {model_name} already exists in Ollama")
                    return True
        
        # Model not found, pull it
        print(f"Model {model_name} not found, pulling it now...")
        pull_response = requests.post(
            f"{ollama_host}/api/pull",
            json={"name": model_name}
        )
        
        if pull_response.status_code == 200:
            print(f"Successfully pulled model {model_name}")
            return True
        else:
            print(f"Failed to pull model {model_name}: {pull_response.status_code}")
            print(pull_response.text)
            return False
    
    except Exception as e:
        print(f"Error checking/pulling Ollama model: {e}")
        return False


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
    result = base_prompts.copy()
    
    variations = [
        " consider that",
        " remember that",
        " understand that",
        " realize that",
        " know that"
    ]
    
    # Add variations until we reach num_samples
    i = 0
    while len(result) < num_samples:
        base_idx = i % len(base_prompts)
        var_idx = (i // len(base_prompts)) % len(variations)
        result.append(base_prompts[base_idx] + variations[var_idx])
        i += 1
    
    return result[:num_samples]


def evaluate_model(model_name, reward_model, prompts, ollama_host):
    """Evaluate the model using TD reward."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make sure reward model is on the right device
    reward_model.language_model = reward_model.language_model.to(device)
    reward_model.belief_state_model = reward_model.belief_state_model.to(device)
    reward_model.value_function = reward_model.value_function.to(device)
    
    results = []
    
    for prompt in tqdm(prompts, desc=f"Evaluating {model_name}"):
        try:
            # Get prediction from Ollama model
            ollama_response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 5,
                    "stream": False
                }
            )
            
            if ollama_response.status_code != 200:
                print(f"Error from Ollama API: {ollama_response.status_code}")
                print(ollama_response.text)
                continue
            
            # Extract predicted tokens
            ollama_text = ollama_response.json().get("response", "")
            if not ollama_text:
                print(f"Empty response for prompt: {prompt}")
                continue
            
            # Tokenize prompt
            context_ids = reward_model.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Get "actual" next token from the base model (this serves as our ground truth)
            with torch.no_grad():
                base_outputs = reward_model.language_model(context_ids, return_dict=True)
                base_logits = base_outputs.logits[:, -1]
                expected_token_id = torch.argmax(base_logits, dim=-1)
                expected_token = reward_model.tokenizer.decode(expected_token_id)
            
            # Get first token of Ollama prediction
            first_token = ollama_text.split()[0] if ollama_text.split() else ollama_text[:1]
            
            # Calculate TD reward
            reward = reward_model.compute_td_reward(
                context=prompt,
                prediction=first_token,
                actual_next_token=expected_token
            )
            
            # Save result
            result = {
                "prompt": prompt,
                "prediction": ollama_text,
                "first_token": first_token,
                "expected_token": expected_token,
                "td_reward": reward
            }
            
            results.append(result)
            
            print(f"Prompt: {prompt}")
            print(f"Prediction: {ollama_text}")
            print(f"Expected next token: {expected_token}")
            print(f"TD reward: {reward:.4f}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            continue
    
    return results


def analyze_results(results, output_dir):
    """Analyze and visualize the results."""
    if not results:
        print("No valid results to analyze.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    rewards = [r["td_reward"] for r in results]
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)
    
    print("\nResults Summary:")
    print(f"Mean TD reward: {mean_reward:.4f}")
    print(f"Median TD reward: {median_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    
    # Save raw results to JSON
    results_file = os.path.join(output_dir, "gpt2_instruct_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "mean_reward": mean_reward,
                "median_reward": median_reward,
                "std_reward": std_reward
            }
        }, f, indent=2)
    
    print(f"Detailed results saved to {results_file}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Reward distribution
    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=10, alpha=0.7)
    plt.axvline(mean_reward, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward:.4f}')
    plt.axvline(median_reward, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_reward:.4f}')
    plt.xlabel('TD Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of TD Rewards')
    plt.legend()
    
    # Individual rewards
    plt.subplot(1, 2, 2)
    y_pos = range(len(rewards))
    plt.barh(y_pos, rewards)
    plt.yticks(y_pos, [f"Prompt {i+1}" for i in range(len(rewards))])
    plt.xlabel('TD Reward')
    plt.title('TD Reward by Prompt')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "gpt2_instruct_rewards.png")
    plt.savefig(plot_file)
    print(f"Visualization saved to {plot_file}")


def main():
    args = parse_args()
    
    # Check/pull Ollama model
    if not check_ollama_model(args.ollama_model, args.ollama_host):
        print("Failed to ensure model is available. Exiting.")
        return
    
    # Initialize reward model
    print(f"Initializing TD reward model with {args.base_model}")
    reward_model = TDRewardModel(
        model_name=args.base_model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Generate test prompts
    prompts = generate_test_prompts(args.num_samples)
    print(f"Generated {len(prompts)} test prompts")
    
    # Evaluate model
    print(f"Evaluating {args.ollama_model} with TD reward...")
    results = evaluate_model(args.ollama_model, reward_model, prompts, args.ollama_host)
    
    # Analyze results
    analyze_results(results, args.output_dir)


if __name__ == "__main__":
    main()
