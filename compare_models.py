"""
Compare different models using TD-inspired reward function.

This script compares multiple models (both HuggingFace and Ollama models)
using our TD-inspired reward function.
"""

import os
import argparse
import torch
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from td_reward import TDRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models with TD reward")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model for reward calculation")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["gpt2", "distilgpt2", "Hudson/gpt2-instruct:345m-q8_0"],
                        help="Models to compare (prefix with 'ollama:' for Ollama models)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test prompts to evaluate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for retrospective penalty")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for exploration bonus")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--output_dir", type=str, default="./comparison_results", help="Directory to save results")
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


def evaluate_hf_model(model_name, reward_model, prompts):
    """Evaluate a HuggingFace model using TD reward."""
    print(f"Loading HuggingFace model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Make sure reward model is on the right device
    reward_model.language_model = reward_model.language_model.to(device)
    reward_model.belief_state_model = reward_model.belief_state_model.to(device)
    reward_model.value_function = reward_model.value_function.to(device)
    
    results = []
    
    for prompt in tqdm(prompts, desc=f"Evaluating {model_name}"):
        try:
            # Tokenize input
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            context_ids = input_ids.clone()  # Create a copy for context
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, 
                    max_length=input_ids.shape[1] + 5,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                # Get model's prediction
                prediction = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
                
                # Get base model's expected next token
                base_outputs = reward_model.language_model(context_ids, return_dict=True)
                base_logits = base_outputs.logits[:, -1]
                expected_token_id = torch.argmax(base_logits, dim=-1)
                expected_token = reward_model.tokenizer.decode(expected_token_id)
                
                # Get first token of the prediction
                first_token = prediction.split()[0] if prediction.split() else prediction[:1]
                
                # Calculate TD reward
                token_ids = reward_model.tokenizer.encode(first_token, add_special_tokens=False)
                if token_ids:
                    prediction_id = token_ids[0]
                    prediction_tensor = torch.tensor([[prediction_id]]).to(device)
                    
                    # Compute TD reward
                    reward = reward_model.compute_td_reward(
                        context=context_ids,
                        prediction=prediction_tensor,
                        actual_next_token=expected_token_id.unsqueeze(0)
                    )
                else:
                    print(f"Warning: Could not tokenize first token '{first_token}'")
                    reward = -5.0  # Assign a penalty
                
                # Save result
                result = {
                    "prompt": prompt,
                    "prediction": prediction,
                    "first_token": first_token,
                    "expected_token": expected_token,
                    "td_reward": reward
                }
                
                results.append(result)
                
                # Print result
                print(f"Prompt: {prompt}")
                print(f"Prediction: {prediction}")
                print(f"Expected next token: {expected_token}")
                print(f"TD reward: {reward:.4f}")
                print("-" * 40)
                
        except Exception as e:
            print(f"Error processing prompt '{prompt}' with model {model_name}: {e}")
            continue
    
    return results


def evaluate_ollama_model(model_name, reward_model, prompts, ollama_host):
    """Evaluate an Ollama model using TD reward."""
    print(f"Evaluating Ollama model: {model_name}")
    
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
            token_ids = reward_model.tokenizer.encode(first_token, add_special_tokens=False)
            if token_ids:
                prediction_id = token_ids[0]
                prediction_tensor = torch.tensor([[prediction_id]]).to(device)
                
                # Compute TD reward
                reward = reward_model.compute_td_reward(
                    context=context_ids,
                    prediction=prediction_tensor,
                    actual_next_token=expected_token_id.unsqueeze(0)
                )
            else:
                print(f"Warning: Could not tokenize first token '{first_token}'")
                reward = -5.0  # Assign a penalty
            
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
            print(f"Error processing prompt '{prompt}' with model {model_name}: {e}")
            continue
    
    return results


def analyze_results(all_results, output_dir):
    """Analyze and visualize the results for all models."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare summary data
    summary = {}
    for model_name, results in all_results.items():
        if not results:
            print(f"No valid results for model {model_name}. Skipping analysis.")
            continue
            
        rewards = [r["td_reward"] for r in results]
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
    
    # Save detailed results to JSON
    results_file = os.path.join(output_dir, "model_comparison_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": all_results,
            "summary": summary
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Create visualizations
    if len(summary) > 0:
        # Bar chart comparison
        plt.figure(figsize=(12, 8))
        
        # Sort models by mean reward
        sorted_models = sorted(summary.keys(), key=lambda m: summary[m]["mean_reward"], reverse=True)
        mean_rewards = [summary[m]["mean_reward"] for m in sorted_models]
        std_rewards = [summary[m]["std_reward"] for m in sorted_models]
        
        # Bar chart of mean rewards
        plt.subplot(2, 1, 1)
        bars = plt.bar(sorted_models, mean_rewards, yerr=std_rewards, capsize=10)
        plt.xlabel('Model')
        plt.ylabel('Mean TD Reward')
        plt.title('Comparison of Models by Mean TD Reward')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add values on top of bars
        for bar, value in zip(bars, mean_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{value:.4f}", ha="center")
        
        # Box plot comparison
        plt.subplot(2, 1, 2)
        data = [[r["td_reward"] for r in all_results[model]] for model in sorted_models]
        plt.boxplot(data, labels=sorted_models)
        plt.xlabel('Model')
        plt.ylabel('TD Reward')
        plt.title('Distribution of TD Rewards by Model')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(plot_file)
        print(f"Visualization saved to {plot_file}")


def main():
    args = parse_args()
    
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
    
    # Evaluate models
    all_results = {}
    
    for model_name in args.models:
        if model_name.startswith("ollama:"):
            # Extract Ollama model name
            ollama_model = model_name[len("ollama:"):]
            
            # Check/pull Ollama model
            if not check_ollama_model(ollama_model, args.ollama_host):
                print(f"Failed to ensure Ollama model {ollama_model} is available. Skipping.")
                continue
            
            # Evaluate Ollama model
            results = evaluate_ollama_model(ollama_model, reward_model, prompts, args.ollama_host)
            all_results[model_name] = results
        else:
            # Evaluate HuggingFace model
            results = evaluate_hf_model(model_name, reward_model, prompts)
            all_results[model_name] = results
    
    # Analyze results
    analyze_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
