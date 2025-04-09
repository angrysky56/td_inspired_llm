"""
Evaluation script for a language model trained with TD-inspired reward function.

This script evaluates the perplexity and generation quality of a fine-tuned model
compared to the base model.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from td_reward import TDRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM with TD-inspired reward")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model for comparison")
    parser.add_argument("--fine_tuned_model", type=str, required=True, help="Fine-tuned model to evaluate")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use for evaluation")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for generation evaluation")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", help="File to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_perplexity(model, tokenizer, dataset, max_length, batch_size, device):
    """
    Evaluate perplexity of a model on a dataset.
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer
        dataset: The evaluation dataset
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Average perplexity across the dataset
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating perplexity"):
        batch = dataset[i:min(i + batch_size, len(dataset))]
        
        # Tokenize batch
        inputs = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids,
                return_dict=True
            )
            
            loss = outputs.loss
            
            # Calculate token count (excluding padding)
            tokens = inputs.attention_mask.sum().item()
            
            total_loss += loss.item() * tokens
            total_tokens += tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def evaluate_td_rewards(base_model, fine_tuned_model, reward_model, dataset, max_length, batch_size, device):
    """
    Compare TD rewards for base and fine-tuned models.
    
    Args:
        base_model: The base language model
        fine_tuned_model: The fine-tuned language model
        reward_model: The TD reward model
        dataset: The evaluation dataset
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Average TD rewards for base and fine-tuned models
    """
    base_model.eval()
    fine_tuned_model.eval()
    
    base_rewards = []
    fine_tuned_rewards = []
    
    for i in tqdm(range(0, min(len(dataset), 100), batch_size), desc="Evaluating TD rewards"):
        batch = dataset[i:min(i + batch_size, len(dataset), 100)]
        
        # Tokenize batch
        inputs = reward_model.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # For simplicity, use all but last token as context and last token as target
        context_ids = inputs.input_ids[:, :-1]
        context_mask = inputs.attention_mask[:, :-1]
        target_ids = inputs.input_ids[:, -1]
        
        # Get predictions from both models
        with torch.no_grad():
            base_outputs = base_model(
                input_ids=context_ids,
                attention_mask=context_mask,
                return_dict=True
            )
            
            fine_tuned_outputs = fine_tuned_model(
                input_ids=context_ids,
                attention_mask=context_mask,
                return_dict=True
            )
            
            # Get top predicted token for each model
            base_pred = torch.argmax(base_outputs.logits[:, -1], dim=-1)
            fine_tuned_pred = torch.argmax(fine_tuned_outputs.logits[:, -1], dim=-1)
        
        # Calculate TD rewards
        for j in range(len(batch)):
            base_reward = reward_model.compute_td_reward(
                context=context_ids[j].unsqueeze(0),
                prediction=base_pred[j].unsqueeze(0),
                actual_next_token=target_ids[j].unsqueeze(0)
            )
            
            fine_tuned_reward = reward_model.compute_td_reward(
                context=context_ids[j].unsqueeze(0),
                prediction=fine_tuned_pred[j].unsqueeze(0),
                actual_next_token=target_ids[j].unsqueeze(0)
            )
            
            base_rewards.append(base_reward)
            fine_tuned_rewards.append(fine_tuned_reward)
    
    # Calculate average rewards
    avg_base_reward = np.mean(base_rewards)
    avg_fine_tuned_reward = np.mean(fine_tuned_rewards)
    
    return avg_base_reward, avg_fine_tuned_reward


def generate_and_compare(base_model, fine_tuned_model, tokenizer, dataset, num_samples, device, output_file):
    """
    Generate text using both models and compare outputs.
    
    Args:
        base_model: The base language model
        fine_tuned_model: The fine-tuned language model
        tokenizer: The tokenizer
        dataset: The evaluation dataset
        num_samples: Number of samples to generate
        device: Device to run generation on
        output_file: File to save generation samples
        
    Returns:
        None (writes results to output_file)
    """
    base_model.eval()
    fine_tuned_model.eval()
    
    # Sample prompts from dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    prompts = [dataset[i]["text"][:100] for i in indices]  # Use first 100 chars as prompt
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Text Generation Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        for i, prompt in enumerate(prompts):
            f.write(f"Sample {i+1}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Prompt: {prompt}\n\n")
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate from base model
            with torch.no_grad():
                base_outputs = base_model.generate(
                    inputs.input_ids,
                    max_length=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                fine_tuned_outputs = fine_tuned_model.generate(
                    inputs.input_ids,
                    max_length=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            # Decode outputs
            base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            fine_tuned_text = tokenizer.decode(fine_tuned_outputs[0], skip_special_tokens=True)
            
            f.write(f"Base model: {base_text}\n\n")
            f.write(f"Fine-tuned model: {fine_tuned_text}\n\n")
            f.write("=" * 80 + "\n\n")
    
    print(f"Generation samples saved to {output_file}")


def plot_rewards_comparison(base_rewards, fine_tuned_rewards, output_file):
    """
    Plot comparison of TD rewards between base and fine-tuned models.
    
    Args:
        base_rewards: Average TD rewards for base model
        fine_tuned_rewards: Average TD rewards for fine-tuned model
        output_file: File to save plot
        
    Returns:
        None (saves plot to output_file)
    """
    labels = ["Base Model", "Fine-tuned Model"]
    rewards = [base_rewards, fine_tuned_rewards]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, rewards, color=["blue", "orange"])
    plt.ylabel("Average TD Reward")
    plt.title("Comparison of TD Rewards")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(rewards):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Rewards comparison plot saved to {output_file}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base and fine-tuned models
    print(f"Loading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    
    print(f"Loading fine-tuned model: {args.fine_tuned_model}")
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(args.fine_tuned_model)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(args.fine_tuned_model).to(device)
    
    # Initialize reward model
    reward_model = TDRewardModel(model_name=args.base_model)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    
    # Evaluate perplexity
    print("Evaluating perplexity...")
    base_ppl = evaluate_perplexity(
        base_model, base_tokenizer, dataset, args.max_length, args.batch_size, device
    )
    fine_tuned_ppl = evaluate_perplexity(
        fine_tuned_model, fine_tuned_tokenizer, dataset, args.max_length, args.batch_size, device
    )
    
    print(f"Base model perplexity: {base_ppl:.4f}")
    print(f"Fine-tuned model perplexity: {fine_tuned_ppl:.4f}")
    
    # Evaluate TD rewards
    print("Evaluating TD rewards...")
    base_rewards, fine_tuned_rewards = evaluate_td_rewards(
        base_model, fine_tuned_model, reward_model, dataset,
        args.max_length, args.batch_size, device
    )
    
    print(f"Base model average TD reward: {base_rewards:.4f}")
    print(f"Fine-tuned model average TD reward: {fine_tuned_rewards:.4f}")
    
    # Generate and compare samples
    print("Generating text samples...")
    generate_and_compare(
        base_model, fine_tuned_model, base_tokenizer,
        dataset, args.num_samples, device, args.output_file
    )
    
    # Plot rewards comparison
    plot_rewards_comparison(
        base_rewards, fine_tuned_rewards,
        output_file="rewards_comparison.png"
    )
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
