"""
Training script for a language model with TD-inspired reward function.

This script implements a simplified Proximal Policy Optimization (PPO) algorithm
to train a language model using the TD-inspired reward function.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import wandb
from tqdm import tqdm

from td_reward import TDRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM with TD-inspired reward")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use for training")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for retrospective penalty")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for exploration bonus")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--log_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPOTrainer:
    """
    Implements PPO training for language models with TD-inspired rewards.
    """
    
    def __init__(
        self,
        args,
        model,
        tokenizer,
        reward_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.args = args
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.device = device
        
        # Initialize optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.value_optimizer = optim.Adam(
            list(self.reward_model.belief_state_model.parameters()) +
            list(self.reward_model.value_function.parameters()),
            lr=args.lr
        )
        
        # PPO-specific parameters
        self.eps_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Reference model (for KL divergence calculation)
        self.ref_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        epoch_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "td_reward": 0.0,
            "kl_div": 0.0,
        }
        
        for batch in tqdm(dataloader, desc="Training"):
            # Process batch
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Split into context and target
            # (Use all but last token as context, last token as target)
            context_ids = input_ids[:, :-1]
            context_mask = attention_mask[:, :-1]
            target_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Forward pass with current policy
            outputs = self.model(
                input_ids=context_ids,
                attention_mask=context_mask,
                return_dict=True
            )
            
            # Get action probabilities
            logits = outputs.logits[:, -1, :]  # Last token predictions
            probs = torch.softmax(logits, dim=-1)
            
            # Forward pass with reference model (for KL penalty)
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=context_ids,
                    attention_mask=context_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits[:, -1, :]
                ref_probs = torch.softmax(ref_logits, dim=-1)
            
            # Sample actions (next tokens) from the policy
            dist = torch.distributions.Categorical(probs)
            action_ids = dist.sample()
            
            # Compute log probabilities
            log_probs = dist.log_prob(action_ids)
            ref_log_probs = torch.log(torch.gather(ref_probs, 1, action_ids.unsqueeze(-1)).squeeze(-1) + 1e-10)
            
            # Compute TD rewards for each sequence in batch
            rewards = []
            for i in range(input_ids.size(0)):
                reward = self.reward_model.compute_td_reward(
                    context=context_ids[i].unsqueeze(0),
                    prediction=action_ids[i].unsqueeze(0),
                    actual_next_token=target_ids[i]
                )
                rewards.append(reward)
            
            rewards = torch.tensor(rewards, device=self.device)
            
            # Compute KL divergence from reference model
            kl_div = (log_probs - ref_log_probs).mean()
            
            # Add KL penalty to rewards
            kl_penalty = 0.1  # Weight for KL penalty
            rewards = rewards - kl_penalty * kl_div
            
            # Get value estimates
            with torch.no_grad():
                # Extract hidden states from model
                context_outputs = self.model(
                    input_ids=context_ids,
                    attention_mask=context_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = context_outputs.hidden_states[-1][:, -1]  # Last token, last layer
                
                # Compute belief states
                belief_states = self.reward_model.belief_state_model(
                    hidden_states.unsqueeze(1)
                ).squeeze(1)
                
                # Get value estimates
                values = self.reward_model.value_function(
                    hidden_states, belief_states
                ).squeeze(-1)
            
            # Compute advantage estimates
            advantages = rewards - values
            
            # PPO policy loss
            ratio = torch.exp(log_probs - ref_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = self.value_coef * (rewards - values).pow(2).mean()
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # Total loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.value_optimizer.step()
            
            # Update statistics
            epoch_stats["policy_loss"] += policy_loss.item()
            epoch_stats["value_loss"] += value_loss.item()
            epoch_stats["td_reward"] += rewards.mean().item()
            epoch_stats["kl_div"] += kl_div.item()
        
        # Compute average statistics
        for key in epoch_stats:
            epoch_stats[key] /= len(dataloader)
        
        return epoch_stats
    
    def save_checkpoint(self, epoch, output_dir):
        """Save model checkpoint."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save model checkpoint
        model_path = os.path.join(output_dir, f"model_epoch_{epoch}")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save reward model components
        reward_path = os.path.join(output_dir, f"reward_model_epoch_{epoch}")
        if not os.path.exists(reward_path):
            os.makedirs(reward_path)
        
        torch.save(
            self.reward_model.belief_state_model.state_dict(),
            os.path.join(reward_path, "belief_state_model.pt")
        )
        torch.save(
            self.reward_model.value_function.state_dict(),
            os.path.join(reward_path, "value_function.pt")
        )
        
        print(f"Saved checkpoint for epoch {epoch} to {output_dir}")


def prepare_dataset(args, tokenizer):
    """Prepare dataset for training."""
    # Load dataset
    dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    return dataloader


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize WandB
    if args.log_wandb:
        wandb.init(
            project="td-inspired-llm",
            config=vars(args)
        )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Initialize TD reward model
    reward_model = TDRewardModel(
        model_name=args.base_model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Prepare dataset
    dataloader = prepare_dataset(args, tokenizer)
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model
    )
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        epoch_stats = trainer.train_epoch(dataloader)
        
        # Log metrics
        print(f"Epoch {epoch+1} stats:")
        for key, value in epoch_stats.items():
            print(f"  {key}: {value:.4f}")
        
        if args.log_wandb:
            wandb.log(epoch_stats)
        
        # Save checkpoint
        trainer.save_checkpoint(epoch+1, args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
