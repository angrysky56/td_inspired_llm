"""
Low-Rank Adaptation (LoRA) training script for TD-inspired LLM reward.

This script implements LoRA fine-tuning using the TD-inspired reward function
for more efficient training of language models.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftConfig
)

from td_reward import TDRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with TD-inspired reward")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use for training")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Directory to save LoRA weights")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=0.1, help="TD reward alpha (retrospective penalty weight)")
    parser.add_argument("--beta", type=float, default=0.01, help="TD reward beta (exploration bonus weight)")
    parser.add_argument("--gamma", type=float, default=0.99, help="TD reward gamma (discount factor)")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Scale factor for TD reward")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TDRewardTrainer(Trainer):
    """
    Custom Trainer class that uses TD reward for loss calculation.
    """
    
    def __init__(self, reward_model, alpha, beta, gamma, reward_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reward_scale = reward_scale
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss using TD reward.
        
        Note: Added **kwargs to handle any additional parameters passed by the Trainer class.
        """
        # Standard language modeling loss
        outputs = model(**inputs)
        standard_loss = outputs.loss
        
        # Calculate TD reward
        # We'll compute TD reward on a subset of the batch to save computation
        batch_size = inputs["input_ids"].size(0)
        reward_batch_size = min(batch_size, 2)  # Use at most 2 examples for reward calculation
        
        td_rewards = []
        device = inputs["input_ids"].device
        
        for i in range(reward_batch_size):
            # Use all but last token as context and last token as target
            context_ids = inputs["input_ids"][i:i+1, :-1]
            target_id = inputs["input_ids"][i:i+1, -1]
            
            # Get model's prediction
            with torch.no_grad():
                pred_outputs = model(context_ids)
                pred_logits = pred_outputs.logits[:, -1]
                pred_token_id = torch.argmax(pred_logits, dim=-1).unsqueeze(-1)
            
            # Calculate TD reward
            with torch.no_grad():
                try:
                    reward = self.reward_model.compute_td_reward(
                        context=context_ids,
                        prediction=pred_token_id,
                        actual_next_token=target_id
                    )
                    td_rewards.append(reward)
                except Exception as e:
                    print(f"Error calculating TD reward: {e}")
                    # Use a default penalty if reward calculation fails
                    td_rewards.append(-1.0)
        
        # Average rewards and apply scale
        avg_reward = np.mean(td_rewards) if td_rewards else 0.0
        reward_term = -self.reward_scale * avg_reward  # Negative because we want to maximize reward
        
        # Combined loss: standard LM loss + reward-based loss
        combined_loss = standard_loss + reward_term
        
        return (combined_loss, outputs) if return_outputs else combined_loss


def prepare_lora_config(args):
    """
    Prepare LoRA configuration.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )


def load_model_and_tokenizer(args):
    """
    Load base model and tokenizer with LoRA configuration.
    """
    print(f"Loading base model: {args.base_model}")
    
    # Set BitsAndBytes configuration if needed
    bnb_config = None
    if args.use_8bit or args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit,
                load_in_8bit=args.use_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            print(f"Using {'4-bit' if args.use_4bit else '8-bit'} quantization")
        except ImportError:
            print("BitsAndBytes not available. Continuing without quantization.")
            bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for LoRA fine-tuning
    if args.use_8bit or args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA config
    lora_config = prepare_lora_config(args)
    model = get_peft_model(model, lora_config)
    
    print(f"Model prepared with LoRA (rank={args.lora_r}, alpha={args.lora_alpha})")
    
    return model, tokenizer


def prepare_dataset(args, tokenizer):
    """
    Prepare dataset for training.
    """
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    
    max_length = args.max_length
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    return tokenized_dataset, data_collator


def train_model(args, model, tokenizer, dataset, data_collator, reward_model):
    """
    Train the model using LoRA and TD reward.
    """
    print("Setting up training arguments...")
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard etc.
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize TD reward trainer
    trainer = TDRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        reward_model=reward_model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        reward_scale=args.reward_scale
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return model


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Initialize TD reward model
    print(f"Initializing TD reward model (alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})")
    reward_model = TDRewardModel(
        model_name=args.base_model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Prepare dataset
    dataset, data_collator = prepare_dataset(args, tokenizer)
    
    # Train model
    model = train_model(args, model, tokenizer, dataset, data_collator, reward_model)
    
    print("Training complete!")
    print(f"LoRA adapter saved to {args.output_dir}")


def test_lora_model(base_model, lora_adapter, prompt, max_length=100):
    """
    Test a LoRA-adapted model.
    """
    # Load config
    config = PeftConfig.from_pretrained(lora_adapter)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter)
    
    # Generate text
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    main()
