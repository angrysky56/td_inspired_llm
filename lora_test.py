"""
Test script for the LoRA trainer with TD-inspired reward.
This script runs a minimal test to ensure the LoRA trainer works correctly.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from td_reward import TDRewardModel
from lora_trainer import TDRewardTrainer, prepare_lora_config

# Define test parameters
MODEL_NAME = "distilgpt2"  # Smaller model for faster testing
OUTPUT_DIR = "./lora_test_output"
NUM_SAMPLES = 10
BATCH_SIZE = 2
NUM_EPOCHS = 1
MAX_LENGTH = 32

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Testing LoRA training with model: {MODEL_NAME}")
print(f"Using {NUM_SAMPLES} samples, {BATCH_SIZE} batch size, {NUM_EPOCHS} epoch")

# Initialize reward model
reward_model = TDRewardModel(
    model_name=MODEL_NAME,
    alpha=0.1,
    beta=0.01,
    gamma=0.99
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Prepare model for LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,  # Small rank for testing
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],  # For DistilGPT-2
    bias="none"
)

model = get_peft_model(model, lora_config)
print(f"LoRA model prepared with rank: {lora_config.r}")

# Load a tiny dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10]")
print(f"Loaded dataset with {len(dataset)} samples")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Set up training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=1,
    save_steps=5,
    save_total_limit=1,
    no_cuda=not torch.cuda.is_available(),
    push_to_hub=False,
    report_to="none",
)

# Create data collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize TD reward trainer
trainer = TDRewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    reward_model=reward_model,
    alpha=0.1,
    beta=0.01,
    gamma=0.99,
    reward_scale=0.5
)

# Train for a single step to test
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
    
    # Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
    # Test loading the model
    from peft import PeftModel, PeftConfig
    
    config = PeftConfig.from_pretrained(OUTPUT_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    print("Successfully loaded the fine-tuned LoRA model")
    
    # Test generation with the fine-tuned model
    prompt = "The quick brown fox jumps over the"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = lora_model.generate(
        inputs.input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
    
except Exception as e:
    print(f"Error during training: {e}")
