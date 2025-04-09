"""
Simple test script for using a LoRA-adapted model.
This script tests text generation with a LoRA model without complex evaluation.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Define test parameters
MODEL_NAME = "distilgpt2"  # Base model
LORA_ADAPTER = "./lora_test_output"  # Path to LoRA adapter
TEST_PROMPTS = [
    "The quick brown fox jumps over the",
    "In the beginning, there was",
    "Once upon a time, a brave knight"
]

print(f"Testing LoRA model generation with base model: {MODEL_NAME}")
print(f"LoRA adapter: {LORA_ADAPTER}")

# Force CPU to avoid CUDA issues
device = "cpu"

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
try:
    config = PeftConfig.from_pretrained(LORA_ADAPTER)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    lora_model = lora_model.to(device)
    
    print("Successfully loaded the LoRA model")
    
    # Generate text with LoRA model
    print("\nGenerating text with LoRA model:")
    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = lora_model.generate(
            inputs.input_ids,
            max_length=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    
    # Also generate with base model for comparison
    print("\nGenerating text with base model (no LoRA):")
    base_model = base_model.to(device)
    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = base_model.generate(
            inputs.input_ids,
            max_length=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    
except Exception as e:
    print(f"Error loading or using LoRA model: {e}")
