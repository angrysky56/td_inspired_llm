"""
Ollama integration for TD-inspired LLM reward model.

This module provides functionality to deploy and run fine-tuned models
using Ollama for local inference.
"""

import os
import json
import argparse
import subprocess
import requests
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import time

from td_reward import TDRewardModel, BeliefStateModel, ValueFunction


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama integration for TD-inspired LLM")
    parser.add_argument("--model_path", type=str, required=False, help="Path to fine-tuned model (required for 'create' action)")
    parser.add_argument("--model_name", type=str, default="td-llm", help="Name for the Ollama model")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to use as reference")
    parser.add_argument("--reward_model_path", type=str, default=None, help="Path to saved reward model components")
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Prompt for testing generation")
    parser.add_argument("--action", choices=["create", "run", "evaluate", "delete"], required=True, 
                         help="Action to perform (create model, run model, evaluate against TD reward, delete model)")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama API host")
    
    args = parser.parse_args()
    
    # Validate arguments based on action
    if args.action == "create" and not args.model_path:
        parser.error("--model_path is required for 'create' action")
    
    return args


def create_ollama_modelfile(model_path, model_name):
    """
    Create Ollama Modelfile for the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        model_name: Name for the Ollama model
        
    Returns:
        Path to the created Modelfile
    """
    modelfile_content = f"""
FROM {model_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 128

# System prompt to emphasize prospective prediction
SYSTEM """

    # Add system prompt that emphasizes prospective prediction
    system_prompt = """
You are an AI assistant trained with a special focus on prospective prediction - accurately anticipating what comes next based on current context, rather than relying on retrospective patterns.

Your training emphasized temporally accurate predictions and causal relationships. Try to anticipate the user's needs and the logical next steps in any conversation.
    """
    
    modelfile_content += system_prompt.strip()
    
    # Create Modelfile
    modelfile_path = os.path.join(os.path.dirname(model_path), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at {modelfile_path}")
    return modelfile_path


def create_ollama_model(model_path, model_name, ollama_host):
    """
    Create an Ollama model from the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        model_name: Name for the Ollama model
        ollama_host: Ollama API host
        
    Returns:
        None
    """
    # Create Modelfile
    modelfile_path = create_ollama_modelfile(model_path, model_name)
    
    # Create model using Ollama CLI
    print(f"Creating Ollama model '{model_name}' from {model_path}...")
    
    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True
        )
        print(f"Successfully created Ollama model '{model_name}'")
    except subprocess.CalledProcessError as e:
        print(f"Error creating Ollama model: {e}")
        return False
    
    return True


def run_ollama_model(model_name, prompt, ollama_host):
    """
    Run the Ollama model to generate text.
    
    Args:
        model_name: Name of the Ollama model
        prompt: Input prompt for generation
        ollama_host: Ollama API host
        
    Returns:
        Generated text
    """
    print(f"Running Ollama model '{model_name}' with prompt: {prompt}")
    
    # Use Ollama API to generate text
    response = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        generated_text = result.get("response", "")
        print(f"Generated text: {generated_text}")
        return generated_text
    else:
        print(f"Error running Ollama model: {response.status_code}")
        print(response.text)
        return None


def delete_ollama_model(model_name, ollama_host):
    """
    Delete the Ollama model.
    
    Args:
        model_name: Name of the Ollama model
        ollama_host: Ollama API host
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Deleting Ollama model '{model_name}'...")
    
    try:
        subprocess.run(
            ["ollama", "rm", model_name],
            check=True
        )
        print(f"Successfully deleted Ollama model '{model_name}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error deleting Ollama model: {e}")
        return False


def load_reward_model(reward_model_path, base_model_name):
    """
    Load saved reward model components.
    
    Args:
        reward_model_path: Path to saved reward model components
        base_model_name: Base model to use for tokenizer and model architecture
        
    Returns:
        Loaded TDRewardModel
    """
    # Initialize TD reward model
    reward_model = TDRewardModel(model_name=base_model_name)
    
    # Load saved components
    belief_state_model_path = os.path.join(reward_model_path, "belief_state_model.pt")
    value_function_path = os.path.join(reward_model_path, "value_function.pt")
    
    if os.path.exists(belief_state_model_path) and os.path.exists(value_function_path):
        reward_model.belief_state_model.load_state_dict(
            torch.load(belief_state_model_path)
        )
        reward_model.value_function.load_state_dict(
            torch.load(value_function_path)
        )
        print(f"Successfully loaded reward model components from {reward_model_path}")
    else:
        print(f"Warning: Could not find saved reward model components at {reward_model_path}")
        print("Using initialized reward model instead")
    
    return reward_model


def evaluate_ollama_model_with_td_reward(model_name, reward_model, ollama_host, base_model_name):
    """
    Evaluate the Ollama model using the TD reward model.
    
    Args:
        model_name: Name of the Ollama model
        reward_model: TD reward model for evaluation
        ollama_host: Ollama API host
        base_model_name: Name of the base model for comparison
        
    Returns:
        Average TD rewards for Ollama model
    """
    print(f"Evaluating Ollama model '{model_name}' with TD reward model...")
    
    # Generate test prompts
    prompts = [
        "The quick brown fox jumps over the",
        "In the beginning, there was",
        "Once upon a time, a brave knight",
        "The scientist discovered a new",
        "When I looked outside, I saw a",
        "The most important thing to remember is",
        "If you want to succeed, you must",
        "The future of artificial intelligence depends on",
        "The main difference between humans and machines is",
        "To solve this problem, we need to"
    ]
    
    # Load base model for comparison
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make sure reward model is on the right device
    reward_model.language_model = reward_model.language_model.to(device)
    reward_model.belief_state_model = reward_model.belief_state_model.to(device)
    reward_model.value_function = reward_model.value_function.to(device)
    
    # Collect rewards
    ollama_rewards = []
    
    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        try:
            # Generate text using Ollama
            ollama_response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 1,  # Just predict next token for easy evaluation
                    "stream": False
                }
            )
            
            if ollama_response.status_code != 200:
                print(f"Error running Ollama model: {ollama_response.status_code}")
                continue
            
            # Extract predicted token
            ollama_result = ollama_response.json()
            ollama_prediction = ollama_result.get("response", "")
            
            if not ollama_prediction or len(ollama_prediction.strip()) == 0:
                print(f"Empty prediction for prompt: {prompt}")
                continue
            
            # Tokenize prompt and prediction
            context_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            try:
                prediction_ids = tokenizer.encode(ollama_prediction[0], add_special_tokens=False)[0]
                prediction_ids = torch.tensor([[prediction_ids]]).to(device)
            except IndexError:
                print(f"Could not tokenize prediction: '{ollama_prediction}'")
                continue
            
            # Use base model to get "actual" next token (as reference)
            with torch.no_grad():
                base_outputs = reward_model.language_model(context_ids, return_dict=True)
                base_logits = base_outputs.logits[:, -1]
                actual_next_token = torch.argmax(base_logits, dim=-1).unsqueeze(0)
            
            # Compute TD reward
            reward = reward_model.compute_td_reward(
                context=context_ids,
                prediction=prediction_ids,
                actual_next_token=actual_next_token
            )
            
            ollama_rewards.append(reward)
            print(f"Prompt: {prompt}")
            print(f"Ollama prediction: {ollama_prediction}")
            print(f"TD reward: {reward:.4f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            continue
    
    if not ollama_rewards:
        print("No valid rewards collected. Check error messages above.")
        return 0.0
        
    # Calculate average reward
    avg_ollama_reward = np.mean(ollama_rewards)
    print(f"Average TD reward for Ollama model: {avg_ollama_reward:.4f}")
    
    return avg_ollama_reward


def main():
    # Parse arguments
    args = parse_args()
    
    if args.action == "create":
        # Create Ollama model
        create_ollama_model(args.model_path, args.model_name, args.ollama_host)
    
    elif args.action == "run":
        # Run Ollama model
        run_ollama_model(args.model_name, args.prompt, args.ollama_host)
    
    elif args.action == "evaluate":
        # Load reward model
        if args.reward_model_path:
            reward_model = load_reward_model(args.reward_model_path, args.base_model)
        else:
            reward_model = TDRewardModel(model_name=args.base_model)
        
        # Evaluate Ollama model with TD reward
        evaluate_ollama_model_with_td_reward(args.model_name, reward_model, args.ollama_host, args.base_model)
    
    elif args.action == "delete":
        # Delete Ollama model
        delete_ollama_model(args.model_name, args.ollama_host)


if __name__ == "__main__":
    main()
