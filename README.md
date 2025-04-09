# TD-Inspired Reward Function for LLMs

A neuroscience-inspired approach to language model evaluation and training, based on temporal difference (TD) learning and dopamine signaling research.

## Overview

This project implements a reward function for language models inspired by temporal difference (TD) learning and dopamine signaling, as described in research on contingency learning and prediction. The approach favors "prospective" predictions over retrospective associations, aligning with research showing that dopamine responses primarily reflect forward-looking predictive relationships.

## Key Findings

From our experiments comparing standard GPT-2 and instruction-tuned models:

- Both general and instruction-tuned models receive similar TD rewards, suggesting that while instruction-tuned models generate more helpful and coherent responses, they don't necessarily predict the next token better according to TD-inspired metrics.
- Instruction-tuned models tend to generate longer, more detailed responses, optimized for helpfulness rather than pure next-token prediction.
- TD reward values are generally negative for current models, indicating room for improvement in training models better at prospective prediction.
- The belief-state representation is crucial for properly evaluating models according to TD learning principles, capturing the temporal dynamics of language contexts.

## Core Concepts

### Temporal Difference Learning

TD learning is a reinforcement learning technique where models learn to predict future rewards based on current states. The TD error is calculated as:

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

Where:
- `r_t` is the immediate reward at time t
- `V(s_t)` is the predicted value of state s_t
- `γ` is the discount factor
- `V(s_{t+1})` is the predicted value of the next state

In our implementation:
- The state s_t is represented by the context (previous tokens)
- The next state s_{t+1} is the context plus the predicted token
- The immediate reward is a function of prediction accuracy, contingency, and exploration

### Belief-State Representation

The Belief-State model uses a neural network to learn a probability distribution over possible states, allowing it to capture uncertainty and temporal dynamics. This is inspired by research showing that dopamine neurons are sensitive to state uncertainty and hidden-state inference.

## Installation

```bash
# Clone the repository
git clone https://github.com/angrysky56/td_inspired_llm.git
cd td_inspired_llm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (optional, for local model deployment)
# See https://ollama.ai/download for installation instructions
```

## Usage

### Quick Start

The easiest way to get started is to run the setup script, which will create a virtual environment, install dependencies, and run a small experiment:

```bash
chmod +x setup.sh
./setup.sh
```

### Evaluating Models with TD Reward

You can evaluate different models using the TD reward function:

```bash
python compare_models.py --models gpt2 distilgpt2 --num_samples 10
```

To evaluate Ollama models, prefix the model name with `ollama:`:

```bash
python compare_models.py --models gpt2 ollama:Hudson/gpt2-instruct:345m-q8_0 --num_samples 5
```

### Testing a Specific Model

```bash
python test_gpt2_instruct.py --ollama_model Hudson/gpt2-instruct:345m-q8_0 --num_samples 10
```

### Training with TD-Inspired Reward

#### Standard Training

```bash
python train.py \
  --base_model gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --output_dir ./output \
  --epochs 3 \
  --batch_size 4 \
  --alpha 0.1 \
  --beta 0.01 \
  --gamma 0.99
```

#### LoRA Fine-Tuning

For more efficient training of larger models, you can use Low-Rank Adaptation (LoRA) with our TD-inspired reward function:

```bash
python lora_trainer.py \
  --base_model gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --output_dir ./lora_output \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --use_8bit  # Enable for larger models
  --batch_size 4 \
  --num_epochs 3 \
  --alpha 0.1 \
  --beta 0.01 \
  --gamma 0.99
```

For 4-bit quantization (for even larger models):

```bash
python lora_trainer.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --use_4bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --output_dir ./llama2_lora_output
```

### Ollama Integration

#### Creating an Ollama Model

```bash
python ollama_integration.py \
  --model_path ./output/model_epoch_3 \
  --model_name td-llm \
  --action create
```

#### Running an Ollama Model

```bash
python ollama_integration.py \
  --model_name td-llm \
  --prompt "The quick brown fox jumps over the" \
  --action run
```

#### Evaluating an Ollama Model

```bash
python ollama_integration.py \
  --model_name td-llm \
  --base_model gpt2 \
  --reward_model_path ./output/reward_model_epoch_3 \
  --action evaluate
```

#### Fine-tuning with Ollama

```bash
python ollama_finetune.py \
  --base_model llama2:7b \
  --output_model td-llama \
  --dataset ./data/sample_training_data.jsonl \
  --epochs 3 \
  --alpha 0.1 \
  --beta 0.01 \
  --gamma 0.99
```

## Components

### 1. TD Reward Module (`td_reward.py`)

The core implementation of the TD-inspired reward function:

- `BeliefStateModel`: Implements a belief state representation for language contexts
- `ValueFunction`: Estimates the value (expected future reward) for a given state
- `TDRewardModel`: Computes rewards based on TD errors

### 2. Training Script (`train.py`)

Implements a PPO-based reinforcement learning algorithm for fine-tuning language models using the TD-inspired reward function.

### 3. Evaluation Tools (`evaluate.py`, `run_experiment.py`, `compare_models.py`)

Tools for evaluating and comparing different models using the TD-inspired reward function.

### 4. LoRA Fine-Tuning (`lora_trainer.py`)

Implements Low-Rank Adaptation (LoRA) fine-tuning with TD-inspired reward for efficient training of large language models:

- Supports 8-bit and 4-bit quantization for larger models
- Custom `TDRewardTrainer` class that incorporates TD reward into the loss function
- Compatible with Hugging Face's PEFT library

### 5. Ollama Integration (`ollama_integration.py`, `ollama_finetune.py`)

Integration with Ollama for local deployment and fine-tuning of models.

## Development

### Project Structure

```
td_inspired_llm/
├── td_reward.py         # Core implementation of TD reward function
├── train.py             # Training script for fine-tuning models
├── evaluate.py          # Evaluation script for fine-tuned models
├── run_experiment.py    # Script for running comparative experiments
├── ollama_integration.py # Integration with Ollama for local deployment
├── ollama_finetune.py   # Fine-tuning script using Ollama
├── test_gpt2_instruct.py # Script for testing GPT-2 instruct models
├── compare_models.py    # Script for comparing different models
├── data/                # Sample training data
├── setup.sh             # Setup script
└── requirements.txt     # Project dependencies
```

### Adding New Models

To add support for a new model:

1. For HuggingFace models, simply specify the model name when calling the evaluation scripts.
2. For Ollama models, ensure the model is available in Ollama and prefix the model name with `ollama:` when using the comparison tools.

### Customizing the TD Reward Function

You can customize the TD reward function by adjusting the following parameters:

- `alpha`: Weight for retrospective penalty component (default: 0.1)
- `beta`: Weight for exploration/novelty bonus (default: 0.01)
- `gamma`: Discount factor for future rewards (default: 0.99)

These parameters can be adjusted in the command-line arguments for the various scripts.

### Implementing Custom State Representations

The project currently uses a belief-state representation with two states (Wait and Pre-transition). You can implement custom state representations by extending the `BeliefStateModel` class in `td_reward.py`.

## Future Directions

1. **Fine-tuning with TD Rewards**: Implement a full fine-tuning pipeline that uses the TD reward as a training signal.
2. **Multi-Reward Learning**: Combine TD rewards with other objectives (like human preference) to train models that are both good at prediction and aligned with human values.
3. **Improved State Representations**: Develop more sophisticated belief-state representations that better capture the temporal dynamics of language.
4. **Larger-Scale Evaluation**: Test the TD reward function on a wider range of models and tasks.
5. **Real-time Adaptation**: Use the TD reward signal for real-time adaptation of model outputs.

## Acknowledgements

This project is inspired by research on dopamine signaling and TD learning, particularly the paper "The role of prospective contingency in the control of behavior and dopamine signals during associative learning" by Qian et al. (2024).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{TDInspiredLLM,
  author = {AngryStorm},
  title = {TD-Inspired Reward Function for LLMs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/angrysky56/td_inspired_llm}}
}
```
