#!/bin/bash

# Setup script for TD-Inspired LLM project
# This script sets up the environment and runs a simple experiment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up TD-Inspired LLM project...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed. For Ollama integration, please install it from https://ollama.ai/download${NC}"
    echo -e "${YELLOW}Continuing without Ollama...${NC}"
fi

# Create output directories
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p output
mkdir -p experiment_results

# Run a small experiment
echo -e "${YELLOW}Running a small experiment with TD reward model...${NC}"
echo -e "${YELLOW}This will download small models like GPT-2 if not already downloaded.${NC}"

python run_experiment.py \
  --models gpt2 distilgpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --num_samples 10 \
  --output_dir ./experiment_results

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}Experiment results are in ./experiment_results${NC}"
echo -e "${GREEN}For next steps, see README.md${NC}"

# Deactivate virtual environment
deactivate
