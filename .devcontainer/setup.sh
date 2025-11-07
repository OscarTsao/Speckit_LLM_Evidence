#!/bin/bash
set -e

echo "Setting up Speckit LLM Evidence development environment..."

# Initialize conda for bash
source /opt/conda/etc/profile.d/conda.sh

# Create conda environment from environment.yml
echo "Creating conda environment..."
cd /workspaces/Speckit_LLM_Evidence
conda env create -f environment.yml

# Activate the environment
conda activate speckit-llm-evidence

# Install the package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Verify CUDA availability
echo "Verifying CUDA setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "Setup complete! Environment: speckit-llm-evidence"
