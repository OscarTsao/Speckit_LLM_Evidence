#!/bin/bash
# Complete environment setup script

set -e

echo "========================================="
echo "Speckit LLM Evidence - Environment Setup"
echo "========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Remove existing environment if it exists
if conda env list | grep -q "speckit-llm-evidence"; then
    echo "Removing existing environment..."
    conda env remove -n speckit-llm-evidence -y
fi

# Create conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate speckit-llm-evidence

# Verify CUDA setup
echo ""
echo "Verifying CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo ""
    echo "WARNING: CUDA is not available. GPU training will not work."
    echo "Please check your NVIDIA drivers and CUDA installation."
fi

# Install pre-commit hooks
echo ""
echo "Installing pre-commit hooks..."
pre-commit install

# Setup PostgreSQL for MLflow
echo ""
read -p "Do you want to set up PostgreSQL for MLflow? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash scripts/setup_postgresql.sh
else
    echo "Skipping PostgreSQL setup. You can run scripts/setup_postgresql.sh later."
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/redsm5
mkdir -p mlruns
mkdir -p logs
mkdir -p outputs/checkpoints
mkdir -p outputs/predictions

# Download dataset
echo ""
read -p "Do you want to download the redsm5 dataset now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading dataset..."
    python scripts/download_dataset.py
else
    echo "Skipping dataset download. Run 'python scripts/download_dataset.py' later."
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate speckit-llm-evidence"
echo ""
echo "To start MLflow UI (if PostgreSQL is set up), run:"
echo "  mlflow server --backend-store-uri \$MLFLOW_TRACKING_URI --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"
echo ""
echo "Next steps:"
echo "  1. Review configs/training_config.yaml"
echo "  2. Download dataset: python scripts/download_dataset.py"
echo "  3. Start training: python scripts/train.py"
echo ""
