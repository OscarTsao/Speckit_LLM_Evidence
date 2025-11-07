# Speckit LLM Evidence - Build Guide

This document provides comprehensive instructions for building and setting up the Speckit LLM Evidence project using the Spec-Kit methodology.

## Prerequisites

- **Hardware**: NVIDIA RTX 4090 GPU (or compatible GPU with 24GB+ VRAM)
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **NVIDIA Drivers**: Version 525+ (for CUDA 12.1)
- **Conda**: Miniconda or Anaconda installed
- **PostgreSQL**: Version 12+ (can be installed via setup script)
- **Git**: For version control

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Speckit_LLM_Evidence
```

### 2. Run Setup Script

```bash
bash scripts/setup_environment.sh
```

This script will:
- Create conda environment from `environment.yml`
- Install all dependencies (conda + pip)
- Verify CUDA installation
- Set up pre-commit hooks
- Optionally set up PostgreSQL for MLflow
- Optionally download the redsm5 dataset

### 3. Activate Environment

```bash
conda activate speckit-llm-evidence
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print('Transformers installed')"
```

## Manual Setup

If you prefer manual setup or the automated script fails:

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate speckit-llm-evidence
```

### Step 2: Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi  # Should show your RTX 4090
```

### Step 3: Set Up PostgreSQL

```bash
bash scripts/setup_postgresql.sh
```

Or manually:

```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
CREATE DATABASE mlflow_db OWNER mlflow_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
\q
```

Create `.env` file:

```bash
cat > .env << EOF
MLFLOW_TRACKING_URI=postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db
MLFLOW_ARTIFACT_ROOT=file:./mlruns
EOF
```

### Step 4: Download Dataset

```bash
python scripts/download_dataset.py
```

### Step 5: Install Pre-commit Hooks

```bash
pre-commit install
```

## Development Container

For VS Code users, the project includes a dev container with full CUDA support:

### Using Dev Container

1. Install **Docker** and **VS Code** with **Dev Containers** extension
2. Ensure NVIDIA Container Toolkit is installed:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3. Open project in VS Code
4. Press `F1` → "Dev Containers: Reopen in Container"
5. Wait for container to build (first time takes 10-15 minutes)

The dev container will automatically:
- Set up CUDA 12.1 environment
- Install Miniconda
- Create conda environment
- Install all dependencies
- Configure VS Code extensions

## Project Structure

```
Speckit_LLM_Evidence/
├── .devcontainer/          # Dev container configuration
│   ├── devcontainer.json   # Container settings
│   ├── Dockerfile          # CUDA-enabled container
│   └── setup.sh            # Container setup script
├── .github/workflows/      # CI/CD workflows
│   ├── spec-validation.yml # Spec-kit validation
│   ├── code-quality.yml    # Linting and testing
│   └── build-test.yml      # Build verification
├── .specify/               # Spec-kit specifications
│   ├── memory/
│   │   └── constitution.md # Project principles
│   └── specs/
│       └── 01-extractive-qa-system.md
├── configs/                # Configuration files
│   ├── mlflow_config.yaml
│   └── training_config.yaml
├── data/                   # Dataset storage
│   └── redsm5/             # Downloaded dataset
├── scripts/                # Utility scripts
│   ├── setup_environment.sh
│   ├── setup_postgresql.sh
│   ├── download_dataset.py
│   ├── train.py            # (to be implemented)
│   └── evaluate.py         # (to be implemented)
├── src/                    # Source code
│   └── Project/SubProject/
│       ├── data/
│       ├── models/
│       ├── engine/
│       └── utils/
├── tests/                  # Unit tests
├── mlruns/                 # MLflow artifacts
├── logs/                   # Training logs
├── environment.yml         # Conda environment
├── requirements.txt        # Pip dependencies
├── pyproject.toml          # Project metadata
└── BUILD.md                # This file
```

## Configuration

### Training Configuration

Edit `configs/training_config.yaml`:

```yaml
model:
  name: "google/gemma-2b"  # or "google/gemma-7b"

data:
  batch_size: 8             # Adjust based on VRAM
  max_seq_length: 512

training:
  learning_rate: 2.0e-5
  num_epochs: 5
  gradient_accumulation_steps: 4
```

### MLflow Configuration

Edit `configs/mlflow_config.yaml` or set environment variables in `.env`:

```bash
MLFLOW_TRACKING_URI=postgresql://user:pass@localhost:5432/mlflow_db
MLFLOW_ARTIFACT_ROOT=file:./mlruns
```

## Running MLflow UI

```bash
source .env
mlflow server \
  --backend-store-uri $MLFLOW_TRACKING_URI \
  --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000
```

Access at: http://localhost:5000

## Building the Project

### Linting and Formatting

```bash
# Check code style
ruff check src tests

# Format code
black src tests

# Type checking
mypy src
```

### Running Tests

```bash
pytest tests/ -v
```

### Training (Once Implemented)

```bash
python scripts/train.py --config configs/training_config.yaml
```

## Spec-Kit Workflow

This project follows the Spec-Kit methodology:

1. **Constitution** (`.specify/memory/constitution.md`): Core principles
2. **Specifications** (`.specify/specs/`): What to build
3. **Implementation**: Build according to specs
4. **Validation**: CI/CD ensures compliance

### Key Principles

- **Reproducibility First**: All experiments tracked in MLflow
- **Performance Optimization**: Optimized for RTX 4090
- **Data Integrity**: Clean train/val/test splits
- **Code Quality**: Linting, type hints, tests required

## CI/CD

GitHub Actions automatically run:

- **Spec Validation**: Ensures specs are properly formatted
- **Code Quality**: Linting, formatting, type checking
- **Build Test**: Verifies environment builds correctly

## Troubleshooting

### CUDA Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### PostgreSQL Connection Failed

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U mlflow_user -d mlflow_db
```

### Out of Memory During Training

- Reduce `batch_size` in `configs/training_config.yaml`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use a smaller model (`gemma-2b` instead of `gemma-7b`)

### Dataset Download Failed

```bash
# Manually download
python scripts/download_dataset.py

# Or set HuggingFace token if needed
export HF_TOKEN=your_token_here
```

## Next Steps

1. Review `.specify/memory/constitution.md` for project principles
2. Read `.specify/specs/01-extractive-qa-system.md` for detailed requirements
3. Implement training and evaluation scripts (currently in planning)
4. Run experiments and track in MLflow
5. Register best model in MLflow Model Registry

## Resources

- **Spec-Kit**: https://github.com/github/spec-kit
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **Transformers**: https://huggingface.co/docs/transformers
- **Gemma**: https://ai.google.dev/gemma
- **REDSM5 Dataset**: https://huggingface.co/datasets/irlab-udc/redsm5

## Support

For issues or questions:
1. Check this BUILD.md guide
2. Review the constitution and specifications
3. Check GitHub Issues
4. Review MLflow logs for training issues
