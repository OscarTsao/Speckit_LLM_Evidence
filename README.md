# Speckit LLM Evidence

Fine-tuning extractive QA models for criteria matching tasks using Gemma Encoder and the [Spec-Kit](https://github.com/github/spec-kit) methodology.

## Overview

This project adapts decoder-based language models (Gemma) for extractive question-answering to identify evidence in posts that support diagnostic criteria. Built with reproducibility, performance optimization (RTX 4090), and spec-driven development in mind.

### Key Features

- **Extractive QA**: SQuAD-style start/end token classification
- **Dataset**: REDSM5 from HuggingFace (`irlab-udc/redsm5`)
- **Model**: Gemma-based encoder (decoder-to-encoder adaptation)
- **Tracking**: MLflow with PostgreSQL backend
- **Optimization**: CUDA 12.1, mixed precision, flash attention
- **Spec-Kit**: Specification-driven development workflow

## Quick Start

### Prerequisites

- NVIDIA RTX 4090 GPU (or compatible with 24GB+ VRAM)
- CUDA 12.1+ drivers
- Conda/Miniconda
- PostgreSQL 12+

### Setup

```bash
# Clone and navigate to project
cd Speckit_LLM_Evidence

# Run automated setup
bash scripts/setup_environment.sh

# Activate environment
conda activate speckit-llm-evidence

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

For detailed build instructions, see [BUILD.md](BUILD.md).

## Spec-Kit Methodology

This project follows the [Spec-Kit](https://github.com/github/spec-kit) spec-driven development approach:

### Project Structure

```
.specify/
├── memory/
│   └── constitution.md          # Core principles and guidelines
└── specs/
    └── 01-extractive-qa-system.md  # Detailed system specification
```

### Constitution Highlights

- **Reproducibility First**: All experiments tracked, version-locked dependencies
- **Performance Optimization**: Optimized for RTX 4090 single GPU
- **Data Integrity**: Proper train/val/test splits, no leakage
- **Code Quality**: Type hints, tests, linting required

See [.specify/memory/constitution.md](.specify/memory/constitution.md) for full principles.

### Specifications

- **[Extractive QA System](.specify/specs/01-extractive-qa-system.md)**: Complete specification including:
  - Data pipeline and preprocessing
  - Model architecture (Gemma Encoder)
  - Training pipeline with RTX 4090 optimizations
  - MLflow integration and model registry
  - Evaluation metrics and testing

## Project Layout

```
Speckit_LLM_Evidence/
├── .devcontainer/          # CUDA-enabled dev container
├── .github/workflows/      # CI/CD (spec validation, tests)
├── .specify/               # Spec-Kit specifications
├── configs/                # YAML configurations
│   ├── mlflow_config.yaml
│   └── training_config.yaml
├── data/redsm5/            # Dataset (downloaded)
├── scripts/                # Setup and utility scripts
├── src/                    # Source code
│   └── Project/SubProject/
│       ├── data/           # Dataset handling
│       ├── models/         # Model architecture
│       ├── engine/         # Training/evaluation
│       └── utils/          # Utilities, logging, MLflow
├── tests/                  # Unit tests
├── environment.yml         # Conda environment
├── requirements.txt        # Pip dependencies
└── BUILD.md                # Detailed build guide
```

## Configuration

### Training

Edit `configs/training_config.yaml`:

```yaml
model:
  name: "google/gemma-2b"

training:
  batch_size: 8
  learning_rate: 2.0e-5
  num_epochs: 5
  fp16: true
```

### MLflow

PostgreSQL backend for experiment tracking, local artifact store:

```bash
# Setup (automated via scripts/setup_postgresql.sh)
MLFLOW_TRACKING_URI=postgresql://mlflow_user:password@localhost:5432/mlflow_db

# Start MLflow UI
mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root ./mlruns
```

## Development

### Dev Container

VS Code dev container with full CUDA support:

1. Install Docker + NVIDIA Container Toolkit
2. Open in VS Code
3. `F1` → "Dev Containers: Reopen in Container"

### Linting and Testing

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type check
mypy src

# Run tests
pytest tests/ -v
```

### Pre-commit Hooks

```bash
pre-commit install
```

## CI/CD

GitHub Actions automatically validate:

- ✅ Spec-Kit structure and formatting
- ✅ YAML configuration validity
- ✅ Code quality (linting, formatting)
- ✅ Test suite
- ✅ Build environment

## MLflow Tracking

All training runs are tracked with:

- **Backend**: PostgreSQL database
- **Artifacts**: Local `mlruns/` directory
- **Registry**: Model versioning and promotion
- **Metrics**: EM, F1, loss curves
- **Logs**: Training configurations, predictions

Access MLflow UI at http://localhost:5000

## Usage (When Implemented)

```bash
# Train model
python scripts/train.py --config configs/training_config.yaml

# Evaluate
python scripts/evaluate.py --model-path mlruns/1/abc123/artifacts/model

# Inference
python scripts/predict.py --model-path <path> --post "..." --criterion "..."
```

## Acceptance Criteria

- [ ] Model trains without OOM on RTX 4090
- [ ] MLflow tracks all runs with PostgreSQL
- [ ] F1 Score > 60% on test set
- [ ] EM > 45% on test set
- [ ] Best model registered in MLflow
- [ ] GPU utilization > 80% during training
- [ ] Results reproducible with same seed

## Resources

- **Spec-Kit**: https://github.com/github/spec-kit
- **Paper**: [Adapting Decoder-Based LMs](https://arxiv.org/abs/2503.02656)
- **Dataset**: [REDSM5](https://huggingface.co/datasets/irlab-udc/redsm5)
- **MLflow**: https://mlflow.org/docs/latest/
- **Gemma**: https://ai.google.dev/gemma

## Contributing

1. Review `.specify/memory/constitution.md` for principles
2. Read relevant specs in `.specify/specs/`
3. Implement according to specifications
4. Ensure tests pass and linting succeeds
5. Update specs if needed
6. Submit PR (CI/CD will validate)

## License

MIT License (or your chosen license)

---

**Built with [Spec-Kit](https://github.com/github/spec-kit)** - Specification-driven development for AI/ML projects.

