# Speckit LLM Evidence - Makefile
# Provides convenient commands for training with different model configurations

.PHONY: help setup clean train-7b train-2b eval-7b eval-2b mlflow-ui download-data lint test

# Default target
help:
	@echo "Speckit LLM Evidence - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup              - Set up conda environment and dependencies"
	@echo "  make download-data      - Download REDSM5 dataset"
	@echo ""
	@echo "Training (Gemma-7b - default):"
	@echo "  make train-7b           - Train with Gemma-7b model"
	@echo "  make train-7b-fast      - Train with Gemma-7b (smaller batch, faster)"
	@echo "  make train-7b-optuna    - Hyperparameter optimization with Optuna"
	@echo ""
	@echo "Training (Gemma-2b - faster alternative):"
	@echo "  make train-2b           - Train with Gemma-2b model"
	@echo "  make train-2b-fast      - Train with Gemma-2b (larger batch, faster)"
	@echo "  make train-2b-optuna    - Hyperparameter optimization with Optuna"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-7b RUN_ID=... - Evaluate Gemma-7b model"
	@echo "  make eval-2b RUN_ID=... - Evaluate Gemma-2b model"
	@echo "  make eval-best          - Evaluate best registered model"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow-ui          - Start MLflow UI (port 5000)"
	@echo "  make mlflow-sqlite      - Switch to SQLite backend"
	@echo "  make mlflow-postgres    - Switch to PostgreSQL backend"
	@echo ""
	@echo "Development:"
	@echo "  make lint               - Run linting (ruff, black)"
	@echo "  make format             - Format code (black)"
	@echo "  make test               - Run tests"
	@echo "  make clean              - Clean build artifacts"
	@echo ""

# ==================== Setup ====================

setup:
	@echo "Setting up environment..."
	bash scripts/setup_environment.sh

download-data:
	@echo "Downloading REDSM5 dataset..."
	python scripts/download_dataset.py

# ==================== Training (Gemma-7b) ====================

train-7b:
	@echo "Training with Gemma-7b (default configuration)..."
	python scripts/train.py \
		--config configs/training_config.yaml \
		--model-config configs/model_config.yaml \
		--data-config configs/data_config.yaml \
		--model-name google/gemma-7b \
		--batch-size 4 \
		--gradient-accumulation-steps 4 \
		--max-seq-length 1024

train-7b-fast:
	@echo "Training with Gemma-7b (fast mode - smaller batch)..."
	python scripts/train.py \
		--config configs/training_config.yaml \
		--model-config configs/model_config.yaml \
		--data-config configs/data_config.yaml \
		--model-name google/gemma-7b \
		--batch-size 2 \
		--gradient-accumulation-steps 8 \
		--max-seq-length 512

train-7b-optuna:
	@echo "Hyperparameter optimization with Optuna (Gemma-7b)..."
	python scripts/optimize_hyperparameters.py \
		--model-name google/gemma-7b \
		--n-trials 50 \
		--study-name gemma-7b-optuna

# ==================== Training (Gemma-2b) ====================

train-2b:
	@echo "Training with Gemma-2b (default configuration)..."
	python scripts/train.py \
		--config configs/training_config.yaml \
		--model-config configs/model_config.yaml \
		--data-config configs/data_config.yaml \
		--model-name google/gemma-2b \
		--batch-size 8 \
		--gradient-accumulation-steps 2 \
		--max-seq-length 1024

train-2b-fast:
	@echo "Training with Gemma-2b (fast mode - larger batch)..."
	python scripts/train.py \
		--config configs/training_config.yaml \
		--model-config configs/model_config.yaml \
		--data-config configs/data_config.yaml \
		--model-name google/gemma-2b \
		--batch-size 16 \
		--gradient-accumulation-steps 1 \
		--max-seq-length 512

train-2b-optuna:
	@echo "Hyperparameter optimization with Optuna (Gemma-2b)..."
	python scripts/optimize_hyperparameters.py \
		--model-name google/gemma-2b \
		--n-trials 50 \
		--study-name gemma-2b-optuna

# ==================== Evaluation ====================

eval-7b:
	@echo "Evaluating Gemma-7b model (RUN_ID=$(RUN_ID))..."
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Error: RUN_ID not specified. Usage: make eval-7b RUN_ID=<mlflow_run_id>"; \
		exit 1; \
	fi
	python scripts/evaluate.py \
		--model-path mlruns/0/$(RUN_ID)/artifacts/model \
		--data-config configs/data_config.yaml \
		--split test

eval-2b:
	@echo "Evaluating Gemma-2b model (RUN_ID=$(RUN_ID))..."
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Error: RUN_ID not specified. Usage: make eval-2b RUN_ID=<mlflow_run_id>"; \
		exit 1; \
	fi
	python scripts/evaluate.py \
		--model-path mlruns/0/$(RUN_ID)/artifacts/model \
		--data-config configs/data_config.yaml \
		--split test

eval-best:
	@echo "Evaluating best registered model..."
	python scripts/evaluate.py \
		--model-name gemma-criteria-matching \
		--model-version latest \
		--data-config configs/data_config.yaml \
		--split test

# ==================== MLflow ====================

mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000..."
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

mlflow-sqlite:
	@echo "Switching to SQLite backend..."
	@echo "MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db" > .env.mlflow
	@echo "Switched to SQLite backend (mlflow.db)"

mlflow-postgres:
	@echo "Switching to PostgreSQL backend..."
	@bash scripts/setup_postgresql.sh
	@echo "Switched to PostgreSQL backend"

# ==================== Development ====================

lint:
	@echo "Running linters..."
	ruff check src tests scripts
	black --check src tests scripts
	mypy src --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src tests scripts
	ruff check --fix src tests scripts

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

test-fast:
	@echo "Running fast tests (no coverage)..."
	pytest tests/ -v -x

# ==================== Inference ====================

predict:
	@echo "Running inference (interactive mode)..."
	python scripts/predict.py \
		--model-name gemma-criteria-matching \
		--interactive

# ==================== Cleanup ====================

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "Cleanup complete!"

clean-data:
	@echo "WARNING: This will delete downloaded datasets!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/redsm5/*.json data/redsm5/arrow; \
		echo "Dataset cleaned!"; \
	fi

clean-mlflow:
	@echo "WARNING: This will delete all MLflow runs!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf mlruns/ mlflow.db; \
		echo "MLflow data cleaned!"; \
	fi

# ==================== Docker ====================

docker-build:
	@echo "Building dev container..."
	docker build -f .devcontainer/Dockerfile -t speckit-llm-evidence:latest .

docker-run:
	@echo "Running dev container..."
	docker run --gpus all -it -v $(PWD):/workspace speckit-llm-evidence:latest

# ==================== Quick Commands ====================

# Quick training with sensible defaults
quick-train: train-2b-fast

# Quick evaluation
quick-eval: eval-best

# Full pipeline (download data + train + eval)
full-pipeline: download-data train-7b eval-best

# ==================== Environment Info ====================

info:
	@echo "Environment Information:"
	@echo "========================"
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
	@python -c "import torch; print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
	@python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
	@echo ""
	@echo "MLflow Backend: $(shell cat .env.mlflow 2>/dev/null || echo 'sqlite:///mlflow.db')"
	@echo "Project Path: $(PWD)"
