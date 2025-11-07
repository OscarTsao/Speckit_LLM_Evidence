# Speckit LLM Evidence - Project Constitution

## Mission

Build a fine-tuned extractive question-answering system for criteria matching tasks, adapting decoder-based language models (Gemma Encoder approach) to extract evidence from posts that support diagnostic criteria.

## Core Principles

### 1. Reproducibility First
- All experiments must be tracked in MLflow with PostgreSQL backend
- Best models must be registered in MLflow Model Registry
- Artifacts stored in local `mlruns/` directory for portability
- Random seeds must be set consistently across all runs
- Environment specifications (conda + pip) must be version-locked

### 2. Performance Optimization
- Target hardware: NVIDIA RTX 4090 (single GPU)
- Utilize CUDA 12.1 with cuDNN 8.9
- Implement mixed-precision training (fp16/bf16)
- Optimize batch sizes for 24GB VRAM
- Use gradient accumulation when needed
- Leverage flash-attention and optimized kernels

### 3. Data Integrity
- Source dataset: `irlab-udc/redsm5` from HuggingFace
- Local copy in `data/redsm5/` directory
- No data leakage between train/val/test splits
- Proper preprocessing and tokenization pipelines
- Validate data quality before training

### 4. Model Architecture
- Base approach: Gemma Encoder (decoder-to-encoder adaptation)
- Task: Extractive QA with SQuAD-style start/end token classification
- Input format: `"Retrieve the evidence from the post:{post} that support the diagnosis of the criterion:{criterion}"`
- Output: Start and end position logits for evidence span extraction
- Handle special cases appropriately

### 5. Evaluation Standards
- Primary metrics: Exact Match (EM) and F1 Score
- Secondary metrics: Precision, Recall at various thresholds
- Validation strategy: Hold-out validation set
- Test only on final model selection
- Log all metrics to MLflow

### 6. Code Quality
- Follow PEP 8 style guidelines
- Type hints for all function signatures
- Comprehensive docstrings
- Unit tests for critical components
- Linting: ruff, black formatting
- Pre-commit hooks for consistency

### 7. Logging and Observability
- Structured logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Progress visualization with tqdm
- MLflow autologging for hyperparameters and metrics
- TensorBoard integration for training curves
- Save checkpoints at regular intervals

### 8. Environment Management
- Conda for system-level dependencies (CUDA, PostgreSQL)
- Pip for Python packages
- Lock files for exact version control
- Dev container with GPU support for consistent development
- Separate requirements for dev/prod

### 9. Experimentation Workflow
- Use Optuna for hyperparameter optimization
- Track all hyperparameter sweeps in MLflow
- Document experiment rationale in MLflow run descriptions
- Compare runs systematically before model selection
- Archive failed experiments with learnings

### 10. Build and Deployment
- Automated testing in CI/CD pipeline
- GitHub Actions for spec validation
- Docker containers for deployment
- Model versioning through MLflow registry
- Clear model promotion criteria (staging → production)

## Non-Negotiables

1. **GPU Requirement**: Must run on CUDA-enabled hardware
2. **MLflow Tracking**: Every training run must be logged
3. **Data Location**: Dataset must be in `data/redsm5/`
4. **Format Consistency**: Input must follow specified template
5. **Metric Reporting**: EM and F1 must be reported for all evaluations
6. **Version Control**: All code changes must be committed with clear messages
7. **No Manual Steps**: Build process must be fully automated

## Decision Framework

When making technical decisions, prioritize in this order:
1. Correctness and reproducibility
2. Performance on RTX 4090
3. Code maintainability and clarity
4. Development velocity
5. Cost and resource efficiency

## Validation Checklist

Before considering any feature complete:
- [ ] Unit tests pass
- [ ] Linting passes (ruff, black)
- [ ] MLflow tracking enabled
- [ ] Logs are informative and structured
- [ ] Documentation updated
- [ ] Spec-kit specification updated
- [ ] Works in dev container
- [ ] GPU utilization is optimized
- [ ] Results are reproducible
