# Specification: Extractive QA System for Criteria Matching

## Overview

Build an extractive question-answering system that identifies evidence spans in posts supporting diagnostic criteria, using a fine-tuned Gemma-based encoder model.

## Requirements

### 1. Data Pipeline

#### 1.1 Dataset Handling
- **Source**: HuggingFace dataset `irlab-udc/redsm5`
- **Local Storage**: `data/redsm5/` directory
- **Format**: JSON/CSV with fields: `post`, `criterion`, `evidence_start`, `evidence_end`
- **Preprocessing**:
  - Tokenization with Gemma tokenizer
  - Character-to-token position mapping
  - Truncation strategy for long sequences
  - Special case handling (when no evidence exists)

#### 1.2 Data Loading
- Implement `RedsM5Dataset` class in `src/Project/SubProject/data/dataset.py`
- Support train/validation/test splits
- Efficient batching with DataLoader
- Caching for faster iteration

### 2. Model Architecture

#### 2.1 Base Model
- **Architecture**: Gemma-based encoder (decoder-to-encoder adaptation)
- **Pretrained Checkpoint**: `google/gemma-2b` or `google/gemma-7b`
- **Modification**: Replace causal attention with bidirectional attention
- **Output Heads**:
  - Start position classifier (linear layer)
  - End position classifier (linear layer)

#### 2.2 Input Format
```
Template: "Retrieve the evidence from the post:{post} that support the diagnosis of the criterion:{criterion}"
Example: "Retrieve the evidence from the post:Patient shows severe anxiety and recurring nightmares that support the diagnosis of the criterion:PTSD symptoms"
```

#### 2.3 Output Format
- **Start Logits**: [batch_size, seq_length] tensor of start position scores
- **End Logits**: [batch_size, seq_length] tensor of end position scores
- **Predicted Span**: (start_idx, end_idx) with highest joint probability

### 3. Training Pipeline

#### 3.1 Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 2e-5 (initial), with linear warmup and decay
- **Batch Size**: Maximize for 4090 (24GB VRAM), likely 8-16 with gradient accumulation
- **Mixed Precision**: Use `torch.cuda.amp` for fp16 training
- **Gradient Clipping**: Max norm 1.0
- **Epochs**: 3-5 with early stopping

#### 3.2 Loss Function
- Cross-entropy loss for start positions
- Cross-entropy loss for end positions
- Combined loss: `loss = (start_loss + end_loss) / 2`
- Handle impossible spans (end before start)

#### 3.3 Optimization for RTX 4090
- Enable TF32 for matrix multiplications
- Use flash-attention if supported
- Gradient checkpointing for larger models
- Pin memory for DataLoader
- Set `torch.backends.cudnn.benchmark = True`

#### 3.4 Training Loop
- Implement in `src/Project/SubProject/engine/train_engine.py`
- Progress tracking with tqdm
- Validation every N steps
- Checkpoint saving (best model, last model)
- MLflow logging integration

### 4. Evaluation Pipeline

#### 4.1 Metrics
- **Exact Match (EM)**: Percentage of predictions with exact character match
- **F1 Score**: Token-level overlap F1
- **Precision/Recall**: At character and token level
- **Latency**: Inference time per sample

#### 4.2 Evaluation Script
- Implement in `src/Project/SubProject/engine/eval_engine.py`
- Support batch evaluation
- Generate detailed error analysis
- Export predictions for manual review

### 5. MLflow Integration

#### 5.1 Tracking Configuration
- **Backend**: PostgreSQL database
  - Database: `mlflow_db`
  - Connection string: `postgresql://user:password@localhost:5432/mlflow_db`
- **Artifact Store**: Local directory `mlruns/`
- **Experiment Name**: `criteria-matching-extractive-qa`

#### 5.2 Logged Artifacts
- **Parameters**: All hyperparameters (lr, batch_size, model_name, etc.)
- **Metrics**: EM, F1, loss curves
- **Artifacts**:
  - Best model checkpoint
  - Training configuration (YAML)
  - Evaluation predictions (JSON)
  - Training logs
- **Tags**: `stage`, `dataset_version`, `gpu_type`

#### 5.3 Model Registry
- Register best model with name: `gemma-criteria-matching`
- Versioning strategy: semantic versioning
- Model metadata: performance metrics, training date, dataset info

### 6. Logging and Monitoring

#### 6.1 Structured Logging
- Use Python `logging` module with configured handlers
- Log levels:
  - DEBUG: Detailed training loop info
  - INFO: Epoch results, validation scores
  - WARNING: Gradient issues, convergence concerns
  - ERROR: Training failures, data issues
- Output to both console and file: `logs/training_{timestamp}.log`

#### 6.2 Visualization
- TensorBoard logs for loss curves
- MLflow UI for experiment comparison
- Progress bars for training/evaluation

### 7. Configuration Management

#### 7.1 Config Files
- YAML configs in `configs/` directory
- Separate configs for:
  - Model architecture (`model_config.yaml`)
  - Training hyperparameters (`training_config.yaml`)
  - Data processing (`data_config.yaml`)
  - MLflow settings (`mlflow_config.yaml`)

#### 7.2 Environment Variables
- Store sensitive info in `.env`:
  - MLflow PostgreSQL credentials
  - HuggingFace API token (if needed)
  - Wandb API key (optional)

### 8. Testing

#### 8.1 Unit Tests
- Test dataset loading and preprocessing
- Test model forward pass
- Test metric calculations
- Test MLflow logging

#### 8.2 Integration Tests
- End-to-end training on small dataset
- Evaluation pipeline
- Model loading and inference

### 9. Scripts

#### 9.1 Training Script
```bash
python scripts/train.py --config configs/training_config.yaml
```

#### 9.2 Evaluation Script
```bash
python scripts/evaluate.py --model-path mlruns/1/abc123/artifacts/model --data data/redsm5/test.json
```

#### 9.3 Inference Script
```bash
python scripts/predict.py --model-path mlruns/1/abc123/artifacts/model --post "..." --criterion "..."
```

#### 9.4 Setup Script
```bash
bash scripts/setup_environment.sh
```

## Acceptance Criteria

### Must Have
- [x] Dataset downloads and preprocesses correctly
- [ ] Model trains without OOM errors on RTX 4090
- [ ] MLflow tracks all runs with PostgreSQL backend
- [ ] Achieves baseline performance (EM > 40%, F1 > 50%)
- [ ] Best model registered in MLflow
- [ ] All tests pass
- [ ] GPU utilization > 80% during training
- [ ] Reproducible results with same seed

### Should Have
- [ ] Hyperparameter optimization with Optuna
- [ ] TensorBoard integration
- [ ] Prediction export for error analysis
- [ ] Training time < 4 hours on full dataset
- [ ] Inference latency < 100ms per sample

### Nice to Have
- [ ] Multi-GPU support
- [ ] Quantization for faster inference
- [ ] API endpoint for serving predictions
- [ ] Gradio demo interface
- [ ] Comprehensive documentation

## Open Questions

1. What is the exact format of the redsm5 dataset? Need to inspect after download.
2. Should we use Gemma-2b or Gemma-7b? (7b likely better but slower)
3. What constitutes a "special case" where no evidence exists?
4. Do we need to handle multi-span evidence extraction?
5. What's the maximum sequence length we should support?

## Dependencies

- Transformers library for Gemma models
- Datasets library for HuggingFace integration
- PyTorch 2.2+ with CUDA 12.1
- MLflow with PostgreSQL driver
- Accelerate for training optimization

## Timeline Estimate

- Dataset setup and exploration: 1 day
- Model implementation: 2 days
- Training pipeline: 2 days
- Evaluation and metrics: 1 day
- MLflow integration: 1 day
- Testing and documentation: 1 day
- **Total**: ~8 days of development

## Success Metrics

- **Primary**: F1 Score > 60% on test set
- **Secondary**: EM > 45%, Training time < 4 hours
- **Operational**: All runs logged, model registered, tests passing
