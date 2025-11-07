# Specification: Extractive QA System for Criteria Matching

## Overview

Build an extractive question-answering system that identifies evidence spans in posts supporting diagnostic criteria, using a fine-tuned Gemma-based encoder model.

## Requirements

### 1. Data Pipeline

#### 1.1 Dataset Handling
- **Source**: HuggingFace dataset `irlab-udc/redsm5`
- **Local Storage**: `data/redsm5/` directory
- **Dataset Schema**: Contains `post` and `evidence_sentence` fields
- **Filtering**: Exclude samples with "special case" symptom category (keep only 9 DSM-5 symptoms)
- **Evidence Position Extraction**:
  - Dataset provides evidence sentence text (not positions)
  - Programmatically locate evidence sentence in post using string matching
  - Handle edge cases (sentence may not be exact substring due to whitespace/formatting)
  - Convert evidence sentence positions to token indices after tokenization
- **Preprocessing**:
  - Tokenization with Gemma tokenizer
  - Character-to-token position mapping for evidence spans
  - Truncation strategy: Truncate post if exceeding max_seq_length, preserve evidence sentence
  - Single evidence span per example (one evidence sentence per post-criterion pair)

#### 1.2 Data Loading
- Implement `RedsM5Dataset` class in `src/Project/SubProject/data/dataset.py`
- Support train/validation/test splits
- Efficient batching with DataLoader
- Caching for faster iteration

### 2. Model Architecture

#### 2.1 Base Model
- **Source**: Standard Gemma from HuggingFace
- **Default**: `google/gemma-7b` (better performance, ~7B parameters)
- **Alternative**: `google/gemma-2b` (faster training, ~2B parameters)
- **Model Selection**: Configurable via Makefile commands

#### 2.1.1 Encoder Conversion (Critical Implementation)
Following paper methodology for decoder-to-encoder adaptation:
- **Load pretrained Gemma decoder** from HuggingFace
- **Replace causal attention mask** with bidirectional attention:
  - Remove causal masking in self-attention layers
  - Enable full attention across all positions (not just previous tokens)
  - Modify attention mask from lower-triangular to full matrix
- **Keep pretrained weights** (transfer learning from decoder)
- **Add task-specific heads**:
  - Start position classifier: Linear layer `[hidden_size] → [1]`
  - End position classifier: Linear layer `[hidden_size] → [1]`
  - Both heads produce logits over sequence length

#### 2.1.2 Implementation Approach
```python
# Pseudocode for encoder conversion
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

# Modify attention mechanism
for layer in model.model.layers:
    # Replace causal attention with bidirectional
    layer.self_attn.is_causal = False

# Add QA heads
model.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits
```

#### 2.2 Input Format

**Prompt Optimization** (based on paper methodology):

Following arxiv:2503.02656 approach for encoder tasks:
- Use special tokens to delineate sections
- Optimize for evidence extraction task
- Format: `[INST] {instruction} [/INST] [POST] {post} [/POST] [CRITERION] {criterion} [/CRITERION]`

**Recommended Format**:
```
[INST] Extract the evidence sentence that supports the diagnostic criterion. [/INST] [POST] {post_text} [/POST] [CRITERION] {criterion_text} [/CRITERION]
```

**Example**:
```
[INST] Extract the evidence sentence that supports the diagnostic criterion. [/INST]
[POST] Patient reports persistent feelings of sadness. They have trouble sleeping most nights. They lost interest in activities they once enjoyed. [/POST]
[CRITERION] Depressed mood most of the day [/CRITERION]
```

**Alternative Simpler Format** (if special tokens cause issues):
```
Question: Find evidence for criterion.
Post: {post_text}
Criterion: {criterion_text}
```

**Token Handling**:
- Add custom tokens `[INST]`, `[/INST]`, `[POST]`, `[/POST]`, `[CRITERION]`, `[/CRITERION]` to tokenizer
- Or use existing Gemma special tokens if available
- Evidence span should be extracted from the `[POST]` section only

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

Following SQuAD evaluation methodology exactly:

**Primary Metrics**:
- **Exact Match (EM)**: Percentage of predictions matching gold answer exactly (after normalization)
- **F1 Score**: Token-level overlap F1 between prediction and gold answer

**Normalization** (SQuAD-style):
- Lowercase text
- Remove punctuation
- Remove articles (a, an, the)
- Normalize whitespace

**Multi-level Metrics** (comprehensive evaluation):
1. **Character-level**: Exact substring match in original text
2. **Token-level**: F1 score on tokenized spans (primary metric)
3. **Word-level**: F1 score on word-level spans

**Additional Metrics**:
- **Precision**: Ratio of correct tokens in prediction
- **Recall**: Ratio of correct tokens found
- **Partial Match**: Count predictions with >50% overlap
- **No-Answer Accuracy**: If implementing no-answer prediction
- **Latency**: Inference time per sample (ms)

**Evaluation Code**:
Use official SQuAD evaluation script or HuggingFace `evaluate` library with `squad` metric

#### 4.2 Evaluation Script
- Implement in `src/Project/SubProject/engine/eval_engine.py`
- Support batch evaluation
- Generate detailed error analysis
- Export predictions for manual review

### 5. MLflow Integration

#### 5.1 Tracking Configuration

**Development Setup** (SQLite - simpler, faster iteration):
- **Backend**: SQLite database
  - Database file: `mlflow.db` in project root
  - Connection string: `sqlite:///mlflow.db`
- **Artifact Store**: Local directory `mlruns/`
- **Experiment Name**: `criteria-matching-extractive-qa`

**Production Setup** (PostgreSQL - for scale and team collaboration):
- **Backend**: PostgreSQL database
  - Database: `mlflow_db`
  - Connection string: `postgresql://mlflow_user:password@localhost:5432/mlflow_db`
- **Artifact Store**: Local directory `mlruns/` or S3/cloud storage
- **Experiment Name**: `criteria-matching-extractive-qa-production`

**Switching Backends**:
- Use environment variable `MLFLOW_BACKEND` to switch
- Default to SQLite for development
- Scripts should support both backends

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

## Implementation Decisions (All Resolved)

### Core Architecture
✅ **Dataset Format**: REDSM5 provides post and evidence_sentence (not positions)
✅ **Base Model**: Gemma-7b default, Gemma-2b alternative (both supported via Makefile)
✅ **Encoder Conversion**: True bidirectional attention (set `is_causal=False`)
✅ **Special Cases**: Filter out "special case" symptom category (keep only 9 DSM-5 symptoms)
✅ **Evidence Spans**: Single evidence sentence per example (no multi-span)

### Data Processing
✅ **Position Extraction**: Programmatic string matching to locate evidence in post
✅ **Evidence Matching**: Hybrid approach (exact match → fuzzy fallback at 85% threshold)
✅ **Truncation**: Truncate post if exceeds max_seq_length, preserve evidence
✅ **Max Sequence Length**: 1024 tokens for optimal context-performance balance
✅ **No-Answer Handling**: Skip examples where evidence cannot be located

### Model & Training
✅ **Prompt Format**: Optimized with special tokens following paper methodology
✅ **Attention Modification**: Mask-only approach, keep pretrained position embeddings
✅ **MLflow Backend**: SQLite for development, PostgreSQL for production
✅ **Hyperparameter Tuning**: After baseline training with Optuna

### Evaluation
✅ **Evaluation**: SQuAD-style metrics at character, token, and word levels
✅ **Normalization**: Lowercase, remove punctuation/articles, normalize whitespace

## Resolved Implementation Questions

All open questions have been clarified and decisions documented:

### 1. Evidence Matching Strategy ✅
**Decision**: Hybrid approach (C)
- First attempt: Exact match with whitespace normalization
- Fallback: Fuzzy matching with 85% threshold (difflib)
- Rationale: Balance between precision (exact match) and recall (fuzzy fallback)
- Implementation: Configured in `data_config.yaml`

### 2. Maximum Sequence Length ✅
**Decision**: 1024 tokens
- Provides good balance between context and performance
- Fits comfortably on RTX 4090 (24GB VRAM)
- Allows batch sizes of 4-8 for Gemma-7b
- Can truncate posts while preserving evidence sentence
- Implementation: Set in `data_config.yaml` and `model_config.yaml`

### 3. No-Answer Handling ✅
**Decision**: Skip examples (A)
- Skip training examples where evidence sentence cannot be located
- Log all skipped examples to `data/problematic_samples.json`
- Generate preprocessing report with skip statistics
- Rationale: Maintains training data quality, avoids noisy supervision
- Implementation: `not_found_action: "skip"` in `data_config.yaml`

### 4. Encoder Conversion Approach ✅
**Decision**: Attention mask modification only (A)
- Set `layer.self_attn.is_causal = False` for all layers
- Keep pretrained position embeddings (RoPE)
- No retraining of positional embeddings needed
- Rationale: Paper shows this is sufficient, preserves pretrained knowledge
- Implementation: Documented in `docs/ENCODER_CONVERSION_GUIDE.md`

## Remaining Open Questions

None - all implementation decisions have been resolved and documented.

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
