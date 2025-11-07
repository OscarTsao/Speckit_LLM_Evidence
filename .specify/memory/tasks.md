# Implementation Tasks - Extractive QA System

**Project**: Speckit LLM Evidence
**Date**: 2025-11-07
**Status**: Ready to Execute

This document breaks down the implementation plan into specific, actionable tasks with clear acceptance criteria.

---

## Phase 1: Data Pipeline (Priority: Critical)

### Task 1.1: Create Dataset Class Structure
**File**: `src/Project/SubProject/data/dataset.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Create `RedsM5Dataset` class inheriting from `torch.utils.data.Dataset`
- [ ] Implement `__init__` with config loading
- [ ] Implement `__len__` method
- [ ] Implement `__getitem__` method stub
- [ ] Add proper type hints and docstrings

**Acceptance Criteria**:
- [ ] Class instantiates without errors
- [ ] Returns correct length
- [ ] Has proper documentation

**Dependencies**: None

---

### Task 1.2: Implement Dataset Loading
**File**: `src/Project/SubProject/data/dataset.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Implement `_load_data` method to load from HuggingFace
- [ ] Add local file loading support (JSON/Arrow format)
- [ ] Handle train/validation/test splits
- [ ] Cache dataset locally in `data/redsm5/`
- [ ] Log dataset statistics (sample counts, splits)

**Acceptance Criteria**:
- [ ] Successfully loads `irlab-udc/redsm5` from HuggingFace
- [ ] Falls back to local cache if available
- [ ] All splits load correctly
- [ ] Logs informative statistics

**Dependencies**: Task 1.1

**Code Snippet**:
```python
def _load_data(self, data_path: str, split: str):
    from datasets import load_dataset

    # Try local first
    local_path = Path(data_path) / "arrow"
    if local_path.exists():
        dataset = load_from_disk(str(local_path))
    else:
        dataset = load_dataset("irlab-udc/redsm5", split=split)

    return dataset
```

---

### Task 1.3: Implement Special Case Filtering
**File**: `src/Project/SubProject/data/dataset.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Implement `_filter_special_cases` method
- [ ] Identify "special case" symptom category field in dataset
- [ ] Filter out all special case samples
- [ ] Log filtering statistics (before/after counts)
- [ ] Add validation that only 9 DSM-5 symptoms remain

**Acceptance Criteria**:
- [ ] Special cases successfully filtered
- [ ] Filtering statistics logged
- [ ] No special case samples in final dataset
- [ ] Unit test verifies filtering

**Dependencies**: Task 1.2

**Code Snippet**:
```python
def _filter_special_cases(self, dataset):
    original_len = len(dataset)

    # Filter out special case category
    dataset = dataset.filter(
        lambda x: x['symptom_category'] != 'special case'
    )

    filtered_len = len(dataset)
    logger.info(f"Filtered {original_len - filtered_len} special case samples")

    return dataset
```

---

### Task 1.4: Implement Evidence Position Extraction
**File**: `src/Project/SubProject/data/dataset.py`
**Estimated Time**: 4 hours

**Subtasks**:
- [ ] Implement `_find_evidence_position` with exact match
- [ ] Add whitespace normalization
- [ ] Implement fuzzy matching fallback (difflib)
- [ ] Set 85% threshold for fuzzy match
- [ ] Convert character positions to token positions
- [ ] Handle cases where evidence not found (return None)
- [ ] Log all failed matches to `data/problematic_samples.json`

**Acceptance Criteria**:
- [ ] Exact match works for well-formed examples
- [ ] Fuzzy match catches slight variations
- [ ] Token positions correctly mapped
- [ ] Failed matches logged properly
- [ ] Unit tests for both exact and fuzzy matching

**Dependencies**: Task 1.2

**Code Snippet**:
```python
def _find_evidence_position(
    self,
    post: str,
    evidence_sentence: str,
    tokenizer
) -> tuple[int, int] | tuple[None, None]:
    from difflib import SequenceMatcher

    # Normalize whitespace
    post_clean = ' '.join(post.split())
    evidence_clean = ' '.join(evidence_sentence.split())

    # Try exact match
    char_start = post_clean.find(evidence_clean)

    if char_start == -1:
        # Fuzzy match fallback
        matcher = SequenceMatcher(None, post_clean, evidence_clean)
        match = matcher.find_longest_match(0, len(post_clean), 0, len(evidence_clean))

        similarity = match.size / len(evidence_clean) if len(evidence_clean) > 0 else 0

        if similarity >= 0.85:
            char_start = match.a
            char_end = match.a + match.size
        else:
            # Log failed match
            self._log_failed_match(post, evidence_sentence, similarity)
            return None, None
    else:
        char_end = char_start + len(evidence_clean)

    # Convert to token positions
    encoding = tokenizer(
        post_clean,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    start_token = self._char_to_token(encoding, char_start)
    end_token = self._char_to_token(encoding, char_end)

    return start_token, end_token
```

---

### Task 1.5: Implement Input Formatting
**File**: `src/Project/SubProject/data/dataset.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Implement `_prepare_input` method
- [ ] Format with special tokens: `[INST]`, `[POST]`, `[CRITERION]`
- [ ] Handle tokenization with Gemma tokenizer
- [ ] Implement truncation preserving evidence
- [ ] Set max_seq_length to 1024
- [ ] Return input_ids, attention_mask, start_positions, end_positions

**Acceptance Criteria**:
- [ ] Input formatted correctly with special tokens
- [ ] Truncation preserves evidence sentence
- [ ] Returns all required tensors
- [ ] Position labels correctly aligned with tokens

**Dependencies**: Task 1.4

**Code Snippet**:
```python
def _prepare_input(
    self,
    post: str,
    criterion: str,
    evidence_start: int,
    evidence_end: int,
    tokenizer
) -> dict:
    # Format prompt
    prompt = (
        f"[INST] Extract the evidence sentence that supports the diagnostic criterion. [/INST]"
        f"[POST] {post} [/POST]"
        f"[CRITERION] {criterion} [/CRITERION]"
    )

    # Tokenize
    encoding = tokenizer(
        prompt,
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True
    )

    # Adjust positions for special tokens and truncation
    # ... (handle position mapping)

    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'start_positions': torch.tensor(start_pos),
        'end_positions': torch.tensor(end_pos)
    }
```

---

### Task 1.6: Write Unit Tests for Dataset
**File**: `tests/data/test_dataset.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Test dataset loading from HuggingFace
- [ ] Test special case filtering
- [ ] Test evidence position extraction (exact match)
- [ ] Test evidence position extraction (fuzzy match)
- [ ] Test input formatting
- [ ] Test truncation behavior
- [ ] Test edge cases (empty post, missing evidence)

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Test coverage > 80% for dataset.py
- [ ] Edge cases handled

**Dependencies**: Tasks 1.1-1.5

---

## Phase 2: Model Architecture (Priority: Critical)

### Task 2.1: Create Model Wrapper Class
**File**: `src/Project/SubProject/models/model.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Create `GemmaForQuestionAnswering` class
- [ ] Implement `__init__` with model loading
- [ ] Load pretrained Gemma (7b or 2b based on config)
- [ ] Add type hints and docstrings
- [ ] Handle device placement (auto device map)

**Acceptance Criteria**:
- [ ] Model loads without errors
- [ ] Supports both Gemma-7b and Gemma-2b
- [ ] Proper documentation

**Dependencies**: None

---

### Task 2.2: Implement Encoder Conversion
**File**: `src/Project/SubProject/models/model.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Implement `_convert_to_encoder` method
- [ ] Iterate through all model layers
- [ ] Set `layer.self_attn.is_causal = False`
- [ ] Verify attention mask is full (not causal)
- [ ] Add logging to confirm conversion
- [ ] Test bidirectional attention with dummy inputs

**Acceptance Criteria**:
- [ ] All layers converted to bidirectional
- [ ] Attention mechanism allows full attention
- [ ] Conversion verified with test
- [ ] No causal masking remains

**Dependencies**: Task 2.1

**Code Snippet**:
```python
def _convert_to_encoder(self):
    """Convert Gemma decoder to encoder by disabling causal attention."""
    logger.info("Converting Gemma decoder to encoder...")

    converted_layers = 0
    for layer in self.gemma.model.layers:
        if hasattr(layer.self_attn, 'is_causal'):
            layer.self_attn.is_causal = False
            converted_layers += 1

    logger.info(f"Converted {converted_layers} layers to bidirectional attention")
```

---

### Task 2.3: Add QA Classification Heads
**File**: `src/Project/SubProject/models/model.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Implement `_add_qa_heads` method
- [ ] Create linear layer for start/end logits: `nn.Linear(hidden_size, 2)`
- [ ] Initialize with Xavier uniform
- [ ] Zero initialize bias

**Acceptance Criteria**:
- [ ] QA heads added successfully
- [ ] Proper initialization
- [ ] Output shape matches expected (batch, seq_len)

**Dependencies**: Task 2.1

**Code Snippet**:
```python
def _add_qa_heads(self):
    """Add QA classification heads for start/end positions."""
    hidden_size = self.gemma.config.hidden_size
    self.qa_outputs = nn.Linear(hidden_size, 2)

    # Initialize
    nn.init.xavier_uniform_(self.qa_outputs.weight)
    nn.init.zeros_(self.qa_outputs.bias)

    logger.info(f"Added QA heads with hidden_size={hidden_size}")
```

---

### Task 2.4: Implement Forward Pass
**File**: `src/Project/SubProject/models/model.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Implement `forward` method
- [ ] Pass inputs through Gemma encoder
- [ ] Extract last hidden states
- [ ] Apply QA heads to get start/end logits
- [ ] Compute loss if labels provided (cross-entropy)
- [ ] Return dict with loss, start_logits, end_logits

**Acceptance Criteria**:
- [ ] Forward pass completes without errors
- [ ] Loss calculated correctly
- [ ] Output shapes correct
- [ ] Gradient flow works

**Dependencies**: Tasks 2.2, 2.3

**Code Snippet**:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    start_positions: torch.Tensor = None,
    end_positions: torch.Tensor = None,
    **kwargs
):
    # Get encoder outputs
    outputs = self.gemma.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs
    )

    sequence_output = outputs.last_hidden_state

    # Get start and end logits
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    # Compute loss if labels provided
    total_loss = None
    if start_positions is not None and end_positions is not None:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

    return {
        'loss': total_loss,
        'start_logits': start_logits,
        'end_logits': end_logits,
        'hidden_states': sequence_output
    }
```

---

### Task 2.5: Add Special Tokens to Tokenizer
**File**: `src/Project/SubProject/models/model.py`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Implement `_add_special_tokens` method
- [ ] Add custom tokens: `[INST]`, `[/INST]`, `[POST]`, `[/POST]`, `[CRITERION]`, `[/CRITERION]`
- [ ] Resize model embeddings
- [ ] Log number of tokens added

**Acceptance Criteria**:
- [ ] Special tokens added successfully
- [ ] Model embeddings resized
- [ ] Tokens accessible in tokenizer

**Dependencies**: Task 2.1

---

### Task 2.6: Write Unit Tests for Model
**File**: `tests/models/test_model.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Test model instantiation
- [ ] Test encoder conversion
- [ ] Test QA heads added
- [ ] Test forward pass
- [ ] Test loss calculation
- [ ] Test with dummy data
- [ ] Test both Gemma-7b and Gemma-2b

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Coverage > 80%
- [ ] Both model sizes tested

**Dependencies**: Tasks 2.1-2.5

---

## Phase 3: Training Engine (Priority: Critical)

### Task 3.1: Create Trainer Class Structure
**File**: `src/Project/SubProject/engine/train_engine.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Create `Trainer` class
- [ ] Implement `__init__` with model, datasets, config
- [ ] Set up optimizer (AdamW)
- [ ] Set up learning rate scheduler (linear with warmup)
- [ ] Configure mixed precision (fp16)
- [ ] Enable TF32 and cudnn optimizations

**Acceptance Criteria**:
- [ ] Trainer instantiates successfully
- [ ] Optimizer and scheduler configured
- [ ] Mixed precision enabled

**Dependencies**: Phase 1, Phase 2

---

### Task 3.2: Implement Training Loop
**File**: `src/Project/SubProject/engine/train_engine.py`
**Estimated Time**: 4 hours

**Subtasks**:
- [ ] Implement `train` method
- [ ] Implement `_train_epoch` method
- [ ] Add gradient accumulation (4 steps)
- [ ] Add gradient clipping (max norm 1.0)
- [ ] Add progress bar with tqdm
- [ ] Validate every 500 steps
- [ ] Save checkpoints every 500 steps

**Acceptance Criteria**:
- [ ] Training loop runs
- [ ] Loss decreases over steps
- [ ] Validation runs at intervals
- [ ] Checkpoints saved

**Dependencies**: Task 3.1

**Code Snippet**:
```python
def _train_epoch(self, epoch: int):
    self.model.train()
    total_loss = 0

    progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs['loss']

        # Backward pass
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

        # Validate
        if (step + 1) % self.eval_steps == 0:
            val_metrics = self.validate()
            self._log_metrics(val_metrics, step)

    return total_loss / len(self.train_loader)
```

---

### Task 3.3: Implement Validation Loop
**File**: `src/Project/SubProject/engine/train_engine.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Implement `validate` method
- [ ] Run model in eval mode
- [ ] Compute validation loss
- [ ] Call evaluation engine for metrics
- [ ] Return metrics dict

**Acceptance Criteria**:
- [ ] Validation runs without errors
- [ ] Returns EM and F1 metrics
- [ ] Model returned to train mode after

**Dependencies**: Task 3.2, Phase 4 (evaluation)

---

### Task 3.4: Integrate MLflow Tracking
**File**: `src/Project/SubProject/engine/train_engine.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Import mlflow
- [ ] Set tracking URI from config (SQLite default)
- [ ] Start MLflow run in `train` method
- [ ] Log hyperparameters
- [ ] Log metrics at each step
- [ ] Log model checkpoints as artifacts
- [ ] End run after training

**Acceptance Criteria**:
- [ ] MLflow run created
- [ ] All params logged
- [ ] Metrics logged correctly
- [ ] Artifacts saved

**Dependencies**: Task 3.2

**Code Snippet**:
```python
def train(self):
    import mlflow

    # Configure MLflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('criteria-matching-extractive-qa')

    with mlflow.start_run(run_name=f"gemma-{self.model_name}-{timestamp}"):
        # Log config
        mlflow.log_params(self.config)
        mlflow.log_param('model_name', self.model_name)
        mlflow.log_param('gpu', torch.cuda.get_device_name(0))

        best_f1 = 0

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.validate()

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_f1': val_metrics['f1'],
                'val_em': val_metrics['exact_match']
            }, step=epoch)

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                mlflow.pytorch.log_model(self.model, 'best_model')
```

---

### Task 3.5: Implement Checkpoint Management
**File**: `src/Project/SubProject/engine/train_engine.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Implement `_save_checkpoint` method
- [ ] Save model state_dict
- [ ] Save optimizer state
- [ ] Save scheduler state
- [ ] Save to `outputs/checkpoints/`
- [ ] Keep only last 3 checkpoints
- [ ] Implement checkpoint loading

**Acceptance Criteria**:
- [ ] Checkpoints saved correctly
- [ ] Old checkpoints deleted
- [ ] Can resume from checkpoint

**Dependencies**: Task 3.2

---

### Task 3.6: Write Unit Tests for Training
**File**: `tests/engine/test_train.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Test trainer instantiation
- [ ] Test single training step
- [ ] Test validation step
- [ ] Test checkpoint saving/loading
- [ ] Test MLflow logging (with mock)

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Can run mini training on small data

**Dependencies**: Tasks 3.1-3.5

---

## Phase 4: Evaluation Engine (Priority: High)

### Task 4.1: Implement Text Normalization
**File**: `src/Project/SubProject/engine/eval_engine.py`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Implement `_normalize_text` method (SQuAD-style)
- [ ] Lowercase text
- [ ] Remove articles (a, an, the)
- [ ] Remove punctuation
- [ ] Normalize whitespace

**Acceptance Criteria**:
- [ ] Normalization matches SQuAD behavior
- [ ] Unit tests verify normalization

**Dependencies**: None

---

### Task 4.2: Implement Exact Match Metric
**File**: `src/Project/SubProject/engine/eval_engine.py`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Implement `_compute_em` method
- [ ] Normalize both prediction and reference
- [ ] Compare strings
- [ ] Return percentage

**Acceptance Criteria**:
- [ ] EM calculation correct
- [ ] Unit tests pass

**Dependencies**: Task 4.1

---

### Task 4.3: Implement F1 Score Metric
**File**: `src/Project/SubProject/engine/eval_engine.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Implement `_compute_f1` method (token-level)
- [ ] Tokenize predictions and references
- [ ] Compute precision and recall
- [ ] Calculate F1 score
- [ ] Handle edge cases (empty predictions)

**Acceptance Criteria**:
- [ ] F1 calculation matches SQuAD
- [ ] Unit tests verify correctness

**Dependencies**: Task 4.1

---

### Task 4.4: Implement Multi-level Metrics
**File**: `src/Project/SubProject/engine/eval_engine.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Implement character-level EM
- [ ] Implement token-level F1
- [ ] Implement word-level F1
- [ ] Aggregate all metrics

**Acceptance Criteria**:
- [ ] All three levels computed
- [ ] Results make sense

**Dependencies**: Tasks 4.2, 4.3

---

### Task 4.5: Create Evaluator Class
**File**: `src/Project/SubProject/engine/eval_engine.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Create `Evaluator` class
- [ ] Implement `evaluate` method
- [ ] Run model on test data
- [ ] Extract predictions
- [ ] Compute all metrics
- [ ] Generate predictions file
- [ ] Return metrics dict

**Acceptance Criteria**:
- [ ] Evaluation runs on full dataset
- [ ] All metrics computed
- [ ] Predictions exported to JSON

**Dependencies**: Tasks 4.1-4.4

---

### Task 4.6: Write Unit Tests for Evaluation
**File**: `tests/engine/test_eval.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Test normalization
- [ ] Test EM calculation
- [ ] Test F1 calculation
- [ ] Test evaluator on dummy data

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Known examples produce expected metrics

**Dependencies**: Tasks 4.1-4.5

---

## Phase 5: Scripts and CLI (Priority: Medium)

### Task 5.1: Create Training Script
**File**: `scripts/train.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Set up argparse
- [ ] Load config files
- [ ] Initialize dataset
- [ ] Initialize model
- [ ] Initialize trainer
- [ ] Run training
- [ ] Handle errors gracefully

**Acceptance Criteria**:
- [ ] Script runs end-to-end
- [ ] Arguments parsed correctly
- [ ] Errors handled

**Dependencies**: Phases 1-3

---

### Task 5.2: Create Evaluation Script
**File**: `scripts/evaluate.py`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Set up argparse
- [ ] Load model from MLflow or checkpoint
- [ ] Load test data
- [ ] Run evaluation
- [ ] Print metrics
- [ ] Save predictions

**Acceptance Criteria**:
- [ ] Script runs successfully
- [ ] Metrics printed clearly
- [ ] Predictions saved

**Dependencies**: Phases 1, 2, 4

---

### Task 5.3: Create Prediction Script
**File**: `scripts/predict.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Set up argparse
- [ ] Support interactive mode
- [ ] Load model
- [ ] Format input
- [ ] Run inference
- [ ] Extract predicted span
- [ ] Display result

**Acceptance Criteria**:
- [ ] Can predict on single example
- [ ] Interactive mode works
- [ ] Output is clear

**Dependencies**: Phases 1, 2

---

### Task 5.4: Update Makefile Commands
**File**: `Makefile`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Verify train-7b command works
- [ ] Verify train-2b command works
- [ ] Verify eval commands work
- [ ] Add any missing commands
- [ ] Test all commands

**Acceptance Criteria**:
- [ ] All Makefile commands work
- [ ] Help text is accurate

**Dependencies**: Tasks 5.1-5.3

---

### Task 5.5: Write Integration Tests
**File**: `tests/integration/test_pipeline.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Test full pipeline: data → model → training → evaluation
- [ ] Use small subset of data
- [ ] Run for 1 epoch
- [ ] Verify loss decreases
- [ ] Verify metrics computed
- [ ] Verify MLflow logging

**Acceptance Criteria**:
- [ ] Integration test passes
- [ ] Pipeline runs end-to-end
- [ ] Takes < 5 minutes

**Dependencies**: All phases

---

## Task Dependencies Graph

```
Phase 1 (Data)
├── 1.1 Dataset Structure
├── 1.2 Dataset Loading ← 1.1
├── 1.3 Filtering ← 1.2
├── 1.4 Evidence Extraction ← 1.2
├── 1.5 Input Formatting ← 1.4
└── 1.6 Tests ← 1.1-1.5

Phase 2 (Model)
├── 2.1 Model Class
├── 2.2 Encoder Conversion ← 2.1
├── 2.3 QA Heads ← 2.1
├── 2.4 Forward Pass ← 2.2, 2.3
├── 2.5 Special Tokens ← 2.1
└── 2.6 Tests ← 2.1-2.5

Phase 3 (Training)
├── 3.1 Trainer Structure ← Phase 1, 2
├── 3.2 Training Loop ← 3.1
├── 3.3 Validation ← 3.2, Phase 4
├── 3.4 MLflow ← 3.2
├── 3.5 Checkpoints ← 3.2
└── 3.6 Tests ← 3.1-3.5

Phase 4 (Evaluation)
├── 4.1 Normalization
├── 4.2 EM Metric ← 4.1
├── 4.3 F1 Metric ← 4.1
├── 4.4 Multi-level ← 4.2, 4.3
├── 4.5 Evaluator ← 4.1-4.4
└── 4.6 Tests ← 4.1-4.5

Phase 5 (Scripts)
├── 5.1 Train Script ← Phase 1-3
├── 5.2 Eval Script ← Phase 1, 2, 4
├── 5.3 Predict Script ← Phase 1, 2
├── 5.4 Makefile ← 5.1-5.3
└── 5.5 Integration Tests ← All Phases
```

---

## Execution Schedule

### Day 1-2: Data Pipeline
- Morning: Tasks 1.1, 1.2, 1.3
- Afternoon: Task 1.4
- Evening: Task 1.5, 1.6

### Day 3-4: Model Architecture
- Morning: Tasks 2.1, 2.2
- Afternoon: Tasks 2.3, 2.4
- Evening: Task 2.5, 2.6

### Day 5: Evaluation Engine
- Morning: Tasks 4.1, 4.2, 4.3
- Afternoon: Tasks 4.4, 4.5, 4.6

### Day 6-7: Training Engine
- Morning: Tasks 3.1, 3.2
- Afternoon: Tasks 3.3, 3.4
- Evening: Tasks 3.5, 3.6

### Day 8: Scripts & Integration
- Morning: Tasks 5.1, 5.2, 5.3
- Afternoon: Task 5.4, 5.5
- Evening: Final testing and fixes

---

## Progress Tracking

Use MLflow and todo list to track:
- [ ] Phase 1 Complete (6/6 tasks)
- [ ] Phase 2 Complete (6/6 tasks)
- [ ] Phase 3 Complete (6/6 tasks)
- [ ] Phase 4 Complete (6/6 tasks)
- [ ] Phase 5 Complete (5/5 tasks)

**Total: 29 tasks across 5 phases**

---

## Quick Start

To begin implementation immediately:

```bash
# Task 1.1: Create dataset structure
touch src/Project/SubProject/data/dataset.py
touch tests/data/test_dataset.py

# Open your editor and start with Task 1.1
code src/Project/SubProject/data/dataset.py
```

**Start with Task 1.1 and work sequentially!**
