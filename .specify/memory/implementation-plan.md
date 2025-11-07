# Technical Implementation Plan - Extractive QA System

**Project**: Speckit LLM Evidence
**Date**: 2025-11-07
**Phase**: Planning
**Status**: Ready for Implementation

---

## Overview

This plan outlines the technical approach for implementing an extractive QA system for criteria matching, using Gemma encoder architecture with bidirectional attention following arxiv:2503.02656.

## Implementation Phases

### Phase 1: Data Pipeline (Priority: High, Days: 2-3)

#### 1.1 Dataset Loader
**File**: `src/Project/SubProject/data/dataset.py`

**Components**:
```python
class RedsM5Dataset(torch.utils.data.Dataset):
    """
    Dataset for REDSM5 criteria matching task.
    """
    - __init__(self, data_path, tokenizer, config)
    - __getitem__(self, idx)
    - __len__(self)
    - _load_data(self)
    - _filter_special_cases(self)
    - _find_evidence_position(self, post, evidence_sentence)
    - _prepare_input(self, post, criterion)
```

**Technical Approach**:
1. Load dataset from HuggingFace or local JSON
2. Filter out "special case" symptom category
3. For each sample:
   - Extract post, criterion, evidence_sentence
   - Locate evidence sentence in post (hybrid matching)
   - Convert to token positions
   - Format with special tokens
   - Handle truncation preserving evidence

**Evidence Position Extraction Algorithm**:
```python
def find_evidence_position(post: str, evidence: str) -> tuple[int, int]:
    # 1. Normalize whitespace
    post_clean = ' '.join(post.split())
    evidence_clean = ' '.join(evidence.split())

    # 2. Try exact match
    char_start = post_clean.find(evidence_clean)

    # 3. Fallback to fuzzy match (85% threshold)
    if char_start == -1:
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, post_clean, evidence_clean)
        match = matcher.find_longest_match(...)
        if match.size / len(evidence_clean) >= 0.85:
            char_start, char_end = match.a, match.a + match.size
        else:
            return None, None  # Skip this example

    # 4. Convert char positions to token positions
    encoding = tokenizer(post_clean, return_offsets_mapping=True)
    start_token = char_to_token(encoding, char_start)
    end_token = char_to_token(encoding, char_end)

    return start_token, end_token
```

**Dependencies**:
- HuggingFace datasets library
- difflib (stdlib)
- Gemma tokenizer

**Testing**:
- Unit test: Evidence position extraction with exact match
- Unit test: Evidence position extraction with fuzzy match
- Unit test: Special case filtering
- Integration test: Full dataset loading

---

### Phase 2: Model Architecture (Priority: High, Days: 2-3)

#### 2.1 Gemma Encoder with QA Heads
**File**: `src/Project/SubProject/models/model.py`

**Components**:
```python
class GemmaEncoder(nn.Module):
    """
    Gemma decoder converted to encoder for extractive QA.
    """
    - __init__(self, model_name: str, config: dict)
    - forward(self, input_ids, attention_mask, start_positions, end_positions)
    - _convert_to_encoder(self)
    - _add_qa_heads(self)
    - _add_special_tokens(self, tokenizer)
```

**Technical Approach**:

1. **Load Pretrained Gemma**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b",  # or gemma-2b
    torch_dtype=torch.float16,
    device_map="auto"
)
```

2. **Convert to Encoder**:
```python
# Disable causal attention
for layer in model.model.layers:
    layer.self_attn.is_causal = False
```

3. **Add QA Heads**:
```python
class GemmaForQuestionAnswering(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.gemma = base_model
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start, end
        nn.init.xavier_uniform_(self.qa_outputs.weight)

    def forward(self, input_ids, attention_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.gemma.model(input_ids, attention_mask)
        logits = self.qa_outputs(outputs.last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)

        if start_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {
            'loss': total_loss,
            'start_logits': start_logits.squeeze(-1),
            'end_logits': end_logits.squeeze(-1)
        }
```

4. **Add Special Tokens**:
```python
special_tokens = {
    'additional_special_tokens': [
        '[INST]', '[/INST]',
        '[POST]', '[/POST]',
        '[CRITERION]', '[/CRITERION]'
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.gemma.resize_token_embeddings(len(tokenizer))
```

**Dependencies**:
- Transformers library
- PyTorch 2.2+
- CUDA 12.1

**Testing**:
- Unit test: Model forward pass
- Unit test: Loss calculation
- Unit test: Special token integration
- Test: Bidirectional attention (not causal)

---

### Phase 3: Training Engine (Priority: High, Days: 2-3)

#### 3.1 Training Loop
**File**: `src/Project/SubProject/engine/train_engine.py`

**Components**:
```python
class Trainer:
    """
    Training engine with MLflow integration.
    """
    - __init__(self, model, train_dataset, val_dataset, config)
    - train(self)
    - validate(self)
    - _train_epoch(self, epoch)
    - _save_checkpoint(self, epoch, metrics)
    - _log_metrics(self, metrics, step)
```

**Technical Approach**:

1. **Training Configuration**:
```python
# RTX 4090 optimized settings
training_args = {
    'learning_rate': 2e-5,
    'batch_size': 4,  # For Gemma-7b
    'gradient_accumulation_steps': 4,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'fp16': True,
    'eval_steps': 500,
    'save_steps': 500,
}

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

2. **Training Loop**:
```python
def train_epoch(self, epoch):
    self.model.train()
    for step, batch in enumerate(tqdm(self.train_loader)):
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

        # Log metrics
        if step % self.log_steps == 0:
            mlflow.log_metric('train_loss', loss.item(), step=step)

        # Validate
        if step % self.eval_steps == 0:
            val_metrics = self.validate()
            mlflow.log_metrics(val_metrics, step=step)
```

3. **MLflow Integration**:
```python
import mlflow

# Start run
with mlflow.start_run(run_name=f"gemma-7b-{timestamp}"):
    # Log config
    mlflow.log_params(config)
    mlflow.log_param('model_name', 'google/gemma-7b')

    # Train
    for epoch in range(num_epochs):
        train_metrics = train_epoch(epoch)
        val_metrics = validate()

        # Log metrics
        mlflow.log_metrics({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_em': val_metrics['exact_match']
        })

        # Save checkpoint
        if val_metrics['f1'] > best_f1:
            mlflow.pytorch.log_model(model, 'model')
```

**Dependencies**:
- MLflow
- PyTorch
- tqdm
- tensorboard

**Testing**:
- Unit test: Training step
- Integration test: Single epoch training on small dataset
- Test: Checkpoint saving/loading

---

### Phase 4: Evaluation Engine (Priority: High, Days: 1-2)

#### 4.1 SQuAD-style Evaluation
**File**: `src/Project/SubProject/engine/eval_engine.py`

**Components**:
```python
class Evaluator:
    """
    SQuAD-style evaluation with multi-level metrics.
    """
    - __init__(self, model, dataset, config)
    - evaluate(self) -> dict
    - _compute_em(self, predictions, references) -> float
    - _compute_f1(self, predictions, references) -> float
    - _normalize_text(self, text) -> str
    - _compute_multi_level_metrics(self, predictions, references)
```

**Technical Approach**:

1. **Text Normalization** (SQuAD-style):
```python
def normalize_text(text: str) -> str:
    """SQuAD-style text normalization."""
    import re
    import string

    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace
    text = ' '.join(text.split())

    return text
```

2. **Exact Match**:
```python
def compute_em(predictions, references):
    """Exact match after normalization."""
    em_count = 0
    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            em_count += 1
    return em_count / len(predictions)
```

3. **F1 Score** (token-level):
```python
def compute_f1(prediction, reference):
    """Token-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1
```

4. **Multi-level Metrics**:
```python
def compute_multi_level_metrics(predictions, references):
    """Character, token, and word-level metrics."""
    return {
        'char_level_em': compute_char_em(predictions, references),
        'token_level_f1': compute_token_f1(predictions, references),
        'word_level_f1': compute_word_f1(predictions, references),
    }
```

**Dependencies**:
- HuggingFace evaluate library
- collections.Counter

**Testing**:
- Unit test: Normalization
- Unit test: EM calculation
- Unit test: F1 calculation
- Test: Multi-level metrics

---

### Phase 5: Scripts (Priority: Medium, Days: 1-2)

#### 5.1 Training Script
**File**: `scripts/train.py`

```python
#!/usr/bin/env python3
import argparse
import yaml
import mlflow
from src.Project.SubProject.data.dataset import RedsM5Dataset
from src.Project.SubProject.models.model import GemmaForQuestionAnswering
from src.Project.SubProject.engine.train_engine import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/training_config.yaml')
    parser.add_argument('--model-name', default='google/gemma-7b')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup MLflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('criteria-matching-extractive-qa')

    # Load data
    train_dataset = RedsM5Dataset('data/redsm5', split='train', config=config)
    val_dataset = RedsM5Dataset('data/redsm5', split='validation', config=config)

    # Load model
    model = GemmaForQuestionAnswering(args.model_name, config)

    # Train
    trainer = Trainer(model, train_dataset, val_dataset, config)
    trainer.train()

if __name__ == '__main__':
    main()
```

#### 5.2 Evaluation Script
**File**: `scripts/evaluate.py`

```python
#!/usr/bin/env python3
import argparse
from src.Project.SubProject.engine.eval_engine import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    # Load model and evaluate
    evaluator = Evaluator(args.model_path, split=args.split)
    metrics = evaluator.evaluate()

    print(f"Evaluation Results:")
    print(f"  Exact Match: {metrics['exact_match']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")

if __name__ == '__main__':
    main()
```

---

## Implementation Order

### Week 1: Foundation
1. **Day 1-2**: Data pipeline
   - Implement RedsM5Dataset
   - Evidence position extraction
   - Unit tests

2. **Day 3-4**: Model architecture
   - Implement GemmaForQuestionAnswering
   - Encoder conversion
   - Special tokens

3. **Day 5**: Integration testing
   - Test data → model flow
   - Verify bidirectional attention
   - Test on small batch

### Week 2: Training & Evaluation
1. **Day 1-2**: Training engine
   - Implement Trainer class
   - MLflow integration
   - Checkpoint management

2. **Day 3**: Evaluation engine
   - Implement SQuAD metrics
   - Multi-level evaluation

3. **Day 4**: Scripts & CLI
   - Training script
   - Evaluation script
   - Makefile integration

4. **Day 5**: Testing & Documentation
   - End-to-end testing
   - Documentation updates
   - Code review

---

## Technical Considerations

### Memory Optimization (RTX 4090 - 24GB)
- **Gemma-7b**: Batch size 4, gradient accumulation 4 → effective batch 16
- **Gemma-2b**: Batch size 8, gradient accumulation 2 → effective batch 16
- Use fp16 mixed precision
- Enable gradient checkpointing if OOM

### Performance Optimization
- Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`
- Use cudnn benchmark: `torch.backends.cudnn.benchmark = True`
- Pin memory in DataLoader
- Prefetch factor = 2

### Data Quality
- Log all skipped examples to `data/problematic_samples.json`
- Generate preprocessing statistics
- Validate evidence positions

---

## Testing Strategy

### Unit Tests
- `tests/data/test_dataset.py`: Dataset loading and preprocessing
- `tests/models/test_model.py`: Model forward pass and loss
- `tests/engine/test_train.py`: Training step
- `tests/engine/test_eval.py`: Metrics calculation

### Integration Tests
- `tests/integration/test_pipeline.py`: End-to-end data → model → evaluation
- `tests/integration/test_training.py`: Small-scale training run

### Validation
- Train on 10% of data for 1 epoch (smoke test)
- Verify loss decreases
- Check GPU utilization > 80%

---

## Dependencies Map

```
Data Pipeline
├── datasets (HuggingFace)
├── transformers (tokenizer)
└── difflib (fuzzy matching)

Model
├── transformers (Gemma)
├── torch
└── torch.nn

Training
├── mlflow
├── tqdm
├── tensorboard
└── accelerate (optional)

Evaluation
├── evaluate (HuggingFace)
└── collections

Scripts
├── argparse
├── yaml
└── logging
```

---

## Risk Mitigation

### Risk 1: OOM during training
**Mitigation**:
- Start with Gemma-2b
- Reduce batch size
- Enable gradient checkpointing

### Risk 2: Evidence matching failures
**Mitigation**:
- Fuzzy matching fallback
- Log all failures
- Manual review of problematic samples

### Risk 3: Poor baseline performance
**Mitigation**:
- Verify bidirectional attention works
- Check evidence position accuracy
- Use pretrained SQuAD model for comparison

### Risk 4: Slow training
**Mitigation**:
- Use fp16 mixed precision
- Optimize DataLoader
- Profile with PyTorch profiler

---

## Success Criteria

### Phase 1 Complete
- [ ] Dataset loads successfully
- [ ] Evidence positions extracted correctly
- [ ] Special cases filtered
- [ ] Unit tests pass

### Phase 2 Complete
- [ ] Model instantiates without errors
- [ ] Bidirectional attention verified
- [ ] Forward pass works
- [ ] Loss calculation correct

### Phase 3 Complete
- [ ] Training loop runs
- [ ] Loss decreases over time
- [ ] MLflow logs metrics
- [ ] Checkpoints saved

### Phase 4 Complete
- [ ] Evaluation script runs
- [ ] Metrics match SQuAD style
- [ ] Multi-level metrics computed

### Project Complete
- [ ] F1 > 60% on test set
- [ ] EM > 45% on test set
- [ ] Training time < 8 hours (no strict limit)
- [ ] All tests pass
- [ ] Documentation complete

---

## Next Steps

1. Review this plan with team
2. Set up development environment (`make setup`)
3. Begin Phase 1: Data Pipeline
4. Daily progress tracking in MLflow

---

**Plan Status**: ✅ Approved for Implementation
**Estimated Timeline**: 8-10 days
**Target Start**: Immediately
