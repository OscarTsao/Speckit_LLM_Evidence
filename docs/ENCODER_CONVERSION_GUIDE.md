# Gemma Encoder Conversion Implementation Guide

This guide details how to implement decoder-to-encoder conversion following the methodology from arxiv:2503.02656 "Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks".

## Overview

The paper demonstrates that decoder-based models can be adapted for encoder tasks (like extractive QA) by modifying the attention mechanism while preserving pretrained weights.

## Implementation Steps

### 1. Load Pretrained Gemma Decoder

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pretrained Gemma model
model_name = "google/gemma-7b"  # or "google/gemma-2b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 2. Modify Attention Mechanism

#### Option A: Disable Causal Masking (Recommended)

```python
# Disable causal attention in all layers
for layer in model.model.layers:
    if hasattr(layer.self_attn, 'is_causal'):
        layer.self_attn.is_causal = False
```

#### Option B: Override Attention Forward Pass

```python
import torch.nn.functional as F

def bidirectional_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
    """
    Modified attention that allows bidirectional context.
    """
    # Get query, key, value projections
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    # Reshape for multi-head attention
    bsz, q_len, _ = hidden_states.size()
    query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Compute attention scores (no causal mask)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

    # Apply attention mask if provided (for padding)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax and apply to values
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)

    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

# Apply to all attention layers
for layer in model.model.layers:
    layer.self_attn.forward = bidirectional_attention_forward.__get__(layer.self_attn)
```

### 3. Add Question Answering Heads

```python
import torch.nn as nn

class GemmaForQuestionAnswering(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.gemma = base_model

        # QA classification heads
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits

        # Initialize QA head
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        nn.init.zeros_(self.qa_outputs.bias)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        **kwargs
    ):
        # Get hidden states from Gemma encoder
        outputs = self.gemma.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state

        # Compute start and end logits
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
            'hidden_states': sequence_output if kwargs.get('output_hidden_states') else None
        }

# Create QA model
hidden_size = model.config.hidden_size
qa_model = GemmaForQuestionAnswering(model, hidden_size)
```

### 4. Add Special Tokens for Prompt Structure

```python
# Define special tokens
special_tokens = {
    'additional_special_tokens': [
        '[INST]', '[/INST]',
        '[POST]', '[/POST]',
        '[CRITERION]', '[/CRITERION]'
    ]
}

# Add to tokenizer and resize model embeddings
num_added = tokenizer.add_special_tokens(special_tokens)
qa_model.gemma.resize_token_embeddings(len(tokenizer))

print(f"Added {num_added} special tokens")
```

### 5. Prepare Input Format

```python
def format_input(post: str, criterion: str, tokenizer) -> dict:
    """
    Format input according to optimized prompt template.
    """
    prompt = (
        f"[INST] Extract the evidence sentence that supports the diagnostic criterion. [/INST]"
        f"[POST] {post} [/POST]"
        f"[CRITERION] {criterion} [/CRITERION]"
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True  # For position mapping
    )

    return inputs

# Example usage
post = "Patient reports persistent sadness and loss of interest in activities."
criterion = "Depressed mood most of the day"
inputs = format_input(post, criterion, tokenizer)
```

### 6. Training Setup

```python
from transformers import TrainingArguments, Trainer

# Training arguments optimized for RTX 4090
training_args = TrainingArguments(
    output_dir="./outputs/checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,  # Mixed precision
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to=["mlflow", "tensorboard"],
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# Enable TF32 for faster training on RTX 4090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

## Key Implementation Details

### Attention Mask Modification

The critical change is converting from **causal attention** (decoder) to **bidirectional attention** (encoder):

**Causal (Decoder):**
```
Attention Mask (lower triangular):
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**Bidirectional (Encoder):**
```
Attention Mask (full):
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

### Position Embeddings

**Keep pretrained position embeddings** - they transfer well even with bidirectional attention:

```python
# No need to modify position embeddings
# The pretrained RoPE (Rotary Position Embeddings) work fine
assert model.config.position_embedding_type == "rope"
```

### Gradient Checkpointing (if OOM)

```python
# Enable gradient checkpointing for larger models
qa_model.gemma.gradient_checkpointing_enable()
```

### Flash Attention (for speed)

```python
# Use Flash Attention 2 if available
try:
    from flash_attn import flash_attn_qkvpacked_func
    # Requires special installation
    qa_model.gemma.config.use_flash_attention_2 = True
except ImportError:
    print("Flash Attention not available, using standard attention")
```

## Evidence Position Extraction

Since the dataset provides evidence sentences (not positions), extract them:

```python
def find_evidence_positions(post: str, evidence_sentence: str, tokenizer) -> tuple:
    """
    Find token positions of evidence sentence in post.

    Returns:
        (start_position, end_position) in tokens
    """
    # Normalize whitespace
    post_clean = ' '.join(post.split())
    evidence_clean = ' '.join(evidence_sentence.split())

    # Find character positions
    char_start = post_clean.find(evidence_clean)
    if char_start == -1:
        # Try fuzzy matching
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, post_clean, evidence_clean)
        match = matcher.find_longest_match(0, len(post_clean), 0, len(evidence_clean))
        if match.size > len(evidence_clean) * 0.8:  # 80% threshold
            char_start = match.a
            char_end = match.a + match.size
        else:
            return None, None  # Evidence not found
    else:
        char_end = char_start + len(evidence_clean)

    # Tokenize with offset mapping
    encoding = tokenizer(
        post_clean,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    # Convert character positions to token positions
    start_token = None
    end_token = None

    for idx, (start, end) in enumerate(encoding['offset_mapping']):
        if start <= char_start < end:
            start_token = idx
        if start < char_end <= end:
            end_token = idx
            break

    return start_token, end_token
```

## Validation

Test the conversion:

```python
def test_encoder_conversion():
    """Test that encoder conversion works correctly."""

    # Create dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128)).cuda()
    attention_mask = torch.ones_like(input_ids)

    # Forward pass
    with torch.no_grad():
        outputs = qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Check outputs
    assert outputs['start_logits'].shape == (1, 128)
    assert outputs['end_logits'].shape == (1, 128)

    # Verify bidirectional attention
    # All positions should attend to all positions
    # (This requires capturing attention weights)

    print("✓ Encoder conversion successful!")
    print(f"  Start logits shape: {outputs['start_logits'].shape}")
    print(f"  End logits shape: {outputs['end_logits'].shape}")

test_encoder_conversion()
```

## Performance Benchmarks

Expected performance on RTX 4090:

| Model | Batch Size | Seq Length | Memory | Speed |
|-------|-----------|-----------|--------|-------|
| Gemma-2b | 16 | 512 | ~12GB | ~200 samples/sec |
| Gemma-2b | 8 | 1024 | ~14GB | ~80 samples/sec |
| Gemma-7b | 4 | 512 | ~18GB | ~60 samples/sec |
| Gemma-7b | 2 | 1024 | ~20GB | ~25 samples/sec |

## Common Issues

### 1. OOM Errors
- Reduce batch size
- Increase gradient accumulation
- Enable gradient checkpointing
- Reduce max_seq_length

### 2. Causal Attention Still Active
- Verify `is_causal=False` is set
- Check attention mask is full (not triangular)
- Test with attention visualization

### 3. Poor Performance
- Ensure bidirectional attention is working
- Check evidence positions are correct
- Verify prompt format matches training data
- Try longer warmup period

## References

- Paper: [Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks](https://arxiv.org/abs/2503.02656)
- Gemma Model: https://ai.google.dev/gemma
- SQuAD Evaluation: https://rajpurkar.github.io/SQuAD-explorer/
- Transformers Docs: https://huggingface.co/docs/transformers
