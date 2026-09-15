# Training and Fine-Tuning with HKTrainer

In this guide, I cover all training and adaptation procedures supported in HK: Full Fine-Tuning (FFT), Parameter-Efficient Fine-Tuning (QLoRA), Supervised Fine-Tuning (SFT), Continued Pre-Training (CPT), and Pre-Training from scratch.

---

## Supported Training Procedures

I designed `HKTrainer` to be fully compatible with whatever training procedure your project requires:

```
+--------------------------------------------------------------------------------+
|                           HK TRAINING CAPABILITIES                             |
+--------------------------------------------------------------------------------+
| 1. Full Fine-Tuning (FFT)       | Updates 100% of model parameters             |
| 2. Parameter-Efficient (QLoRA)  | Native low-rank adapters, zero extra packages|
| 3. Supervised Fine-Tuning (SFT) | Instruction datasets, chat template masking  |
| 4. Continued Pre-Training (CPT) | Domain adaptation with vocab expansion       |
| 5. Pre-Training from Scratch    | Distributed raw sharding, 4KB/16KB page DMA  |
| 6. Adaptive Growth Training     | Plateau-triggered Net2Net SwiGLU widening    |
+--------------------------------------------------------------------------------+
```

---

## 1. Full Fine-Tuning (FFT)

In Full Fine-Tuning, every single parameter across every layer (self-attention projections, SwiGLU MLPs, RMSNorm weights, and embeddings) receives gradient updates.

### How I Implemented FFT in HK
- Set `use_qlora=False` in `HKTrainingArguments`.
- Base weights are memory-mapped directly from the `.hk` file with copy-on-write page safety.
- The AdamW optimizer tracks first and second moment states.
- When checkpoints are saved, HK writes delta updates and metrics directly into the file's Appendix DAG, eliminating the need to duplicate 10 separate full model copies on disk.

```python
import torch
from hk import HKConfig, HKForCausalLM
from hk.trainer import HKTrainer, HKTrainingArguments

# Load model
config = HKConfig.from_pretrained("model.hk")
model = HKForCausalLM(config)

# Configure Full Fine-Tuning (FFT)
args = HKTrainingArguments(
    output_dir="./checkpoints_fft",
    learning_rate=5e-5,
    batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=50,
    
    # FFT mode: 100% parameter updates
    use_qlora=False,
    
    # Optional: Enable plateau growth to widen layers if FFT plateaus
    enable_adaptive_growth=True,
    growth_patience=3,
    growth_width_factor=1.20,
)

trainer = HKTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

model.save_pretrained("./checkpoints_fft/final_model.hk")
```

---

## 2. Parameter-Efficient Fine-Tuning (Native QLoRA)

If you have limited VRAM (for example, on a laptop with a 4GB or 6GB GPU), training all parameters with FFT can exceed hardware limits.

I built native QLoRA directly into the engine without needing third-party libraries (`peft`, `bitsandbytes`, or custom CUDA builds):

```python
args = HKTrainingArguments(
    output_dir="./checkpoints_qlora",
    learning_rate=2e-4,
    batch_size=4,
    num_train_epochs=3,
    
    # Native QLoRA
    use_qlora=True,
    lora_rank=8,
    lora_alpha=16.0,
    
    # Protect base capacity
    protect_base_capacity=True,
)
```

How it works:
1. Base weights remain frozen in memory.
2. Low-rank adapter matrices ($A$ and $B$) are attached to attention and MLP projection layers.
3. Only the low-rank adapters receive gradients, cutting training memory by over 70%.
4. Trained adapters are persisted into the container's Appendix region.

---

## 3. Supervised Fine-Tuning (SFT) & Instruction Tuning

For instruction datasets (Alpaca, ShareGPT, or custom Q&A pairs), SFT trains the model to follow user prompts:

- Format your dataset using the model's embedded Jinja2 chat template (`tokenizer.apply_chat_template`).
- Use prompt loss masking: set the labels for user instruction tokens to `-100` so that cross-entropy loss is computed exclusively on the assistant response tokens.

```python
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.examples = []
        for s in samples:
            prompt_text = f"<user>\n{s['instruction']}\n<assistant>\n"
            full_text = prompt_text + s["response"]
            
            prompt_ids = tokenizer.encode(prompt_text)
            full_ids = tokenizer.encode(full_text)
            
            input_ids = full_ids[:max_len]
            labels = list(input_ids)
            # Mask instruction tokens from loss
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
                
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
```

---

## 4. Continued Pre-Training (CPT) & Domain Adaptation

When adapting a generalist model to a new programming language (e.g. Zig or Rust) or technical domain (e.g. legal or clinical):

1. Dynamic Vocabulary Expansion:
   If the domain contains specialized syntax or terms, expand the vocabulary first so tokens are not split into tiny byte fragments:
   ```python
   from hk.adaptive.growth import expand_vocab
   expand_vocab(model.model.embed_tokens, model.lm_head, new_vocab_size=32500)
   ```
2. Plasticity Isolation:
   Enable `protect_base_capacity=True`. HK generates gradient masks that dampen updates to pre-existing base neurons while allowing newly added neurons to learn domain patterns rapidly.
3. Training:
   Run causal language modeling on raw domain text files with a low learning rate and cosine decay.

---

## 5. Pre-Training from Scratch

If you are training a new architecture from scratch:
- Define dimensions using `HKConfig`.
- Use `save_sharded_raw` to partition checkpoints into multi-file shards (`model.hk.index.json`) for multi-GPU data parallel or tensor parallel setups.
- Benefit from universal 4KB / 16KB page alignment: storage engines stream directly into host RAM and PCIe DMA channels without format conversions.

---

## 6. Plateau-Triggered Dynamic Growth

When training on complex datasets, models frequently hit a convergence plateau where validation loss stops dropping.

With `enable_adaptive_growth=True`:
1. `HKTrainer` monitors validation loss across evaluation intervals.
2. If loss does not improve for `growth_patience` evaluations, the loop automatically pauses.
3. The native Zig `GrowthGovernor` checks available physical RAM/VRAM.
4. Net2WiderNet widens intermediate SwiGLU layers by `growth_width_factor` with exact **0.000000** Day-0 output preservation.
5. Training resumes immediately, allowing the model to break through the plateau.

---

## Training Arguments Reference (`HKTrainingArguments`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `output_dir` | `"./results"` | Directory where checkpoints and exports are written. |
| `learning_rate` | `3e-4` | Peak learning rate for AdamW optimizer. |
| `batch_size` | `4` | Training batch size per device. |
| `num_train_epochs` | `3` | Total number of training epochs. |
| `gradient_accumulation_steps` | `1` | Number of update steps to accumulate before backward pass. |
| `warmup_steps` | `0` | Linear warmup steps for learning rate schedule. |
| `weight_decay` | `0.01` | Weight decay rate for AdamW optimizer. |
| `max_grad_norm` | `1.0` | Maximum gradient norm for gradient clipping. |
| `use_qlora` | `False` | Enable native low-rank adapter fine-tuning. If `False`, Full Fine-Tuning (FFT) is performed. |
| `lora_rank` | `8` | Rank dimension for LoRA adapter matrices. |
| `lora_alpha` | `16.0` | Scaling factor for LoRA updates. |
| `enable_adaptive_growth` | `False` | Automatically widen model when validation loss plateaus. |
| `growth_patience` | `5` | Number of evaluations with stagnant loss before triggering growth. |
| `growth_width_factor` | `1.25` | Multiplier for widening feed-forward dimensions. |
| `protect_base_capacity` | `True` | Apply gradient masks to prevent catastrophic forgetting. |
| `alignment` | `128` | Memory alignment boundary for saved checkpoints (128, 4096, 16384). |
