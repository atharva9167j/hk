# Transitioning to HK from Existing Tools

If you are currently using tools like Ollama, LM Studio, llama.cpp, Unsloth, Hugging Face Transformers, PEFT, or SafeTensors, moving over to HK is straightforward. 

I designed HK specifically so you do not have to throw away your existing models, weights, or datasets. You can substitute HK directly into your workflow and immediately get faster execution, lighter memory usage, and access to continuous growth features.

Here is how to transition from each major tool.

---

## 1. Transitioning from Ollama, LM Studio, and llama.cpp (Local Inference)

If you run models locally using Ollama, LM Studio, or llama.cpp, you usually deal with GGUF files. While GGUF works for static quantized inference, it is locked into static architectures, requires heavy C++ runtimes, and lacks dynamic growth or training inside the container.

### Converting Existing GGUF Models to HK
You can convert any existing GGUF model directly into an HK container using the native CLI:

```bash
# Convert a GGUF file directly to HK format
hk convert-gguf path/to/model.gguf path/to/model.hk
```

The conversion happens in a streaming pass. The metadata, vocabulary, and tensor weights are preserved.

### Running Local Inference (Replacing `ollama run`)
Instead of running a heavy background daemon, I built a standalone native binary that uses less than 3 MB of idle RAM and starts in under 60 milliseconds:

```bash
# One-off prompt generation
hk run model.hk -p "Explain how a compiler works in simple terms" -n 256

# Interactive chat session in your terminal
hk chat model.hk --temp 0.7
```

### Offloading Layers to GPU
Just like `-ngl` in llama.cpp or GPU offloading in Ollama, I added `-ngl` so you can choose how many layers run on your GPU:

```bash
# Offload 24 layers to GPU, keeping the rest on CPU
hk chat model.hk -ngl 24
```

If you do not have a dedicated GPU, you do not need to configure anything. HK automatically detects your CPU vector extensions (AVX2, AVX-512, ARM NEON) and runs optimized 4-row unrolled SIMD math across your registers.

---

## 2. Transitioning from Hugging Face Transformers & SafeTensors

If you use Hugging Face `transformers` and `safetensors`, I built direct drop-in replacements for both loading and saving.

### Drop-in Replacement for `safetensors.torch`
In any script where you currently use `safetensors.torch`:

```python
# Old way:
# from safetensors.torch import save_file, load_file, safe_open

# New way with HK:
from hk.torch import save_file, load_file, safe_open

# Saving tensors (identical syntax, plus automatic tied-weight detection)
save_file(tensors_dict, "model.hk")

# Loading tensors (identical syntax, zero-copy memory mapped)
loaded = load_file("model.hk")

# Lazy inspection and multidimensional slicing
with safe_open("model.hk", framework="pt") as f:
    for key in f.keys():
        print(key, f.get_slice(key).get_shape())
    # Grab only a slice of a large matrix without loading the whole file
    head_weights = f.get_slice("model.layers.0.mlp.gate_proj.weight")[0:128, :]
```

### Drop-in Replacement for `AutoModelForCausalLM`
Instead of pulling through massive dependency trees, use HK's native model loader:

```python
# Old way:
# from transformers import AutoModelForCausalLM, AutoTokenizer

# New way with HK:
from hk import AutoModelForCausalLM, AutoTokenizer

# Load directly from an HK file or directory
model = AutoModelForCausalLM.from_pretrained("model.hk", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("model.hk")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))
```

### Converting SafeTensors to HK
If you have folders with `.safetensors` files from Hugging Face:

```bash
# Convert a SafeTensors checkpoint to HK
hk convert-safetensors model.safetensors model.hk
```

Or in Python:

```python
from hk.native import convert_safetensors_to_hk

convert_safetensors_to_hk("model.safetensors", "model.hk")
```

---

## 3. Transitioning from Unsloth, PEFT, and Hugging Face Trainer

Training and fine-tuning models with traditional setups often means stitching together PyTorch, Hugging Face `Trainer`, `peft` for LoRA, `bitsandbytes` for quantization, and external scripts for logging. When loss plateaus, you have no way to grow the model, so you end up stuck.

I brought all training workflows into a single, clean training engine: `HKTrainer`. Whether you are performing Full Fine-Tuning (FFT) across all parameters or parameter-efficient fine-tuning (QLoRA), I designed it for direct substitution.

### Side-by-Side Comparison

#### Traditional Setup (Transformers + PEFT):
```python
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, peft_config)

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

#### How I Built It in HK (Full Fine-Tuning or QLoRA):
```python
from hk import HKConfig, HKForCausalLM
from hk.trainer import HKTrainer, HKTrainingArguments

# 1. Load model
config = HKConfig.from_pretrained("model.hk")
model = HKForCausalLM(config)

# 2. Configure training
args = HKTrainingArguments(
    output_dir="./results",
    batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    
    # Mode A: Full Fine-Tuning (FFT) - update 100% of parameters
    use_qlora=False,
    
    # Mode B: Parameter-Efficient QLoRA - native low-rank adapters (no peft needed)
    # use_qlora=True,
    # lora_rank=8,
    # lora_alpha=16.0,
    
    # When loss stagnates for 5 steps, automatically widen capacity (Net2Net)
    enable_adaptive_growth=True,
    growth_patience=5,
    growth_width_factor=1.25,
    protect_base_capacity=True, # Prevent catastrophic forgetting
)

# 3. Train
trainer = HKTrainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# 4. Save directly with version history and rollback support
model.save_pretrained("./results/evolved_model.hk")
```

### Why This Transition Helps You
1. Full Fine-Tuning (FFT) & QLoRA in One Engine: Switch between FFT and QLoRA with a single boolean flag (`use_qlora`). No need to rewrite training scripts or reformat datasets.
2. Zero Extra Dependencies: You do not need `bitsandbytes`, `peft`, or complicated CUDA compilation flags just to do LoRA fine-tuning. It is built in.
3. Plateau-Triggered Widening: If your dataset has new concepts that the existing parameter count cannot absorb, traditional training overfits or gets stuck. HK automatically widens the intermediate layers (SwiGLU) without breaking what the model already knows.
4. In-Container Checkpointing: Instead of saving 10 separate folders with duplicate 5 GB weights, training checkpoints and adapter deltas are recorded directly into the file appendix.
5. Instant Rollback: If a fine-tuning run overfits or gets poisoned, run `hk rollback model.hk --generation 1` to restore the previous state in less than a millisecond.

---

## 4. Transitioning PyTorch `nn.Module` Checkpoints (`.pt` / `.pth`)

If you currently use `torch.save(model.state_dict(), "model.pt")`:

```python
# Old way:
# torch.save(model.state_dict(), "model.pt")
# model.load_state_dict(torch.load("model.pt"))

# New way with HK:
import hk.torch as hkt

# Saves with zero-overhead binary layout and tied-weight deduplication
hkt.save_model(model, "model.hk")

# Loads cleanly with strict parameter matching and zero memory leak
hkt.load_model(model, "model.hk", strict=True)
```

Advantages:
- Safe against arbitrary code execution (no Python `pickle` vulnerabilities).
- Zero-copy memory mapping on Windows, Linux, and macOS.
- Files do not leak file descriptors on Windows when unlinked.

---

## 5. Transitioning NumPy, JAX, and Flax Workflows

If your pipeline uses NumPy arrays, JAX arrays, or Flax modules, HK provides direct native bridges:

### NumPy
```python
import hk.numpy as hknp
import numpy as np

arrays = {"features": np.random.randn(128, 256).astype(np.float32)}
hknp.save_file(arrays, "data.hk")

loaded = hknp.load_file("data.hk")
print(loaded["features"].shape)
```

### JAX / Flax
```python
import hk.jax as hkjax
import jax.numpy as jnp

params = {"kernel": jnp.ones((64, 64), dtype=jnp.float32)}
hkjax.save_file(params, "flax_weights.hk")

loaded = hkjax.load_file("flax_weights.hk")
```

---

## 6. Command Equivalents Cheat Sheet

| Action | Old Way (Ollama / LLaMA.cpp / HF) | HK Equivalent |
| :--- | :--- | :--- |
| **Run model locally** | `ollama run llama3` | `hk run model.hk -p "Hello"` |
| **Interactive chat** | `ollama run llama3` or `./llama-cli -m model.gguf` | `hk chat model.hk` |
| **Offload to GPU** | `./llama-cli -m model.gguf -ngl 33` | `hk chat model.hk -ngl 33` |
| **Convert GGUF** | `python convert-hf-to-gguf.py` | `hk convert-gguf model.gguf model.hk` |
| **Convert SafeTensors** | Manual conversion script | `hk convert-safetensors in.safetensors out.hk` |
| **Export to GGUF** | Complex export script | `hk export -f gguf model.hk out.gguf` |
| **Export to SafeTensors**| Complex export script | `hk export -f safetensors model.hk out.safetensors` |
| **Inspect Model TOC** | `gguf-dump` or Python script | `hk inspect model.hk` |
| **Verify Integrity** | Manual SHA check | `hk verify model.hk` |
| **Edit Chat Template** | Rewrite entire file with Python | `hk metadata set model.hk tokenizer.chat_template "..."` |
| **Fine-tune with LoRA** | Hugging Face + PEFT + bitsandbytes | `HKTrainer` with `use_qlora=True` |
| **Rollback Fine-tune** | Re-download or restore from backup | `hk rollback model.hk --generation 1` |
