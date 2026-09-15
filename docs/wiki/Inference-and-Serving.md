# Inference and Serving

In this guide, I explain how to run high-performance inference using HK, both via the native standalone CLI and through Python.

---

## 1. Local CLI Inference (Fast, Lightweight, Zero-Config)

I built a standalone native executable written in Zig for HK. It does not require Python, PyTorch, or CUDA toolkits to run high-throughput CPU inference.

### Process Footprint
- Idle Memory: 2.8 MB of RAM (compared to 200+ MB for Python/PyTorch runtimes).
- Startup Time: Under 55 milliseconds.
- SIMD Kernels: Automatically detects AVX2, AVX-512, or ARM NEON on your processor.

### One-Off Text Generation (`hk run`)
To generate text from a prompt:

```bash
hk run model.hk -p "Write a quick Python script to download an image from a URL" -n 128
```

Key flags:
- `-p, --prompt`: Input text prompt.
- `-n, --n-predict`: Maximum number of tokens to generate (default: 128).
- `--temp`: Sampling temperature (e.g. `0.7`). Higher means more creative; `0.0` is greedy argmax.
- `--top-p`: Nucleus sampling cutoff (e.g. `0.9`).
- `--top-k`: Top-K candidate pool size (e.g. `40`).
- `--repeat-penalty`: Penalizes repeated token loops (default: `1.10`).

### Interactive Chat Session (`hk chat`)
To start an interactive chat session directly in your terminal:

```bash
hk chat model.hk --temp 0.7
```

In chat mode:
- HK reads the chat template embedded inside the model metadata.
- Context history is preserved across turns.
- Type your prompt and press Enter. Type `/exit` or `/quit` to leave.

### Dynamic CPU/GPU Offloading (`-ngl`)
If your machine has a dedicated GPU (e.g., an NVIDIA RTX laptop card or desktop GPU), I added `-ngl` so you can offload layers to VRAM while keeping the rest on CPU:

```bash
# Offload 16 transformer blocks to GPU
hk chat model.hk -ngl 16

# Offload all layers to GPU (if VRAM is large enough)
hk run model.hk -p "Explain gravity" -ngl 99
```

If you do not specify `-ngl`, HK executes cleanly on your CPU using register-unrolled SIMD kernels.

---

## 2. Python Inference

If your application lives in Python, you have multiple ways to run inference:

### Using `AutoModelForCausalLM`
HK provides a familiar interface matching Hugging Face conventions:

```python
from hk import AutoModelForCausalLM, AutoTokenizer

# 1. Load model with zero-copy memory mapping
model = AutoModelForCausalLM.from_pretrained("model.hk", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("model.hk")

# 2. Tokenize input
prompt = "The key difference between CPU and GPU compute is"
inputs = tokenizer(prompt, return_tensors="pt")

# 3. Generate tokens
output_ids = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# 4. Decode
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### High-Level Pipelines (`HKPipeline`)
For standard tasks, you can use the unified pipeline:

```python
from hk.pipeline import pipeline

# Create a text-generation pipeline
generator = pipeline("text-generation", model="model.hk")

results = generator("What are the advantages of zero-copy storage?", max_new_tokens=100)
print(results[0]["generated_text"])
```

---

## 3. High-Performance Native Zig C-ABI Engine

If you are building an application in C, C++, Rust, or Go, you can connect directly to the native C-ABI dynamic library (`hk.dll`, `libhk.so`, or `libhk.dylib`) without Python overhead:

```c
#include "hk.h"

// 1. Open model container
hk_model_t* model = hk_model_open("model.hk");

// 2. Initialize inference context
hk_context_t* ctx = hk_context_create(model, 32); // 32 layers to GPU

// 3. Tokenize & forward step
int token = 1204;
int pos = 0;
float* logits = hk_forward_step(ctx, token, pos);

// 4. Clean up
hk_context_free(ctx);
hk_model_close(model);
```

---

## 4. Context Window Management

HK includes zero-copy context window management in native Zig (`src/context.zig`).

When a conversation exceeds the maximum token length:
- Traditional Python setups rebuild and reallocate the token buffer, which takes hundreds of microseconds.
- HK executes dynamic buffer truncation in native memory in **14.3 microseconds** on 65,000-token buffers (over 13x faster).
- The KV cache is maintained without thrashing host memory.
