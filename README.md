# HK Neural Tensor Framework

[![Version](https://img.shields.io/badge/Version-1.1.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Android-purple.svg)](bindings/)

I built HK as a high-performance neural framework and unified binary container format (`.hk`) for continuous model evolution, dynamic architecture growth, fast local training, and zero-overhead storage across every type of hardware.

Instead of treating neural networks as frozen, immutable files on disk, I created HK to provide a living architecture that can widen its layers, learn new domains, and adapt over time without catastrophic forgetting.

I have written comprehensive documentation and deep dives in the [HK GitHub Wiki](https://github.com/harshitkhandelwal208/hk/wiki).

---

## Why I Built HK

I started building HK out of sheer frustration late one night sitting with my laptop.

I wanted to experiment with modern open-weight language models locally. I had an ordinary machine: 16 GB of system RAM and a modest laptop GPU. I was genuinely excited to poke around under the hood, try fine-tuning a small model on some custom code, and see how far I could push it.

Instead, reality hit like a brick wall.

First came the local inference experience. Just importing packages took what felt like forever. Loading a modest 1B or 3B model into memory dragged my entire laptop to a crawl. The fans spun up to maximum speed, the cursor started stuttering, and before I could even run a single token forward pass, the process was killed with an out-of-memory error. Standard runtimes were allocating gigabytes of RAM just for Python runtime wrappers and unaligned byte buffers.

Then came training, and that was an absolute nightmare.

If you have ever tried fine-tuning or training models on consumer hardware, you know the pain:
- Doing Full Fine-Tuning (FFT) where you update all parameters across the network was practically impossible locally because standard PyTorch training state (weights, gradients, Adam first and second moments) easily eats up 16 to 24 bytes per parameter. An 8B model suddenly demands 80+ GB of VRAM.
- Even trying parameter-efficient methods like LoRA meant installing a fragile house of cards: specific PyTorch builds, matching CUDA toolkits, bitsandbytes DLLs that frequently crashed on Windows, and conflicting dependencies.
- Worst of all was the rigidity of the architectures themselves. The moment pre-training finishes, traditional models are frozen in stone. Layer widths are fixed. Vocabularies are locked. If you try to teach an existing model a new programming language or new domain vocabulary, it struggles because it does not have the capacity to absorb new concepts. And if you force the weights to update through standard fine-tuning, the model suffers from catastrophic forgetting, corrupting its earlier knowledge.

I remember staring at my screen thinking: why does adding a few new capabilities to a 1B model require throwing the whole thing away and retraining from scratch on an expensive GPU cluster?

When you look at biological brains, nothing works like that. When you learned to code, or learned a second language, you did not erase your childhood memories. You did not have to reboot your brain from scratch. Biological brains are dynamic. They develop through continuous neurogenesis, widen neural pathways, strengthen synapses, and adapt over time while keeping earlier memories intact.

When I started analyzing traditional formats like SafeTensors and GGUF, I saw so many fundamental things that could be fixed:
- They treat neural networks as static, dead files on disk rather than living architectures.
- Their memory layouts force an artificial choice between unaligned bloated floats and lossy low-bit quantizations that degrade reasoning.
- Memory reads do not align with how hardware memory controllers, DMA channels, and GPU warps actually read data.
- Checkpoint history is outsourced to external Git LFS repos or duplicate multi-gigabyte folders.

I decided to build HK to test a different path:
1. Neural networks should be dynamic like the human brain, capable of growing their layers (Net2Net), expanding their vocabulary, and learning continuously without catastrophic forgetting.
2. The framework must work seamlessly on whatever hardware you have. Whether you are on a regular laptop CPU, a consumer GPU, an Apple Silicon Mac, or a multi-GPU server, I built HK to run reliably, cleanly, and faster than traditional tools.

---

## Transitioning to HK from Your Current Setup

You do not need to throw away your existing checkpoints, datasets, or codebases. I built everything in HK with clean substitution paths so you can transition immediately.

### 1. Replacing Ollama, LM Studio, and llama.cpp (Local Inference)
If you run models locally using GGUF files in Ollama, LM Studio, or llama.cpp, you can convert them directly to HK format in a single streaming pass:

```bash
# Convert existing GGUF model to HK container
hk convert-gguf model.gguf model.hk

# Run fast command-line inference (starts in under 55 ms, uses under 3 MB idle RAM)
hk run model.hk -p "Explain how an operating system kernel works" -n 128

# Interactive chat session in your terminal
hk chat model.hk --temp 0.7

# Offload specific number of layers to GPU (keeps the rest on CPU SIMD)
hk chat model.hk -ngl 16
```

### 2. Replacing Hugging Face Transformers & SafeTensors
I designed HK to provide direct drop-in replacements for `safetensors.torch` and `AutoModelForCausalLM`:

```python
# Drop-in replacement for safetensors.torch
from hk.torch import save_file, load_file, safe_open

# Save weights with automatic tied-weight deduplication
save_file(tensors_dict, "model.hk")

# Load weights with zero-copy memory mapping
tensors = load_file("model.hk")

# Lazy inspection and multidimensional tensor slicing without loading full matrix
with safe_open("model.hk", framework="pt") as f:
    slice_data = f.get_slice("model.layers.0.mlp.gate_proj.weight")[0:64, :]
```

And for high-level model loading:

```python
from hk import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model.hk", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("model.hk")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))
```

### 3. Training Compatibility: Full Fine-Tuning (FFT), QLoRA, SFT, and CPT
Whether you do Full Fine-Tuning (FFT) across all parameters or parameter-efficient training, I built `HKTrainer` to support all standard training procedures with drop-in compatibility:

```python
from hk import HKConfig, HKForCausalLM
from hk.trainer import HKTrainer, HKTrainingArguments

config = HKConfig.from_pretrained("model.hk")
model = HKForCausalLM.from_pretrained("model.hk", config=config)

args = HKTrainingArguments(
    output_dir="./checkpoints",
    learning_rate=2e-4,
    batch_size=4,
    num_train_epochs=3,
    
    # 1. Full Fine-Tuning (FFT) Mode:
    # Set use_qlora=False to train 100% of parameters with full gradient backpropagation
    use_qlora=False,
    
    # 2. Or Parameter-Efficient QLoRA Mode:
    # Set use_qlora=True to train low-rank adapters with zero external library dependencies
    # use_qlora=True,
    # lora_rank=8,
    # lora_alpha=16.0,
    
    # 3. Dynamic Capacity Growth:
    # When loss plateaus, automatically widen intermediate SwiGLU layers (Net2Net)
    enable_adaptive_growth=True,
    growth_patience=5,
    growth_width_factor=1.25,
    protect_base_capacity=True, # Plasticity isolation prevents catastrophic forgetting
)

trainer = HKTrainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()

# Saves evolved model with in-container checkpoint history
model.save_pretrained("./checkpoints/evolved_model.hk")
```

### 4. Replacing PyTorch `.pt` Checkpoints
```python
import hk.torch as hkt

# Save state dict safely without Python pickle vulnerabilities
hkt.save_model(model, "model.hk")

# Load state dict back with strict parameter checking
hkt.load_model(model, "model.hk", strict=True)
```

---

## Universal Silicon Invariance: How It Runs Fast on Any Chip

A foundational goal when I built HK was to ensure a single file runs at peak efficiency across completely different chip architectures without duplicate files or conversion steps:

- NVIDIA GPUs: Require 128-byte alignment so 32-thread GPU warps can fetch memory in a single bus transaction.
- AMD GPUs and Intel CPUs: Use 4096-byte (4 KB) page boundaries for direct memory mapping and DMA transfers.
- Apple Silicon (M1/M2/M3/M4): Uses 16384-byte (16 KB) pages for zero-copy Metal Unified Memory mapping (`newBufferWithBytesNoCopy`).

Because 4096 is divisible by 128 (32 x 128) and 16384 is divisible by 128 (128 x 128), aligning weights to 4KB or 16KB automatically satisfies NVIDIA 128-byte warp coalescing.

A single `.hk` container provides zero-copy mapping on an Apple Silicon MacBook, native DMA on AMD/Intel, and warp coalescing on NVIDIA. Zero conversion steps, zero wasted RAM, zero compute overhead.

---

## Quick Start

### Installation

Install the Python package via pip:
```bash
pip install hknt
```

Or install the Node.js / TypeScript SDK:
```bash
npm install hknt
```

Or compile the standalone native engine and CLI from source using Zig (0.13+):
```bash
# Build native binary
zig build -Doptimize=ReleaseFast

# Run native unit test suite (49/49 tests passing)
zig build test
```

---

## Core Capabilities at a Glance

### 1. Dynamic Architecture Growth (Net2Net)
- Net2WiderNet: Widens feed-forward intermediate layers, attention projections, and SwiGLU MLPs during runtime. On Day 0, the output is mathematically identical to the unexpanded model ($f_{\text{new}}(x) \equiv f_{\text{old}}(x)$ with 0.000000 deviation).
- Net2DeeperNet: Inserts modular identity residual blocks to increase depth without disrupting earlier layers.
- Dynamic Vocabulary Expansion: Add new domain keywords or syntax tokens to the tokenizer and embedding table on the fly without breaking pre-existing token weights.
- Native GrowthGovernor: A native Zig hardware manager evaluates physical RAM and VRAM constraints in microseconds (50,000 checks in 3.57 ms), preventing out-of-memory crashes before allocation.
- Plasticity Isolation: Generates gradient masks that protect established base weights and channel learning updates into newly expanded neurons.

### 2. Comprehensive Training Support: FFT, QLoRA, SFT, and CPT
- Full Fine-Tuning (FFT): Train 100% of parameters with full gradient backprop, AdamW, and linear warmup schedules.
- Native QLoRA: Fine-tune low-rank adapters directly attached to projection matrices with zero external dependencies.
- Supervised Fine-Tuning (SFT): Instruction tuning with embedded Jinja2 chat templates and prompt loss masking.
- Continued Pre-Training (CPT): Domain adaptation with dynamic vocabulary expansion and Net2Net capacity injection.
- Plateau-Triggered Widening: Automatically widens intermediate dimensions when validation loss stagnates.

### 3. Autonomous Self-Training & Self-Play
- Closed-Loop Pipeline: Probes model weaknesses using `ExpansionEvaluator`.
- Inner Monologue: Generates structured `<think> ... </think>` reasoning traces before emitting candidate code.
- Self-Play Fine-Tuning: Uses SPIN loss to iteratively bootstrap depth by contrasting new generations against past answers.
- Secure Sandboxed Execution: Evaluates candidate code solutions in an AST-validated, process-isolated sandbox with strict timeouts and memory caps.

### 4. In-Container Version Lineage & Instant Rollback
- The Appendix DAG: An append-only audit trail residing at the end of the `.hk` file tracks LoRA adapter deltas, training loss history, validation accuracy, and growth receipts.
- Cryptographic Lineage: Every generation links to its parent container via SHA-256 hash chaining, guaranteeing tamper-proof history.
- Instant Rollback: Restore a model to any earlier generation in under a millisecond with `hk rollback model.hk --generation 1` without touching base weights.

### 5. Zero Compute Headroom Raw Storage
- Stores full-precision and half-precision floats (`bfloat16`, `float16`, `float32`, `int8`) as contiguous bit representations.
- Zero decoding or dequantization overhead during inference. Compute kernels execute directly on memory-mapped pointers.
- Virtual Deduplication: `SHARED_REF` virtual aliases deduplicate tied embeddings (such as `lm_head.weight == embed_tokens.weight`) on disk, saving hundreds of megabytes. `NULL_REF` stores pruned layers with zero physical bytes.

### 6. Lossless Hardware Structured Sparsity
- Native Ampere 2:4 structured sparsity with compact 2-bit nibble indexing.
- Cuts physical storage and memory bandwidth by 50% (1.88x physical compression).
- Non-zero values remain full 16-bit floats, providing exact 0.000000 numerical deviation compared to the dense baseline.

### 7. Microsecond In-Place Metadata Updates
- Pre-allocated elastic header padding lets you update chat templates, licenses, and version strings in place in under 5 milliseconds with `hk metadata set` without rewriting gigabytes of weights.

---

## Standalone Native CLI Tools

I compiled the standalone `hk` binary to provide instant command-line workflows:

```bash
# 1. Profile system CPU extensions and page alignment
hk hardware-profile

# 2. Inspect container header, TOC, and tensor layouts
hk inspect model.hk

# 3. Verify physical alignment, checksums, and container integrity
hk verify model.hk

# 4. In-place metadata editing
hk metadata set model.hk general.version "1.1.0"
hk metadata set model.hk tokenizer.chat_template "{% for msg in messages %}..."

# 5. Expand model capacity (widen layers and vocabulary)
hk expand in.hk out.hk --width 1.25 --vocab 32500

# 6. Measure memory traversal and SIMD compute throughput
hk benchmark model.hk

# 7. List in-container version history records
hk appendix list model.hk

# 8. Instantly rollback container to previous generation
hk rollback model.hk --generation 1

# 9. Lossless GGUF and SafeTensors transcoding
hk convert-gguf model.gguf model.hk
hk export -f safetensors model.hk model.safetensors
hk export -f gguf model.hk exported.gguf

# 10. Fast command-line inference and interactive chat REPL
hk run model.hk -p "Explain gravity simply" -n 128 -ngl 16
hk chat model.hk -ngl 16 --temp 0.7

# 11. Launch visual Model Studio GUI
hk gui model.hk
```

---

## Performance Highlights

I evaluated HK on a production model with 873,438,784 bfloat16 parameters (488 tensors, 1.75 GB) comparing standard Hugging Face / PyTorch against my native HK Tensor-Optimized Engine:

| Metric | Standard PyTorch / Hugging Face | HK Tensor Engine | Advantage |
| :--- | :--- | :--- | :--- |
| **Layer GEMV ($y = W \cdot x$)** | 0.37 ms (20.04 GFLOPS) | **0.21 ms (34.38 GFLOPS)** | **1.72x faster** (+71.6% throughput) |
| **Warm Layer Retrieval** | 28.84 microseconds | **0.08 microseconds (80 ns)** | **346.1x faster** (sub-microsecond) |
| **Cold Layer Retrieval** | 506.23 microseconds | **136.07 microseconds** | **3.72x faster** |
| **Growth Governor Decision** | 7.89 ms (Python) | **3.57 ms (Native Zig)** | **2.21x faster** (50,000 checks) |
| **CLI Idle Memory Footprint** | 197.4 MB (Python runtime) | **2.80 MB (Native `hk`)** | **98.6% less memory** |
| **Process Startup Latency** | 4,880 ms (Python runtime) | **53.9 ms (Native `hk`)** | **90.4x faster startup** |
| **Ampere 2:4 Structured Sparsity**| None (1.00x) | **1.88x physical compression** | **0.000000 error (bit-exact)** |

For full empirical benchmarks, physical page fault profiling, memory bus measurements, and quantization tradeoffs, see the [Benchmarks and Performance Guide](https://github.com/harshitkhandelwal208/hk/wiki/Benchmarks-and-Performance).

---

## Multi-Language SDK Ecosystem

I built native, zero-overhead bindings across 7 programming environments:

- **Python**: [`python/hk/`](python/hk/) - Full framework with PyTorch, NumPy, JAX, Flax, `HKTrainer`, and `SelfTrainingPipeline`.
- **Rust**: [`bindings/rust/`](bindings/rust/) - Safe idiomatic Rust crate with compile-time lifetime checks.
- **TypeScript / Node.js**: [`bindings/js/`](bindings/js/) - Zero-dependency browser, Node.js, and WebAssembly SDK (`hknt`).
- **C# / .NET 9**: [`bindings/csharp/`](bindings/csharp/) - Modern .NET package for Unity, Windows desktop, and enterprise servers.
- **Go**: [`bindings/go/`](bindings/go/) - Idiomatic Go module using cgo.
- **Java / Android**: [`bindings/java/`](bindings/java/) - JNI package with direct NIO `ByteBuffer` zero-copy buffer views.
- **C / C++**: [`include/hk.h`](include/hk.h), [`include/hk.hpp`](include/hk.hpp) - C99 ABI and C++20 RAII headers.

---

## Documentation & Wiki

I have written detailed guides in the [HK GitHub Wiki](https://github.com/harshitkhandelwal208/hk/wiki):

- [Wiki Home](https://github.com/harshitkhandelwal208/hk/wiki)
- [Getting Started Guide](https://github.com/harshitkhandelwal208/hk/wiki/Getting-Started)
- [Transition Guide](https://github.com/harshitkhandelwal208/hk/wiki/Transition-Guide)
- [Architecture and Ideology](https://github.com/harshitkhandelwal208/hk/wiki/Architecture-and-Ideology)
- [Format Specification](https://github.com/harshitkhandelwal208/hk/wiki/Format-Specification)
- [Raw Storage and Super-Coalescing](https://github.com/harshitkhandelwal208/hk/wiki/Raw-Storage-and-Super-Coalescing)
- [Dynamic Architecture Growth](https://github.com/harshitkhandelwal208/hk/wiki/Dynamic-Architecture-Growth)
- [Training and Fine-Tuning: FFT, QLoRA, SFT](https://github.com/harshitkhandelwal208/hk/wiki/Training-and-Fine-Tuning)
- [Autonomous Self-Training](https://github.com/harshitkhandelwal208/hk/wiki/Autonomous-Self-Training)
- [In-Container Version Lineage](https://github.com/harshitkhandelwal208/hk/wiki/In-Container-Version-Lineage)
- [Inference and Serving](https://github.com/harshitkhandelwal208/hk/wiki/Inference-and-Serving)
- [Storage Innovations and Sparsity](https://github.com/harshitkhandelwal208/hk/wiki/Storage-and-Sparsity)
- [CLI Reference Manual](https://github.com/harshitkhandelwal208/hk/wiki/CLI-Reference)
- [Python API Reference](https://github.com/harshitkhandelwal208/hk/wiki/Python-API-Reference)
- [Multi-Language SDKs](https://github.com/harshitkhandelwal208/hk/wiki/Multi-Language-SDKs)
- [Empirical Benchmarks](https://github.com/harshitkhandelwal208/hk/wiki/Benchmarks-and-Performance)
- [FAQ and Troubleshooting](https://github.com/harshitkhandelwal208/hk/wiki/FAQ-and-Troubleshooting)

---

## License

I have licensed HK as open-source software under the [Apache License, Version 2.0](LICENSE).

