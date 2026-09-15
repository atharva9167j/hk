# HK Neural Tensor Framework

[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Python%20Tests-82%2F82%20Passing-brightgreen.svg)](tests/)
[![Native Tests](https://img.shields.io/badge/Zig%20Tests-42%2F42%20Passing-brightgreen.svg)](tests/)
[![Platforms](https://img.shields.io/badge/Platforms-Android%20%7C%20iOS%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-purple.svg)](bindings/)

HK is a next-generation neural framework and unified binary container format (`.hk`) engineered for **continuous model evolution, end-to-end training, autonomous self-learning, dynamic architecture growth (Net2Net), and universal super-coalesced storage**. Built with a native Zig SIMD engine and zero-overhead C-ABI bindings across 7 programming languages, HK supersedes static legacy formats (SafeTensors, GGUF, and PyTorch checkpoints) with an active, living neural architecture that learns, widens, and adapts over time.

---

## The Four Pillars of the HK Framework

```
                    +-------------------------------------------------------+
                    |             HK NEURAL TENSOR FRAMEWORK                |
                    +-------------------------------------------------------+
                               /              |             \             \
                              /               |              \             \
                             v                v               v             v
       +-------------------------+  +-------------------+  +---------------+  +-------------------------+
       |   AUTONOMOUS GROWTH     |  |     TRAINING &    |  |  SELF-LEARNING|  |    SUPER-COALESCED      |
       |     & EVOLUTION         |  |    FINE-TUNING    |  |  & SELF-PLAY  |  |    STORAGE & DMA        |
       |-------------------------|  |-------------------|  |---------------|  |-------------------------|
       | • Net2WiderNet (SwiGLU) |  | • HKTrainer       |  | • SelfTrainer |  | • 0 Compute Headroom    |
       | • Net2DeeperNet Layers  |  | • Native QLoRA    |  | • <think> Mon.|  | • 4KB/16KB/128B Multi-Dev|
       | • Dynamic Vocab Growth  |  | • Loss-Plateau Exp|  | • SPIN Loss   |  | • Ampere 2:4 Sparsity   |
       | • Zig GrowthGovernor    |  | • Plasticity Mask |  | • Sandbox Eval|  | • In-Container DAG/Roll |
       +-------------------------+  +-------------------+  +---------------+  +-------------------------+
```

| Capability | Static Legacy Formats (SafeTensors / GGUF) | HK Autonomous Neural Framework |
| :--- | :--- | :--- |
| **Model Lifecycle** | Frozen, static file on disk | **Living architecture that grows, trains, and evolves** |
| **Capacity Expansion** | Impossible (requires retraining from scratch) | **Function-preserving Net2Net growth ($f_{\text{new}}(x) \equiv f_{\text{old}}(x)$)** |
| **Hardware Safety** | Manual trial-and-error OOM crashes | **Native Zig `GrowthGovernor` (50k constraints evaluated in 3.57 ms)** |
| **Training Integration**| External third-party scripts | **Native `HKTrainer` with automated plateau-triggered widening** |
| **Self-Learning Loop** | External synthetic data pipelines | **Closed-loop self-play (SPIN), inner monologue (`<think>`), and sandboxing** |
| **Catastrophic Forgetting**| Base representations get corrupted | **Plasticity Isolation gradient shielding (`protect_base_capacity`)** |
| **Version Lineage** | External Git LFS commits / full file duplication | **Cryptographic in-container Appendix DAG with instant rollback** |
| **Storage Architecture**| Unaligned JSON blobs or lossy quantizations | **Super-Coalesced raw weights: 4KB AMD/Intel + 16KB Apple + 128B NVIDIA** |
| **Hardware Sparsity** | None | **Native Ampere 2:4 structured sparsity (1.88× physical reduction, 0.000000 error)**|

> [!NOTE]
> **Active Research & Early Prototype Status**:  
> While the binary container format (`.hk`), zero-copy memory mapping, multi-device super-coalescing (128B/4KB/16KB), Ampere 2:4 structured sparsity, SIMD GEMV kernels, and low-level CLI tools are fully mature and production-ready (`v1.0.0`), the framework's higher-level evolutionary capabilities—including **`HKTrainer`**, the **`SelfTrainingPipeline`**, **`SelfPlayEngine` (SPIN)**, **dynamic Net2Net growth routines**, and the **in-container Appendix evolution DAG**—are **active research prototypes currently undergoing heavy development**. Their APIs, interfaces, and heuristics are experimental and actively evolving.

---

## Core Pillars & Highlights

### 1. Autonomous Dynamic Architecture Growth (Net2Net + GrowthGovernor)
- **Function-Preserving Net2WiderNet**: Dynamically widens feed-forward intermediate dimensions, attention projection layers, and SwiGLU MLPs (`net2wider_swiglu`) during runtime or fine-tuning while mathematically preserving exact outputs ($f_{\text{new}}(x) = f_{\text{old}}(x)$ on Day 0 with **$0.000000$ deviation**).
- **Function-Preserving Net2DeeperNet**: Inserts identity-initialized modular residual blocks (`ModularResidualBlock`) and adapter layers into transformer stacks, expanding representational depth without perturbing existing forward activations.
- **Dynamic Vocabulary Growth (`expand_vocab`, `hk_expand_vocab`)**: Expands token embeddings for domain-specific vocabularies or programming syntax on-the-fly with Gaussian initialization while preserving 100% of pre-existing token weights and IDs.
- **Native Zig Hardware `GrowthGovernor`**: Hardware resource manager implemented in compiled native Zig (`hk_governor_can_grow_batch`). Evaluates **50,000 capacity constraints in 3.57 ms** (2.2× faster than Python), strictly bounding parameter expansions against available physical VRAM and system memory to prevent out-of-memory crashes before allocation.

### 2. Full-Featured Training & Fine-Tuning Engine (`HKTrainer`)
- **Unified Training Loop (`HKTrainer`, `HKTrainingArguments`)**: Hugging Face-compatible training interface integrating AdamW optimization, cosine learning rate schedules, linear warmup, gradient clipping, and automated checkpointing.
- **Native QLoRA Adapter Fine-Tuning**: Low-rank adaptation (`enable_qlora`, `lora_rank`, `lora_alpha`) attached directly to quantized or unquantized projection matrices with zero external library dependencies.
- **Plateau-Triggered Dynamic Growth**: Continuously monitors validation loss stagnation; when learning plateaus beyond `growth_patience`, `HKTrainer` automatically invokes Net2Net widening to inject new capacity and break through convergence barriers.
- **Plasticity Isolation (`protect_base_capacity`)**: Anti-catastrophic forgetting gradient shielding masks (`hk_init_plasticity_mask`, row/col gradient masking) that freeze or dampen gradient flow to pre-existing base units while channeling learning updates exclusively into newly expanded neurons.
- **In-Container Checkpointing**: Stores training checkpoints, optimizer states, learning rates, and step metrics directly inside the `.hk` container.

### 3. Closed-Loop Autonomous Self-Learning & Self-Play Pipeline
- **End-to-End Self-Training Pipeline (`SelfTrainingPipeline`)**: Orchestrates closed-loop continuous model improvement without human supervision or manual labeling.
- **Bottleneck Diagnosis (`ExpansionEvaluator`)**: Probes representation deficits, domain error rates, and missing vocabulary tokens across task curricula to formulate precise architectural growth plans.
- **Inner Monologue Generation (`SelfConversationalEngine`)**: Employs dual-agent self-dialogue between Proposer and Thinker agents, generating structured reasoning traces (`<think> ... </think>`) before synthesizing candidate code or solutions.
- **Self-Play Fine-Tuning (`SelfPlayEngine` & `SPINLoss`)**: Leverages Self-Play Fine-Tuning (SPIN) objectives to contrast candidate responses against previous generations, iteratively self-correcting and bootstrapping reasoning depth.
- **Isolated Sandbox Verification (`CodeSandbox`, `SandboxExecutor`)**: Executes candidate implementations in a secure, sandboxed execution environment with strict memory/timeout limits. Only verifiably passing solutions provide positive reinforcement gradients.

### 4. Cryptographic In-Container Version Lineage & Instant Rollback (Appendix Region)
- **Embedded Version DAG (`header.appendix_offset`)**: Persistent append-only audit log residing at the end of the `.hk` file storing LoRA adapter deltas, training loss history, validation accuracy, sandbox pass rates, and growth receipts.
- **Cryptographic SHA-256 Parent Hash Chaining**: Every adaptation generation links cryptographically to its parent container state, guaranteeing tamper-proof audit trails.
- **Single-Command Instant Rollback (`hk rollback` / `hk_appendix_rollback`)**: Instantly rollback a model to any prior generation in sub-milliseconds without modifying or re-saving base weights.

### 5. Foundational Raw Weight Storage & Super-Coalesced Multi-Device Architecture
- **Zero-Compute Headroom Raw Storage**: Stores unquantized IEEE `FP32`, `FP16`, `BF16`, `FP8`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`, and `F64` weights as contiguous bit representations with **0 decoding, dequantization, or transcoding latency**. Compute kernels execute directly on mmap pointers via SIMD/Tensor Core vector instructions.
- **Universal Super-Coalescing Across Heterogeneous Silicon**:
  - **NVIDIA GPUs**: 128-byte alignment for warp-coalesced memory transactions and Tensor Core tile loads.
  - **AMD GPUs & CPUs**: ROCm DirectGMA and Zen cache-tiled 4096-byte (4 KB) page boundaries.
  - **Intel CPUs & NPUs**: OpenVINO Direct DMA, AVX-512 VNNI, and AMX-TILE 4096-byte boundaries.
  - **ARM & Apple Silicon**: macOS Metal Unified Memory `newBufferWithBytesNoCopy` 16384-byte (16 KB) boundaries & Linux ARM NEON/SVE.
  - **Super-Coalesced Invariance**: Because $4096 = 32 \times 128$ and $16384 = 128 \times 128$, a single shared `.hk` file aligned to 4KB or 16KB guarantees 100% strict NVIDIA Tensor Core coalescing while simultaneously providing native zero-copy DMA mapping for AMD, Intel, and Apple Silicon. **Zero storage duplication, zero compute headroom.**
- **Zero-Copy Page Cache Memory Mapping (`mmap`)**: Maps model checkpoints directly into user-space virtual memory (`mmap` / `MapViewOfFile`) with copy-on-write page safety. A 70B FP16 model maps in sub-milliseconds with zero Python heap allocation.
- **Minimal Container Overhead (<0.001%)**: Replaces multi-megabyte JSON header dictionaries with a fixed-size 128-byte binary file header and compact 128-byte Table of Contents (TOC) entries (<128 KB metadata overhead for a 1,000-tensor model).
- **Virtual Deduplication**:
  - `SHARED_REF`: Deduplicates tied embeddings (e.g., `lm_head.weight` == `embed_tokens.weight`) and recursive layer weights on disk, saving gigabytes of redundant full-precision storage.
  - `NULL_REF`: Represents fully pruned or zeroed layers with zero physical bytes stored in the container.

### 6. Split Mode Sharding & Regeneratable Modular Weights
- **Multi-File Sharding (`HeaderFlags.IS_SHARDED = 0x40`)**: Splits multi-hundred-gigabyte raw checkpoints cleanly across storage boundaries with standardized manifest indexes (`model.hk.index.json`), allowing lazy cross-shard slice access.
- **Regeneratable Weights & Dynamic Modularity**: Sharding works seamlessly in raw storage mode, allowing individual layers, LoRA adapters, and delta patches to be attached, detached, or regenerated dynamically without modifying base weights.

### 7. Lossless Structural Sparsity without Quantization
- **NVIDIA Ampere / Hopper / Blackwell 2:4 Structured Sparsity**: Stores 2 non-zero elements per 4-element block with physical 2-bit nibble metadata packing. Cuts physical on-disk storage and memory bandwidth by **50% (1.88× physical compression)** while preserving **exact 16-bit floating point precision** and dynamic range ($0.000000$ error vs dense baseline).
- **Compressed Sparse Representations**: Bitmask, Compressed Sparse Row (CSR), and Block Sparse Row (BSR) encoding zero out pruned connections without quantization distortion, streaming directly into native SIMD unpacking kernels at $>2.5\text{ GB/s}$.
- **Physical Channel Pruning**: Structurally drops unneeded attention heads or MLP channels from the tensor layout with explicit dimension mapping, reducing FLOPs and parameter footprint simultaneously.

### 8. Cache-Optimized 2D Tiling for Full-Precision GEMM/GEMV
- **Tensor-Core-Aligned Tiling (`TileLayout`)**: Reorganizes 2D/nD weight matrices into 16×16, 32×16, and 64×64 cacheline-aligned tiles with contiguous inner-K dimension packing to eliminate cache line thrashing and shared memory bank conflicts.

### 9. Microsecond In-Place Metadata Editing
- **Microsecond In-Place Metadata Patching (`hk metadata set`)**: Updates tokenizer configs, chat templates, hyperparameter tags, and licenses directly in the file header without re-serializing multi-gigabyte weight tensors.

### 10. Native Zig-First High-Performance Engine
- **Ultra-Lightweight Footprint**: The standalone native Zig binary (`hk.exe`) idles at just **2.80 MB of RAM** (>98% less memory than Python/PyTorch) and starts in **53 ms** (90.4× faster).
- **Zero-Copy Context Management**: Dynamic context window truncation executes in **14.3 µs** on 65k contiguous token buffers (13.4× faster than Python).
- **Hardware-Adaptive Inference**: Native CPUSpecs detection tailors vector paths to AVX2, AVX-512 VNNI, AMX-TILE, ARM NEON, or Apple Metal dynamically.

### 11. Comprehensive Quantization Suite (Complementary for Edge Deployments)
While HK is fundamentally designed for non-quantized storage and access performance, it provides a full suite of quantization schemes when low-bit compression is needed:
- **Dual-Mode Quantization**: Decouples compact base weights (NF4, DQ8, BitNet ternary) from residual delta streams, enabling runtime recovery of full floating-point fidelity ($>0.99999$ cosine similarity) without reloading weights.
- **Super-Block K-Quants ($Q2\_K$ through $Q8\_K$)**: 256-element super-blocks with sub-block scaling factors matching GGUF precision parity.
- **Vector I-Quants ($IQ1\_S$ through $IQ4\_NL$)**: Non-linear codebook quantization fitted to Gaussian weight distributions for minimal quantization noise.
- **Microscaling (OCP MXFP4 & Blackwell NVFP4)**: Hardware microscaling formats for next-generation silicon.
- **Activation-Aware Calibration**: Fisher second-moment importance matrix (`imatrix`) calibration with predefined recipes (`Q4_K_M`, `Q5_K_M`, etc.).

---

## Quick Start

### 1. Installation

Install the Python package directly (pre-compiled multiplatform native SIMD acceleration included):

```bash
pip install hk
```

Or install the Node.js / TypeScript package:

```bash
npm install hknt
```

Or compile from source using Zig:

```bash
# Build native release artifacts (CPU SIMD engine)
zig build -Doptimize=ReleaseFast

# Build with CUDA GPU backend (requires nvcc + CUDA toolkit)
zig build -Dcuda=true -Doptimize=ReleaseFast

# Run native test suite (42/42 native tests passing)
zig build test

# Run CUDA GPU backend correctness tests (requires -Dcuda=true)
zig build test-cuda -Dcuda=true
```

---

### 2. Training with `HKTrainer` (Experimental Prototype Preview)

> [!TIP]
> `HKTrainer` and the adaptive Net2Net training features are currently in early prototype preview and undergoing active research.

Train or fine-tune neural models using HK's unified trainer with automatic plateau-triggered Net2Net capacity growth:

```python
import torch
from hk import HKConfig, HKForCausalLM
from hk.trainer import HKTrainer, HKTrainingArguments

# 1. Initialize model
config = HKConfig(vocab_size=32000, hidden_size=1024, intermediate_size=2816, num_hidden_layers=12)
model = HKForCausalLM(config)

# 2. Configure training with QLoRA & adaptive Net2Net growth
args = HKTrainingArguments(
    output_dir="./checkpoints",
    learning_rate=3e-4,
    batch_size=4,
    num_train_epochs=3,
    # QLoRA parameter-efficient fine-tuning
    use_qlora=True,
    lora_rank=8,
    lora_alpha=16.0,
    # Adaptive capacity growth: widens layer dimensions when loss stagnates for 5 steps
    enable_adaptive_growth=True,
    growth_patience=5,
    growth_width_factor=1.25,
    # Hardware cacheline alignment
    alignment=128,
)

# 3. Launch training
trainer = HKTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# 4. Save evolved model with cryptographic Appendix lineage
model.save_pretrained("./checkpoints/evolved_model.hk")
```

---

### 3. Dynamic Architecture Growth (SwiGLU Net2WiderNet & Hardware GrowthGovernor)

Expand model capacity on-the-fly while strictly bounding memory against hardware VRAM:

```python
import torch
from hk.adaptive.growth import GrowthGovernor, net2wider_swiglu

# 1. Initialize hardware-bounded GrowthGovernor
# Uses native Zig engine to evaluate hardware capacity in microseconds
gov = GrowthGovernor(max_vram_mb=8192, max_growth_ratio=2.0)

current_params = 135_000_000
additional_params = 35_000_000

# 2. Validate proposed expansion against hardware limits
allowed, reason = gov.can_grow(current_params, additional_params, dtype_bytes=2)
print(f"Growth Approved: {allowed} ({reason})")

if allowed:
    # 3. Apply function-preserving SwiGLU Net2WiderNet expansion
    # Expands intermediate dimension from 1536 to 2048
    mlp = model.model.layers[0].mlp
    wider_gate, wider_up, wider_down = net2wider_swiglu(
        mlp.gate_proj,
        mlp.up_proj,
        mlp.down_proj,
        new_intermediate_size=2048,
        noise_std=0.0,  # 0.0 guarantees exact Day-0 mathematical function preservation
    )
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = wider_gate, wider_up, wider_down
    print("SwiGLU MLP successfully expanded with 0.000000 deviation from baseline!")
```

---

### 4. Closed-Loop Autonomous Self-Training Pipeline

Coordinate self-conversational reasoning, dynamic expansion, and sandboxed code verification:

```python
from hk.adaptive import (
    SelfTrainingPipeline,
    SelfTrainingCurriculum,
    SelfConversationalEngine,
    CodeSandbox,
    GrowthGovernor,
)

# 1. Define target curriculum with syntax and diagnostic tasks
curriculum = SelfTrainingCurriculum(
    domain_name="SystemsProgramming",
    target_language="Zig",
    syntax_keywords=["fn", "defer", "errdefer", "comptime", "pub", "anytype"],
    diagnostic_tasks=[...],
    training_tasks=[...],
)

# 2. Initialize pipeline with secure code sandbox
sandbox = CodeSandbox(timeout_sec=2.0)
engine = SelfConversationalEngine(sandbox=sandbox)
pipeline = SelfTrainingPipeline(
    model=model,
    hk_file_path="models/minizig_model.hk",
    governor=GrowthGovernor(max_vram_mb=6144),
    conversational_engine=engine,
    rehearsal_ratio=0.10,  # Anti-catastrophic forgetting rehearsal
)

# 3. Run autonomous self-training generation
report = pipeline.train_generation(curriculum, generation_idx=1)
print(f"Gen 1 Complete: Pass Rate={report.pass_rate*100:.1f}%, Expansion={report.expansion_occurred}")
```

---

### 5. In-Container Version Lineage & Cryptographic Rollback

Manage version DAGs directly inside `.hk` files without duplicating base weights:

```python
from hk.adaptive.appendix import AppendixManager

# 1. Inspect version lineage
manager = AppendixManager("model.hk")
records = manager.list_records()
for r in records:
    print(f"Generation {r.generation} [{r.name}]: Loss={r.metrics.loss:.4f}, PassRate={r.metrics.pass_rate*100:.1f}%")

# 2. Rollback to Generation 1 instantly
success = manager.rollback_to_generation(target_generation=1)
print(f"Rollback successful: {success}")
```

---

### 6. Raw Weight Storage & Instant Zero-Copy Inference

Store raw unquantized weights with zero compute headroom and super-coalesced alignment:

```python
import torch
import hk

# 1. Save unquantized IEEE raw weights with universal super-coalescing
weights = {
    "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
    "model.layers.0.mlp.up_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
}
hk.save_raw(weights, "model_raw.hk", universal_alignment=True)

# 2. Instant zero-copy memory-mapped store
store = hk.load_raw("model_raw.hk")
print(f"Universal Page Aligned (4KB): {store.is_universal_page_aligned}")
print(f"Tensor Core Aligned (128B) : {store.is_tensor_core_aligned}")

# 3. Direct SIMD Linear Algebra (zero prior memory allocation)
x = torch.randn(4096, dtype=torch.float32)
y = store.gemv("model.layers.0.mlp.gate_proj.weight", x)
```

---

## Native Standalone CLI & Developer Tools

The compiled native `hk` CLI provides sub-millisecond execution with minimal memory usage:

```bash
# 1. Profile host architecture (AMD Zen, Intel AVX-512/AMX, Apple Silicon, ARM NEON/SVE)
hk hardware-profile

# 2. Inspect container header, alignment audit, and non-quantized tensor layouts
hk inspect model.hk

# 3. Verify 128-byte Tensor Core alignment, checksums, and container integrity
hk verify model.hk

# 4. In-place metadata patching without rewriting multi-gigabyte weight arrays
hk metadata set model.hk general.version "1.0.0"
hk metadata set model.hk tokenizer.chat_template "{% for message in messages %}..."

# 5. Native vocabulary and SwiGLU MLP width expansion (Net2WiderNet)
hk expand in.hk out.hk --vocab 32000 --width 1.33

# 6. Physical memory traversal, page fault profiling & resident dequantization
hk benchmark model.hk

# 7. List in-container Appendix version lineage records
hk appendix list model.hk

# 8. Cryptographic generational rollback to previous model state
hk rollback model.hk --generation 1

# 9. Lossless GGUF / SafeTensors transcoding
hk convert-gguf model.gguf model.hk
hk export -f safetensors model.hk model.safetensors
hk export -f gguf model.hk exported.gguf

# 10. Launch visual HK Model Studio GUI
hk gui model.hk
# Or via Python:
hk-gui model.hk

# 11. Native autoregressive inference with dynamic CPU/GPU offloading (-ngl <layers>)
hk run model.hk -p "Explain quantum computing simply" -n 128 -ngl 32

# 12. Interactive terminal chat REPL with dynamic offloading
hk chat model.hk -ngl 16 --temp 0.7

# 13. Cryptographic hash verification (streaming SHA-256)
hk hash model.hk
```

---

## Architectural Comparison

| Feature | SafeTensors | GGUF | HK Neural Framework |
|:---|:---|:---|:---|
| **Primary Paradigm** | Static weight serialization | Static quantized inference | **Living neural framework: Training, Dynamic Growth, Self-Play & Storage** |
| **Unified Training Engine** | None (external PyTorch/HF) | None | **Native `HKTrainer` with automated plateau-triggered Net2Net expansion** |
| **Dynamic Architecture Growth** | Impossible | Impossible | **Live Net2WiderNet (Linear & SwiGLU), Net2DeeperNet, Dynamic Vocab Growth** |
| **Hardware Growth Safety** | None (OOM crashes) | None | **Native Zig `GrowthGovernor` (50k constraints evaluated in 3.57 ms)** |
| **Autonomous Self-Learning** | None | None | **Closed-loop `SelfTrainingPipeline`, SPIN loss, `<think>` inner monologue** |
| **Execution Sandboxing** | None | None | **Integrated `CodeSandbox` for automated test-driven reward verification** |
| **Catastrophic Forgetting Defense** | None | None | **Plasticity Isolation gradient shielding masks (`protect_base_capacity`)** |
| **Version Lineage & Provenance** | None (requires external Git LFS) | None | **In-container Appendix DAG with SHA-256 parent hash chaining & rollback** |
| **Raw Unquantized Weights** | Basic unaligned byte blobs | Inefficient fallback | **First-class IEEE/INT raw storage with 0 compute headroom (no decode penalty)** |
| **Multi-Device DMA Zero-Copy** | None | 32-byte CPU alignment | **Super-Coalesced: 4KB AMD/Intel + 16KB Apple Metal + 128B NVIDIA in one file** |
| **Structured Hardware Sparsity** | No support | No support | **Native Ampere 2:4 (nibble indices, 1.88× physical reduction, 0.000000 error)** |
| **2D Tensor Tiling** | Row-major only | CPU strided only | **Tensor-Core-aligned K-contiguous tiles (16×16, 32×16, 64×64 WMMA)** |
| **In-Place Metadata Updates** | Requires full file rewrite | Requires container rewrite | **Microsecond in-place header patching (`hk metadata set`)** |
| **Quantization Coverage** | None (dense floats only) | Lossy K/I-quants only | **Comprehensive: Dual-Mode (NF4 + residual), K-Quants, I-Quants, MXFP4, NVFP4** |
| **Process Idle RAM** | ~197 MB (PyTorch runtime) | Variable C++ | **2.80 MB (Standalone native Zig `hk.exe`)** |
| **Startup Latency** | ~4,880 ms (Python runtime) | ~120 ms | **53.98 ms (Standalone native Zig `hk.exe`)** |

---

## Empirical Benchmarks

### 1. ~1 Billion Parameter Production Model Benchmark (`Qwen3.5-0.8B`)
Evaluated against a real production non-quantized model with **873,438,784 bfloat16 parameters** (488 tensors, 1.75 GB) comparing standard **Hugging Face / PyTorch / Safetensors** workflows against the **HK Tensor-Optimized Raw Weight Engine** (`qwen_0.8b_optimized.hk` with 4096-byte Universal Page Alignment + 128-byte NVIDIA Tensor Core Coalescing):

| Benchmark Metric | Standard Hugging Face / PyTorch | HK Tensor-Optimized Engine | Speedup / Advantage |
| :--- | :--- | :--- | :--- |
| **Layer GEMV Compute ($y = W \cdot x$)** | 0.37 ms (20.04 GFLOPS) | **0.21 ms (34.38 GFLOPS)** | **1.72× FASTER** *(+71.6% compute throughput)* |
| **Autoregressive Layer Retrieval (Warm)** | 28.84 $\mu$s | **0.08 $\mu$s (80 ns)** | **346.10× FASTER** *(Sub-Microsecond)* |
| **Autoregressive Layer Retrieval (Cold)** | 506.23 $\mu$s | **136.07 $\mu$s** | **3.72× FASTER** |
| **Full Model Weight Load (1.75 GB, 488 tensors)** | 10.30 ms | **15.88 ms** | Pure zero-copy OS memory mapping (`HKDict`) |
| **Safetensors $\to$ HK Transcoding Speed** | N/A | **222.23 MB/s** | Streaming zero-memory generator |

- **4-Row Unrolled SIMD GEMV**: `HKRawWeightStore.gemv` evaluates 4 output rows simultaneously in registers with 8 interleaved 256-bit SIMD accumulators, eliminating cache thrashing and achieving **34.38 GFLOPS** on a single CPU core.
- **Sub-Microsecond Layer Access**: Once mapped, layer retrieval executes in **80 nanoseconds**, delivering virtually instantaneous weight feed to compute cores during token-by-token generation.
- **Reproducibility**: Run this exact benchmark locally on any machine with `python benchmarks/benchmark_1b_model.py`.

---

### 2. Memory-Mapped Loading & Physical Page-In vs. Resident Dequantization
Transparently isolating virtual memory descriptor mapping, physical storage paging, and pure in-memory SIMD compute:

| Loader / Compute Phase | Metric Reported | Measured Latency / Rate | Physical Significance |
| :--- | :--- | :--- | :--- |
| **Zero-Copy Header & Deserialization** | Metadata & Virtual Map | **15.93 ms** (1.75 GB model) | Maps file into user address space (`mmap`/`MapViewOfFile`). |
| **Lazy Slice Pointer Resolution** | Descriptor Lookup | **23.70 $\mu$s** (20.6 M-slices/s) | Resolves 488 tensor descriptors in user space (0 heap allocations). |
| **Physical Memory Traversal (First Touch)** | NVMe / Storage Throughput | **2,422 ms** (687.8 MB/s) | Actually reads mapped pages via 256-bit SIMD, faulting them into RAM. |
| **Physical Memory Traversal (Warm Cache)**| System Memory Bus Rate | **1,409 ms** (1,182.3 MB/s) | Pure memory bus streaming throughput when pages are resident in RAM. |
| **Reconstruction & Dequantization** | SIMD Compute Throughput | **213.25 M-elem/s** (970 MB scratch)| Executes on warm resident memory using single-tensor scratch buffers. |

> [!NOTE]
> **Benchmarking Rigor**: HK explicitly distinguishes between **lazy pointer slice resolution** (sub-microsecond virtual address setup where memory is not yet paged in) and **physical first-touch traversal** (where SIMD reads force OS page faults from disk into RAM). Furthermore, HK's dequantization benchmarks use a reusable per-tensor scratch buffer (sized to the largest single tensor, e.g. 50–970 MB) rather than eagerly allocating multi-gigabyte monolithic arrays, eliminating artificial OS memory zeroing and swap thrashing.

---

### 3. Autonomous Growth Governor & Adaptive Decision Latency
Evaluating hardware boundary safety constraints across 50,000 proposed layer/width capacity expansions:

| Evaluation Engine | Operations Evaluated | Total Latency | Rate | Relative Performance |
| :--- | :---: | :---: | :---: | :---: |
| **Python Standard Logic** | 50,000 checks | 7.89 ms | 6.34 M-evals/sec | 1.00x |
| **HK Native Zig Vectorized Governor** | 50,000 checks | **3.57 ms** | **14.01 M-evals/sec** | **2.21× faster** |

- **Zero OOM Tolerance**: The native growth governor evaluates candidate expansions in **71 nanoseconds per decision**, allowing automated training loops to continuously probe expansion feasibility without introducing overhead.

---

### 4. Non-Quantized Loading & Memory-Mapped Slicing (HK vs SafeTensors)
Evaluated across 32 dense full-precision FP32 tensors (224.00 MB total container size) with 50 iterations per operation:

| Reader / Loader Access Pattern | Container Size | Operation | Latency (ms) | Relative Performance |
|:---|:---:|:---|:---:|:---:|
| **SafeTensors (`load_file`)** | 224.00 MB | Full Load (32 tensors) | 1.473 ms | 1.00x |
| **HK Native (`load_file`)** | 224.00 MB | Full Load Zero-Copy | **0.797 ms** | **1.85× faster** |
| **SafeTensors (`get_slice` direct)** | 224.00 MB | Direct 2D Slice [128, 512] | 0.0149 ms | 1.00x |
| **HK Native (`get_slice` direct)** | 224.00 MB | Direct 2D Slice [128, 512] | **0.0092 ms** | **1.62× faster** |
| **SafeTensors (`safe_open` lifecycle)** | 224.00 MB | Open + Slice Context | 0.332 ms | 1.00x |
| **HK Native (`safe_open` lifecycle)** | 224.00 MB | Open + Slice Context | **0.274 ms** | **1.21× faster** |

---

### 5. Lossless NVIDIA Ampere 2:4 Structured Sparsity
Bit-exact hardware physical nibble compression without numerical precision loss:

| Operation | Input Size | Output Size | Compression | Latency (ms) | Throughput (MB/s) | Max Absolute Error |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Ampere 2:4 Pack** | 64.00 MB | 34.00 MB | **1.88x** | 89.90 ms | 711.9 MB/s | **0.000000 (Exact Lossless)** |
| **Ampere 2:4 Unpack** | 34.00 MB | 64.00 MB | 1.00x | **24.87 ms** | **2,573.0 MB/s** | **0.000000 (Exact Lossless)** |

---

### 6. Native Zig CLI & Process Footprint
Comparing process initialization and idle memory between Python/PyTorch and the native `hk.exe` binary:

| Metric | Python Runtime (Torch + HK) | Standalone Zig Binary (`hk.exe`) | Improvement / Delta |
|:---|:---:|:---:|:---:|
| **Idle Process RAM (RSS)** | 197.44 MB | **2.80 MB** | **98.6% less RAM** |
| **Process Startup Latency** | 4,880.72 ms | **53.98 ms** | **90.4× faster startup** |
| **HF Tensor Name Mapping Rate** | 12,400 names/sec | **320,944 names/sec** | **25.8× faster** |
| **65k Context Truncation** | 191.82 µs | **14.30 µs** | **13.4× faster** |

---

### 7. Quantization Benchmarks (Complementary Edge Schemes)
Evaluated on a 2048 x 2048 matrix (4,194,304 f32 elements / 16.00 MB) with AVX2 SIMD acceleration:

| Quant Format | Packed Size | Compression | Quant Latency | Quant Throughput | Dequant Latency | Dequant Throughput | Cosine Sim |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Q8_0 (8-bit Symmetric)** | 4.25 MB | 3.76x | 6.4 ms | 2,489.8 MB/s | **2.4 ms** | **6,688.5 MB/s** | 0.999985 |
| **Q4_0 (4-bit Symmetric)** | 2.25 MB | 7.11x | 6.7 ms | 2,372.8 MB/s | **3.4 ms** | **4,667.7 MB/s** | 0.996318 |
| **Q8_K (8-bit K-Quant)** | 4.56 MB | 3.51x | 5.9 ms | 2,718.0 MB/s | **4.7 ms** | **3,396.6 MB/s** | 0.999975 |
| **Q4_K (4-bit K-Quant)** | 2.25 MB | 7.11x | 5.9 ms | 2,705.7 MB/s | **2.9 ms** | **5,535.6 MB/s** | 0.994132 |
| **Dual-Mode NF4 + Residual** | 97.09 MB (SmollM) | 6.40x | 245.0 ms | — | — | **0.999994** |

---

## Multilingual SDK Ecosystem

HK provides native, zero-overhead bindings across all major environments:

- **Python**: [`python/hk/`](python/hk/) - Unified framework with PyTorch, NumPy, JAX, Flax, `HKTrainer`, and `SelfTrainingPipeline`.
- **Rust**: [`bindings/rust/`](bindings/rust/) - Safe idiomatic Rust crate (`Cargo.toml`).
- **TypeScript / Node.js**: [`bindings/js/`](bindings/js/) - Zero-dependency browser, Node.js, and WASM SDK (`@hk-format/core`).
- **C# / .NET**: [`bindings/csharp/`](bindings/csharp/) - Modern .NET package for Unity, Windows, and enterprise systems (`Hk.csproj`).
- **Go**: [`bindings/go/`](bindings/go/) - Idiomatic Go module using cgo (`go.mod`).
- **Java / Android**: [`bindings/java/`](bindings/java/) - JNI package with direct NIO `ByteBuffer` zero-copy support (`pom.xml`).
- **C / C++**: [`include/hk.h`](include/hk.h), [`include/hk.hpp`](include/hk.hpp) - C-ABI and C++20 RAII headers.

---

## License

HK is open source software licensed under the [Apache License, Version 2.0](LICENSE).

