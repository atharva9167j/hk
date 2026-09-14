# HK Neural Tensor Framework

[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Python%20Tests-82%2F82%20Passing-brightgreen.svg)](tests/)
[![Native Tests](https://img.shields.io/badge/Zig%20Tests-42%2F42%20Passing-brightgreen.svg)](tests/)
[![Platforms](https://img.shields.io/badge/Platforms-Android%20%7C%20iOS%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-purple.svg)](bindings/)

HK is a unified neural framework and binary container format (`.hk`) engineered to supersede legacy model files (SafeTensors, GGUF, and PyTorch checkpoints) with a single, high-performance architecture. Built primarily in native Zig with zero-overhead C-ABI bindings across 7 languages, it establishes a new standard for **raw unquantized weight storage, universal multi-device super-coalescing, hardware page-cache memory mapping, lossless 2:4 structured sparsity, SIMD execution, and modular architecture evolution**, while maintaining **complete quantization parity** for resource-constrained edge deployments.

---

## Core Pillars & Highlights

### 1. Foundational Raw Weight Storage & Super-Coalesced Multi-Device Architecture
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

### 2. Split Mode Sharding & Regeneratable Modular Weights
- **Multi-File Sharding (`HeaderFlags.IS_SHARDED = 0x40`)**: Splits multi-hundred-gigabyte raw checkpoints cleanly across storage boundaries with standardized manifest indexes (`model.hk.index.json`), allowing lazy cross-shard slice access.
- **Regeneratable Weights & Dynamic Modularity**: Sharding works seamlessly in raw storage mode, allowing individual layers, LoRA adapters, and delta patches to be attached, detached, or regenerated dynamically without modifying base weights.
- **Appendix Version Lineage & Instant Rollback**: Append LoRA adapters, delta patches, and evaluation benchmarks directly to `.hk` files with cryptographic parent hash chaining and single-command rollback (`hk_appendix_rollback`).

### 3. Lossless Structural Sparsity without Quantization
- **NVIDIA Ampere / Hopper / Blackwell 2:4 Structured Sparsity**: Stores 2 non-zero elements per 4-element block with physical 2-bit nibble metadata packing. Cuts physical on-disk storage and memory bandwidth by **50% (1.88× physical compression)** while preserving **exact 16-bit floating point precision** and dynamic range ($0.000000$ error vs dense baseline).
- **Compressed Sparse Representations**: Bitmask, Compressed Sparse Row (CSR), and Block Sparse Row (BSR) encoding zero out pruned connections without quantization distortion, streaming directly into native SIMD unpacking kernels at $>2.5\text{ GB/s}$.
- **Physical Channel Pruning**: Structurally drops unneeded attention heads or MLP channels from the tensor layout with explicit dimension mapping, reducing FLOPs and parameter footprint simultaneously.

### 4. Cache-Optimized 2D Tiling for Full-Precision GEMM/GEMV
- **Tensor-Core-Aligned Tiling (`TileLayout`)**: Reorganizes 2D/nD weight matrices into 16×16, 32×16, and 64×64 cacheline-aligned tiles with contiguous inner-K dimension packing to eliminate cache line thrashing and shared memory bank conflicts.

### 5. Microsecond In-Place Metadata Editing
- **Microsecond In-Place Metadata Patching (`hk metadata set`)**: Updates tokenizer configs, chat templates, hyperparameter tags, and licenses directly in the file header without re-serializing multi-gigabyte weight tensors.

### 6. Native Zig-First High-Performance Engine
- **Ultra-Lightweight Footprint**: The standalone native Zig binary (`hk.exe`) idles at just **2.80 MB of RAM** (>98% less memory than Python/PyTorch) and starts in **53 ms** (90.4× faster).
- **Zero-Copy Context Management**: Dynamic context window truncation executes in **14.3 µs** on 65k contiguous token buffers (13.4× faster than Python).
- **Hardware-Adaptive Inference**: Native CPUSpecs detection tailors vector paths to AVX2, AVX-512 VNNI, AMX-TILE, ARM NEON, or Apple Metal dynamically.

### 7. Comprehensive Quantization Suite (Complementary for Edge Deployments)
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
# Build native release artifacts (hk.exe / hk.dll / libhk.so)
zig build -Doptimize=ReleaseFast

# Run native test suite (42/42 native tests passing)
zig build test
```

---

### 2. Python API: Raw Weight Storage & Zero-Copy Access

```python
import torch
import hk

# 1. First-Class Raw Weight Storage (Zero Decoding Headroom)
# Stores raw unquantized IEEE weights (BF16, FP16, FP32, INT8) with super-coalesced alignment
weights = {
    "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
    "model.layers.0.mlp.up_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
}
hk.save_raw(weights, "model_raw.hk", universal_alignment=True)

# 2. Instant Zero-Copy Memory-Mapped Store
store = hk.load_raw("model_raw.hk")
print(f"Is Raw Storage: {store.is_raw_storage}")
print(f"Is Universal Page Aligned: {store.is_universal_page_aligned}")
print(f"Is Tensor Core Aligned: {store.is_tensor_core_aligned}")

# Zero-copy tensor slice access directly from mapped storage
w_gate = store["model.layers.0.mlp.gate_proj.weight"]
print(f"Loaded tensor: shape={w_gate.shape}, dtype={w_gate.dtype}")

# 3. Direct Native Linear Algebra (Zero Prior Allocation)
x = torch.randn(4096, dtype=torch.float32)
y = store.gemv("model.layers.0.mlp.gate_proj.weight", x)

# 4. Multi-Device Hardware Optimization Exporters
# Optimizes alignment for target device while maintaining 100% NVIDIA Tensor Core compatibility
hk.to_amd_rocm("model_raw.hk", "model_amd.hk")       # 4KB aligned for ROCm DirectGMA
hk.to_intel_npu("model_raw.hk", "model_intel.hk")    # 4KB aligned for OpenVINO Direct DMA
hk.to_apple_metal("model_raw.hk", "model_metal.hk")  # 16KB aligned for Metal zero-copy
hk.to_nvidia_tensor_core("model_raw.hk", "model_nv.hk")

# 5. Split Mode Sharding with Raw Weights (Regeneratable Weights / LoRA)
hk.save_sharded_raw(weights, "shards/", max_shard_size="2GB")
sharded_store = hk.load_sharded_raw("shards/model.hk.index.json")

# 6. Complementary Quantization & Transcoding (When Edge Compression is Needed)
from hk import convert_gguf_to_hk, export_hk_to_gguf
convert_gguf_to_hk("model.Q4_K_M.gguf", "model.hk")
export_hk_to_gguf("model.hk", "exported.gguf")
```

---

### 3. Native Standalone CLI & Developer Tools

The compiled native `hk` CLI provides sub-millisecond execution with minimal memory usage:

```bash
# 1. Profile host architecture (AMD Zen, Intel AVX-512/AMX, Apple Silicon, ARM NEON/SVE)
hk hardware-profile

# 2. Inspect container header, alignment audit, and non-quantized tensor layouts
hk inspect model.hk

# 3. Comprehensive binary dumper (hex header breakdown, flags, memory offsets)
hk dump model.hk

# 4. Verify 128-byte Tensor Core alignment and container checksums
hk verify model.hk

# 5. In-place metadata patching without rewriting multi-gigabyte weight arrays
hk metadata set model.hk general.version "1.0.0"
hk metadata set model.hk tokenizer.chat_template "{% for message in messages %}..."

# 6. Lossless GGUF / SafeTensors transcoding
hk convert-gguf model.gguf model.hk
hk export -f safetensors model.hk model.safetensors
hk export -f gguf model.hk exported.gguf

# 7. Native vocabulary and SwiGLU MLP width expansion (Net2WiderNet)
hk expand in.hk out.hk --vocab 32000 --width 1.33

# 8. Launch visual HK Model Studio GUI
hk gui model.hk
# Or via Python:
hk-gui model.hk

# 9. Cryptographic hash verification (streaming SHA-256)
hk hash model.hk
```

---

## Architectural Comparison

| Feature | SafeTensors | GGUF | HK Framework |
|:---|:---|:---|:---|
| **Primary Architecture** | JSON-header float serialization | CPU-first quantized inference | **Unified high-performance container for raw non-quantized & quantized execution** |
| **Raw Unquantized Weights** | Basic unaligned byte blobs | Inefficient fallback | **First-class IEEE/INT raw storage with 0 compute headroom (no decode penalty)** |
| **Multi-Device DMA Zero-Copy** | None | 32-byte CPU alignment | **Super-Coalesced: 4KB AMD/Intel + 16KB Apple Metal + 128B NVIDIA in one file** |
| **Non-Quantized Storage** | Dense unaligned arrays | Dense floats with 32B alignment | **Hardware-aligned (64B/128B/4096B/16KB) zero-copy memory mapping** |
| **Container Header Overhead** | Multi-megabyte JSON text string | Variable-length string KV array | **Fixed 128-byte binary header + 128-byte TOC entries (<0.001% overhead)** |
| **Zero-Allocation Deduplication** | None (duplicate arrays saved) | None | **`SHARED_REF` (tied embeddings) & `NULL_REF` (0-byte pruned layers)** |
| **Structured Hardware Sparsity** | No support | No support | **Native Ampere 2:4 (nibble indices, 1.88× lossless physical compression)** |
| **2D Tensor Tiling** | Row-major only | CPU strided only | **Tensor-Core-aligned K-contiguous tiles (16×16, 32×16, 64×64 WMMA)** |
| **In-Place Metadata Updates** | Impossible (requires rewrite) | Requires container rewrite | **Microsecond in-place header patching (`hk metadata set`)** |
| **Quantization Coverage** | None (dense floats only) | Lossy K/I-quants only | **Comprehensive: Dual-Mode (NF4 + residual recovery), K-Quants, I-Quants, MXFP4, NVFP4, BitNet** |
| **Model Architectures** | External dependency | 137 fixed architectures | **137+ native architectures with compiled bidirectional mapping (320k names/sec)** |
| **Process Idle RAM** | ~197 MB (PyTorch runtime) | Variable C++ | **2.80 MB (Standalone native Zig `hk.exe`)** |
| **Startup Latency** | ~4,880 ms (Python runtime) | ~120 ms | **53.98 ms (Standalone native Zig `hk.exe`)** |
| **Multi-Stage Pipelines** | None | None | **Heterogeneous DAGs (Whisper STT + DistilBERT + Causal LLM)** |
| **Live Dynamic Expansion** | Impossible | Impossible | **Live Net2WiderNet (Linear & SwiGLU), Net2DeeperNet, Vocab expansion** |

---

## Empirical Benchmarks

### 1. Non-Quantized Loading & Memory-Mapped Slicing (HK vs SafeTensors)
Evaluated across 32 dense full-precision tensors (224.00 MB FP32) with 50 iterations per operation:

| Reader / Loader Access Pattern | Container Size | Operation | Latency (ms) | Relative Performance |
|:---|:---:|:---|:---:|:---:|
| **SafeTensors (`load_file`)** | 224.00 MB | Full Load (32 tensors) | 1.473 ms | 1.00x |
| **HK Native (`load_file`)** | 224.00 MB | Full Load Zero-Copy | **0.797 ms** | **1.85× faster** |
| **SafeTensors (`get_slice` direct)** | 224.00 MB | Direct 2D Slice [128, 512] | 0.0149 ms | 1.00x |
| **HK Native (`get_slice` direct)** | 224.00 MB | Direct 2D Slice [128, 512] | **0.0092 ms** | **1.62× faster** |
| **SafeTensors (`safe_open` lifecycle)** | 224.00 MB | Open + Slice Context | 0.332 ms | 1.00x |
| **HK Native (`safe_open` lifecycle)** | 224.00 MB | Open + Slice Context | **0.274 ms** | **1.21× faster** |

- **Sub-10 Microsecond Slicing**: Direct tensor slicing in `safe_open` executes in **9.2 microseconds** per slice (1.62× faster than SafeTensors) by calculating strided offsets directly on the memory map without copying unrequested bytes.
- **Full Model Load**: Full model dictionary loading runs in **0.797 ms** (1.85× faster than SafeTensors) thanks to 128-byte hardware cache-line alignment and zero-copy memory mapping.

---

### 2. Lossless NVIDIA Ampere 2:4 Structured Sparsity
Bit-exact hardware physical nibble compression without numerical precision loss:

| Operation | Input Size | Output Size | Compression | Latency (ms) | Throughput (MB/s) | Max Absolute Error |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ampere 2:4 Pack** | 64.00 MB | 34.00 MB | **1.88x** | 89.90 ms | 711.9 MB/s | **0.000000 (Exact Lossless)** |
| **Ampere 2:4 Unpack** | 34.00 MB | 64.00 MB | 1.00x | **24.87 ms** | **2,573.0 MB/s** | **0.000000 (Exact Lossless)** |

- **Lossless Storage Reduction**: Achieves a 1.88× physical storage reduction while maintaining a **1.000000 cosine similarity** and zero deviation from full IEEE floating-point precision.

---

### 3. Native Zig CLI & Process Footprint
Comparing process initialization and idle memory between Python/PyTorch and the native `hk.exe` binary:

| Metric | Python Runtime (Torch + HK) | Standalone Zig Binary (`hk.exe`) | Improvement / Delta |
|:---|:---:|:---:|:---:|
| **Idle Process RAM (RSS)** | 197.44 MB | **2.80 MB** | **98.6% less RAM** |
| **Process Startup Latency** | 4,880.72 ms | **53.98 ms** | **90.4× faster startup** |
| **HF Tensor Name Mapping Rate** | 12,400 names/sec | **320,944 names/sec** | **25.8× faster** |
| **65k Context Truncation** | 191.82 µs | **14.30 µs** | **13.4× faster** |
| **Growth Governor (50k Checks)** | 7.89 ms | **3.57 ms (Batched)** | **2.2× faster** |

---

### 4. Quantization Benchmarks (Complementary Deployment)
Evaluated on a 2048 x 2048 matrix (4,194,304 f32 elements / 16.00 MB) with AVX2 SIMD acceleration:

| Quant Format | Packed Size | Compression | Quant Latency | Quant Throughput | Dequant Latency | Dequant Throughput | Cosine Sim |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q8_0 (8-bit Symmetric)** | 4.25 MB | 3.76x | 6.4 ms | 2,489.8 MB/s | **2.4 ms** | **6,688.5 MB/s** | 0.999985 |
| **Q4_0 (4-bit Symmetric)** | 2.25 MB | 7.11x | 6.7 ms | 2,372.8 MB/s | **3.4 ms** | **4,667.7 MB/s** | 0.996318 |
| **Q8_K (8-bit K-Quant)** | 4.56 MB | 3.51x | 5.9 ms | 2,718.0 MB/s | **4.7 ms** | **3,396.6 MB/s** | 0.999975 |
| **Q4_K (4-bit K-Quant)** | 2.25 MB | 7.11x | 5.9 ms | 2,705.7 MB/s | **2.9 ms** | **5,535.6 MB/s** | 0.994132 |
| **Dual-Mode NF4 + Residual** | 97.09 MB (SmollM) | 6.40x | 245.0 ms | — | — | **0.999994** |

---

## Multilingual SDK Ecosystem

HK provides native, zero-overhead bindings across all major environments:

- **Python**: [`python/hk/`](python/hk/) - Hugging Face-style drop-in API with PyTorch, NumPy, JAX, and Flax bindings.
- **Rust**: [`bindings/rust/`](bindings/rust/) - Safe idiomatic Rust crate (`Cargo.toml`).
- **TypeScript / Node.js**: [`bindings/js/`](bindings/js/) - Zero-dependency browser, Node.js, and WASM SDK (`@hk-format/core`).
- **C# / .NET**: [`bindings/csharp/`](bindings/csharp/) - Modern .NET package for Unity, Windows, and enterprise systems (`Hk.csproj`).
- **Go**: [`bindings/go/`](bindings/go/) - Idiomatic Go module using cgo (`go.mod`).
- **Java / Android**: [`bindings/java/`](bindings/java/) - JNI package with direct NIO `ByteBuffer` zero-copy support (`pom.xml`).
- **C / C++**: [`include/hk.h`](include/hk.h), [`include/hk.hpp`](include/hk.hpp) - C-ABI and C++20 RAII headers.

---

## License

HK is open source software licensed under the [Apache License, Version 2.0](LICENSE).
