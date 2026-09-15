# Raw Storage and Super-Coalesced Silicon Alignment

In this guide, I explain how I designed HK to achieve zero compute headroom and universal hardware alignment across NVIDIA, AMD, Intel, and Apple Silicon.

---

## What Does Zero Compute Headroom Mean?

When you run inference with traditional quantized formats (such as 4-bit or 8-bit GGUF files), every single forward pass requires a dequantization step:
1. Load compressed blocks from memory into registers.
2. Read block scales and offsets.
3. Multiply each compressed nibble by its scale factor to reconstruct an approximate 16-bit or 32-bit float.
4. Execute the actual matrix-vector multiplication (GEMV).

This reconstruction step burns memory bandwidth and CPU/GPU compute cycles on every generated token.

I built HK with a different design goal: **Zero Compute Headroom**. 

In HK, full-precision and half-precision weights (`bfloat16`, `float16`, `float32`, `int8`) are saved as contiguous bit representations aligned directly with the host memory bus. When you load a model, the OS maps the file directly into virtual address space via `mmap` or `MapViewOfFile`. 

There is:
- Zero decoding overhead
- Zero dequantization step
- Zero Python heap allocation
- Zero memory copying

The CPU SIMD or GPU Tensor Core kernels read directly from the memory-mapped virtual address. The compute cores spend 100% of their execution time doing actual linear algebra rather than unpacking bytes.

---

## The Super-Coalesced Silicon Invariance

Modern chips have very specific, non-negotiable memory alignment requirements for high-performance data transfers:

1. NVIDIA GPUs (Tensor Cores & Warp Coalescing):
   A warp consists of 32 threads. To fetch memory in a single bus transaction, the starting memory address must be aligned to a 128-byte boundary. Unaligned addresses split a single transaction into multiple serialized reads, cutting bandwidth in half.

2. AMD GPUs (ROCm DirectGMA) and Intel CPUs:
   Modern Linux and Windows kernels manage virtual memory in 4096-byte (4 KB) pages. Zero-copy Direct Memory Access (DMA) between storage controllers, host RAM, and PCIe devices requires 4KB alignment.

3. Apple Silicon (M-Series Metal Unified Memory):
   On macOS ARM64, the Metal API provides zero-copy buffer creation via `newBufferWithBytesNoCopy`. This API requires that the buffer pointer and its length be aligned to the macOS page boundary, which is 16384 bytes (16 KB).

### The Mathematical Invariance

Historically, developers had to build different files for different chips: one for Apple Metal, one for NVIDIA CUDA, and one for CPU servers.

I solved this in HK using a simple mathematical truth:

```
4096 bytes  = 32 x 128 bytes
16384 bytes = 128 x 128 bytes
```

Because 4096 and 16384 are exact integer multiples of 128:
- A tensor payload aligned to 4096 bytes (4 KB) automatically satisfies NVIDIA's 128-byte warp coalescing.
- A tensor payload aligned to 16384 bytes (16 KB) simultaneously satisfies Apple Metal zero-copy, AMD/Intel Direct DMA, and NVIDIA warp coalescing.

A single `.hk` file can be opened on an Apple M3 MacBook, copied over to an NVIDIA RTX workstation, or loaded on an AMD Ryzen Linux server, and every single chip gets native, uncompromised, zero-copy alignment out of the box.

---

## Virtual Deduplication: `SHARED_REF` and `NULL_REF`

Large language models frequently share weights between layers. For example, tied embedding models use the exact same weight matrix for both `model.embed_tokens.weight` and `lm_head.weight`.

In standard formats like SafeTensors, both matrices are often serialized separately, duplicating hundreds of megabytes of raw floats on disk.

HK eliminates this with virtual references:

### `SHARED_REF` (0x31)
When HK saves a model, it inspects the memory pointers (`data_ptr`) of each tensor. If two keys point to the exact same physical memory address, HK marks the second entry in the Table of Contents as `SHARED_REF` and points its `data_offset` directly to the first tensor.
- On disk: Only one copy of the weight is stored.
- On load: Both tensor names resolve to the exact same underlying memory-mapped pointer.
- Result: Hundreds of megabytes saved without losing tied-weight identity.

### `NULL_REF` (0x30)
If you prune a layer, drop an unused attention projection, or zero out weights, HK marks the TOC entry as `NULL_REF`. The tensor name and metadata remain intact for architecture compatibility, but `data_size = 0`. Zero bytes of disk space are used.

---

## Python Usage with `hk.raw`

HK provides high-level Python utilities for saving, inspecting, and running math on raw unquantized stores:

```python
import torch
import hk
from hk.raw import HKRawWeightStore, save_raw, load_raw

# 1. Save unquantized weights with universal silicon alignment
weights = {
    "layer0.gate.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
    "layer0.up.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
}
save_raw(weights, "model_raw.hk", universal_alignment=True)

# 2. Open zero-copy memory-mapped store
store = load_raw("model_raw.hk")

# 3. Check alignment guarantees
print("Universal Page Aligned (4KB):", store.is_universal_page_aligned)
print("Tensor Core Aligned (128B) :", store.is_tensor_core_aligned)

# 4. Perform direct SIMD GEMV multiplication
# Computes y = W * x directly from memory-mapped pages without prior RAM allocation
x = torch.randn(4096, dtype=torch.float32)
y = store.gemv("layer0.gate.weight", x)
print("Output vector shape:", y.shape)

# Close cleanly to unmap memory
store.close()
```

---

## Verifying Alignment via CLI

You can verify any `.hk` file directly from your terminal:

```bash
# Verify alignment, checksums, and container integrity
hk verify model.hk
```

The tool checks that every tensor payload starts on a clean 128-byte boundary and confirms whether universal 4KB or 16KB alignment is active.
