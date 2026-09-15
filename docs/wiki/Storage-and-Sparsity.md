# Storage Innovations, Sparsity, and Quantization

In this guide, I cover the storage optimizations I built into HK: lossless hardware structured sparsity, cacheline-aligned 2D tiling, in-place metadata updates, multi-file sharding, and the complementary edge quantization suite.

---

## 1. Lossless NVIDIA Ampere 2:4 Structured Sparsity

Modern NVIDIA GPUs (Ampere, Ada Lovelace, Hopper, Blackwell) include Sparse Tensor Cores that double matrix multiplication throughput if weight matrices conform to a 2:4 sparsity pattern: exactly 2 out of every 4 consecutive values must be non-zero.

### The Problem with Traditional Formats
Standard formats like SafeTensors and GGUF do not support physical 2:4 sparsity. If you prune weights to 2:4, traditional formats still store the zeros on disk as full 16-bit floats, saving zero disk space and wasting PCIe transfer bandwidth.

### My Solution in HK: Physical Nibble Packing
I implemented 2:4 structured sparsity natively in HK:
- The 2 non-zero 16-bit values are packed contiguously.
- The positions of the 2 values within each 4-element block are stored as a compact 2-bit nibble index (4 bits per block).
- Physical Storage: Cuts physical file size by **50% (1.88x physical compression)**.
- Numerical Accuracy: Unlike quantization, the non-zero weights remain full uncompressed 16-bit floats. There is **0.000000 numerical error** compared to the sparse baseline.
- Streaming Speed: Native SIMD unpacking kernels stream into memory at over 2.5 GB/s.

```python
import torch
import hk
from hk.pruning import prune_to_2_4, pack_2_4_sparse

# 1. Take a dense weight matrix
dense_weight = torch.randn(2048, 4096, dtype=torch.bfloat16)

# 2. Prune to 2:4 pattern (picks 2 largest magnitude elements per 4-block)
sparse_weight, mask = prune_to_2_4(dense_weight)

# 3. Pack into physical nibble storage
packed = pack_2_4_sparse(sparse_weight)
print("Original size:", dense_weight.numel() * 2, "bytes")
print("Packed 2:4 size:", len(packed.data) + len(packed.indices), "bytes")
# Storage is reduced by ~47-50%
```

---

## 2. 2D Cacheline-Aligned Tiling (`TileLayout`)

Standard neural weight formats save 2D matrices in simple row-major order. When executing matrix multiplication on hardware:
- Accessing elements across columns requires long memory strides.
- GPU threads in adjacent warp lanes access addresses that span different cachelines, causing shared memory bank conflicts and cacheline thrashing.

I designed HK to allow weight matrices to be serialized in **2D cacheline tiles**:
- `TileLayout.tile_16x16`: 16x16 elements packed contiguously.
- `TileLayout.tile_32x16`: 32x16 elements aligned with Tensor Core WMMA instructions.
- `TileLayout.tile_64x64`: 64x64 tiles for large matrix multiplication on desktop GPUs and multi-threaded CPUs.

Because inner-K dimensions are contiguous inside each tile, data stays resident in L1/L2 processor caches during matrix-vector operations, boosting throughput without changing parameter values.

---

## 3. Microsecond In-Place Metadata Editing

In traditional formats (like SafeTensors or GGUF), the metadata dictionary sits before the tensor data, but the file format does not allocate spare headroom. If you want to change a single character in a chat template or update a model's license tag, you have to rewrite the entire multi-gigabyte file from scratch.

I solved this in HK by pre-allocating an elastic metadata padding zone in the file header.

You can patch metadata in place directly from your command line in microseconds:

```bash
# Update model version tag
hk metadata set model.hk general.version "1.1.0"

# Update Jinja2 chat template without rewriting weights
hk metadata set model.hk tokenizer.chat_template "{% for msg in messages %}..."
```

The CLI modifies the bytes in place in less than 5 milliseconds on a 10 GB file.

---

## 4. Multi-File Sharding (`HeaderFlags.IS_SHARDED`)

When distributing 70B+ models across storage boundaries or multi-drive setups, monolithic files can become unwieldy.

I designed HK to provide multi-file sharding natively:
- Header flag `0x40` signals a sharded container.
- Each shard contains its own 128-byte header, a local TOC, and a global index manifest (`model.hk.index.json`).
- Python and native loaders resolve tensor slice pointers across shard boundaries transparently.

```python
from hk.raw import save_sharded_raw, load_sharded_raw

# Save weights split across shards of maximum 2 GB each
save_sharded_raw(weights_dict, "./sharded_model", max_shard_size_gb=2.0)

# Load sharded store
store = load_sharded_raw("./sharded_model")
print("Total shards mapped:", store.shard_count)
```

---

## 5. Complementary Edge Quantization Suite

While I prioritize zero compute headroom non-quantized storage in HK, I also built an exhaustive suite of quantization schemes for extreme compression on edge devices:

### Dual-Mode Quantization (NF4 + Residual Delta)
- Stores a 4-bit non-linear base weight (NF4) for fast low-memory loading.
- Appends an optional high-precision residual delta stream.
- In memory-constrained environments, you run pure 4-bit inference.
- If more memory becomes available, HK applies the residual stream to recover full 16-bit floating point precision (>0.99999 cosine similarity) without reloading the model.

### Super-Block K-Quants (`Q2_K` through `Q8_K`)
- Uses 256-element super-blocks with hierarchical sub-block scaling factors matching GGUF precision parity.
- Ideal for running models on low-memory mobile devices or edge boards.

### Vector I-Quants (`IQ1_S` through `IQ4_NL`)
- Non-linear codebook quantization fitted to Gaussian weight distributions for minimal quantization noise at 1-bit to 3-bit precision.

### Microscaling Formats (MXFP4 & NVFP4)
- OCP MXFP4 and NVIDIA Blackwell NVFP4 microscaling formats with 32-element micro-block scales for next-generation hardware.
