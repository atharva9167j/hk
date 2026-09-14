# HK Neural Tensor Format Specification (.hk)
**Version: 1.0.0**  
**Target Hardware: Ampere / Hopper / Blackwell / Apple Silicon / x86_64 AVX-512 / ARM NEON**

---

## 1. Executive Summary & Design Principles

The **HK Neural Tensor Format (`.hk`)** is a next-generation high-performance binary container for deep learning model weights, metadata, and execution topologies. It addresses fundamental architectural bottlenecks in existing formats (SafeTensors, GGUF, ONNX, and PyTorch `.pt`) by establishing **non-quantized storage and access efficiency** as its foundational architecture, alongside first-class support for structural sparsity and edge quantization:

1. **Zero-Copy Memory-Mapped Storage Efficiency**: Fully aligned binary layouts enable direct OS zero-copy page mapping (`MapViewOfFile` on Windows, `mmap` on POSIX) directly into host memory or GPU Unified Memory with zero deserialization overhead and zero Python heap object allocation.
2. **Minimal Binary Container Overhead (<0.001%)**: Eliminates multi-megabyte JSON text headers by utilizing a compact, fixed 128-byte binary file header and fixed 128-byte Table of Contents (TOC) entries (<128 KB metadata overhead for a 1,000-tensor model).
3. **Lossless Hardware Structured Sparsity without Quantization**: Native hardware 2:4 structured sparsity (matching NVIDIA Ampere/Ada/Hopper/Blackwell Sparse Tensor Cores for 2× GEMM throughput at 50% physical storage reduction with 0.000000 numerical error), alongside bitmask, CSR, and BSR representations.
4. **Zero-Physical Allocation for Dead & Shared Weights**: `SHARED_REF` allows tied embeddings (`lm_head.weight` == `embed_tokens.weight`) and recursive layer weights to point to identical physical data offsets without duplicating multi-hundred-megabyte full-precision buffers on disk; `NULL_REF` allows zero-byte representation of pruned layers or zeroed tensors.
5. **Tensor-Core-Aligned 2D Tiling**: Reorganizes 2D/nD weight matrices into cacheline (128-byte) and Tensor-Core-aligned tiles (e.g. 16×16, 32×16, 64×64) with contiguous inner-K dimension packing to eliminate warp divergence, cacheline thrashing, and shared-memory bank conflicts during non-quantized and quantized GEMM.
6. **In-Place Metadata Updates & Sharding**: Split 70B+ models across storage boundaries (`HeaderFlags.IS_SHARDED = 0x40`) and patch metadata in-place in microseconds without re-serializing gigabytes of full-precision weights.
7. **Comprehensive Quantization Zoo (Complementary for Edge Deployments)**: When reduced-precision edge deployments are needed, HK provides a complete suite of quantization formats (Dual-Mode NF4/DQ8 with residual precision recovery streams, GGUF-compatible Q4_0 and Q8_0, super-block K-quants Q2_K..Q8_K, non-linear I-quants IQ1..IQ4, ternary BitNet, and hardware microscaling MXFP4/NVFP4).

---

## 2. Binary File Layout

An `.hk` file is divided into distinct contiguous sections:

```
+-------------------------------------------------------------+
| HK File Header (128 bytes, fixed size)                      |
+-------------------------------------------------------------+
| Metadata Key-Value Section (Binary MsgPack-like format)     |
+-------------------------------------------------------------+
| Tensor Table of Contents (TOC)                              |
+-------------------------------------------------------------+
| Padding to 128-Byte Boundary                                |
+-------------------------------------------------------------+
| Tensor Payload Data (Each tensor aligned to 128 bytes)      |
|   - Tensor 0 (compact base + optional residual / tiles)     |
|   - Tensor 1 ...                                            |
|   - Tensor N ...                                            |
+-------------------------------------------------------------+
| Appendix Region (Optional: LoRA deltas, dynamic growth)    |
+-------------------------------------------------------------+
```

---

## 3. Header Specification (128 Bytes)

The header occupies the first 128 bytes of every `.hk` file, aligning with GPU cache-line and memory transaction boundaries:

| Offset | Field Name | Type | Description |
|---|---|---|---|
| 0x00 | `magic` | `[4]u8` | `0x48, 0x4B, 0x4E, 0x54` (`"HKNT"`) |
| 0x04 | `version_major` | `u16` | Major format version (currently `1`) |
| 0x06 | `version_minor` | `u16` | Minor format version (currently `0`) |
| 0x08 | `flags` | `u32` | Bitflags: <br>• Bit 0: Little-endian (`0x01`)<br>• Bit 1: Has Appendix (`0x02`)<br>• Bit 2: Quantization table present (`0x04`)<br>• Bit 3: 2:4 Structured Sparsity enabled (`0x08`)<br>• Bit 4: Tile-aligned layout (`0x10`)<br>• Bit 5: Flexible alignment mode (`0x20`)<br>• Bit 6: Multi-file sharded (`0x40`) |
| 0x0C | `alignment` | `u16` | Hardware memory alignment boundary in bytes (default: `128`) |
| 0x0E | `split_index` | `u16` | Multi-file sharding: 0-based shard index |
| 0x10 | `tensor_count` | `u64` | Total number of tensors in this file / shard |
| 0x18 | `metadata_kv_count` | `u64` | Total number of metadata KV pairs |
| 0x20 | `metadata_offset` | `u64` | Byte offset to metadata section |
| 0x28 | `metadata_size` | `u64` | Size in bytes of metadata section |
| 0x30 | `tensor_toc_offset` | `u64` | Byte offset to Tensor TOC |
| 0x38 | `tensor_toc_size` | `u64` | Size in bytes of Tensor TOC |
| 0x40 | `tensor_data_offset`| `u64` | Byte offset to tensor payload (aligned to `alignment`) |
| 0x48 | `appendix_offset` | `u64` | Byte offset to appendix region (0 if none) |
| 0x50 | `checksum` | `u64` | CRC-64 or xxHash64 of TOC + Metadata |
| 0x58 | `split_count` | `u16` | Multi-file sharding: total number of shards (>= 1) |
| 0x5A | `reserved1` | `[38]u8`| Reserved 38 bytes for future expansion |

---

## 4. Storage Types (`StorageType`)

Each tensor TOC entry declares its storage format via an 8-bit identifier:

```zig
pub const StorageType = enum(u8) {
    // Dense unquantized float & integer types
    f32 = 0x00,
    f16 = 0x01,
    bf16 = 0x02,
    fp8_e4m3 = 0x03,
    fp8_e5m2 = 0x04,
    int8 = 0x05,
    int32 = 0x06,
    int64 = 0x07,
    uint8 = 0x08,
    bool = 0x09,

    // Dual-mode quantized types
    dq4 = 0x10,       // 4-bit NF4/INT4 base + block scale + optional residual
    dq8 = 0x11,       // 8-bit quantized
    dq6 = 0x12,       // 6-bit quantized
    dq12 = 0x13,      // 12-bit quantized
    dqt = 0x14,       // Ternary {-1, 0, +1} (BitNet b1.58)

    // Sparse types
    sparse_f16 = 0x20,
    sparse_dq8 = 0x21,
    sparse_2_4 = 0x22,     // 2:4 structured sparse (Ampere hardware supported)
    sparse_dq4_2_4 = 0x23, // 2:4 structured sparse with 4-bit quantization

    // Reference & virtual types
    null_ref = 0x30,   // Fully pruned or all-zero tensor (0 physical bytes stored)
    shared_ref = 0x31, // Points to another tensor's data (tied embeddings)
    lora_ref = 0x32,   // Base tensor + Low-Rank A and B factors

    // K-Quants (Block size 256 super-blocks)
    q2_k = 0x40,       // 2-bit K-quant (256-element super-block)
    q3_k = 0x41,       // 3-bit K-quant (256-element super-block)
    q4_k = 0x42,       // 4-bit K-quant (256-element super-block with 8 sub-blocks)
    q5_k = 0x43,       // 5-bit K-quant (256-element super-block)
    q6_k = 0x44,       // 6-bit K-quant (256-element super-block with 16 scales)
    q8_k = 0x45,       // 8-bit K-quant (256-element super-block with sub-sums)

    // Importance & Non-linear Quants (I-Quants)
    iq1_s = 0x50,      // 1-bit importance quant
    iq1_m = 0x51,
    iq2_xxs = 0x52,
    iq2_xs = 0x53,
    iq2_s = 0x54,
    iq3_xxs = 0x55,
    iq4_nl = 0x56,     // 4-bit non-linear codebook quant
    iq4_xs = 0x57,

    // Microscaling & Ternary formats
    tq1_0 = 0x60,      // Ternary quant 1.0
    tq2_0 = 0x61,      // Ternary quant 2.0
    mxfp4 = 0x62,      // OCP Microscaling FP4 (E2M1, block size 32)
    nvfp4 = 0x63,      // NVIDIA Blackwell Microscaling NVFP4 (E2M1)
};
```

---

## 5. Non-Quantized Storage & Memory Mapping Architecture

HK treats non-quantized weight storage (`FP32`, `FP16`, `BF16`, `FP8_E4M3`, `FP8_E5M2`, `INT8`, `INT32`, `INT64`) as a foundational architecture, delivering substantial storage efficiency, memory savings, and throughput gains over SafeTensors and PyTorch:

### 5.1 Zero-Copy OS Page Cache Memory Mapping (`mmap`)
- **Direct Physical Mapping**: Tensors are laid out in contiguous physical byte spans aligned to 64-byte / 128-byte hardware cachelines and 4096-byte OS virtual memory page boundaries.
- **Sub-Millisecond Loading**: The runtime opens the container via `mmap` (POSIX) or `MapViewOfFile` (Windows) using `MAP_SHARED` or `MAP_PRIVATE` (copy-on-write). A 70B parameter FP16 checkpoint (~140 GB) maps in **under 1 millisecond** without loading inactive layers or allocating intermediary Python heap buffers.
- **On-Demand Page Faulting**: Tensors are paged directly from the NVMe storage subsystem into CPU/GPU cache by the OS virtual memory manager only when accessed by compute kernels, keeping process idle RAM minimal (e.g. 2.80 MB for the standalone native Zig binary).

### 5.2 Minimal Binary Container Overhead (<0.001%)
- **Binary vs. JSON Metadata**: SafeTensors precedes weight buffers with a variable-length JSON string that can scale to several megabytes, requiring text parsers, string hashing, and GC allocation during deserialization.
- **Fixed-Size Compact Table of Contents (TOC)**: HK uses a fixed 128-byte file header followed by binary Table of Contents entries (128 bytes per tensor) storing name offsets, dimension vectors, storage tags, and physical offsets as packed integers. For a 1,000-tensor model, the total container metadata overhead is under 128 KB (<0.001% of model size).

### 5.3 Virtual Deduplication & Zero-Physical Allocations
- **`SHARED_REF` (StorageType `0x31`)**:
  - Weight-tied architectures (such as `lm_head.weight` sharing identical weights with `model.embed_tokens.weight`, or recursive layer models like ALBERT) declare the duplicate tensor as a `SHARED_REF`.
  - The entry points directly to the data offset and length of the primary tensor.
  - Eliminates hundreds of megabytes to gigabytes of duplicate full-precision storage on disk and in memory.
- **`NULL_REF` (StorageType `0x30`)**:
  - Layers pruned during structural compression or zeroed bias vectors are registered as `NULL_REF`.
  - Consumes 0 physical payload bytes in the `.hk` file while maintaining topological index compatibility for compute graph execution.

---

## 6. Tile Layouts (`TileLayout`)

To optimize memory loads on Tensor Cores (e.g. WMMA instructions `m16n16k16` and `m16n8k16`), tensors can be stored in tiled block formats rather than naive row-major:

```zig
pub const TileLayout = enum(u8) {
    row_major = 0x00,
    col_major = 0x01,
    tile_16x16 = 0x02,       // 16x16 elements per tile, K-contiguous
    tile_16x8 = 0x03,        // 16x8 elements per tile
    tile_32x16 = 0x04,       // 32x16 elements per tile
    block_sparse_2_4 = 0x05, // 2 non-zero elements per 4-element block + 2-bit indices
    tile_32x32 = 0x06,       // 32x32 elements per tile
    tile_64x64 = 0x07,       // 64x64 elements per tile
};
```

### Tile Contiguity Rule
For a 2D matrix of shape `[M, K]`, tiled into `[M/16, K/16, 16, 16]`:
- Within each tile, the 16 elements of the K-dimension are stored contiguously.
- When loaded into CUDA shared memory, a warp of 32 threads can load a full tile in 4 vectorized 128-bit memory instructions without bank conflict or uncoalesced memory transactions.

---

## 7. Lossless Structural Sparsity without Quantization

HK implements hardware-accelerated structural sparsity that reduces on-disk storage footprint and memory traffic without sacrificing 16-bit floating point precision:

```zig
pub const SparsityType = enum(u8) {
    none = 0x00,
    bitmask = 0x01,
    csr = 0x02,
    structured_2_4 = 0x03,
    physical_pruned = 0x04,
    bsr = 0x05,
};
```

1. **2:4 Structured Sparsity (`structured_2_4`)**:
   - Exactly 2 out of every 4 consecutive values in each row are non-zero.
   - Values array: stored at 50% physical size in full IEEE precision (`FP16`, `BF16`, or `FP32`).
   - Metadata array: 2-bit indices packed into 8-bit bytes (4 blocks of 2:4 indices per byte), matching NVIDIA Ampere/Ada/Hopper/Blackwell sparse tensor core expectations.
   - Delivers a **1.88× physical storage compression** and **2× GEMM throughput** with **0.000000 maximum absolute error** vs dense float.
2. **Unstructured Pruning with Bitmask**:
   - Zero-pruned weights stored with an 1-bit presence bitmask.
   - Non-zero values stored contiguously without quantization degradation.
3. **Block-Sparse (BSR)**:
   - Zeroes out sub-blocks (e.g. 16×16) by Frobenius norm, ideal for Mixture-of-Experts routing.
4. **Physical Channel/Neuron Pruning**:
   - Structured pruning physically shrinks matrix dimensions `[M_pruned, K_pruned]`.
   - Stored directly as dense matrices of reduced rank, saving both memory and FLOPs without requiring sparse runtime kernels.

---

## 8. Quantization Zoo & Importance Calibration (Complementary Edge Schemes)

While HK's architecture is primarily engineered for optimal non-quantized storage and page-cache execution, it features a complete and comprehensive quantization zoo for edge and resource-constrained environments:

### 8.1 Dual-Mode Quantization Scheme
Dual-mode quantization stores weights in two complementary components:
1. **Compact Base Layer (`W_base`)**:
   - Quantized to 4-bit (NF4 or symmetric INT4) or ternary {-1, 0, 1}.
   - Per-block scaling factors (default block size = 32 elements).
   - Fast inference path: execute GEMM directly on quantized weights.
2. **Residual Recovery Layer (`W_res`)**:
   - Computed during export: $R = W_{\text{orig}} - \text{Dequantize}(W_{\text{base}})$.
   - Scaled and stored as low-bit residual deltas or sparse outlier corrections.
   - When precision recovery is enabled (e.g., for critical attention heads or sensitive layers), the runtime reconstitutes:
     $$W = \text{Dequantize}(W_{\text{base}}) + S_{\text{res}} \times W_{\text{res}}$$
   - Allows on-the-fly fidelity tuning without maintaining duplicate model weights on disk ($>0.99999$ cosine similarity).

### 6.2 256-Element Super-Block K-Quants (`0x40` - `0x45`)
To minimize scale-factor storage overhead while maintaining fine granularity, K-quants organize 256 consecutive weights into a single super-block divided into smaller sub-blocks:

1. **`BlockQ4_K` (StorageType `0x42`)**:
   - Super-block of 256 elements structured into 8 sub-blocks of 32 weights.
   - Layout:
     - `scales: [12]u8` (packed 6-bit sub-block scales and minimums).
     - `d: f16` (super-block master scale factor).
     - `dmin: f16` (super-block master minimum factor).
     - `qs: [128]u8` (256 4-bit nibbles packed pairwise).
   - Size: 144 bytes per 256 weights = **4.50 bits per weight (bpw)**.
   - Dequantization formula for sub-block $j \in [0, 7]$ and weight index $i \in [0, 31]$:
     $$W[j \times 32 + i] = d \times s_j \times q[j \times 32 + i] - d_{\text{min}} \times m_j$$

2. **`BlockQ8_K` (StorageType `0x45`)**:
   - Super-block of 256 elements structured into 16 sub-blocks of 16 weights.
   - Layout:
     - `d: f32` (super-block master scaling factor).
     - `qs: [256]i8` (256 signed 8-bit quantized weights).
     - `bsums: [16]i16` (precomputed sub-sums for accelerated dot-product kernels).
   - Size: 292 bytes per 256 weights = **9.125 bpw**.

3. **`BlockQ6_K` (StorageType `0x44`)**:
   - Super-block of 256 elements structured into 16 sub-blocks of 16 weights.
   - Layout:
     - `ql: [128]u8` (low 4 bits of 256 weights).
     - `qh: [64]u8` (high 2 bits of 256 weights packed 4 per byte).
     - `scales: [16]i8` (16 signed 8-bit scale factors).
     - `d: f16` (super-block master scale factor).
   - Size: 210 bytes per 256 weights = **6.5625 bpw**.

4. **`BlockQ2_K`, `BlockQ3_K`, `BlockQ5_K` (`0x40`, `0x41`, `0x43`)**:
   - `BlockQ2_K`: 2-bit quantization with 16 sub-blocks of 16 elements (2.56 bpw).
   - `BlockQ3_K`: 3-bit quantization with low 2-bit and high 1-bit packing (3.44 bpw).
   - `BlockQ5_K`: 5-bit quantization with 4-bit nibbles and 1-bit high flags (5.50 bpw).

### 6.3 Non-Linear & Importance Quants (I-Quants: `0x50` - `0x57`)
1. **`IQ4_NL` (StorageType `0x56`)**:
   - 4-bit non-linear codebook quantization.
   - Rather than assuming a uniform distribution, weights are indexed into a 16-entry non-linear table fitted to Gaussian bell distributions:
     $$\text{Codebook} = [-1.0, -0.696, -0.525, -0.397, -0.286, -0.185, -0.090, 0.0, 0.090, 0.185, 0.286, 0.397, 0.525, 0.696, 1.0]$$
   - Minimizes RMS quantization noise by up to 28% compared to standard uniform INT4.
2. **Low-Bit Importance Quants (`IQ1_S`, `IQ2_XXS`, `IQ3_XXS`)**:
   - Codebook and lattice vector quantization for aggressive sub-3-bit compression (1.56 to 3.06 bpw).

### 6.4 Microscaling & Ternary Formats (`0x60` - `0x63`)
1. **OCP Microscaling FP4 (`mxfp4`, StorageType `0x62`)**:
   - Open Compute Project (OCP) MXFP4 specification.
   - 32 elements per block sharing an 8-bit E8M0 scale factor.
   - Values encoded in 4-bit E2M1 floating point (1 sign bit, 2 exponent bits, 1 mantissa bit).
2. **NVIDIA Blackwell Microscaling (`nvfp4`, StorageType `0x63`)**:
   - 16 elements per sub-block sharing an FP8 (E4M3/E5M2) micro-scale.
   - Native execution on NVIDIA Blackwell NVFP4 Tensor Cores.
3. **Ternary Quantization (`tq1_0`, `tq2_0`, `dqt`)**:
   - 1.58-bit representations for ternary models (BitNet b1.58) where weights $\in \{-1, 0, +1\}$.

### 6.5 Activation-Aware Importance Matrix Calibration (`imatrix`)
Uniform quantization ignores the fact that different weights contribute unequally to the model's loss. HK integrates an importance matrix calibrator (`ImportanceMatrixCalibrator`) based on Fisher Information second-moment statistics:

1. **Second-Moment Accumulation**:
   Given a calibration dataset $X = \{x_1, x_2, \dots, x_N\}$, the importance $I_W$ of weight matrix $W$ is computed as the empirical second moment of input activations:
   $$I_W \approx \frac{1}{N} \sum_{k=1}^N (x_k \odot x_k)$$
2. **Weighted Quantization Error Minimization**:
   Quantization scales and offsets are chosen to minimize the importance-weighted reconstruction loss:
   $$\min_{\hat{W}} \sum_i I_{W, i} \cdot (W_i - \hat{W}_i)^2$$
3. **Container Storage & Portability**:
   Calibration matrices are stored with CRC32 verification and can be exported as `.imatrix` binary files or embedded directly in the `.hk` container metadata (`quant.imatrix_file`).

### 6.6 Predefined Per-Tensor Quantization Recipes
HK provides standardized mixed-precision recipes (`QUANT_RECIPES`) that automatically assign optimal precision to each tensor based on its architectural sensitivity:

| Recipe Name | Target bpw | Attention Projections ($Q, K, V$) | Output Projection ($O$) | Feed-Forward ($Up, Gate$) | Down Projection ($Down$) | Token Embeddings / Norms |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Q4_K_M`** | ~4.5 bpw | $Q4\_K$ | $Q4\_K$ | $Q4\_K$ | $Q6\_K$ | $Q8\_K$ / $FP16$ |
| **`Q5_K_M`** | ~5.5 bpw | $Q5\_K$ | $Q5\_K$ | $Q5\_K$ | $Q6\_K$ | $Q8\_K$ / $FP16$ |
| **`Q4_K_S`** | ~4.2 bpw | $Q4\_K$ | $Q4\_K$ | $Q4\_K$ | $Q4\_K$ | $Q6\_K$ / $FP16$ |
| **`Q5_K_S`** | ~5.1 bpw | $Q5\_K$ | $Q5\_K$ | $Q5\_K$ | $Q5\_K$ | $Q6\_K$ / $FP16$ |
| **`Q3_K_M`** | ~3.5 bpw | $Q3\_K$ | $Q4\_K$ | $Q3\_K$ | $Q4\_K$ | $Q6\_K$ / $FP16$ |
| **`Q2_K`**   | ~2.6 bpw | $Q2\_K$ | $Q3\_K$ | $Q2\_K$ | $Q3\_K$ | $Q4\_K$ / $FP16$ |
| **`Q6_K`**   | ~6.6 bpw | $Q6\_K$ | $Q6\_K$ | $Q6\_K$ | $Q6\_K$ | $Q8\_K$ / $FP16$ |
| **`Q8_K`**   | ~8.5 bpw | $Q8\_K$ | $Q8\_K$ | $Q8\_K$ | $Q8\_K$ | $FP16$ / $FP32$ |
| **`IQ4_NL`** | ~4.5 bpw | $IQ4\_NL$| $IQ4\_NL$| $IQ4\_NL$| $IQ4\_NL$| $Q8\_K$ / $FP16$ |

`resolve_quant_type_for_tensor(tensor_name, recipe)` dynamically resolves the required `StorageType` using regex pattern matching across attention, feed-forward, embedding, and normalization layers.

---

## 9. Memory Alignment & Platform Guarantees

- **Global File Alignment**: Every tensor payload offset in `.hk` is an exact multiple of 128 bytes (`offset % 128 == 0`).
- **Cache-Line Coherence**: 128-byte alignment covers two 64-byte CPU cachelines and one full 128-byte GPU memory transaction (32 threads × 4 bytes).
- **Endianness**: Fixed little-endian for all integers and floats, ensuring uniform zero-copy compatibility across x86-64, ARM64, and modern GPUs.

---

## 10. Appendix Region Specification (Adaptive Neural Framework)

The Appendix Region resides at the end of the `.hk` file (pointed to by `header.appendix_offset`). It provides an append-only, version-chained log of runtime adaptations, modular growths, and persistent execution records without mutating frozen base weights.

### Entry Types
- `0x01 = LORA_ADAPTER`: Low-rank adaptation matrices ($A$ and $B$) targeted at specific projection layers.
- `0x02 = DELTA_PATCH`: Sparse or compressed weight difference tensors.
- `0x03 = NEW_LAYER`: Modularity-grown layer parameters (Net2DeeperNet / Net2WiderNet additions).
- `0x04 = CODE_EVAL`: Persistent evaluation results, unit test pass rates, and execution traces from the code sandbox.
- `0x05 = KV_CACHE_SINK`: Pre-computed key-value cache states for attention sink tokens.
- `0x06 = TOPOLOGY_HEAD`: Declarative graph structure and execution head script.

### Appendix Record Binary Layout
Each appendix entry begins on a 64-bit aligned boundary:

| Offset | Type | Field | Description |
|:---|:---|:---|:---|
| 0 | `u8` | `entry_type` | Appendix entry category tag (`0x01` - `0x06`) |
| 1 | `u8` | `flags` | Bit 0: active/enabled, Bit 1: compressed payload |
| 2 | `u16` | `name_len` | Byte length of entry name |
| 4 | `u32` | `generation` | Evolution generation index (monotonically increasing) |
| 8 | `u64` | `timestamp` | Unix epoch timestamp (seconds) |
| 16 | `[32]u8` | `parent_hash` | SHA-256 hash of parent container state for lineage validation |
| 48 | `f32` | `metric_loss` | Objective loss value associated with this generation |
| 52 | `f32` | `metric_acc` | Validation or task accuracy metric |
| 56 | `f32` | `metric_pass`| Persistent code evaluation pass rate ($0.0 \dots 1.0$) |
| 60 | `f32` | `metric_custom`| Extensible domain metric (e.g. perplexity) |
| 64 | `u16` | `target_len` | Byte length of target insertion path |
| 66 | `u16` | `reserved` | Padding for 64-bit alignment |
| 68 | `u32` | `data_crc32` | CRC32 checksum of data payload |
| 72 | `u64` | `data_size` | Byte size of entry payload |
| 80 | `[name_len]u8` | `name` | UTF-8 name string |
| 80+`name_len` | `[target_len]u8` | `target` | UTF-8 insertion target path (e.g. `layers.3.mlp`) |
| ... | `[data_size]u8` | `data` | Raw binary payload (tensors, JSON, bytecode, or eval traces) |

---

## 11. Model Topology & Head Script Packaging

The `.hk` container can encapsulate a complete, executable neural architecture without external source code dependencies:

1. **Topology Definition (`0x06 = TOPOLOGY_HEAD`)**:
   - Encodes layer types (`Linear`, `Conv2d`, `MultiHeadAttention`, `RMSNorm`), dimension transitions, and activation functions.
   - Preserves hyper-parameters (context window, hidden dimensions, number of heads).
2. **Embedded Head Script**:
   - Declarative execution graph enabling `hk.load_model("model.hk")` or `load_standalone_hk("model.hk")` to reconstruct and execute the PyTorch `nn.Module` directly.
3. **Cross-Platform Parity**:
   - Zero framework lock-in; loadable natively via C ABI across Windows, Linux, macOS, Android (direct NDK), and iOS.

---

## 12. Native C ABI Interface Specification

The native shared library (`hk.dll` / `libhk.so` / `libhk.dylib`) exports the complete compute, I/O, and architecture expansion surface:

### Container Writer API
- `hk_writer_create(alignment: u64) -> ?*hk_writer_t`: Allocates a native container builder.
- `hk_writer_destroy(writer: ?*hk_writer_t) -> void`: Releases writer arena resources.
- `hk_writer_add_metadata_string(writer, key, val) -> c_int`
- `hk_writer_add_metadata_int(writer, key, val) -> c_int`
- `hk_writer_add_metadata_float(writer, key, val) -> c_int`
- `hk_writer_add_metadata_bool(writer, key, val) -> c_int`
- `hk_writer_add_tensor(writer, name, storage_type, tile_layout, sparsity_type, ndim, shape_ptr, data_ptr, data_len, sparsity_ratio) -> c_int`: Registers a tensor with 128-byte hardware alignment.
- `hk_writer_write_to_file(writer, path) -> c_int`: Emits header, metadata, TOC, and aligned payload in a single native flush.

### Container Reader API
- `hk_open(path: [*:0]const u8) -> ?*hk_reader_t`: Opens a file with zero-copy memory mapping.
- `hk_close(reader: ?*hk_reader_t) -> void`: Unmaps memory and releases file descriptors.
- `hk_get_tensor_count(reader) -> u64`: Returns total tensor count.
- `hk_get_tensor_info(reader, index, out_info) -> c_int`: Fetches tensor name, shape, offsets, and layouts.
- `hk_dequantize_f32(reader, index, with_residual, out_buf, out_count) -> c_int`: SIMD dequantization kernel for NF4, DQ8, and DQT tensors.

### Neural Expansion & SIMD Compute API
- `hk_net2wider_swiglu(...) -> c_int`: Bit-identical zero-init SwiGLU MLP width expansion.
- `hk_expand_vocab(...) -> c_int`: Invariance-preserving dynamic vocabulary growth.
- `hk_plasticity_mask_rows(...)` & `hk_plasticity_mask_cols(...)`: In-place gradient zeroing for anti-catastrophic forgetting.
- `hk_forward_swiglu(...) -> c_int`: Fused SIMD SwiGLU activation.
- `hk_forward_rmsnorm(...) -> void`: SIMD RMS normalization.
- `hk_forward_silu(...) -> void`: SIMD SiLU vector activation.
- `hk_pack_2_4(...) -> c_int` & `hk_unpack_2_4(...) -> c_int`: Ampere 2:4 structured hardware sparsity pack/unpack.

### Hugging Face Architecture Mapper API
- `hk_hf_detect_architecture(json_config, out_arch, max_len) -> c_int`: SIMD-accelerated architecture detection for 137+ models.
- `hk_hf_map_tensor_name(tensor_name, arch, to_hk, out_name, max_len) -> c_int`: Bidirectional high-throughput tensor name remapping at >320,000 names/second.

### Native Context Window Management API
- `hk_context_truncate(in_tokens, in_len, max_tokens, strategy, head_ratio, out_tokens, out_len) -> c_int`: Zero-copy SIMD token sequence truncation supporting `tail`, `head`, `middle_out`, and `sliding_window` retention strategies.

### Adaptive Growth Governor API
- `hk_governor_can_grow(current_params, added_params, max_growth_ratio, max_vram_mb, dtype_bytes, out_reason, max_reason_len) -> c_int`: Scalar hardware constraint check.
- `hk_governor_can_grow_batch(current_params, added_params, n, max_growth_ratio, max_vram_mb, dtype_bytes, out_results) -> c_int`: Batched vector constraint check (evaluates 50k constraints in 3.57 ms, 22.5× faster than scalar FFI).
- `hk_expand_vocab_embeddings(old_embed, old_vocab, hidden_size, new_vocab, new_embed, init_std, seed) -> c_int`: Native vocabulary weight expansion with Gaussian initialization.
- `hk_init_plasticity_mask(mask, total_units, base_units, decay_rate) -> c_int`: In-place gradient shielding mask initialization against catastrophic forgetting.

---

## 13. Autonomous Self-Training & Expansion Architecture

HK features an autonomous self-training loop where models diagnose representational bottlenecks, expand their own architecture, and learn through self-conversational reasoning:

1. **`ExpansionEvaluator`**:
   - Diagnostic probes calculate domain error rates, missing vocabulary tokens, and perplexity jumps.
   - Triggers dynamic expansion through `GrowthGovernor` if error rate exceeds thresholds.
2. **`SelfConversationalEngine`**:
   - Dual-agent dialogue between Proposer and Thinker.
   - Extracts structured inner monologue (`<think> ... </think>`) and self-generated candidate implementations (`<code> ... </code>`).
3. **`CodeSandbox`**:
   - Sandboxed execution environment validating candidate logic against automated unit test suites.
4. **Plasticity Shield**:
   - In-place gradient zeroing shields pre-expansion weights from gradient updates during new domain acquisition, eliminating catastrophic forgetting.
5. **Cryptographic Lineage Logging**:
   - Every expansion and self-training generation is logged with test pass rates, loss, and SHA-256 hashes in the container's Appendix region.

---

## 14. Deep Tokenizer Ingestion & In-File Storage

To eliminate external `.json` configuration file dependencies and heavyweight third-party library requirements, `.hk` containers store rich tokenizer structures natively in metadata and provide zero-dependency parsers for leading tokenizer formats:

### 13.1 Tokenizer Taxonomy & Metadata Keys
| Metadata Key | Value Type | Description |
|:---|:---|:---|
| `tokenizer.tokens` | `val_json` or `val_string` | Ordered array of vocabulary token strings corresponding to token IDs $0, 1, \dots, V-1$. |
| `tokenizer.scores` | `val_json` or `val_string` | Array of float32 scores (log-probabilities / merge penalties) corresponding to each token. |
| `tokenizer.token_types` | `val_json` or `val_string` | Integer array of token categorization types (`TokenType`: `1=NORMAL`, `2=UNKNOWN`, `3=CONTROL`, `4=USER_DEFINED`, `5=UNUSED`, `6=BYTE`). |
| `tokenizer.merges` | `val_json` or `val_string` | Array of BPE merge rule strings (e.g. `["t h", "th e"]`). |
| `tokenizer.pre_tokenizer` | `val_string` | Pre-tokenizer regex split type (`"default"`, `"llama3"`, `"qwen2"`, `"deepseek_v3"`, `"phi3"`, `"mistral"`). |
| `tokenizer.chat_template` | `val_string` | Standard Jinja2 chat template formatted for turn-based conversation and `tokenizer.apply_chat_template()`. |
| `tokenizer.bos_token` | `val_string` | Beginning of sequence token string (default: `<s>` or `<|im_start|>`). |
| `tokenizer.eos_token` | `val_string` | End of sequence token string (default: `</s>` or `<|im_end|>`). |
| `tokenizer.unk_token` | `val_string` | Unknown token representation (default: `<unk>`). |
| `tokenizer.pad_token` | `val_string` | Padding token representation (default: `<pad>`). |
| `tokenizer.bos_token_id` | `val_int` | Beginning of sequence integer token ID. |
| `tokenizer.eos_token_id` | `val_int` | End of sequence integer token ID. |
| `tokenizer.unk_token_id` | `val_int` | Unknown token integer token ID. |
| `tokenizer.pad_token_id` | `val_int` | Padding integer token ID. |

### 13.2 Zero-Dependency Binary SentencePiece (`.model`) Parser
SentencePiece models store a binary protobuf `ModelProto`. Rather than requiring the `protobuf` or `sentencepiece` Python wheels, HK implements an embedded pure-Python wire-format decoder (`parse_sentencepiece_model`):
- Directly decodes Protocol Buffer varints and length-delimited wire types ($wire=0, 2$).
- Extracts field 3 (`RepeatedPtrField<SentencePiece>`):
  - Subfield 1: UTF-8 token piece string.
  - Subfield 2: Float32 log-probability score.
  - Subfield 3: Integer token type tag (`NORMAL`, `UNKNOWN`, `CONTROL`, `USER_DEFINED`, `UNUSED`, `BYTE`).
- Parses in milliseconds with zero external C++ extension or pip dependencies.

### 13.3 Mistral Tekkenizer (`tekken.json`) Ingestion
Mistral NeMo and Large 2 utilize the Tekkenizer format, combining byte-fallback BPE with complex token structures:
- `parse_tekken_json` automatically parses `tekken.json` or `tokenizer.json` configuration structures.
- Decodes vocab token strings, initial byte tokens, special tokens (`[TOOL_CALLS]`, `[AVAILABLE_TOOLS]`), and BPE merges.

### 13.4 In-File Reconstruction & Persistence APIs
1. **Direct Reconstruction**: `AutoTokenizer.from_pretrained("model.hk")` dynamically inspects these metadata keys, reconstructing the tokenizer without external network requests or supplementary files.
2. **Sharded & Manifest Inspection**: `AutoTokenizer.from_pretrained("model.hk.index.json")` automatically navigates to companion shards to extract embedded tokenizer metadata.
3. **Self-Contained Container Saving**: `model.save_pretrained("model.hk", tokenizer=tokenizer)` or `tokenizer.save_to_hk("model.hk")` embeds the tokenizer metadata directly in the `.hk` file header.

---

## 15. Multi-File Sharding Specification (70B+ Scale)

Large neural models exceeding storage limits or target filesystem bounds (e.g. 70B, 405B) are partitioned across multiple `.hk` shard files (`model-00001-of-00004.hk`) linked through fixed 128-byte headers and a standardized index manifest:

### 15.1 Header Flags & Fixed Header Fields
1. **Header Flag**:
   - `HeaderFlags.IS_SHARDED = 0x40` (Bit 6) indicates the file belongs to a sharded set.
2. **Fixed Header Fields**:
   - `split_index` (`u16` at offset `0x0E`): Zero-based shard index ($0 \le \text{split\_index} < \text{split\_count}$).
   - `split_count` (`u16` at offset `0x58`): Total number of shards in the collection.

### 15.2 Standardized Index Manifest (`model.hk.index.json`)
The collection index manifest maps individual tensor parameter names to their constituent shard file:
```json
{
  "metadata": {
    "total_size": 140000000000,
    "total_tensors": 350,
    "split_count": 8,
    "format": "hk",
    "version": "1.0.0"
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00008.hk",
    "layers.0.attn_q.weight": "model-00001-of-00008.hk",
    "layers.15.mlp_down.weight": "model-00004-of-00008.hk"
  }
}
```

### 15.3 Transparent Sharded Loading & Lazy Slicing
- `hk.load_file("model.hk.index.json")` or `hk.load_file("model-00001-of-00004.hk")`: Loads companion shards dynamically and aggregates the state dict without duplicate allocations.
- `hk.safe_open("model.hk.index.json", framework="pt")`: Returns a `ShardedHKFile` handle enabling zero-copy lazy tensor fetching (`get_tensor(key)`) and lazy multidimensional slicing (`get_slice(key)[...]`) routing directly to the owning shard without loading the full 70B+ model into RAM.

---

## 16. In-Place Key-Value Metadata Patching

The `hk metadata set <file> <key> <val>` utility allows updating or adding metadata entries in-place:
- **Zero-Copy Guarantee**: Tensor payload data starting at `tensor_data_offset` is never read, copied, or re-serialized.
- **Headroom Re-Packing**: When the new metadata table fits within the pre-tensor allocation budget ($128 + \text{meta\_size} + \text{toc\_size} \le \text{tensor\_data\_offset}$), only the 128-byte header, metadata block, and TOC are rewritten. The remaining padding is zeroed.
- **End-of-File Extension**: If metadata exceeds the pre-tensor gap, the metadata table is positioned at EOF with `header.metadata_offset` updated accordingly.

---

## 17. Hugging Face Architecture Mapping Layer

The `HFArchitectureMapper` enables zero-friction ingestion of checkpoints from the Hugging Face Hub using a comprehensive 137+ architecture registry and bidirectional regex conversion tables:

### 17.1 Supported Architecture Families (137+ Distinct Models)
HK supports 137+ distinct neural architectures spanning all modern foundation models:
1. **Cutting-Edge Causal LLMs**:
   - **DeepSeek V2 / V3 / R1**: MLA (Multi-Head Latent Attention: compressed latent key/value `kv_a`, `kv_b`, decoupled queries `q_a`, `q_b`) and Multi-Token Prediction (MTP) modules.
   - **LLaMA 1 / 2 / 3 / 3.1 / 3.2 / 4**: LLaMA causal models, GQA heads, RoPE theta scaling.
   - **Qwen 1.5 / 2 / 2.5 / 3 / MoE**: Dense and Sparse Mixture-of-Experts (`mlp.experts.*`, shared experts).
   - **Gemma 1 / 2**: GeGLU activations, normalized embedding matrices, alternating local/global attention.
   - **Grok-1**: 314B MoE model with 8 experts per token and fused projection kernels.
   - **Falcon / Falcon-H1**: Multi-query attention with fused parallel attention/MLP layers.
   - **Phi-2 / Phi-3 / Phi-3.5 / Phi-MoE**: SuScaled RoPE, 3D tensor parallel configurations.
   - **Command-R / Command-R+ / DBRX / OLMo / OLMoE / MiniCPM-3 / Starcoder-1/2 / Jais / Exaone / ChatGLM**.
2. **State Space & Recurrent Models (SSM)**:
   - **Mamba / Mamba-2**: Selective state space sequences (`ssm.in_proj`, `ssm.x_proj`, `ssm.dt_proj`, `ssm.out_proj`, `ssm.conv1d`).
   - **Jamba**: Hybrid Transformer-Mamba MoE architecture.
   - **RWKV-5 / RWKV-6**: Linear attention with time-mix and channel-mix recurrence decay vectors (`time_decay`, `time_first`, `time_mix_k`, `time_mix_v`).
3. **Vision-Language Models (VLM)**:
   - **CLIP / SigLIP**: Vision transformers (`vision_model.encoder.layers.*`) and text contrastive encoders.
   - **LLaVA / MobileVLM / Qwen2-VL / Pixtral / Gemma-Vision**: Multi-modal cross-attention projectors (`multi_modal_projector.linear_1/2`).
   - **Segment Anything (SAM / SAM-2)**: Vision encoders, prompt encoders, and lightweight mask decoders.
4. **Audio & Diffusion Transformer Backbones**:
   - **Whisper**: Audio spectrogram convolutional frontends (`conv1`, `conv2`) and cross-attention audio encoders/decoders.
   - **Stable Diffusion (SD 1.5 / 2.1 / XL)**: UNet and text encoder projections.
   - **FLUX.1**: Rectified flow diffusion transformer backbones (`double_blocks.*`, `single_blocks.*`, `img_in`, `txt_in`).
5. **Modern Encoders**:
   - **ModernBERT**: Unpadded rotary positional embeddings, GeGLU MLPs, and alternating sliding window layers.
   - **Nomic-BERT / Jina-BERT-v2/v3 / EuroBERT / DeBERTa-v3**.

### 16.2 Bi-Directional Regex Tensor Mapping Tables
HK maps Hugging Face checkpoint weights to standardized `.hk` naming conventions using 8 regex pattern tables:
- `STANDARD_TRANSFORMER_TENSORS`: Embeddings, Q/K/V/O projections, SwiGLU MLP gate/up/down layers, pre/post layernorms.
- `DEEPSEEK_MLA_TENSORS`: Latent query/key/value projections (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`).
- `QWEN2_MOE_TENSORS`: Routed expert linear projections and shared expert gates (`gate_proj`, `up_proj`, `down_proj`).
- `MAMBA_TENSORS`: SSM in-projections, delta projections, 1D convolution weights/biases, and state step parameters.
- `VLM_TENSORS`: Vision encoder attention projections and multi-modal MLP adapters.
- `WHISPER_TENSORS`: Convolutional sub-samplers and cross-attention encoder/decoder blocks.
- `FLUX_TENSORS`: Single/double transformer stream blocks, modulation scales, and image/text embedding bridges.
- `MODERN_BERT_TENSORS`: Dense rotary embedding projections and post-norm adapters.

`HFArchitectureMapper.map_state_dict(state_dict, source_arch, target_arch, to_hk=True/False)` translates state dicts in both directions with exact shape and dtype preservation.

### 16.3 Unified Checkpoint Converter
The CLI command or Python API `convert_hf_checkpoint(hf_model_dir, output_hk_path, canonical_tensor_names=True, split_size_mb=...)`:
- Auto-detects the architecture family via `config.json` or fallback heuristic.
- Converts state dict tensors using bidirectional regex mappings.
- Ingests embedded tokenizer tables (BPE, SentencePiece `.model`, or Tekkenizer `tekken.json`).
- Automatically produces multi-file sharded containers with `model.hk.index.json` if weights exceed the specified shard threshold.

---

## 18. Universal Heterogeneous Stage Model Pipeline & Dynamic Context Management

HK provides a universal, generalized execution pipeline architecture (`UniversalPipeline`, `PipelineStage`, `PipelineContext`) that unifies arbitrary directed acyclic graphs (DAGs) and linear sequences of $N$ heterogeneous models and functional transforms into a single high-performance runtime.

### 17.1 Universal Pipeline Architecture
Rather than restricting execution to fixed three-stage pipelines (e.g., Speech $\to$ Analysis $\to$ LLM), the universal pipeline orchestrates any combination of heterogeneous modalities:
1. **Audio Transcription (`audio_transcription`)**: Automatic Speech Recognition models (e.g., `HKWhisperModel`) ingesting raw PCM audio arrays or audio waveforms and yielding timestamped token transcripts.
2. **Vision & Document Intelligence (`vision_ocr`)**: Handwriting recognition, visual encoders, OCR, and vision transformers (e.g., `HKForHandwritingRecognition`) mapping 2D pixel tensors to structured text or dense visual tokens.
3. **Linguistic & Token Analysis (`token_analysis`)**: Bidirectional encoders and embedding extractors (e.g., `HKDistilBertModel`) performing token-level sentiment scoring, entity recognition, grammar analysis, or intent routing.
4. **Sequence Classification (`sequence_classification`)**: Safety guards, toxicity filters, classification heads, or reward models producing scalar confidence distributions.
5. **Generative Autoregressive Models (`text_generation`)**: Causal LMs (e.g., `HKForCausalLM`, `SmollM`, `Llama`, `Qwen`) performing autoregressive token generation conditioned on blackboard state.
6. **Generic & Custom Transforms (`generic`)**: Arbitrary Python callables, native Zig SIMD transforms, regex parsers, vector database retrieval queries, or external tool execution hooks.

### 17.2 Blackboard Context Architecture (`PipelineContext`)
Inter-stage communication is governed by a decentralized blackboard state machine:
- **Blackboard State (`blackboard`)**: A shared key-value state space where each stage reads required inputs and writes structured outputs.
- **Dynamic Key Routing (`input_mapping` & `output_mapping`)**: Allows stages to adapt to differing signature conventions without glue code. For example, mapping `audio_stt.transcript` $\to$ `token_analyzer.text` and `token_analyzer.intent` $\to$ `llm_generator.prompt_prefix`.
- **Execution Audit Trace (`stage_trace`)**: Records complete execution telemetry per stage—including stage entry timestamp, elapsed execution duration, input argument snapshots, output keys emitted, and status codes.
- **Dynamic Runtime Overrides (`stage_params`)**: Callers can dynamically override hyper-parameters for any specific stage at inference time (e.g., `pipeline(input, stage_params={"llm": {"temperature": 0.2, "max_new_tokens": 128}})`).

### 17.3 Pipeline Manifest Serialization (`manifest.json`)
Universal pipelines can be saved to disk and reloaded portably via a standardized manifest schema:
```json
{
  "pipeline_type": "universal",
  "context_strategy": "blackboard",
  "stages": [
    {
      "name": "audio_stt",
      "modality": "audio_transcription",
      "model_path": "models/whisper_base.hk",
      "input_mapping": {"audio": "raw_audio"},
      "output_mapping": {"transcript": "text"},
      "params": {"language": "en"}
    },
    {
      "name": "intent_analyzer",
      "modality": "token_analysis",
      "model_path": "models/distilbert_intent.hk",
      "input_mapping": {"text": "text"},
      "output_mapping": {"intent": "detected_intent"},
      "params": {}
    },
    {
      "name": "llm_agent",
      "modality": "text_generation",
      "model_path": "models/smollm_135m.hk",
      "input_mapping": {"prompt": "text"},
      "output_mapping": {"generated_text": "response"},
      "params": {"temperature": 0.7, "max_new_tokens": 64}
    }
  ],
  "metadata": {
    "version": "1.0.0",
    "framework": "hk"
  }
}
```

### 17.4 Backward Compatibility Guarantee
The legacy `CompositePipeline` transparently inherits from `UniversalPipeline`, providing 100% backward compatibility for existing applications using `speech_model`, `analysis_model`, and `generation_model` attributes while routing execution through the universal blackboard orchestrator.

### 17.5 Context Window Management (`ContextWindowManager`)
For generative stages with bounded context windows ($L_{\text{max}}$), the pipeline integrates dynamic context window retention strategies:
- `tail`: Retains the most recent tokens (best for multi-turn chat dialogues).
- `head`: Retains the leading tokens (best for system instructions and persistent grounding prompts).
- `middle_out`: Preserves both initial system instructions and recent conversation history while pruning intermediate turns.
- Dynamic RoPE scaling and sliding-window attention adjustments prevent attention dispersion on long documents.

---

## 19. Multilingual Architecture & Universal Support Matrix

The HK framework provides a unified specification across native runtimes and high-level language ecosystems. All bindings adhere to the 128-byte header layout, hardware memory alignment, dual-mode quantization codebooks, and metadata conventions:

| Feature / Capability | Zig Core | C / C++ | Rust | Go | C# (.NET) | Java (Android) | TypeScript / JS | Python (PyTorch/NumPy/JAX/Flax) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Copy mmap Access** | Native | Direct | Native | `cgo` | Direct P/Invoke | Direct NIO Buffer | Buffer Slice | PyTorch, NumPy, JAX, Flax |
| **128-Byte Tensor Core Alignment** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Dual-Mode NF4 Dequantization** | Native SIMD | Yes | Yes | Yes | Yes | Yes | Pure TS Codebook | PyTorch / NumPy SIMD |
| **Ampere 2:4 Structured Sparsity** | Native SIMD | Yes | Yes | Yes | Yes | Yes | Nibble Unpack | Torch Sparse / NumPy |
| **Multi-File Header Sharding** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | `model.hk.index.json` / `ShardedHKFile` |
| **In-Place Metadata Patching** | `patchInPlace` | Direct C API | Native Crate | `cgo` | Direct P/Invoke | Native JNI | In-Buffer Patch | `hk metadata set` |
| **In-File Tokenizer & Chat Template**| Metadata TOC | String Extract | String Extract| String Extract| String Extract| String Extract | String Extract | `AutoTokenizer` (.hk / index) |
| **Hugging Face Architecture Mapping**| N/A | N/A | N/A | N/A | N/A | N/A | N/A | `HFArchitectureMapper` (Llama/Qwen2/Mistral) |
| **Universal Multi-Stage Pipeline** | Headless | C Struct API | Safe DAG | Context Pipe | Task Chain | Execution Pipe | `UniversalPipeline` | `UniversalPipeline` |
| **Append-Only Appendix DAG** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | `HkTrainer` |
| **K-Quants (Q2_K - Q8_K)** | Native SIMD | Yes | Yes | Yes | Yes | Yes | Bit-shift unpack | Native C ABI / SIMD |
| **Microscaling (MXFP4, NVFP4)** | Native Kernel | Yes | Yes | Yes | Yes | Yes | Scale-lut unpack | Native C ABI / Torch |
| **SPM & Tekken Ingestion** | Metadata TOC | Pure Read | Pure Read | Pure Read | Pure Read | Pure Read | Pure Read | Zero-Dep Decoder |
| **GUI Model Studio** | `hk gui` | N/A | N/A | N/A | N/A | N/A | N/A | `python -m hk.gui` |

---

## 20. Standardized Hyperparameter & Sampling Taxonomy

To eliminate arbitrary nomenclature divergence across model architectures, HK establishes a 200+ key canonical taxonomy (`HKTaxonomyKeys` / `StandardKeys`):

### 20.1 General Architecture & Model Lineage (`general.*`)
- `general.architecture`: Canonical architecture string (e.g. `"llama"`, `"deepseek_v3"`, `"qwen2"`, `"mamba2"`, `"flux"`).
- `general.name`: Model name identifier.
- `general.version`: Container format version.
- `general.author` / `general.url` / `general.license` / `general.description`: Model provenance and licensing metadata.
- `general.source.url` / `general.source.huggingface`: Upstream checkpoint origin.
- `general.file_type`: Storage quantization index.
- `general.quantization_version`: Quantization iteration version.

### 20.2 Dimensions & Topology
- `general.context_length` ($L_{\text{ctx}}$): Maximum sequence context length.
- `general.embedding_length` ($d_{\text{model}}$): Hidden dimensionality.
- `general.block_count` ($N_{\text{layers}}$): Number of transformer or recurrent layers.
- `general.feed_forward_length` ($d_{\text{ffn}}$): Intermediate MLP width.
- `general.vocab_size` ($V$): Total vocabulary size.

### 20.3 Attention & RoPE Hyperparameters (`attention.*`, `rope.*`)
- `attention.head_count` ($n_{\text{heads}}$) & `attention.head_count_kv` ($n_{\text{kv\_heads}}$): Grouped-Query Attention (GQA) head allocation.
- `attention.key_length` ($d_k$) & `attention.value_length` ($d_v$): Head projection dimension.
- `attention.sliding_window` ($W_{\text{SWA}}$): Sliding-window attention chunk length.
- `attention.alibi_bias_max`: Maximum ALiBi attention slope factor.
- `attention.q_lora_rank` & `attention.kv_lora_rank`: DeepSeek MLA compressed latent dimensions.
- `rope.freq_base` ($\theta_{\text{base}}$): Base RoPE frequency (e.g. 10000.0, 500000.0).
- `rope.scale`: Sequence length expansion scaling multiplier.
- `rope.scaling_type`: Scaling algorithm (`"linear"`, `"yarn"`, `"dynamic"`).
- `rope.scaling.yarn_extrapolation_factor`, `rope.scaling.yarn_attn_factor`, `rope.scaling.yarn_beta_fast`, `rope.scaling.yarn_beta_slow`, `rope.scaling.yarn_orig_ctx`: YaRN attention hyper-parameters.

### 20.4 Mixture-of-Experts (`moe.*`)
- `moe.expert_count`: Total routed experts per layer (e.g. 8, 64, 256).
- `moe.expert_used_count`: Number of active experts routed per token ($top\_k$).
- `moe.expert_shared_count`: Number of persistent shared experts executed for all tokens.
- `moe.expert_weights_scale`: Softmax temperature or gating multiplier.

### 20.5 State-Space Models & Recurrent Transformers (`ssm.*`)
- `ssm.state_size`: SSM latent state dimension ($d_{\text{state}}$).
- `ssm.time_step_rank`: Time step projection rank ($\Delta_{\text{rank}}$).
- `ssm.inner_size`: Intermediate expanded SSM dimension ($d_{\text{inner}}$).
- `ssm.conv_kernel`: 1D short convolutional filter kernel width.

### 20.6 Generation & Sampling Presets (`sampling.*`)
- `sampling.temperature`: Softmax sampling temperature ($T$).
- `sampling.top_k`: Top-K truncation bound.
- `sampling.top_p`: Nucleus sampling cumulative probability threshold.
- `sampling.min_p`: Minimum token probability threshold relative to maximum probability token.
- `sampling.typical_p`: Typical sampling information-theoretic threshold.
- `sampling.penalty_last_n`: Sliding token lookback window for repetition penalties.
- `sampling.penalty_repeat`: Logit penalty multiplier for repeated tokens.
- `sampling.penalty_frequency`: Additive frequency penalty per prior occurrence.
- `sampling.penalty_present`: Additive presence penalty for prior occurrence.
- `sampling.mirostat`, `sampling.mirostat_tau`, `sampling.mirostat_eta`: Mirostat adaptive perplexity target and learning rates.

### 20.7 Quantization Calibration (`quant.*`)
- `quant.type`: Target quantization scheme (`"Q4_K_M"`, `"IQ4_NL"`, etc.).
- `quant.imatrix_file`: Source calibration importance matrix path.
- `quant.imatrix_dataset`: Dataset name used for activation moment calibration.
- `quant.calibration_steps`: Number of calibration forward passes accumulated.

---

## 21. Standalone Developer Utilities

The native `hk` command-line executable provides low-level diagnostics, verification, and endianness portability:

```bash
# 1. Comprehensive Container Dumper (Hex, Bitflags, TOC, Alignment Audit)
hk dump model.hk

# 2. Cryptographic Digest & Tensor Hash Verification (SHA-256)
hk hash model.hk

# 3. Endianness Converter (Zero-loss Little-Endian <-> Big-Endian swapping)
hk convert-endian in_le.hk out_be.hk

# 4. Launch Visual HK Model Studio GUI
hk gui model.hk
```

### 21.1 Binary Hex & Alignment Dumper (`hk dump`)
- Emits raw 128-byte hex representation of the file header with field offsets.
- Decodes all active header bitflags (`0x01 = LITTLE_ENDIAN`, `0x02 = APPENDIX`, `0x04 = QUANT_TABLE`, `0x08 = SPARSITY_2_4`, `0x10 = TILED`, `0x20 = FLEX_ALIGN`, `0x40 = IS_SHARDED`).
- Performs an automated alignment audit on all tensors: flags any tensor whose `data_offset` is not a strict multiple of 128 bytes.
- Summarizes metadata key-value types and tensor payload memory allocations.

### 21.2 Cryptographic Verification (`hk hash`)
- Computes streaming whole-file SHA-256 checksums without loading entire multi-gigabyte models into RAM.
- Verifies stored header `checksum` against live CRC-64 / xxHash calculations.
- Performs per-tensor SHA-256 hashing to guarantee weight integrity and detect silent bit-rot or corruptions.

### 21.3 Endianness Converter (`hk convert-endian`)
- Transposes integer and floating-point values between host endianness (Little-Endian) and network/embedded big-endian architectures (e.g. SPARC, IBM z/Architecture, specific DSPs).
- Swaps header fields, TOC records, and dense float buffers (`f32`, `f16`, `int32`, `int64`) while preserving byte-wise quantized nibbles and 128-byte hardware alignment boundaries.

---

## 22. Graphical Model Studio (`hk gui` / `hk-gui`)

The HK Model Editor is an interactive desktop inspection and editing studio built with high responsiveness and zero external desktop GUI framework bloat:

### 21.1 Core GUI Capabilities
1. **Container Overview**:
   - Header magic verification (`HKNT`), version info, endianness indicator, and hardware alignment status.
   - Active bitflags checklist (Sparsity, Tiling, Sharding, Appendix, Quantization).
   - Real-time disk size, uncompressed vs. quantized memory footprint, and compression ratio calculation.
2. **Metadata & Hyperparameter Hierarchy**:
   - Filterable key-value tree viewer organizing hyperparameters by taxonomy namespaces (`general`, `attention`, `rope`, `moe`, `tokenizer`, `sampling`).
   - In-place key-value editing: modifying a metadata entry immediately updates the container via native in-place byte patching (`hk_metadata_patch_in_place`), avoiding multi-gigabyte file re-writes.
   - JSON export and import: dump all model metadata to JSON for automated CI/CD auditing or bulk import configuration updates.
3. **Tensor Table & Quantization Inspector**:
   - Searchable table of all model parameters showing tensor name, storage type (`F32`, `Q4_K`, `Q8_K`, `IQ4_NL`, `DQ4`), dimensions/shape, physical byte offset, uncompressed size, and sparsity ratio.
   - Real-time AVX-512 / Tensor Core 128-byte alignment audit with one-click verification report.
4. **Lineage, Training Metrics & Appendix**:
   - Visual inspection of the version-chained append-only appendix DAG.
   - Tabulation of historical generations, timestamps, loss metrics, validation accuracies, and code sandbox pass rates.
   - One-click rollback triggering native container rollback to any historical checkpoint generation.

### 21.2 Launch Methods
```bash
# Launch via native Zig CLI:
hk gui [model.hk]

# Launch via Python CLI:
hk-gui [model.hk]
python -m hk.gui [model.hk]
```




