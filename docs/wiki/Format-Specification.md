# HK Binary Format Specification (.hk)

In this document, I provide a detailed breakdown of the HK binary container format (`.hk`) that I designed.

---

## 1. High-Level File Layout

An `.hk` file is a single contiguous binary container organized into five primary sections:

```
+-------------------------------------------------------------+
| HK File Header (Fixed 128 bytes, cacheline aligned)         |
+-------------------------------------------------------------+
| Metadata Key-Value Section (Binary MsgPack-like format)     |
+-------------------------------------------------------------+
| Tensor Table of Contents (TOC) (Fixed 128 bytes per entry)  |
+-------------------------------------------------------------+
| Alignment Padding (Pads to alignment boundary: 128B / 4KB)  |
+-------------------------------------------------------------+
| Tensor Payloads (Zero-copy memory mapped raw arrays)        |
|   - Tensor 0 (aligned to hardware boundary)                 |
|   - Tensor 1 ...                                            |
|   - Tensor N ...                                            |
+-------------------------------------------------------------+
| Appendix Region (Optional append-only version DAG & deltas) |
+-------------------------------------------------------------+
```

---

## 2. File Header (128 Bytes Fixed)

The header sits at offset `0x00` and occupies exactly 128 bytes. This aligns with GPU cache lines and hardware memory controller read buffers.

| Byte Offset | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `magic` | `[4]u8` | Format identifier: `0x48, 0x4B, 0x4E, 0x54` (`"HKNT"`). |
| `0x04` | `version_major` | `u16` | Major version (currently `1`). |
| `0x06` | `version_minor` | `u16` | Minor version (currently `0`). |
| `0x08` | `flags` | `u32` | Bitflags for container features (see Header Flags below). |
| `0x0C` | `alignment` | `u16` | Hardware memory boundary in bytes (128, 4096, 16384, or 65536). |
| `0x0E` | `split_index` | `u16` | Multi-file sharding: index of this shard (0-based). |
| `0x10` | `tensor_count` | `u64` | Total number of tensors stored in this file. |
| `0x18` | `metadata_kv_count` | `u64` | Total number of metadata key-value pairs. |
| `0x20` | `metadata_offset` | `u64` | Byte offset where metadata section begins. |
| `0x28` | `metadata_size` | `u64` | Total byte size of the metadata section. |
| `0x30` | `tensor_toc_offset` | `u64` | Byte offset where the Table of Contents begins. |
| `0x38` | `tensor_toc_size` | `u64` | Total byte size of the Table of Contents. |
| `0x40` | `tensor_data_offset`| `u64` | Byte offset where tensor payload data begins (aligned). |
| `0x48` | `appendix_offset` | `u64` | Byte offset to the Appendix DAG (0 if none present). |
| `0x50` | `checksum` | `u64` | CRC-64 or xxHash64 checksum of metadata and TOC. |
| `0x58` | `split_count` | `u16` | Multi-file sharding: total number of shards (>= 1). |
| `0x5A` | `reserved` | `[38]u8` | Reserved for future format extensions. |

### Header Flags (`flags`)
- Bit 0 (`0x01`): Little-endian byte order.
- Bit 1 (`0x02`): Appendix region present at end of file.
- Bit 2 (`0x04`): Quantization scale table present.
- Bit 3 (`0x08`): Ampere 2:4 structured sparsity enabled.
- Bit 4 (`0x10`): Cache-tiled 2D layout enabled.
- Bit 5 (`0x20`): Flexible alignment mode.
- Bit 6 (`0x40`): Multi-file sharded container.
- Bit 7 (`0x80`): Raw weight storage mode (zero compute headroom).
- Bit 8 (`0x100`): Universal page aligned (4KB / 16KB).

---

## 3. Metadata Section

Metadata is stored as a sequence of length-prefixed key-value pairs. Keys are UTF-8 strings. Values can be strings, integers, floats, booleans, or arrays (such as token lists or chat templates).

Because the header stores `metadata_offset` and `metadata_size`, tools like `hk metadata set` can update values in-place in microseconds if the new data fits inside pre-allocated padding, without having to rewrite gigabytes of weights.

Common metadata keys include:
- `general.architecture`: Model family (e.g., `llama`, `qwen2`, `mistral`).
- `general.name`: Model identifier string.
- `tokenizer.chat_template`: Jinja2 chat template string.
- `tokenizer.ggml.tokens`: List of vocabulary tokens.
- `tokenizer.ggml.bos_token_id`: Beginning-of-sequence token ID.
- `tokenizer.ggml.eos_token_id`: End-of-sequence token ID.

---

## 4. Table of Contents (TOC) Entry (128 Bytes Fixed)

Every tensor has an entry in the Table of Contents. Each TOC entry is fixed at 128 bytes, allowing constant-time O(1) index lookups by tensor ordinal.

| Offset | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `name` | `[64]u8` | Null-terminated UTF-8 tensor identifier (e.g. `model.layers.0.mlp.gate_proj.weight`). |
| `0x40` | `storage_type` | `u8` | Data type identifier (see Storage Types below). |
| `0x41` | `tile_layout` | `u8` | Tiling format (`0` = row-major, `1` = 16x16, `2` = 32x16, `3` = 64x64). |
| `0x42` | `sparsity_type` | `u8` | Sparsity scheme (`0` = dense, `1` = 2:4 structured, `2` = bitmask, `3` = CSR). |
| `0x43` | `ndim` | `u8` | Number of dimensions in tensor (1 to 8). |
| `0x44` | `shape` | `[8]u64` | Extents of each dimension (up to 8D). |
| `0x84` | `data_offset` | `u64` | Absolute byte offset to the main tensor payload. |
| `0x8C` | `data_size` | `u64` | Byte size of the main tensor payload. |
| `0x94` | `residual_offset`| `u64` | Byte offset to precision recovery residual stream (or 0). |
| `0x9C` | `residual_size` | `u64` | Byte size of residual stream (or 0). |
| `0xA4` | `scale_offset` | `u64` | Byte offset to block quantization scales (or 0). |
| `0xAC` | `scale_size` | `u64` | Byte size of block quantization scales (or 0). |
| `0xB4` | `block_size` | `u16` | Number of elements per quantization block (e.g. 32 or 256). |
| `0xB6` | `sparsity_ratio`| `f32` | Fraction of zero elements (e.g. 0.50 for 2:4 sparsity). |
| `0xBA` | `reserved` | `[6]u8` | Reserved padding to complete 128-byte alignment. |

---

## 5. Storage Types (`storage_type`)

I classified tensors into three primary families:

### Raw IEEE Floats and Integers (Zero Compute Headroom)
These formats require no dequantization or decoding. SIMD and Tensor Core kernels execute directly on the memory-mapped pointer:
- `0x00`: `f32` (IEEE 754 32-bit single precision)
- `0x01`: `f16` (IEEE 754 16-bit half precision)
- `0x02`: `bf16` (Brain Floating Point 16-bit)
- `0x03`: `fp8_e4m3` (8-bit float, 4-bit exponent, 3-bit mantissa)
- `0x04`: `fp8_e5m2` (8-bit float, 5-bit exponent, 2-bit mantissa)
- `0x05`: `int8` (Signed 8-bit integer)
- `0x06`: `int32` (Signed 32-bit integer)
- `0x07`: `int64` (Signed 64-bit integer)
- `0x08`: `uint8` (Unsigned 8-bit integer)
- `0x09`: `bool` (Boolean byte)
- `0x0A`: `int16` (Signed 16-bit integer)
- `0x0B`: `uint16` (Unsigned 16-bit integer)
- `0x0C`: `uint32` (Unsigned 32-bit integer)
- `0x0D`: `uint64` (Unsigned 64-bit integer)
- `0x0E`: `f64` (IEEE 754 64-bit double precision)

### Dual-Mode Quantization & Edge Formats
- `0x10`: `dq4` (4-bit NF4/INT4 base + block scale + optional residual stream)
- `0x11`: `dq8` (8-bit quantized)
- `0x14`: `dqt` (Ternary {-1, 0, +1} BitNet b1.58)
- `0x20` - `0x27`: GGUF-compatible symmetric quants (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`)
- `0x30` - `0x36`: Super-block K-quants (`q2_k` through `q8_k`)
- `0x40` - `0x47`: Importance matrix I-quants (`iq1_s` through `iq4_nl`)

### Virtual Reference Types (Zero Storage Overhead)
- `0x30`: `NULL_REF`: Represents a pruned or zeroed layer. The tensor metadata exists in the TOC, but `data_size = 0`. Zero bytes are stored on disk.
- `0x31`: `SHARED_REF`: Virtual alias for tied weights (for example, when `lm_head.weight` shares the exact memory address of `embed_tokens.weight`). Its `data_offset` points directly to the target tensor. Zero duplicate bytes are saved to disk.

---

## 6. Appendix DAG Region

The Appendix resides at the end of the `.hk` file (`header.appendix_offset`). I designed it as an append-only audit trail and version DAG that tracks the model's evolutionary history without modifying or duplicating the base model weights.

Each Appendix record stores:
- `record_type`: Type of event (`0x01` = LoRA adapter checkpoint, `0x02` = Net2Net growth receipt, `0x03` = Evaluation metrics, `0x04` = Self-training generation).
- `generation`: Incremental generation index (e.g. `1`, `2`, `3`).
- `parent_hash`: SHA-256 hash of the parent container state, forming a cryptographic chain of custody.
- `step`: Global training step.
- `metrics`: Evaluation metrics (loss, perplexity, sandbox pass rate).
- `payload_size`: Size of attached adapter deltas or growth maps.
- `payload`: Binary delta weights.

Because records are chained chronologically, you can run `hk rollback model.hk --generation 1` to restore the container to an earlier generation in under a millisecond.
