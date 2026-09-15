# Empirical Benchmarks and Performance Analysis

In this document, I share full empirical benchmark measurements comparing HK against standard frameworks (PyTorch, SafeTensors, GGUF, and Hugging Face).

All tests are reproducible on consumer laptops, desktop GPUs, and server hardware.

---

## 1. Production Model Benchmark: Qwen3.5-0.8B (873M bfloat16 Parameters)

I evaluated HK against a real production non-quantized model with 873,438,784 parameters across 488 tensors (1.75 GB total). I compared standard Hugging Face / PyTorch / SafeTensors workflows against my HK Tensor-Optimized Raw Weight Engine (`qwen_0.8b_optimized.hk` with 4096-byte Universal Page Alignment + 128-byte NVIDIA Tensor Core Coalescing):

| Benchmark Metric | Standard Hugging Face / PyTorch | HK Tensor-Optimized Engine | Speedup / Advantage |
| :--- | :--- | :--- | :--- |
| Layer GEMV Compute ($y = W \cdot x$) | 0.37 ms (20.04 GFLOPS) | 0.21 ms (34.38 GFLOPS) | **1.72x faster** (+71.6% compute throughput) |
| Autoregressive Layer Retrieval (Warm) | 28.84 microseconds | 0.08 microseconds (80 ns) | **346.1x faster** (sub-microsecond) |
| Autoregressive Layer Retrieval (Cold) | 506.23 microseconds | 136.07 microseconds | **3.72x faster** |
| Full Model Weight Load (1.75 GB, 488 tensors) | 10.30 ms | 15.88 ms | Pure zero-copy OS memory mapping (`HKDict`) |
| SafeTensors to HK Transcoding Speed | N/A | 222.23 MB/s | Streaming zero-memory generator |

Key Takeaways:
- 4-Row Unrolled SIMD GEMV: `HKRawWeightStore.gemv` computes 4 output rows simultaneously in registers with 8 interleaved 256-bit SIMD accumulators, achieving 34.38 GFLOPS on a single CPU core.
- Sub-Microsecond Layer Access: Once mapped, retrieving layer weight descriptors takes just 80 nanoseconds, providing instant weight access during token-by-token generation.
- Reproducibility: Run this benchmark locally with `python benchmarks/benchmark_1b_model.py`.

---

## 2. Memory-Mapped Loading & Physical Page-In vs Resident Dequantization

This benchmark isolates virtual memory descriptor mapping, physical storage paging, and pure in-memory SIMD compute:

| Loader / Compute Phase | Metric Reported | Measured Latency / Rate | Physical Significance |
| :--- | :--- | :--- | :--- |
| Zero-Copy Header & Deserialization | Metadata & Virtual Map | 15.93 ms (1.75 GB model) | Maps file into user address space (`mmap`/`MapViewOfFile`). |
| Lazy Slice Pointer Resolution | Descriptor Lookup | 23.70 microseconds (20.6 M-slices/s) | Resolves 488 tensor descriptors in user space (0 heap allocations). |
| Physical Memory Traversal (First Touch) | NVMe / Storage Throughput | 2,422 ms (687.8 MB/s) | Actually reads mapped pages via 256-bit SIMD, faulting them into RAM. |
| Physical Memory Traversal (Warm Cache) | System Memory Bus Rate | 1,409 ms (1,182.3 MB/s) | Pure memory bus streaming throughput when pages are resident in RAM. |
| Reconstruction & Dequantization | SIMD Compute Throughput | 213.25 M-elem/s (970 MB scratch) | Executes on warm resident memory using single-tensor scratch buffers. |

Benchmarking Rigor:
I explicitly distinguish between lazy pointer slice resolution (sub-microsecond virtual address setup where memory is not yet paged in) and physical first-touch traversal (where SIMD reads force OS page faults from disk into RAM). Furthermore, I used a reusable per-tensor scratch buffer rather than eagerly allocating multi-gigabyte monolithic arrays, eliminating artificial OS memory zeroing and swap thrashing.

---

## 3. Autonomous Growth Governor Decision Latency

Evaluating hardware boundary safety constraints across 50,000 proposed layer/width capacity expansions:

| Evaluation Engine | Operations Evaluated | Total Latency | Rate | Relative Performance |
| :--- | :---: | :---: | :---: | :---: |
| Python Standard Logic | 50,000 checks | 7.89 ms | 6.34 M-evals/sec | 1.00x |
| HK Native Zig Vectorized Governor | 50,000 checks | 3.57 ms | 14.01 M-evals/sec | **2.21x faster** |

The native growth governor evaluates candidate expansions in 71 nanoseconds per decision, allowing automated training loops to continuously probe expansion feasibility without introducing overhead.

---

## 4. Non-Quantized Loading & Memory-Mapped Slicing (HK vs SafeTensors)

Evaluated across 32 dense full-precision FP32 tensors (224.00 MB total container size) with 50 iterations per operation:

| Reader / Loader Access Pattern | Container Size | Operation | Latency (ms) | Relative Performance |
| :--- | :---: | :--- | :---: | :---: |
| SafeTensors (`load_file`) | 224.00 MB | Full Load (32 tensors) | 1.473 ms | 1.00x |
| HK Native (`load_file`) | 224.00 MB | Full Load Zero-Copy | 0.797 ms | **1.85x faster** |
| SafeTensors (`get_slice` direct) | 224.00 MB | Direct 2D Slice [128, 512] | 0.0149 ms | 1.00x |
| HK Native (`get_slice` direct) | 224.00 MB | Direct 2D Slice [128, 512] | 0.0092 ms | **1.62x faster** |
| SafeTensors (`safe_open` lifecycle) | 224.00 MB | Open + Slice Context | 0.332 ms | 1.00x |
| HK Native (`safe_open` lifecycle) | 224.00 MB | Open + Slice Context | 0.274 ms | **1.21x faster** |

---

## 5. Lossless NVIDIA Ampere 2:4 Structured Sparsity

Bit-exact hardware physical nibble compression without numerical precision loss:

| Operation | Input Size | Output Size | Compression | Latency (ms) | Throughput (MB/s) | Max Absolute Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ampere 2:4 Pack | 64.00 MB | 34.00 MB | **1.88x** | 89.90 ms | 711.9 MB/s | **0.000000 (Exact Lossless)** |
| Ampere 2:4 Unpack | 34.00 MB | 64.00 MB | 1.00x | 24.87 ms | 2,573.0 MB/s | **0.000000 (Exact Lossless)** |

---

## 6. Native Zig CLI & Process Footprint

Comparing process initialization and idle memory between Python/PyTorch and the native `hk.exe` binary:

| Metric | Python Runtime (Torch + HK) | Standalone Zig Binary (`hk.exe`) | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| Idle Process RAM (RSS) | 197.44 MB | 2.80 MB | **98.6% less RAM** |
| Process Startup Latency | 4,880.72 ms | 53.98 ms | **90.4x faster startup** |
| HF Tensor Name Mapping Rate | 12,400 names/sec | 320,944 names/sec | **25.8x faster** |
| 65k Context Truncation | 191.82 microseconds | 14.30 microseconds | **13.4x faster** |

---

## 7. Quantization Benchmarks (Complementary Edge Schemes)

Evaluated on a 2048 x 2048 matrix (4,194,304 f32 elements / 16.00 MB) with AVX2 SIMD acceleration:

| Quant Format | Packed Size | Compression | Quant Latency | Quant Throughput | Dequant Latency | Dequant Throughput | Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Q8_0 (8-bit Symmetric) | 4.25 MB | 3.76x | 6.4 ms | 2,489.8 MB/s | 2.4 ms | 6,688.5 MB/s | 0.999985 |
| Q4_0 (4-bit Symmetric) | 2.25 MB | 7.11x | 6.7 ms | 2,372.8 MB/s | 3.4 ms | 4,667.7 MB/s | 0.996318 |
| Q8_K (8-bit K-Quant) | 4.56 MB | 3.51x | 5.9 ms | 2,718.0 MB/s | 4.7 ms | 3,396.6 MB/s | 0.999975 |
| Q4_K (4-bit K-Quant) | 2.25 MB | 7.11x | 5.9 ms | 2,705.7 MB/s | 2.9 ms | 5,535.6 MB/s | 0.994132 |
| Dual-Mode NF4 + Residual | 97.09 MB (SmollM) | 6.40x | 245.0 ms | - | - | **0.999994** |
