# HK High-Performance Benchmark Report
**Engine**: Native Zig 0.16.0 SIMD Engine with Full-Tensor AVX2 Kernels  
**Environment**: Windows x86_64 | PyTorch 2.x | Python 3.12  
**Total Benchmark Duration**: 15.53 seconds

---

## 1. Full-Tensor SIMD Quantization & Dequantization Throughput
Evaluated on a 4096 x 4096 weight matrix (16,777,216 elements / 67.11 MB FP32):

| Quant Format           | Packed Size | Compression | Quant Latency | Quant Throughput | Dequant Latency | Dequant Throughput | RMSE    | Cosine Sim |
|:-----------------------|-:------------|-:------------|-:--------------|-:-----------------|-:----------------|-:-------------------|-:--------|-:----------:|
| Q8_0 (8-bit Symmetric) | 4.25 MB     | 3.76x       | 13.4 ms       | 1190.4 MB/s      | 10.1 ms         | 1591.0 MB/s        | 0.00535 | 0.999985   |
| Q4_0 (4-bit Symmetric) | 2.25 MB     | 7.11x       | 13.3 ms       | 1204.2 MB/s      | 6.0 ms          | 2655.0 MB/s        | 0.08589 | 0.996318   |
| Q8_K (8-bit K-Quant)   | 4.56 MB     | 3.51x       | 10.6 ms       | 1512.3 MB/s      | 4.6 ms          | 3452.0 MB/s        | 0.00695 | 0.999975   |
| Q6_K (6-bit K-Quant)   | 3.28 MB     | 4.88x       | 14.3 ms       | 1118.0 MB/s      | 11.3 ms         | 1413.3 MB/s        | 0.01930 | 0.999819   |
| Q5_K (5-bit K-Quant)   | 2.75 MB     | 5.82x       | 20.0 ms       | 800.6 MB/s       | 10.6 ms         | 1515.6 MB/s        | 0.03817 | 0.999278   |
| Q4_K (4-bit K-Quant)   | 2.25 MB     | 7.11x       | 10.0 ms       | 1605.3 MB/s      | 9.0 ms          | 1780.7 MB/s        | 0.10886 | 0.994132   |
| Q3_K (3-bit K-Quant)   | 1.72 MB     | 9.31x       | 17.8 ms       | 899.9 MB/s       | 11.1 ms         | 1438.9 MB/s        | 2.14444 | -0.458345  |
| Q2_K (2-bit K-Quant)   | 1.31 MB     | 12.19x      | 18.7 ms       | 857.6 MB/s       | 6.6 ms          | 2432.2 MB/s        | 0.32834 | 0.951503   |

---

## 2. SIMD GEMV Kernel Latency & GFLOPS
Single-vector matrix multiplication ($M=4096, K=4096$, 33.55 MFLOP) comparing PyTorch CPU FP32 vs HK Native Kernels:

| Kernel Engine / Precision   | Weight RAM | Latency (ms) | Throughput (GFLOPS) | Speedup vs PyTorch |
|:----------------------------|-:-----------|-:-------------|-:--------------------|-:------------------:|
| PyTorch CPU F.linear (FP32) | 64.00 MB   | 3.542 ms     | 9.47 GFLOPS         | 1.00x              |
| HK Native SIMD GEMV (FP32)  | 64.00 MB   | 5.647 ms     | 5.94 GFLOPS         | 0.63x              |
| HK Native SIMD GEMV (Q8_0)  | 17.00 MB   | 5.512 ms     | 6.09 GFLOPS         | 0.64x              |
| HK Native SIMD GEMV (Q4_0)  | 9.00 MB    | 13.077 ms    | 2.57 GFLOPS         | 0.27x              |
| HK Native SIMD GEMV (Q4_K)  | 9.00 MB    | 14.886 ms    | 2.25 GFLOPS         | 0.24x              |

---

## 3. Native BPE Tokenizer Throughput
High-throughput tokenization over a 30 KB multilingual text corpus:

| Tokenizer Operation | Payload Size | Tokens Count | Latency (ms) | Throughput (KB/s) | Rate (Tokens/sec) |
|:--------------------|-:-------------|-:-------------|-:-------------|-:------------------|-:-----------------:|
| Native Zig Encode   | 27.2 KB      | 20,402       | 6.72 ms      | 4054.6 KB/s       | 3,036,104 tok/s   |
| Native Zig Decode   | 27.2 KB      | 20,402       | 3.50 ms      | 7794.2 KB/s       | 5,836,347 tok/s   |

---

## 4. NVIDIA Ampere 2:4 Hardware Sparsity Pack & Unpack
Bit-exact hardware physical nibble compression:

| Sparsity Operation | Input Size | Output Size | Compression | Latency (ms) | Throughput (MB/s) | Max Abs Error |
|:-------------------|-:-----------|-:------------|-:------------|-:-------------|-:------------------|-:-------------:|
| Ampere 2:4 Pack    | 64.00 MB   | 34.00 MB    | 1.88x       | 110.99 ms    | 576.6 MB/s        | 0.000000      |
| Ampere 2:4 Unpack  | 34.00 MB   | 64.00 MB    | 1.00x       | 23.28 ms     | 2749.7 MB/s       | 0.000000      |

---

## 5. Zero-Copy Container Slicing & Memory-Mapped Latency
Evaluated across an 8-layer transformer container (32 tensors, ~67 MB):

| Reader / Loader Access Pattern  | Container Size | Operation                     | Latency (ms) | Speedup |
|:--------------------------------|-:---------------|-:------------------------------|-:-------------|-:-------:|
| SafeTensors (Full dict load)    | 224.00 MB      | Full Load (32 tensors)        | 4.899 ms     | 1.00x   |
| HK Native (safe_open full read) | 224.00 MB      | Iterative Mmap Read           | 159.421 ms   | 0.03x   |
| HK Native (safe_open slice)     | 224.00 MB      | Zero-Copy 2D Slice [128, 512] | 17.696 ms    | 0.3x    |
