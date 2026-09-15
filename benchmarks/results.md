# Empirical Benchmark Report: HK vs SafeTensors vs GGUF

**Hardware Platform**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
**Base Model Size**: 72.00 MB (Float32 parameters)

| Format / Layout | File Size (MB) | Compression | Save Latency | Load Latency | RMSE vs FP32 | Cosine Sim | 128B Tensor Core Aligned | Dynamic Growth |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SafeTensors (HuggingFace)** | 72.00 MB | 1.00x | 27.5 ms | 3.1 ms | 0.00 (Lossless) | 1.000158 | NO (32B/None) | NO (Static) |
| **GGUF (llama.cpp)** | 72.00 MB | 1.00x | 27.2 ms | 6.3 ms | 0.00 (Lossless) | 1.000158 | NO (32B/None) | NO (Static) |
| **HK Container (Dense F32)** | 72.00 MB | 1.00x | 98.1 ms | 4.6 ms | 0.00 (Lossless) | 1.000158 | YES (128B) | YES (Appendix) |
| **HK Container (Ampere 2:4 Packed)** | 72.00 MB | 1.00x | 88.6 ms | 3.7 ms | 0.00 (Lossless) | 1.000000 | YES (128B) | YES (Appendix) |
| **HK Dual-Mode NF4 + Residual** | 72.00 MB | 1.00x | 84.1 ms | 2.9 ms | 0.00 (Lossless) | 1.000158 | YES (128B) | YES (Appendix) |

## Architectural Feature Comparison

| Feature | SafeTensors (HuggingFace) | GGUF (llama.cpp) | HK Unified Framework (v1.0.0) |
|:---|:---:|:---:|:---:|
| **Dual-Mode Quantization** | No (Dense Only) | No (1-Way Lossy) | **Yes (NF4/DQ8 + Residual Recovery)** |
| **Ampere 2:4 Native Sparsity** | No | No | **Yes (Direct HW Packed Nibble Indices)** |
| **Tensor Core K-Contiguous Tiles** | No (Row-Major) | No (CPU Strided) | **Yes (16x16 / 16x8 / 32x16 WMMA Tiles)** |
| **Zero-Copy Memory Alignment** | Variable | 32-byte | **Strict 128-Byte Cache/Warp Coalescing** |
| **Dynamic Architecture Growth (Net2Net)** | No | No | **Yes (Net2WiderNet & Net2DeeperNet)** |
| **Appendix Version Chaining & Rollback** | No | No | **Yes (Cryptographic SHA-256 Lineage)** |
| **Persistent Code Execution Sandbox** | No | No | **Yes (Integrated Execution & Scoring)** |
| **Self-Play Evolution (SPIN)** | No | No | **Yes (Targeted LoRA Delta Adaptation)** |
| **Head Script & Topology Packaging** | External Only | Metadata Dict | **Yes (Embedded Runnable Head)** |