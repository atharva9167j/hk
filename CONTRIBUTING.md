# Contributing to HK Neural Tensor Framework

Thank you for your interest in contributing to HK! We welcome contributions across all areas: native Zig kernels, high-performance quantizers, quantization zoo expansions, multilingual SDKs, Python framework bridges, documentation, and benchmarks.

---

## 1. Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive, and collaborative in all discussions and pull requests.

---

## 2. Architecture Overview

HK is structured as a layered, multi-language system:
- **`src/`**: High-performance core engine written in Zig 0.16. Includes:
  - `format.zig`: Binary container layout, headers, TOC, metadata, and chunk formats.
  - `quantization.zig`: Quantization algorithms (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `IQ` codebooks, etc.) with SIMD vectorization.
  - `tensor_ops.zig`: Packed-weight SIMD GEMV kernels (`gemvQ8_0`, `gemvQ4_0`, `gemvQ4_K`), RoPE coordinates, and LayerNorm offsets.
  - `gguf.zig`: Native binary GGUF reader, writer, and zero-copy bitstream transcoders.
  - `c_api.zig`: ABI-stable C interface exported as a shared library (`hk.dll`, `libhk.so`, `libhk.dylib`).
- **`include/`**: C and C++ header files (`hk.h`, `hk.hpp`).
- **`python/hk/`**: Python SDK with PyTorch zero-copy tensors, safe open, packed execution (`HKQuantizedLinear`), and tokenizer integration.
- **`bindings/`**: Official foreign function interfaces for Rust, C#, Go, Java, and TypeScript.
- **`tests/`**: Comprehensive pytest and native unit test suites.

---

## 3. Development Environment Setup

### Prerequisites
- **Zig**: `0.16.x` or latest master build. Verify with `zig version`.
- **Python**: Python `3.10`+ (Python 3.12 recommended).
- **C/C++ Compiler**: Clang, GCC, or MSVC (optional, Zig acts as C compiler).
- **Rust** (optional, for Rust SDK): `rustc` & `cargo`.
- **.NET SDK** (optional, for C# SDK): .NET 8.0+.
- **JDK** (optional, for Java SDK): Java 17+.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshitkhandelwal208/hk.git
   cd hk
   ```

2. **Build the native Zig engine and run native tests**:
   ```bash
   zig build
   zig build test --summary all
   ```

3. **Install the Python package in editable mode**:
   ```bash
   pip install -e ".[dev]"
   # or with core dependencies:
   pip install torch numpy safetensors transformers gguf pytest
   pip install -e .
   ```

4. **Verify the Python CLI**:
   ```bash
   python -m hk.cli --help
   hk --help
   ```

5. **Run the Python test suite**:
   ```bash
   pytest tests/
   ```

---

## 4. Testing Guidelines

Any PR touching native code or quantization kernels **must** satisfy:
- **Numerical Parity**: Quantizers and dequantizers must match reference implementations within acceptable epsilon (`max_discrepancy == 0.0` for bit-identical dequantization, or $\le 10^{-4}$ for floating point math).
- **Zero Failures, Zero Warnings**: All tests in `tests/` must pass with 0 pytest warnings and 0 errors.
- **Cross-Platform Portability**: File I/O and SIMD intrinsics must compile cleanly across Windows (`x86_64`), Linux (`x86_64`, `aarch64`), and macOS (`aarch64`).
  ```bash
  zig build -Dtarget=x86_64-linux
  zig build -Dtarget=aarch64-linux
  zig build -Dtarget=aarch64-macos
  zig build -Dtarget=x86_64-windows
  ```

---

## 5. Submitting a Pull Request

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feat/my-quant-kernel
   ```
2. **Commit your changes**:
   - Write clear, concise commit messages.
   - Reference any relevant issues (e.g. `Fixes #123`).
3. **Run all tests locally before opening the PR**:
   ```bash
   zig build test
   pytest tests/
   ```
4. **Open a Pull Request** against `main`:
   - Fill out the PR template completely.
   - CI will automatically run multi-platform tests and multilingual SDK verification.
