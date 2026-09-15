# HK Documentation Hub

This directory contains the documentation and GitHub Wiki pages I wrote for the HK Neural Tensor Framework.

---

## Documentation Index (`docs/wiki/`)

### Getting Started & Concepts
- [Home](wiki/Home.md): Welcome, philosophy, and overview.
- [Getting Started](wiki/Getting-Started.md): Quick installation guide across Python, Zig, Node.js, and CLI.
- [Transition Guide](wiki/Transition-Guide.md): Drop-in guide for replacing Ollama, LM Studio, Unsloth, Hugging Face, and SafeTensors.
- [Architecture and Ideology](wiki/Architecture-and-Ideology.md): The motivation behind HK, biological brain principles, and hardware inclusivity.

### Framework Deep Dives
- [Format Specification](wiki/Format-Specification.md): Complete binary layout of `.hk` (Header, TOC, Payloads, Alignment).
- [Raw Storage and Super-Coalescing](wiki/Raw-Storage-and-Super-Coalescing.md): Zero compute headroom and universal 128B/4KB/16KB silicon alignment.
- [Dynamic Architecture Growth](wiki/Dynamic-Architecture-Growth.md): Net2WiderNet (SwiGLU/Linear), Net2DeeperNet, dynamic vocabulary, and the GrowthGovernor.
- [Training and Fine-Tuning](wiki/Training-and-Fine-Tuning.md): Training with HKTrainer, native QLoRA, and plateau-triggered widening.
- [Autonomous Self-Training](wiki/Autonomous-Self-Training.md): Closed-loop self-training, `<think>` reasoning traces, SPIN loss, and AST sandboxing.
- [In-Container Version Lineage](wiki/In-Container-Version-Lineage.md): Appendix DAG, cryptographic parent hash chaining, and instant rollback.
- [Inference and Serving](wiki/Inference-and-Serving.md): Native CLI execution (`hk run`, `hk chat`), dynamic layer offloading (`-ngl`), and Python inference.
- [Storage Innovations and Sparsity](wiki/Storage-and-Sparsity.md): Ampere 2:4 structured sparsity with nibble indexing, 2D tiling, in-place metadata editing, and edge quantization.

### Reference & Benchmarks
- [CLI Reference](wiki/CLI-Reference.md): Complete command-line manual for `hk`.
- [Python API Reference](wiki/Python-API-Reference.md): Reference manual for all Python packages and classes.
- [Multi-Language SDKs](wiki/Multi-Language-SDKs.md): Using HK from Rust, TypeScript, C#, Go, Java/Android, and C/C++.
- [Benchmarks and Performance](wiki/Benchmarks-and-Performance.md): Complete empirical benchmark data and physical memory measurements.
- [FAQ and Troubleshooting](wiki/FAQ-and-Troubleshooting.md): Frequently asked questions and common setup tips.

---

## Syncing with GitHub Wiki

I have structured all files in `docs/wiki/` to be dropped directly into the GitHub Wiki repository (`https://github.com/<owner>/<repo>.wiki.git`). See my [FAQ and Troubleshooting](wiki/FAQ-and-Troubleshooting.md) guide for synchronization instructions.

