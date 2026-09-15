# Welcome to the HK Wiki

Welcome to the documentation I put together for the HK Neural Tensor Framework.

I built HK as a neural framework and unified binary format (.hk) for continuous model evolution, dynamic architecture growth, fast training, and zero-overhead storage across every type of hardware.

Whether you are running inference on a standard laptop, fine-tuning on a desktop with a consumer GPU, training large runs on a workstation, or deploying on Apple Silicon, I designed HK to work seamlessly without forcing you to deal with bloated runtimes, out-of-memory crashes, or static models frozen in stone.

---

## The Core Philosophy Behind HK

Traditional deep learning treated neural networks as static, immutable files. You trained a model once, exported it to disk as a frozen blob (SafeTensors, GGUF, or PyTorch checkpoints), and that was it. If you needed more capacity, better reasoning in a specific domain, or support for a new programming language, your only option was to throw the checkpoint away and train a brand new model from scratch.

I started HK based on a very simple observation: real biological brains do not work that way. Biological brains grow, widen connections, form new neural pathways, and adapt continuously while preserving what they have already learned.

Working with standard frameworks on my everyday laptop was deeply frustrating. Running modern open weights locally meant sluggish load times, huge memory footprints, and dependency tangles. Training or fine-tuning them on anything less than an expensive multi-GPU cluster was an absolute nightmare.

When I dug into the plumbing of traditional model architectures and file formats, I discovered layers of unnecessary overhead:
- Multilingual and cross-platform runtimes allocate hundreds of megabytes just to start an inference loop.
- Formats force an artificial choice between unaligned raw floats and lossy low-bit quantizations that destroy fine-grained reasoning.
- Weights are tied to fixed dimension matrices with no clean mathematical way to grow them during fine-tuning without corrupting earlier representations.
- Model lineage and version history are pushed to external Git LFS repos or full file duplicates, eating up hundreds of gigabytes of disk space.

I built HK from the ground up in native Zig and Python to solve every one of these problems. I engineered it to run faster, use less memory, support function-preserving dynamic expansion (Net2Net), and guarantee multi-device super-coalescing on all silicon.

---

## Wiki Navigation

Explore the sections below for complete documentation on every part of the framework:

### Getting Started & Transitioning
- [Getting Started](Getting-Started.md): Installation across Python, Node.js, Zig source, and pre-built binaries.
- [Transition Guide](Transition-Guide.md): Effortless substitution guide for replacing Ollama, LM Studio, llama.cpp, Unsloth, Hugging Face Transformers, and SafeTensors.
- [Architecture and Ideology](Architecture-and-Ideology.md): The core concepts, biological inspiration, and hardware inclusivity guarantees.

### Core Framework Deep-Dives
- [Format Specification](Format-Specification.md): Complete binary layout of the .hk container (Header, TOC, Payloads, Alignment).
- [Raw Storage and Super-Coalescing](Raw-Storage-and-Super-Coalescing.md): How universal 128-byte, 4KB, and 16KB alignment delivers zero compute headroom on NVIDIA, AMD, Intel, and Apple Silicon.
- [Dynamic Architecture Growth](Dynamic-Architecture-Growth.md): Net2WiderNet, SwiGLU widening, Net2DeeperNet, dynamic vocabulary expansion, and the native GrowthGovernor.
- [Training and Fine-Tuning](Training-and-Fine-Tuning.md): Training with HKTrainer, native QLoRA, plateau-triggered adaptive widening, and in-container checkpoints.
- [Autonomous Self-Training](Autonomous-Self-Training.md): The closed-loop self-learning loop with inner monologue reasoning (<think>), SPIN self-play, and secure AST sandboxing.
- [In-Container Version Lineage](In-Container-Version-Lineage.md): The Appendix DAG, cryptographic parent hash chaining, and sub-millisecond instant rollback.
- [Inference and Serving](Inference-and-Serving.md): Running models locally, KV cache optimization, interactive CLI chat, and dynamic CPU/GPU offloading.
- [Storage and Sparsity](Storage-and-Sparsity.md): Ampere 2:4 structured sparsity with nibble indexing, 2D cache tiling, microsecond metadata editing, and edge quantization.

### References & Guides
- [CLI Reference](CLI-Reference.md): Complete manual for the native hk command line utility.
- [Python API Reference](Python-API-Reference.md): Comprehensive reference for all Python classes, functions, and modules.
- [Multi-Language SDKs](Multi-Language-SDKs.md): Using HK from Rust, TypeScript, C#, Go, Java/Android, and C/C++.
- [Benchmarks and Performance](Benchmarks-and-Performance.md): Full empirical benchmarks on production models, memory mapping, SIMD GEMV, and CLI footprint.
- [FAQ and Troubleshooting](FAQ-and-Troubleshooting.md): Common questions and setup troubleshooting.

---

## Hardware Inclusivity

I built HK to deliver top-tier performance on whatever hardware you have:
- Laptop CPU (x86_64 or ARM): Native SIMD vector kernels (AVX2, AVX-512, NEON) unroll GEMV across registers, running inference with sub-microsecond layer retrieval.
- Apple Silicon (M1/M2/M3/M4): Direct 16KB page alignment maps cleanly into macOS Metal Unified Memory with zero memory copies.
- NVIDIA GPUs: 128-byte warp coalescing and native Ampere 2:4 structured sparsity deliver double the compute throughput.
- AMD GPUs & Intel NPUs: 4KB page alignment connects directly to ROCm DirectGMA and OpenVINO DMA without conversion.
