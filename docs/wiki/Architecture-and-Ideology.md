# Architecture and Ideology

## The Spark Behind HK

I started building HK out of sheer frustration late one night sitting with my laptop.

I wanted to experiment with modern open-weight language models locally. I had an ordinary machine: 16 GB of system RAM and a modest laptop GPU. I was genuinely excited to poke around under the hood, try fine-tuning a small model on some custom code, and see how far I could push it.

Instead, reality hit like a brick wall.

First came the local inference experience. Just importing packages in Python took what felt like forever. Loading a modest 1B or 3B model into memory dragged my entire laptop to a crawl. The fans spun up to maximum speed, the cursor started stuttering, and before I could even run a single token forward pass, the process crashed with an out-of-memory error. Standard runtimes were allocating gigabytes of RAM just for Python runtime wrappers, framework overhead, and unaligned byte buffers.

Then came training, and that was an absolute nightmare.

If you have ever tried fine-tuning or training models on consumer hardware, you know the pain:
- Doing Full Fine-Tuning (FFT) where you update 100% of the parameters across the network was practically impossible locally. Standard PyTorch training state (weights, gradients, and Adam first and second moments) easily eats up 16 to 24 bytes per parameter. An 8B model suddenly demands 80+ GB of VRAM.
- Even trying parameter-efficient methods like LoRA meant installing a fragile house of cards: specific PyTorch builds, matching CUDA toolkits, bitsandbytes DLLs that frequently crashed on Windows, and conflicting dependencies.
- Worst of all was the rigidity of the architectures themselves. The moment pre-training finishes, traditional models are frozen in stone. Layer widths are fixed. Vocabularies are locked. If you try to teach an existing model a new programming language or new domain vocabulary, it struggles because it does not have the capacity to absorb new concepts. And if you force the weights to update through standard fine-tuning, the model suffers from catastrophic forgetting, corrupting its earlier knowledge.

I remember staring at my screen thinking: why does adding a few new capabilities to a 1B model require throwing the whole thing away and retraining from scratch on an expensive GPU cluster?

When you look at biological brains, nothing works like that. When you learned to code, or learned a second language, you did not erase your childhood memories. You did not have to reboot your brain from scratch. Biological brains are dynamic. They develop through continuous neurogenesis, widen neural pathways, strengthen synapses, and adapt over a lifetime while keeping earlier memories intact.

When I started analyzing traditional formats like SafeTensors and GGUF, I saw so many fundamental things that could be fixed:
- They treat neural networks as static, dead files on disk rather than living architectures.
- Their memory layouts force an artificial choice between unaligned bloated floats and lossy low-bit quantizations that degrade reasoning.
- Memory reads do not align with how hardware memory controllers, DMA channels, and GPU warps actually read data.
- Checkpoint history is outsourced to external Git LFS repos or duplicate multi-gigabyte folders.

I decided to build HK to test a different path:
1. Neural networks should be dynamic like the human brain, capable of growing their layers (Net2Net), expanding their vocabulary, and learning continuously without catastrophic forgetting.
2. The framework must work seamlessly on whatever hardware you have. Whether you are on a regular laptop CPU, a consumer GPU, an Apple Silicon Mac, or a multi-GPU server, I built HK to run reliably, cleanly, and faster than traditional tools.

---

## How Models Are Traditionally Trained vs. My Approach in HK

To understand how HK works, it helps to look at how neural models are traditionally trained and where I found the biggest bottlenecks:

### 1. Pre-Training from Scratch
- Traditional approach: Models start with randomly initialized weights and train over trillions of tokens using cross-entropy loss and AdamW. Checkpoints are written to disk as massive monolithic PyTorch `.pt` or SafeTensors files. Loading or verifying checkpoints requires duplicating memory across storage, OS buffers, and framework tensors.
- My approach in HK: I store pre-trained weights with zero compute headroom and universal page alignment (4KB / 16KB). When loading or verifying checkpoints, HK uses direct zero-copy OS memory mapping (`mmap`). Multi-file sharding (`HeaderFlags.IS_SHARDED`) allows terabyte-scale models to be partitioned cleanly with instant slice access.

### 2. Full Fine-Tuning (FFT)
- Traditional approach: All parameters in the network are updated. While FFT yields high quality, it requires maintaining gradients and optimizer states for every single weight in the network. If training destabilizes or overfits, restoring an earlier state requires keeping full copies of previous multi-gigabyte checkpoints on disk.
- My approach in HK: I built `HKTrainer` to natively support Full Fine-Tuning (`use_qlora=False`). Base weights are memory-mapped directly into the process. Optimizer states and intermediate checkpoints are tracked within the file's Appendix DAG, cutting redundant disk duplication while allowing sub-millisecond rollback to any previous step.

### 3. Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA)
- Traditional approach: Base weights are frozen, and low-rank adapter matrices ($W + B \cdot A$) are trained. In traditional stacks, this requires external packages (`peft`, `bitsandbytes`, `accelerate`), which frequently suffer from installation incompatibilities on Windows and macOS.
- My approach in HK: I built QLoRA natively into `HKTrainer` with zero external dependencies. Adapters are attached directly to projection layers and saved cleanly into the container's Appendix region.

### 4. Continued Pre-Training (CPT) & Domain Adaptation
- Traditional approach: Continuing pre-training on domain-specific data (e.g., medical texts or programming languages) causes two major issues: token fragmentation (new keywords are split into 4-5 tiny tokens) and catastrophic forgetting (base capabilities degrade).
- My approach in HK: I implemented Dynamic Vocabulary Expansion (`expand_vocab`) to add new domain tokens without breaking existing embeddings. When representational capacity limits are reached, Net2WiderNet injects fresh capacity into intermediate SwiGLU layers, and Plasticity Isolation gradient masks shield the base weights.

### 5. Preference Optimization & Alignment (DPO, SPIN)
- Traditional approach: RLHF requires training separate reward models and value networks, demanding huge compute clusters. DPO improves upon this by using pairwise preferences.
- My approach in HK: I incorporated Self-Play Fine-Tuning (`SPINLoss`) directly into the autonomous self-training loop. The model plays against its own past generations to iteratively bootstrap reasoning depth without requiring human-labeled pairs.

---

## Universal Hardware Inclusivity

One of the foundational pillars when I designed HK is that it must run seamlessly on whatever hardware you have.

Whether you are on:
- A regular thin-and-light laptop with only an integrated CPU
- An older desktop with a consumer GPU (like an RTX 3050 or GTX 1660)
- An Apple Silicon Mac (M1, M2, M3, M4)
- A high-end workstation or server with multiple enterprise GPUs

I engineered HK to run cleanly and deliver maximum performance without manual hacks.

### Why It Works So Well Across Different Silicon

The core of my design lies in the memory layout of the `.hk` file:

```
+---------------------------------------------------------------+
|                       HK Super-Coalesced File                 |
|                                                               |
|  [Header: 128B] [TOC: 128B entries] [Aligned Tensor Payloads] |
+---------------------------------------------------------------+
       |                         |                       |
       v                         v                       v
128-Byte Alignment        4096-Byte Alignment    16384-Byte Alignment
(NVIDIA Tensor Cores)     (AMD ROCm / Intel CPU) (Apple Silicon Metal)
```

1. NVIDIA GPUs require 128-byte alignment so that GPU warps can coalesce memory reads into a single transaction.
2. AMD GPUs, Intel CPUs, and Linux systems use 4096-byte (4 KB) page boundaries for direct memory mapping and DMA.
3. Apple Silicon uses 16384-byte (16 KB) pages for unified memory buffer mapping (`newBufferWithBytesNoCopy`).

Because 4096 is divisible by 128 (32 x 128), and 16384 is divisible by 128 (128 x 128), aligning a file to 4KB or 16KB automatically satisfies NVIDIA 128-byte warp coalescing.

This means a single `.hk` file gives you zero-copy, direct memory mapping on Apple Silicon, native DMA on AMD/Intel, and fully coalesced reads on NVIDIA. Zero conversion steps, zero wasted memory, and zero compute overhead.

---

## Native Zig SIMD Engine

Instead of relying on heavy C++ frameworks or runtime interpreters, I wrote HK's core engine in native Zig:
- The standalone binary (`hk.exe` or `hk`) idles at just 2.8 MB of RAM, compared to 200+ MB for standard Python runtimes.
- Cold startup takes under 55 milliseconds.
- I wrote the SIMD GEMV kernels with 4-row register unrolling and 8 interleaved accumulators, delivering high-throughput matrix-vector multiplication even on a single CPU core.
- Dynamic hardware detection probes your CPU features at startup (AVX2, AVX-512, AMX, ARM NEON, Apple Silicon) and automatically picks the fastest vector instruction path.
