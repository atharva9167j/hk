# Getting Started with HK

In this guide, I'll walk you through installing HK and running your first model in just a few minutes.

HK runs on Windows, Linux, and macOS (Intel and Apple Silicon). You do not need a high-end GPU to get started.

---

## 1. Python Installation

The easiest way to use HK in Python is via pip:

```bash
pip install hknt
```

If you are developing locally or want to build the native C-ABI bridge from source:

```bash
# Clone the repository
git clone https://github.com/harshitkhandelwal208/hk.git
cd hk

# Build the native high-performance engine using Zig
zig build -Doptimize=ReleaseFast

# Copy the compiled dynamic library into the Python package directory
# On Windows:
copy zig-out\bin\hk.dll python\hk\
# On Linux:
cp zig-out/lib/libhk.so python/hk/
# On macOS:
cp zig-out/lib/libhk.dylib python/hk/

# Install in editable mode with dependencies
pip install -e .
```

### Verifying Python Installation
Run a quick sanity check in Python:

```python
import hk
from hk.native import native_detect_hardware

hw = native_detect_hardware()
print("HK hardware detection:", hw)
print("HK native version:", getattr(hk, "__version__", "1.0.0"))
```

---

## 2. Standalone Native CLI Installation

If you just want to run models, chat with them, inspect weights, or convert checkpoints without touching Python, you can use the standalone native CLI binary.

### Building from Source (Requires Zig 0.13+)
```bash
# Build the native CLI executable
zig build -Doptimize=ReleaseFast

# The executable will be in zig-out/bin/
# On Windows: zig-out\bin\hk.exe
# On Linux/macOS: zig-out/bin/hk
```

You can add `zig-out/bin` to your system `PATH` to run `hk` directly from anywhere in your terminal.

### Verifying the CLI
```bash
# Check version and available subcommands
hk --help

# Profile your machine's CPU features and page alignment
hk hardware-profile
```

---

## 3. Node.js and TypeScript Installation

If you are working in JavaScript or TypeScript, I created a zero-dependency npm package:

```bash
npm install hknt
```

Or build the local bindings:

```bash
cd bindings/js
npm install
npm run build
```

---

## 4. Your First Model: Quick Walkthrough

Here is a quick script showing how I save tensors to an `.hk` container, inspect them with lazy slicing, and run matrix math:

```python
import torch
import hk
from hk.torch import save_file, load_file, safe_open

# 1. Create a dictionary of PyTorch tensors
tensors = {
    "model.embed_tokens.weight": torch.randn(1000, 256, dtype=torch.float32),
    "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 256, dtype=torch.float32),
    "model.layers.0.mlp.up_proj.weight": torch.randn(512, 256, dtype=torch.float32),
}

# 2. Save to an HK container with metadata
metadata = {
    "architecture": "llama",
    "author": "Harshit",
    "version": "1.0.0",
}
save_file(tensors, "my_model.hk", metadata=metadata)
print("Saved my_model.hk successfully.")

# 3. Inspect keys and shapes without loading the whole file into RAM
with safe_open("my_model.hk", framework="pt") as f:
    print("Metadata:", f.metadata())
    print("Tensors in container:", f.keys())
    
    # Grab a slice of the gate projection (first 64 rows only)
    slice_tensor = f.get_slice("model.layers.0.mlp.gate_proj.weight")
    print("Slice shape:", slice_tensor.get_shape())
    first_block = slice_tensor[0:64, :]
    print("Fetched block shape:", first_block.shape)

# 4. Load the full dictionary with zero-copy memory mapping
loaded = load_file("my_model.hk")
print("Loaded tensors:", list(loaded.keys()))
```

---

## 5. Converting an Existing Model

If you have a model in GGUF or SafeTensors format, you can convert it immediately:

```bash
# Convert a GGUF file
hk convert-gguf model.gguf model.hk

# Convert a SafeTensors file
hk convert-safetensors model.safetensors model.hk
```

Once converted, you can inspect it in your terminal:

```bash
hk inspect model.hk
```

Or start an interactive terminal chat session:

```bash
hk chat model.hk --temp 0.7
```

---

## Next Steps
- Read the [Transition Guide](Transition-Guide.md) to replace Ollama, Unsloth, and Hugging Face pipelines.
- Read [Dynamic Architecture Growth](Dynamic-Architecture-Growth.md) to see how models widen without retraining.
- Explore the [CLI Reference](CLI-Reference.md) for all available command flags.
