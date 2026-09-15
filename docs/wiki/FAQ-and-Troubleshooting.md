# Frequently Asked Questions and Troubleshooting

---

## General Questions

### Can I run HK on a standard laptop without a dedicated GPU?
Yes. Universal hardware inclusivity is one of the core principles I built HK on. HK includes 4-row unrolled SIMD GEMV kernels in native Zig. When running on CPU, it automatically detects your processor's vector extensions (AVX2, AVX-512, ARM NEON) and unrolls math directly across registers. You do not need an NVIDIA card to get fast inference.

### How do I offload layers to an NVIDIA GPU?
Use the `-ngl` flag in the CLI:
```bash
# Offload 24 layers to GPU
hk chat model.hk -ngl 24
```
In Python:
```python
from hk import AutoModelForCausalLM

# If CUDA is available, weights move seamlessly to GPU
model = AutoModelForCausalLM.from_pretrained("model.hk", device_map="auto")
```

### Does HK work on Apple Silicon Macs (M1/M2/M3/M4)?
Yes. HK aligns tensor payloads to 16384-byte (16 KB) boundaries by default when targeting Apple Silicon. This matches the macOS page size, allowing direct zero-copy buffer views into Metal Unified Memory via `newBufferWithBytesNoCopy`.

### How does HK compare to SafeTensors and GGUF?
- SafeTensors: Static weight format with unaligned floats and JSON headers. It cannot grow, does not train, and does not support structured sparsity.
- GGUF: Static quantized inference format requiring a heavy C++ runtime. It does not support training, Net2Net widening, or an in-container version DAG.
- HK: A living neural framework I built to support zero compute headroom raw unquantized storage, 2:4 Ampere structured sparsity, native Net2Net widening, in-container training (`HKTrainer`), self-learning (`SelfTrainingPipeline`), and instant generational rollback.

---

## Troubleshooting

### Windows File Lock Error (`PermissionError: [WinError 32]`)
On Windows, holding an open memory-mapped file descriptor (`mmap` or `MapViewOfFile`) prevents deleting or overwriting the file.

In HK, `hk.torch.load_file` and `safe_open` implement explicit lifecycle cleanup. If you need to delete a file right after loading it:
```python
# Use safe_open context manager to ensure handles are closed on exit
with safe_open("model.hk", framework="pt") as f:
    weights = {k: f.get_tensor(k) for k in f.keys()}
# Handles are now completely closed; file can be safely deleted or replaced
```

### CUDA Backend Tests on Cloud CI
If you run `zig build test` on a cloud CI runner (like GitHub Actions) that does not have an NVIDIA GPU driver installed, HK detects that CUDA is unavailable and automatically skips GPU-specific assertions (`SkipZigTest`). On machines with an NVIDIA GPU, all CUDA tests execute and verify mathematical parity against CPU reference kernels.

### How to Sync `docs/wiki/` to the GitHub Wiki Git Repository
GitHub Wikis are separate git repositories located at:
`https://github.com/<owner>/<repo>.wiki.git`

To push all pages from `docs/wiki/` to your GitHub Wiki:
```bash
# Clone the wiki repo
git clone https://github.com/harshitkhandelwal208/hk.wiki.git local-wiki

# Copy all markdown files from docs/wiki into the wiki repo
# On Windows (PowerShell):
Copy-Item docs/wiki/* local-wiki/ -Force

# Commit and push
cd local-wiki
git add .
git commit -m "Update documentation from main repository"
git push origin master
```
