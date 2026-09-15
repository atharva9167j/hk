# HK Standalone CLI Reference

I wrote the compiled native `hk` executable in Zig as a self-contained developer tool. It starts in under 55 milliseconds, uses less than 3 MB of idle RAM, and provides complete inspection, conversion, execution, and expansion capabilities.

---

## Command Overview

```bash
hk <subcommand> [arguments...] [options...]
```

| Subcommand | Description |
| :--- | :--- |
| `hardware-profile` | Probe host CPU vector extensions, vendor, and optimal memory page alignment. |
| `inspect` | Dump container header, TOC entries, storage types, and alignment details. |
| `verify` | Check container integrity, 128-byte Tensor Core alignment, and checksums. |
| `metadata set` | In-place microsecond metadata update without rewriting weights. |
| `expand` | Native vocabulary expansion and SwiGLU MLP width expansion (Net2WiderNet). |
| `benchmark` | Measure zero-copy mapping, first-touch page faults, memory bus throughput, and SIMD GEMV. |
| `appendix list` | Display version DAG records, generation indexes, loss, and parent hashes. |
| `rollback` | Instantly rollback a model container to a previous generation state. |
| `convert-gguf` | Transcode a GGUF file to an HK container. |
| `convert-safetensors`| Transcode a SafeTensors checkpoint to an HK container. |
| `export` | Export an HK container to legacy formats (SafeTensors or GGUF). |
| `run` | Run non-interactive autoregressive text generation from a prompt. |
| `chat` | Start an interactive terminal chat REPL with embedded template rendering. |
| `gui` | Launch the visual HK Model Studio graphical interface. |
| `hash` | Compute streaming cryptographic SHA-256 hash of the container. |

---

## Subcommand Details and Examples

### 1. `hardware-profile`
Probes your system hardware and displays supported instruction sets:

```bash
hk hardware-profile
```

Sample output:
```
================================================================================
HK HARDWARE PROFILE & SILICON ALIGNMENT AUDIT
================================================================================
Host Architecture      : x86_64
Operating System       : Windows
Optimal Page Alignment : 4096 bytes (4 KB)
Hugepage DMA Alignment : 65536 bytes (64 KB)
Vector Features        : AVX2: YES | AVX-512F: YES | AVX-512-VNNI: YES | AMX: NO
Universal Compatibility: YES (4096 is divisible by 128 - Tensor Core ready)
```

---

### 2. `inspect`
Dumps the complete Table of Contents and file header:

```bash
hk inspect model.hk
```

Add `--verbose` to see every single tensor offset, dimensions, and storage types.

---

### 3. `verify`
Audits the physical file alignment and container checksums:

```bash
hk verify model.hk
```

Checks:
- Magic bytes match `"HKNT"`.
- Table of Contents checksum matches header.
- Every single tensor payload starts on a multiple of `header.alignment` (128B, 4KB, or 16KB).
- Flags are consistent with payload data.

---

### 4. `metadata set`
Updates metadata fields in place without rewriting weights:

```bash
# Update model version string
hk metadata set model.hk general.version "2.0.0"

# Update author string
hk metadata set model.hk general.author "Harshit"

# Update Jinja2 chat template
hk metadata set model.hk tokenizer.chat_template "{% for message in messages %}..."
```

---

### 5. `expand`
Expands intermediate layer width (Net2WiderNet) and token vocabulary:

```bash
# Widen SwiGLU MLPs by 25% and increase vocabulary to 32,500
hk expand input.hk output.hk --width 1.25 --vocab 32500
```

Flags:
- `--width <float>`: Multiplier for feed-forward intermediate size (e.g. `1.25` for +25%).
- `--vocab <int>`: New target vocabulary size.
- `--noise-std <float>`: Initialization noise (default: `0.0` for bit-exact function preservation).

---

### 6. `benchmark`
Profiles memory traversal and compute throughput:

```bash
hk benchmark model.hk
```

Measures:
- Zero-copy virtual memory descriptor mapping latency.
- Lazy slice pointer resolution rate.
- Physical storage first-touch paging throughput (NVMe to RAM).
- Resident warm memory streaming rate.
- Register-unrolled SIMD GEMV compute throughput.

---

### 7. `appendix list`
Lists all version lineage records stored in the file:

```bash
hk appendix list model.hk
```

Displays generational indexes, record types, global steps, evaluation metrics, and parent hashes.

---

### 8. `rollback`
Restores container state to an earlier generation:

```bash
hk rollback model.hk --generation 1
```

Executes in under 1 millisecond. The Table of Contents is adjusted to point to the generation state requested.

---

### 9. `convert-gguf`
Converts a GGUF file directly to HK:

```bash
hk convert-gguf llama-3-8b.gguf llama-3-8b.hk
```

Preserves all metadata, tokenizers, and tensor layouts in a single streaming pass.

---

### 10. `convert-safetensors`
Converts a SafeTensors checkpoint to HK:

```bash
hk convert-safetensors model.safetensors model.hk
```

---

### 11. `export`
Exports an HK container back into GGUF or SafeTensors:

```bash
# Export to SafeTensors
hk export -f safetensors model.hk exported.safetensors

# Export to GGUF
hk export -f gguf model.hk exported.gguf
```

---

### 12. `run`
Runs fast command-line autoregressive inference:

```bash
hk run model.hk -p "Explain neural network weights" -n 128 --temp 0.7
```

Flags:
- `-p, --prompt`: Input text prompt.
- `-n, --n-predict`: Maximum tokens to generate (default: 128).
- `-ngl, --n-gpu-layers`: Number of layers to offload to GPU.
- `--temp`: Sampling temperature (default: `0.7`).
- `--top-p`: Nucleus sampling cutoff (default: `0.9`).
- `--top-k`: Top-K candidate pool size (default: `40`).

---

### 13. `chat`
Starts an interactive terminal chat session:

```bash
hk chat model.hk -ngl 16 --temp 0.7
```

---

### 14. `gui`
Launches the HK Model Studio GUI for visual inspection:

```bash
hk gui model.hk
```

Opens a visual window displaying tensor distribution, weight histograms, metadata tree, and alignment status.

---

### 15. `hash`
Calculates the streaming SHA-256 hash of the model container:

```bash
hk hash model.hk
```
