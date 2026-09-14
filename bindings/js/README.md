# @hk-format/core (HK TypeScript / JavaScript SDK)

Zero-dependency TypeScript & JavaScript SDK for the **HK Neural Tensor Framework**.  
Provides fast container reading, memory-mapped tensor access, GGUF/Safetensors compatibility, and browser/Node.js/WASM runtime support.

---

## Features

- **Non-Quantized Storage Efficiency**: Direct memory-mapped zero-copy access to FP32, FP16, BF16, and FP8 unquantized tensors with zero deserialization overhead.
- **Zero-Dependency**: Reads `.hk` containers directly in browser, Web Worker, or Node.js without native binary dependencies.
- **Minimal Container Overhead**: Fixed 128-byte header and binary TOC inspection in sub-milliseconds.
- **Comprehensive Quantization Support**: Full support for quantized models (Q4_0, Q8_0, K-quants, I-quants) for edge and mobile execution.
- **Typed & Pure**: Full TypeScript typings (`.d.ts`), tree-shakeable, and ESM/CJS compatible.

---

## Installation

```bash
npm install @hk-format/core
# or
yarn add @hk-format/core
# or
pnpm add @hk-format/core
```

---

## Usage

### In Node.js / Server-side

```typescript
import * as fs from "node:fs";
import { HkModel, StorageType } from "@hk-format/core";

// Read a local .hk container file
const buffer = fs.readFileSync("model.hk");
const model = HkModel.fromArrayBuffer(buffer.buffer);

// Inspect metadata
console.log("Model Architecture:", model.getMetadata("general.architecture"));
console.log("All Metadata:", model.getAllMetadata());

// List all tensors
for (const tensor of model.listTensors()) {
  console.log(`Tensor: ${tensor.name}, Shape: [${tensor.shape}], Type: ${StorageType[tensor.storageType]}`);
}

// Extract a tensor's raw binary data
const weightBytes = model.getTensorBytes("layers.0.feed_forward.w1.weight");
```

### In Browser / Web Workers

```typescript
import { HkModel } from "@hk-format/core";

// Fetch container header via HTTP range request
const response = await fetch("https://huggingface.co/org/model/resolve/main/model.hk");
const arrayBuffer = await response.arrayBuffer();

const model = HkModel.fromArrayBuffer(arrayBuffer);
console.log("Loaded model with tensors:", model.tensorNames);
```

---

## License

Apache-2.0 © HK AI Research Team
