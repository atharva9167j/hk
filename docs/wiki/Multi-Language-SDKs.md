# Multi-Language SDK Ecosystem

I built native, zero-overhead SDKs across 7 major programming environments for HK. Every SDK connects directly to my compiled Zig SIMD engine through a standardized C-ABI bridge with zero memory copies.

---

## 1. Rust SDK (`bindings/rust/`)

The Rust crate provides safe, idiomatic abstractions with compile-time lifetime and bounds guarantees.

### Adding Dependency
```toml
[dependencies]
hk-tensor = { path = "bindings/rust" }
```

### Example Usage
```rust
use hk_tensor::{HkModel, HkContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open model container
    let model = HkModel::open("model.hk")?;
    println!("Model vocab size: {}", model.vocab_size());

    // Create execution context with GPU offloading
    let mut ctx = HkContext::new(&model, 16)?; // 16 layers to GPU

    // Step autoregressive forward pass
    let token_id = 1;
    let pos = 0;
    let logits = ctx.forward_step(token_id, pos)?;
    println!("First logit: {}", logits[0]);

    Ok(())
}
```

---

## 2. TypeScript and Node.js (`bindings/js/`)

The JavaScript / TypeScript SDK runs in Node.js, modern browsers, and WebAssembly runtimes with zero external npm dependencies.

### Installation
```bash
npm install hknt
```

### Example Usage
```typescript
import { HkModel, safeOpen } from 'hknt';

async function run() {
    // Open model container
    const model = await HkModel.open('model.hk');
    console.log(`Loaded model with ${model.tensorCount} tensors.`);

    // Inspect tensor metadata
    const info = model.getTensorInfo('model.layers.0.mlp.gate_proj.weight');
    console.log(`Tensor shape: ${info.shape}, Storage: ${info.storageType}`);

    // Read lazy slice
    const slice = await model.readSlice('model.layers.0.mlp.gate_proj.weight', [0, 0], [64, 128]);
    console.log(`Fetched slice bytes: ${slice.byteLength}`);
}

run();
```

---

## 3. C# and .NET 9 (`bindings/csharp/`)

The .NET SDK targets modern C# 12 and .NET 9, ideal for Unity game engines, desktop apps, and enterprise backends.

### Example Usage
```csharp
using System;
using Hk;

class Program {
    static void Main() {
        using var model = new HkModel("model.hk");
        Console.WriteLine($"Model Architecture: {model.Architecture}");
        Console.WriteLine($"Vocab Size: {model.VocabSize}");

        using var ctx = model.CreateContext(n_gpu_layers: 24);
        float[] logits = ctx.ForwardStep(token: 1, pos: 0);
        Console.WriteLine($"Logits evaluated. Logit[0] = {logits[0]}");
    }
}
```

---

## 4. Go Module (`bindings/go/`)

The Go SDK provides idiomatic Go packages using `cgo` for integration with microservices and server applications.

### Example Usage
```go
package main

import (
    "fmt"
    "github.com/hk/bindings/go/hk"
)

func main() {
    model, err := hk.Open("model.hk")
    if err != nil {
        panic(err)
    }
    defer model.Close()

    fmt.Printf("Model loaded: %d tensors\n", model.TensorCount())

    ctx, err := model.NewContext(16)
    if err != nil {
        panic(err)
    }
    defer ctx.Close()

    logits, err := ctx.ForwardStep(1, 0)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Logit sample: %f\n", logits[0])
}
```

---

## 5. Java and Android (`bindings/java/`)

The Java SDK uses JNI with direct NIO `ByteBuffer` pointers, allowing zero-copy buffer views directly on Android devices and enterprise JVM backends.

### Example Usage
```java
import com.hk.HkModel;
import com.hk.HkContext;

public class Main {
    public static void main(String[] args) {
        try (HkModel model = HkModel.open("model.hk")) {
            System.out.println("Model tensors: " + model.getTensorCount());
            
            try (HkContext ctx = model.createContext(0)) { // 0 for CPU
                float[] logits = ctx.forwardStep(1, 0);
                System.out.println("Logits evaluated: " + logits.length);
            }
        }
    }
}
```

---

## 6. C and C++ (`include/hk.h`, `include/hk.hpp`)

Direct C99 ABI headers and C++20 RAII wrappers for embedding in custom game engines, robotics frameworks, or edge microcontrollers.

### C Example
```c
#include "hk.h"
#include <stdio.h>

int main() {
    hk_model_t* model = hk_model_open("model.hk");
    if (!model) return 1;

    printf("Tensors in container: %llu\n", hk_model_tensor_count(model));

    hk_context_t* ctx = hk_context_create(model, 0);
    float* logits = hk_forward_step(ctx, 1, 0);
    printf("Logit 0: %f\n", logits[0]);

    hk_context_free(ctx);
    hk_model_close(model);
    return 0;
}
```

### C++20 RAII Example
```cpp
#include "hk.hpp"
#include <iostream>

int main() {
    hk::Model model("model.hk");
    std::cout << "Model architecture: " << model.architecture() << "\n";

    auto ctx = model.create_context(/*gpu_layers=*/32);
    auto logits = ctx.forward_step(1, 0);
    std::cout << "Logit 0: " << logits[0] << "\n";
    return 0;
}
```
