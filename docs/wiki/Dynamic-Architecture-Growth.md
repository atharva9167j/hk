# Dynamic Architecture Growth (Net2Net)

In this guide, I explain how I designed HK to enable dynamic model expansion, allowing you to widen layers, add depth, and grow vocabulary on the fly while mathematically preserving existing model outputs.

---

## The Problem with Static Models

In traditional deep learning, once a model finishes pre-training, its architecture is frozen forever:
- If your 1B model struggles to understand code or medical texts, you cannot simply add 200 million parameters to its intermediate layers.
- If you need to add syntax tokens for a new programming language, modifying the embedding table corrupts vocabulary offsets.
- If you attempt to train new layers naively, the new parameters start with random values, breaking the model's activations and causing catastrophic forgetting.
- If you try to guess how much bigger you can make a model on your hardware, you often face frustrating out-of-memory (OOM) crashes after hours of setup.

I solved this in HK with function-preserving dynamic growth routines combined with a native hardware resource governor.

---

## 1. Function-Preserving Net2WiderNet (SwiGLU & Linear)

Net2WiderNet allows you to widen the intermediate dimension of feed-forward networks (such as SwiGLU MLPs in Llama, Qwen, and Mistral) or attention projection matrices during training or fine-tuning.

### The Mathematical Guarantee
When widening a layer from dimension $D_{\text{old}}$ to $D_{\text{new}}$ in HK, the routine copies the original weights into the upper-left partition and initializes the newly added neurons such that the initial output is mathematically identical to the unexpanded model:

$$f_{\text{new}}(x) \equiv f_{\text{old}}(x)$$

On Day 0 (the moment of expansion), the output deviation is exactly **0.000000**. The model produces the exact same logits and predictions as it did before widening. It then has extra capacity to absorb new training data without forgetting its base knowledge.

### Expanding a SwiGLU MLP in Python
```python
import torch
from hk.adaptive.growth import net2wider_swiglu

# Assume layer has gate_proj (hidden -> intermediate), up_proj, down_proj
mlp = model.model.layers[0].mlp

# Current intermediate size: 1536
# Target intermediate size: 2048
wider_gate, wider_up, wider_down = net2wider_swiglu(
    gate_proj=mlp.gate_proj,
    up_proj=mlp.up_proj,
    down_proj=mlp.down_proj,
    new_intermediate_size=2048,
    noise_std=0.0,  # 0.0 guarantees exact mathematical function preservation
)

# Assign widened weights back to model
mlp.gate_proj = wider_gate
mlp.up_proj = wider_up
mlp.down_proj = wider_down

print("SwiGLU MLP successfully expanded to 2048 intermediate units.")
```

---

## 2. Function-Preserving Net2DeeperNet

Net2DeeperNet increases the depth of a neural network by inserting new layers.

To prevent destroying the model's forward pass, HK inserts **Modular Residual Blocks** (`ModularResidualBlock`). These blocks are initialized as identity operations:

$$x_{\text{out}} = x + 0 \cdot g(x) = x$$

Because the transformation initially evaluates to an identity pass, inserting three or four new adapter blocks into an existing transformer stack does not alter the model's predictions. As training begins, the new blocks gradually learn specialized residual representations.

---

## 3. Dynamic Vocabulary Expansion

When teaching a model a new domain (such as legal texts or a new programming language like Zig), existing tokenizers often split domain keywords into 4 or 5 fragmented byte tokens. This bloats context windows and degrades comprehension.

HK allows you to expand the vocabulary dynamically:

```python
from hk.adaptive.growth import expand_vocab

# Expand token embedding matrix from 32,000 to 32,500 tokens
new_embed, new_head = expand_vocab(
    embed_tokens=model.model.embed_tokens,
    lm_head=model.lm_head,
    new_vocab_size=32500,
    init_std=0.02,
)

model.model.embed_tokens = new_embed
model.lm_head = new_head
```

Pre-existing token weights and IDs are preserved with bit-exact precision. Newly introduced token embeddings are initialized with Gaussian noise centered on existing embedding statistics, ready to learn new domain tokens.

---

## 4. Hardware Safety: Native Zig `GrowthGovernor`

A major hazard of dynamic expansion is running out of physical RAM or VRAM midway through a run.

I prevented this in HK with the **GrowthGovernor**, which I implemented in compiled native Zig (`hk_governor_can_grow_batch`). The governor evaluates capacity expansion constraints against physical system RAM, GPU VRAM, and maximum growth ratios in microseconds.

In benchmarks, the native governor evaluates **50,000 capacity constraints in 3.57 milliseconds** (around 71 nanoseconds per decision).

```python
from hk.adaptive.growth import GrowthGovernor

# Set physical memory boundary (e.g. 6 GB VRAM on a laptop GPU)
governor = GrowthGovernor(max_vram_mb=6144, max_growth_ratio=1.5)

current_params = 135_000_000
additional_params = 25_000_000

# Check if expansion fits within physical memory
allowed, reason = governor.can_grow(current_params, additional_params, dtype_bytes=2)
if allowed:
    print("Expansion approved by GrowthGovernor. Safe to proceed.")
else:
    print(f"Expansion rejected: {reason}")
```

---

## 5. Plasticity Isolation (Anti-Catastrophic Forgetting)

When training an expanded model, you do not want the new gradient updates to overwrite the pre-existing representations that the model spent millions of steps learning.

I built in **Plasticity Isolation** to prevent this:
- It generates a gradient mask that freezes or dampens updates to original neuron indices (`protect_base_capacity = True`).
- Gradient updates flow predominantly or exclusively into newly added rows and columns.
- The model learns new capabilities while the original capabilities remain shielded.

---

## 6. Expanding via the CLI

You can also widen an existing `.hk` model directly from your command line without writing any Python:

```bash
# Widen feed-forward layers by 33% and expand vocabulary to 32,000
hk expand input_model.hk output_model.hk --width 1.33 --vocab 32000
```

The output file is a valid `.hk` container with updated dimensions and aligned weights, ready for training or immediate inference.
