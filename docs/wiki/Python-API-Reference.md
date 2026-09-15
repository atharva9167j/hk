# Python API Reference

In this document, I provide a comprehensive reference for the primary Python modules, classes, and functions in the `hk` package (`hknt` on PyPI).

---

## 1. Top-Level Package (`hk`)

### `AutoModelForCausalLM`
Factory class for loading causal language models from `.hk` containers.
- `from_pretrained(pretrained_model_name_or_path, torch_dtype="bfloat16", **kwargs) -> HKForCausalLM`: Loads model using zero-copy memory-mapped tensors.

### `AutoTokenizer`
Factory class for tokenizers embedded in `.hk` containers.
- `from_pretrained(pretrained_model_name_or_path, **kwargs) -> HKTokenizer`: Loads tokenizer vocabulary, merge rules, and special token mappings.

### `AutoConfig` / `HKConfig`
Model architecture hyperparameter configuration.
- Fields: `model_type`, `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`, `rope_theta`, `rms_norm_eps`.
- `from_pretrained(path) -> HKConfig`: Reads config dictionary from `.hk` metadata section.
- `save_pretrained(path)`: Serializes config into metadata section.

### `HKForCausalLM`
PyTorch `nn.Module` implementation supporting causal language modeling, autoregressive generation, and dynamic capacity expansion.
- `generate(input_ids, max_new_tokens=128, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1, do_sample=True, **kwargs) -> torch.Tensor`: Autoregressive token generation loop.
- `save_pretrained(save_directory, metadata=None)`: Saves weights directly into `.hk` container.

---

## 2. PyTorch Integration (`hk.torch`)

I designed this as a drop-in replacement for `safetensors.torch` with automatic tied-weight deduplication and memory leak protection.

### Functions
- `save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Optional[Dict[str, str]] = None) -> None`: Serializes dictionary of PyTorch tensors into an `.hk` container. Automatically detects shared `data_ptr` addresses and marks them as `SHARED_REF` without duplicate storage.
- `load_file(filename: str, device: str = "cpu") -> Dict[str, torch.Tensor]`: Loads all tensors from an `.hk` file using zero-copy memory mapping.
- `save_model(model: torch.nn.Module, filename: str, metadata: Optional[Dict[str, str]] = None) -> None`: Extracts state dictionary and persists model weights.
- `load_model(model: torch.nn.Module, filename: str, strict: bool = True) -> None`: Loads `.hk` weights into an existing `nn.Module` instance.

### `safe_open` Class
Context manager for lazy inspection and partial tensor slicing.
```python
with safe_open(filename: str, framework: str = "pt") as f:
    keys = f.keys()
    metadata = f.metadata()
    tensor = f.get_tensor(name: str)
    slice_obj = f.get_slice(name: str)
    # Multidimensional slicing without loading full matrix:
    partial_tensor = slice_obj[0:10, 0:20]
```

---

## 3. Raw Weight Store (`hk.raw`)

Zero compute headroom storage and direct SIMD execution.

### `HKRawWeightStore`
- `load_raw(filepath: str) -> HKRawWeightStore`: Maps an `.hk` file directly into user address space.
- Attributes:
  - `is_universal_page_aligned`: True if all payloads start on 4KB or 16KB boundaries.
  - `is_tensor_core_aligned`: True if all payloads start on 128-byte boundaries.
  - `tensor_names`: List of stored tensor identifiers.
- Methods:
  - `gemv(tensor_name: str, x: torch.Tensor) -> torch.Tensor`: Executes direct 4-row unrolled SIMD matrix-vector multiplication $y = W \cdot x$ directly from mapped pages.
  - `get_tensor(tensor_name: str) -> torch.Tensor`: Returns a zero-copy PyTorch tensor viewing the mapped memory.
  - `close() -> None`: Safely closes file descriptors and unmaps memory.

### Functions
- `save_raw(tensors: Dict[str, torch.Tensor], filepath: str, universal_alignment: bool = True) -> None`: Saves unquantized IEEE tensors with 4KB/16KB universal alignment.
- `save_sharded_raw(tensors: Dict, output_dir: str, max_shard_size_gb: float = 2.0) -> None`: Splits large models across multi-file shards with global manifest index.
- `load_sharded_raw(index_or_dir: str) -> HKShardedRawStore`: Loads a sharded checkpoint.

---

## 4. Dynamic Architecture Growth (`hk.adaptive.growth`)

Functions and classes for runtime capacity expansion.

### `GrowthGovernor`
Hardware resource manager that evaluates physical RAM and GPU VRAM boundaries in native Zig.
- `__init__(max_vram_mb: int = 8192, max_ram_mb: Optional[int] = None, max_growth_ratio: float = 2.0)`
- `can_grow(current_params: int, additional_params: int, dtype_bytes: int = 2) -> Tuple[bool, str]`: Checks whether a proposed parameter expansion is safe to allocate.

### Growth Routines
- `net2wider_swiglu(gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, new_intermediate_size: int, noise_std: float = 0.0) -> Tuple[nn.Linear, nn.Linear, nn.Linear]`: Widens SwiGLU MLP intermediate dimension with exact Day-0 function preservation ($f_{\text{new}}(x) \equiv f_{\text{old}}(x)$ when `noise_std = 0.0`).
- `net2wider_linear(linear: nn.Linear, new_out_features: int, noise_std: float = 0.0) -> nn.Linear`: Widens standard linear layers.
- `expand_vocab(embed_tokens: nn.Embedding, lm_head: nn.Linear, new_vocab_size: int, init_std: float = 0.02) -> Tuple[nn.Embedding, nn.Linear]`: Expands token vocabulary while preserving all existing token embeddings.

---

## 5. Training Engine (`hk.trainer`)

### `HKTrainingArguments`
Dataclass holding all training hyperparameters:
- `output_dir: str = "./results"`
- `learning_rate: float = 3e-4`
- `batch_size: int = 4`
- `num_train_epochs: int = 3`
- `use_qlora: bool = False`
- `lora_rank: int = 8`
- `lora_alpha: float = 16.0`
- `enable_adaptive_growth: bool = False`
- `growth_patience: int = 5`
- `growth_width_factor: float = 1.25`
- `protect_base_capacity: bool = True`

### `HKTrainer`
Hugging Face-compatible unified training loop:
- `__init__(model, args: HKTrainingArguments, train_dataset, eval_dataset=None, tokenizer=None)`
- `train() -> Dict[str, float]`: Runs training, evaluates validation loss, triggers Net2Net expansion on plateaus, and saves in-container checkpoints.

---

## 6. Version Lineage (`hk.adaptive.appendix`)

### `AppendixManager`
Manages the append-only version DAG embedded at the end of `.hk` files.
- `__init__(hk_path: str)`
- `list_records() -> List[AppendixRecord]`: Returns full generational history.
- `append_record(record_type: int, generation: int, metrics: Dict, payload: bytes) -> str`: Appends a new generation delta record with cryptographic SHA-256 parent hash.
- `verify_chain_integrity() -> Tuple[bool, str]`: Validates cryptographic hashes across all generations.
- `rollback_to_generation(target_generation: int) -> bool`: Restores container active state to an earlier generation in sub-milliseconds.

---

## 7. Autonomous Self-Training (`hk.adaptive`)

- `SelfTrainingPipeline`: Orchestrates closed-loop self-learning runs.
- `SelfTrainingCurriculum`: Holds diagnostic and training task definitions.
- `SelfConversationalEngine`: Manages Proposer/Thinker dialogue and `<think>` reasoning traces.
- `CodeSandbox`: Secure, AST-screened multiprocess execution sandbox with timeouts.
- `SelfPlayEngine` / `SPINLoss`: Computes self-play fine-tuning objectives.
- `ExpansionEvaluator`: Probes error rates to formulate architectural growth plans.

---

## 8. Structural Sparsity (`hk.pruning`)

- `prune_to_2_4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Prunes dense matrix to conform to Ampere 2:4 structured sparsity pattern.
- `pack_2_4_sparse(weight: torch.Tensor) -> Packed24Tensor`: Packs 2:4 sparse float matrix into contiguous non-zeros and 2-bit nibble index buffers (50% storage reduction).
- `unpack_2_4_sparse(packed: Packed24Tensor) -> torch.Tensor`: Unpacks 2:4 sparse buffer back into dense tensor with 0.000000 error.
