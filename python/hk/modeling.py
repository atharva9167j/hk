"""
HK Neural Modeling & AutoModel
Hugging Face-compatible transformer architectures, task heads, and native SIMD compute execution.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch import load_hk, save_hk
from .native import NativeHKReader as HKModelReader, NativeHKWriter as HKModelWriter
from .adaptive import (
    expand_vocab as adaptive_expand_vocab,
    expand_model_width as adaptive_expand_width,
    protect_base_capacity as adaptive_protect_base,
)

from .config import HKConfig, AutoConfig
from .format import StorageType
from .quantization import (
    quantize_q4_0,
    dequantize_q4_0,
    quantize_q8_0,
    dequantize_q8_0,
    quantize_q4_k,
    dequantize_q4_k,
)
from .native import (
    is_native_available,
    native_gemm,
    native_gemv,
    native_net2wider,
    native_net2deeper,
)


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class HKLinear(nn.Module):
    """Linear layer supporting standard PyTorch, native SIMD Zig dispatch, and QLoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_native = (device == "native") and is_native_available()

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # LoRA adapter slots
        self.has_lora = False
        self.lora_r = 0
        self.lora_alpha = 1.0
        self.lora_scaling = 1.0
        self.lora_A = None
        self.lora_B = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def attach_lora(self, r: int = 8, alpha: float = 16.0):
        self.has_lora = True
        self.lora_r = r
        self.lora_alpha = alpha
        self.lora_scaling = alpha / r
        # Freeze base weight
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If native device requested and running on CPU evaluation
        if self.use_native and not self.training and x.device.type == "cpu" and not self.has_lora:
            x_shape = x.shape
            x_flat = x.view(-1, self.in_features).detach().numpy()
            w_np = self.weight.detach().numpy()
            b_np = self.bias.detach().numpy() if self.bias is not None else None

            # Dispatch to native SIMD Zig GEMM
            # (x @ W.T) = (x[M, K] * W.T[K, N]) where K = in_features, N = out_features
            W_T = np.ascontiguousarray(w_np.T)
            out_np = native_gemm(x_flat, W_T)
            if b_np is not None:
                out_np += b_np
            out_tensor = torch.from_numpy(out_np).view(*x_shape[:-1], self.out_features)
            return out_tensor

        # PyTorch standard forward pass
        out = F.linear(x, self.weight, self.bias)
        if self.has_lora and self.lora_A is not None and self.lora_B is not None:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.lora_scaling
            out = out + lora_out
        return out


class HKQuantizedLinear(nn.Module):
    """
    Quantized Linear layer with packed weights (Q8_0, Q4_0, Q4_K, NF4).
    Executes native SIMD GEMV kernels without full FP32 weight matrix materialization in RAM/VRAM!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        storage_type: StorageType = StorageType.Q4_0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.storage_type = storage_type

        # Register packed weights as byte buffer
        self.register_buffer("packed_weight", torch.empty(0, dtype=torch.uint8))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # LoRA adapter slots
        self.has_lora = False
        self.lora_r = 0
        self.lora_alpha = 1.0
        self.lora_scaling = 1.0
        self.lora_A = None
        self.lora_B = None

    def attach_lora(self, r: int = 8, alpha: float = 16.0):
        self.has_lora = True
        self.lora_r = r
        self.lora_alpha = alpha
        self.lora_scaling = alpha / r
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        storage_type: StorageType = StorageType.Q4_0,
    ) -> "HKQuantizedLinear":
        qlin = cls(
            linear.in_features,
            linear.out_features,
            storage_type=storage_type,
            bias=linear.bias is not None,
        )
        if linear.bias is not None:
            qlin.bias.data.copy_(linear.bias.data)

        # Quantize row by row
        w_f32 = linear.weight.detach().cpu().to(torch.float32)
        packed_blocks = []

        if storage_type == StorageType.Q8_0:
            for r in range(linear.out_features):
                packed_blocks.append(quantize_q8_0(w_f32[r]))
        elif storage_type == StorageType.Q4_0:
            for r in range(linear.out_features):
                packed_blocks.append(quantize_q4_0(w_f32[r]))
        elif storage_type == StorageType.Q4_K:
            for r in range(linear.out_features):
                packed_blocks.append(quantize_q4_k(w_f32[r]))
        else:
            # Default to Q4_0
            for r in range(linear.out_features):
                packed_blocks.append(quantize_q4_0(w_f32[r]))

        all_bytes = b"".join(packed_blocks)
        qlin.packed_weight = torch.from_numpy(np.frombuffer(all_bytes, dtype=np.uint8).copy())
        return qlin

    def dequantize_weight(self) -> torch.Tensor:
        raw_bytes = bytes(self.packed_weight.cpu().numpy())
        rows = []
        if self.storage_type == StorageType.Q8_0:
            b_size = (self.in_features // 32) * 34
            for r in range(self.out_features):
                rows.append(dequantize_q8_0(raw_bytes[r * b_size : (r + 1) * b_size], [self.in_features]))
        elif self.storage_type == StorageType.Q4_0:
            b_size = (self.in_features // 32) * 18
            for r in range(self.out_features):
                rows.append(dequantize_q4_0(raw_bytes[r * b_size : (r + 1) * b_size], [self.in_features]))
        elif self.storage_type == StorageType.Q4_K:
            b_size = (self.in_features // 256) * 144
            for r in range(self.out_features):
                rows.append(dequantize_q4_k(raw_bytes[r * b_size : (r + 1) * b_size], [self.in_features]))
        else:
            raise NotImplementedError(f"Dequantize not supported for {self.storage_type}")
        return torch.stack(rows, dim=0)

    def get_dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = getattr(self, "_cached_weight", None)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        w = self.dequantize_weight().to(dtype=dtype, device=device)
        w.requires_grad = False
        self._cached_weight = w
        return self._cached_weight

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility property returning dequantized base weight (frozen / no gradient)."""
        w = self.dequantize_weight()
        w.requires_grad = False
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        in_dim = orig_shape[-1]
        assert in_dim == self.in_features, f"Expected input dim {self.in_features}, got {in_dim}"

        raw_bytes = bytes(self.packed_weight.cpu().numpy())
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None

        # Fast 1D / single-vector SIMD GEMV path without full dequantization (eval only)
        x_flat = x.view(-1, self.in_features)
        batch_size = x_flat.shape[0]

        from .native import (
            native_gemv_q8_0,
            native_gemv_q4_0,
            native_gemv_q4_k,
            is_native_available,
        )

        if is_native_available() and x.device.type == "cpu" and not self.training:
            out_rows = []
            for b in range(batch_size):
                x_vec = x_flat[b].detach().cpu().to(torch.float32).numpy()
                if self.storage_type == StorageType.Q8_0:
                    y = native_gemv_q8_0(raw_bytes, x_vec, bias_np, self.out_features, self.in_features)
                elif self.storage_type == StorageType.Q4_0:
                    y = native_gemv_q4_0(raw_bytes, x_vec, bias_np, self.out_features, self.in_features)
                elif self.storage_type == StorageType.Q4_K:
                    y = native_gemv_q4_k(raw_bytes, x_vec, bias_np, self.out_features, self.in_features)
                else:
                    break
                out_rows.append(torch.from_numpy(y))

            if len(out_rows) == batch_size:
                out = torch.stack(out_rows, dim=0).view(*orig_shape[:-1], self.out_features).to(dtype=x.dtype, device=x.device)
                if self.has_lora and self.lora_A is not None and self.lora_B is not None:
                    x_f32 = x.to(torch.float32)
                    lora_out = (x_f32 @ self.lora_A.t()) @ self.lora_B.t() * self.lora_scaling
                    out = out + lora_out.to(dtype=out.dtype)
                return out

        # Fast on-device path using cached dequantized weights (eliminates CPU roundtrips during QLoRA)
        w = self.get_dequantized_weight(x.device, x.dtype)
        out = F.linear(x, w, self.bias)
        if self.has_lora and self.lora_A is not None and self.lora_B is not None:
            x_f32 = x.to(torch.float32)
            lora_out = (x_f32 @ self.lora_A.t()) @ self.lora_B.t() * self.lora_scaling
            out = out + lora_out.to(dtype=out.dtype)
        return out


class HKTransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP feed-forward."""

    def __init__(self, config: HKConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Attention
        self.q_proj = HKLinear(config.hidden_size, config.hidden_size, bias=False, device=device)
        self.k_proj = HKLinear(config.hidden_size, config.hidden_size, bias=False, device=device)
        self.v_proj = HKLinear(config.hidden_size, config.hidden_size, bias=False, device=device)
        self.out_proj = HKLinear(config.hidden_size, config.hidden_size, bias=False, device=device)

        # Norms
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

        # MLP
        self.mlp_fc1 = HKLinear(config.hidden_size, config.intermediate_size, bias=True, device=device)
        self.mlp_fc2 = HKLinear(config.intermediate_size, config.hidden_size, bias=True, device=device)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Multi-head self-attention
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(normed).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_out = self.out_proj(attn_out)
        hidden_states = residual + attn_out

        # 2. MLP
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_act = F.gelu(self.mlp_fc1(normed))
        mlp_out = self.mlp_fc2(mlp_act)
        hidden_states = residual + mlp_out

        return hidden_states


class HKPreTrainedModel(nn.Module):
    """Base class for all HK pre-trained PyTorch models."""

    def __init__(self, config: HKConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        config: Optional[HKConfig] = None,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> "HKPreTrainedModel":
        path = Path(pretrained_model_name_or_path)
        model_file = None
        if path.is_dir():
            hk_files = list(path.glob("*.hk"))
            if hk_files:
                model_file = hk_files[0]
            else:
                st_files = list(path.glob("*.safetensors"))
                if st_files:
                    model_file = st_files[0]
                else:
                    raise FileNotFoundError(f"No .hk or .safetensors model file found in directory {path}")
        else:
            model_file = path

        if not model_file.is_file():
            raise FileNotFoundError(f"Model file {model_file} does not exist.")

        # Resolve target device and device map
        target_device = device
        dynamic_map: Optional[Dict[str, str]] = None
        if device_map is not None:
            if isinstance(device_map, str) and device_map.lower() in ("auto", "dynamic", "dynamic_offload"):
                from .offload import DynamicOffloadPlanner
                target_device = "cpu"
                if config is None:
                    config = AutoConfig.from_pretrained(model_file, **kwargs)
                dynamic_map = DynamicOffloadPlanner.create_device_map(
                    config,
                    torch_dtype=torch_dtype or torch.float16,
                    max_memory=kwargs.get("max_memory", None),
                )
            elif isinstance(device_map, dict):
                dynamic_map = device_map
                target_device = "cpu"
            elif isinstance(device_map, str):
                target_device = device_map

        if config is None:
            config = AutoConfig.from_pretrained(model_file, **kwargs)

        model = cls(config, device=target_device)
        model._model_file = str(model_file)

        # Load weights
        if model_file.suffix.lower() == ".safetensors":
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(str(model_file), device=target_device)
            except ImportError:
                raise ImportError("safetensors package required to load .safetensors models directly.")
        else:
            from .torch import load_file
            state_dict = load_file(str(model_file), device=target_device)

        model.load_state_dict(state_dict, strict=False)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        if dynamic_map is not None:
            from .offload import AutoDeviceDispatcher
            AutoDeviceDispatcher.apply_device_map(model, dynamic_map)
        elif target_device != "cpu":
            model = model.to(target_device)

        return model

    def to_dynamic_offload(
        self,
        safety_headroom_ratio: float = 0.0,
        min_vram_headroom_bytes: int = 128 * 1024 * 1024,  # 128 MB buffer memory
        max_memory: Optional[Dict[Any, Any]] = None,
    ) -> "HKPreTrainedModel":
        """
        Dynamically distributes model layers across available GPUs and CPU RAM based on
        live hardware VRAM headroom, automatically preventing CUDA Out-Of-Memory (OOM) errors.
        """
        from .offload import DynamicOffloadPlanner, AutoDeviceDispatcher
        param_dtype = next(self.parameters()).dtype if list(self.parameters()) else torch.float32
        device_map = DynamicOffloadPlanner.create_device_map(
            self.config,
            torch_dtype=param_dtype,
            safety_headroom_ratio=safety_headroom_ratio,
            min_vram_headroom_bytes=min_vram_headroom_bytes,
            max_memory=max_memory,
        )
        AutoDeviceDispatcher.apply_device_map(self, device_map)
        return self

    def save_pretrained(
        self,
        save_directory_or_path: Union[str, Path],
        alignment: int = 128,
        quantize_mode: str = "none",
        sparsity_mode: str = "none",
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Saves the model directly to the ultra-portable HK container format with optional embedded tokenizer."""
        target = Path(save_directory_or_path)
        if target.suffix.lower() == ".hk":
            output_path = target
        else:
            os.makedirs(target, exist_ok=True)
            output_path = target / "model.hk"
            # Also save config.json alongside
            self.config.save_pretrained(target)
            if tokenizer is not None:
                tokenizer.save_pretrained(target)

        metadata = {
            "model_type": self.config.model_type,
            "config": self.config.to_json_string(),
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "quantization": quantize_mode,
            "sparsity": sparsity_mode,
        }

        if tokenizer is not None:
            if hasattr(tokenizer, "export_to_metadata"):
                metadata.update(tokenizer.export_to_metadata())

        q_mode = None
        if quantize_mode in ("nf4", "dq4"):
            q_mode = "dq4"
        elif quantize_mode == "dq8":
            q_mode = "dq8"
        elif quantize_mode == "dqt":
            q_mode = "dqt"

        save_hk(
            str(output_path),
            state_dict=self.state_dict(),
            metadata=metadata,
            quantize_mode=q_mode,
            auto_pack_sparse=(sparsity_mode == "ampere_2_4"),
            alignment=alignment,
        )
        return str(output_path)

    def enable_qlora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        storage_type: StorageType = StorageType.Q4_0,
    ):
        """
        True QLoRA: Quantizes base weights into 4-bit packed buffers (reducing base weight memory/gradients)
        and attaches low-rank trainable adapters (lora_A, lora_B).
        """
        for p in self.parameters():
            p.requires_grad = False

        def replace_with_qlora(parent_module: nn.Module, prefix: str = ""):
            for child_name, child in list(parent_module.named_children()):
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                is_target = target_modules is None or any(t in full_name for t in target_modules)
                if isinstance(child, HKQuantizedLinear):
                    if is_target:
                        child.attach_lora(r=rank, alpha=alpha)
                elif isinstance(child, (HKLinear, nn.Linear)):
                    if is_target:
                        qlin = HKQuantizedLinear.from_float(child, storage_type=storage_type)
                        qlin.attach_lora(r=rank, alpha=alpha)
                        setattr(parent_module, child_name, qlin)
                else:
                    replace_with_qlora(child, full_name)

        replace_with_qlora(self)


class HKForCausalLM(HKPreTrainedModel):
    """Transformer language model with auto-regressive generation and Net2Net expansion."""

    def __init__(self, config: HKConfig, device: str = "cpu"):
        super().__init__(config)
        self.device_name = device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HKTransformerBlock(config, device=device) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = HKLinear(config.hidden_size, config.vocab_size, bias=False, device=device)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        batch_size, seq_len = input_ids.shape
        embed_device = self.embed_tokens.weight.device
        if input_ids.device != embed_device:
            input_ids = input_ids.to(embed_device)

        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask if not provided (strictly matching hidden_states device and dtype)
        if attention_mask is None and seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            attention_mask = causal_mask.view(1, 1, seq_len, seq_len)
        elif attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)

        for layer in self.layers:
            try:
                layer_dev = next(layer.parameters()).device
            except StopIteration:
                layer_dev = hidden_states.device

            if hidden_states.device != layer_dev:
                hidden_states = hidden_states.to(layer_dev, non_blocking=True)
            if attention_mask is not None and attention_mask.device != layer_dev:
                attention_mask = attention_mask.to(layer_dev, non_blocking=True)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        try:
            norm_dev = next(self.norm.parameters()).device
        except StopIteration:
            norm_dev = hidden_states.device

        if hidden_states.device != norm_dev:
            hidden_states = hidden_states.to(norm_dev, non_blocking=True)
        hidden_states = self.norm(hidden_states)

        try:
            head_dev = next(self.lm_head.parameters()).device
        except StopIteration:
            head_dev = hidden_states.device

        if hidden_states.device != head_dev:
            hidden_states = hidden_states.to(head_dev, non_blocking=True)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Greedy / Top-k auto-regressive text generation."""
        # Fast native Zig execution path if model is an .hk file on CPU
        model_file = getattr(self, "_model_file", None)
        if model_file and str(model_file).endswith(".hk") and input_ids.shape[0] == 1 and input_ids.device.type == "cpu":
            try:
                from .native import NativeHKEngine, is_native_available
                if is_native_available():
                    engine = NativeHKEngine(model_file)
                    ids_list = input_ids[0].tolist()
                    last_logits = None
                    for pos, tid in enumerate(ids_list):
                        last_logits = engine.forward_step(tid, pos)
                    pos = len(ids_list)
                    for _ in range(max_new_tokens):
                        if last_logits is None:
                            break
                        next_tok = int(np.argmax(last_logits))
                        ids_list.append(next_tok)
                        last_logits = engine.forward_step(next_tok, pos)
                        pos += 1
                    return torch.tensor([ids_list], dtype=input_ids.dtype, device=input_ids.device)
            except Exception:
                pass

        self.eval()
        curr_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self(curr_ids)
            next_token_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True).to(curr_ids.device)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
        return curr_ids

    def grow_width(self, layer_idx: int, new_intermediate_size: int, noise_std: float = 0.0):
        """Dynamically expands the MLP intermediate dimension using native Zig Net2WiderNet."""
        assert 0 <= layer_idx < len(self.layers)
        layer = self.layers[layer_idx]

        w1_old = layer.mlp_fc1.weight.detach().cpu().numpy()
        b1_old = layer.mlp_fc1.bias.detach().cpu().numpy() if layer.mlp_fc1.bias is not None else None
        w2_old = layer.mlp_fc2.weight.detach().cpu().numpy()

        w1_new, b1_new, w2_new = native_net2wider(
            w_in_old=w1_old,
            b_in_old=b1_old,
            new_out=new_intermediate_size,
            w_out_old=w2_old,
            noise_std=noise_std,
            seed=42,
        )

        # Construct new linear modules
        new_fc1 = HKLinear(self.config.hidden_size, new_intermediate_size, bias=b1_old is not None, device=self.device_name)
        new_fc1.weight = nn.Parameter(torch.from_numpy(w1_new))
        if b1_new is not None:
            new_fc1.bias = nn.Parameter(torch.from_numpy(b1_new))

        new_fc2 = HKLinear(new_intermediate_size, self.config.hidden_size, bias=layer.mlp_fc2.bias is not None, device=self.device_name)
        new_fc2.weight = nn.Parameter(torch.from_numpy(w2_new))
        if layer.mlp_fc2.bias is not None:
            new_fc2.bias = nn.Parameter(layer.mlp_fc2.bias.clone())

        layer.mlp_fc1 = new_fc1
        layer.mlp_fc2 = new_fc2
        self.config.intermediate_size = new_intermediate_size

    def grow_depth(self, insert_after_layer_idx: int):
        """Dynamically inserts a new identity transformer block using Net2DeeperNet."""
        assert 0 <= insert_after_layer_idx < len(self.layers)
        new_block = HKTransformerBlock(self.config, device=self.device_name)

        # Zero out the second MLP linear layer and attention output projection so f(x) = x upon insertion
        nn.init.zeros_(new_block.mlp_fc2.weight)
        if new_block.mlp_fc2.bias is not None:
            nn.init.zeros_(new_block.mlp_fc2.bias)
        nn.init.zeros_(new_block.out_proj.weight)

        self.layers.insert(insert_after_layer_idx + 1, new_block)
        self.config.num_hidden_layers = len(self.layers)

    def expand_vocab(self, new_vocab_size: int, init_std: float = 0.02) -> "HKForCausalLM":
        """Expands embedding table and LM head while preserving all existing token logits."""
        return adaptive_expand_vocab(self, new_vocab_size=new_vocab_size, init_std=init_std)

    def expand_width(
        self,
        expansion_ratio: float = 1.33,
        layers: Union[str, List[int]] = "all",
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> "HKForCausalLM":
        """Model-wide Net2WiderNet capacity expansion with exact function preservation."""
        layer_indices = None if layers == "all" else layers
        return adaptive_expand_width(
            self,
            expansion_ratio=expansion_ratio,
            noise_std=noise_std,
            layer_indices=layer_indices,
            seed=seed,
        )

    def enable_continual_learning(self, protect_base: bool = True) -> List[Any]:
        """
        Activates plasticity isolation (anti-catastrophic forgetting).
        Zeros out gradients on pre-expansion base weights while permitting 100% learning
        on newly expanded capacity and new vocabulary tokens.
        """
        if protect_base:
            return adaptive_protect_base(self)
        return []

    def create_self_training_pipeline(
        self,
        hk_file_path: str,
        governor: Optional[Any] = None,
        learning_rate: float = 3e-3,
    ) -> Any:
        """Creates an autonomous SelfTrainingPipeline bound to this model."""
        from .adaptive import SelfTrainingPipeline
        return SelfTrainingPipeline(self, hk_file_path=hk_file_path, governor=governor, learning_rate=learning_rate)


class HKForSequenceClassification(HKPreTrainedModel):
    """Transformer model with classification head for NLP classification / sentiment analysis."""

    def __init__(self, config: HKConfig, device: str = "cpu"):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HKTransformerBlock(config, device=device) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.classifier = HKLinear(config.hidden_size, config.num_classes, bias=True, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        # Mean pooling across tokens
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(loss=loss, logits=logits)


class HKForHandwritingRecognition(HKPreTrainedModel):
    """Multi-device OCR & Handwriting Recognition model with CNN backbone and CTC / projection head."""

    def __init__(self, config: HKConfig, device: str = "cpu"):
        super().__init__(config)
        in_c = getattr(config, "in_channels", 1)
        hidden_dim = config.hidden_size

        # Lightweight CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.proj = HKLinear(hidden_dim, config.num_classes, bias=True, device=device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        # pixel_values: [batch, in_channels, H, W]
        features = self.cnn(pixel_values)  # [batch, hidden_dim, H', W']
        # Pool height dimension
        features = features.mean(dim=2).permute(0, 2, 1)  # [batch, W', hidden_dim]
        logits = self.proj(features)

        loss = None
        if labels is not None:
            # Flatten classification loss
            loss = F.cross_entropy(logits.view(-1, self.config.num_classes), labels.view(-1))

        return ModelOutput(loss=loss, logits=logits)


class AutoModel:
    """Automatic model factory supporting all HK model architectures."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs) -> HKPreTrainedModel:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_type = getattr(config, "model_type", "causal_lm").lower()

        if model_type in ("causal_lm", "text_generation", "lm"):
            return HKForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        elif model_type in ("sequence_classification", "classification", "sentiment_analysis"):
            return HKForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        elif model_type in ("handwriting_recognition", "hwr", "ocr"):
            return HKForHandwritingRecognition.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        else:
            return HKForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
