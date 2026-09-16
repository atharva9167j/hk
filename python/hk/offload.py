"""
HK Dynamic GPU/CPU Offloading Engine
Automated multi-device placement, memory budgeting, pipeline activation dispatch,
and dynamic OOM prevention for large model serving.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import os
import torch
import torch.nn as nn

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .config import HKConfig

logger = logging.getLogger("hk.offload")


@dataclass
class DeviceMemoryInfo:
    """Memory status for a compute device."""
    device: str
    device_type: str  # 'cuda' or 'cpu'
    device_index: Optional[int]
    total_bytes: int
    free_bytes: int
    usable_bytes: int


class HardwareMemoryInspector:
    """Inspects hardware topology and real-time available memory across all GPUs and host RAM."""

    @staticmethod
    def get_available_devices() -> List[str]:
        """Returns ordered list of available compute devices: cuda:0, cuda:1, ..., cpu."""
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        devices.append("cpu")
        return devices

    @staticmethod
    def get_device_memory(
        device: str,
        safety_headroom_ratio: float = 0.0,
        min_vram_headroom_bytes: int = 128 * 1024 * 1024,  # 128 MB buffer memory
    ) -> DeviceMemoryInfo:
        """
        Queries free and total memory for the given device.
        Applies dynamic buffer memory headroom (128 MB) to reserve space for activations,
        prefill buffers, and KV cache while maximizing on-device weight capacity.
        """
        device_str = str(device).lower()
        if device_str.startswith("cuda"):
            dev_idx = 0
            if ":" in device_str:
                try:
                    dev_idx = int(device_str.split(":")[1])
                except ValueError:
                    dev_idx = 0

            # Synchronize and get exact CUDA memory info
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev_idx)
                free_b, total_b = torch.cuda.mem_get_info(dev_idx)
            else:
                free_b, total_b = 0, 0

            # Compute safe usable weight budget reserving 128 MB buffer
            headroom = max(int(total_b * safety_headroom_ratio), min_vram_headroom_bytes)
            usable_b = max(0, free_b - headroom)

            return DeviceMemoryInfo(
                device=f"cuda:{dev_idx}",
                device_type="cuda",
                device_index=dev_idx,
                total_bytes=total_b,
                free_bytes=free_b,
                usable_bytes=usable_b,
            )
        else:
            # Host CPU memory
            if HAS_PSUTIL:
                vm = psutil.virtual_memory()
                total_b = vm.total
                free_b = vm.available
            else:
                # Fallback: estimate 32 GB total, 16 GB free
                total_b = 32 * 1024 * 1024 * 1024
                free_b = 16 * 1024 * 1024 * 1024

            # For CPU, leave at least 2 GB for OS/runtime
            usable_b = max(0, free_b - 2 * 1024 * 1024 * 1024)

            return DeviceMemoryInfo(
                device="cpu",
                device_type="cpu",
                device_index=None,
                total_bytes=total_b,
                free_bytes=free_b,
                usable_bytes=usable_b,
            )


class LayerMemoryEstimator:
    """Estimates static parameter footprints for each sub-module of a transformer model."""

    @staticmethod
    def get_dtype_bytes(dtype: Optional[torch.dtype]) -> int:
        if dtype in (torch.float32, torch.int32):
            return 4
        elif dtype in (torch.float16, torch.bfloat16, torch.int16):
            return 2
        elif dtype in (torch.int8, torch.uint8):
            return 1
        elif dtype is None:
            return 2  # Default to FP16 assumption
        return 2

    @classmethod
    def estimate_module_footprints(
        cls,
        config: HKConfig,
        torch_dtype: Optional[torch.dtype] = torch.float16,
    ) -> Dict[str, int]:
        """
        Calculates exact byte footprint for:
        - embed_tokens
        - layers.0 ... layers.(N-1)
        - norm
        - lm_head
        """
        elem_size = cls.get_dtype_bytes(torch_dtype)
        hidden = config.hidden_size
        vocab = config.vocab_size
        inter = config.intermediate_size
        num_layers = config.num_hidden_layers

        # Embedding: vocab_size * hidden_size
        embed_bytes = vocab * hidden * elem_size

        # Single Transformer Layer:
        # Self-Attention: Q, K, V, O projections -> 4 * hidden * hidden
        # MLP: Gate, Up, Down projections -> 3 * hidden * inter
        # LayerNorms: input_layernorm + post_attention_layernorm -> 2 * hidden
        layer_params = (4 * hidden * hidden) + (3 * hidden * inter) + (2 * hidden)
        layer_bytes = layer_params * elem_size

        # Final Norm: hidden_size
        norm_bytes = hidden * elem_size

        # LM Head: hidden_size * vocab_size (if tied, 0 extra storage)
        if config.tie_word_embeddings:
            lm_head_bytes = 0
        else:
            lm_head_bytes = hidden * vocab * elem_size

        footprints: Dict[str, int] = {
            "embed_tokens": embed_bytes,
        }
        for i in range(num_layers):
            footprints[f"layers.{i}"] = layer_bytes

        footprints["norm"] = norm_bytes
        footprints["lm_head"] = lm_head_bytes
        return footprints


class DynamicOffloadPlanner:
    """
    Automated dynamic layer-to-device partitioner.
    Eliminates manual layer count calculations and dynamically distributes weights
    across GPUs and CPU RAM based on live hardware memory headroom.
    """

    @classmethod
    def create_device_map(
        cls,
        config: HKConfig,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        safety_headroom_ratio: float = 0.0,
        min_vram_headroom_bytes: int = 128 * 1024 * 1024,  # 128 MB buffer memory
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        force_cpu: bool = False,
    ) -> Dict[str, str]:
        """
        Generates an automated, zero-configuration device map dictionary.
        Format:
            {
                "embed_tokens": "cuda:0",
                "layers.0": "cuda:0",
                ...,
                "layers.7": "cuda:0",
                "layers.8": "cuda:1",
                ...,
                "norm": "cuda:1",
                "lm_head": "cuda:1",
            }
        """
        num_layers = config.num_hidden_layers
        device_map: Dict[str, str] = {}

        if force_cpu or not torch.cuda.is_available():
            device_map["embed_tokens"] = "cpu"
            for i in range(num_layers):
                device_map[f"layers.{i}"] = "cpu"
            device_map["norm"] = "cpu"
            device_map["lm_head"] = "cpu"
            return device_map

        # Parse user max_memory overrides if provided
        user_budgets: Dict[str, int] = {}
        if max_memory:
            for k, v in max_memory.items():
                dev_key = f"cuda:{k}" if isinstance(k, int) else str(k).lower()
                if isinstance(v, str):
                    v_str = v.strip().upper()
                    if v_str.endswith("GB") or v_str.endswith("GIB"):
                        user_budgets[dev_key] = int(float(v_str.replace("GIB", "").replace("GB", "")) * 1024**3)
                    elif v_str.endswith("MB") or v_str.endswith("MIB"):
                        user_budgets[dev_key] = int(float(v_str.replace("MIB", "").replace("MB", "")) * 1024**2)
                    else:
                        user_budgets[dev_key] = int(v_str)
                else:
                    user_budgets[dev_key] = int(v)

        # Collect usable budgets for all available devices
        devices = HardwareMemoryInspector.get_available_devices()
        usable_budgets: Dict[str, int] = {}
        for dev in devices:
            mem_info = HardwareMemoryInspector.get_device_memory(
                dev,
                safety_headroom_ratio=safety_headroom_ratio,
                min_vram_headroom_bytes=min_vram_headroom_bytes,
            )
            if dev in user_budgets:
                usable_budgets[dev] = min(mem_info.free_bytes, user_budgets[dev])
            else:
                usable_budgets[dev] = mem_info.usable_bytes

        # Calculate module footprints
        footprints = LayerMemoryEstimator.estimate_module_footprints(config, torch_dtype=torch_dtype)

        gpu_devices = [d for d in devices if d.startswith("cuda")]
        if not gpu_devices:
            gpu_devices = ["cpu"]

        # Water-filling greedy allocation
        current_gpu_idx = 0
        current_device = gpu_devices[current_gpu_idx]

        # 1. Place embed_tokens on GPU 0 if budget allows, else CPU
        embed_fp = footprints["embed_tokens"]
        if usable_budgets.get(current_device, 0) >= embed_fp:
            device_map["embed_tokens"] = current_device
            usable_budgets[current_device] -= embed_fp
        else:
            device_map["embed_tokens"] = "cpu"

        # 2. Place transformer layers iteratively
        for i in range(num_layers):
            layer_fp = footprints[f"layers.{i}"]
            placed = False

            # Check if current GPU can hold this layer
            while current_gpu_idx < len(gpu_devices):
                cand_device = gpu_devices[current_gpu_idx]
                if usable_budgets.get(cand_device, 0) >= layer_fp:
                    device_map[f"layers.{i}"] = cand_device
                    usable_budgets[cand_device] -= layer_fp
                    placed = True
                    break
                else:
                    # Move to next GPU
                    current_gpu_idx += 1

            if not placed:
                # All GPUs full: spill to host CPU RAM
                device_map[f"layers.{i}"] = "cpu"

        # 3. Place final norm and lm_head
        last_layer_device = device_map.get(f"layers.{num_layers - 1}", "cpu")
        norm_fp = footprints["norm"]
        head_fp = footprints["lm_head"]
        total_head_fp = norm_fp + head_fp

        if last_layer_device != "cpu" and usable_budgets.get(last_layer_device, 0) >= total_head_fp:
            device_map["norm"] = last_layer_device
            device_map["lm_head"] = last_layer_device
            usable_budgets[last_layer_device] -= total_head_fp
        elif gpu_devices and usable_budgets.get(gpu_devices[0], 0) >= total_head_fp:
            # Fallback to GPU 0 if it has headroom
            device_map["norm"] = gpu_devices[0]
            device_map["lm_head"] = gpu_devices[0]
        else:
            # Place on CPU or same as last layer
            device_map["norm"] = last_layer_device
            device_map["lm_head"] = last_layer_device

        logger.info(f"Automated Dynamic Device Map generated: {device_map}")
        return device_map


class AutoDeviceDispatcher:
    """Handles transparent cross-device activation streaming with non-blocking stream transfers."""

    @staticmethod
    def transfer_tensor(
        tensor: Optional[torch.Tensor],
        target_device: Union[str, torch.device],
        non_blocking: bool = True,
    ) -> Optional[torch.Tensor]:
        """Asynchronously transfers a tensor to target device if not already there."""
        if tensor is None:
            return None
        target_dev_str = str(target_device)
        curr_dev_str = str(tensor.device)
        if curr_dev_str == target_dev_str:
            return tensor
        return tensor.to(target_device, non_blocking=non_blocking)

    @staticmethod
    def apply_device_map(model: nn.Module, device_map: Dict[str, str]) -> nn.Module:
        """
        Shards submodules of model onto their target devices according to device_map.
        """
        # Embed tokens
        if hasattr(model, "embed_tokens") and "embed_tokens" in device_map:
            target_dev = device_map["embed_tokens"]
            model.embed_tokens = model.embed_tokens.to(target_dev)

        # Layers
        if hasattr(model, "layers"):
            for i, layer in enumerate(model.layers):
                layer_key = f"layers.{i}"
                if layer_key in device_map:
                    target_dev = device_map[layer_key]
                    model.layers[i] = layer.to(target_dev)

        # Norm
        if hasattr(model, "norm") and "norm" in device_map:
            model.norm = model.norm.to(device_map["norm"])

        # LM Head
        if hasattr(model, "lm_head") and "lm_head" in device_map:
            model.lm_head = model.lm_head.to(device_map["lm_head"])

        model._hk_device_map = device_map
        return model


class DynamicOOMGuard:
    """
    Runtime context manager and monitor for catching CUDA memory spikes
    and dynamically spilling boundary layers to host pinned memory.
    """
    def __init__(self, model: nn.Module, critical_vram_threshold_bytes: int = 256 * 1024 * 1024):
        self.model = model
        self.critical_vram_threshold = critical_vram_threshold_bytes

    def check_and_spill(self) -> bool:
        """
        Checks if any GPU is critically close to OOM (<256MB free).
        If so, migrates the highest indexed GPU layer to host CPU and cleans cache.
        Returns True if a layer was spilled.
        """
        if not torch.cuda.is_available():
            return False

        spilled = False
        device_map = getattr(self.model, "_hk_device_map", {})
        if not device_map:
            return False

        for i in range(torch.cuda.device_count()):
            free_b, _ = torch.cuda.mem_get_info(i)
            if free_b < self.critical_vram_threshold:
                target_gpu = f"cuda:{i}"
                # Find highest layer on this GPU
                candidate_layer_idx = -1
                for key, dev in device_map.items():
                    if dev == target_gpu and key.startswith("layers."):
                        l_idx = int(key.split(".")[1])
                        if l_idx > candidate_layer_idx:
                            candidate_layer_idx = l_idx

                if candidate_layer_idx >= 0 and hasattr(self.model, "layers"):
                    logger.warning(
                        f"[DynamicOOMGuard] Free VRAM on {target_gpu} critically low ({free_b / 1024**2:.1f} MB). "
                        f"Dynamically spilling layer {candidate_layer_idx} to host CPU."
                    )
                    self.model.layers[candidate_layer_idx] = self.model.layers[candidate_layer_idx].to("cpu")
                    device_map[f"layers.{candidate_layer_idx}"] = "cpu"
                    torch.cuda.empty_cache()
                    spilled = True

        return spilled
