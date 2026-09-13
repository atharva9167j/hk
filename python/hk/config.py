"""
HK Model Configurations & AutoConfig
Hugging Face-compatible architecture and parameter specifications.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .native import NativeHKReader as HKModelReader


class HKConfig:
    """Base configuration class for HK neural models."""

    def __init__(
        self,
        model_type: str = "causal_lm",
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        num_classes: int = 2,
        in_channels: int = 1,
        image_height: int = 32,
        image_width: int = 128,
        quantization: str = "none",
        sparsity: str = "none",
        alignment: int = 128,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        num_key_value_heads: Optional[int] = None,
        rms_norm_eps: Optional[float] = None,
        rope_theta: Optional[float] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        sliding_window: Optional[int] = None,
        key_length_mla: Optional[int] = None,
        value_length_mla: Optional[int] = None,
        expert_count: Optional[int] = None,
        expert_used_count: Optional[int] = None,
        ssm_conv_kernel: Optional[int] = None,
        ssm_inner_size: Optional[int] = None,
        ssm_state_size: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_height = image_height
        self.image_width = image_width
        self.quantization = quantization
        self.sparsity = sparsity
        self.alignment = alignment
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.key_length_mla = key_length_mla
        self.value_length_mla = value_length_mla
        self.expert_count = expert_count
        self.expert_used_count = expert_used_count
        self.ssm_conv_kernel = ssm_conv_kernel
        self.ssm_inner_size = ssm_inner_size
        self.ssm_state_size = ssm_state_size
        self.extra_kwargs = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        output = copy_dict = self.__dict__.copy()
        if "extra_kwargs" in output:
            del output["extra_kwargs"]
        return output

    def to_json_string(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_pretrained(self, save_directory: Union[str, Path]) -> str:
        os.makedirs(save_directory, exist_ok=True)
        config_file = Path(save_directory) / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())
        return str(config_file)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HKConfig":
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file: Union[str, Path]) -> "HKConfig":
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs) -> "HKConfig":
        path = Path(pretrained_model_name_or_path)

        # 1. Direct .hk binary file inspection
        if path.is_file() and path.suffix == ".hk":
            reader = HKModelReader(str(path))
            meta = reader.metadata
            config_dict = {}

            # If full config JSON was stored in metadata
            if "config" in meta and isinstance(meta["config"], str):
                try:
                    config_dict = json.loads(meta["config"])
                except Exception:
                    pass

            # Override with explicit metadata keys
            key_mappings = {
                "model_type": ("model_type", str),
                "vocab_size": ("vocab_size", int),
                "hidden_size": ("hidden_size", int),
                "num_hidden_layers": ("num_hidden_layers", int),
                "num_attention_heads": ("num_attention_heads", int),
                "intermediate_size": ("intermediate_size", int),
                "max_position_embeddings": ("max_position_embeddings", int),
                "num_classes": ("num_classes", int),
                "quantization": ("quantization", str),
                "sparsity": ("sparsity", str),
            }
            for meta_k, (cfg_k, type_fn) in key_mappings.items():
                if meta_k in meta:
                    try:
                        config_dict[cfg_k] = type_fn(meta[meta_k])
                    except Exception:
                        pass

            if hasattr(reader, "header") and hasattr(reader.header, "alignment"):
                config_dict["alignment"] = reader.header.alignment
            elif hasattr(reader, "alignment"):
                config_dict["alignment"] = reader.alignment
            else:
                config_dict["alignment"] = 128
            config_dict.update(kwargs)
            return cls.from_dict(config_dict)

        # 2. Directory inspection
        if path.is_dir():
            config_file = path / "config.json"
            if config_file.is_file():
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                from .hf_mapper import HFArchitectureMapper, SUPPORTED_ARCHITECTURES
                archs = cfg_data.get("architectures", [])
                m_type = cfg_data.get("model_type", "")
                is_hf = any(a in SUPPORTED_ARCHITECTURES for a in archs) or (m_type in SUPPORTED_ARCHITECTURES)
                if is_hf:
                    cfg = HFArchitectureMapper.map_config(cfg_data)
                else:
                    cfg = cls.from_dict(cfg_data)
                for k, v in kwargs.items():
                    setattr(cfg, k, v)
                return cfg

            # Look for any .hk file in the dir
            hk_files = list(path.glob("*.hk"))
            if hk_files:
                return cls.from_pretrained(hk_files[0], **kwargs)

        raise FileNotFoundError(
            f"Could not locate valid HK model file (.hk) or config.json at {pretrained_model_name_or_path}"
        )


class AutoConfig:
    """Factory for loading model configurations automatically."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs) -> HKConfig:
        return HKConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
