"""
HK Hugging Face Architecture Mapping Layer
Provides zero-friction translation of popular Hugging Face Hub checkpoints (Llama, Qwen2, Mistral)
directly into HK metadata and native container format (.hk).
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .config import HKConfig
from .torch import save_file, save_sharded_file
from .models import (
    ARCHITECTURES_REGISTRY,
    get_architecture_tensor_mappings,
    build_reverse_mappings,
    STANDARD_TRANSFORMER_TENSORS,
)
from .constants import HKKeys, format_arch_key


# Support all 137+ architectures from registry
SUPPORTED_ARCHITECTURES = ARCHITECTURES_REGISTRY

# ---------------------------------------------------------------------------
# Architecture Configuration Conversion Tables (Comprehensive Families)
# ---------------------------------------------------------------------------
ARCHITECTURE_CONFIG_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "llama": {
        "architecture_name": "LlamaForCausalLM",
        "family": "llama",
        "default_params": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps", "pretraining_tp"],
    },
    "llama4": {
        "architecture_name": "Llama4ForCausalLM",
        "family": "llama",
        "default_params": {
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps"],
    },
    "qwen2": {
        "architecture_name": "Qwen2ForCausalLM",
        "family": "qwen2",
        "default_params": {
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "sliding_window": 131072,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps", "sliding_window", "max_window_layers", "rope_scaling"],
    },
    "qwen3": {
        "architecture_name": "Qwen3ForCausalLM",
        "family": "qwen2",
        "default_params": {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 22016,
            "vocab_size": 152064,
            "max_position_embeddings": 65536,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps"],
    },
    "mistral": {
        "architecture_name": "MistralForCausalLM",
        "family": "mistral",
        "default_params": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "sliding_window": 4096,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps", "sliding_window"],
    },
    "deepseek": {
        "architecture_name": "DeepseekForCausalLM",
        "family": "deepseek",
        "default_params": {
            "hidden_size": 4096,
            "num_hidden_layers": 30,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 102400,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps"],
    },
    "deepseek2": {
        "architecture_name": "DeepseekV2ForCausalLM",
        "family": "deepseek",
        "default_params": {
            "hidden_size": 5120,
            "num_hidden_layers": 60,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "intermediate_size": 12288,
            "vocab_size": 102400,
            "max_position_embeddings": 163840,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "n_routed_experts": 160,
            "num_experts_per_tok": 6,
        },
        "extra_fields": ["kv_lora_rank", "q_lora_rank", "n_routed_experts", "num_experts_per_tok"],
    },
    "deepseek3": {
        "architecture_name": "DeepseekV3ForCausalLM",
        "family": "deepseek",
        "default_params": {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "intermediate_size": 18432,
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        },
        "extra_fields": ["kv_lora_rank", "q_lora_rank", "n_routed_experts", "num_experts_per_tok", "n_shared_experts"],
    },
    "deepseek_r1": {
        "architecture_name": "DeepseekV3ForCausalLM",
        "family": "deepseek",
        "default_params": {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "intermediate_size": 18432,
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["kv_lora_rank", "q_lora_rank", "n_routed_experts"],
    },
    "gemma": {
        "architecture_name": "GemmaForCausalLM",
        "family": "gemma",
        "default_params": {
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "intermediate_size": 16384,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
        },
        "extra_fields": ["rope_theta", "rms_norm_eps"],
    },
    "gemma2": {
        "architecture_name": "Gemma2ForCausalLM",
        "family": "gemma",
        "default_params": {
            "hidden_size": 3584,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "sliding_window": 4096,
        },
        "extra_fields": ["sliding_window", "attn_logit_softcapping", "final_logit_softcapping"],
    },
    "grok": {
        "architecture_name": "Grok1ForCausalLM",
        "family": "grok",
        "default_params": {
            "hidden_size": 6144,
            "num_hidden_layers": 64,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "intermediate_size": 32768,
            "vocab_size": 131072,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        "extra_fields": ["num_local_experts", "num_experts_per_tok"],
    },
    "falcon": {
        "architecture_name": "FalconForCausalLM",
        "family": "falcon",
        "default_params": {
            "hidden_size": 4544,
            "num_hidden_layers": 32,
            "num_attention_heads": 71,
            "num_key_value_heads": 1,
            "intermediate_size": 18176,
            "vocab_size": 65024,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["multi_query", "parallel_attn"],
    },
    "phi3": {
        "architecture_name": "Phi3ForCausalLM",
        "family": "phi3",
        "default_params": {
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32064,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["original_max_position_embeddings", "rope_scaling"],
    },
    "mamba": {
        "architecture_name": "MambaForCausalLM",
        "family": "mamba",
        "default_params": {
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": 4096,
            "vocab_size": 50280,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "ssm_state_size": 16,
            "ssm_conv_kernel": 4,
            "ssm_inner_size": 4096,
        },
        "extra_fields": ["ssm_state_size", "ssm_conv_kernel", "ssm_inner_size"],
    },
    "whisper": {
        "architecture_name": "WhisperForConditionalGeneration",
        "family": "whisper",
        "default_params": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "intermediate_size": 5120,
            "vocab_size": 51865,
            "max_position_embeddings": 1500,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["num_mel_bins"],
    },
    "modern_bert": {
        "architecture_name": "ModernBertModel",
        "family": "modern_bert",
        "default_params": {
            "hidden_size": 768,
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "intermediate_size": 1152,
            "vocab_size": 50368,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 160000.0,
            "tie_word_embeddings": False,
        },
        "extra_fields": ["global_attn_every_n_layers"],
    },
}

# Backward compatibility alias
ARCHITECTURE_TENSOR_MAPPINGS = {
    "llama": STANDARD_TRANSFORMER_TENSORS,
    "qwen2": STANDARD_TRANSFORMER_TENSORS,
    "mistral": STANDARD_TRANSFORMER_TENSORS,
}
ARCHITECTURE_REVERSE_TENSOR_MAPPINGS = {
    "llama": build_reverse_mappings(STANDARD_TRANSFORMER_TENSORS),
    "qwen2": build_reverse_mappings(STANDARD_TRANSFORMER_TENSORS),
    "mistral": build_reverse_mappings(STANDARD_TRANSFORMER_TENSORS),
}
HF_TO_HK_TENSOR_PATTERNS = STANDARD_TRANSFORMER_TENSORS
HK_TO_HF_TENSOR_PATTERNS = build_reverse_mappings(STANDARD_TRANSFORMER_TENSORS)


class HFArchitectureMapper:
    """
    Translates Hugging Face model configurations, state dicts, and tokenizer
    metadata into standardized HK specifications for 137+ architectures.
    """

    @staticmethod
    def detect_architecture(hf_config: Dict[str, Any]) -> str:
        """Determines canonical architecture name from HF config dict (137+ model support)."""
        archs = hf_config.get("architectures", [])
        if isinstance(archs, list) and archs:
            for arch_name in archs:
                if arch_name in SUPPORTED_ARCHITECTURES:
                    return SUPPORTED_ARCHITECTURES[arch_name]
                for k, v in SUPPORTED_ARCHITECTURES.items():
                    if k.lower() == str(arch_name).lower():
                        return v

        model_type = str(hf_config.get("model_type", "")).lower()
        if model_type in SUPPORTED_ARCHITECTURES:
            return SUPPORTED_ARCHITECTURES[model_type]

        # Check against arch_names as well
        all_candidates = [model_type]
        if isinstance(archs, list):
            all_candidates.extend(str(a).lower() for a in archs)

        # Substring fuzzy fallbacks
        for cand in all_candidates:
            for key, canonical in [
                ("deepseek_r1", "deepseek_r1"),
                ("deepseekv3", "deepseek3"),
                ("deepseek3", "deepseek3"),
                ("deepseekv2", "deepseek2"),
                ("deepseek2", "deepseek2"),
                ("deepseek", "deepseek3"),
                ("qwen3", "qwen3"),
                ("qwen2", "qwen2"),
                ("qwen", "qwen2"),
                ("mistral", "mistral"),
                ("mixtral", "mixtral"),
                ("gemma2", "gemma2"),
                ("gemma", "gemma2"),
                ("mamba2", "mamba2"),
                ("mamba", "mamba"),
                ("falcon", "falcon"),
                ("phi3", "phi3"),
                ("whisper", "whisper"),
                ("flux", "flux"),
                ("modern_bert", "modern_bert"),
                ("modernbert", "modern_bert"),
                ("bert", "modern_bert"),
            ]:
                if key in cand:
                    return canonical

        return "llama"

    @classmethod
    def get_conversion_table(cls, architecture: str = "llama") -> Dict[str, Any]:
        """Returns the complete configuration and tensor conversion table for the specified architecture."""
        canonical = SUPPORTED_ARCHITECTURES.get(architecture, architecture.lower())
        forward = get_architecture_tensor_mappings(canonical)
        reverse = build_reverse_mappings(forward)
        return {
            "config": ARCHITECTURE_CONFIG_MAPPINGS.get(canonical, ARCHITECTURE_CONFIG_MAPPINGS["llama"]),
            "forward_tensor_patterns": forward,
            "reverse_tensor_patterns": reverse,
        }

    @classmethod
    def map_config(cls, hf_config: Dict[str, Any]) -> HKConfig:
        """
        Translates a Hugging Face configuration dictionary to an HKConfig instance
        using architecture-specific parameter tables and advanced taxonomy extraction.
        """
        canonical_arch = cls.detect_architecture(hf_config)
        table = ARCHITECTURE_CONFIG_MAPPINGS.get(canonical_arch, ARCHITECTURE_CONFIG_MAPPINGS["llama"])
        defaults = table["default_params"]

        hidden_size = hf_config.get("hidden_size", defaults.get("hidden_size", 4096))
        num_hidden_layers = hf_config.get("num_hidden_layers", defaults.get("num_hidden_layers", 32))
        num_attention_heads = hf_config.get("num_attention_heads", defaults.get("num_attention_heads", 32))
        num_key_value_heads = hf_config.get("num_key_value_heads", hf_config.get("num_attention_heads", defaults.get("num_key_value_heads", num_attention_heads)))
        intermediate_size = hf_config.get("intermediate_size", defaults.get("intermediate_size", 4 * hidden_size))
        vocab_size = hf_config.get("vocab_size", defaults.get("vocab_size", 32000))
        max_pos = hf_config.get("max_position_embeddings", defaults.get("max_position_embeddings", 2048))
        rms_norm_eps = hf_config.get("rms_norm_eps", defaults.get("rms_norm_eps", 1e-6))
        rope_theta = hf_config.get("rope_theta", defaults.get("rope_theta", 10000.0))
        tie_word_embeddings = hf_config.get("tie_word_embeddings", defaults.get("tie_word_embeddings", False))

        # Advanced hyperparameter extraction: MLA, MoE, SWA, YaRN, SSM
        extra_kwargs = {}
        if "sliding_window" in hf_config:
            extra_kwargs["sliding_window"] = hf_config["sliding_window"]
        if "rope_scaling" in hf_config:
            extra_kwargs["rope_scaling"] = hf_config["rope_scaling"]
        if "kv_lora_rank" in hf_config:
            extra_kwargs["key_length_mla"] = hf_config["kv_lora_rank"]
            extra_kwargs["value_length_mla"] = hf_config.get("v_head_dim", 128)
        if "q_lora_rank" in hf_config:
            extra_kwargs["q_lora_rank"] = hf_config["q_lora_rank"]
        if "n_routed_experts" in hf_config:
            extra_kwargs["expert_count"] = hf_config["n_routed_experts"]
        elif "num_local_experts" in hf_config:
            extra_kwargs["expert_count"] = hf_config["num_local_experts"]
        if "num_experts_per_tok" in hf_config:
            extra_kwargs["expert_used_count"] = hf_config["num_experts_per_tok"]
        if "ssm_state_size" in hf_config:
            extra_kwargs["ssm_state_size"] = hf_config["ssm_state_size"]
        if "ssm_conv_kernel" in hf_config:
            extra_kwargs["ssm_conv_kernel"] = hf_config["ssm_conv_kernel"]
        if "ssm_inner_size" in hf_config:
            extra_kwargs["ssm_inner_size"] = hf_config["ssm_inner_size"]

        return HKConfig(
            model_type="causal_lm",
            architecture=canonical_arch,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=hf_config.get("bos_token_id", 1),
            eos_token_id=hf_config.get("eos_token_id", 2),
            pad_token_id=hf_config.get("pad_token_id", 0),
            **extra_kwargs,
        )

    @classmethod
    def map_config_to_metadata(cls, hf_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds standardized HK metadata key-values from Hugging Face config adhering
        to the formal taxonomy namespaces (general.*, attention.*, rope.*, moe.*, ssm.*).
        """
        canonical_arch = cls.detect_architecture(hf_config)
        table = ARCHITECTURE_CONFIG_MAPPINGS.get(canonical_arch, ARCHITECTURE_CONFIG_MAPPINGS["llama"])
        defaults = table["default_params"]

        meta: Dict[str, Any] = {
            HKKeys.GENERAL_ARCHITECTURE: canonical_arch,
            HKKeys.GENERAL_NAME: hf_config.get("_name_or_path", canonical_arch),
            "model_type": "causal_lm",
            "vocab_size": hf_config.get("vocab_size", defaults.get("vocab_size", 32000)),
            "hidden_size": hf_config.get("hidden_size", defaults.get("hidden_size", 4096)),
            "num_hidden_layers": hf_config.get("num_hidden_layers", defaults.get("num_hidden_layers", 32)),
            "num_attention_heads": hf_config.get("num_attention_heads", defaults.get("num_attention_heads", 32)),
            "intermediate_size": hf_config.get("intermediate_size", defaults.get("intermediate_size", 11008)),
            "max_position_embeddings": hf_config.get("max_position_embeddings", defaults.get("max_position_embeddings", 2048)),
            format_arch_key(HKKeys.CONTEXT_LENGTH, canonical_arch): hf_config.get("max_position_embeddings", defaults.get("max_position_embeddings", 2048)),
            format_arch_key(HKKeys.EMBEDDING_LENGTH, canonical_arch): hf_config.get("hidden_size", defaults.get("hidden_size", 4096)),
            format_arch_key(HKKeys.BLOCK_COUNT, canonical_arch): hf_config.get("num_hidden_layers", defaults.get("num_hidden_layers", 32)),
            format_arch_key(HKKeys.FEED_FORWARD_LENGTH, canonical_arch): hf_config.get("intermediate_size", defaults.get("intermediate_size", 11008)),
            format_arch_key(HKKeys.ATTENTION_HEAD_COUNT, canonical_arch): hf_config.get("num_attention_heads", defaults.get("num_attention_heads", 32)),
        }

        kv_heads = hf_config.get("num_key_value_heads", defaults.get("num_key_value_heads", meta["num_attention_heads"]))
        meta["num_key_value_heads"] = kv_heads
        meta[format_arch_key(HKKeys.ATTENTION_HEAD_COUNT_KV, canonical_arch)] = kv_heads

        if "rms_norm_eps" in hf_config or "rms_norm_eps" in defaults:
            eps = hf_config.get("rms_norm_eps", defaults.get("rms_norm_eps"))
            meta["rms_norm_eps"] = eps
            meta[format_arch_key(HKKeys.ATTENTION_LAYER_NORM_RMS_EPSILON, canonical_arch)] = eps

        if "rope_theta" in hf_config or "rope_theta" in defaults:
            theta = hf_config.get("rope_theta", defaults.get("rope_theta"))
            meta["rope_theta"] = theta
            meta[format_arch_key(HKKeys.ROPE_FREQ_BASE, canonical_arch)] = theta

        if "sliding_window" in hf_config:
            meta["sliding_window"] = hf_config["sliding_window"]
            meta[format_arch_key(HKKeys.ATTENTION_SLIDING_WINDOW, canonical_arch)] = hf_config["sliding_window"]

        # MLA Attention
        if "kv_lora_rank" in hf_config:
            meta[format_arch_key(HKKeys.ATTENTION_KEY_LENGTH_MLA, canonical_arch)] = hf_config["kv_lora_rank"]
            meta[format_arch_key(HKKeys.ATTENTION_KV_LORA_RANK, canonical_arch)] = hf_config["kv_lora_rank"]
        if "q_lora_rank" in hf_config:
            meta[format_arch_key(HKKeys.ATTENTION_Q_LORA_RANK, canonical_arch)] = hf_config["q_lora_rank"]

        # MoE Parameters
        if "n_routed_experts" in hf_config or "num_local_experts" in hf_config:
            exp_count = hf_config.get("n_routed_experts", hf_config.get("num_local_experts"))
            meta[format_arch_key(HKKeys.MOE_EXPERT_COUNT, canonical_arch)] = exp_count
        if "num_experts_per_tok" in hf_config:
            meta[format_arch_key(HKKeys.MOE_EXPERT_USED_COUNT, canonical_arch)] = hf_config["num_experts_per_tok"]

        # RoPE YaRN / Scaling
        if "rope_scaling" in hf_config and isinstance(hf_config["rope_scaling"], dict):
            rs = hf_config["rope_scaling"]
            scale_type = rs.get("type", rs.get("rope_type", "linear"))
            meta[format_arch_key(HKKeys.ROPE_SCALE_TYPE, canonical_arch)] = scale_type
            meta[format_arch_key(HKKeys.ROPE_SCALING_TYPE, canonical_arch)] = scale_type
            if "factor" in rs:
                meta[format_arch_key(HKKeys.ROPE_SCALE, canonical_arch)] = rs["factor"]
            if "extrapolation_factor" in rs:
                meta[format_arch_key(HKKeys.ROPE_YARN_EXT_FACTOR, canonical_arch)] = rs["extrapolation_factor"]
            if "attn_factor" in rs:
                meta[format_arch_key(HKKeys.ROPE_YARN_ATTN_FACTOR, canonical_arch)] = rs["attn_factor"]

        # Sampling Defaults
        meta[HKKeys.SAMPLING_TEMP] = 0.7
        meta[HKKeys.SAMPLING_TOP_P] = 0.95
        meta[HKKeys.SAMPLING_MIN_P] = 0.05

        meta["config"] = json.dumps(hf_config)
        return meta

    convert_hf_config_to_metadata = map_config_to_metadata

    @classmethod
    def map_tensor_name_to_hk(cls, name: str, architecture: str = "llama") -> str:
        """Translates a Hugging Face tensor name into HK format using architecture-specific tables."""
        canonical = SUPPORTED_ARCHITECTURES.get(architecture, architecture.lower())
        patterns = get_architecture_tensor_mappings(canonical)
        for pattern, repl in patterns:
            new_name, count = re.subn(pattern, repl, name)
            if count > 0:
                return new_name
        for pattern, repl in STANDARD_TRANSFORMER_TENSORS:
            new_name, count = re.subn(pattern, repl, name)
            if count > 0:
                return new_name
        return name

    @classmethod
    def map_tensor_name_to_hf(cls, name: str, architecture: str = "llama") -> str:
        """Translates an HK tensor name back into Hugging Face name using architecture tables."""
        canonical = SUPPORTED_ARCHITECTURES.get(architecture, architecture.lower())
        patterns = build_reverse_mappings(get_architecture_tensor_mappings(canonical))
        for pattern, repl in patterns:
            new_name, count = re.subn(pattern, repl, name)
            if count > 0:
                return new_name
        for pattern, repl in HK_TO_HF_TENSOR_PATTERNS:
            new_name, count = re.subn(pattern, repl, name)
            if count > 0:
                return new_name
        return name

    @classmethod
    def map_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        architecture: Optional[str] = None,
        to_hk: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Translates all tensor names in a state dict between HF and HK conventions."""
        arch = architecture or "llama"
        if to_hk:
            return {cls.map_tensor_name_to_hk(k, architecture=arch): v for k, v in state_dict.items()}
        else:
            return {cls.map_tensor_name_to_hf(k, architecture=arch): v for k, v in state_dict.items()}



def extract_hf_tokenizer_metadata(hf_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts tokenizer vocabulary, merges, special tokens, and chat templates
    from Hugging Face tokenizer files into standard HK metadata keys.
    """
    dir_path = Path(hf_dir)
    meta: Dict[str, Any] = {}

    tok_file = dir_path / "tokenizer.json"
    tok_cfg_file = dir_path / "tokenizer_config.json"
    spec_tokens_file = dir_path / "special_tokens_map.json"

    # 1. Read tokenizer.json
    if tok_file.is_file():
        try:
            with open(tok_file, "r", encoding="utf-8") as f:
                raw_json = f.read()
                tok_data = json.loads(raw_json)

            # Verbatim embedding of full Hugging Face tokenizer.json
            meta["tokenizer.huggingface.json"] = raw_json

            model_data = tok_data.get("model", {})
            vocab = model_data.get("vocab", {})
            if isinstance(vocab, dict):
                sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
                tokens_list = [token for token, _ in sorted_tokens]
                meta["tokenizer.tokens"] = json.dumps(tokens_list)

            # Scores (e.g. from BPE or SentencePiece if present)
            scores = model_data.get("scores", None)
            if isinstance(scores, list):
                meta["tokenizer.scores"] = json.dumps(scores)

            # Merges
            merges = model_data.get("merges", None)
            if isinstance(merges, list):
                formatted_merges = []
                for m in merges:
                    if isinstance(m, (list, tuple)):
                        formatted_merges.append(f"{m[0]} {m[1]}")
                    elif isinstance(m, str):
                        formatted_merges.append(m)
                meta["tokenizer.merges"] = json.dumps(formatted_merges)
        except Exception:
            pass

    # 2. Read tokenizer_config.json
    if tok_cfg_file.is_file():
        try:
            with open(tok_cfg_file, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)

            chat_tmpl = tok_cfg.get("chat_template")
            if isinstance(chat_tmpl, str):
                meta["tokenizer.chat_template"] = chat_tmpl
                meta["tokenizer.chat_templates"] = json.dumps({"default": chat_tmpl})
            elif isinstance(chat_tmpl, dict):
                meta["tokenizer.chat_templates"] = json.dumps(chat_tmpl)
                if "default" in chat_tmpl:
                    meta["tokenizer.chat_template"] = chat_tmpl["default"]
            elif isinstance(chat_tmpl, list):
                tmpls = {}
                for t in chat_tmpl:
                    if isinstance(t, dict) and "name" in t and "template" in t:
                        tmpls[t["name"]] = t["template"]
                if tmpls:
                    meta["tokenizer.chat_templates"] = json.dumps(tmpls)
                    meta["tokenizer.chat_template"] = tmpls.get("default", list(tmpls.values())[0])

            for fim_key, meta_key in [
                ("prefix_token_id", "tokenizer.ggml.prefix_token_id"),
                ("middle_token_id", "tokenizer.ggml.middle_token_id"),
                ("suffix_token_id", "tokenizer.ggml.suffix_token_id"),
                ("pad_token_id", "tokenizer.ggml.pad_token_id"),
                ("eot_token_id", "tokenizer.ggml.eot_token_id"),
            ]:
                if fim_key in tok_cfg:
                    meta[meta_key] = tok_cfg[fim_key]

            for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
                val = tok_cfg.get(key)
                if isinstance(val, str):
                    meta[f"tokenizer.{key}"] = val
                elif isinstance(val, dict) and "content" in val:
                    meta[f"tokenizer.{key}"] = val["content"]
        except Exception:
            pass

    # 3. Read special_tokens_map.json
    if spec_tokens_file.is_file():
        try:
            with open(spec_tokens_file, "r", encoding="utf-8") as f:
                spec_map = json.load(f)
            for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
                if f"tokenizer.{key}" not in meta and key in spec_map:
                    val = spec_map[key]
                    if isinstance(val, str):
                        meta[f"tokenizer.{key}"] = val
                    elif isinstance(val, dict) and "content" in val:
                        meta[f"tokenizer.{key}"] = val["content"]
        except Exception:
            pass

    return meta


def convert_hf_checkpoint(
    hf_model_dir: Union[str, Path],
    output_hk_path: Union[str, Path],
    quantize_mode: Optional[str] = None,
    canonical_tensor_names: bool = False,
    split_size_mb: Optional[int] = None,
    alignment: int = 128,
) -> str:
    """
    Converts a Hugging Face model directory (safetensors or pytorch_model.bin)
    into standard .hk file(s) with full metadata and tokenizer tables.
    """
    src_dir = Path(hf_model_dir)
    cfg_file = src_dir / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Missing config.json in {hf_model_dir}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        hf_config = json.load(f)

    # 1. Map config to metadata
    metadata = HFArchitectureMapper.map_config_to_metadata(hf_config)

    # 2. Extract in-file tokenizer metadata
    tok_meta = extract_hf_tokenizer_metadata(src_dir)
    metadata.update(tok_meta)

    # 3. Load model weights
    state_dict: Dict[str, torch.Tensor] = {}

    st_files = list(src_dir.glob("*.safetensors"))
    if st_files:
        try:
            import safetensors.torch
            for sf in sorted(st_files):
                sub_dict = safetensors.torch.load_file(str(sf), device="cpu")
                state_dict.update(sub_dict)
        except ImportError:
            pass

    if not state_dict:
        bin_files = list(src_dir.glob("pytorch_model*.bin"))
        for bf in sorted(bin_files):
            sub_dict = torch.load(str(bf), map_location="cpu")
            if isinstance(sub_dict, dict):
                state_dict.update(sub_dict)

    if not state_dict:
        pt_file = src_dir / "model.pt"
        if pt_file.is_file():
            sub_dict = torch.load(str(pt_file), map_location="cpu")
            if isinstance(sub_dict, dict):
                state_dict.update(sub_dict)

    if not state_dict:
        raise ValueError(f"No valid weights found in {hf_model_dir} (expected .safetensors or .bin)")

    # 4. Optional tensor name mapping
    if canonical_tensor_names:
        arch = HFArchitectureMapper.detect_architecture(hf_config)
        state_dict = HFArchitectureMapper.map_state_dict(state_dict, architecture=arch, to_hk=True)

    # 5. Write to .hk container (single or sharded)
    out_path = Path(output_hk_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if split_size_mb is not None and split_size_mb > 0:
        shard_paths = save_sharded_file(
            state_dict=state_dict,
            base_path=str(out_path),
            max_shard_size_bytes=split_size_mb * 1024 * 1024,
            metadata=metadata,
            quantize_mode=quantize_mode,
            alignment=alignment,
        )
        return shard_paths[0]
    else:
        save_file(
            state_dict=state_dict,
            filename=str(out_path),
            metadata=metadata,
            quantize_mode=quantize_mode,
            alignment=alignment,
        )
        return str(out_path)


# ---------------------------------------------------------------------------
# Mathematical Tensor Transformation Routines (RoPE, LayerNorm, QKV, MoE)
# ---------------------------------------------------------------------------

def permute_hf_to_gguf_rope(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Permutes weight matrix from Hugging Face half-split RoPE layout to GGUF interleaved layout.
    HF layout: [x0, x1, ..., x_{d/2-1}, y0, y1, ..., y_{d/2-1}]
    GGUF layout: [x0, y0, x1, y1, ..., x_{d/2-1}, y_{d/2-1}]
    """
    from .native import native_rope_permute_hf_to_gguf, is_native_available
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    orig_device = weight.device

    if is_native_available():
        arr = weight.detach().cpu().to(torch.float32).numpy()
        perm = native_rope_permute_hf_to_gguf(arr, n_heads, head_dim)
        return torch.from_numpy(perm).to(dtype=orig_dtype, device=orig_device).view(orig_shape)

    # PyTorch fallback
    half = head_dim // 2
    w = weight.view(-1, n_heads, head_dim)
    w_out = torch.empty_like(w)
    w_out[..., 0::2] = w[..., :half]
    w_out[..., 1::2] = w[..., half:]
    return w_out.view(orig_shape)


def unpermute_gguf_to_hf_rope(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Unpermutes weight matrix from GGUF interleaved layout to Hugging Face half-split RoPE layout.
    """
    from .native import native_rope_unpermute_gguf_to_hf, is_native_available
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    orig_device = weight.device

    if is_native_available():
        arr = weight.detach().cpu().to(torch.float32).numpy()
        unperm = native_rope_unpermute_gguf_to_hf(arr, n_heads, head_dim)
        return torch.from_numpy(unperm).to(dtype=orig_dtype, device=orig_device).view(orig_shape)

    # PyTorch fallback
    half = head_dim // 2
    w = weight.view(-1, n_heads, head_dim)
    w_out = torch.empty_like(w)
    w_out[..., :half] = w[..., 0::2]
    w_out[..., half:] = w[..., 1::2]
    return w_out.view(orig_shape)


def add_layernorm_offset(weight: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
    """
    Adds offset (+1.0) to LayerNorm/RMSNorm weights (e.g., converting Gemma/T5 HF weights to effective GGUF weights).
    """
    from .native import native_layernorm_offset, is_native_available
    if is_native_available():
        arr = weight.detach().cpu().to(torch.float32).numpy()
        res = native_layernorm_offset(arr, offset)
        return torch.from_numpy(res).to(dtype=weight.dtype, device=weight.device)
    return weight + offset


def sub_layernorm_offset(weight: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
    """
    Subtracts offset (-1.0) from LayerNorm/RMSNorm weights.
    """
    from .native import native_layernorm_offset, is_native_available
    if is_native_available():
        arr = weight.detach().cpu().to(torch.float32).numpy()
        res = native_layernorm_offset(arr, -offset)
        return torch.from_numpy(res).to(dtype=weight.dtype, device=weight.device)
    return weight - offset


def split_fused_qkv(
    qkv_weight: torch.Tensor,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits a single fused QKV projection weight matrix into separate Q, K, and V projection tensors.
    """
    q_dim = q_heads * head_dim
    k_dim = k_heads * head_dim
    v_dim = v_heads * head_dim

    w_q = qkv_weight[:q_dim, ...]
    w_k = qkv_weight[q_dim : q_dim + k_dim, ...]
    w_v = qkv_weight[q_dim + k_dim : q_dim + k_dim + v_dim, ...]
    return w_q, w_k, w_v


def merge_fused_qkv(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Merges separate Q, K, and V projection tensors into a single fused QKV projection tensor.
    """
    return torch.cat([q_weight, k_weight, v_weight], dim=0)


def pack_moe_experts(expert_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Packs a list of 2D expert weight tensors [E x [in_dim, out_dim]] into a single 3D stacked tensor [E, in_dim, out_dim].
    """
    return torch.stack(expert_tensors, dim=0)


def unpack_moe_experts(stacked_tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Unpacks a 3D stacked MoE tensor [E, in_dim, out_dim] into a list of 2D expert weight tensors.
    """
    return [stacked_tensor[i].clone() for i in range(stacked_tensor.shape[0])]

