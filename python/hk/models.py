"""
HK Architecture Definitions & Comprehensive Tensor Mapping Tables
Defines configuration schemas, bidirectional tensor regexes, and registry for 137+ distinct model architectures:
- Cutting-edge LLMs (DeepSeek MLA/MTP, LLaMA 4, Qwen 2.5/3/MoE, Gemma, Grok, Falcon, Phi-3, DBRX, Command-R+, OLMoE, etc.)
- State Space & Recurrent Models (Mamba, Mamba-2, Jamba, RWKV-5/6)
- Vision-Language Models (CLIP, LLaVA, MobileVLM, Qwen2-VL, Pixtral, Gemma-Vision, SAM)
- Audio & Diffusion (Whisper, Stable Diffusion, FLUX)
- Modern Encoders (ModernBERT, Nomic-BERT, Jina-BERT, EuroBERT, BERT, RoBERTa)
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from .constants import ModelArchitecture, HKKeys, format_arch_key


# ---------------------------------------------------------------------------
# Comprehensive Architecture Registry (137+ Model Classes / Model Types)
# ---------------------------------------------------------------------------
ARCHITECTURES_REGISTRY: Dict[str, str] = {
    # DeepSeek Series
    "DeepseekV3ForCausalLM": "deepseek3",
    "DeepseekV2ForCausalLM": "deepseek2",
    "DeepseekForCausalLM": "deepseek",
    "DeepseekV3": "deepseek3",
    "DeepseekV2": "deepseek2",
    "deepseek_v3": "deepseek3",
    "deepseek_v2": "deepseek2",
    "deepseek": "deepseek",
    "deepseek_r1": "deepseek_r1",

    # LLaMA Family
    "LlamaForCausalLM": "llama",
    "LLaMAForCausalLM": "llama",
    "Llama4ForCausalLM": "llama4",
    "CodeLlamaForCausalLM": "llama",
    "llama": "llama",
    "llama4": "llama4",

    # Qwen Family
    "Qwen2ForCausalLM": "qwen2",
    "Qwen2MoeForCausalLM": "qwen2_moe",
    "Qwen2_5ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3_moe",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2_moe",
    "qwen": "qwen",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3_moe",

    # Mistral & Mixtral
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "mistral": "mistral",
    "mixtral": "mixtral",

    # Gemma Series
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma2",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "recurrent_gemma": "recurrent_gemma",
    "RecurrentGemmaForCausalLM": "recurrent_gemma",

    # Grok
    "Grok1ForCausalLM": "grok",
    "GrokForCausalLM": "grok",
    "grok": "grok",

    # Falcon Series
    "FalconForCausalLM": "falcon",
    "FalconH1ForCausalLM": "falcon_h1",
    "RWForCausalLM": "falcon",
    "falcon": "falcon",
    "falcon_h1": "falcon_h1",

    # Phi Series
    "PhiForCausalLM": "phi",
    "Phi2ForCausalLM": "phi2",
    "Phi3ForCausalLM": "phi3",
    "Phi3SmallForCausalLM": "phi3",
    "PhiMoEForCausalLM": "phi3_moe",
    "Phi4ForCausalLM": "phi4",
    "phi": "phi",
    "phi2": "phi2",
    "phi3": "phi3",
    "phi3_moe": "phi3_moe",
    "phi4": "phi4",

    # DBRX
    "DbrxForCausalLM": "dbrx",
    "dbrx": "dbrx",

    # Command-R & Command-R+
    "CohereForCausalLM": "command-r",
    "Cohere2ForCausalLM": "command-r-plus",
    "cohere": "command-r",
    "command-r": "command-r",
    "command-r-plus": "command-r-plus",

    # OLMo & OLMoE
    "OLMoForCausalLM": "olmo",
    "Olmo2ForCausalLM": "olmo2",
    "OlmoeForCausalLM": "olmoe",
    "olmo": "olmo",
    "olmo2": "olmo2",
    "olmoe": "olmoe",

    # MiniCPM Series
    "MiniCPMForCausalLM": "minicpm",
    "MiniCPM3ForCausalLM": "minicpm3",
    "minicpm": "minicpm",
    "minicpm3": "minicpm3",

    # StarCoder Series
    "GPTBigCodeForCausalLM": "starcoder",
    "Starcoder2ForCausalLM": "starcoder2",
    "starcoder": "starcoder",
    "starcoder2": "starcoder2",

    # Jais & Exaone
    "JaisLMHeadModel": "jais",
    "JaisForCausalLM": "jais",
    "ExaoneForCausalLM": "exaone",
    "jais": "jais",
    "exaone": "exaone",

    # ChatGLM & GLM-4
    "ChatGLMModel": "chatglm",
    "ChatGLMForConditionalGeneration": "chatglm",
    "GlmForCausalLM": "glm4",
    "chatglm": "chatglm",
    "glm4": "glm4",

    # Baichuan
    "BaichuanForCausalLM": "baichuan",
    "BaiChuanForCausalLM": "baichuan",
    "baichuan": "baichuan",

    # InternLM
    "InternLMForCausalLM": "internlm",
    "InternLM2ForCausalLM": "internlm2",
    "InternLM3ForCausalLM": "internlm3",
    "internlm": "internlm",
    "internlm2": "internlm2",
    "internlm3": "internlm3",

    # Yi, Xverse, Orion, Arctic, BitNet
    "YiForCausalLM": "yi",
    "XverseForCausalLM": "xverse",
    "OrionForCausalLM": "orion",
    "ArcticForCausalLM": "arctic",
    "BitNetForCausalLM": "bitnet",
    "SmolLMForCausalLM": "smollm",
    "SmolLM2ForCausalLM": "smollm2",
    "yi": "yi",
    "xverse": "xverse",
    "orion": "orion",
    "arctic": "arctic",
    "bitnet": "bitnet",
    "smollm": "smollm",
    "smollm2": "smollm2",

    # State Space & Recurrent
    "MambaForCausalLM": "mamba",
    "Mamba2ForCausalLM": "mamba2",
    "JambaForCausalLM": "jamba",
    "RWKVForCausalLM": "rwkv5",
    "RWKV5ForCausalLM": "rwkv5",
    "RWKV6ForCausalLM": "rwkv6",
    "mamba": "mamba",
    "mamba2": "mamba2",
    "jamba": "jamba",
    "rwkv": "rwkv5",
    "rwkv5": "rwkv5",
    "rwkv6": "rwkv6",

    # Vision-Language Models (VLM)
    "CLIPModel": "clip",
    "CLIPVisionModel": "clip",
    "LlavaForConditionalGeneration": "llava",
    "LlavaNextForConditionalGeneration": "llava_next",
    "MobileVLMForConditionalGeneration": "mobilevlm",
    "Qwen2VLForConditionalGeneration": "qwen2_vl",
    "PixtralForConditionalGeneration": "pixtral",
    "GemmaVisionForConditionalGeneration": "gemma_vision",
    "PaliGemmaForConditionalGeneration": "paligemma",
    "SamModel": "sam",
    "Sam2Model": "sam2",
    "CogVLMForConditionalGeneration": "cogvlm",
    "MiniCPMV": "minicpm_v",
    "InternVLChatModel": "internvl",
    "clip": "clip",
    "llava": "llava",
    "mobilevlm": "mobilevlm",
    "qwen2_vl": "qwen2_vl",
    "pixtral": "pixtral",
    "sam": "sam",
    "paligemma": "paligemma",

    # Audio & Diffusion
    "WhisperForConditionalGeneration": "whisper",
    "WhisperModel": "whisper",
    "whisper": "whisper",
    "Wav2Vec2ForCTC": "wav2vec2",
    "wav2vec2": "wav2vec2",
    "StableDiffusionPipeline": "stable_diffusion",
    "FluxPipeline": "flux",
    "flux": "flux",
    "stable_diffusion": "stable_diffusion",

    # Encoders
    "ModernBertModel": "modern_bert",
    "ModernBertForMaskedLM": "modern_bert",
    "NomicBertModel": "nomic_bert",
    "JinaBertModel": "jina_bert_v2",
    "EuroBertModel": "euro_bert",
    "BertModel": "bert",
    "BertForMaskedLM": "bert",
    "RobertaModel": "roberta",
    "RobertaForMaskedLM": "roberta",
    "modern_bert": "modern_bert",
    "nomic_bert": "nomic_bert",
    "jina_bert_v2": "jina_bert_v2",
    "euro_bert": "euro_bert",
    "bert": "bert",
    "roberta": "roberta",
    "gpt2": "gpt2",
    "GPT2LMHeadModel": "gpt2",
    "bloom": "bloom",
    "BloomForCausalLM": "bloom",
}

# Ensure all canonical enum architectures are indexed
for arch in ModelArchitecture:
    if arch.value not in ARCHITECTURES_REGISTRY:
        ARCHITECTURES_REGISTRY[arch.value] = arch.value
    # Also register sanitized variants (e.g. qwen2_5 for qwen2.5)
    clean_val = arch.value.replace(".", "_")
    if clean_val not in ARCHITECTURES_REGISTRY:
        ARCHITECTURES_REGISTRY[clean_val] = arch.value

ARCHITECTURES_REGISTRY["qwen2.5"] = "qwen2"
ARCHITECTURES_REGISTRY["qwen2_5"] = "qwen2"
ARCHITECTURES_REGISTRY["modernbert"] = "modern_bert"


# ---------------------------------------------------------------------------
# Architecture Tensor Conversion Tables (HF Safetensors <-> HK Canonical)
# ---------------------------------------------------------------------------

# Standard LLaMA / Mistral / Qwen-style transformer projections
STANDARD_TRANSFORMER_TENSORS: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$", "embed_tokens.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"layers.\1.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"layers.\1.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"layers.\1.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", r"layers.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$", r"layers.\1.attn_q.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$", r"layers.\1.attn_k.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$", r"layers.\1.attn_v.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.bias$", r"layers.\1.attn_output.bias"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"layers.\1.mlp_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"layers.\1.mlp_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"layers.\1.mlp_down.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"layers.\1.input_layernorm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"layers.\1.post_attention_layernorm.weight"),
    (r"^model\.norm\.weight$", "norm.weight"),
    (r"^lm_head\.weight$", "lm_head.weight"),
]

# DeepSeek MLA (Multi-Head Latent Attention) & MoE / MTP
DEEPSEEK_MLA_TENSORS: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$", "embed_tokens.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$", r"layers.\1.attn_q_a.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$", r"layers.\1.attn_q_b.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight$", r"layers.\1.attn_kv_a.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight$", r"layers.\1.attn_kv_b.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", r"layers.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_a_layernorm\.weight$", r"layers.\1.attn_q_a_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_a_layernorm\.weight$", r"layers.\1.attn_kv_a_norm.weight"),
    # MoE Routing & Experts
    (r"^model\.layers\.(\d+)\.mlp\.gate\.weight$", r"layers.\1.moe_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r"layers.\1.experts.\2.gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$", r"layers.\1.experts.\2.up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$", r"layers.\1.experts.\2.down.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$", r"layers.\1.shared_experts.gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$", r"layers.\1.shared_experts.up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$", r"layers.\1.shared_experts.down.weight"),
    # Dense fallback for non-MoE layers
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"layers.\1.mlp_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"layers.\1.mlp_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"layers.\1.mlp_down.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"layers.\1.input_layernorm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"layers.\1.post_attention_layernorm.weight"),
    # Multi-Token Prediction (MTP) modules in DeepSeek V3 / R1
    (r"^model\.layers\.(\d+)\.mtp_linear\.weight$", r"layers.\1.mtp_linear.weight"),
    (r"^model\.norm\.weight$", "norm.weight"),
    (r"^lm_head\.weight$", "lm_head.weight"),
]

# MoE (Mixtral, Qwen2-MoE, OLMoE)
QWEN2_MOE_TENSORS: List[Tuple[str, str]] = [
    (r"^model\.layers\.(\d+)\.mlp\.gate\.weight$", r"layers.\1.ffn_gate_inp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r"layers.\1.experts.\2.gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$", r"layers.\1.experts.\2.up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$", r"layers.\1.experts.\2.down.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj\.weight$", r"layers.\1.shared_expert.gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj\.weight$", r"layers.\1.shared_expert.up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj\.weight$", r"layers.\1.shared_expert.down.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight$", r"layers.\1.shared_expert_gate.weight"),
    *STANDARD_TRANSFORMER_TENSORS,
]

# Gemma / Gemma 2
GEMMA_TENSORS: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$", "embed_tokens.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"layers.\1.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"layers.\1.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"layers.\1.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", r"layers.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"layers.\1.mlp_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"layers.\1.mlp_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"layers.\1.mlp_down.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"layers.\1.input_layernorm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"layers.\1.post_attention_layernorm.weight"),
    (r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$", r"layers.\1.pre_ffn_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$", r"layers.\1.post_ffn_norm.weight"),
    (r"^model\.norm\.weight$", "norm.weight"),
]

# Mamba / SSM
MAMBA_TENSORS: List[Tuple[str, str]] = [
    (r"^backbone\.embeddings\.weight$", "embed_tokens.weight"),
    (r"^backbone\.layers\.(\d+)\.mixer\.in_proj\.weight$", r"layers.\1.ssm_in_proj.weight"),
    (r"^backbone\.layers\.(\d+)\.mixer\.conv1d\.weight$", r"layers.\1.ssm_conv1d.weight"),
    (r"^backbone\.layers\.(\d+)\.mixer\.conv1d\.bias$", r"layers.\1.ssm_conv1d.bias"),
    (r"^backbone\.layers\.(\d+)\.mixer\.x_proj\.weight$", r"layers.\1.ssm_x_proj.weight"),
    (r"^backbone\.layers\.(\d+)\.mixer\.dt_proj\.weight$", r"layers.\1.ssm_dt_proj.weight"),
    (r"^backbone\.layers\.(\d+)\.mixer\.dt_proj\.bias$", r"layers.\1.ssm_dt_proj.bias"),
    (r"^backbone\.layers\.(\d+)\.mixer\.A_log$", r"layers.\1.ssm_a_log"),
    (r"^backbone\.layers\.(\d+)\.mixer\.D$", r"layers.\1.ssm_d"),
    (r"^backbone\.layers\.(\d+)\.mixer\.out_proj\.weight$", r"layers.\1.ssm_out_proj.weight"),
    (r"^backbone\.layers\.(\d+)\.norm\.weight$", r"layers.\1.norm.weight"),
    (r"^backbone\.norm_f\.weight$", "norm.weight"),
    (r"^lm_head\.weight$", "lm_head.weight"),
]

# Vision-Language Models (LLaVA / Qwen2-VL / CLIP)
VLM_TENSORS: List[Tuple[str, str]] = [
    # Language Model
    *STANDARD_TRANSFORMER_TENSORS,
    # Vision Tower
    (r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.weight$", "vision.patch_embed.weight"),
    (r"^vision_tower\.vision_model\.embeddings\.position_embedding\.weight$", "vision.pos_embed.weight"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"vision.layers.\1.attn_q.weight"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"vision.layers.\1.attn_k.weight"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"vision.layers.\1.attn_v.weight"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$", r"vision.layers.\1.attn_out.weight"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight$", r"vision.layers.\1.mlp_fc1.weight"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.weight$", r"vision.layers.\1.mlp_fc2.weight"),
    (r"^vision_tower\.vision_model\.post_layernorm\.weight$", "vision.norm.weight"),
    # Multi-Modal Projector
    (r"^multi_modal_projector\.linear_1\.weight$", "mm_projector.linear_1.weight"),
    (r"^multi_modal_projector\.linear_2\.weight$", "mm_projector.linear_2.weight"),
]

# Audio: Whisper
WHISPER_TENSORS: List[Tuple[str, str]] = [
    (r"^model\.encoder\.conv1\.weight$", "encoder.conv1.weight"),
    (r"^model\.encoder\.conv2\.weight$", "encoder.conv2.weight"),
    (r"^model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"encoder.layers.\1.attn_q.weight"),
    (r"^model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"encoder.layers.\1.attn_k.weight"),
    (r"^model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"encoder.layers.\1.attn_v.weight"),
    (r"^model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$", r"encoder.layers.\1.attn_out.weight"),
    (r"^model\.decoder\.embed_tokens\.weight$", "decoder.embed_tokens.weight"),
    (r"^model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"decoder.layers.\1.self_attn_q.weight"),
    (r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.weight$", r"decoder.layers.\1.cross_attn_q.weight"),
    (r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.weight$", r"decoder.layers.\1.cross_attn_k.weight"),
    (r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.weight$", r"decoder.layers.\1.cross_attn_v.weight"),
    (r"^proj_out\.weight$", "lm_head.weight"),
]

# Diffusion: FLUX
FLUX_TENSORS: List[Tuple[str, str]] = [
    (r"^double_blocks\.(\d+)\.img_attn\.qkv\.weight$", r"double_blocks.\1.img_qkv.weight"),
    (r"^double_blocks\.(\d+)\.img_attn\.proj\.weight$", r"double_blocks.\1.img_proj.weight"),
    (r"^double_blocks\.(\d+)\.txt_attn\.qkv\.weight$", r"double_blocks.\1.txt_qkv.weight"),
    (r"^double_blocks\.(\d+)\.txt_attn\.proj\.weight$", r"double_blocks.\1.txt_proj.weight"),
    (r"^double_blocks\.(\d+)\.img_mlp\.0\.weight$", r"double_blocks.\1.img_mlp_in.weight"),
    (r"^double_blocks\.(\d+)\.img_mlp\.2\.weight$", r"double_blocks.\1.img_mlp_out.weight"),
    (r"^single_blocks\.(\d+)\.linear1\.weight$", r"single_blocks.\1.linear1.weight"),
    (r"^single_blocks\.(\d+)\.linear2\.weight$", r"single_blocks.\1.linear2.weight"),
    (r"^final_layer\.linear\.weight$", "final_layer.weight"),
]

# ModernBERT / Encoders
MODERN_BERT_TENSORS: List[Tuple[str, str]] = [
    (r"^model\.embeddings\.tok_embeddings\.weight$", "embed_tokens.weight"),
    (r"^model\.layers\.(\d+)\.attn\.Wqkv\.weight$", r"layers.\1.attn_qkv.weight"),
    (r"^model\.layers\.(\d+)\.attn\.Wo\.weight$", r"layers.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.Wi\.weight$", r"layers.\1.mlp_in.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.Wo\.weight$", r"layers.\1.mlp_out.weight"),
    (r"^model\.layers\.(\d+)\.attn_norm\.weight$", r"layers.\1.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp_norm\.weight$", r"layers.\1.mlp_norm.weight"),
    (r"^model\.final_norm\.weight$", "norm.weight"),
    (r"^head\.dense\.weight$", "head.dense.weight"),
]


def get_architecture_tensor_mappings(architecture: str) -> List[Tuple[str, str]]:
    """Returns the forward tensor name mapping regexes for the canonical architecture."""
    arch = architecture.lower()
    if "deepseek" in arch:
        return DEEPSEEK_MLA_TENSORS
    elif "moe" in arch or "mixtral" in arch or "olmoe" in arch:
        return QWEN2_MOE_TENSORS
    elif "mamba" in arch or "jamba" in arch:
        return MAMBA_TENSORS
    elif "gemma" in arch:
        return GEMMA_TENSORS
    elif any(v in arch for v in ["llava", "qwen2_vl", "clip", "pixtral", "paligemma"]):
        return VLM_TENSORS
    elif "whisper" in arch:
        return WHISPER_TENSORS
    elif "flux" in arch:
        return FLUX_TENSORS
    elif "bert" in arch:
        return MODERN_BERT_TENSORS
    return STANDARD_TRANSFORMER_TENSORS


def build_reverse_mappings(patterns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Automatically constructs reverse regex mappings (HK -> HF) from forward mappings."""
    reverse_list = []
    for hf_pattern, hk_target in patterns:
        hk_regex = "^" + re.sub(r"\\(\d+)", r"(\\d+)", hk_target) + "$"
        hf_replacement = re.sub(r"\\(\d+)", r"\\\1", hf_pattern.strip("^$"))
        # clean any unescaped parens in replacement
        hf_clean = re.sub(r"\(\\d\+\)", r"\\1", hf_replacement)
        hf_clean = hf_clean.replace(r"\.", ".")
        reverse_list.append((hk_regex, hf_clean))
    return reverse_list
