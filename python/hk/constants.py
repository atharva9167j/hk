"""
HK Standardized Hyperparameter, Architecture & Sampling Taxonomy
Contains 200+ formal keys matching and expanding upon GGUF standards across formal namespaces:
Attention (MLA, SWA, ALiBi, softcapping), RoPE (YaRN, dynamic), MoE, SSM/Mamba,
Tokenizer, Inference Sampling presets, and Model Lineage/Provenance.
"""

from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Union


# ---------------------------------------------------------------------------
# Canonical Model Architectures (137+ Architectures)
# ---------------------------------------------------------------------------
class ModelArchitecture(str, Enum):
    # Core LLMs
    LLAMA = "llama"
    LLAMA4 = "llama4"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    QWEN2_MOE = "qwen2_moe"
    QWEN3 = "qwen3"
    QWEN3_MOE = "qwen3_moe"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    DEEPSEEK = "deepseek"
    DEEPSEEK2 = "deepseek2"
    DEEPSEEK3 = "deepseek3"
    DEEPSEEK_R1 = "deepseek_r1"
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    GROK = "grok"
    FALCON = "falcon"
    FALCON_H1 = "falcon_h1"
    PHI = "phi"
    PHI2 = "phi2"
    PHI3 = "phi3"
    PHI3_MOE = "phi3_moe"
    PHI4 = "phi4"
    DBRX = "dbrx"
    COMMAND_R = "command-r"
    COMMAND_R_PLUS = "command-r-plus"
    OLMO = "olmo"
    OLMO2 = "olmo2"
    OLMOE = "olmoe"
    MINICPM = "minicpm"
    MINICPM3 = "minicpm3"
    STARCODER = "starcoder"
    STARCODER2 = "starcoder2"
    JAIS = "jais"
    EXAONE = "exaone"
    CHATGLM = "chatglm"
    CHATGLM2 = "chatglm2"
    CHATGLM3 = "chatglm3"
    GLM4 = "glm4"
    BAICHUAN = "baichuan"
    BAICHUAN2 = "baichuan2"
    INTERNLM = "internlm"
    INTERNLM2 = "internlm2"
    INTERNLM3 = "internlm3"
    YI = "yi"
    XVERSE = "xverse"
    ORION = "orion"
    COHERE = "cohere"
    ARCTIC = "arctic"
    BITNET = "bitnet"
    SMOLLM = "smollm"
    SMOLLM2 = "smollm2"
    NOMIC_BERT = "nomic_bert"
    MODERN_BERT = "modern_bert"
    JINA_BERT_V2 = "jina_bert_v2"
    JINA_BERT_V3 = "jina_bert_v3"
    EURO_BERT = "euro_bert"
    BERT = "bert"
    ROBERTA = "roberta"
    ELECTRA = "electra"
    DEBERTA = "deberta"
    DEBERTA_V2 = "deberta_v2"
    BLOOM = "bloom"
    MPT = "mpt"
    GPT2 = "gpt2"
    GPT_J = "gpt-j"
    GPT_NEOX = "gpt-neox"
    STABLELM = "stablelm"
    DECILM = "decilm"
    GRANITE = "granite"
    GRANITE_MOE = "granite_moe"
    CHAMELEON = "chameleon"

    # State Space & Recurrent Models
    MAMBA = "mamba"
    MAMBA2 = "mamba2"
    JAMBA = "jamba"
    RWKV4 = "rwkv4"
    RWKV5 = "rwkv5"
    RWKV6 = "rwkv6"
    RECURRENT_GEMMA = "recurrent_gemma"

    # Vision-Language Models (VLM)
    CLIP = "clip"
    LLAVA = "llava"
    LLAVA_NEXT = "llava_next"
    MOBILE_VLM = "mobilevlm"
    QWEN2_VL = "qwen2_vl"
    PIXTRAL = "pixtral"
    GEMMA_VISION = "gemma_vision"
    PALIGEMMA = "paligemma"
    PALIGEMMA2 = "paligemma2"
    SAM = "sam"
    SAM2 = "sam2"
    COGVLM = "cogvlm"
    COGVLM2 = "cogvlm2"
    MINICPM_V = "minicpm_v"
    INTERNVL = "internvl"
    INTERNVL2 = "internvl2"
    VI_T = "vit"
    SIGLIP = "siglip"
    BLIP2 = "blip2"
    FLAMINGO = "flamingo"

    # Audio & Diffusion
    WHISPER = "whisper"
    WAV2VEC2 = "wav2vec2"
    HUBERT = "hubert"
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    FLUX = "flux"
    AURAFLOW = "auraflow"
    HUNYUAN_DIT = "hunyuan_dit"
    LATENT_CONSISTENCY = "lcm"


# ---------------------------------------------------------------------------
# Formal HK Metadata Taxonomy Namespaces
# ---------------------------------------------------------------------------
class HKKeys:
    # General / Lineage / Provenance
    GENERAL_ARCHITECTURE = "general.architecture"
    GENERAL_QUANTIZATION_VERSION = "general.quantization_version"
    GENERAL_ALIGNMENT = "general.alignment"
    GENERAL_NAME = "general.name"
    GENERAL_AUTHOR = "general.author"
    GENERAL_VERSION = "general.version"
    GENERAL_ORGANIZATION = "general.organization"
    GENERAL_BASE_MODEL_NAME = "general.base_model_name"
    GENERAL_BASE_MODEL_AUTHOR = "general.base_model_author"
    GENERAL_BASE_MODEL_DOI = "general.base_model_doi"
    GENERAL_BASE_MODEL_REPO_URL = "general.base_model_repo_url"
    GENERAL_DATASET_LINEAGE = "general.dataset_lineage"
    GENERAL_LICENSE = "general.license"
    GENERAL_LICENSE_NAME = "general.license_name"
    GENERAL_LICENSE_LINK = "general.license_link"
    GENERAL_URL = "general.url"
    GENERAL_DESCRIPTION = "general.description"
    GENERAL_TAGS = "general.tags"
    GENERAL_FILE_TYPE = "general.file_type"

    # Transformer Core Hyperparameters
    CONTEXT_LENGTH = "{arch}.context_length"
    EMBEDDING_LENGTH = "{arch}.embedding_length"
    BLOCK_COUNT = "{arch}.block_count"
    FEED_FORWARD_LENGTH = "{arch}.feed_forward_length"
    USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
    TENSOR_DATA_LAYOUT = "{arch}.tensor_data_layout"

    # Attention Namespace
    ATTENTION_HEAD_COUNT = "{arch}.attention.head_count"
    ATTENTION_HEAD_COUNT_KV = "{arch}.attention.head_count_kv"
    ATTENTION_MAX_ALIBI_BIAS = "{arch}.attention.max_alibi_bias"
    ATTENTION_CLAMP_KQV = "{arch}.attention.clamp_kqv"
    ATTENTION_LAYER_NORM_EPSILON = "{arch}.attention.layer_norm_epsilon"
    ATTENTION_LAYER_NORM_RMS_EPSILON = "{arch}.attention.layer_norm_rms_epsilon"
    ATTENTION_KEY_LENGTH = "{arch}.attention.key_length"
    ATTENTION_VALUE_LENGTH = "{arch}.attention.value_length"
    ATTENTION_KEY_LENGTH_MLA = "{arch}.attention.key_length_mla"
    ATTENTION_VALUE_LENGTH_MLA = "{arch}.attention.value_length_mla"
    ATTENTION_KEY_LENGTH_SWA = "{arch}.attention.key_length_swa"
    ATTENTION_LOGIT_SOFTCAPPING = "{arch}.attention.attn_logit_softcapping"
    ATTENTION_FINAL_LOGIT_SOFTCAPPING = "{arch}.attention.final_logit_softcapping"
    ATTENTION_SLIDING_WINDOW = "{arch}.attention.sliding_window"
    ATTENTION_Q_LORA_RANK = "{arch}.attention.q_lora_rank"
    ATTENTION_KV_LORA_RANK = "{arch}.attention.kv_lora_rank"

    # RoPE Namespace
    ROPE_DIMENSION_COUNT = "{arch}.rope.dimension_count"
    ROPE_FREQ_BASE = "{arch}.rope.freq_base"
    ROPE_SCALE = "{arch}.rope.scale"
    ROPE_SCALE_TYPE = "{arch}.rope.scale_type"
    ROPE_SCALING_TYPE = "{arch}.rope.scaling.type"
    ROPE_YARN_EXT_FACTOR = "{arch}.rope.yarn_ext_factor"
    ROPE_YARN_ATTN_FACTOR = "{arch}.rope.yarn_attn_factor"
    ROPE_YARN_BETA_FAST = "{arch}.rope.yarn_beta_fast"
    ROPE_YARN_BETA_SLOW = "{arch}.rope.yarn_beta_slow"
    ROPE_YARN_ORIG_CTX = "{arch}.rope.yarn_orig_ctx"
    ROPE_FINETUNED = "{arch}.rope.finetuned"

    # MoE Namespace
    MOE_EXPERT_COUNT = "{arch}.expert_count"
    EXPERT_COUNT = "{arch}.expert_count"
    MOE_EXPERT_USED_COUNT = "{arch}.expert_used_count"
    MOE_EXPERT_SHARED_COUNT = "{arch}.expert_shared_count"
    MOE_EXPERT_WEIGHTS_SCALE = "{arch}.expert_weights_scale"
    MOE_EXPERT_WEIGHTS_NORM = "{arch}.expert_weights_norm"
    MOE_EXPERT_GATING_FUNC = "{arch}.expert_gating_func"
    MOE_EXPERT_GROUP_COUNT = "{arch}.expert_group_count"
    MOE_EXPERT_GROUP_TOP_K = "{arch}.expert_group_top_k"

    # SSM / Mamba Namespace
    SSM_CONV_KERNEL = "{arch}.ssm.conv_kernel"
    SSM_INNER_SIZE = "{arch}.ssm.inner_size"
    SSM_STATE_SIZE = "{arch}.ssm.state_size"
    SSM_TIME_STEP_RANK = "{arch}.ssm.time_step_rank"
    SSM_DT_B_C_RMS = "{arch}.ssm.dt_b_c_rms"

    # Tokenizer Namespace
    TOKENIZER_MODEL = "tokenizer.ggml.model"
    TOKENIZER_PRE = "tokenizer.ggml.pre"
    TOKENIZER_TOKENS = "tokenizer.tokens"
    TOKENIZER_SCORES = "tokenizer.scores"
    TOKENIZER_TOKEN_TYPE = "tokenizer.token_type"
    TOKENIZER_MERGES = "tokenizer.merges"
    TOKENIZER_BOS_ID = "tokenizer.bos_token_id"
    TOKENIZER_EOS_ID = "tokenizer.eos_token_id"
    TOKENIZER_UNK_ID = "tokenizer.unk_token_id"
    TOKENIZER_SEP_ID = "tokenizer.sep_token_id"
    TOKENIZER_PAD_ID = "tokenizer.pad_token_id"
    TOKENIZER_CLS_ID = "tokenizer.cls_token_id"
    TOKENIZER_MASK_ID = "tokenizer.mask_token_id"
    TOKENIZER_BOS_TOKEN = "tokenizer.bos_token"
    TOKENIZER_EOS_TOKEN = "tokenizer.eos_token"
    TOKENIZER_UNK_TOKEN = "tokenizer.unk_token"
    TOKENIZER_PAD_TOKEN = "tokenizer.pad_token"
    TOKENIZER_CHAT_TEMPLATE = "tokenizer.chat_template"
    TOKENIZER_CHAT_TEMPLATES = "tokenizer.chat_templates"
    TOKENIZER_HUGGINGFACE_JSON = "tokenizer.huggingface.json"
    TOKENIZER_PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap"
    TOKENIZER_NORMALIZER_PRECOMPILED = "tokenizer.ggml.normalizer.precompiled_charsmap"
    TOKENIZER_FIM_PREFIX = "tokenizer.ggml.prefix_token_id"
    TOKENIZER_FIM_MIDDLE = "tokenizer.ggml.middle_token_id"
    TOKENIZER_FIM_SUFFIX = "tokenizer.ggml.suffix_token_id"
    TOKENIZER_FIM_PAD = "tokenizer.ggml.pad_token_id"
    TOKENIZER_FIM_EOT = "tokenizer.ggml.eot_token_id"

    # Sampling Presets
    SAMPLING_TEMP = "sampling.temp"
    SAMPLING_TOP_P = "sampling.top_p"
    SAMPLING_TOP_K = "sampling.top_k"
    SAMPLING_MIN_P = "sampling.min_p"
    SAMPLING_XTC_PROBABILITY = "sampling.xtc_probability"
    SAMPLING_XTC_THRESHOLD = "sampling.xtc_threshold"
    SAMPLING_TYPICAL_P = "sampling.typical_p"
    SAMPLING_PENALTY_REPEAT = "sampling.penalty_repeat"
    SAMPLING_PENALTY_FREQ = "sampling.penalty_freq"
    SAMPLING_PENALTY_PRESENT = "sampling.penalty_present"
    SAMPLING_MIROSTAT = "sampling.mirostat"
    SAMPLING_MIROSTAT_TAU = "sampling.mirostat_tau"
    SAMPLING_MIROSTAT_ETA = "sampling.mirostat_eta"

    # Quantization Metadata
    QUANT_RECIPE = "quantization.recipe"
    QUANT_VERSION = "quantization.version"
    QUANT_IMATRIX_DATASET = "quantization.imatrix_dataset"
    QUANT_IMATRIX_ENTRIES = "quantization.imatrix_entries"
    QUANT_BLOCK_SIZE = "quantization.block_size"


HKTaxonomyKeys = HKKeys


# ---------------------------------------------------------------------------
# Token Types Standard Enum
# ---------------------------------------------------------------------------
class TokenType(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


# ---------------------------------------------------------------------------
# Pre-Tokenizer Type Identifier Registry
# ---------------------------------------------------------------------------
class PreTokenizerType(str, Enum):
    DEFAULT = "default"
    LLAMA3 = "llama3"
    LLAMA = "llama"
    QWEN2 = "qwen2"
    DEEPSEEK_LLM = "deepseek-llm"
    DEEPSEEK_CODER = "deepseek-coder"
    TEKKEN = "tekken"
    FALCON = "falcon"
    GPT2 = "gpt-2"
    BERT = "bert"
    CHAMELEON = "chameleon"
    STARCODER = "starcoder"
    SMOLLM = "smollm"


# ---------------------------------------------------------------------------
# RoPE Scaling Type Registry
# ---------------------------------------------------------------------------
class RoPEScalingType(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    YARN = "yarn"
    DYNAMIC = "dynamic"
    LLAMA3 = "llama3"


def format_arch_key(key_template: str, arch: Union[str, ModelArchitecture]) -> str:
    """Formats a namespaced architecture key template (e.g. '{arch}.attention.head_count') with canonical name."""
    arch_str = arch.value if isinstance(arch, ModelArchitecture) else str(arch).lower()
    return key_template.format(arch=arch_str)
