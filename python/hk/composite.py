"""
HK Universal & Composite Multi-Modal Pipelines & Dynamic Context Management
Provides universal heterogeneous model orchestration (Audio STT, Vision OCR,
Token/Intent Analysis, Classification, and Generative LLMs) with dynamic routing,
per-stage hyperparameter controls, robust context window management, and container serialization.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HKConfig, AutoConfig
from .modeling import (
    HKLinear,
    HKTransformerBlock,
    HKPreTrainedModel,
    HKForCausalLM,
    HKForSequenceClassification,
    HKForHandwritingRecognition,
    AutoModel,
    ModelOutput,
)
from .tokenizer import HKTokenizer, AutoTokenizer


# ---------------------------------------------------------------------------
# Dynamic Context Window Management
# ---------------------------------------------------------------------------

class ContextWindowManager:
    """
    Manages sequence length, token budgets, and long-context scaling for single
    and multi-stage inference pipelines.
    """

    def __init__(
        self,
        max_context_length: int = 2048,
        default_strategy: str = "middle_out",
        head_ratio: float = 0.25,
    ):
        self.max_context_length = max_context_length
        self.default_strategy = default_strategy
        self.head_ratio = head_ratio

    def truncate(
        self,
        tokens: Union[List[int], torch.Tensor],
        max_tokens: Optional[int] = None,
        strategy: Optional[str] = None,
        head_ratio: Optional[float] = None,
    ) -> Union[List[int], torch.Tensor]:
        """
        Truncates tokens according to the specified retention strategy:
        - 'tail': Keeps the most recent tokens (conversation recency).
        - 'head': Keeps the earliest tokens (system prompt / initial instructions).
        - 'middle_out': Preserves head for instructions and tail for recent turns,
                        dropping tokens from the middle.
        - 'sliding_window': Keeps the final window of tokens.
        """
        limit = max_tokens if max_tokens is not None else self.max_context_length
        strat = strategy if strategy is not None else self.default_strategy
        ratio = head_ratio if head_ratio is not None else self.head_ratio

        is_tensor = isinstance(tokens, torch.Tensor)
        if is_tensor:
            device = tokens.device
            dtype = tokens.dtype
            # Handle 2D [batch, seq] or 1D [seq]
            if tokens.ndim == 2:
                batch_size, seq_len = tokens.shape
                if seq_len <= limit:
                    return tokens
                truncated_rows = [
                    self.truncate(tokens[i].tolist(), max_tokens=limit, strategy=strat, head_ratio=ratio)
                    for i in range(batch_size)
                ]
                return torch.tensor(truncated_rows, dtype=dtype, device=device)
            elif tokens.ndim == 1:
                token_list = tokens.tolist()
            else:
                return tokens
        else:
            token_list = list(tokens)

        total = len(token_list)
        if total <= limit:
            return torch.tensor(token_list) if is_tensor else token_list

        if strat == "tail":
            result = token_list[-limit:]
        elif strat == "head":
            result = token_list[:limit]
        elif strat == "middle_out":
            head_len = max(1, int(limit * ratio))
            tail_len = limit - head_len
            if tail_len <= 0:
                result = token_list[:limit]
            else:
                result = token_list[:head_len] + token_list[-tail_len:]
        elif strat == "sliding_window":
            result = token_list[-limit:]
        else:
            result = token_list[-limit:]

        return torch.tensor(result, dtype=dtype, device=device) if is_tensor else result

    @staticmethod
    def apply_rope_scaling(
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
        scale_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically applies linear position interpolation / NTK scaling to RoPE embeddings
        for extending context windows beyond original training bounds.
        """
        if scale_factor <= 1.0:
            return cos_cached, sin_cached

        orig_shape = cos_cached.shape
        seq_len, dim = orig_shape[-2], orig_shape[-1]
        scaled_inv_freq = 1.0 / (scale_factor ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, scaled_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        res_cos = emb.cos().to(cos_cached.device).view(*orig_shape)
        res_sin = emb.sin().to(sin_cached.device).view(*orig_shape)
        return res_cos, res_sin


# ---------------------------------------------------------------------------
# Heterogeneous Pipeline Models (Whisper STT & DistilBERT Token Analyzer)
# ---------------------------------------------------------------------------

class HKWhisperModel(HKPreTrainedModel):
    """
    Speech-to-text / acoustic model for processing audio waveforms or spectrograms
    into token sequences.
    """

    def __init__(self, config: Optional[HKConfig] = None, device: str = "cpu"):
        if config is None:
            config = HKConfig(
                model_type="whisper",
                vocab_size=51865,
                hidden_size=384,
                num_hidden_layers=4,
                num_attention_heads=6,
            )
        super().__init__(config)
        self.device_name = device
        hidden_dim = config.hidden_size

        # Acoustic convolutional feature extractor (1D waveform -> frames)
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

        # Acoustic transformer layers
        self.layers = nn.ModuleList([
            HKTransformerBlock(config, device=device) for _ in range(config.num_hidden_layers)
        ])

        # CTC / token projection head
        self.proj = HKLinear(hidden_dim, config.vocab_size, bias=True, device=device)

    def forward(self, audio: torch.Tensor) -> ModelOutput:
        """
        Processes audio waveform [batch, samples] or [batch, channels, samples].
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(1)

        x = F.gelu(self.conv1(audio))
        x = F.gelu(self.conv2(x))  # [batch, hidden_dim, seq_len]
        x = x.permute(0, 2, 1)    # [batch, seq_len, hidden_dim]
        x = self.norm(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.proj(x)
        return ModelOutput(logits=logits)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[float]],
        temperature: float = 0.0,
        tokenizer: Optional[Any] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Transcribes audio into token IDs and string representation.
        """
        self.eval()
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        elif isinstance(audio, list):
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
        else:
            audio_tensor = audio.float()

        out = self.forward(audio_tensor)
        logits = out.logits[0]  # [seq_len, vocab_size]

        if temperature <= 0.0:
            pred_ids = torch.argmax(logits, dim=-1).tolist()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            pred_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        # Deduplicate consecutive identical tokens (CTC decoding)
        dedup_ids = []
        last_id = -1
        for tid in pred_ids:
            if tid != last_id and tid != 0:
                dedup_ids.append(tid)
            last_id = tid

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            text = tokenizer.decode(dedup_ids, skip_special_tokens=True)
        else:
            text = "".join(chr((t % 26) + ord("a")) for t in dedup_ids[:40])

        return {
            "token_ids": dedup_ids,
            "text": text,
            "confidence": float(torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()),
        }


class HKDistilBertModel(HKPreTrainedModel):
    """
    Lightweight DistilBERT-style token and intent analyzer for intent classification,
    sentiment analysis, and token importance scoring.
    """

    def __init__(self, config: Optional[HKConfig] = None, device: str = "cpu"):
        if config is None:
            config = HKConfig(
                model_type="distilbert",
                vocab_size=32000,
                hidden_size=256,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_classes=6,  # 6 intent categories
            )
        super().__init__(config)
        self.device_name = device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            HKTransformerBlock(config, device=device) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        # Multi-task heads
        self.intent_head = HKLinear(config.hidden_size, config.num_classes, bias=True, device=device)
        self.sentiment_head = HKLinear(config.hidden_size, 3, bias=True, device=device)  # neg, neu, pos
        self.importance_head = HKLinear(config.hidden_size, 1, bias=True, device=device)

        self.intent_labels = ["question", "command", "statement", "summarization", "code", "clarification"]
        self.sentiment_labels = ["negative", "neutral", "positive"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)

        pooled = x.mean(dim=1)
        intent_logits = self.intent_head(pooled)
        sentiment_logits = self.sentiment_head(pooled)
        importance_scores = torch.sigmoid(self.importance_head(x)).squeeze(-1)

        return ModelOutput(
            logits=intent_logits,
            hidden_states=(pooled, sentiment_logits, importance_scores),
        )

    @torch.no_grad()
    def analyze(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive linguistic and intent analysis over input tokens.
        """
        self.eval()
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        out = self.forward(input_ids, attention_mask=attention_mask)
        intent_probs = torch.softmax(out.logits[0], dim=-1)
        intent_idx = int(torch.argmax(intent_probs).item())
        intent_name = self.intent_labels[intent_idx % len(self.intent_labels)]

        _, sentiment_logits, importance = out.hidden_states
        sentiment_probs = torch.softmax(sentiment_logits[0], dim=-1)
        sentiment_idx = int(torch.argmax(sentiment_probs).item())
        sentiment_name = self.sentiment_labels[sentiment_idx % len(self.sentiment_labels)]

        token_scores = importance[0].tolist()
        scored_tokens = sorted(enumerate(token_scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_tokens[:return_top_k]]

        return {
            "intent": intent_name,
            "intent_confidence": round(float(intent_probs[intent_idx].item()), 4),
            "sentiment": sentiment_name,
            "sentiment_confidence": round(float(sentiment_probs[sentiment_idx].item()), 4),
            "salient_token_positions": top_indices,
            "token_importance_scores": [round(s, 4) for s in token_scores],
        }


# ---------------------------------------------------------------------------
# Universal Pipeline Context Blackboard
# ---------------------------------------------------------------------------

class PipelineContext(dict):
    """
    Blackboard state dictionary shared across stages in a UniversalPipeline.
    Accumulates raw inputs, intermediate representations, and final stage predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "stages_executed" not in self:
            self["stages_executed"] = []
        if "stage_outputs" not in self:
            self["stage_outputs"] = {}
        if "metadata" not in self:
            self["metadata"] = {}

    def set_output(self, stage_name: str, key: str, value: Any) -> None:
        self[key] = value
        self["stage_outputs"][stage_name] = value
        if stage_name not in self["stages_executed"]:
            self["stages_executed"].append(stage_name)

    def get_latest_text(self) -> Optional[str]:
        """Returns the most recent text representation generated in the context."""
        for key in ["generated_text", "transcription", "ocr_text", "text", "prompt"]:
            val = self.get(key)
            if val is not None and isinstance(val, str) and val.strip():
                return val.strip()
        return None


# ---------------------------------------------------------------------------
# Universal Pipeline Stage Abstraction
# ---------------------------------------------------------------------------

class PipelineStage:
    """
    Universal representation of an executable stage within a multi-modal pipeline.
    Can wrap Audio, Vision, NLP Analysis, Classification, Generative LLM, or custom transforms.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        task_type: str = "auto",
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        transform_fn: Optional[Callable[[PipelineContext, Dict[str, Any]], Any]] = None,
        enabled: bool = True,
    ):
        self.name = name
        self.model = model
        self.task_type = task_type
        self.input_mapping = dict(input_mapping or {})
        self.output_key = output_key or name
        self.params = dict(params or {})
        self.transform_fn = transform_fn
        self.enabled = enabled

        # Auto-detect task type from model architecture if requested
        if self.task_type == "auto" and self.model is not None:
            m_name = type(self.model).__name__
            if hasattr(self.model, "transcribe") or "Whisper" in m_name:
                self.task_type = "audio_transcription"
            elif hasattr(self.model, "analyze") or "DistilBert" in m_name:
                self.task_type = "token_analysis"
            elif "Handwriting" in m_name or "Vision" in m_name or "OCR" in m_name:
                self.task_type = "vision_ocr"
            elif "SequenceClassification" in m_name:
                self.task_type = "sequence_classification"
            elif hasattr(self.model, "generate") or "CausalLM" in m_name:
                self.task_type = "text_generation"
            else:
                self.task_type = "generic"

    def execute(
        self,
        context: PipelineContext,
        params_override: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        context_manager: Optional[ContextWindowManager] = None,
    ) -> Any:
        """
        Executes this individual stage against the current pipeline blackboard context.
        """
        if not self.enabled:
            return None

        effective_params = dict(self.params)
        if params_override:
            effective_params.update(params_override)

        # If custom transformation function provided, delegate to it
        if self.transform_fn is not None:
            res = self.transform_fn(context, effective_params)
            context.set_output(self.name, self.output_key, res)
            return res

        res = None

        # 1. Audio Transcription Stage (Whisper)
        if self.task_type in ("audio", "audio_transcription", "whisper", "stt"):
            audio_key = self.input_mapping.get("audio", "audio")
            audio_data = context.get(audio_key)
            if audio_data is not None and self.model is not None:
                trans_res = self.model.transcribe(
                    audio=audio_data,
                    temperature=effective_params.get("temperature", 0.0),
                    tokenizer=tokenizer,
                    sample_rate=effective_params.get("sample_rate", 16000),
                )
                res = trans_res
                text = trans_res.get("text", "")
                context.set_output(self.name, self.output_key, trans_res)
                context["transcription"] = text
                context["transcription_meta"] = trans_res
                current_text = context.get("text", "")
                context["text"] = (current_text + " " + text).strip() if current_text else text
                return res

        # 2. Vision / OCR Stage (Handwriting / Image-to-Text)
        elif self.task_type in ("vision", "vision_ocr", "ocr", "image_to_text", "hwr"):
            img_key = self.input_mapping.get("image", "image")
            img_data = context.get(img_key, context.get("pixel_values"))
            if img_data is not None and self.model is not None:
                if isinstance(img_data, np.ndarray):
                    if img_data.ndim == 2:
                        img_data = img_data[None, None, ...]
                    elif img_data.ndim == 3:
                        img_data = img_data[None, ...]
                    img_tensor = torch.from_numpy(img_data).float()
                elif isinstance(img_data, torch.Tensor):
                    img_tensor = img_data if img_data.ndim == 4 else img_data.unsqueeze(0)
                else:
                    img_tensor = torch.zeros(1, 1, 32, 128)

                with torch.no_grad():
                    out = self.model(img_tensor)
                    preds = torch.argmax(out.logits, dim=-1)[0].tolist()

                if tokenizer is not None and hasattr(tokenizer, "decode"):
                    ocr_text = tokenizer.decode(preds, skip_special_tokens=True)
                else:
                    ocr_text = "".join(chr(p % 26 + ord("a")) for p in preds if p > 0)

                res = {"transcription": ocr_text, "logits": out.logits}
                context.set_output(self.name, self.output_key, res)
                context["ocr_text"] = ocr_text
                current_text = context.get("text", "")
                context["text"] = (current_text + " " + ocr_text).strip() if current_text else ocr_text
                return res

        # 3. Token & Linguistic Analysis Stage (DistilBERT)
        elif self.task_type in ("analysis", "token_analysis", "intent", "sentiment"):
            text_val = context.get_latest_text() or "Hello"
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                input_ids = tokenizer.encode(text_val, add_special_tokens=True)
            else:
                input_ids = [ord(c) % 1000 + 1 for c in text_val]

            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            if self.model is not None:
                res = self.model.analyze(
                    input_tensor,
                    return_top_k=effective_params.get("return_top_k", 3),
                )
                context.set_output(self.name, self.output_key, res)
                context["analysis"] = res
                return res

        # 4. Sequence Classification / Sentiment Stage
        elif self.task_type in ("classification", "sequence_classification", "sentiment_analysis"):
            text_val = context.get_latest_text() or "Hello"
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                input_ids = tokenizer.encode(text_val, add_special_tokens=True)
            else:
                input_ids = [ord(c) % 1000 + 1 for c in text_val]

            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            if self.model is not None:
                with torch.no_grad():
                    out = self.model(input_tensor)
                    probs = torch.softmax(out.logits, dim=-1)[0]
                    pred_class = int(torch.argmax(probs).item())
                    confidence = float(probs[pred_class].item())

                res = {"label": f"LABEL_{pred_class}", "score": round(confidence, 4), "logits": out.logits}
                context.set_output(self.name, self.output_key, res)
                context["classification"] = res
                return res

        # 5. Generative LLM Stage (Causal LM)
        elif self.task_type in ("generation", "text_generation", "causal_lm", "llm"):
            sys_prompt = context.get("system_prompt")
            base_text = context.get_latest_text() or ""
            analysis = context.get("analysis")

            augmented_prompt = base_text
            if sys_prompt:
                augmented_prompt = f"System: {sys_prompt}\nUser: {base_text}"
            elif analysis and "intent" in analysis:
                augmented_prompt = f"[Intent: {analysis['intent']}] {base_text}"

            if tokenizer is not None and hasattr(tokenizer, "encode"):
                prompt_ids = tokenizer.encode(augmented_prompt, add_special_tokens=True)
            else:
                prompt_ids = [ord(c) % 1000 + 1 for c in (augmented_prompt or "Hello")]

            if context_manager is not None:
                strat = effective_params.get("context_strategy", context_manager.default_strategy)
                prompt_ids = context_manager.truncate(
                    prompt_ids,
                    max_tokens=effective_params.get("max_context_length"),
                    strategy=strat,
                    head_ratio=context_manager.head_ratio,
                )

            if self.model is not None:
                dev = getattr(self.model, "device_name", "cpu")
                inp_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
                gen_ids = self.model.generate(
                    input_ids=inp_tensor,
                    max_new_tokens=effective_params.get("max_new_tokens", 40),
                    temperature=effective_params.get("temperature", 0.7),
                    top_k=effective_params.get("top_k", 40),
                )[0].tolist()

                new_tokens = gen_ids[len(prompt_ids):] if len(gen_ids) >= len(prompt_ids) else gen_ids
                if tokenizer is not None and hasattr(tokenizer, "decode"):
                    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                else:
                    gen_text = "".join(chr((t % 26) + ord("a")) for t in new_tokens)

                res = gen_text
                context.set_output(self.name, self.output_key, gen_text)
                context["generated_text"] = gen_text
                context["tokens_in"] = len(prompt_ids)
                return res

        # 6. Generic / Fallback execution
        elif self.model is not None:
            # Try calling model directly with available context tensors
            try:
                out = self.model(**{k: v for k, v in context.items() if isinstance(v, torch.Tensor)})
                context.set_output(self.name, self.output_key, out)
                return out
            except Exception:
                pass

        return res


# ---------------------------------------------------------------------------
# Universal Multi-Modal Pipeline Engine
# ---------------------------------------------------------------------------

class UniversalPipeline:
    """
    Universal multi-modal heterogeneous pipeline execution engine.
    Chains arbitrary sequences of heterogeneous models (Audio STT, Vision OCR,
    Linguistic/Intent Analysis, Classification, and Generative LLMs)
    with fine-grained per-stage hyperparameters, dynamic blackboard routing,
    robust context window management, and container serialization.
    """

    def __init__(
        self,
        stages: Optional[List[PipelineStage]] = None,
        tokenizer: Optional[Any] = None,
        context_manager: Optional[ContextWindowManager] = None,
        device: str = "cpu",
    ):
        self.stages: List[PipelineStage] = list(stages or [])
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(".")
        self.context_manager = context_manager or ContextWindowManager(max_context_length=2048)
        self.device = device
        self.stage_params: Dict[str, Dict[str, Any]] = {}

        for stage in self.stages:
            self.stage_params[stage.name] = dict(stage.params)

    def add_stage(
        self,
        stage_or_name: Optional[Union[PipelineStage, str]] = None,
        model: Optional[Any] = None,
        task_type: str = "auto",
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        transform_fn: Optional[Callable[[PipelineContext, Dict[str, Any]], Any]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> "UniversalPipeline":
        """Appends a new stage to the universal pipeline."""
        target = stage_or_name if stage_or_name is not None else name
        if isinstance(target, PipelineStage):
            stage = target
        else:
            stage_name = str(target) if target is not None else "stage"
            stage = PipelineStage(
                name=stage_name,
                model=model,
                task_type=task_type,
                input_mapping=input_mapping,
                output_key=output_key,
                params=params,
                transform_fn=transform_fn,
            )

        # Replace existing if name matches, else append
        existing_idx = [i for i, s in enumerate(self.stages) if s.name == stage.name]
        if existing_idx:
            self.stages[existing_idx[0]] = stage
        else:
            self.stages.append(stage)

        self.stage_params[stage.name] = dict(stage.params)
        return self

    def remove_stage(self, name: str) -> bool:
        """Removes a stage from the pipeline by name."""
        initial_len = len(self.stages)
        self.stages = [s for s in self.stages if s.name != name]
        if name in self.stage_params:
            del self.stage_params[name]
        return len(self.stages) < initial_len

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Returns the requested pipeline stage object."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def set_stage_param(self, stage_name: str, **kwargs) -> None:
        """Configures per-model runtime parameters for a specific stage."""
        if stage_name not in self.stage_params:
            self.stage_params[stage_name] = {}
        self.stage_params[stage_name].update(kwargs)

        stage = self.get_stage(stage_name)
        if stage is not None:
            stage.params.update(kwargs)

    def get_stage_param(self, stage_name: str) -> Dict[str, Any]:
        """Retrieves configured parameters for a given stage."""
        return dict(self.stage_params.get(stage_name, {}))

    def set_context_strategy(self, strategy: str, head_ratio: Optional[float] = None) -> None:
        """Configures context retention strategy globally across text generation stages."""
        self.context_manager.default_strategy = strategy
        if head_ratio is not None:
            self.context_manager.head_ratio = head_ratio

    def __call__(
        self,
        *args,
        audio: Optional[Any] = None,
        image: Optional[Any] = None,
        text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stage_params: Optional[Dict[str, Dict[str, Any]]] = None,
        context_strategy: Optional[str] = None,
        **kwargs,
    ) -> PipelineContext:
        """
        Executes all stages in the universal pipeline sequentially,
        threading the blackboard context between them.
        """
        context = PipelineContext()
        context["text"] = text or ""
        if audio is not None:
            context["audio"] = audio
        if image is not None:
            context["image"] = image
        if system_prompt is not None:
            context["system_prompt"] = system_prompt

        context.update(kwargs)

        strat = context_strategy or self.context_manager.default_strategy

        for stage in self.stages:
            if not stage.enabled:
                continue

            stage_override = {}
            if stage.name in self.stage_params:
                stage_override.update(self.stage_params[stage.name])
            if stage_params and stage.name in stage_params:
                stage_override.update(stage_params[stage.name])
            if context_strategy:
                stage_override["context_strategy"] = strat

            stage.execute(
                context=context,
                params_override=stage_override,
                tokenizer=self.tokenizer,
                context_manager=self.context_manager,
            )

        # Consolidate standard summary properties
        if "generated_text" not in context:
            context["generated_text"] = context.get_latest_text() or ""
        if "transcription" not in context:
            context["transcription"] = None

        return context

    def save_pipeline(self, output_dir: Union[str, Path]) -> str:
        """
        Saves the universal pipeline manifest and all stage models into a directory.
        """
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "pipeline_type": "universal",
            "stages": [],
            "stage_params": self.stage_params,
            "context_manager": {
                "max_context_length": self.context_manager.max_context_length,
                "default_strategy": self.context_manager.default_strategy,
                "head_ratio": self.context_manager.head_ratio,
            },
        }

        for stage in self.stages:
            stage_entry: Dict[str, Any] = {
                "name": stage.name,
                "task_type": stage.task_type,
                "input_mapping": stage.input_mapping,
                "output_key": stage.output_key,
                "params": stage.params,
                "enabled": stage.enabled,
            }

            if stage.model is not None:
                model_fname = f"{stage.name}.hk"
                model_path = target_dir / model_fname
                if hasattr(stage.model, "save_pretrained"):
                    stage.model.save_pretrained(model_path)
                    stage_entry["model_file"] = model_fname
                elif isinstance(stage.model, nn.Module):
                    from .torch import save_file
                    save_file(stage.model.state_dict(), str(model_path))
                    stage_entry["model_file"] = model_fname

            manifest["stages"].append(stage_entry)

        manifest_file = target_dir / "pipeline_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return str(manifest_file)

    @classmethod
    def load_pipeline(cls, pipeline_dir: Union[str, Path], device: str = "cpu") -> "UniversalPipeline":
        """
        Reconstructs a universal pipeline and loads its component models from disk.
        """
        p_dir = Path(pipeline_dir)
        manifest_file = p_dir / "pipeline_manifest.json"
        if not manifest_file.is_file():
            # Check for composite_manifest.json fallback
            alt = p_dir / "composite_manifest.json"
            if alt.is_file():
                manifest_file = alt
            else:
                raise FileNotFoundError(f"Missing pipeline_manifest.json in {pipeline_dir}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cm_data = data.get("context_manager", {})
        ctx_mgr = ContextWindowManager(
            max_context_length=cm_data.get("max_context_length", 2048),
            default_strategy=cm_data.get("default_strategy", "middle_out"),
            head_ratio=cm_data.get("head_ratio", 0.25),
        )

        pipeline = cls(context_manager=ctx_mgr, device=device)

        stages_list = data.get("stages", [])
        if isinstance(stages_list, list):
            for entry in stages_list:
                name = entry["name"]
                task_type = entry.get("task_type", "auto")
                m_file = entry.get("model_file")
                model = None
                if m_file:
                    full_m_path = p_dir / m_file
                    if full_m_path.is_file():
                        try:
                            model = AutoModel.from_pretrained(str(full_m_path), device=device)
                        except Exception:
                            if task_type in ("audio", "audio_transcription", "whisper"):
                                model = HKWhisperModel(device=device)
                            elif task_type in ("analysis", "token_analysis"):
                                model = HKDistilBertModel(device=device)
                            else:
                                model = HKForCausalLM(HKConfig(hidden_size=256), device=device)

                pipeline.add_stage(
                    name=name,
                    model=model,
                    task_type=task_type,
                    input_mapping=entry.get("input_mapping"),
                    output_key=entry.get("output_key"),
                    params=entry.get("params"),
                )
        elif isinstance(stages_list, dict):
            for name, m_file in stages_list.items():
                pipeline.add_stage(name=name, model=None, task_type="auto")

        return pipeline

    # --- Presets & Factory Builders ---

    @classmethod
    def speech_analysis_generation(
        cls,
        whisper_model: Optional[HKWhisperModel] = None,
        analyzer_model: Optional[HKDistilBertModel] = None,
        llm_model: Optional[HKForCausalLM] = None,
        tokenizer: Optional[Any] = None,
        context_manager: Optional[ContextWindowManager] = None,
        device: str = "cpu",
    ) -> "UniversalPipeline":
        """Factory for 3-stage Audio STT -> Intent Analysis -> LLM Generation."""
        pipe = cls(tokenizer=tokenizer, context_manager=context_manager, device=device)
        if whisper_model is None:
            whisper_model = HKWhisperModel(device=device)
        if analyzer_model is None:
            analyzer_model = HKDistilBertModel(device=device)
        if llm_model is None:
            llm_model = HKForCausalLM(
                HKConfig(model_type="causal_lm", hidden_size=256, num_hidden_layers=2, num_attention_heads=4),
                device=device,
            )

        pipe.add_stage("whisper", whisper_model, task_type="audio_transcription", params={"temperature": 0.0})
        pipe.add_stage("analyzer", analyzer_model, task_type="token_analysis", params={"return_top_k": 3})
        pipe.add_stage("llm", llm_model, task_type="text_generation", params={"temperature": 0.7, "max_new_tokens": 40})
        return pipe

    @classmethod
    def vision_analysis_generation(
        cls,
        ocr_model: Optional[HKForHandwritingRecognition] = None,
        analyzer_model: Optional[HKDistilBertModel] = None,
        llm_model: Optional[HKForCausalLM] = None,
        tokenizer: Optional[Any] = None,
        context_manager: Optional[ContextWindowManager] = None,
        device: str = "cpu",
    ) -> "UniversalPipeline":
        """Factory for 3-stage Vision/OCR -> Token Analysis -> LLM Generation."""
        pipe = cls(tokenizer=tokenizer, context_manager=context_manager, device=device)
        if ocr_model is None:
            ocr_model = HKForHandwritingRecognition(HKConfig(model_type="hwr", hidden_size=64, num_classes=26), device=device)
        if analyzer_model is None:
            analyzer_model = HKDistilBertModel(device=device)
        if llm_model is None:
            llm_model = HKForCausalLM(
                HKConfig(model_type="causal_lm", hidden_size=256, num_hidden_layers=2, num_attention_heads=4),
                device=device,
            )

        pipe.add_stage("ocr", ocr_model, task_type="vision_ocr")
        pipe.add_stage("analyzer", analyzer_model, task_type="token_analysis", params={"return_top_k": 3})
        pipe.add_stage("llm", llm_model, task_type="text_generation", params={"temperature": 0.7, "max_new_tokens": 40})
        return pipe


# ---------------------------------------------------------------------------
# Backward-Compatible CompositePipeline
# ---------------------------------------------------------------------------

class CompositePipeline(UniversalPipeline):
    """
    Backward-compatible composite multi-model pipeline (Whisper -> DistilBERT -> LLM)
    built on top of the UniversalPipeline architecture.
    """

    def __init__(
        self,
        whisper_model: Optional[HKWhisperModel] = None,
        analyzer_model: Optional[HKDistilBertModel] = None,
        llm_model: Optional[HKForCausalLM] = None,
        tokenizer: Optional[Any] = None,
        context_manager: Optional[ContextWindowManager] = None,
        device: str = "cpu",
    ):
        super().__init__(tokenizer=tokenizer, context_manager=context_manager, device=device)

        self.whisper = whisper_model or HKWhisperModel(device=device)
        self.analyzer = analyzer_model or HKDistilBertModel(device=device)
        self.llm = llm_model or HKForCausalLM(
            HKConfig(model_type="causal_lm", hidden_size=256, num_hidden_layers=2, num_attention_heads=4),
            device=device,
        )

        self.add_stage("whisper", self.whisper, task_type="audio_transcription", params={"temperature": 0.0, "sample_rate": 16000})
        self.add_stage("analyzer", self.analyzer, task_type="token_analysis", params={"return_top_k": 3})
        self.add_stage("llm", self.llm, task_type="text_generation", params={"temperature": 0.7, "max_new_tokens": 50, "top_k": 40})

    def save_composite(self, output_dir: Union[str, Path]) -> str:
        """Alias for save_pipeline for backward compatibility."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = self.save_pipeline(output_dir)

        # Also write legacy composite_manifest.json for 100% backward compatibility
        legacy_manifest = {
            "pipeline_type": "composite",
            "stages": {
                "whisper": "whisper.hk",
                "analyzer": "analyzer.hk",
                "llm": "llm.hk",
            },
            "stage_params": self.stage_params,
            "context_manager": {
                "max_context_length": self.context_manager.max_context_length,
                "default_strategy": self.context_manager.default_strategy,
                "head_ratio": self.context_manager.head_ratio,
            },
        }
        with open(target_dir / "composite_manifest.json", "w", encoding="utf-8") as f:
            json.dump(legacy_manifest, f, indent=2)

        return str(target_dir / "composite_manifest.json")
