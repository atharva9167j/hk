"""
HK Adaptive Neural Framework: Self-Play Evolution Engine & LoRA Delta Adaptation
Implements self-play fine-tuning (SPIN-style), closed-loop reward updates,
and automated regression detection with rollback to mitigate context poisoning.
"""

import os
import io
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from .appendix import AppendixRecord, AppendixEntryType, AppendixMetrics, append_record, rollback_appendix, compute_parent_hash

class LoRAAdapter(nn.Module):
    """Low-rank parameter adapter attached to a frozen base linear layer."""
    def __init__(self, target_layer: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.target_layer = target_layer
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze base layer
        for param in self.target_layer.parameters():
            param.requires_grad = False

        in_f = target_layer.in_features
        out_f = target_layer.out_features

        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.target_layer(x)
        delta_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + delta_out

    def serialize_weights(self) -> bytes:
        """Serializes A and B matrices to compact bytes."""
        buf = io.BytesIO()
        torch.save({
            "rank": self.rank,
            "scaling": self.scaling,
            "A": self.lora_A.data.cpu(),
            "B": self.lora_B.data.cpu()
        }, buf)
        return buf.getvalue()

    def load_serialized_weights(self, data: bytes):
        """Loads A and B matrices from binary payload."""
        buf = io.BytesIO(data)
        state = torch.load(buf, weights_only=True)
        self.rank = state["rank"]
        self.scaling = state["scaling"]
        self.lora_A.data.copy_(state["A"])
        self.lora_B.data.copy_(state["B"])

class SelfPlayEvolutionEngine:
    """
    Self-Play evolutionary controller.
    Runs self-refinement iterations, evaluates candidate adaptations against
    task objectives/code evaluation rewards, and appends cryptographic
    version-chained LoRA records to the HK container with regression safeguards.
    """
    def __init__(
        self,
        hk_file_path: str,
        base_model: nn.Module,
        target_layer_name: str,
        lora_rank: int = 4,
        tolerance_drop: float = 0.02
    ):
        self.hk_path = hk_file_path
        self.model = base_model
        self.target_name = target_layer_name
        self.tolerance = tolerance_drop
        self.current_generation = 0
        self.last_payload_hash = b"\x00" * 32
        self.best_metric = 0.0

        # Attach LoRA adapter to target module
        self.adapter = self._attach_adapter(lora_rank)

    def _attach_adapter(self, rank: int) -> LoRAAdapter:
        # Traverse named modules to find target
        components = self.target_name.split(".")
        parent = self.model
        for c in components[:-1]:
            parent = getattr(parent, c)
        target_layer = getattr(parent, components[-1])
        adapter = LoRAAdapter(target_layer, rank=rank)
        setattr(parent, components[-1], adapter)
        return adapter

    def run_evolution_step(
        self,
        train_step_fn: Callable[[nn.Module], float],
        eval_fn: Callable[[nn.Module], Tuple[float, float]], # returns (loss, accuracy/pass_rate)
        step_name: str = "self_play.adapter"
    ) -> Tuple[bool, str, AppendixMetrics]:
        """
        Executes one evolutionary refinement step:
        1. Updates LoRA parameters using train_step_fn.
        2. Evaluates model on validation metrics.
        3. Verifies against regression to prevent context poisoning.
        4. If approved, appends version-chained record to HK container.
        """
        # Snapshot current adapter weights in case of rollback
        prior_state_A = self.adapter.lora_A.data.clone()
        prior_state_B = self.adapter.lora_B.data.clone()

        # Step 1: Run self-play training step
        train_loss = train_step_fn(self.model)

        # Step 2: Evaluate validation metric
        val_loss, val_metric = eval_fn(self.model)

        # Step 3: Check for performance regression
        if self.current_generation > 0 and (val_metric < self.best_metric - self.tolerance):
            # Revert weights to prevent context poisoning
            self.adapter.lora_A.data.copy_(prior_state_A)
            self.adapter.lora_B.data.copy_(prior_state_B)
            return False, f"Regression detected: {val_metric:.3f} < {self.best_metric:.3f}. Rolled back.", AppendixMetrics(loss=val_loss, accuracy=val_metric)

        # Step 4: Evolution approved! Save to HK container
        self.current_generation += 1
        if val_metric > self.best_metric:
            self.best_metric = val_metric

        payload = self.adapter.serialize_weights()
        metrics = AppendixMetrics(
            loss=val_loss,
            accuracy=val_metric,
            pass_rate=val_metric if val_metric <= 1.0 else 1.0,
            custom=train_loss
        )

        record = AppendixRecord(
            entry_type=AppendixEntryType.LORA_ADAPTER,
            name=f"{step_name}.gen{self.current_generation}",
            target=self.target_name,
            generation=self.current_generation,
            parent_hash=self.last_payload_hash,
            metrics=metrics,
            data=payload
        )

        append_record(self.hk_path, record)
        self.last_payload_hash = compute_parent_hash(payload)

        return True, f"Evolution Step {self.current_generation} approved (metric: {val_metric:.3f})", metrics


class SPINLoss(nn.Module):
    """Self-Play Fine-Tuning (SPIN) loss function."""
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, real_logits: torch.Tensor, gen_logits: torch.Tensor) -> torch.Tensor:
        diff = real_logits.mean() - gen_logits.mean()
        return -torch.log(torch.sigmoid(self.beta * diff) + 1e-8)

