"""
HK Adaptive Neural Framework: Autonomous Self-Training Pipeline
Coordinates self-conversational thinking, autonomous expansion need evaluation,
dynamic Net2Net growth with plasticity isolation, and sandboxed code self-training.
"""

import os
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Set

import torch
import torch.nn as nn

from .appendix import (
    AppendixRecord,
    AppendixEntryType,
    AppendixFlags,
    AppendixMetrics,
    append_record,
)
from .growth import (
    GrowthGovernor,
    expand_vocab,
    expand_model_width,
    protect_base_capacity,
)
from .code_eval import CodeSandbox, TestCase, EvalResult
from .expansion_evaluator import ExpansionEvaluator, ExpansionDiagnosis
from .self_conversation import SelfConversationalEngine, SelfDialogue


@dataclass
class SelfTrainingCurriculum:
    """Specification of the target domain/programming language curriculum."""
    domain_name: str
    target_language: str
    syntax_keywords: List[str] = field(default_factory=list)
    diagnostic_tasks: List[Dict[str, Any]] = field(default_factory=list)
    training_tasks: List[Dict[str, Any]] = field(default_factory=list)
    validation_tasks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationReport:
    """Summary of a single self-training generation."""
    generation: int
    train_loss: float
    pass_rate: float
    expansion_occurred: bool
    expansion_details: str
    evaluated_tasks: int
    successful_dialogues: int


class SelfTrainingPipeline:
    """
    Autonomous End-to-End Self-Training Pipeline.
    1. Evaluates whether the model needs architecture expansion for a given domain/language.
    2. If yes, autonomously expands vocabulary and model width with zero-init function preservation.
    3. Engages plasticity isolation to shield pre-existing native capabilities.
    4. Trains the model through self-conversational inner monologue (<think>...</think>) and CodeSandbox verification.
    5. Persists the evolved container with cryptographic Appendix lineage.
    """

    def __init__(
        self,
        model: nn.Module,
        hk_file_path: str,
        governor: Optional[GrowthGovernor] = None,
        evaluator: Optional[ExpansionEvaluator] = None,
        conversational_engine: Optional[SelfConversationalEngine] = None,
        learning_rate: float = 3e-3,
        rehearsal_ratio: float = 0.10,
    ):
        self.model = model
        self.hk_path = hk_file_path
        self.governor = governor or GrowthGovernor(max_growth_ratio=2.0)
        self.evaluator = evaluator or ExpansionEvaluator(governor=self.governor)
        self.engine = conversational_engine or SelfConversationalEngine()
        self.lr = learning_rate
        self.rehearsal_ratio = rehearsal_ratio

        self.current_generation = 0
        self.expansion_applied = False
        self.plasticity_hooks: List[Any] = []
        self.known_vocab: Set[str] = set()

    def set_known_vocab(self, vocab_tokens: Set[str]):
        """Sets the set of known vocabulary tokens for deficit evaluation."""
        self.known_vocab = set(vocab_tokens)

    def evaluate_expansion_need(
        self,
        curriculum: SelfTrainingCurriculum,
        diagnostic_loss_fn: Optional[Callable[[nn.Module], float]] = None,
    ) -> ExpansionDiagnosis:
        """
        Probes the model's current capability on the target programming language curriculum
        and evaluates whether dynamic architecture expansion is required.
        """
        # Diagnostic pass rate evaluation
        passed = 0
        total = len(curriculum.diagnostic_tasks)

        for task in curriculum.diagnostic_tasks:
            test_cases = [
                TestCase(input_call=tc["call"], expected_output=tc["expected"], description=tc.get("desc", ""))
                for tc in task.get("test_cases", [])
            ]
            eval_res = self.engine.sandbox.execute_code(task.get("baseline_code", ""), test_cases=test_cases)
            if eval_res.success:
                passed += 1

        pass_rate = (passed / max(total, 1)) if total > 0 else 0.0

        # Loss measurement
        loss_val = 5.0
        if diagnostic_loss_fn is not None:
            loss_val = diagnostic_loss_fn(self.model)
        elif pass_rate < 0.20:
            loss_val = 4.6052 # Random guessing ceiling

        diagnosis = self.evaluator.evaluate_domain_capacity(
            model=self.model,
            diagnostic_loss=loss_val,
            diagnostic_pass_rate=pass_rate,
            known_vocab=self.known_vocab,
            domain_keywords=curriculum.syntax_keywords,
        )
        return diagnosis

    def apply_autonomous_expansion(
        self,
        diagnosis: ExpansionDiagnosis,
        curriculum: SelfTrainingCurriculum,
    ) -> Dict[str, Any]:
        """
        Autonomously executes vocabulary growth, intermediate width expansion,
        and registers plasticity isolation hooks.
        """
        expansion_details = []

        # 1. Vocab expansion
        if diagnosis.needs_vocab_expansion and diagnosis.suggested_new_tokens:
            current_vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 100)
            tokens_to_add = len(diagnosis.suggested_new_tokens)
            new_vocab_size = current_vocab_size + tokens_to_add

            expand_vocab(self.model, new_vocab_size=new_vocab_size)
            self.known_vocab.update(diagnosis.suggested_new_tokens)
            expansion_details.append(f"Vocab expanded: {current_vocab_size} -> {new_vocab_size} (+{tokens_to_add} tokens)")

        # 2. Width expansion
        if diagnosis.needs_width_expansion and diagnosis.suggested_width_ratio > 1.0:
            expand_model_width(
                self.model,
                expansion_ratio=diagnosis.suggested_width_ratio,
                noise_std=0.0,
            )
            new_inter = getattr(getattr(self.model, "config", None), "intermediate_size", 128)
            expansion_details.append(f"Width expanded by {diagnosis.suggested_width_ratio:.2f}x (intermediate_size={new_inter})")

        # 3. Plasticity Protection
        self.plasticity_hooks = protect_base_capacity(self.model)
        expansion_details.append(f"Plasticity shield active: {len(self.plasticity_hooks)} gradient hooks registered")

        self.expansion_applied = True
        self.current_generation += 1

        # Record expansion in HK container Appendix if path provided
        if os.path.exists(self.hk_path):
            details_str = "; ".join(expansion_details)
            record = AppendixRecord(
                entry_type=AppendixEntryType.NEW_LAYER,
                flags=AppendixFlags.ACTIVE,
                name=f"autonomous_expansion.gen{self.current_generation}",
                target="model.architecture",
                generation=self.current_generation,
                metrics=AppendixMetrics(
                    loss=diagnosis.diagnostics.get("diagnostic_loss", 0.0),
                    accuracy=diagnosis.diagnostics.get("diagnostic_pass_rate", 0.0),
                    pass_rate=diagnosis.diagnostics.get("diagnostic_pass_rate", 0.0),
                    custom=float(getattr(getattr(self.model, "config", None), "intermediate_size", 128)),
                ),
                data=json.dumps({
                    "reason": diagnosis.rationale,
                    "curriculum": curriculum.domain_name,
                    "details": details_str,
                }).encode("utf-8"),
            )
            append_record(self.hk_path, record)

        return {
            "success": True,
            "generation": self.current_generation,
            "details": "; ".join(expansion_details),
            "hooks_count": len(self.plasticity_hooks),
        }

    def train_generation(
        self,
        curriculum: SelfTrainingCurriculum,
        solution_generator_fn: Callable[[str], str],
        optimizer_step_fn: Callable[[List[SelfDialogue], nn.Module], float],
        base_rehearsal_batch: Optional[Any] = None,
    ) -> GenerationReport:
        """
        Executes one complete generation of self-training:
        1. Explores curriculum tasks through self-conversational inner dialogue.
        2. Evaluates code solutions in CodeSandbox.
        3. Filters dialogues with verified pass rate.
        4. Trains expanded parameters using the verified reasoning traces.
        5. Logs CodeSandbox evaluation results into Appendix.
        """
        dialogues: List[SelfDialogue] = []
        passed_count = 0

        # Step 1 & 2: Self-conversational thinking and sandbox verification
        for task in curriculum.training_tasks:
            task_id = task.get("id", f"task_{random.randint(1000, 9999)}")
            prompt = task["prompt"]
            test_cases = [
                TestCase(input_call=tc["call"], expected_output=tc["expected"], description=tc.get("desc", ""))
                for tc in task.get("test_cases", [])
            ]

            dialogue = self.engine.conduct_self_dialogue(
                task_id=task_id,
                task_prompt=prompt,
                initial_solution_fn=solution_generator_fn,
                test_cases=test_cases,
                target_language=curriculum.target_language,
            )
            dialogues.append(dialogue)
            if dialogue.success:
                passed_count += 1

        total_tasks = len(dialogues)
        pass_rate = (passed_count / max(total_tasks, 1)) if total_tasks > 0 else 0.0

        # Step 3: Optimize model on successful reasoning trajectories
        successful_dialogues = [d for d in dialogues if d.success]
        train_loss = 0.0
        if successful_dialogues:
            train_loss = optimizer_step_fn(successful_dialogues, self.model)

        # Step 4: Persist evaluation report into HK container appendix
        if os.path.exists(self.hk_path):
            eval_record = AppendixRecord(
                entry_type=AppendixEntryType.CODE_EVAL,
                flags=AppendixFlags.ACTIVE,
                name=f"self_training_eval.gen{self.current_generation}",
                target=f"curriculum.{curriculum.target_language}",
                generation=self.current_generation,
                metrics=AppendixMetrics(
                    loss=train_loss,
                    accuracy=pass_rate,
                    pass_rate=pass_rate,
                    custom=float(passed_count),
                ),
                data=json.dumps({
                    "total_tasks": total_tasks,
                    "passed_tasks": passed_count,
                    "pass_rate": pass_rate,
                    "train_loss": train_loss,
                }).encode("utf-8"),
            )
            append_record(self.hk_path, eval_record)

        return GenerationReport(
            generation=self.current_generation,
            train_loss=train_loss,
            pass_rate=pass_rate,
            expansion_occurred=self.expansion_applied,
            expansion_details="Model capacity active",
            evaluated_tasks=total_tasks,
            successful_dialogues=passed_count,
        )
