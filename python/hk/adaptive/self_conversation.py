"""
HK Adaptive Neural Framework: Self-Conversational Engine
Implements multi-role inner monologue and self-conversational thinking:
Proposer <-> Thinker (<think>...</think>) <-> Coder <-> Sandboxed Critic.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn

from .code_eval import CodeSandbox, TestCase, EvalResult


@dataclass
class ConversationalTurn:
    """A single turn in the self-conversational thinking trajectory."""
    role: str # "proposer", "thinker", "coder", "critic"
    content: str
    thinking: str = ""
    code: str = ""
    eval_result: Optional[EvalResult] = None


@dataclass
class SelfDialogue:
    """A complete self-conversational thinking trajectory."""
    task_id: str
    task_prompt: str
    target_language: str
    turns: List[ConversationalTurn] = field(default_factory=list)
    success: bool = False
    pass_rate: float = 0.0
    final_code: str = ""
    final_thinking: str = ""

    def full_transcript(self) -> str:
        lines = [f"=== Self-Conversation [Task: {self.task_id} | Lang: {self.target_language}] ==="]
        for t in self.turns:
            lines.append(f"[{t.role.upper()}]: {t.content}")
            if t.thinking:
                lines.append(f"  <think>\n  {t.thinking}\n  </think>")
            if t.code:
                lines.append(f"  <code>\n  {t.code}\n  </code>")
            if t.eval_result:
                status = "PASS" if t.eval_result.success else "FAIL"
                lines.append(f"  [EVAL {status}]: pass_rate={t.eval_result.pass_rate:.1%} | time={t.eval_result.execution_time_ms:.1f}ms")
        return "\n".join(lines)


class SelfConversationalEngine:
    """
    Drives autonomous self-dialogue and inner reasoning.
    Pairs cognitive self-reflection with empirical code execution via CodeSandbox.
    """

    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        max_reflection_steps: int = 2,
    ):
        self.sandbox = sandbox or CodeSandbox(timeout_sec=3.0)
        self.max_reflection_steps = max_reflection_steps

    def extract_thinking_and_code(self, response_text: str) -> Tuple[str, str]:
        """
        Parses structured inner reasoning (<think>...</think>) and executable code block.
        """
        thinking = ""
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()

        # Extract code from ```...``` or <code>...</code> or use response if raw code
        code = ""
        code_match = re.search(r"```(?:\w+)?\n?(.*?)```", response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            tag_match = re.search(r"<code>(.*?)</code>", response_text, re.DOTALL)
            if tag_match:
                code = tag_match.group(1).strip()
            elif not think_match:
                code = response_text.strip()

        return thinking, code

    def format_training_pair(
        self,
        task_prompt: str,
        thinking: str,
        code: str,
        target_language: str = "python",
    ) -> str:
        """
        Formats a verified self-conversational reasoning trace into a standardized training sequence.
        """
        return (
            f"<|im_start|>user\nSolve this programming problem in {target_language}:\n{task_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{thinking}\n</think>\n"
            f"```{target_language}\n{code}\n```<|im_end|>"
        )

    def conduct_self_dialogue(
        self,
        task_id: str,
        task_prompt: str,
        initial_solution_fn: Any,
        test_cases: Optional[List[TestCase]] = None,
        target_language: str = "python",
        reflector_fn: Optional[Any] = None,
    ) -> SelfDialogue:
        """
        Runs an autonomous self-conversational thinking session:
        1. Proposer frames the task.
        2. Thinker produces internal reasoning & initial code.
        3. Sandboxed Critic executes and tests code.
        4. If failing, Thinker reflects on error messages and produces refined solution.
        """
        dialogue = SelfDialogue(
            task_id=task_id,
            task_prompt=task_prompt,
            target_language=target_language,
        )

        # Turn 1: Proposer
        dialogue.turns.append(ConversationalTurn(
            role="proposer",
            content=f"Implement requirement in {target_language}: {task_prompt}",
        ))

        # Turn 2: Initial Thinker & Coder
        raw_output = initial_solution_fn(task_prompt)
        thinking, code = self.extract_thinking_and_code(raw_output)

        eval_res = self.sandbox.execute_code(code, test_cases=test_cases)
        dialogue.turns.append(ConversationalTurn(
            role="thinker",
            content="Generated initial solution.",
            thinking=thinking,
            code=code,
            eval_result=eval_res,
        ))

        # Reflective iterations if needed
        step = 0
        current_code = code
        current_thinking = thinking

        while not eval_res.success and step < self.max_reflection_steps and reflector_fn is not None:
            step += 1
            # Critic turn
            error_feedback = eval_res.error_message or eval_res.stderr
            critic_msg = (
                f"Execution failed on step {step}: {error_feedback}. "
                f"Passed {eval_res.passed_tests}/{eval_res.total_tests} tests."
            )
            dialogue.turns.append(ConversationalTurn(
                role="critic",
                content=critic_msg,
            ))

            # Thinker reflective turn
            reflection_prompt = (
                f"Task: {task_prompt}\n"
                f"Previous Code:\n{current_code}\n"
                f"Error Encountered:\n{error_feedback}\n"
                f"Reflect on the syntax and logic error, then write the corrected code."
            )
            refined_output = reflector_fn(reflection_prompt)
            current_thinking, current_code = self.extract_thinking_and_code(refined_output)

            eval_res = self.sandbox.execute_code(current_code, test_cases=test_cases)
            dialogue.turns.append(ConversationalTurn(
                role="thinker",
                content=f"Refinement iteration {step}.",
                thinking=current_thinking,
                code=current_code,
                eval_result=eval_res,
            ))

        dialogue.success = eval_res.success
        dialogue.pass_rate = eval_res.pass_rate
        dialogue.final_code = current_code
        dialogue.final_thinking = current_thinking

        return dialogue
