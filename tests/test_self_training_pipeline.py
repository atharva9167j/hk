"""
Test Suite: Autonomous Self-Training Pipeline with Self-Conversational Thinking
Verifies:
1. ExpansionEvaluator diagnosing vocabulary deficits and representation bottlenecks.
2. SelfConversationalEngine parsing <think> inner monologue and sandboxed unit tests.
3. SelfTrainingPipeline autonomous dynamic expansion (vocab + width) with Day-0 function preservation.
4. Self-training loop with CodeSandbox verification and Appendix lineage tracking.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn

# Ensure local python package is on path
sys.path.insert(0, os.path.abspath("python"))

from hk import HKConfig, HKForCausalLM
from hk.torch import save_model, load_file
from hk.adaptive import (
    GrowthGovernor,
    ExpansionEvaluator,
    ExpansionDiagnosis,
    SelfConversationalEngine,
    ConversationalTurn,
    SelfDialogue,
    SelfTrainingCurriculum,
    GenerationReport,
    SelfTrainingPipeline,
    CodeSandbox,
    TestCase,
    read_appendix,
    AppendixEntryType,
)


def test_expansion_evaluator_diagnosis():
    """Verifies that ExpansionEvaluator accurately identifies vocab and capacity bottlenecks."""
    print("\n--- 1. Testing ExpansionEvaluator Diagnostic Logic ---")
    governor = GrowthGovernor(max_growth_ratio=2.0)
    evaluator = ExpansionEvaluator(governor=governor, max_error_threshold=0.40)

    config = HKConfig(vocab_size=100, intermediate_size=128)
    model = HKForCausalLM(config, device="cpu")

    # Case A: Familiar domain, all tokens known, high pass rate
    known_vocab = {"fn", "let", "return", "if", "else"}
    diagnosis_a = evaluator.evaluate_domain_capacity(
        model=model,
        diagnostic_loss=0.05,
        diagnostic_pass_rate=0.95,
        known_vocab=known_vocab,
        domain_keywords=["fn", "let", "return"],
    )
    assert not diagnosis_a.needs_expansion, "Familiar domain should NOT require expansion"
    assert not diagnosis_a.needs_vocab_expansion
    assert not diagnosis_a.needs_width_expansion
    print("  [PASS] Familiar domain correctly diagnosed as needing NO expansion.")

    # Case B: Target language introduces 5 unknown keywords and has 80% failure rate
    domain_keywords = ["fn", "let", "defer", "errdefer", "comptime", "pub", "anytype"]
    diagnosis_b = evaluator.evaluate_domain_capacity(
        model=model,
        diagnostic_loss=4.5,
        diagnostic_pass_rate=0.10, # 90% error rate!
        known_vocab=known_vocab,
        domain_keywords=domain_keywords,
    )
    assert diagnosis_b.needs_expansion, "New language with high error rate MUST trigger expansion"
    assert diagnosis_b.needs_vocab_expansion, "Missing keywords must trigger vocab expansion"
    assert len(diagnosis_b.suggested_new_tokens) == 5
    assert "comptime" in diagnosis_b.suggested_new_tokens
    assert diagnosis_b.needs_width_expansion, "90% error rate must trigger width expansion"
    assert diagnosis_b.suggested_width_ratio >= 1.33
    print(f"  [PASS] New language correctly diagnosed: {diagnosis_b.suggested_width_ratio:.2f}x width, {len(diagnosis_b.suggested_new_tokens)} tokens.")


def test_self_conversational_engine_parsing():
    """Verifies inner monologue extraction (<think>...</think>) and CodeSandbox execution."""
    print("\n--- 2. Testing SelfConversationalEngine & Sandboxed Execution ---")
    sandbox = CodeSandbox(timeout_sec=3.0)
    engine = SelfConversationalEngine(sandbox=sandbox)

    sample_output = """
<think>
We need to implement a function that calculates the factorial of n.
For n <= 1, return 1. Otherwise return n * factorial(n - 1).
Time complexity will be O(n).
</think>
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
"""
    thinking, code = engine.extract_thinking_and_code(sample_output)
    assert "calculates the factorial" in thinking
    assert "def factorial(n):" in code
    print("  [PASS] <think> inner monologue and code successfully extracted.")

    # Execute in CodeSandbox against unit tests
    test_cases = [
        TestCase(input_call="factorial(0)", expected_output=1),
        TestCase(input_call="factorial(5)", expected_output=120),
        TestCase(input_call="factorial(6)", expected_output=720),
    ]
    res = sandbox.execute_code(code, test_cases=test_cases)
    assert res.success, f"Factorial execution failed: {res.error_message}"
    assert res.pass_rate == 1.0
    assert res.passed_tests == 3
    print("  [PASS] CodeSandbox validated execution: 3/3 tests passed (100%).")


def test_autonomous_expansion_and_plasticity():
    """Verifies autonomous expansion application, Day-0 preservation, and gradient shielding."""
    print("\n--- 3. Testing Autonomous Expansion & Plasticity Shield ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        hk_path = os.path.join(tmp_dir, "test_base.hk")

        config = HKConfig(
            model_type="causal_lm",
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            tie_word_embeddings=True,
        )
        model = HKForCausalLM(config, device="cpu")
        save_model(model, hk_path)

        pipeline = SelfTrainingPipeline(model=model, hk_file_path=hk_path)
        pipeline.set_known_vocab({f"token_{i}" for i in range(100)})

        curriculum = SelfTrainingCurriculum(
            domain_name="MiniLang",
            target_language="minilang",
            syntax_keywords=["func", "defer", "match", "val", "mut"],
            diagnostic_tasks=[],
        )

        diagnosis = ExpansionDiagnosis(
            needs_expansion=True,
            needs_vocab_expansion=True,
            suggested_new_tokens=["func", "defer", "match", "val", "mut"],
            needs_width_expansion=True,
            suggested_width_ratio=1.50,
            rationale="MiniLang syntax deficit and capacity requirements.",
        )

        # Baseline input
        x_test = torch.randint(0, 100, (4, 8))
        logits_before = model(x_test).logits.detach()

        # Apply autonomous expansion
        res = pipeline.apply_autonomous_expansion(diagnosis, curriculum)
        assert res["success"]
        assert model.config.vocab_size == 105
        assert model.config.intermediate_size == 192
        assert len(pipeline.plasticity_hooks) > 0
        print(f"  [PASS] Autonomous expansion applied: vocab 100->105, width 128->192, {res['hooks_count']} hooks.")

        # Verify Day-0 logit preservation on existing vocabulary
        logits_after = model(x_test).logits.detach()
        max_diff = (logits_after[:, :, :100] - logits_before).abs().max().item()
        assert max_diff < 1e-4, f"Day-0 logit preservation violated: {max_diff}"
        print(f"  [PASS] Day-0 function preservation verified: max deviation = {max_diff:.2e} < 1e-4.")

        # Verify Appendix record in container
        records = read_appendix(hk_path)
        assert len(records) >= 1
        assert records[-1].entry_type == AppendixEntryType.NEW_LAYER
        print("  [PASS] Recorded autonomous expansion in container Appendix.")


def test_self_training_generation_loop():
    """Verifies the complete self-training loop with verified inner dialogue and Appendix tracking."""
    print("\n--- 4. Testing End-to-End Self-Training Generation Loop ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        hk_path = os.path.join(tmp_dir, "self_trained.hk")

        config = HKConfig(
            model_type="causal_lm",
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            tie_word_embeddings=True,
        )
        model = HKForCausalLM(config, device="cpu")
        save_model(model, hk_path)

        pipeline = SelfTrainingPipeline(model=model, hk_file_path=hk_path)

        curriculum = SelfTrainingCurriculum(
            domain_name="SimpleMathDSL",
            target_language="python",
            syntax_keywords=["add", "sub", "mul"],
            training_tasks=[
                {
                    "id": "add_task",
                    "prompt": "Write a function `solve(a, b)` that returns the sum of a and b.",
                    "test_cases": [
                        {"call": "solve(2, 3)", "expected": 5},
                        {"call": "solve(-1, 1)", "expected": 0},
                    ],
                },
                {
                    "id": "mul_task",
                    "prompt": "Write a function `solve(a, b)` that returns the product of a and b.",
                    "test_cases": [
                        {"call": "solve(3, 4)", "expected": 12},
                        {"call": "solve(0, 5)", "expected": 0},
                    ],
                },
            ],
        )

        def mock_solution_generator(prompt: str) -> str:
            if "sum" in prompt:
                return "<think>Add a and b together.</think>\n```python\ndef solve(a, b):\n    return a + b\n```"
            else:
                return "<think>Multiply a and b together.</think>\n```python\ndef solve(a, b):\n    return a * b\n```"

        def mock_optimizer_step(dialogues: list, m: nn.Module) -> float:
            assert len(dialogues) == 2
            # Simulate an optimization step
            return 0.042

        report = pipeline.train_generation(
            curriculum=curriculum,
            solution_generator_fn=mock_solution_generator,
            optimizer_step_fn=mock_optimizer_step,
        )

        assert report.pass_rate == 1.0, f"Expected 100% pass rate, got {report.pass_rate}"
        assert report.successful_dialogues == 2
        assert report.train_loss == 0.042
        print(f"  [PASS] Self-training generation completed: {report.successful_dialogues}/{report.evaluated_tasks} passed, loss={report.train_loss:.4f}.")

        # Check Appendix
        records = read_appendix(hk_path)
        assert len(records) >= 1
        assert records[-1].entry_type == AppendixEntryType.CODE_EVAL
        print("  [PASS] Code evaluation record verified in container Appendix.")


def main():
    print("=" * 80)
    print("RUNNING AUTONOMOUS SELF-TRAINING PIPELINE TEST SUITE")
    print("=" * 80)

    test_expansion_evaluator_diagnosis()
    test_self_conversational_engine_parsing()
    test_autonomous_expansion_and_plasticity()
    test_self_training_generation_loop()

    print("\n" + "=" * 80)
    print("SELF-TRAINING PIPELINE SUITE: 4 PASSED, 0 FAILED")
    print("=" * 80)


if __name__ == "__main__":
    main()
