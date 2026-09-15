# Autonomous Self-Training and Self-Play

In this guide, I explain HK's autonomous self-learning pipeline, which allows models to self-diagnose weaknesses, formulate reasoning traces, generate candidate solutions, and verify them inside a secure code sandbox.

---

## Overview

Traditional model improvement relies heavily on massive external synthetic data pipelines, human labelers, and manual prompt engineering.

I designed HK with a self-contained evolutionary loop for autonomous improvement:
1. Diagnose: Identifies representation deficits, syntax error patterns, or missing tokens.
2. Reflect: Uses dual-agent dialogue to generate structured inner monologue reasoning traces (`<think> ... </think>`).
3. Synthesize: Produces candidate implementations or problem solutions.
4. Verify: Executes code candidates inside an AST-validated, process-isolated sandbox.
5. Reinforce: Applies Self-Play Fine-Tuning (SPIN) loss, rewarding verifiably correct solutions and penalizing failed attempts.
6. Adapt: If the task complexity exceeds the model's current capacity, it triggers Net2Net widening under the supervision of the GrowthGovernor.

---

## The Four Stages of the Self-Training Loop

```
+-------------------------------------------------------------+
| 1. Bottleneck Diagnosis (ExpansionEvaluator)                |
|    Probes error rates and missing tokens across curriculum  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 2. Inner Monologue Reflection (SelfConversationalEngine)    |
|    Generates <think> reasoning traces before coding         |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 3. Sandboxed Execution (CodeSandbox)                        |
|    Validates code in isolated process with strict timeout   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 4. Self-Play Fine-Tuning (SPINLoss) & Appendix Recording    |
|    Reinforces passing solutions, writes audit record        |
+-------------------------------------------------------------+
```

---

## 1. Bottleneck Diagnosis (`ExpansionEvaluator`)

Before blindly adding parameters, the `ExpansionEvaluator` probes the model across a domain curriculum (such as systems programming or math). It measures:
- Exact-match pass rates
- Syntax compilation error distributions
- Missing vocabulary tokens (keywords split into multiple subword pieces)
- Gradient saturation on key projection layers

If the evaluator discovers that errors are due to insufficient intermediate capacity rather than training duration, it creates a recommended architectural growth plan.

---

## 2. Inner Monologue Reasoning (`<think>`)

When we solve difficult problems, we think through the steps before acting. I implemented this in HK via `SelfConversationalEngine`, using a dual-agent Proposer and Thinker dialogue:

```
User Prompt: "Write a function in Zig to reverse a slice in-place."

Generated Output:
<think>
To reverse a slice in-place in Zig:
1. We need two indices: start = 0, and end = slice.len - 1.
2. While start < end, swap slice[start] with slice[end].
3. Increment start, decrement end.
4. We must handle empty slices and single-element slices safely.
5. In Zig, std.mem.swap or a temp variable can be used.
</think>
pub fn reverse(comptime T: type, slice: []T) void {
    if (slice.len <= 1) return;
    var start: usize = 0;
    var end: usize = slice.len - 1;
    while (start < end) : ({ start += 1; end -= 1; }) {
        const tmp = slice[start];
        slice[start] = slice[end];
        slice[end] = tmp;
    }
}
```

The model explicitly learns to deliberate before emitting final code, improving logical consistency on complex tasks.

---

## 3. Secure Execution Sandbox (`CodeSandbox`)

Allowing an autonomous model to execute code it wrote requires strict security. The `CodeSandbox` provides multiple layers of defense:

1. AST Safety Validation: Pre-screens the abstract syntax tree to reject forbidden imports (e.g. `os`, `subprocess`, `socket`, `shutil` in Python) and dangerous builtins (`eval`, `exec`, `__import__`).
2. Subprocess Isolation: Executes candidate programs in separate child processes.
3. Strict Timeouts: Automatically terminates infinite loops after a user-defined threshold (e.g., 2.0 seconds).
4. Memory Caps: Limits the virtual memory address space of the sandboxed process to prevent memory exhaustion attacks.

Only candidate solutions that compile cleanly and pass all unit tests provide positive reinforcement gradients.

---

## 4. Self-Play Fine-Tuning (`SPINLoss`)

Rather than relying on static external rewards, `SPINLoss` (Self-Play Fine-Tuning) pits the current generation of the model against its past self:
- The current model acts as the player attempting to distinguish and improve upon candidate outputs generated by the previous model generation.
- The loss function mathematically drives the model to produce answers that are more coherent, verified, and complete than what it generated in the past iteration.
- This creates an iterative bootstrapping loop that increases reasoning depth over multiple generations.

---

## Python Example: Running a Self-Training Generation

```python
from hk import HKConfig, HKForCausalLM
from hk.adaptive import (
    SelfTrainingPipeline,
    SelfTrainingCurriculum,
    SelfConversationalEngine,
    CodeSandbox,
    GrowthGovernor,
)

# 1. Initialize model
config = HKConfig(vocab_size=32000, hidden_size=1024, num_hidden_layers=8)
model = HKForCausalLM(config)

# 2. Setup curriculum with diagnostic tasks
curriculum = SelfTrainingCurriculum(
    domain_name="Algorithms",
    target_language="Python",
    syntax_keywords=["def", "return", "yield", "class", "async"],
    diagnostic_tasks=[
        {"prompt": "Write a binary search function.", "tests": ["assert binary_search([1,2,3], 2) == 1"]},
    ],
    training_tasks=[
        {"prompt": "Write a function to check if a string is a palindrome.", "tests": ["assert is_palindrome('racecar') == True"]},
    ],
)

# 3. Setup sandbox and conversational engine
sandbox = CodeSandbox(timeout_sec=2.0)
engine = SelfConversationalEngine(sandbox=sandbox)

# 4. Initialize pipeline
pipeline = SelfTrainingPipeline(
    model=model,
    hk_file_path="checkpoints/autonomous_model.hk",
    governor=GrowthGovernor(max_vram_mb=8192),
    conversational_engine=engine,
    rehearsal_ratio=0.10,  # 10% old task rehearsal to prevent forgetting
)

# 5. Run a self-training generation
report = pipeline.train_generation(curriculum, generation_idx=1)

print(f"Generation 1 Complete.")
print(f"Pass Rate: {report.pass_rate * 100:.1f}%")
print(f"Capacity Expanded: {report.expansion_occurred}")
print(f"New Parameters Added: {report.new_parameters}")
```
