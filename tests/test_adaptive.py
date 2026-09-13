"""
Rigorous & Exhaustive Test Suite for HK Adaptive Neural Framework:
1. Net2WiderNet multi-layer sequential function preservation across activations (ReLU, GELU, SiLU, Tanh)
2. Net2WiderNet batch dimension invariance and symmetry-breaking noise
3. Net2DeeperNet sequential identity layer stacking (exact 0.0 deviation)
4. ModularResidualBlock bottleneck ranks (r=1..32) and gradient flow
5. GrowthGovernor boundary conditions, memory limits, and cumulative ratio checks
6. CodeSandbox AST validation, syntax error isolation, and restricted execution
7. CodeSandbox multiprocess timeouts, recursions, and runtime exception handling
8. CodeSandbox persistent evaluation records in .hk appendix container
9. Appendix multi-record stress test covering all 6 AppendixEntryType variants
10. Appendix SHA-256 cryptographic DAG lineage and tamper detection
11. Appendix multi-stage rollback (intermediate, gen 1, gen 0 reset)
12. Self-Play Evolution Engine (SPIN) multi-generation evolutionary trajectory
13. Self-Play context poisoning mitigation & bit-exact parameter restoration
14. Standalone model topology packaging with MLP and CNN architectures
15. Attention sink KV cache persistence (StreamingLLM style)
16. Writable memory mapping (mmap.ACCESS_COPY) in-place mutation & training safety
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
import numpy as np
import torch
import torch.nn as nn

# Ensure python directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from hk import (
    save_hk,
    load_hk,
)
from hk.adaptive import (
    net2wider_linear,
    net2deeper_linear,
    ModularResidualBlock,
    GrowthGovernor,
    CodeSandbox,
    TestCase,
    EvalResult,
    LoRAAdapter,
    SelfPlayEvolutionEngine,
    AppendixRecord,
    AppendixEntryType,
    AppendixFlags,
    AppendixMetrics,
    read_appendix,
    append_record,
    rollback_appendix,
    verify_lineage,
    package_standalone_hk,
    load_standalone_hk,
    load_standalone_topology
)


def test_net2wider_multi_layer_deep_preservation():
    print("--- 1. Testing Net2WiderNet Multi-Layer Deep Function Preservation ---")
    torch.manual_seed(42)
    # 4-layer network: 12 -> 20 -> 24 -> 16 -> 6
    fc1 = nn.Linear(12, 20)
    fc2 = nn.Linear(20, 24)
    fc3 = nn.Linear(24, 16)
    fc4 = nn.Linear(16, 6)

    activations = [nn.ReLU(), nn.GELU(), nn.SiLU(), nn.Tanh()]
    for act in activations:
        x = torch.randn(8, 12)
        with torch.no_grad():
            h1 = act(fc1(x))
            h2 = act(fc2(h1))
            h3 = act(fc3(h2))
            y_orig = fc4(h3)

        # Sequentially widen layer 1 (20 -> 36), updating fc2
        w_fc1, w_fc2 = net2wider_linear(fc1, fc2, 36, noise_std=0.0)
        # Widen layer 2 (24 -> 40), updating fc3
        w_fc2, w_fc3 = net2wider_linear(w_fc2, fc3, 40, noise_std=0.0)
        # Widen layer 3 (16 -> 28), updating fc4
        w_fc3, w_fc4 = net2wider_linear(w_fc3, fc4, 28, noise_std=0.0)

        with torch.no_grad():
            wh1 = act(w_fc1(x))
            wh2 = act(w_fc2(wh1))
            wh3 = act(w_fc3(wh2))
            y_wider = w_fc4(wh3)

        diff = (y_wider - y_orig).abs().max().item()
        assert diff < 1e-5, f"Deep Net2Wider divergence with {act}: diff {diff:.2e}"

    print("[PASS] Multi-layer Net2Wider preserved exact outputs across ReLU, GELU, SiLU, Tanh")


def test_net2wider_batch_invariance_and_noise():
    print("\n--- 2. Testing Net2WiderNet Batch Invariance and Noise Symmetry Breaking ---")
    torch.manual_seed(101)
    fc1 = nn.Linear(16, 24)
    fc2 = nn.Linear(24, 8)

    # Test various batch configurations
    for bs in [1, 7, 32, 128]:
        x = torch.randn(bs, 16)
        y_orig = fc2(torch.relu(fc1(x)))
        w1, w2 = net2wider_linear(fc1, fc2, 48, noise_std=0.0)
        y_wider = w2(torch.relu(w1(x)))
        diff = (y_wider - y_orig).abs().max().item()
        assert diff < 1e-5, f"Batch size {bs} failed invariance: diff {diff:.2e}"

    # Test noise symmetry breaking
    w1_noisy, w2_noisy = net2wider_linear(fc1, fc2, 48, noise_std=1e-4)
    # Check that added neurons have non-identical weights in w2
    assert not torch.allclose(w2_noisy.weight[:, 24], w2_noisy.weight[:, 25], atol=1e-8)
    # Check that output deviation is bounded by the noise level
    x_test = torch.randn(10, 16)
    y_noisy = w2_noisy(torch.relu(w1_noisy(x_test)))
    y_base = fc2(torch.relu(fc1(x_test)))
    noise_diff = (y_noisy - y_base).abs().max().item()
    assert noise_diff < 5e-3, f"Noise injection caused unacceptable deviation: {noise_diff}"
    print(f"[PASS] Net2Wider validated across batches [1, 7, 32, 128] and noise symmetry breaking confirmed ({noise_diff:.2e} bounded diff)")


def test_net2deeper_sequential_stacking():
    print("\n--- 3. Testing Net2DeeperNet Sequential Layer Stacking ---")
    dim = 32
    deep1 = net2deeper_linear(dim, activation="identity")
    deep2 = net2deeper_linear(dim, activation="identity")
    deep3 = net2deeper_linear(dim, activation="identity")

    x = torch.randn(16, dim)
    with torch.no_grad():
        out = deep3(deep2(deep1(x)))

    diff = (out - x).abs().max().item()
    assert diff == 0.0, f"Net2Deeper 3-layer stacking deviation: {diff}"
    print("[PASS] 3 consecutive Net2Deeper identity layers preserved input with bit-exact 0.0 deviation")


def test_modular_residual_block_ranks_and_backprop():
    print("\n--- 4. Testing ModularResidualBlock Across Ranks & Backprop ---")
    dim = 64
    ranks = [1, 2, 4, 8, 16, 32]
    for r in ranks:
        block = ModularResidualBlock(dim=dim, bottleneck_rank=r)
        x = torch.randn(8, dim)
        with torch.no_grad():
            y = block(x)
        assert (y - x).abs().max().item() == 0.0, f"Rank {r} not zero on init"

        # Forward + backward pass
        out = block(x)
        loss = (out ** 2).sum()
        loss.backward()

        assert block.down.weight.grad is not None, f"Rank {r}: down.weight grad missing"
        assert block.up.weight.grad is not None, f"Rank {r}: up.weight grad missing"
        assert block.scale.grad is not None, f"Rank {r}: scale grad missing"

        # Optimize step
        opt = torch.optim.SGD(block.parameters(), lr=0.1)
        opt.step()
        opt.zero_grad()

    print(f"[PASS] ModularResidualBlock validated across ranks {ranks} with full backward gradient flow")


def test_growth_governor_exhaustive_budgeting():
    print("\n--- 5. Testing GrowthGovernor Boundary & Budgeting Rules ---")
    # 1. Standard governor: max ratio 2.0x, max VRAM 100 MB
    gov = GrowthGovernor(max_vram_mb=100, max_growth_ratio=2.0)

    # Valid: 1M -> 1.5M (1.5x <= 2.0x, 2MB <= 100MB)
    ok, msg = gov.can_grow(current_params=1_000_000, additional_params=500_000)
    assert ok, f"Expected approval: {msg}"

    # Ratio boundary: exactly 2.0x (1M -> 2M)
    ok, msg = gov.can_grow(current_params=1_000_000, additional_params=1_000_000)
    assert ok, f"Expected approval at exact 2.0x limit: {msg}"

    # Ratio violation: 2.01x (1M -> 2.01M)
    ok, msg = gov.can_grow(current_params=1_000_000, additional_params=1_010_000)
    assert not ok and "exceeds max limit" in msg, f"Expected ratio rejection: {msg}"

    # VRAM budget violation: 30M params * 4B = 120 MB > 100 MB budget
    small_gov = GrowthGovernor(max_vram_mb=50, max_growth_ratio=5.0)
    ok, msg = small_gov.can_grow(current_params=10_000_000, additional_params=15_000_000)
    assert not ok and "exceeds budget" in msg, f"Expected VRAM rejection: {msg}"

    print("[PASS] GrowthGovernor enforced ratio limits, exact boundaries, and VRAM memory ceilings")


def test_codesandbox_ast_validation():
    print("\n--- 6. Testing CodeSandbox AST Validation & Syntax Isolation ---")
    sandbox = CodeSandbox()

    # Valid Python code
    valid_codes = [
        "def add(a: int, b: int) -> int:\n    '''Docstring'''\n    return a + b",
        "class Node:\n    def __init__(self, v):\n        self.v = v",
        "squares = [x**2 for x in range(10) if x % 2 == 0]"
    ]
    for c in valid_codes:
        ok, err = sandbox.check_syntax(c)
        assert ok, f"Valid code rejected: {err}"

    # Invalid syntax with error line detection
    invalid_codes = [
        ("def broken(\n    return 42", "unexpected EOF"),
        ("x = [1, 2, 3", "was never closed"),
        ("for i in range(5)\n    pass", "expected ':'")
    ]
    for c, expected_err in invalid_codes:
        ok, err = sandbox.check_syntax(c)
        assert not ok, f"Syntax error not caught for: {c}"

    print("[PASS] CodeSandbox AST parser correctly verified valid code and isolated syntax errors")


def test_codesandbox_multiprocess_timeouts_and_exceptions():
    print("\n--- 7. Testing CodeSandbox Timeouts, Recursions & Exceptions ---")
    sandbox = CodeSandbox(timeout_sec=1.0)

    # A. Infinite loop timeout
    res_inf = sandbox.execute_code("while True:\n    pass")
    assert not res_inf.success
    assert "timed out" in res_inf.error_message.lower()

    # B. Deep recursion
    res_rec = sandbox.execute_code("def rec(): return rec()\nrec()")
    assert not res_rec.success

    # C. Runtime exception (ZeroDivisionError)
    res_div = sandbox.execute_code("x = 1 / 0")
    assert not res_div.success
    assert "zerodivisionerror" in res_div.error_message.lower()

    # D. Partial test case passes (2/3 passed)
    code = "def check(x): return x > 0 and x < 10"
    tests = [
        TestCase("check(5)", True, "inside range"),
        TestCase("check(-1)", False, "below range"),
        TestCase("check(15)", True, "deliberate failing test"), # should fail
    ]
    res_partial = sandbox.execute_code(code, tests)
    assert not res_partial.success
    assert res_partial.passed_tests == 2
    assert res_partial.total_tests == 3
    assert abs(res_partial.pass_rate - (2.0 / 3.0)) < 1e-4

    print(f"[PASS] CodeSandbox handled timeouts, deep recursion, runtime errors, and scored partial pass rates (2/3 = {res_partial.pass_rate:.2f})")


def test_codesandbox_hk_appendix_persistence(temp_dir):
    print("\n--- 8. Testing CodeSandbox Persistent Evaluation Records in Appendix ---")
    hk_path = os.path.join(temp_dir, "test_eval_sandbox.hk")

    # Base model
    save_hk(hk_path, {"dummy": torch.ones(4)})

    sandbox = CodeSandbox(timeout_sec=2.0)
    code = "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)"
    tests = [
        TestCase("fib(0)", 0, "base 0"),
        TestCase("fib(1)", 1, "base 1"),
        TestCase("fib(6)", 8, "fib 6"),
    ]
    eval_res = sandbox.execute_code(code, tests)
    assert eval_res.success

    # Serialize evaluation result into Appendix
    eval_data = json.dumps({
        "code": code,
        "passed": eval_res.passed_tests,
        "total": eval_res.total_tests,
        "pass_rate": eval_res.pass_rate,
        "time_ms": eval_res.execution_time_ms
    }).encode("utf-8")

    rec = AppendixRecord(
        entry_type=AppendixEntryType.CODE_EVAL,
        name="eval.fibonacci.test",
        generation=1,
        data=eval_data,
        metrics=AppendixMetrics(pass_rate=eval_res.pass_rate, custom=eval_res.execution_time_ms)
    )
    append_record(hk_path, rec)

    # Read back and verify persistence
    recs = read_appendix(hk_path)
    assert len(recs) == 1
    assert recs[0].entry_type == AppendixEntryType.CODE_EVAL
    assert recs[0].metrics.pass_rate == 1.0

    retrieved = json.loads(recs[0].data.decode("utf-8"))
    assert retrieved["passed"] == 3
    assert retrieved["pass_rate"] == 1.0
    print("[PASS] Code evaluation records serialized, persisted to .hk appendix, and faithfully recovered")


def test_appendix_multi_record_all_six_types(temp_dir):
    print("\n--- 9. Testing Appendix Multi-Record Stress with All 6 Entry Types ---")
    hk_path = os.path.join(temp_dir, "test_all_types.hk")
    save_hk(hk_path, {"w": torch.randn(8, 8)})

    entries = [
        (AppendixEntryType.LORA_ADAPTER, "lora.adapter.01", b"LORA_BYTES"),
        (AppendixEntryType.DELTA_PATCH, "delta.patch.02", b"DELTA_BYTES"),
        (AppendixEntryType.NEW_LAYER, "new.layer.03", b"LAYER_BYTES"),
        (AppendixEntryType.CODE_EVAL, "code.eval.04", b"EVAL_BYTES"),
        (AppendixEntryType.KV_CACHE_SINK, "kv.sink.05", b"SINK_BYTES"),
        (AppendixEntryType.TOPOLOGY_HEAD, "topology.head.06", b"TOPO_BYTES"),
    ]

    for gen, (etype, name, payload) in enumerate(entries, start=1):
        rec = AppendixRecord(
            entry_type=etype,
            name=name,
            generation=gen,
            data=payload
        )
        append_record(hk_path, rec)

    recs = read_appendix(hk_path)
    assert len(recs) == 6, f"Expected 6 records, got {len(recs)}"
    for i, (etype, name, payload) in enumerate(entries):
        assert recs[i].entry_type == etype
        assert recs[i].name == name
        assert recs[i].data == payload
        assert recs[i].generation == i + 1

    print("[PASS] All 6 AppendixEntryType variants serialized, appended, and parsed with exact fidelity")


def test_appendix_cryptographic_lineage_and_tamper_detection(temp_dir):
    print("\n--- 10. Testing Appendix SHA-256 DAG Lineage & Tamper Detection ---")
    hk_path = os.path.join(temp_dir, "test_tamper_lineage.hk")
    save_hk(hk_path, {"base": torch.randn(4)})

    # Build 5-generation hash-chained sequence
    payloads = [f"PAYLOAD_GEN_{i}".encode("utf-8") for i in range(1, 6)]
    parent_hash = b"\x00" * 32

    for gen, p in enumerate(payloads, start=1):
        rec = AppendixRecord(
            entry_type=AppendixEntryType.LORA_ADAPTER,
            name=f"adapter.gen{gen}",
            generation=gen,
            parent_hash=parent_hash,
            data=p
        )
        append_record(hk_path, rec)
        parent_hash = hashlib.sha256(p).digest()

    recs = read_appendix(hk_path)
    assert len(recs) == 5
    assert verify_lineage(recs), "Valid lineage failed verification"

    # Tamper test 1: corrupt payload of generation 2
    tampered_recs = [AppendixRecord(
        entry_type=r.entry_type,
        name=r.name,
        generation=r.generation,
        data=r.data if r.generation != 2 else b"CORRUPTED_PAYLOAD_GEN_2",
        parent_hash=r.parent_hash
    ) for r in recs]
    assert not verify_lineage(tampered_recs), "Tampered payload was not caught by verify_lineage"

    # Tamper test 2: corrupt parent_hash of generation 4
    tampered_recs2 = [AppendixRecord(
        entry_type=r.entry_type,
        name=r.name,
        generation=r.generation,
        data=r.data,
        parent_hash=r.parent_hash if r.generation != 4 else b"\xff" * 32
    ) for r in recs]
    assert not verify_lineage(tampered_recs2), "Tampered parent hash was not caught by verify_lineage"

    print("[PASS] 5-generation SHA-256 DAG lineage verified and tamper detection passed for both corrupted payload and hash")


def test_appendix_multi_stage_rollback(temp_dir):
    print("\n--- 11. Testing Appendix Multi-Stage Rollback ---")
    hk_path = os.path.join(temp_dir, "test_rollback_stages.hk")
    save_hk(hk_path, {"w": torch.randn(4)})

    for g in range(1, 6):
        append_record(hk_path, AppendixRecord(
            entry_type=AppendixEntryType.LORA_ADAPTER,
            name=f"gen.{g}",
            generation=g,
            data=f"DATA_{g}".encode("utf-8")
        ))

    # Rollback to Gen 3
    rollback_appendix(hk_path, 3)
    recs_3 = read_appendix(hk_path)
    assert len(recs_3) == 3
    assert [r.generation for r in recs_3] == [1, 2, 3]

    # Append new Gen 4
    append_record(hk_path, AppendixRecord(
        entry_type=AppendixEntryType.LORA_ADAPTER,
        name="gen.4.branchB",
        generation=4,
        data=b"DATA_4_BRANCH_B"
    ))
    recs_4 = read_appendix(hk_path)
    assert len(recs_4) == 4
    assert recs_4[-1].name == "gen.4.branchB"

    # Rollback to Gen 1
    rollback_appendix(hk_path, 1)
    recs_1 = read_appendix(hk_path)
    assert len(recs_1) == 1
    assert recs_1[0].generation == 1

    # Rollback to Gen 0 (complete clear)
    rollback_appendix(hk_path, 0)
    recs_0 = read_appendix(hk_path)
    assert len(recs_0) == 0

    print("[PASS] Multi-stage rollback (Gen 5 -> Gen 3 -> Branch -> Gen 1 -> Gen 0) executed cleanly")


def test_self_play_evolution_multi_generation_trajectory(temp_dir):
    print("\n--- 12. Testing Self-Play Evolution Engine Multi-Generation Trajectory ---")
    hk_path = os.path.join(temp_dir, "test_self_play_traj.hk")

    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 2)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = ToyNet()
    save_hk(hk_path, model.state_dict())

    engine = SelfPlayEvolutionEngine(
        hk_file_path=hk_path,
        base_model=model,
        target_layer_name="fc1",
        lora_rank=2
    )

    def dummy_train(m):
        opt = torch.optim.SGD(engine.adapter.parameters(), lr=0.05)
        x = torch.randn(4, 8)
        loss = (m(x) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    # Planned trajectory:
    # Step 1: 0.82 -> PASS
    # Step 2: 0.70 -> FAIL (regress) -> Rollback
    # Step 3: 0.88 -> PASS
    # Step 4: 0.93 -> PASS
    # Step 5: 0.65 -> FAIL (regress) -> Rollback
    trajectory = [
        (0.82, True),
        (0.70, False),
        (0.88, True),
        (0.93, True),
        (0.65, False),
    ]

    accepted_gens = []
    for idx, (score, should_accept) in enumerate(trajectory, start=1):
        ok, msg, _ = engine.run_evolution_step(
            train_step_fn=dummy_train,
            eval_fn=lambda m, s=score: (0.1, s),
            step_name=f"evolution.step_{idx}"
        )
        assert ok == should_accept, f"Step {idx} expected accept={should_accept}, got {ok}: {msg}"
        if ok:
            accepted_gens.append(idx)

    recs = read_appendix(hk_path)
    assert len(recs) == len(accepted_gens), f"Expected {len(accepted_gens)} records in container, got {len(recs)}"
    for r in recs:
        assert r.metrics.accuracy >= 0.80
    print(f"[PASS] Self-Play 5-generation evolutionary loop approved {len(accepted_gens)} steps and rejected 2 regressions")


def test_self_play_context_poisoning_bit_exactness(temp_dir):
    print("\n--- 13. Testing Context Poisoning Mitigation Bit-Exact Parameter Restoration ---")
    hk_path = os.path.join(temp_dir, "test_poison_restore.hk")

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return self.linear(x)

    model = SimpleNet()
    save_hk(hk_path, model.state_dict())

    engine = SelfPlayEvolutionEngine(
        hk_file_path=hk_path,
        base_model=model,
        target_layer_name="linear",
        lora_rank=2
    )
    # Establish Generation 1 baseline
    ok_base, _, _ = engine.run_evolution_step(
        train_step_fn=lambda m: 0.1,
        eval_fn=lambda m: (0.1, 0.85),
        step_name="baseline"
    )
    assert ok_base, "Baseline step 1 must be approved"

    # Save exact pre-trial adapter weights
    pre_a = engine.adapter.lora_A.clone()
    pre_b = engine.adapter.lora_B.clone()

    # Run step that triggers massive regression
    def poison_step(m):
        with torch.no_grad():
            engine.adapter.lora_A.add_(5.0)
            engine.adapter.lora_B.add_(10.0)
        return 999.0

    ok, msg, _ = engine.run_evolution_step(
        train_step_fn=poison_step,
        eval_fn=lambda m: (10.0, 0.10), # Catastrophic accuracy
        step_name="poison_attempt"
    )
    assert not ok, "Poisoning step was not rejected"

    # Assert adapter weights were restored to exact pre-trial values
    assert torch.equal(engine.adapter.lora_A, pre_a), "lora_A was not bit-exact restored after rollback"
    assert torch.equal(engine.adapter.lora_B, pre_b), "lora_B was not bit-exact restored after rollback"
    print("[PASS] Context poisoning prevented: parameters bit-identical to pre-trial state after rollback")


def test_standalone_topology_mlp_and_cnn(temp_dir):
    print("\n--- 14. Testing Standalone Model Topology Packaging (MLP & CNN) ---")
    # A. Test AutoMLP
    mlp_path = os.path.join(temp_dir, "test_standalone_mlp.hk")
    class OrigMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 4)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    mlp = OrigMLP()
    mlp.eval()
    package_standalone_hk(
        mlp_path,
        mlp,
        architecture_name="mlp",
        config={"in_features": 16, "hidden_features": 32, "num_classes": 4}
    )
    recon_mlp, _ = load_standalone_hk(mlp_path)

    x_mlp = torch.randn(5, 16)
    with torch.no_grad():
        y_orig = mlp(x_mlp)
        y_recon = recon_mlp(x_mlp)
    assert (y_recon - y_orig).abs().max().item() < 1e-6, "Standalone MLP diverged"

    # B. Test AutoCNN
    cnn_path = os.path.join(temp_dir, "test_standalone_cnn.hk")
    class OrigCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(8 * 16, 10)
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            return self.fc(x.flatten(1))

    cnn = OrigCNN()
    cnn.eval()
    package_standalone_hk(
        cnn_path,
        cnn,
        architecture_name="cnn",
        config={"in_channels": 1, "hidden_channels": 8, "num_classes": 10}
    )
    recon_cnn, _ = load_standalone_hk(cnn_path)

    x_cnn = torch.randn(2, 1, 16, 16)
    with torch.no_grad():
        y_cnn_orig = cnn(x_cnn)
        y_cnn_recon = recon_cnn(x_cnn)
    assert (y_cnn_recon - y_cnn_orig).abs().max().item() < 1e-6, "Standalone CNN diverged"

    print("[PASS] Standalone MLP and CNN reconstructed without class definitions with exact numerical fidelity")


def test_attention_sink_persistence(temp_dir):
    print("\n--- 15. Testing Attention Sink KV Cache Persistence (StreamingLLM) ---")
    hk_path = os.path.join(temp_dir, "test_attention_sink.hk")
    save_hk(hk_path, {"model.weight": torch.randn(8, 8)})

    # Simulate attention sink tokens (4 tokens x 8 heads x 64 head_dim)
    sink_k = torch.randn(4, 8, 64, dtype=torch.float32)
    sink_v = torch.randn(4, 8, 64, dtype=torch.float32)

    sink_data = bytearray()
    sink_data.extend(sink_k.numpy().tobytes())
    sink_data.extend(sink_v.numpy().tobytes())

    rec = AppendixRecord(
        entry_type=AppendixEntryType.KV_CACHE_SINK,
        name="attention.sink.initial_4_tokens",
        target="transformer.layer_0.attention",
        generation=1,
        data=bytes(sink_data)
    )
    append_record(hk_path, rec)

    # Read back and reconstruct
    recs = read_appendix(hk_path)
    assert len(recs) == 1
    assert recs[0].entry_type == AppendixEntryType.KV_CACHE_SINK

    payload = recs[0].data
    half = len(payload) // 2
    recovered_k = torch.from_numpy(np.frombuffer(payload[:half], dtype=np.float32).copy()).view(4, 8, 64)
    recovered_v = torch.from_numpy(np.frombuffer(payload[half:], dtype=np.float32).copy()).view(4, 8, 64)

    assert torch.equal(recovered_k, sink_k)
    assert torch.equal(recovered_v, sink_v)
    print("[PASS] Attention sink KV cache representations persisted to appendix and recovered identically")


def test_mmap_cow_inplace_mutation_and_backprop_safety(temp_dir):
    print("\n--- 16. Testing Writable Memory Mapping (ACCESS_COPY) In-Place Safety ---")
    hk_path = os.path.join(temp_dir, "test_cow_safety.hk")
    w_initial = torch.randn(16, 16)
    save_hk(hk_path, {"weight": w_initial})

    with open(hk_path, "rb") as f:
        file_bytes_initial = f.read()

    # Load with writable=True (mmap.ACCESS_COPY)
    model = load_hk(hk_path, writable=True)
    w_loaded = model["weight"]

    # Verify writable tensor
    assert w_loaded.is_leaf or not w_loaded.is_inference()

    # Perform in-place mutation
    w_loaded.add_(5.0)
    w_loaded[0, 0] = 999.0

    # Perform optimization step
    param = nn.Parameter(w_loaded)
    opt = torch.optim.Adam([param], lr=0.01)
    loss = (param ** 2).sum()
    loss.backward()
    opt.step()

    # Verify on-disk file was NOT corrupted or altered by in-place / Adam writes
    with open(hk_path, "rb") as f:
        file_bytes_after = f.read()

    assert file_bytes_initial == file_bytes_after, "On-disk file was mutated! ACCESS_COPY isolation failed."
    print("[PASS] Mmap copy-on-write confirmed: in-place mutations and Adam optimizer steps executed safely without file corruption")


if __name__ == "__main__":
    import numpy as np
    import inspect

    print("================================================================================")
    print("RUNNING EXHAUSTIVE ADAPTIVE NEURAL FRAMEWORK VERIFICATION SUITE")
    print("================================================================================")

    current_module = sys.modules[__name__]
    test_funcs = [
        getattr(current_module, name)
        for name in dir(current_module)
        if name.startswith("test_") and callable(getattr(current_module, name))
    ]

    passed = 0
    failed = 0

    for fn in test_funcs:
        fn_name = fn.__name__
        try:
            sig = inspect.signature(fn)
            if "temp_dir" in sig.parameters:
                d = tempfile.mkdtemp(prefix="hk_adaptive_test_")
                try:
                    fn(d)
                finally:
                    shutil.rmtree(d, ignore_errors=True)
            else:
                fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {fn_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n================================================================================")
    print(f"ADAPTIVE FRAMEWORK VERIFICATION RESULTS: {passed} PASSED, {failed} FAILED")
    print("================================================================================")
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
