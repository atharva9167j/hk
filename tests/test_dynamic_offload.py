"""
Test Suite for Automated Dynamic GPU/CPU Offloading Engine & Framework Optimizations
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

from hk.config import HKConfig
from hk.modeling import HKForCausalLM, AutoModel
from hk.offload import (
    HardwareMemoryInspector,
    LayerMemoryEstimator,
    DynamicOffloadPlanner,
    AutoDeviceDispatcher,
    DynamicOOMGuard,
)
from hk.tokenizer import HKTokenizer
from hk.quantization import make_2_4_sparse
from hk.torch import save_file, load_file
from hk.pipeline import pipeline


def test_hardware_inspection():
    """Verify hardware discovery and available memory reporting."""
    devices = HardwareMemoryInspector.get_available_devices()
    assert "cpu" in devices
    assert len(devices) >= 1

    cpu_mem = HardwareMemoryInspector.get_device_memory("cpu")
    assert cpu_mem.device == "cpu"
    assert cpu_mem.total_bytes > 0
    assert cpu_mem.usable_bytes > 0
    assert cpu_mem.usable_bytes <= cpu_mem.free_bytes


def test_layer_memory_estimation():
    """Verify calculation of parameter footprint per sub-module."""
    cfg = HKConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        tie_word_embeddings=False,
    )
    footprints = LayerMemoryEstimator.estimate_module_footprints(cfg, torch_dtype=torch.float16)

    # FP16 = 2 bytes/element
    # embed: 1000 * 256 * 2 = 512,000 bytes
    assert footprints["embed_tokens"] == 1000 * 256 * 2
    # 4 layers
    for i in range(4):
        assert f"layers.{i}" in footprints
        assert footprints[f"layers.{i}"] > 0
    # norm
    assert footprints["norm"] == 256 * 2
    # lm_head
    assert footprints["lm_head"] == 256 * 1000 * 2


def test_dynamic_offload_planner_cpu_fallback():
    """Verify automated device map generation falls back cleanly to CPU when no GPUs are available."""
    cfg = HKConfig(
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=2,
    )
    dmap = DynamicOffloadPlanner.create_device_map(cfg, force_cpu=True)
    assert dmap["embed_tokens"] == "cpu"
    for i in range(3):
        assert dmap[f"layers.{i}"] == "cpu"
    assert dmap["norm"] == "cpu"
    assert dmap["lm_head"] == "cpu"


def test_fp16_bf16_causal_mask_parity():
    """Verify that HKForCausalLM forward pass succeeds without dtype mismatch in FP16 and BF16."""
    cfg = HKConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    # Test FP16
    m_fp16 = HKForCausalLM(cfg, device="cpu").to(torch.float16)
    x = torch.randint(0, 100, (2, 8))
    out_fp16 = m_fp16(x)
    assert out_fp16.logits.shape == (2, 8, 100)
    assert out_fp16.logits.dtype == torch.float16

    # Test BF16
    m_bf16 = HKForCausalLM(cfg, device="cpu").to(torch.bfloat16)
    out_bf16 = m_bf16(x)
    assert out_bf16.logits.shape == (2, 8, 100)
    assert out_bf16.logits.dtype == torch.bfloat16


def test_dynamic_device_cross_boundary_forward_and_generate():
    """Verify model forward pass and auto-regressive generation across multiple partitioned modules."""
    cfg = HKConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
    )
    model = HKForCausalLM(cfg, device="cpu")

    # Simulate multi-stage device placement on CPU
    device_map = {
        "embed_tokens": "cpu",
        "layers.0": "cpu",
        "layers.1": "cpu",
        "layers.2": "cpu",
        "layers.3": "cpu",
        "norm": "cpu",
        "lm_head": "cpu",
    }
    AutoDeviceDispatcher.apply_device_map(model, device_map)

    inp = torch.randint(0, 200, (1, 6))
    out = model(inp)
    assert out.logits.shape == (1, 6, 200)

    # Test generate
    gen_ids = model.generate(inp, max_new_tokens=5)
    assert gen_ids.shape == (1, 11)


def test_to_dynamic_offload_helper():
    """Verify model.to_dynamic_offload() runs without errors."""
    cfg = HKConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    model = HKForCausalLM(cfg, device="cpu")
    model.to_dynamic_offload()
    assert hasattr(model, "_hk_device_map")
    assert "embed_tokens" in model._hk_device_map


def test_zero_copy_serialization_roundtrip():
    """Verify zero-copy saving and loading preserves exact tensor contents and shapes."""
    tensors = {
        "weight_f32": torch.randn(16, 32, dtype=torch.float32),
        "weight_f16": torch.randn(8, 16, dtype=torch.float16),
        "bias": torch.randn(32, dtype=torch.float32),
        "scalar_val": torch.tensor(42.0, dtype=torch.float32),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_weights.hk")
        save_file(tensors, path)
        assert os.path.exists(path)

        loaded = load_file(path, device="cpu")
        assert len(loaded) == len(tensors)

        for k in tensors:
            assert k in loaded
            torch.testing.assert_close(tensors[k], loaded[k])


def test_tokenizer_bpe_cache_and_streaming():
    """Verify that BPE caching and stream tokenization match exact standard tokenization."""
    tok = HKTokenizer()
    text = "Antigravity dynamic automated offloading scales seamlessly across multi-GPU environments!"

    # Encode with caching
    ids1 = tok.encode(text)
    ids2 = tok.encode(text)
    assert ids1 == ids2
    assert len(tok._bpe_cache) > 0

    # Test streaming
    chunks = ["Antigravity dynamic ", "automated offloading ", "scales seamlessly across ", "multi-GPU environments!"]
    stream_ids = list(tok.encode_stream(chunks, add_special_tokens=False))
    standard_ids = tok.encode("".join(chunks), add_special_tokens=False)
    assert stream_ids == standard_ids


def test_2_4_sparse_scale_correction():
    """Verify that 2:4 structured sparsity with scale correction achieves 50% sparsity with high fidelity."""
    x = torch.randn(64, 128, dtype=torch.float32)
    sparse_x, ratio = make_2_4_sparse(x, scale_correction=True)

    assert ratio == 0.5
    # Check 50% elements zeroed out
    num_zeros = (sparse_x == 0).sum().item()
    assert num_zeros == x.numel() // 2

    # Check that energy correction maintained Frobenius norm closer to original
    norm_orig = torch.norm(x).item()
    norm_sparse = torch.norm(sparse_x).item()
    rel_diff = abs(norm_orig - norm_sparse) / norm_orig
    assert rel_diff < 0.15  # Energy conserved within 15% (uncorrected loses ~30-50% energy)


def test_pipeline_device_auto():
    """Verify pipeline instantiation with device='auto'."""
    pipe = pipeline("text-generation", model=None, device="auto")
    assert pipe is not None
    assert isinstance(pipe.model, HKForCausalLM)
    res = pipe("Hello", max_new_tokens=3)
    assert len(res) == 1
    assert "generated_text" in res[0]
