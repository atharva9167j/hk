"""
Tests for HK Raw Weight Storage Engine and Multi-Device Universal Hardware Alignment
"""

import os
import pytest
import numpy as np
import torch

import hk
from hk.raw import (
    HKRawWeightStore,
    save_raw,
    load_raw,
    save_sharded_raw,
    load_sharded_raw,
    to_amd_rocm,
    to_intel_npu,
    to_apple_metal,
    to_nvidia_tensor_core,
)
from hk.native import (
    native_detect_hardware,
    native_get_optimal_alignment,
    native_gemv_bf16,
    native_gemv_f16,
    native_gemv_int8,
)


def test_hardware_detection():
    hw = native_detect_hardware()
    assert isinstance(hw, dict)
    assert "vendor" in hw
    assert "optimal_page_alignment" in hw
    assert "dma_hugepage_alignment" in hw

    align = hw["optimal_page_alignment"]
    assert align >= 4096
    # Crucial guarantee: Universal page alignment MUST satisfy NVIDIA Tensor Core 128-byte coalescing
    assert (align % 128) == 0, f"Page alignment {align} must be divisible by 128 for Tensor Core compatibility"
    assert (hw["dma_hugepage_alignment"] % 128) == 0

    opt_align = native_get_optimal_alignment()
    assert opt_align == align


def test_save_and_load_raw_tensors(tmp_path):
    out_file = str(tmp_path / "test_raw_model.hk")

    w_f32 = torch.tensor([1.25, -2.5, 3.75, 4.0, -5.5], dtype=torch.float32)
    w_bf16 = torch.tensor([0.5, -1.0, 2.0, -0.25, 1.5], dtype=torch.bfloat16)
    w_f16 = torch.tensor([0.125, -0.25, 0.5, 1.0, -2.0], dtype=torch.float16)
    w_int8 = torch.tensor([10, -20, 30, -40, 50], dtype=torch.int8)
    w_int16 = torch.tensor([1000, -2000, 3000, -4000, 5000], dtype=torch.int16)
    w_int32 = torch.tensor([100000, -200000, 300000, -400000, 500000], dtype=torch.int32)
    w_int64 = torch.tensor([10000000000, -20000000000, 30000000000, -40000000000, 50000000000], dtype=torch.int64)
    w_uint8 = torch.tensor([0, 50, 100, 150, 255], dtype=torch.uint8)

    tensors = {
        "layer.f32": w_f32,
        "layer.bf16": w_bf16,
        "layer.f16": w_f16,
        "layer.int8": w_int8,
        "layer.int16": w_int16,
        "layer.int32": w_int32,
        "layer.int64": w_int64,
        "layer.uint8": w_uint8,
    }

    meta = {"architecture": "universal_raw_transformer", "version": "1.0"}
    save_raw(out_file, tensors, metadata=meta, alignment=4096)

    # Inspect via HKRawWeightStore
    with HKRawWeightStore(out_file) as store:
        assert store.is_raw_storage is True
        assert store.is_universal_page_aligned is True
        assert store.is_tensor_core_aligned is True
        assert store.alignment == 4096

        stored_meta = store.metadata()
        assert stored_meta.get("architecture") == "universal_raw_transformer"

        for name in tensors:
            assert name in store

        # Direct PyTorch tensor extraction
        assert torch.allclose(store["layer.f32"], w_f32)
        assert torch.allclose(store["layer.bf16"], w_bf16)
        assert torch.allclose(store["layer.f16"], w_f16)
        assert torch.equal(store["layer.int8"], w_int8)
        assert torch.equal(store["layer.int16"], w_int16)
        assert torch.equal(store["layer.int32"], w_int32)
        assert torch.equal(store["layer.int64"], w_int64)
        assert torch.equal(store["layer.uint8"], w_uint8)

    # Direct load_raw function
    loaded = load_raw(out_file, as_torch=True)
    assert torch.allclose(loaded["layer.f32"], w_f32)
    assert torch.allclose(loaded["layer.bf16"], w_bf16)
    assert torch.allclose(loaded["layer.f16"], w_f16)
    assert torch.equal(loaded["layer.int8"], w_int8)

    # Direct NumPy load
    loaded_np = load_raw(out_file, as_torch=False)
    assert np.allclose(loaded_np["layer.f32"], w_f32.numpy())
    assert np.array_equal(loaded_np["layer.int8"], w_int8.numpy())


def test_raw_sharding(tmp_path):
    base_file = str(tmp_path / "model_sharded.hk")

    shard0 = {
        "transformer.layer.0.weight": torch.randn(4, 4, dtype=torch.float32),
        "transformer.layer.1.weight": torch.randn(4, 4, dtype=torch.float32),
    }
    shard1 = {
        "transformer.layer.2.weight": torch.randn(4, 4, dtype=torch.float32),
        "transformer.layer.3.weight": torch.randn(4, 4, dtype=torch.float32),
    }

    shard_files = save_sharded_raw(base_file, [shard0, shard1], alignment=4096)
    assert len(shard_files) == 2

    # Check shard 0
    with HKRawWeightStore(shard_files[0]) as s0:
        assert s0.is_sharded is True
        assert s0.split_index == 0
        assert s0.split_count == 2
        assert s0.is_raw_storage is True
        assert s0.is_tensor_core_aligned is True

    # Check shard 1
    with HKRawWeightStore(shard_files[1]) as s1:
        assert s1.is_sharded is True
        assert s1.split_index == 1
        assert s1.split_count == 2
        assert s1.is_raw_storage is True
        assert s1.is_tensor_core_aligned is True

    # Load combined
    combined = load_sharded_raw(shard_files, as_torch=True)
    assert len(combined) == 4
    for k, v in shard0.items():
        assert torch.allclose(combined[k], v)
    for k, v in shard1.items():
        assert torch.allclose(combined[k], v)


def test_raw_gemv_computation(tmp_path):
    # 2x3 weight matrix
    # [ 1.0, 2.0, -1.0 ]
    # [ 0.5, -0.5, 2.0 ]
    w = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]], dtype=np.float32)
    x = np.array([2.0, 1.0, -3.0], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    expected_y = np.dot(w, x) + bias  # [ 2 + 2 + 3 + 0.5, 1 - 0.5 - 6 - 0.5 ] = [ 7.5, -6.0 ]

    # Test raw BF16 GEMV
    # Encode w to BF16 (uint16 representation)
    w_bf16_u16 = (w.view(np.uint32) >> 16).astype(np.uint16)
    y_bf16 = native_gemv_bf16(w_bf16_u16, x, bias=bias)
    assert np.allclose(expected_y, y_bf16, atol=1e-3)

    # Test raw FP16 GEMV
    w_f16 = w.astype(np.float16)
    y_f16 = native_gemv_f16(w_f16, x, bias=bias)
    assert np.allclose(expected_y, y_f16, atol=1e-3)

    # Test raw INT8 GEMV
    w_i8 = np.array([[10, 20, -10], [5, -5, 20]], dtype=np.int8)
    scale = 0.1
    # expected: (w_i8 * x) * 0.1 + bias
    expected_i8 = np.dot(w_i8, x) * scale + bias
    y_i8 = native_gemv_int8(w_i8, x, scale=scale, bias=bias)
    assert np.allclose(expected_i8, y_i8, atol=1e-4)

    # Test HKRawWeightStore.gemv
    hk_file = str(tmp_path / "raw_gemv_model.hk")
    save_raw(
        hk_file,
        {
            "linear.weight": torch.from_numpy(w).to(torch.bfloat16),
        },
        alignment=4096,
    )
    with HKRawWeightStore(hk_file) as store:
        y_store = store.gemv("linear.weight", x, bias=bias)
        assert np.allclose(expected_y, y_store.numpy() if isinstance(y_store, torch.Tensor) else y_store, atol=1e-2)


def test_device_optimizers():
    t_torch = torch.randn(16, 16)
    t_np = np.random.randn(16, 16).astype(np.float32)

    rocm_t = to_amd_rocm(t_torch)
    assert rocm_t.is_contiguous()
    rocm_np = to_amd_rocm(t_np)
    assert rocm_np.flags.c_contiguous

    npu_t = to_intel_npu(t_torch)
    assert npu_t.is_contiguous()

    metal_t = to_apple_metal(t_torch)
    assert metal_t.is_contiguous()

    tc_t = to_nvidia_tensor_core(t_torch)
    assert tc_t.is_contiguous()
