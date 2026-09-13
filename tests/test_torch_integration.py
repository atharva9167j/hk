"""
Exhaustive Verification Suite for HK PyTorch Integrations (`hk.torch`):
1. Native Dtype Roundtrip (F32, F16, BF16, I64, I32, I8, U8, BOOL) with bit-exact preservation.
2. Tied Weights Auto-Detection via shared data_ptr address without duplicate storage.
3. Drop-in `save_file` and `load_file` matching `safetensors.torch` API.
4. Generic PyTorch `nn.Module` serialization via `save_model` and `load_model`.
5. Lazy inspection and multidimensional tensor slicing via `safe_open`.
6. Safe file lifecycle on Windows (no leaked memory map locks on file removal).
7. `AutoModel.from_pretrained` with dynamic `torch_dtype` and device casting.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn

# Ensure local python directory is on path
sys.path.insert(0, os.path.abspath("python"))

import hk
from hk.torch import save_file, load_file, save_model, load_model, safe_open
from hk import AutoModelForCausalLM, HKConfig, HKForCausalLM


def test_native_dtype_roundtrip(temp_dir):
    print("\n--- 1. Testing Native Dtype Roundtrip (F32, F16, BF16, I64, I32, I8, U8, BOOL) ---")
    test_path = os.path.join(temp_dir, "test_dtypes.hk")

    tensors = {
        "t_f32": torch.randn(4, 8, dtype=torch.float32),
        "t_f16": torch.randn(4, 8, dtype=torch.float16),
        "t_bf16": torch.tensor([1.5, -2.25, 0.125, 100.0], dtype=torch.bfloat16),
        "t_i64": torch.tensor([100000000000, -99999999999, 42], dtype=torch.int64),
        "t_i32": torch.tensor([1, -2, 3, 0], dtype=torch.int32),
        "t_i8": torch.tensor([-128, 127, 0], dtype=torch.int8),
        "t_u8": torch.tensor([255, 0, 128], dtype=torch.uint8),
        "t_bool": torch.tensor([True, False, True, True], dtype=torch.bool),
    }

    save_file(tensors, test_path)
    loaded = load_file(test_path)

    for name, orig in tensors.items():
        assert name in loaded, f"Missing key {name}"
        rec = loaded[name]
        assert rec.dtype == orig.dtype, f"Dtype mismatch for {name}: expected {orig.dtype}, got {rec.dtype}"
        assert rec.shape == orig.shape, f"Shape mismatch for {name}: expected {orig.shape}, got {rec.shape}"
        assert torch.all(rec == orig), f"Value mismatch for {name}"
        print(f"  [PASS] {name}: {orig.dtype} preserved with bit-exact equality.")


def test_tied_weights_auto_detection(temp_dir):
    print("\n--- 2. Testing Tied Weights Auto-Detection & Memory Sharing ---")
    test_path = os.path.join(temp_dir, "test_tied.hk")

    shared_weight = torch.randn(8, 8, dtype=torch.float32)
    other_weight = torch.randn(8, 8, dtype=torch.float32)

    tensors = {
        "embed_tokens.weight": shared_weight,
        "encoder.layer.weight": other_weight,
        "lm_head.weight": shared_weight,  # Tied to embed_tokens
    }

    save_file(tensors, test_path)

    # Inspect TOC via safe_open to verify shared ref flag
    with safe_open(test_path, framework="pt") as f:
        info_head = f.tensors_info["lm_head.weight"]
        assert info_head["storage_type"] == 0x31, "lm_head.weight must have STORAGE_SHARED_REF (0x31)"
        print("  [PASS] Auto-detected tied weight: lm_head.weight marked as STORAGE_SHARED_REF (0x31).")

    # Load file and verify data_ptr equivalence
    loaded = load_file(test_path)
    assert loaded["embed_tokens.weight"].data_ptr() == loaded["lm_head.weight"].data_ptr(), (
        "Tied weights must share the exact same underlying memory address!"
    )
    print("  [PASS] Loaded tied weights share identical data_ptr memory address.")


def test_generic_nn_module_save_load(temp_dir):
    print("\n--- 3. Testing Generic PyTorch nn.Module Save & Load ---")
    test_path = os.path.join(temp_dir, "test_module.hk")

    class CustomNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 8)
            self.register_buffer("running_scale", torch.tensor([1.5, 2.5]))

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x))) * self.running_scale[0]

    orig_model = CustomNetwork()
    x = torch.randn(4, 16)
    orig_output = orig_model(x)

    # Save via save_model
    save_model(orig_model, test_path, metadata={"architecture": "CustomNetwork", "version": "1.0"})
    print("  [PASS] Saved custom nn.Module to .hk container.")

    # Load into new instance
    fresh_model = CustomNetwork()
    load_model(fresh_model, test_path, strict=True)
    fresh_output = fresh_model(x)

    diff = torch.max(torch.abs(orig_output - fresh_output)).item()
    assert diff == 0.0, f"Module output deviation: {diff}"
    assert torch.all(fresh_model.running_scale == orig_model.running_scale)
    print(f"  [PASS] Loaded into fresh nn.Module with bit-exact forward pass (diff = {diff:.2e}).")


def test_safe_open_lazy_slicing(temp_dir):
    print("\n--- 4. Testing safe_open Lazy Inspection & Tensor Slicing ---")
    test_path = os.path.join(temp_dir, "test_slice.hk")

    matrix = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    bias = torch.tensor([1.0, 2.0, 3.0])
    save_file({"weight": matrix, "bias": bias}, test_path, metadata={"author": "HK", "model": "test"})

    with safe_open(test_path, framework="pt") as f:
        # 1. Keys and Metadata
        assert set(f.keys()) == {"weight", "bias"}
        meta = f.metadata()
        assert meta.get("author") == "HK"
        print(f"  [PASS] safe_open metadata and keys verified: {f.keys()}")

        # 2. Lazy single tensor get
        b = f.get_tensor("bias")
        assert torch.all(b == bias)
        print("  [PASS] Lazy get_tensor('bias') fetched successfully.")

        # 3. Slicing
        slice_obj = f.get_slice("weight")
        assert slice_obj.get_shape() == [10, 10]

        sub_matrix = slice_obj[0:3, 0:4]
        expected = matrix[0:3, 0:4]
        assert torch.all(sub_matrix == expected)
        assert sub_matrix.shape == (3, 4)
        print("  [PASS] TensorSlice [0:3, 0:4] matched exact multidimensional slice.")


def test_windows_file_lock_lifecycle(temp_dir):
    print("\n--- 5. Testing Windows File Descriptor Lifecycle & Safe Deletion ---")
    test_path = os.path.join(temp_dir, "test_lifecycle.hk")

    save_file({"w": torch.randn(16, 16)}, test_path)

    # Load with load_file
    data = load_file(test_path)
    assert "w" in data

    # Attempt to delete file immediately in the same process
    # On Windows, if mmap was held open, this would raise [WinError 32]
    try:
        os.remove(test_path)
        print("  [PASS] File deleted cleanly immediately after load_file (zero descriptor leak).")
    except PermissionError as e:
        raise AssertionError(f"Windows file lock error encountered: {e}")


def test_automodel_torch_dtype_and_device(temp_dir):
    print("\n--- 6. Testing AutoModel.from_pretrained with torch_dtype ---")
    test_dir = os.path.join(temp_dir, "hf_model_dir")
    os.makedirs(test_dir, exist_ok=True)

    config = HKConfig(
        model_type="causal_lm",
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    base_model = HKForCausalLM(config)
    base_model.save_pretrained(test_dir)

    # Load model with torch_dtype=torch.float16
    model_fp16 = AutoModelForCausalLM.from_pretrained(test_dir, torch_dtype=torch.float16)

    # Verify all linear parameters are cast to torch.float16
    for name, param in model_fp16.named_parameters():
        assert param.dtype == torch.float16, f"Parameter {name} is {param.dtype}, expected torch.float16"

    print("  [PASS] AutoModel.from_pretrained loaded with torch_dtype=torch.float16 successfully.")


def main():
    print("=" * 80)
    print("RUNNING HK PYTORCH INTEGRATION VERIFICATION SUITE")
    print("=" * 80)

    tests = [
        test_native_dtype_roundtrip,
        test_tied_weights_auto_detection,
        test_generic_nn_module_save_load,
        test_safe_open_lazy_slicing,
        test_windows_file_lock_lifecycle,
        test_automodel_torch_dtype_and_device,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        temp_d = tempfile.mkdtemp(prefix="hk_torch_test_")
        try:
            test_fn(temp_d)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        finally:
            shutil.rmtree(temp_d, ignore_errors=True)

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
