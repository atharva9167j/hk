"""
Complete End-to-End Demonstration of HK Neural Tensor Format (.hk) & Neural Pruning Suite:
1. Trains an HWR (Handwritten Recognition) Neural Network on GPU (or CPU fallback).
2. Evaluates Baseline parameters, disk size, inference latency, and validation accuracy.
3. Applies Magnitude-Based Unstructured Pruning (50% sparsity) + Fine-Tuning Recovery.
4. Applies Wanda (Weights and Activations: |W_ij| * ||X_j||_2) Activation-Aware Pruning.
5. Applies NVIDIA Ampere 2:4 Structured Sparsity with Physical Hardware 2:4 Packing in .hk.
6. Applies Block-Sparse (BSR) Pruning based on sub-block Frobenius norms.
7. Applies Structured L2-Norm Physical Pruning (physically shrinking matrix dimensions by 30%).
8. Exports models to HK format (.hk) with:
   - True 2:4 structured hardware byte packing (50% values + 2-bit metadata)
   - Dual-Mode 4-bit NF4 quantization + residual precision recovery
   - Dual-Mode 8-bit DQ8 quantization
   - BitNet b1.58 ternary (DQT) quantization
   - Zero-payload NULL_REF and SHARED_REF
9. Verifies and benchmarks via the native Zig HK CLI (hk.exe).
10. Reports full comparative metrics table.
"""

import os
import sys
import copy
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# Add local python directory to path
sys.path.insert(0, os.path.abspath("python"))

from hk.pruning import (
    prune_unstructured_magnitude,
    prune_wanda,
    prune_structured_2_4,
    prune_block_sparse,
    prune_structured_l2,
    fine_tune_recovery,
    LayerSparsitySchedule,
)
from hk.quantization import make_2_4_sparse
from hk.torch import save_hk, load_hk
from hk.benchmark import benchmark_model, compare_models


class HWRNet(nn.Module):
    """
    Handwriting Recognition Deep Neural Network
    Designed for digit and stroke pattern recognition.
    """
    def __init__(self, in_features=64, hidden_dim1=128, hidden_dim2=128, hidden_dim3=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.relu3 = nn.ReLU()

        self.classifier = nn.Linear(hidden_dim3, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        return self.classifier(x)


def main():
    print("=" * 80)
    print("HK (Neural Tensor Format) - Complete Pruning & Compression Suite")
    print("=" * 80)

    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Hardware Acceleration: {gpu_name} (CUDA {torch.version.cuda})")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not found, using CPU")

    # 2. Data Preparation
    print("\n[Data] Loading Handwritten Digit Recognition dataset (HWR)...")
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # Normalize to [0, 1]
    y = digits.target.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    sample_batch = torch.from_numpy(X_val[:32]).to(device)

    # 3. Model Training
    print("\n[Train] Training baseline HWRNet...")
    model = HWRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    model.train()
    for epoch in range(15):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    os.makedirs("models", exist_ok=True)
    base_pt_path = "models/hwr_baseline.pt"
    torch.save(model.state_dict(), base_pt_path)

    results = []

    # Benchmark 1: Baseline
    print("\n[Benchmark 1/7] Evaluating Baseline Model...")
    res_base = benchmark_model(
        "1. Baseline (Unpruned)",
        model,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=base_pt_path
    )
    results.append(res_base)
    print(f"   Accuracy: {res_base['accuracy_pct']:.2f}% | Latency (B=32): {res_base['latency_b32_ms']:.2f} ms")

    # 4. Magnitude-Based Unstructured Pruning + Recovery
    print("\n[Prune] Applying Magnitude-based Unstructured Pruning (50% target)...")
    schedule = LayerSparsitySchedule(default_ratio=0.50, skip_first=True, skip_last=True)
    model_unstructured = copy.deepcopy(model)
    masks_mag = prune_unstructured_magnitude(model_unstructured, schedule=schedule)

    model_recovered, _ = fine_tune_recovery(
        model_unstructured,
        train_loader,
        val_loader,
        criterion,
        device,
        epochs=3,
        lr=0.0005,
        masks=masks_mag
    )
    recovered_pt_path = "models/hwr_unstructured_recovered.pt"
    torch.save(model_recovered.state_dict(), recovered_pt_path)

    res_rec = benchmark_model(
        "2. Magnitude Pruned (50% Fine-tuned)",
        model_recovered,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=recovered_pt_path
    )
    results.append(res_rec)
    print(f"   Accuracy: {res_rec['accuracy_pct']:.2f}%")

    # 5. Wanda Pruning (Weights & Activations)
    print("\n[Prune] Applying Wanda (Weights & Activations) Pruning...")
    model_wanda = copy.deepcopy(model)
    masks_wanda = prune_wanda(
        model_wanda,
        train_loader,
        device,
        sparsity_ratio=0.50,
        schedule=schedule,
        num_calibration_batches=10
    )
    model_wanda, _ = fine_tune_recovery(
        model_wanda,
        train_loader,
        val_loader,
        criterion,
        device,
        epochs=2,
        lr=0.0003,
        masks=masks_wanda
    )
    wanda_pt_path = "models/hwr_wanda.pt"
    torch.save(model_wanda.state_dict(), wanda_pt_path)

    res_wanda = benchmark_model(
        "3. Wanda Pruned (50% W*X norm)",
        model_wanda,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=wanda_pt_path
    )
    results.append(res_wanda)
    print(f"   Accuracy: {res_wanda['accuracy_pct']:.2f}%")

    # 6. NVIDIA Ampere 2:4 Structured Sparsity
    print("\n[Prune] Applying NVIDIA Ampere 2:4 Structured Hardware Pruning...")
    model_24 = copy.deepcopy(model)
    masks_24 = prune_structured_2_4(model_24, skip_first=True, skip_last=True)
    model_24, _ = fine_tune_recovery(
        model_24,
        train_loader,
        val_loader,
        criterion,
        device,
        epochs=2,
        lr=0.0003,
        masks=masks_24
    )

    hk_sparse_path = "models/hwr_sparse_2_4.hk"
    save_hk(
        hk_sparse_path,
        model_24.state_dict(),
        metadata={
            "architecture": "HWRNet-Sparse-2-4",
            "framework": "HK Neural Tensor Format",
            "sparsity_mode": "Ampere 2:4 Structured Hardware",
        },
        auto_pack_sparse=True
    )
    print(f"   [OK] Saved packed 2:4 sparse container {hk_sparse_path} ({os.path.getsize(hk_sparse_path)/1024:.1f} KB)")

    loaded_24 = load_hk(hk_sparse_path, device=str(device))
    model_loaded_24 = HWRNet().to(device)
    model_loaded_24.load_state_dict(loaded_24.state_dict)

    res_24 = benchmark_model(
        "4. Ampere 2:4 Packed (.hk)",
        model_loaded_24,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=hk_sparse_path
    )
    results.append(res_24)
    print(f"   Accuracy: {res_24['accuracy_pct']:.2f}% | Latency (B=32): {res_24['latency_b32_ms']:.2f} ms")

    # 7. Block-Sparse (BSR) Pruning
    print("\n[Prune] Applying Block-Sparse (BSR) Pruning (16x16 blocks)...")
    model_bsr = copy.deepcopy(model)
    masks_bsr = prune_block_sparse(model_bsr, block_h=16, block_w=16, block_sparsity=0.50, skip_first=True, skip_last=True)
    model_bsr, _ = fine_tune_recovery(
        model_bsr,
        train_loader,
        val_loader,
        criterion,
        device,
        epochs=2,
        lr=0.0003,
        masks=masks_bsr
    )
    bsr_pt_path = "models/hwr_block_sparse.pt"
    torch.save(model_bsr.state_dict(), bsr_pt_path)

    res_bsr = benchmark_model(
        "5. Block-Sparse BSR (16x16)",
        model_bsr,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=bsr_pt_path
    )
    results.append(res_bsr)
    print(f"   Accuracy: {res_bsr['accuracy_pct']:.2f}%")

    # 8. Structured Physical Pruning (-30% dimensions)
    print("\n[Prune] Applying Structured L2-Norm Pruning (30% physical channel/neuron reduction)...")
    model_struct = copy.deepcopy(model)
    model_struct = prune_structured_l2(model_struct, prune_ratio=0.30, skip_first=True, skip_last=True)
    optimizer_struct = torch.optim.AdamW(model_struct.parameters(), lr=0.001)
    model_struct.train()
    for _ in range(3):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_struct.zero_grad()
            loss = criterion(model_struct(bx), by)
            loss.backward()
            optimizer_struct.step()

    struct_pt_path = "models/hwr_structured_physical.pt"
    torch.save(model_struct.state_dict(), struct_pt_path)

    res_struct = benchmark_model(
        "6. Structured Physical (-30% dims)",
        model_struct,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=struct_pt_path
    )
    results.append(res_struct)
    print(f"   Accuracy: {res_struct['accuracy_pct']:.2f}% | Latency (B=32): {res_struct['latency_b32_ms']:.2f} ms")

    # 9. HK Dual-Mode 4-bit NF4 Quantization with Precision Recovery
    print("\n[Export] Exporting to HK Dual-Mode NF4 Format (.hk)...")
    hk_dq4_path = "models/hwr_dual_mode_dq4.hk"
    save_hk(
        hk_dq4_path,
        model_recovered.state_dict(),
        metadata={
            "architecture": "HWRNet-DualMode-NF4",
            "framework": "HK Neural Tensor Format",
            "quantization": "4-bit NF4 + Precision Recovery Residuals",
        },
        quantize_mode="dq4",
        include_residual=True
    )
    print(f"   [OK] Saved {hk_dq4_path} ({os.path.getsize(hk_dq4_path)/1024:.1f} KB)")

    loaded_hk = load_hk(hk_dq4_path, device=str(device), with_residual=True)
    model_loaded = HWRNet().to(device)
    model_loaded.load_state_dict(loaded_hk.state_dict)

    res_hk = benchmark_model(
        "7. HK Dual-Mode (NF4 + Residual .hk)",
        model_loaded,
        val_loader,
        criterion,
        sample_batch,
        device,
        disk_path=hk_dq4_path
    )
    results.append(res_hk)
    print(f"   Accuracy: {res_hk['accuracy_pct']:.2f}%")

    # 10. Native Zig HK CLI Verification & Benchmarks
    print("\n" + "=" * 80)
    print("NATIVE ZIG HK CLI VERIFICATION & BENCHMARK SUITE")
    print("=" * 80)
    cli_exe = os.path.abspath("zig-out/bin/hk.exe")

    if os.path.exists(cli_exe):
        print(f"\n>>> Running: hk.exe verify {hk_dq4_path}")
        subprocess.run([cli_exe, "verify", hk_dq4_path])

        print(f"\n>>> Running: hk.exe inspect {hk_dq4_path}")
        subprocess.run([cli_exe, "inspect", hk_dq4_path])

        print(f"\n>>> Running: hk.exe benchmark {hk_dq4_path}")
        subprocess.run([cli_exe, "benchmark", hk_dq4_path])

        print(f"\n>>> Running: hk.exe verify {hk_sparse_path}")
        subprocess.run([cli_exe, "verify", hk_sparse_path])

        print(f"\n>>> Running: hk.exe inspect {hk_sparse_path}")
        subprocess.run([cli_exe, "inspect", hk_sparse_path])

        # Test CLI retile
        retiled_path = "models/hwr_retiled_16x16.hk"
        print(f"\n>>> Running: hk.exe retile {hk_sparse_path} {retiled_path} tile_16x16")
        subprocess.run([cli_exe, "retile", hk_sparse_path, retiled_path, "tile_16x16"])
    else:
        print(f"[Warning] CLI executable {cli_exe} not found.")

    # 11. Print Final Comparative Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    table_md = compare_models(results)
    print(table_md)
    print("=" * 80)


if __name__ == "__main__":
    main()
