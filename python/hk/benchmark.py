"""
HK Neural Tensor Format: Benchmark & Model Profiling Engine
Measures latency, throughput, accuracy, compression ratio, and memory consumption.
"""

from typing import Dict, Any, List, Optional
import os
import time
import numpy as np
import torch
import torch.nn as nn


def benchmark_model(
    name: str,
    model: nn.Module,
    val_loader: Any,
    criterion: Any,
    sample_batch: torch.Tensor,
    device: torch.device,
    disk_path: Optional[str] = None,
    num_runs: int = 40,
) -> Dict[str, Any]:
    """Exhaustively profiles a model for accuracy, loss, latency, and memory footprint."""
    model.to(device)
    model.eval()

    # 1. Validation Accuracy & Loss
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                bx, by = batch[0].to(device), batch[1].to(device)
            else:
                bx = batch.to(device)
                by = None

            out = model(bx)
            if by is not None and criterion is not None:
                loss = criterion(out, by)
                total_loss += loss.item() * bx.size(0)
                preds = out.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total_samples += by.size(0)

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    acc_pct = (correct / total_samples * 100.0) if total_samples > 0 else 0.0

    # 2. Warmup & Latency
    sample = sample_batch.to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    avg_lat = float(np.median(latencies))

    # 3. Model Parameters & Storage
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    sparsity_pct = (1.0 - (nonzero_params / total_params)) * 100.0 if total_params > 0 else 0.0

    disk_kb = 0.0
    if disk_path and os.path.exists(disk_path):
        disk_kb = os.path.getsize(disk_path) / 1024.0

    return {
        "name": name,
        "accuracy_pct": acc_pct,
        "loss": avg_loss,
        "latency_b32_ms": avg_lat,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity_pct": sparsity_pct,
        "disk_size_kb": disk_kb,
        "disk_path": disk_path or "in-memory",
    }


def compare_models(results: List[Dict[str, Any]]) -> str:
    """Renders a formatted GitHub-flavored Markdown comparison table."""
    if not results:
        return "No benchmark results to compare."

    base = results[0]
    base_size = base["disk_size_kb"] if base["disk_size_kb"] > 0 else 1.0
    base_lat = base["latency_b32_ms"] if base["latency_b32_ms"] > 0 else 1.0

    lines = [
        "| Configuration / Format | Accuracy (%) | Latency (ms) | Speedup | Storage (KB) | Compression | Sparsity (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        size_kb = r["disk_size_kb"]
        comp = f"{base_size / size_kb:.2f}x" if size_kb > 0 else "N/A"
        lat = r["latency_b32_ms"]
        speed = f"{base_lat / lat:.2f}x" if lat > 0 else "1.00x"
        line = (
            f"| {r['name']} | {r['accuracy_pct']:.2f}% | {lat:.2f} | {speed} | "
            f"{size_kb:.1f} | {comp} | {r['sparsity_pct']:.1f}% |"
        )
        lines.append(line)

    return "\n".join(lines)
