"""
HK Neural Tensor Format: Advanced Model Pruning & Sparsification Suite
Provides unstructured magnitude pruning, Wanda (Weights & Activations) pruning,
Ampere 2:4 structured hardware pruning, BSR block sparsity, structured L2 channel reduction,
and mask-constrained fine-tuning recovery.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import make_2_4_sparse


class LayerSparsitySchedule:
    """Manages layer-specific sparsity targets during progressive compression."""
    def __init__(
        self,
        default_ratio: float = 0.50,
        skip_first: bool = True,
        skip_last: bool = True,
        layer_ratios: Optional[Dict[str, float]] = None,
    ):
        self.default_ratio = default_ratio
        self.skip_first = skip_first
        self.skip_last = skip_last
        self.layer_ratios = layer_ratios or {}

    def get_ratio(self, name: str, is_first: bool, is_last: bool) -> float:
        if is_first and self.skip_first:
            return 0.0
        if is_last and self.skip_last:
            return 0.0
        return self.layer_ratios.get(name, self.default_ratio)


def _get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def prune_unstructured_magnitude(
    model: nn.Module,
    sparsity_ratio: float = 0.50,
    schedule: Optional[LayerSparsitySchedule] = None,
) -> Dict[str, torch.Tensor]:
    """Applies unstructured magnitude pruning to linear and convolutional layers."""
    layers = _get_prunable_layers(model)
    masks: Dict[str, torch.Tensor] = {}

    for i, (name, layer) in enumerate(layers):
        is_first = (i == 0)
        is_last = (i == len(layers) - 1)
        ratio = schedule.get_ratio(name, is_first, is_last) if schedule else sparsity_ratio

        if ratio <= 0.0:
            masks[name] = torch.ones_like(layer.weight, dtype=torch.bool)
            continue

        w = layer.weight.data
        k = int(ratio * w.numel())
        if k > 0:
            thresh = torch.kthvalue(w.abs().flatten(), k).values
            mask = w.abs() > thresh
        else:
            mask = torch.ones_like(w, dtype=torch.bool)

        w.mul_(mask.to(w.dtype))
        masks[name] = mask

    return masks


def prune_wanda(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    sparsity_ratio: float = 0.50,
    schedule: Optional[LayerSparsitySchedule] = None,
    num_calibration_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Applies Wanda (Pruning by Weights and activations) pruning.
    Evaluates layer input norm over calibration batches: score = |W| * ||X||_2.
    """
    layers = _get_prunable_layers(model)
    activation_norms: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3: # (B, S, D)
                norm = torch.norm(x, p=2, dim=(0, 1))
            elif x.dim() == 2: # (B, D)
                norm = torch.norm(x, p=2, dim=0)
            else:
                norm = torch.norm(x.view(-1, x.shape[-1]), p=2, dim=0)

            if layer_name not in activation_norms:
                activation_norms[layer_name] = norm
            else:
                activation_norms[layer_name] += norm
        return hook_fn

    for name, layer in layers:
        handles.append(layer.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_calibration_batches:
                break
            if isinstance(batch, (tuple, list)):
                bx = batch[0].to(device)
            else:
                bx = batch.to(device)
            model(bx)

    for h in handles:
        h.remove()

    masks: Dict[str, torch.Tensor] = {}
    for i, (name, layer) in enumerate(layers):
        is_first = (i == 0)
        is_last = (i == len(layers) - 1)
        ratio = schedule.get_ratio(name, is_first, is_last) if schedule else sparsity_ratio

        if ratio <= 0.0 or name not in activation_norms:
            masks[name] = torch.ones_like(layer.weight, dtype=torch.bool)
            continue

        w = layer.weight.data
        act_norm = activation_norms[name].to(w.device)
        score = w.abs() * act_norm.unsqueeze(0)

        k = int(ratio * score.numel())
        if k > 0:
            thresh = torch.kthvalue(score.flatten(), k).values
            mask = score > thresh
        else:
            mask = torch.ones_like(w, dtype=torch.bool)

        w.mul_(mask.to(w.dtype))
        masks[name] = mask

    return masks


def prune_structured_2_4(
    model: nn.Module,
    skip_first: bool = True,
    skip_last: bool = True,
) -> Dict[str, torch.Tensor]:
    """Enforces NVIDIA Ampere 2:4 structured hardware sparsity on weights."""
    layers = _get_prunable_layers(model)
    masks: Dict[str, torch.Tensor] = {}

    for i, (name, layer) in enumerate(layers):
        if (i == 0 and skip_first) or (i == len(layers) - 1 and skip_last):
            masks[name] = torch.ones_like(layer.weight, dtype=torch.bool)
            continue

        sparse_w, _ = make_2_4_sparse(layer.weight.data)
        layer.weight.data.copy_(sparse_w)
        masks[name] = (sparse_w != 0)

    return masks


def prune_block_sparse(
    model: nn.Module,
    block_h: int = 16,
    block_w: int = 16,
    block_sparsity: float = 0.50,
    skip_first: bool = True,
    skip_last: bool = True,
) -> Dict[str, torch.Tensor]:
    """Applies Block-Sparse (BSR) pruning zeroes out entire sub-blocks by Frobenius norm."""
    layers = _get_prunable_layers(model)
    masks: Dict[str, torch.Tensor] = {}

    for i, (name, layer) in enumerate(layers):
        if (i == 0 and skip_first) or (i == len(layers) - 1 and skip_last):
            masks[name] = torch.ones_like(layer.weight, dtype=torch.bool)
            continue

        w = layer.weight.data
        H, W = w.shape
        pad_h = (block_h - (H % block_h)) % block_h
        pad_w = (block_w - (W % block_w)) % block_w
        if pad_h > 0 or pad_w > 0:
            w_padded = F.pad(w, (0, pad_w, 0, pad_h))
        else:
            w_padded = w

        pH, pW = w_padded.shape
        blocks = w_padded.view(pH // block_h, block_h, pW // block_w, block_w).permute(0, 2, 1, 3)
        norms = torch.norm(blocks, p=2, dim=(2, 3)) # (num_blocks_h, num_blocks_w)

        k = int(block_sparsity * norms.numel())
        if k > 0:
            thresh = torch.kthvalue(norms.flatten(), k).values
            block_mask = (norms > thresh).unsqueeze(-1).unsqueeze(-1)
            blocks_masked = blocks * block_mask.to(blocks.dtype)
            reconstructed = blocks_masked.permute(0, 2, 1, 3).reshape(pH, pW)
            w.copy_(reconstructed[:H, :W])
            masks[name] = (w != 0)
        else:
            masks[name] = torch.ones_like(w, dtype=torch.bool)

    return masks


def prune_structured_l2(
    model: nn.Module,
    prune_ratio: float = 0.30,
    skip_first: bool = True,
    skip_last: bool = True,
) -> nn.Module:
    """Physically reduces channels/neurons in linear layers based on L2 norm."""
    linear_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        return model

    # Prune middle linear layer
    target_name = linear_layers[1] if len(linear_layers) > 2 else linear_layers[0]
    for i in range(len(linear_layers) - 1):
        l1_name = linear_layers[i]
        l2_name = linear_layers[i + 1]
        if i == 0 and skip_first:
            continue

        l1: nn.Linear = dict(model.named_modules())[l1_name]
        l2: nn.Linear = dict(model.named_modules())[l2_name]

        out_features = l1.out_features
        keep_features = max(1, int(out_features * (1.0 - prune_ratio)))
        norms = torch.norm(l1.weight.data, p=2, dim=1)
        _, keep_indices = torch.topk(norms, k=keep_features, largest=True)
        keep_indices = torch.sort(keep_indices).values

        # Physically slice weights
        new_l1_w = l1.weight.data[keep_indices, :].clone()
        new_l1_b = l1.bias.data[keep_indices].clone() if l1.bias is not None else None

        new_l2_w = l2.weight.data[:, keep_indices].clone()
        new_l2_b = l2.bias.data.clone() if l2.bias is not None else None

        l1.out_features = keep_features
        l1.weight = nn.Parameter(new_l1_w)
        if new_l1_b is not None:
            l1.bias = nn.Parameter(new_l1_b)

        l2.in_features = keep_features
        l2.weight = nn.Parameter(new_l2_w)
        if new_l2_b is not None:
            l2.bias = nn.Parameter(new_l2_b)

        # Check for intermediate batch norm matching l1
        bn_target_name = l1_name.replace("fc", "bn")
        for m_name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm1d) and (m_name == bn_target_name or (m.num_features == out_features and m_name.endswith(l1_name[-1]))):
                m.num_features = keep_features
                if m.weight is not None:
                    m.weight = nn.Parameter(m.weight.data[keep_indices].clone())
                if m.bias is not None:
                    m.bias = nn.Parameter(m.bias.data[keep_indices].clone())
                if m.running_mean is not None:
                    m.running_mean = m.running_mean[keep_indices].clone()
                if m.running_var is not None:
                    m.running_var = m.running_var[keep_indices].clone()
                break

        break

    return model


def fine_tune_recovery(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    device: torch.device,
    epochs: int = 2,
    lr: float = 0.0003,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[nn.Module, float]:
    """Fine-tunes a pruned model while rigidly freezing pruned zeros."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                bx, by = batch[0].to(device), batch[1].to(device)
            else:
                bx = batch.to(device)
                by = bx

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()

            # Zero out gradients for pruned weights
            if masks is not None:
                for name, layer in model.named_modules():
                    if name in masks and layer.weight.grad is not None:
                        mask = masks[name].to(device)
                        layer.weight.grad.data.mul_(mask.to(layer.weight.grad.dtype))

            optimizer.step()

            # Re-enforce exact zeros
            if masks is not None:
                with torch.no_grad():
                    for name, layer in model.named_modules():
                        if name in masks:
                            mask = masks[name].to(device)
                            layer.weight.data.mul_(mask.to(layer.weight.dtype))

    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                bx, by = batch[0].to(device), batch[1].to(device)
            else:
                bx = batch.to(device)
                by = None
            out = model(bx)
            if by is not None:
                preds = out.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.numel()

    acc = (correct / total * 100.0) if total > 0 else 0.0
    return model, acc
