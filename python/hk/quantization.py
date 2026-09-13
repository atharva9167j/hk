"""
HK Neural Tensor Format: High-Performance Quantization & Sparsity Engine
Dual-mode quantization (lossy baseline + lossless residual recovery),
NormalFloat-4 (NF4), INT8 (DQ8), BitNet ternary (DQT), and NVIDIA Ampere 2:4 structured sparsity.
"""

from typing import Tuple, Optional, List, Union, Any
import struct
import numpy as np
import torch
import torch.nn.functional as F

from .format import (
    StorageType,
    STORAGE_Q2_K, STORAGE_Q3_K, STORAGE_Q4_K, STORAGE_Q5_K, STORAGE_Q6_K, STORAGE_Q8_K,
    STORAGE_IQ1_S, STORAGE_IQ1_M, STORAGE_IQ2_XXS, STORAGE_IQ2_XS, STORAGE_IQ3_XXS, STORAGE_IQ4_NL, STORAGE_IQ4_XS,
    STORAGE_TQ1_0, STORAGE_TQ2_0, STORAGE_MXFP4, STORAGE_NVFP4,
    STORAGE_F32, STORAGE_F16, STORAGE_BF16, STORAGE_DQ4, STORAGE_DQ8,
    STORAGE_Q4_0, STORAGE_Q8_0,
)
from .native import (
    native_pack_2_4, native_unpack_2_4, is_native_available,
    native_quantize_q4_k, native_dequantize_q4_k,
    native_quantize_q8_k, native_dequantize_q8_k,
    native_quantize_q6_k, native_dequantize_q6_k,
    native_quantize_q2_k, native_dequantize_q2_k,
    native_quantize_q4_0, native_dequantize_q4_0,
    native_quantize_q8_0, native_dequantize_q8_0,
    native_quantize_q5_k, native_dequantize_q5_k,
    native_quantize_q3_k, native_dequantize_q3_k,
    native_quantize_tensor_q4_0, native_dequantize_tensor_q4_0,
    native_quantize_tensor_q8_0, native_dequantize_tensor_q8_0,
    native_quantize_tensor_q4_k, native_dequantize_tensor_q4_k,
    native_quantize_tensor_q8_k, native_dequantize_tensor_q8_k,
    native_quantize_tensor_q6_k, native_dequantize_tensor_q6_k,
    native_quantize_tensor_q5_k, native_dequantize_tensor_q5_k,
    native_quantize_tensor_q3_k, native_dequantize_tensor_q3_k,
    native_quantize_tensor_q2_k, native_dequantize_tensor_q2_k,
)

# 16-point NormalFloat-4 (NF4) quantile lookup table matching Zig implementation
NF4_TABLE = np.array([
    -1.0000000,
    -0.6961928,
    -0.5250731,
    -0.3949175,
    -0.2844414,
    -0.1847734,
    -0.0910500,
    0.0000000,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.5626170,
    0.7229568,
    1.0000000,
], dtype=np.float32)


def make_2_4_sparse(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Applies NVIDIA Ampere 2:4 structured hardware sparsity to a tensor.
    In every group of 4 elements along the last dimension, keeps the 2 largest
    by magnitude and zeroes out the remaining 2.
    """
    orig_shape = tensor.shape
    flat = tensor.detach().cpu().to(torch.float32).clone().flatten()
    n = flat.numel()
    pad_len = (4 - (n % 4)) % 4
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    grouped = flat.view(-1, 4)
    # Find top-2 indices by magnitude per group
    _, top2_idx = torch.topk(grouped.abs(), k=2, dim=1)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(1, top2_idx, True)
    sparse_grouped = grouped * mask.to(grouped.dtype)

    sparse_flat = sparse_grouped.flatten()
    if pad_len > 0:
        sparse_flat = sparse_flat[:n]

    result = sparse_flat.view(orig_shape).to(tensor.dtype).to(tensor.device)
    return result, 0.5


def pack_2_4(sparse_tensor: torch.Tensor) -> bytes:
    """Packs a 2:4 sparse tensor into Ampere physical hardware format using compiled Zig engine."""
    native_bytes = native_pack_2_4(sparse_tensor)
    if native_bytes is not None:
        return native_bytes
    raise RuntimeError("Native Zig library required for pack_2_4")


def unpack_2_4(packed_data: Union[bytes, bytearray, memoryview], shape: List[int]) -> torch.Tensor:
    """Unpacks Ampere 2:4 structured sparsity back to dense PyTorch tensor using compiled Zig engine."""
    t = native_unpack_2_4(packed_data, tuple(shape))
    if t is not None:
        return t
    raise RuntimeError("Native Zig library required for unpack_2_4")


def quantize_nf4_dual_mode(
    tensor: torch.Tensor,
    block_size: int = 32,
    compute_residual: bool = True,
) -> Tuple[bytes, bytes, Optional[bytes]]:
    """Quantizes a float tensor to NormalFloat-4 (NF4) with optional residual recovery buffer."""
    flat = tensor.detach().cpu().to(torch.float32).numpy().flatten()
    n = flat.size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    num_blocks = flat.size // block_size

    blocks = flat.reshape(num_blocks, block_size)
    scales = np.max(np.abs(blocks), axis=1)
    scales = np.where(scales == 0.0, 1e-8, scales).astype(np.float32)

    norm_blocks = np.clip(blocks / scales[:, None], -1.0, 1.0)
    # Map to closest NF4 code (0..15)
    # Broadcast norm_blocks against NF4_TABLE: shape (num_blocks, block_size, 16)
    diffs = np.abs(norm_blocks[:, :, None] - NF4_TABLE[None, None, :])
    codes = np.argmin(diffs, axis=2).astype(np.uint8) # shape (num_blocks, block_size)

    # Pack 2 4-bit codes per byte
    codes_flat = codes.flatten()
    low = codes_flat[0::2] & 0x0F
    high = (codes_flat[1::2] & 0x0F) << 4
    packed = (low | high).astype(np.uint8).tobytes()

    scales_bytes = scales.tobytes()

    residual_bytes = None
    if compute_residual:
        approx = NF4_TABLE[codes] * scales[:, None]
        residual = (blocks - approx).astype(np.float32).flatten()
        if pad_len > 0:
            residual = residual[:n]
        residual_bytes = residual.tobytes()

    return packed, scales_bytes, residual_bytes


def dequantize_nf4_dual_mode(
    packed_data: bytes,
    scales_data: bytes,
    shape: List[int],
    block_size: int = 32,
    residual: Optional[bytes] = None,
) -> torch.Tensor:
    """Dequantizes NF4 packed buffer back into PyTorch float tensor with optional residual addition."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (block_size - (total_elements % block_size)) % block_size
    n_padded = total_elements + pad_len
    num_blocks = n_padded // block_size

    scales = np.frombuffer(scales_data, dtype=np.float32)
    packed = np.frombuffer(packed_data, dtype=np.uint8)

    codes = np.empty(n_padded, dtype=np.uint8)
    codes[0::2] = packed & 0x0F
    codes[1::2] = (packed >> 4) & 0x0F

    codes_blocks = codes.reshape(num_blocks, block_size)
    approx = (NF4_TABLE[codes_blocks] * scales[:num_blocks, None]).flatten()

    if pad_len > 0:
        approx = approx[:total_elements]

    if residual is not None and len(residual) > 0:
        res_arr = np.frombuffer(residual, dtype=np.float32)
        approx = approx + res_arr[:total_elements]

    return torch.from_numpy(approx).view(*shape)


def quantize_dq8_dual_mode(
    tensor: torch.Tensor,
    block_size: int = 32,
    compute_residual: bool = False,
) -> Tuple[bytes, bytes, Optional[bytes]]:
    """Quantizes a float tensor to symmetric INT8 with optional residual recovery."""
    flat = tensor.detach().cpu().to(torch.float32).numpy().flatten()
    n = flat.size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    num_blocks = flat.size // block_size

    blocks = flat.reshape(num_blocks, block_size)
    max_abs = np.max(np.abs(blocks), axis=1)
    scales = np.where(max_abs == 0.0, 1e-8, max_abs / 127.0).astype(np.float32)

    q = np.clip(np.round(blocks / scales[:, None]), -128, 127).astype(np.int8)
    packed = q.tobytes()
    scales_bytes = scales.tobytes()

    residual_bytes = None
    if compute_residual:
        approx = (q.astype(np.float32) * scales[:, None]).flatten()
        if pad_len > 0:
            approx = approx[:n]
        residual = (flat[:n] - approx).astype(np.float32)
        residual_bytes = residual.tobytes()

    return packed, scales_bytes, residual_bytes


def dequantize_dq8_dual_mode(
    packed_data: bytes,
    scales_data: bytes,
    shape: List[int],
    block_size: int = 32,
    residual: Optional[bytes] = None,
) -> torch.Tensor:
    """Dequantizes INT8 packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (block_size - (total_elements % block_size)) % block_size
    n_padded = total_elements + pad_len
    num_blocks = n_padded // block_size

    scales = np.frombuffer(scales_data, dtype=np.float32)
    q = np.frombuffer(packed_data, dtype=np.int8).reshape(num_blocks, block_size)

    approx = (q.astype(np.float32) * scales[:num_blocks, None]).flatten()
    if pad_len > 0:
        approx = approx[:total_elements]

    if residual is not None and len(residual) > 0:
        res_arr = np.frombuffer(residual, dtype=np.float32)
        approx = approx + res_arr[:total_elements]

    return torch.from_numpy(approx).view(*shape)


def quantize_dqt(
    tensor: torch.Tensor,
    block_size: int = 32,
    compute_residual: bool = False,
) -> Tuple[bytes, bytes, Optional[bytes]]:
    """Quantizes a float tensor to BitNet 1.58 ternary {-1, 0, +1}."""
    flat = tensor.detach().cpu().to(torch.float32).numpy().flatten()
    n = flat.size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    num_blocks = flat.size // block_size

    blocks = flat.reshape(num_blocks, block_size)
    mean_abs = np.mean(np.abs(blocks), axis=1)
    scales = np.where(mean_abs == 0.0, 1e-8, mean_abs).astype(np.float32)

    # Ternary: -1, 0, 1
    ternary = np.clip(np.round(blocks / scales[:, None]), -1, 1).astype(np.int8)

    # Pack 4 ternary values per byte (2 bits each: 0 -> 00, +1 -> 01, -1 -> 11)
    t_flat = ternary.flatten()
    codes = np.zeros(t_flat.size, dtype=np.uint8)
    codes[t_flat == 1] = 0x01
    codes[t_flat == -1] = 0x03

    p0 = codes[0::4]
    p1 = codes[1::4] << 2
    p2 = codes[2::4] << 4
    p3 = codes[3::4] << 6
    packed = (p0 | p1 | p2 | p3).astype(np.uint8).tobytes()
    scales_bytes = scales.tobytes()

    residual_bytes = None
    if compute_residual:
        approx = (ternary.astype(np.float32) * scales[:, None]).flatten()
        if pad_len > 0:
            approx = approx[:n]
        residual = (flat[:n] - approx).astype(np.float32)
        residual_bytes = residual.tobytes()

    return packed, scales_bytes, residual_bytes


def dequantize_dqt(
    packed_data: bytes,
    scales_data: bytes,
    shape: List[int],
    block_size: int = 32,
    residual: Optional[bytes] = None,
) -> torch.Tensor:
    """Dequantizes BitNet ternary packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (block_size - (total_elements % block_size)) % block_size
    n_padded = total_elements + pad_len
    num_blocks = n_padded // block_size

    scales = np.frombuffer(scales_data, dtype=np.float32)
    packed = np.frombuffer(packed_data, dtype=np.uint8)

    codes = np.empty(n_padded, dtype=np.uint8)
    codes[0::4] = packed & 0x03
    codes[1::4] = (packed >> 2) & 0x03
    codes[2::4] = (packed >> 4) & 0x03
    codes[3::4] = (packed >> 6) & 0x03

    ternary = np.zeros(n_padded, dtype=np.float32)
    ternary[codes == 0x01] = 1.0
    ternary[codes == 0x03] = -1.0

    approx = (ternary.reshape(num_blocks, block_size) * scales[:num_blocks, None]).flatten()
    if pad_len > 0:
        approx = approx[:total_elements]

    if residual is not None and len(residual) > 0:
        res_arr = np.frombuffer(residual, dtype=np.float32)
        approx = approx + res_arr[:total_elements]

    return torch.from_numpy(approx).view(*shape)


# ---------------------------------------------------------------------------
# Advanced K-Quants Full-Tensor Routines (256-element Super-Blocks)
# ---------------------------------------------------------------------------

QK_K = 256
BLOCK_Q4_K_SIZE = 144
BLOCK_Q8_K_SIZE = 292
BLOCK_Q6_K_SIZE = 210
BLOCK_Q2_K_SIZE = 84


def quantize_q4_k(tensor: torch.Tensor, imatrix: Optional[np.ndarray] = None) -> bytes:
    """
    Quantizes a float tensor into HK Q4_K format using 256-element super-blocks.
    Each super-block contains 144 bytes.
    Optionally applies importance matrix calibration weighting.
    """
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK_K - (n % QK_K)) % QK_K
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    if imatrix is not None and imatrix.size >= flat.size:
        flat = flat * np.clip(imatrix[:flat.size], 0.1, 10.0)
    return native_quantize_tensor_q4_k(flat)


def dequantize_q4_k(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes HK Q4_K packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK_K - (total_elements % QK_K)) % QK_K
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q4_k(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def quantize_q8_k(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into HK Q8_K format using 256-element super-blocks (292 bytes)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK_K - (n % QK_K)) % QK_K
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q8_k(flat)


def dequantize_q8_k(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes HK Q8_K packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK_K - (total_elements % QK_K)) % QK_K
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q8_k(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def dequantize_q6_k(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes HK Q6_K packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK_K - (total_elements % QK_K)) % QK_K
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q6_k(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def dequantize_q2_k(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes HK Q2_K packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK_K - (total_elements % QK_K)) % QK_K
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q2_k(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def quantize_q6_k(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into HK Q6_K format (256 weights, 210 bytes per block)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK_K - (n % QK_K)) % QK_K
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q6_k(flat)


def quantize_q2_k(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into HK Q2_K format (256 weights, 84 bytes per block)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK_K - (n % QK_K)) % QK_K
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q2_k(flat)


BLOCK_Q4_0_SIZE = 18
QK4_0 = 32
BLOCK_Q8_0_SIZE = 34
QK8_0 = 32
BLOCK_Q5_K_SIZE = 176
BLOCK_Q3_K_SIZE = 110


def quantize_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into standard Q4_0 format (32 weights, 18 bytes per block)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK4_0 - (n % QK4_0)) % QK4_0
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q4_0(flat)


def dequantize_q4_0(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes Q4_0 packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK4_0 - (total_elements % QK4_0)) % QK4_0
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q4_0(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def quantize_q8_0(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into standard Q8_0 format (32 weights, 34 bytes per block)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK8_0 - (n % QK8_0)) % QK8_0
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q8_0(flat)


def dequantize_q8_0(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes Q8_0 packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK8_0 - (total_elements % QK8_0)) % QK8_0
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q8_0(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def quantize_q5_k(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into Q5_K format (256 weights, 176 bytes per block)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK_K - (n % QK_K)) % QK_K
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q5_k(flat)


def dequantize_q5_k(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes Q5_K packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK_K - (total_elements % QK_K)) % QK_K
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q5_k(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


def quantize_q3_k(tensor: torch.Tensor) -> bytes:
    """Quantizes a float tensor into Q3_K format (256 weights, 110 bytes per block)."""
    flat = tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_len = (QK_K - (n % QK_K)) % QK_K
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    return native_quantize_tensor_q3_k(flat)


def dequantize_q3_k(packed_data: bytes, shape: List[int]) -> torch.Tensor:
    """Dequantizes Q3_K packed buffer back into PyTorch float tensor."""
    total_elements = 1
    for d in shape:
        total_elements *= d
    pad_len = (QK_K - (total_elements % QK_K)) % QK_K
    n_padded = total_elements + pad_len
    out_arr = native_dequantize_tensor_q3_k(packed_data, n_padded)
    if pad_len > 0:
        out_arr = out_arr[:total_elements]
    return torch.from_numpy(out_arr).view(*shape)


# ---------------------------------------------------------------------------
# Importance Matrix Calibration Engine (imatrix)
# ---------------------------------------------------------------------------

class ImportanceMatrixCalibrator:
    """
    Computes activation second-moment statistics (imatrix) across calibration sequences.
    Estimates Fisher information per tensor channel: I = E[activations^2].
    Used to guide mixed-precision K-Quant and I-Quant assignments and protect sensitive weights.
    """
    def __init__(self):
        self.stats: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}

    def observe(self, name: str, activations: Union[torch.Tensor, np.ndarray]):
        """Accumulates squared activation norms for a given module/tensor."""
        if isinstance(activations, torch.Tensor):
            act = activations.detach().cpu().to(torch.float32).numpy()
        else:
            act = np.asarray(activations, dtype=np.float32)

        if act.ndim > 1:
            act = act.reshape(-1, act.shape[-1])
            sq = np.mean(act ** 2, axis=0)
        else:
            sq = act ** 2

        if name not in self.stats:
            self.stats[name] = sq
            self.counts[name] = 1
        else:
            self.stats[name] += sq
            self.counts[name] += 1

    def get_importance(self, name: str) -> Optional[np.ndarray]:
        """Returns the normalized importance weight vector for a given tensor."""
        if name not in self.stats:
            return None
        raw = self.stats[name] / max(1, self.counts[name])
        # Normalize to mean 1.0
        m = np.mean(raw)
        if m > 1e-8:
            raw = raw / m
        return raw

    def save(self, file_path: str):
        """Saves calibration statistics to an .npz file."""
        np.savez(file_path, **{k: self.stats[k] / max(1, self.counts[k]) for k in self.stats})

    @classmethod
    def load(cls, file_path: str) -> "ImportanceMatrixCalibrator":
        """Loads calibration statistics from an .npz file."""
        cal = cls()
        data = np.load(file_path)
        for k in data.files:
            cal.stats[k] = data[k]
            cal.counts[k] = 1
        return cal


# ---------------------------------------------------------------------------
# Predefined Per-Tensor Mixed Quantization Recipes
# ---------------------------------------------------------------------------

QUANT_RECIPES: dict[str, dict[str, StorageType]] = {
    "Q4_K_M": {
        "token_embd": StorageType.Q4_K,
        "output": StorageType.Q6_K,
        "attn_v": StorageType.Q6_K,
        "attn_output": StorageType.Q6_K,
        "ffn_down": StorageType.Q6_K,
        "default": StorageType.Q4_K,
    },
    "Q5_K_M": {
        "token_embd": StorageType.Q5_K,
        "output": StorageType.Q6_K,
        "attn_q": StorageType.Q5_K,
        "attn_k": StorageType.Q5_K,
        "attn_v": StorageType.Q6_K,
        "attn_output": StorageType.Q6_K,
        "ffn_gate": StorageType.Q5_K,
        "ffn_up": StorageType.Q5_K,
        "ffn_down": StorageType.Q6_K,
        "default": StorageType.Q5_K,
    },
    "Q4_K_S": {
        "output": StorageType.Q4_K,
        "default": StorageType.Q4_K,
    },
    "Q5_K_S": {
        "output": StorageType.Q5_K,
        "default": StorageType.Q5_K,
    },
    "Q3_K_M": {
        "output": StorageType.Q4_K,
        "attn_v": StorageType.Q4_K,
        "ffn_down": StorageType.Q4_K,
        "default": StorageType.Q3_K,
    },
    "Q2_K": {
        "default": StorageType.Q2_K,
    },
    "Q6_K": {
        "default": StorageType.Q6_K,
    },
    "Q8_K": {
        "default": StorageType.Q8_K,
    },
    "IQ4_NL": {
        "default": StorageType.IQ4_NL,
    },
}


def resolve_quant_type_for_tensor(recipe_name: str, tensor_name: str) -> StorageType:
    """Resolves the designated StorageType for a tensor according to the chosen quantization recipe."""
    recipe = QUANT_RECIPES.get(recipe_name.upper())
    if not recipe:
        raise ValueError(f"Unknown quantization recipe '{recipe_name}'. Available: {list(QUANT_RECIPES.keys())}")

    # Keep 1D normalization weights, biases, and RoPE frequencies in high precision (FP32/FP16)
    lower_name = tensor_name.lower()
    if any(k in lower_name for k in ["norm", "bias", "rope", "inv_freq", "embed_tokens"]):
        return StorageType.F32

    for key, st in recipe.items():
        if key != "default" and key in lower_name:
            return st

    return recipe.get("default", StorageType.Q4_K)
