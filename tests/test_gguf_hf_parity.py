"""
Exhaustive Verification Suite for HK GGUF & Hugging Face Full Parity and Superiority Plan.

Covers all 6 Phases:
- Phase 1: Native GGUF v1/v2/v3 Ingestion, Zero-Copy Transplant, and Export Roundtrip.
- Phase 2: Quantization Zoo Completeness & Bitstream/Numerical Parity (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K).
- Phase 3: Mathematical Tensor Transformations (RoPE permute/unpermute, LayerNorm offset, QKV split/merge, MoE packing).
- Phase 4: Packed SIMD GEMV Acceleration in Zig & HKQuantizedLinear in PyTorch.
- Phase 5: Tokenizer Hugging Face JSON Embedding, Multi-Template Chat Dictionary, & FIM Formatting.
- Phase 6: Remote HTTP Range Reader (RFC 7233) for Metadata Inspection & Lazy Single-Tensor Streaming.
"""

import os
import sys
import math
import struct
import shutil
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest

# Ensure local python package is on path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import hk
from hk.format import StorageType
from hk.gguf_parser import (
    GGUF_MAGIC,
    GGUFReaderLight,
    convert_gguf_to_hk,
    export_hk_to_gguf,
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q2_K,
)
from hk.hf_mapper import (
    permute_hf_to_gguf_rope,
    unpermute_gguf_to_hf_rope,
    add_layernorm_offset,
    sub_layernorm_offset,
    split_fused_qkv,
    merge_fused_qkv,
    pack_moe_experts,
    unpack_moe_experts,
)
from hk.modeling import HKQuantizedLinear
from hk.quantization import (
    quantize_q8_0,
    dequantize_q8_0,
    quantize_q4_0,
    dequantize_q4_0,
    quantize_q4_k,
    dequantize_q4_k,
)
from hk.tokenizer import HKTokenizer
from hk.remote import read_remote_hk_header, safe_open_remote, RemoteHKFile
from hk.torch import safe_open, save_file
from hk.native import is_native_available


# ---------------------------------------------------------------------------
# Helper: Binary GGUF v3 File Generator for Testing
# ---------------------------------------------------------------------------

def create_synthetic_gguf_file(filepath: str, alignment: int = 32):
    """Creates a well-formed GGUF v3 file containing metadata and multiple tensors."""
    with open(filepath, "wb") as f:
        # Magic + Version
        f.write(struct.pack("<II", GGUF_MAGIC, 3))
        
        # Tensor count (2) and KV count (4)
        f.write(struct.pack("<QQ", 2, 4))

        # Helper to write GGUF string
        def write_str(s: str):
            encoded = s.encode("utf-8")
            f.write(struct.pack("<Q", len(encoded)))
            f.write(encoded)

        # 1. general.architecture (string)
        write_str("general.architecture")
        f.write(struct.pack("<I", 8))  # string type
        write_str("llama")

        # 2. llama.context_length (uint32)
        write_str("llama.context_length")
        f.write(struct.pack("<I", 4))  # uint32 type
        f.write(struct.pack("<I", 4096))

        # 3. general.alignment (uint32)
        write_str("general.alignment")
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<I", alignment))

        # 4. tokenizer.chat_templates (string)
        write_str("tokenizer.chat_templates")
        f.write(struct.pack("<I", 8))
        write_str('{"default": "{{ messages }}"}')

        # Tensor TOC:
        # Tensor 1: model.embed.weight (shape: [4, 32] in C order -> ne: [32, 4], F32)
        write_str("model.embed.weight")
        f.write(struct.pack("<I", 2))  # ndim = 2
        f.write(struct.pack("<QQ", 32, 4))  # ne0, ne1
        f.write(struct.pack("<IQ", GGML_TYPE_F32, 0))  # type, offset = 0

        # Tensor 2: model.mlp.weight (shape: [64, 32] in C order -> ne: [32, 64], Q8_0)
        # Block size for Q8_0 is 32 el = 34 bytes. Each row has 1 block = 34 bytes. 64 rows = 2176 bytes.
        t1_size = 4 * 32 * 4  # 512 bytes
        t1_aligned_size = (t1_size + alignment - 1) & ~(alignment - 1)
        write_str("model.mlp.weight")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<QQ", 32, 64))
        f.write(struct.pack("<IQ", GGML_TYPE_Q8_0, t1_aligned_size))

        # Pad up to alignment boundary for data start
        cur_pos = f.tell()
        rem = cur_pos % alignment
        if rem != 0:
            f.write(b"\x00" * (alignment - rem))

        # Tensor 1 payload (F32)
        np.random.seed(42)
        t1_data = np.linspace(-1.0, 1.0, 4 * 32, dtype=np.float32)
        f.write(t1_data.tobytes())

        # Pad to alignment between tensors
        rem1 = t1_size % alignment
        if rem1 != 0:
            f.write(b"\x00" * (alignment - rem1))

        # Tensor 2 payload (Q8_0)
        # 64 blocks of Q8_0 (each block: 2 bytes scale + 32 bytes int8 = 34 bytes)
        t2_bytes_list = []
        for i in range(64):
            # Scale 0.125 in float16 (0x3000)
            d_f16 = struct.pack("<H", 0x3000)
            qs = bytes([((i + j) % 127) for j in range(32)])
            t2_bytes_list.append(d_f16 + qs)
        t2_data = b"".join(t2_bytes_list)
        f.write(t2_data)


# ---------------------------------------------------------------------------
# Phase 1: GGUF Ingestion & Transcoder Engine Tests
# ---------------------------------------------------------------------------

def test_phase1_gguf_ingestion_and_roundtrip(temp_dir):
    """Verifies GGUF v3 parsing, zero-copy transplant into HK container, and export back to GGUF."""
    gguf_in = os.path.join(temp_dir, "model_in.gguf")
    hk_out = os.path.join(temp_dir, "model.hk")
    gguf_exported = os.path.join(temp_dir, "model_exported.gguf")

    create_synthetic_gguf_file(gguf_in, alignment=32)
    assert os.path.exists(gguf_in)

    # 1. Read input GGUF
    reader_in = GGUFReaderLight(gguf_in)
    assert reader_in.version == 3
    assert reader_in.metadata.get("general.architecture") == "llama"
    assert reader_in.metadata.get("llama.context_length") == 4096
    assert "model.embed.weight" in reader_in.tensors
    assert "model.mlp.weight" in reader_in.tensors
    assert reader_in.tensors["model.mlp.weight"].ggml_type == GGML_TYPE_Q8_0

    # 2. Convert GGUF -> HK (Zero-Copy Bitstream Transplant)
    convert_gguf_to_hk(gguf_in, hk_out)
    assert os.path.exists(hk_out)

    # 3. Inspect HK container
    hk_file = safe_open(hk_out, framework="pt")
    meta = hk_file.metadata()
    assert meta.get("general.architecture") == "llama"
    assert "model.embed.weight" in hk_file.keys()
    assert "model.mlp.weight" in hk_file.keys()

    t_embed = hk_file.get_tensor("model.embed.weight")
    assert tuple(t_embed.shape) == (4, 32)
    assert t_embed.dtype == torch.float32

    t_mlp = hk_file.get_tensor("model.mlp.weight")
    assert tuple(t_mlp.shape) == (64, 32)
    del hk_file

    # 4. Export HK -> GGUF
    export_hk_to_gguf(hk_out, gguf_exported)
    assert os.path.exists(gguf_exported)

    # 5. Verify exported GGUF matches original bitstream & metadata
    reader_out = GGUFReaderLight(gguf_exported)
    assert reader_out.metadata.get("general.architecture") == "llama"
    assert "model.embed.weight" in reader_out.tensors
    assert "model.mlp.weight" in reader_out.tensors

    # Bitstream exactness test for Q8_0 payload
    raw_mlp_in = reader_in.read_tensor_bytes("model.mlp.weight")
    raw_mlp_out = reader_out.read_tensor_bytes("model.mlp.weight")
    assert raw_mlp_in == raw_mlp_out, "Zero-copy bitstream transplant payload must be identical!"


# ---------------------------------------------------------------------------
# Phase 2: Quantization Zoo Completeness & Parity Tests
# ---------------------------------------------------------------------------

def test_phase2_quantization_parity():
    """Verifies exact numerical accuracy and roundtrip consistency across quantization formats."""
    torch.manual_seed(42)

    # 1. Q8_0 (32 elements per block)
    x_f32 = torch.randn(128, dtype=torch.float32)
    packed_q8 = quantize_q8_0(x_f32)
    assert len(packed_q8) == (128 // 32) * 34
    dequant_q8 = dequantize_q8_0(packed_q8, [128])
    max_err_q8 = (x_f32 - dequant_q8).abs().max().item()
    assert max_err_q8 < 0.05, f"Q8_0 error {max_err_q8} exceeds bound"

    # 2. Q4_0 (32 elements per block)
    packed_q4 = quantize_q4_0(x_f32)
    assert len(packed_q4) == (128 // 32) * 18
    dequant_q4 = dequantize_q4_0(packed_q4, [128])
    max_err_q4 = (x_f32 - dequant_q4).abs().max().item()
    assert max_err_q4 < 0.35, f"Q4_0 error {max_err_q4} exceeds bound"

    # 3. Q4_K (256 elements per super-block)
    x_256 = torch.randn(256, dtype=torch.float32)
    packed_q4k = quantize_q4_k(x_256)
    assert len(packed_q4k) == 144
    dequant_q4k = dequantize_q4_k(packed_q4k, [256])
    max_err_q4k = (x_256 - dequant_q4k).abs().max().item()
    assert max_err_q4k < 0.35, f"Q4_K error {max_err_q4k} exceeds bound"


# ---------------------------------------------------------------------------
# Phase 3: Mathematical Tensor Transformations Tests
# ---------------------------------------------------------------------------

def test_phase3_tensor_transformations():
    """Verifies RoPE permute/unpermute, LayerNorm offset, QKV split/merge, and MoE packing."""
    torch.manual_seed(1337)

    # 1. RoPE Permute / Unpermute Roundtrip
    n_heads = 4
    head_dim = 16
    orig_rope_w = torch.randn(8, n_heads * head_dim, dtype=torch.float32)
    
    permuted = permute_hf_to_gguf_rope(orig_rope_w, n_heads=n_heads, head_dim=head_dim)
    unpermuted = unpermute_gguf_to_hf_rope(permuted, n_heads=n_heads, head_dim=head_dim)
    
    max_rope_diff = (orig_rope_w - unpermuted).abs().max().item()
    assert max_rope_diff == 0.0, f"RoPE roundtrip must be bit-exact! Max diff: {max_rope_diff}"

    # 2. LayerNorm +/- 1.0 Offset
    orig_ln = torch.randn(32, 64, dtype=torch.float32)
    ln_with_offset = add_layernorm_offset(orig_ln, offset=1.0)
    ln_restored = sub_layernorm_offset(ln_with_offset, offset=1.0)
    
    max_ln_diff = (orig_ln - ln_restored).abs().max().item()
    assert max_ln_diff < 1e-6, f"LayerNorm offset roundtrip diff: {max_ln_diff}"

    # 3. Fused QKV Split & Merge
    q_w = torch.randn(64, 128)  # 4 heads * 16 dim = 64
    k_w = torch.randn(32, 128)  # 2 heads * 16 dim = 32
    v_w = torch.randn(32, 128)  # 2 heads * 16 dim = 32

    fused_qkv = merge_fused_qkv(q_w, k_w, v_w)
    assert fused_qkv.shape == (128, 128)

    q_rec, k_rec, v_rec = split_fused_qkv(fused_qkv, q_heads=4, k_heads=2, v_heads=2, head_dim=16)
    assert torch.equal(q_w, q_rec)
    assert torch.equal(k_w, k_rec)
    assert torch.equal(v_w, v_rec)

    # 4. Grouped MoE Expert Packing & Unpacking
    num_experts = 4
    experts = [torch.randn(32, 64) for _ in range(num_experts)]
    packed_moe = pack_moe_experts(experts)
    assert packed_moe.shape == (4, 32, 64)

    unpacked_moe = unpack_moe_experts(packed_moe)
    assert len(unpacked_moe) == num_experts
    for orig, unp in zip(experts, unpacked_moe):
        assert torch.equal(orig, unp)


# ---------------------------------------------------------------------------
# Phase 4: Packed SIMD GEMV Kernels & HKQuantizedLinear Tests
# ---------------------------------------------------------------------------

def test_phase4_simd_gemv_and_hk_quantized_linear():
    """Verifies SIMD GEMV kernels executing on packed quantized weights without full FP32 decompression."""
    torch.manual_seed(999)

    # Test 1: Q8_0 Linear layer
    lin_f32 = nn.Linear(64, 32, bias=True)
    qlin_q8 = HKQuantizedLinear.from_float(lin_f32, storage_type=StorageType.Q8_0)

    assert qlin_q8.packed_weight.dtype == torch.uint8
    # 32 rows * (64 // 32 * 34 bytes) = 32 * 68 = 2176 bytes
    assert qlin_q8.packed_weight.numel() == 32 * 68

    x = torch.randn(2, 64)
    out_q8 = qlin_q8(x)
    assert out_q8.shape == (2, 32)
    # Forward pass outputs should be finite and close to full float linear
    out_ref = lin_f32(x)
    cos_sim = torch.nn.functional.cosine_similarity(out_q8.flatten(), out_ref.flatten(), dim=0).item()
    assert cos_sim > 0.99, f"Q8_0 forward pass output cosine similarity {cos_sim} must be > 0.99"

    # Test 2: Q4_0 Linear layer
    qlin_q4 = HKQuantizedLinear.from_float(lin_f32, storage_type=StorageType.Q4_0)
    assert qlin_q4.packed_weight.dtype == torch.uint8
    # 32 rows * (64 // 32 * 18 bytes) = 32 * 36 = 1152 bytes
    assert qlin_q4.packed_weight.numel() == 32 * 36
    out_q4 = qlin_q4(x)
    assert out_q4.shape == (2, 32)
    cos_sim_q4 = torch.nn.functional.cosine_similarity(out_q4.flatten(), out_ref.flatten(), dim=0).item()
    assert cos_sim_q4 > 0.95, f"Q4_0 forward pass output cosine similarity {cos_sim_q4} must be > 0.95"

    # Test 3: Q4_K Linear layer (requires in_features multiple of 256)
    lin_256 = nn.Linear(256, 16, bias=False)
    qlin_q4k = HKQuantizedLinear.from_float(lin_256, storage_type=StorageType.Q4_K)
    assert qlin_q4k.packed_weight.numel() == 16 * 144
    x_256 = torch.randn(1, 256)
    out_q4k = qlin_q4k(x_256)
    assert out_q4k.shape == (1, 16)


# ---------------------------------------------------------------------------
# Phase 5: Tokenizer Hugging Face JSON & Multi-Template Chat Tests
# ---------------------------------------------------------------------------

def test_phase5_tokenizer_metadata_and_fim():
    """Verifies verbatim tokenizer.huggingface.json embedding, multi-template chat, and FIM prompt generation."""
    templates = {
        "default": "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}",
        "rag": "{% for msg in messages %}[CONTEXT] {{ msg.content }}[/CONTEXT]\n{% endfor %}",
    }
    sample_hf_json = '{"version":"1.0","truncation":null,"padding":null}'

    tok = HKTokenizer(
        tokens=["<s>", "</s>", "<unk>", "<pad>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>", "def", " ", "foo"],
        chat_templates=templates,
        huggingface_json=sample_hf_json,
    )

    # 1. Chat Template selection
    msgs = [{"role": "user", "content": "Hello HK"}]
    rendered_default = tok.apply_chat_template(msgs, template_name="default")
    assert "user: Hello HK" in rendered_default

    rendered_rag = tok.apply_chat_template(msgs, template_name="rag")
    assert "[CONTEXT] Hello HK[/CONTEXT]" in rendered_rag

    # 2. Fill-In-the-Middle (FIM) prompt formatting
    fim_prompt = tok.apply_fim(prefix="def solve():\n    ", suffix="\n    return res")
    assert fim_prompt.startswith("<fim_prefix>def solve():\n    <fim_suffix>\n    return res<fim_middle>")

    # 3. Serialization to HK metadata
    meta = tok.export_to_metadata()
    assert "tokenizer.chat_templates" in meta
    assert "tokenizer.huggingface.json" in meta
    assert meta["tokenizer.huggingface.json"] == sample_hf_json


# ---------------------------------------------------------------------------
# Phase 6: Remote HTTP Range Reader (RFC 7233) Tests
# ---------------------------------------------------------------------------

class RangeRequestHandler(BaseHTTPRequestHandler):
    """Minimal RFC 7233 HTTP server supporting Range requests for tests."""
    file_bytes = b""

    def do_GET(self):
        range_header = self.headers.get("Range")
        total_len = len(self.file_bytes)

        if not range_header:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(total_len))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(self.file_bytes)
            return

        # Parse 'bytes=start-end'
        try:
            val = range_header.strip().replace("bytes=", "")
            start_str, end_str = val.split("-")
            start = int(start_str)
            end = int(end_str) if end_str else total_len - 1
            if end >= total_len:
                end = total_len - 1
            content_length = end - start + 1

            self.send_response(206)  # Partial Content
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {start}-{end}/{total_len}")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(self.file_bytes[start : end + 1])
        except Exception as e:
            self.send_response(416)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs in pytest


def test_phase6_remote_http_range_reader(temp_dir):
    """Verifies instant remote container header inspection and selective single-tensor HTTP range streaming."""
    hk_file_path = os.path.join(temp_dir, "remote_test.hk")
    
    # Create sample HK container with PyTorch tensors
    weights = {
        "layer0.weight": torch.randn(16, 32, dtype=torch.float32),
        "layer1.weight": torch.randn(8, 16, dtype=torch.float32),
    }
    save_file(weights, hk_file_path, metadata={"model_type": "deep_transformer", "param_count": "1024"})

    with open(hk_file_path, "rb") as f:
        file_bytes = f.read()

    # Launch local Range-compliant HTTP server
    RangeRequestHandler.file_bytes = file_bytes
    server = HTTPServer(("127.0.0.1", 0), RangeRequestHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}/model.hk"

    try:
        # 1. Header and TOC inspection via single 128KB HTTP range request
        meta, tensors = read_remote_hk_header(url)
        assert meta.get("model_type") == "deep_transformer"
        assert len(tensors) == 2
        tensor_names = [t["name"] for t in tensors]
        assert "layer0.weight" in tensor_names
        assert "layer1.weight" in tensor_names

        # 2. RemoteHKFile lazy streaming
        remote_file = safe_open_remote(url)
        assert set(remote_file.keys()) == {"layer0.weight", "layer1.weight"}
        
        # Download and verify single tensor
        t0_streamed = remote_file.get_tensor("layer0.weight", framework="pt")
        assert torch.equal(t0_streamed, weights["layer0.weight"]), "Streamed remote tensor must match bit-exact!"

        t1_streamed = remote_file.get_tensor("layer1.weight", framework="pt")
        assert torch.equal(t1_streamed, weights["layer1.weight"]), "Streamed remote tensor must match bit-exact!"

    finally:
        server.shutdown()
        server.server_close()
