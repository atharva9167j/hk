"""
Unit Tests for New HK Features:
1. In-File Tokenizer Metadata (AutoTokenizer reconstruction from .hk)
2. Jinja2 Chat Templates (apply_chat_template)
3. Multi-File Sharding Specification (split_index, split_count, IS_SHARDED)
4. In-Place Metadata Patching Utility (Zero-copy update of key-value table)
5. Hugging Face Architecture Mapping Layer (Llama/Qwen2/Mistral conversion)
6. Multi-Model Combined Pipelines & Context Controls (Whisper -> DistilBERT -> LLM)
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

import hk
from hk.tokenizer import HKTokenizer, AutoTokenizer
from hk.torch import (
    save_file,
    load_file,
    save_sharded_file,
    load_sharded_file,
    metadata_set,
    safe_open,
)
from hk.hf_mapper import (
    HFArchitectureMapper,
    extract_hf_tokenizer_metadata,
    convert_hf_checkpoint,
)
from hk.composite import (
    ContextWindowManager,
    HKWhisperModel,
    HKDistilBertModel,
    CompositePipeline,
)


class TestHKNewFeatures(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="hk_test_features_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 1. In-File Tokenizer Metadata
    # -----------------------------------------------------------------------
    def test_in_file_tokenizer_metadata(self):
        """Test AutoTokenizer reconstruction from pure .hk metadata without external .json files."""
        vocab = ["<pad>", "<s>", "</s>", "<unk>", "hello", "world", "hk", "neural", "tensor", "format", " "]
        scores = [0.0, 1.0, 1.0, 0.5, 2.5, 3.1, 4.0, 3.5, 2.0, 1.8, 0.0]
        merges = ["h e", "he l", "hel lo", "w o", "wo r", "wor ld"]

        tok = HKTokenizer(
            vocab=vocab,
            scores=scores,
            merges=merges,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        hk_path = os.path.join(self.test_dir, "tokenizer_model.hk")
        metadata = tok.export_to_metadata()
        dummy_tensors = {"dummy_param": torch.randn(4, 4)}

        # Save into .hk container
        save_file(dummy_tensors, hk_path, metadata=metadata)

        # Ensure no .json files exist alongside
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "tokenizer.json")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "vocab.json")))

        # Reconstruct directly from .hk file
        reconstructed_tok = AutoTokenizer.from_pretrained(hk_path)

        # Validate vocabulary and special tokens
        self.assertEqual(reconstructed_tok.vocab_size, len(vocab))
        self.assertEqual(reconstructed_tok.bos_token, "<s>")
        self.assertEqual(reconstructed_tok.eos_token, "</s>")
        self.assertEqual(reconstructed_tok.unk_token, "<unk>")
        expected_merges = [tuple(m.split()) for m in merges]
        self.assertEqual(reconstructed_tok.merges, expected_merges)

        # Validate encoding and decoding round-trip
        encoded = reconstructed_tok.encode("hello world", add_special_tokens=True)
        self.assertEqual(encoded[0], reconstructed_tok.bos_token_id)
        self.assertEqual(encoded[-1], reconstructed_tok.eos_token_id)

        decoded = reconstructed_tok.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, "hello world")

    # -----------------------------------------------------------------------
    # 2. Standardized Jinja2 Chat Templates
    # -----------------------------------------------------------------------
    def test_jinja2_chat_template(self):
        """Test Jinja2 chat template storage in HK metadata and apply_chat_template formatting."""
        template_str = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

        tok = HKTokenizer(
            vocab=["<|im_start|>", "<|im_end|>", "system", "user", "assistant"],
            chat_template=template_str,
        )

        hk_path = os.path.join(self.test_dir, "chat_model.hk")
        save_file({"weight": torch.ones(2, 2)}, hk_path, metadata=tok.export_to_metadata())

        # Reload from .hk
        loaded_tok = AutoTokenizer.from_pretrained(hk_path)
        self.assertEqual(loaded_tok.chat_template, template_str)

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is HK?"},
        ]

        formatted = loaded_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        expected = (
            "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is HK?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(formatted, expected)

        # Test with tokenize=True
        token_ids = loaded_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)

    # -----------------------------------------------------------------------
    # 3. Multi-File Sharding Specification
    # -----------------------------------------------------------------------
    def test_multifile_sharding_spec(self):
        """Test multi-file sharding creation, header flags, split_index, split_count, and transparent loading."""
        # Create weights that exceed a small threshold
        state_dict = {
            f"layer_{i}.weight": torch.randn(64, 64, dtype=torch.float32)
            for i in range(8)
        }

        base_hk = os.path.join(self.test_dir, "sharded_model.hk")
        # Each tensor is 64*64*4 = 16,384 bytes. Set shard limit to ~35KB so it creates ~4 shards.
        shard_paths = save_sharded_file(
            state_dict=state_dict,
            base_path=base_hk,
            max_shard_size_bytes=35 * 1024,
            metadata={"test_meta": "sharded_v1"},
        )

        self.assertGreater(len(shard_paths), 1)
        split_count = len(shard_paths)

        # Inspect each shard's header
        for idx, s_path in enumerate(shard_paths):
            with safe_open(s_path, framework="pt") as f:
                self.assertTrue(f.is_sharded, f"Shard {s_path} should have IS_SHARDED flag set")
                self.assertEqual(f.split_index, idx, f"Shard {s_path} split_index mismatch")
                self.assertEqual(f.split_count, split_count, f"Shard {s_path} split_count mismatch")

        # Test loading via explicit shard list
        loaded_dict = load_sharded_file(shard_paths)
        self.assertEqual(len(loaded_dict), len(state_dict))
        for k in state_dict:
            self.assertTrue(torch.allclose(state_dict[k], loaded_dict[k]))

        # Test loading via primary shard filename with transparent companion discovery
        auto_loaded = load_file(shard_paths[0])
        self.assertEqual(len(auto_loaded), len(state_dict))
        for k in state_dict:
            self.assertTrue(torch.allclose(state_dict[k], auto_loaded[k]))

    # -----------------------------------------------------------------------
    # 4. In-Place Metadata Patching Utility
    # -----------------------------------------------------------------------
    def test_inplace_metadata_patching(self):
        """Test zero-copy in-place metadata update without modifying tensor payload."""
        file_path = os.path.join(self.test_dir, "patch_target.hk")
        original_tensors = {
            "fc1.weight": torch.randn(16, 16),
            "fc2.weight": torch.randn(16, 8),
        }
        initial_meta = {
            "initial_key": "initial_value",
            "model_version": "1.0.0",
        }

        save_file(original_tensors, file_path, metadata=initial_meta)

        # Record file size and tensor data to verify zero payload alteration
        file_sz_before = os.path.getsize(file_path)

        # Update metadata in-place
        metadata_set(file_path, "model_version", "2.1.0")
        metadata_set(file_path, "patched_entry", "active_patch")

        # Verify updated metadata
        with safe_open(file_path, framework="pt") as f:
            meta = f.metadata()
            self.assertEqual(meta.get("initial_key"), "initial_value")
            self.assertEqual(meta.get("model_version"), "2.1.0")
            self.assertEqual(meta.get("patched_entry"), "active_patch")

            # Verify tensors are 100% intact
            t1 = f.get_tensor("fc1.weight")
            t2 = f.get_tensor("fc2.weight")
            self.assertTrue(torch.allclose(t1, original_tensors["fc1.weight"]))
            self.assertTrue(torch.allclose(t2, original_tensors["fc2.weight"]))

    # -----------------------------------------------------------------------
    # 5. Hugging Face Architecture Mapping Layer
    # -----------------------------------------------------------------------
    def test_hf_architecture_mapping_and_conversion(self):
        """Test HF Llama architecture detection, state dict translation, and checkpoint conversion."""
        hf_dir = Path(self.test_dir) / "mock_llama_hf"
        hf_dir.mkdir()

        # Write HF config.json
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 1000,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "_name_or_path": "meta-llama/Llama-2-7b",
        }
        with open(hf_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(hf_config, f)

        # Write tokenizer.json
        tok_data = {
            "model": {
                "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "llama": 3, "hk": 4},
                "merges": ["l l", "ll a", "lla ma"],
            }
        }
        with open(hf_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tok_data, f)

        # Write weights using safetensors
        import safetensors.torch
        hf_weights = {
            "model.embed_tokens.weight": torch.randn(1000, 128),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 128),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 128),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 128),
            "model.layers.0.mlp.up_proj.weight": torch.randn(256, 128),
            "model.layers.0.mlp.down_proj.weight": torch.randn(128, 256),
            "model.norm.weight": torch.ones(128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        safetensors.torch.save_file(hf_weights, str(hf_dir / "model.safetensors"))

        # Test AutoConfig detects and maps HF directory
        auto_cfg = hk.AutoConfig.from_pretrained(str(hf_dir))
        self.assertEqual(auto_cfg.architecture, "llama")
        self.assertEqual(auto_cfg.hidden_size, 128)
        self.assertEqual(auto_cfg.num_hidden_layers, 2)
        self.assertEqual(auto_cfg.num_attention_heads, 4)

        # Test convert_hf_checkpoint with canonical tensor renaming
        output_hk = os.path.join(self.test_dir, "converted_llama.hk")
        converted_file = convert_hf_checkpoint(
            hf_model_dir=str(hf_dir),
            output_hk_path=output_hk,
            canonical_tensor_names=True,
        )
        self.assertTrue(os.path.exists(converted_file))

        # Inspect converted container
        with safe_open(converted_file, framework="pt") as f:
            keys = f.keys()
            self.assertIn("embed_tokens.weight", keys)
            self.assertIn("layers.0.attn_q.weight", keys)
            self.assertIn("layers.0.mlp_gate.weight", keys)
            self.assertIn("lm_head.weight", keys)

            # Check tokenizer was preserved in metadata
            meta = f.metadata()
            self.assertIn("tokenizer.tokens", meta)
            self.assertIn("tokenizer.merges", meta)

    # -----------------------------------------------------------------------
    # 6. Multi-Model Combined Pipelines & Context Controls
    # -----------------------------------------------------------------------
    def test_context_window_manager(self):
        """Test ContextWindowManager retention strategies (tail, head, middle_out) and RoPE scaling."""
        mgr = ContextWindowManager(max_context_length=10, default_strategy="middle_out", head_ratio=0.3)
        tokens = list(range(20))  # [0, 1, ..., 19]

        # Tail strategy: keep last 10
        tail_tokens = mgr.truncate(tokens, max_tokens=10, strategy="tail")
        self.assertEqual(tail_tokens, list(range(10, 20)))

        # Head strategy: keep first 10
        head_tokens = mgr.truncate(tokens, max_tokens=10, strategy="head")
        self.assertEqual(head_tokens, list(range(0, 10)))

        # Middle-out strategy: keep 3 head + 7 tail
        middle_tokens = mgr.truncate(tokens, max_tokens=10, strategy="middle_out", head_ratio=0.3)
        self.assertEqual(len(middle_tokens), 10)
        self.assertEqual(middle_tokens[:3], [0, 1, 2])
        self.assertEqual(middle_tokens[3:], list(range(13, 20)))

        # RoPE scaling
        cos = torch.ones(1, 1, 64, 32)
        sin = torch.zeros(1, 1, 64, 32)
        scaled_cos, scaled_sin = mgr.apply_rope_scaling(cos, sin, scale_factor=2.0)
        self.assertEqual(scaled_cos.shape, cos.shape)
        self.assertEqual(scaled_sin.shape, sin.shape)

    def test_composite_multi_model_pipeline(self):
        """Test end-to-end multi-model chaining (Whisper -> DistilBERT -> LLM) with per-stage params."""
        whisper = HKWhisperModel()
        analyzer = HKDistilBertModel()
        llm_cfg = hk.HKConfig(model_type="causal_lm", hidden_size=64, num_hidden_layers=2, num_attention_heads=2)
        llm = hk.HKForCausalLM(llm_cfg)
        tok = HKTokenizer(vocab=["<pad>", "<s>", "</s>", "hello", "robot", "start", "engine", "query"])

        composite = CompositePipeline(
            whisper_model=whisper,
            analyzer_model=analyzer,
            llm_model=llm,
            tokenizer=tok,
        )

        # Configure per-model parameters
        composite.set_stage_param("whisper", temperature=0.0)
        composite.set_stage_param("analyzer", return_top_k=2)
        composite.set_stage_param("llm", temperature=0.6, max_new_tokens=15)

        self.assertEqual(composite.get_stage_param("whisper")["temperature"], 0.0)
        self.assertEqual(composite.get_stage_param("llm")["max_new_tokens"], 15)

        # Create synthetic audio waveform (1 second at 16kHz)
        audio = torch.randn(16000)

        # Run composite pipeline end-to-end
        result = composite(audio=audio, system_prompt="Answer concisely.")

        self.assertIn("whisper", result["stages_executed"])
        self.assertIn("analyzer", result["stages_executed"])
        self.assertIn("llm", result["stages_executed"])
        self.assertIsNotNone(result["transcription"])
        self.assertIn("intent", result["analysis"])
        self.assertIn("sentiment", result["analysis"])
        self.assertIsInstance(result["generated_text"], str)

        # Test high-level pipeline factory registration
        high_level_pipe = hk.pipeline(
            "composite",
            whisper_model=whisper,
            analyzer_model=analyzer,
            llm_model=llm,
            tokenizer=tok,
        )
        self.assertIsInstance(high_level_pipe, CompositePipeline)

        # Test composite pipeline serialization
        save_dir = os.path.join(self.test_dir, "saved_composite")
        manifest_path = composite.save_composite(save_dir)
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(manifest_data["pipeline_type"], "composite")
        self.assertIn("whisper", manifest_data["stages"])
        self.assertIn("analyzer", manifest_data["stages"])
        self.assertIn("llm", manifest_data["stages"])


if __name__ == "__main__":
    unittest.main()
