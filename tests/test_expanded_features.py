"""
Comprehensive Verification Suite for:
1. In-File Tokenizer Metadata & AutoTokenizer in-file reconstruction.
2. Standardized Multi-File Sharding Index (model.hk.index.json, in-header split indices, ShardedHKFile).
3. Framework Bindings Expansion: NumPy (hk.numpy), JAX (hk.jax), Flax (hk.flax), multi-framework safe_open.
4. Architecture Conversion Tables: LlamaForCausalLM, Qwen2ForCausalLM (with QKV biases), MistralForCausalLM (with sliding_window).
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import pytest

import numpy as np
import torch

import hk
from hk import (
    HKConfig,
    AutoConfig,
    HKTokenizer,
    AutoTokenizer,
    HKForCausalLM,
    save_file,
    load_file,
    save_sharded_file,
    load_sharded_file,
    safe_open,
    ShardedHKFile,
    HFArchitectureMapper,
    convert_hf_checkpoint,
)
import hk.numpy
import hk.jax
import hk.flax


class TestExpandedFeatures(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="hk_expanded_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 1. In-File Tokenizer Metadata
    # -----------------------------------------------------------------------
    def test_in_file_tokenizer_metadata_standardization(self):
        """Verify standardized metadata keys: tokenizer.tokens, tokenizer.scores, tokenizer.merges, tokenizer.chat_template."""
        tokens = ["<pad>", "<s>", "</s>", "<unk>", "deep", "mind", "anti", "gravity", "tensor", " "]
        scores = [0.0, 1.0, 1.0, 0.5, 2.2, 3.1, 4.0, 3.8, 2.9, 0.0]
        merges = ["d e", "de ep", "m i", "mi nd", "an ti"]
        chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"

        tok = HKTokenizer(
            tokens=tokens,
            scores=scores,
            merges=merges,
            chat_template=chat_template,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        meta = tok.export_to_metadata()

        # Check required standardized keys
        self.assertIn("tokenizer.tokens", meta)
        self.assertIn("tokenizer.scores", meta)
        self.assertIn("tokenizer.merges", meta)
        self.assertIn("tokenizer.chat_template", meta)
        self.assertIn("tokenizer.bos_token", meta)
        self.assertIn("tokenizer.eos_token", meta)

        hk_path = os.path.join(self.test_dir, "model_with_tokenizer.hk")
        save_file({"weight": torch.randn(4, 4)}, hk_path, metadata=meta)

        # Reconstruct directly using AutoTokenizer from .hk container
        reconstructed = AutoTokenizer.from_pretrained(hk_path)
        self.assertEqual(reconstructed.tokens, tokens)
        self.assertEqual(reconstructed.scores, scores)
        self.assertEqual(reconstructed.chat_template, chat_template)
        self.assertEqual(reconstructed.bos_token, "<s>")
        self.assertEqual(reconstructed.eos_token, "</s>")

        # Encode and decode roundtrip
        encoded = reconstructed.encode("deep mind", add_special_tokens=True)
        self.assertEqual(encoded[0], reconstructed.bos_token_id)
        self.assertEqual(encoded[-1], reconstructed.eos_token_id)
        decoded = reconstructed.decode(encoded)
        self.assertEqual(decoded, "deep mind")

    def test_model_save_pretrained_with_embedded_tokenizer(self):
        """Verify model.save_pretrained can embed tokenizer directly in the .hk container."""
        cfg = HKConfig(model_type="causal_lm", vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2)
        model = HKForCausalLM(cfg)

        tok = HKTokenizer(tokens=["<pad>", "<s>", "</s>", "<unk>", "hello", "world"])
        save_hk_path = os.path.join(self.test_dir, "self_contained.hk")

        model.save_pretrained(save_hk_path, tokenizer=tok)
        self.assertTrue(os.path.isfile(save_hk_path))

        # Reconstruct tokenizer from the self-contained container
        loaded_tok = AutoTokenizer.from_pretrained(save_hk_path)
        self.assertEqual(loaded_tok.tokens[:6], ["<pad>", "<s>", "</s>", "<unk>", "hello", "world"])

    # -----------------------------------------------------------------------
    # 2. Standardized Multi-File Sharding Index (model.hk.index.json)
    # -----------------------------------------------------------------------
    def test_sharding_index_manifest_generation_and_loading(self):
        """Verify model.hk.index.json generation with weight_map and transparent sharded loading."""
        tensors = {
            f"layer_{i}.weight": torch.randn(32, 32, dtype=torch.float32)
            for i in range(12)
        }

        base_model_path = os.path.join(self.test_dir, "model.hk")
        # 32*32*4 = 4096 bytes per tensor. Setting shard size to 10KB splits across ~5 shards.
        shard_paths = save_sharded_file(
            state_dict=tensors,
            base_path=base_model_path,
            max_shard_size_bytes=10 * 1024,
            metadata={"architecture": "llama", "model_family": "test_70b"},
        )

        self.assertGreater(len(shard_paths), 1)

        # Verify model.hk.index.json was created
        index_json_path = os.path.join(self.test_dir, "model.hk.index.json")
        self.assertTrue(os.path.isfile(index_json_path), f"Expected index manifest at {index_json_path}")

        with open(index_json_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.assertIn("metadata", index_data)
        self.assertIn("weight_map", index_data)
        self.assertEqual(index_data["metadata"]["split_count"], len(shard_paths))
        self.assertEqual(index_data["metadata"]["format"], "hk")
        self.assertEqual(len(index_data["weight_map"]), len(tensors))

        # Test loading via index manifest directly
        loaded_from_index = load_file(index_json_path)
        self.assertEqual(len(loaded_from_index), len(tensors))
        for k in tensors:
            self.assertTrue(torch.allclose(tensors[k], loaded_from_index[k]))

        # Test loading via directory path
        loaded_from_dir = load_file(self.test_dir)
        self.assertEqual(len(loaded_from_dir), len(tensors))

        # Test lazy inspection and slicing with safe_open on index.json
        with safe_open(index_json_path, framework="pt") as f:
            self.assertIsInstance(f, ShardedHKFile)
            self.assertEqual(len(f.keys()), len(tensors))
            self.assertEqual(f.metadata().get("architecture"), "llama")

            # Lazy get_tensor
            t0 = f.get_tensor("layer_0.weight")
            self.assertTrue(torch.allclose(t0, tensors["layer_0.weight"]))

            # Lazy sliced access
            slice_tensor = f.get_slice("layer_0.weight")[0:4, 0:8]
            self.assertEqual(slice_tensor.shape, (4, 8))
            self.assertTrue(torch.allclose(slice_tensor, tensors["layer_0.weight"][0:4, 0:8]))

    # -----------------------------------------------------------------------
    # 3. Framework Bindings Expansion (NumPy, JAX, Flax)
    # -----------------------------------------------------------------------
    def test_numpy_bindings_save_and_load(self):
        """Verify hk.numpy save_file and load_file with multiple dtypes."""
        np_tensors = {
            "arr_f32": np.random.randn(8, 8).astype(np.float32),
            "arr_f16": np.random.randn(4, 4).astype(np.float16),
            "arr_i32": np.array([1, 2, 3, -10], dtype=np.int32),
            "arr_i64": np.array([1000000000, -2000000000], dtype=np.int64),
            "arr_bool": np.array([True, False, True], dtype=np.bool_),
        }

        save_path = os.path.join(self.test_dir, "numpy_model.hk")
        hk.numpy.save_file(np_tensors, save_path, metadata={"library": "numpy"})

        loaded = hk.numpy.load_file(save_path)
        for k, v in np_tensors.items():
            self.assertIn(k, loaded)
            self.assertEqual(loaded[k].dtype, v.dtype)
            np.testing.assert_array_equal(loaded[k], v)

        # Test safe_open with framework="np"
        with safe_open(save_path, framework="np") as f:
            arr = f.get_tensor("arr_f32")
            self.assertIsInstance(arr, np.ndarray)
            sliced = f.get_slice("arr_f32")[0:2, 0:4]
            self.assertIsInstance(sliced, np.ndarray)
            self.assertEqual(sliced.shape, (2, 4))

    def test_jax_and_flax_bindings(self):
        """Verify hk.jax and hk.flax array and model serialization/deserialization."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        flax = pytest.importorskip("flax")
        from flax.core import freeze

        # JAX arrays
        jax_tensors = {
            "w1": jnp.ones((4, 4), dtype=jnp.float32) * 3.14,
            "b1": jnp.zeros((4,), dtype=jnp.float32),
        }

        jax_path = os.path.join(self.test_dir, "jax_model.hk")
        hk.jax.save_file(jax_tensors, jax_path, metadata={"backend": "jax"})

        loaded_jax = hk.jax.load_file(jax_path)
        for k, v in jax_tensors.items():
            self.assertIn(k, loaded_jax)
            self.assertTrue(isinstance(loaded_jax[k], jax.Array))
            np.testing.assert_allclose(np.array(loaded_jax[k]), np.array(v), rtol=1e-5)

        # safe_open with framework="jax"
        with safe_open(jax_path, framework="jax") as f:
            arr = f.get_tensor("w1")
            self.assertTrue(isinstance(arr, jax.Array))
            sliced = f.get_slice("w1")[:2, :2]
            self.assertTrue(isinstance(sliced, jax.Array))
            self.assertEqual(sliced.shape, (2, 2))

        # Flax nested model variables
        flax_params = freeze({
            "params": {
                "Dense_0": {
                    "kernel": jnp.ones((8, 16), dtype=jnp.float32) * 0.5,
                    "bias": jnp.zeros((16,), dtype=jnp.float32),
                },
                "Dense_1": {
                    "kernel": jnp.ones((16, 4), dtype=jnp.float32) * 0.25,
                    "bias": jnp.ones((4,), dtype=jnp.float32),
                },
            }
        })

        flax_path = os.path.join(self.test_dir, "flax_model.hk")
        hk.flax.save_model(flax_params, flax_path, metadata={"framework": "flax"})

        loaded_flax = hk.flax.load_model(flax_params, flax_path)
        self.assertIn("params", loaded_flax)
        self.assertIn("Dense_0", loaded_flax["params"])
        self.assertIn("kernel", loaded_flax["params"]["Dense_0"])
        np.testing.assert_allclose(
            np.array(loaded_flax["params"]["Dense_0"]["kernel"]),
            np.array(flax_params["params"]["Dense_0"]["kernel"]),
        )

    # -----------------------------------------------------------------------
    # 4. Architecture Conversion Tables (Llama, Qwen2, Mistral)
    # -----------------------------------------------------------------------
    def test_architecture_conversion_tables_and_mappings(self):
        """Verify conversion tables for LlamaForCausalLM, Qwen2ForCausalLM, and MistralForCausalLM."""
        # Check conversion table existence for all 3 architectures
        for arch in ["LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"]:
            table = HFArchitectureMapper.get_conversion_table(arch)
            self.assertIn("config", table)
            self.assertIn("forward_tensor_patterns", table)
            self.assertIn("reverse_tensor_patterns", table)

        # 1. Llama mapping
        llama_name = "model.layers.3.self_attn.q_proj.weight"
        hk_llama = HFArchitectureMapper.map_tensor_name_to_hk(llama_name, architecture="llama")
        self.assertEqual(hk_llama, "layers.3.attn_q.weight")
        rev_llama = HFArchitectureMapper.map_tensor_name_to_hf(hk_llama, architecture="llama")
        self.assertEqual(rev_llama, llama_name)

        # 2. Qwen2 mapping (including attention biases!)
        qwen_bias = "model.layers.5.self_attn.k_proj.bias"
        hk_qwen = HFArchitectureMapper.map_tensor_name_to_hk(qwen_bias, architecture="qwen2")
        self.assertEqual(hk_qwen, "layers.5.attn_k.bias")
        rev_qwen = HFArchitectureMapper.map_tensor_name_to_hf(hk_qwen, architecture="qwen2")
        self.assertEqual(rev_qwen, qwen_bias)

        # 3. Mistral mapping (including sliding window config mapping)
        mistral_config = {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "sliding_window": 4096,
            "rope_theta": 10000.0,
        }
        mapped_cfg = HFArchitectureMapper.map_config(mistral_config)
        self.assertEqual(mapped_cfg.architecture, "mistral")
        self.assertEqual(mapped_cfg.intermediate_size, 14336)
        self.assertEqual(getattr(mapped_cfg, "sliding_window", None), 4096)

        # 4. Qwen2 checkpoint conversion with biases
        qwen_dir = Path(self.test_dir) / "mock_qwen2_hf"
        qwen_dir.mkdir()
        with open(qwen_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 100,
            }, f)

        import safetensors.torch
        qwen_weights = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.q_proj.bias": torch.randn(64),
            "model.layers.0.self_attn.k_proj.bias": torch.randn(64),
            "model.layers.0.self_attn.v_proj.bias": torch.randn(64),
            "model.norm.weight": torch.ones(64),
            "lm_head.weight": torch.randn(100, 64),
        }
        safetensors.torch.save_file(qwen_weights, str(qwen_dir / "model.safetensors"))

        output_hk = os.path.join(self.test_dir, "converted_qwen2.hk")
        convert_hf_checkpoint(
            hf_model_dir=str(qwen_dir),
            output_hk_path=output_hk,
            canonical_tensor_names=True,
        )

        with safe_open(output_hk, framework="pt") as f:
            keys = f.keys()
            self.assertIn("layers.0.attn_q.bias", keys)
            self.assertIn("layers.0.attn_k.bias", keys)
            self.assertIn("layers.0.attn_v.bias", keys)
            self.assertIn("embed_tokens.weight", keys)


if __name__ == "__main__":
    unittest.main()
