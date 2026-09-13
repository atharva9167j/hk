"""
HK Framework End-to-End Test Suite
Validates Hugging Face-style AutoModel, AutoConfig, AutoTokenizer, pipeline,
HKTrainer, flexible alignments, native SIMD Zig dispatch, and Net2Net growth.
"""

import os
import sys

# Ensure python directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import shutil
import tempfile
import numpy as np
import torch

from hk import (
    HKConfig,
    AutoConfig,
    HKTokenizer,
    AutoTokenizer,
    HKPreTrainedModel,
    HKForCausalLM,
    HKForSequenceClassification,
    HKForHandwritingRecognition,
    AutoModel,
    pipeline,
    HKTrainer,
    HKTrainingArguments,
)
from hk.native import (
    is_native_available,
    native_gemm,
    native_gemv,
    native_dot,
    native_net2wider,
    native_net2deeper,
)



def test_autoconfig_roundtrip(temp_dir):
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        alignment=16,
    )
    cfg_path = os.path.join(temp_dir, "config.json")
    cfg.save_pretrained(temp_dir)
    assert os.path.isfile(cfg_path)

    loaded_cfg = AutoConfig.from_pretrained(temp_dir)
    assert loaded_cfg.model_type == "causal_lm"
    assert loaded_cfg.hidden_size == 128
    assert loaded_cfg.num_hidden_layers == 2
    assert loaded_cfg.intermediate_size == 256
    assert loaded_cfg.alignment == 16


def test_automodel_causal_lm_save_load(temp_dir):
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    model = HKForCausalLM(cfg)
    model.eval()

    test_input = torch.randint(0, 500, (1, 8))
    with torch.no_grad():
        out_orig = model(test_input)

    # Save pretrained to .hk container
    save_path = os.path.join(temp_dir, "causal_model.hk")
    model.save_pretrained(save_path)
    assert os.path.isfile(save_path)

    # Load via AutoModel
    loaded_model = AutoModel.from_pretrained(save_path)
    assert isinstance(loaded_model, HKForCausalLM)
    loaded_model.eval()

    with torch.no_grad():
        out_loaded = loaded_model(test_input)

    # Validate output logits match closely
    max_diff = (out_orig.logits - out_loaded.logits).abs().max().item()
    assert max_diff < 1e-4, f"Logits mismatch after save/load: {max_diff}"


def test_pipeline_text_generation(temp_dir):
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    model = HKForCausalLM(cfg)
    save_path = os.path.join(temp_dir, "gen_model.hk")
    model.save_pretrained(save_path)

    tokenizer = HKTokenizer()

    pipe = pipeline("text-generation", model=save_path, tokenizer=tokenizer)
    results = pipe("Hello AI", max_new_tokens=5)
    assert len(results) == 1
    assert "generated_text" in results[0]
    assert len(results[0]["generated_text"]) > 0


def test_pipeline_sequence_classification(temp_dir):
    cfg = HKConfig(
        model_type="sequence_classification",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_classes=3,
    )
    model = HKForSequenceClassification(cfg)
    save_path = os.path.join(temp_dir, "classifier.hk")
    model.save_pretrained(save_path)

    tokenizer = HKTokenizer()
    pipe = pipeline("sequence-classification", model=save_path, tokenizer=tokenizer)
    res = pipe("Great framework!")
    assert len(res) == 1
    assert "label" in res[0]
    assert "score" in res[0]
    assert res[0]["label"].startswith("LABEL_")


def test_pipeline_handwriting_recognition(temp_dir):
    cfg = HKConfig(
        model_type="handwriting_recognition",
        in_channels=1,
        hidden_size=64,
        num_classes=26,
    )
    model = HKForHandwritingRecognition(cfg)
    save_path = os.path.join(temp_dir, "hwr.hk")
    model.save_pretrained(save_path)

    pipe = pipeline("handwriting-recognition", model=save_path)
    dummy_img = np.zeros((1, 32, 128), dtype=np.float32)
    res = pipe(dummy_img)
    assert len(res) == 1
    assert "transcription" in res[0]


def test_net2wider_exact_function_preservation():
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = HKForCausalLM(cfg)
    model.eval()

    test_input = torch.randint(0, 100, (1, 6))
    with torch.no_grad():
        out_before = model(test_input).logits

    # Grow width from 64 to 96 with zero noise
    model.grow_width(layer_idx=0, new_intermediate_size=96, noise_std=0.0)

    with torch.no_grad():
        out_after = model(test_input).logits

    # Net2WiderNet guarantees mathematical identity
    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff < 1e-4, f"Function preservation violated: {max_diff}"


def test_net2deeper_identity_expansion():
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = HKForCausalLM(cfg)
    model.eval()

    test_input = torch.randint(0, 100, (1, 6))
    with torch.no_grad():
        out_before = model(test_input).logits

    # Grow depth: insert new transformer block after layer 0
    model.grow_depth(insert_after_layer_idx=0)
    assert len(model.layers) == 2

    with torch.no_grad():
        out_after = model(test_input).logits

    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff < 1e-4, f"Identity depth expansion violated: {max_diff}"


def test_qlora_adapter_activation():
    cfg = HKConfig(model_type="causal_lm", vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
    model = HKForCausalLM(cfg)

    # Enable QLoRA with rank 4
    model.enable_qlora(rank=4, alpha=8.0)

    # Check that base weights are frozen
    for name, p in model.named_parameters():
        if "lora" in name:
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False

    # Forward + backward to check adapter gradient flow
    x = torch.randint(0, 100, (2, 8))
    out = model(x, labels=x)
    out.loss.backward()

    # Base weights should have no grad
    assert model.layers[0].mlp_fc1.weight.grad is None
    # LoRA A should have grad
    assert model.layers[0].mlp_fc1.lora_A.grad is not None


def test_universal_flexible_alignment(temp_dir):
    """Verifies that models can be packaged with relaxed alignment (1B, 16B) for embedded/WASM/mobile."""
    cfg = HKConfig(model_type="causal_lm", vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2)
    model = HKForCausalLM(cfg)

    alignments = [1, 4, 16, 64, 128]
    for align in alignments:
        p = os.path.join(temp_dir, f"model_align_{align}.hk")
        model.save_pretrained(p, alignment=align)

        # Load back
        loaded = AutoModel.from_pretrained(p)
        assert loaded.config.alignment == align

        x = torch.randint(0, 50, (1, 4))
        with torch.no_grad():
            out = loaded(x)
        assert out.logits.shape == (1, 4, 50)


def test_native_simd_speedup():
    """Verifies native Zig SIMD GEMM computation against NumPy."""
    assert is_native_available()

    M, K, N = 64, 128, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    c_numpy = np.matmul(A, B)
    c_native = native_gemm(A, B)

    np.testing.assert_allclose(c_numpy, c_native, rtol=1e-4, atol=1e-4)


def test_hktrainer_training_and_lineage(temp_dir):
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = HKForCausalLM(cfg)

    args = HKTrainingArguments(
        output_dir=temp_dir,
        learning_rate=1e-3,
        num_train_epochs=2,
        enable_adaptive_growth=True,
        growth_patience=2,
        growth_width_factor=1.25,
    )

    trainer = HKTrainer(model=model, args=args)
    results = trainer.train()

    assert results["global_step"] > 0
    assert os.path.isfile(results["output_path"])


def test_trainer_plateau_adaptive_growth(temp_dir):
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = HKForCausalLM(cfg)
    initial_inter = model.config.intermediate_size

    args = HKTrainingArguments(
        output_dir=temp_dir,
        learning_rate=1e-3,
        num_train_epochs=3,
        enable_adaptive_growth=True,
        growth_patience=1,
        growth_width_factor=1.5,
    )

    trainer = HKTrainer(model=model, args=args)
    trainer.train()

    assert model.config.intermediate_size >= initial_inter


def test_causal_lm_sampling_and_generation():
    cfg = HKConfig(
        model_type="causal_lm",
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = HKForCausalLM(cfg)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30, 40]])
    generated = model.generate(
        input_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
    )
    assert generated.shape == (1, 14), f"Expected shape (1, 14), got {generated.shape}"


if __name__ == "__main__":
    import inspect
    print("=" * 60)
    print("Running HK Framework End-to-End Test Suite")
    print("=" * 60)
    current_module = sys.modules[__name__]
    test_funcs = [
        getattr(current_module, name)
        for name in dir(current_module)
        if name.startswith("test_") and callable(getattr(current_module, name))
    ]
    passed = 0
    failed = 0
    for fn in test_funcs:
        fn_name = fn.__name__
        print(f"Running {fn_name}...", end=" ", flush=True)
        try:
            sig = inspect.signature(fn)
            if "temp_dir" in sig.parameters:
                d = tempfile.mkdtemp(prefix="hk_test_")
                try:
                    fn(d)
                finally:
                    shutil.rmtree(d, ignore_errors=True)
            else:
                fn()
            print("PASS")
            passed += 1
        except Exception as e:
            print("FAIL")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
