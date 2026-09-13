"""
Unit Tests for Universal Heterogeneous Stage Model Pipeline:
- Arbitrary DAG and sequence of N heterogeneous stages (Audio, Vision OCR, NLP Analysis, Custom, LLM)
- Blackboard PipelineContext routing and dynamic transformations
- Fine-grained per-stage hyperparameter controls
- Universal pipeline serialization and deserialization (pipeline_manifest.json)
- Universal pipeline factory presets (vision_analysis_generation, speech_analysis_generation)
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
from hk.composite import (
    UniversalPipeline,
    PipelineStage,
    PipelineContext,
    ContextWindowManager,
    HKWhisperModel,
    HKDistilBertModel,
)
from hk.modeling import HKForCausalLM, HKForHandwritingRecognition, HKForSequenceClassification
from hk.tokenizer import HKTokenizer


class TestUniversalPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="hk_test_universal_")
        self.tokenizer = HKTokenizer(
            vocab=["<pad>", "<s>", "</s>", "<unk>", "hello", "robot", "invoice", "total", "amount", "$100", " "]
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_arbitrary_four_stage_pipeline(self):
        """Test a 4-stage heterogeneous pipeline: OCR -> Intent Analysis -> Custom Filter -> LLM."""
        ocr_cfg = hk.HKConfig(model_type="hwr", hidden_size=64, num_classes=26)
        ocr_model = HKForHandwritingRecognition(ocr_cfg)

        analyzer_model = HKDistilBertModel()

        llm_cfg = hk.HKConfig(model_type="causal_lm", hidden_size=64, num_hidden_layers=2, num_attention_heads=2)
        llm_model = HKForCausalLM(llm_cfg)

        pipeline = UniversalPipeline(tokenizer=self.tokenizer)

        # Stage 1: Vision OCR
        pipeline.add_stage(
            name="document_ocr",
            model=ocr_model,
            task_type="vision_ocr",
            input_mapping={"image": "receipt_image"},
            output_key="ocr_raw",
        )

        # Stage 2: Token / Intent Analysis
        pipeline.add_stage(
            name="intent_classifier",
            model=analyzer_model,
            task_type="token_analysis",
            params={"return_top_k": 2},
        )

        # Stage 3: Custom Context Normalizer / Transformer
        def prompt_enricher(context: PipelineContext, params: dict):
            intent = context.get("analysis", {}).get("intent", "general")
            ocr_text = context.get("ocr_text", "")
            enriched = f"[{intent.upper()}] Process document: {ocr_text}"
            context["enriched_prompt"] = enriched
            context["prompt"] = enriched
            return enriched

        pipeline.add_stage(
            name="context_enricher",
            model=None,
            task_type="custom",
            transform_fn=prompt_enricher,
        )

        # Stage 4: Generative LLM
        pipeline.add_stage(
            name="llm_generator",
            model=llm_model,
            task_type="text_generation",
            params={"temperature": 0.5, "max_new_tokens": 20},
        )

        self.assertEqual(len(pipeline.stages), 4)

        # Create synthetic image input [1, 1, 32, 128]
        dummy_img = torch.randn(1, 1, 32, 128)

        # Execute universal pipeline
        res = pipeline(receipt_image=dummy_img)

        self.assertIsInstance(res, PipelineContext)
        self.assertEqual(len(res["stages_executed"]), 4)
        self.assertIn("document_ocr", res["stages_executed"])
        self.assertIn("intent_classifier", res["stages_executed"])
        self.assertIn("context_enricher", res["stages_executed"])
        self.assertIn("llm_generator", res["stages_executed"])

        self.assertIn("ocr_text", res)
        self.assertIn("analysis", res)
        self.assertIn("enriched_prompt", res)
        self.assertIn("generated_text", res)
        self.assertIsInstance(res["generated_text"], str)

    def test_dynamic_stage_addition_and_removal(self):
        """Test adding, replacing, and removing pipeline stages dynamically at runtime."""
        pipeline = UniversalPipeline(tokenizer=self.tokenizer)

        stage1 = PipelineStage(name="s1", model=None, task_type="custom", transform_fn=lambda ctx, p: "step1")
        stage2 = PipelineStage(name="s2", model=None, task_type="custom", transform_fn=lambda ctx, p: "step2")

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        self.assertEqual(len(pipeline.stages), 2)

        # Test stage parameter configuration
        pipeline.set_stage_param("s1", threshold=0.9, debug=True)
        self.assertEqual(pipeline.get_stage_param("s1")["threshold"], 0.9)

        # Remove stage
        removed = pipeline.remove_stage("s1")
        self.assertTrue(removed)
        self.assertEqual(len(pipeline.stages), 1)
        self.assertIsNone(pipeline.get_stage("s1"))

        # Execute with remaining stage
        res = pipeline()
        self.assertEqual(res["stages_executed"], ["s2"])

    def test_universal_pipeline_serialization_roundtrip(self):
        """Test saving and restoring a multi-stage universal pipeline via pipeline_manifest.json."""
        ocr_model = HKForHandwritingRecognition(hk.HKConfig(model_type="hwr", hidden_size=64, num_classes=26))
        analyzer_model = HKDistilBertModel()
        llm_model = HKForCausalLM(hk.HKConfig(model_type="causal_lm", hidden_size=64, num_hidden_layers=2, num_attention_heads=2))

        pipeline = UniversalPipeline(tokenizer=self.tokenizer)
        pipeline.add_stage("ocr_stage", ocr_model, task_type="vision_ocr")
        pipeline.add_stage("analyzer_stage", analyzer_model, task_type="token_analysis", params={"return_top_k": 3})
        pipeline.add_stage("llm_stage", llm_model, task_type="text_generation", params={"temperature": 0.4})

        # Save pipeline
        save_path = os.path.join(self.test_dir, "saved_universal_pipeline")
        manifest_file = pipeline.save_pipeline(save_path)
        self.assertTrue(os.path.exists(manifest_file))

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["pipeline_type"], "universal")
        self.assertEqual(len(manifest["stages"]), 3)
        self.assertEqual(manifest["stages"][0]["name"], "ocr_stage")

        # Load pipeline from directory
        loaded_pipeline = UniversalPipeline.load_pipeline(save_path)
        self.assertEqual(len(loaded_pipeline.stages), 3)
        self.assertIsNotNone(loaded_pipeline.get_stage("ocr_stage"))
        self.assertIsNotNone(loaded_pipeline.get_stage("analyzer_stage"))
        self.assertIsNotNone(loaded_pipeline.get_stage("llm_stage"))

    def test_vision_analysis_generation_preset(self):
        """Test the vision_analysis_generation factory preset."""
        preset = UniversalPipeline.vision_analysis_generation(tokenizer=self.tokenizer)
        self.assertEqual(len(preset.stages), 3)
        self.assertEqual([s.name for s in preset.stages], ["ocr", "analyzer", "llm"])

        # Execute preset with synthetic handwriting image
        img = torch.randn(1, 1, 32, 128)
        res = preset(image=img)
        self.assertIn("ocr", res["stages_executed"])
        self.assertIn("analyzer", res["stages_executed"])
        self.assertIn("llm", res["stages_executed"])
        self.assertIsNotNone(res.get("ocr_text"))
        self.assertIsNotNone(res.get("analysis"))

    def test_high_level_pipeline_universal_task(self):
        """Test hk.pipeline('universal') and hk.pipeline('vision-text-generation')."""
        pipe1 = hk.pipeline("universal", tokenizer=self.tokenizer)
        self.assertIsInstance(pipe1, UniversalPipeline)

        pipe2 = hk.pipeline("vision-text-generation", tokenizer=self.tokenizer)
        self.assertIsInstance(pipe2, UniversalPipeline)
        self.assertEqual(len(pipe2.stages), 3)


if __name__ == "__main__":
    unittest.main()
