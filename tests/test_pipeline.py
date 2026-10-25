"""
Comprehensive test suite covering:
1. Dataset loading and BIO-label parsing.
2. BIO normalization and span extraction.
3. Subword tokenization and Two-Pass index alignment.
4. Model forward pass shapes (Baseline, JPT, MultiDef JPT).
5. Loss computation and masking.
6. Prediction decoding from logits to BIO tags.
7. Evaluation metric computation (Precision, Recall, Micro-F1).
8. Predictor basic inference and span reconstruction.
"""

import os
import sys
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

from src.data import (
    normalize_tag,
    read_conll_file,
    extract_entity_types_from_tags,
    get_dataset_statistics,
    CANONICAL_DEFINITIONS,
    PARAPHRASED_DEFINITIONS,
    create_definition_embeddings,
    JPTNERDataset,
    collate_jpt_batch
)
from src.model import (
    BaselineTokenClassifier,
    JustPassTwiceNER,
    MultiDefJustPassTwiceNER
)
from src.evaluate import (
    extract_spans_from_bio,
    compute_entity_metrics,
    benchmark_model_speed
)
from src.predict import JPTPredictor


class TestJPTPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        cls.config = AutoConfig.from_pretrained("distilgpt2")
        cls.causal_model = AutoModelForCausalLM.from_config(cls.config)
        weights_path = os.path.join(PROJECT_ROOT, "distilgpt2_model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            cls.causal_model.load_state_dict(state_dict, strict=False)
            cls.causal_model.tie_weights()
        cls.causal_model.eval()
        cls.backbone = cls.causal_model.transformer

    def test_01_tag_normalization(self):
        self.assertEqual(normalize_tag("B-organisation"), "B-ORG")
        self.assertEqual(normalize_tag("B-organization"), "B-ORG")
        self.assertEqual(normalize_tag("B-person"), "B-PER")
        self.assertEqual(normalize_tag("B-location"), "B-LOC")
        self.assertEqual(normalize_tag("O"), "O")
        self.assertEqual(normalize_tag("B-POLITICIAN"), "B-POLITICIAN")

    def test_02_dataset_loading(self):
        ai_train = os.path.join(PROJECT_ROOT, "data", "raw", "ai", "train.txt")
        self.assertTrue(os.path.exists(ai_train), f"Dataset file missing: {ai_train}")
        sentences = read_conll_file(ai_train)
        self.assertGreater(len(sentences), 0, "No sentences loaded from AI train set")
        tokens, tags = sentences[0]
        self.assertEqual(len(tokens), len(tags), "Tokens and tags count must match")
        stats = get_dataset_statistics(sentences)
        self.assertGreater(stats["num_tokens"], 0)
        self.assertGreater(stats["total_mentions"], 0)

    def test_03_span_extraction_and_metrics(self):
        gold = [["B-PER", "I-PER", "O", "B-LOC", "O"]]
        pred_exact = [["B-PER", "I-PER", "O", "B-LOC", "O"]]
        metrics = compute_entity_metrics(gold, pred_exact)
        self.assertEqual(metrics["precision"], 100.0)
        self.assertEqual(metrics["recall"], 100.0)
        self.assertEqual(metrics["f1"], 100.0)

        # Partial prediction
        pred_partial = [["B-PER", "O", "O", "B-LOC", "O"]]
        metrics_partial = compute_entity_metrics(gold, pred_partial)
        # Gold has 2 entities, pred has 2 entities, but only LOC is exactly matched
        self.assertEqual(metrics_partial["total_correct"], 1)
        self.assertEqual(metrics_partial["precision"], 50.0)
        self.assertEqual(metrics_partial["recall"], 50.0)

    def test_04_dataset_two_pass_alignment(self):
        dummy_sentences = [
            (["Marie", "Curie", "discovered", "radium", "."], ["B-PER", "I-PER", "O", "B-MISC", "O"])
        ]
        label_to_id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-MISC": 3}
        dataset = JPTNERDataset(dummy_sentences, self.tokenizer, label_to_id, two_pass=True, max_length=64)
        sample = dataset[0]

        # In two-pass, input_ids contains: pass1 + [SEP] + pass2
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        eval_mask = sample["eval_mask"]

        # Number of active labels in the second pass must equal number of words (5)
        active_eval = (eval_mask == 1).sum().item()
        self.assertEqual(active_eval, 5, "Every word should have exactly one evaluation subword in pass 2")

        # Labels in pass 1 must be all masked (-100)
        sep_indices = (input_ids == (self.tokenizer.eos_token_id or 50256)).nonzero(as_tuple=True)[0]
        self.assertGreater(len(sep_indices), 0)
        sep_pos = sep_indices[0].item()
        pass1_labels = labels[:sep_pos]
        self.assertTrue((pass1_labels == -100).all().item(), "All Pass 1 labels must be masked (-100)")

    def test_05_baseline_model_forward(self):
        model = BaselineTokenClassifier(self.backbone, num_labels=5)
        model.eval()
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        attn_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        labels = torch.tensor([[0, 1, -100]], dtype=torch.long)

        out = model(input_ids, attn_mask, labels=labels)
        self.assertEqual(out["logits"].shape, (1, 3, 5))
        self.assertIsNotNone(out["loss"])
        self.assertGreater(out["loss"].item(), 0)

    def test_06_jpt_bilinear_model_forward(self):
        type_keys = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
        def_embs = create_definition_embeddings(type_keys, CANONICAL_DEFINITIONS, self.tokenizer, self.backbone)

        model = JustPassTwiceNER(
            backbone=self.backbone,
            projection_dim=128,
            entity_emb_dim=self.config.hidden_size,
            num_entity_types=len(type_keys)
        )
        model.eval()

        input_ids = torch.tensor([[10, 20, 50256, 10, 20]], dtype=torch.long)
        attn_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, -100, 1, 2]], dtype=torch.long)

        out = model(input_ids, attn_mask, entity_embeddings=def_embs, labels=labels)
        self.assertEqual(out["logits"].shape, (1, 5, 5))
        self.assertEqual(out["projected_tokens"].shape, (1, 5, 128))
        self.assertEqual(out["projected_entities"].shape, (5, 128))
        self.assertGreater(out["loss"].item(), 0)

    def test_07_multidef_jpt_model_forward(self):
        type_keys = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
        multi_def_embs = create_definition_embeddings(
            type_keys, PARAPHRASED_DEFINITIONS, self.tokenizer, self.backbone, is_multi=True
        )
        self.assertEqual(multi_def_embs.dim(), 3)

        model = MultiDefJustPassTwiceNER(
            backbone=self.backbone,
            projection_dim=128,
            entity_emb_dim=self.config.hidden_size,
            num_entity_types=len(type_keys),
            aggregation="mean"
        )
        model.eval()

        input_ids = torch.tensor([[10, 20, 50256, 10, 20]], dtype=torch.long)
        attn_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, -100, 1, 2]], dtype=torch.long)

        out = model(input_ids, attn_mask, entity_embeddings=multi_def_embs, labels=labels)
        self.assertEqual(out["logits"].shape, (1, 5, 5))
        self.assertGreater(out["loss"].item(), 0)

    def test_08_predictor_end_to_end(self):
        predictor = JPTPredictor(weights_path=os.path.join(PROJECT_ROOT, "distilgpt2_model.safetensors"))
        res = predictor.predict("Albert Einstein won the Nobel Prize in Physics.")
        self.assertIn("tokens", res)
        self.assertIn("bio_tags", res)
        self.assertIn("entities", res)
        self.assertIn("latency_ms", res)
        self.assertEqual(len(res["tokens"]), len(res["bio_tags"]))
        self.assertGreater(res["latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
