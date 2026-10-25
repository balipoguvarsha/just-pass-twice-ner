"""
Inference pipeline for Just Pass Twice (JPT) Named Entity Recognition.
Supports standalone text prediction and span extraction.
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

from src.data import (
    CANONICAL_DEFINITIONS,
    PARAPHRASED_DEFINITIONS,
    create_definition_embeddings
)
from src.model import (
    BaselineTokenClassifier,
    JustPassTwiceNER,
    MultiDefJustPassTwiceNER
)
from src.evaluate import extract_spans_from_bio


class JPTPredictor:
    """End-to-end predictor for Named Entity Recognition using JPT models."""
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "jpt",
        weights_path: str = "distilgpt2_model.safetensors",
        device: str = "cpu"
    ):
        self.device = device
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.config = AutoConfig.from_pretrained("distilgpt2")

        # Load backbone: use local safetensors if available; otherwise fetch official pretrained weights from Hugging Face
        if weights_path and os.path.exists(weights_path):
            self.causal_model = AutoModelForCausalLM.from_config(self.config)
            state_dict = load_file(weights_path)
            self.causal_model.load_state_dict(state_dict, strict=False)
            self.causal_model.tie_weights()
        else:
            self.causal_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        self.causal_model.to(self.device)
        self.causal_model.eval()

        self.backbone = self.causal_model.transformer

        # Default CoNLL labels
        self.labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        self.label_to_id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id_to_label = {i: lbl for i, lbl in enumerate(self.labels)}
        self.type_keys = ["O"] + [lbl.split("-", 1)[1] for lbl in self.labels[1:]]

        # Checkpoint resolution: full checkpoint or lightweight adapter head
        resolved_ckpt = checkpoint_path
        if not resolved_ckpt or not os.path.exists(resolved_ckpt):
            # Check for lightweight adapter head in checkpoints/heads/
            head_path = os.path.join(PROJECT_ROOT, "checkpoints", "heads", f"{self.model_type}_head.pt")
            if os.path.exists(head_path):
                resolved_ckpt = head_path
            else:
                # Check for standard full checkpoint in checkpoints/
                full_path = os.path.join(PROJECT_ROOT, "checkpoints", f"{self.model_type}_model.pt")
                if os.path.exists(full_path):
                    resolved_ckpt = full_path

        self.is_checkpoint_loaded = False
        if resolved_ckpt and os.path.exists(resolved_ckpt):
            chk = torch.load(resolved_ckpt, map_location=device, weights_only=False)
            self.model_type = chk.get("model_type", self.model_type)
            self.labels = chk.get("labels", self.labels)
            self.label_to_id = chk.get("label_to_id", self.label_to_id)
            self.id_to_label = chk.get("id_to_label", self.id_to_label)
            self.type_keys = chk.get("type_keys", ["O"] + [lbl.split("-", 1)[1] for lbl in self.labels[1:]])
            saved_state = chk["state_dict"]
            self.is_checkpoint_loaded = True
        else:
            saved_state = None

        # Build model instance
        num_labels = len(self.labels)
        hidden_size = self.config.hidden_size
        proj_dim = 128

        if self.model_type == "baseline":
            self.model = BaselineTokenClassifier(self.backbone, num_labels=num_labels)
            self.def_embs = None
        elif self.model_type == "multi_def_jpt":
            self.model = MultiDefJustPassTwiceNER(
                backbone=self.backbone,
                projection_dim=proj_dim,
                entity_emb_dim=hidden_size,
                num_entity_types=num_labels,
                aggregation="mean"
            )
            self.def_embs = create_definition_embeddings(
                self.type_keys, PARAPHRASED_DEFINITIONS, self.tokenizer, self.backbone, device=self.device, is_multi=True
            )
        else:  # "jpt"
            self.model = JustPassTwiceNER(
                backbone=self.backbone,
                projection_dim=proj_dim,
                entity_emb_dim=hidden_size,
                num_entity_types=num_labels
            )
            self.def_embs = create_definition_embeddings(
                self.type_keys, CANONICAL_DEFINITIONS, self.tokenizer, self.backbone, device=self.device, is_multi=False
            )

        if saved_state is not None:
            self.model.load_state_dict(saved_state, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        """
        Run NER inference on an arbitrary input string.
        Returns:
            - tokens: list of whitespace/word tokens
            - bio_tags: predicted BIO tags for each token
            - entities: list of recognized entity dictionaries with text, type, start, end
            - latency_ms: forward pass execution time in milliseconds
            - is_trained: boolean indicating if trained weights were used
        """
        if not text or not text.strip():
            return {
                "text": text,
                "tokens": [],
                "bio_tags": [],
                "entities": [],
                "latency_ms": 0.0,
                "is_trained": self.is_checkpoint_loaded
            }

        words = text.strip().split()
        subwords = []
        word_to_subword_indices = []

        for word in words:
            w_sub = self.tokenizer.tokenize(" " + word)
            if not w_sub:
                w_sub = [self.tokenizer.unk_token or ""]
            word_to_subword_indices.append((len(subwords), len(subwords) + len(w_sub)))
            subwords.extend(w_sub)

        subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
        sep_id = getattr(self.tokenizer, "eos_token_id", 50256) or 50256

        t0 = time.perf_counter()

        if self.model_type != "baseline":
            # Two-pass JPT concatenation: [Pass1, SEP, Pass2]
            pass1 = subword_ids
            pass2 = subword_ids
            input_ids = pass1 + [sep_id] + pass2
            offset = len(pass1) + 1
            eval_indices = [offset + start for start, _ in word_to_subword_indices]
        else:
            input_ids = subword_ids
            eval_indices = [start for start, _ in word_to_subword_indices]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            if self.model_type == "baseline":
                out = self.model(input_tensor, attn_mask)
            else:
                out = self.model(input_tensor, attn_mask, entity_embeddings=self.def_embs)

            logits = out["logits"][0]  # [seq_len, num_labels]
            probs = torch.softmax(logits, dim=-1)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        predicted_tags = []
        confidences = []
        for idx in eval_indices:
            if idx < logits.size(0):
                pred_id = torch.argmax(logits[idx]).item()
                prob = probs[idx, pred_id].item()
                predicted_tags.append(self.id_to_label.get(pred_id, "O"))
                confidences.append(round(prob, 4))
            else:
                predicted_tags.append("O")
                confidences.append(0.0)

        # Extract entity spans
        spans = extract_spans_from_bio(predicted_tags)
        entities = []
        for start_idx, end_idx, ent_type in sorted(list(spans), key=lambda x: x[0]):
            mention = " ".join(words[start_idx:end_idx + 1])
            avg_conf = float(np.mean(confidences[start_idx:end_idx + 1])) if confidences else 1.0
            entities.append({
                "entity_text": mention,
                "entity_type": ent_type,
                "token_start": start_idx,
                "token_end": end_idx,
                "confidence": round(avg_conf, 4)
            })

        return {
            "text": text,
            "tokens": words,
            "bio_tags": predicted_tags,
            "entities": entities,
            "latency_ms": latency_ms,
            "is_trained": self.is_checkpoint_loaded,
            "model_type": self.model_type
        }


if __name__ == "__main__":
    predictor = JPTPredictor()
    sample = "Alan Turing developed the Enigma decryption machine at Bletchley Park in the United Kingdom."
    res = predictor.predict(sample)
    print("Sample Prediction Output:")
    print(res)
