"""
Zero-Shot Cross-Domain Evaluation benchmark across CrossNER domains:
- AI
- Literature
- Music
- Politics
- Science

Evaluates Baseline vs. Paper-Inspired JPT vs. Proposed Multi-Def JPT.
In zero-shot transfer, target entity definitions are injected at test time without retraining.
"""

import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import (
    read_conll_file,
    extract_entity_types_from_tags,
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
from src.evaluate import compute_entity_metrics
from src.train import decode_predictions_to_bio


def evaluate_domain_zero_shot(
    domain: str,
    models: dict,
    tokenizer,
    device: str = "cpu",
    max_sentences: int = 150
) -> dict:
    """
    Run zero-shot evaluation on a given domain test set.
    Uses up to max_sentences for fast, deterministic evaluation on CPU.
    """
    test_path = os.path.join(PROJECT_ROOT, "data", "raw", domain, "test.txt")
    if not os.path.exists(test_path):
        print(f"Warning: {test_path} not found. Skipping domain {domain}.")
        return {}

    test_sentences = read_conll_file(test_path)
    if max_sentences and len(test_sentences) > max_sentences:
        test_sentences = test_sentences[:max_sentences]

    # Extract target entity types
    all_tags = []
    for _, tags in test_sentences:
        all_tags.extend(tags)
    domain_types = extract_entity_types_from_tags(all_tags)

    # Domain BIO labels
    domain_labels = ["O"]
    for t in domain_types:
        domain_labels.append(f"B-{t}")
        domain_labels.append(f"I-{t}")
    label_to_id = {lbl: i for i, lbl in enumerate(domain_labels)}
    id_to_label = {i: lbl for i, lbl in enumerate(domain_labels)}

    # Target type keys for definition lookup
    type_keys = ["O"]
    for lbl in domain_labels[1:]:
        t = lbl.split("-", 1)[1]
        type_keys.append(t)

    # Build datasets
    two_pass_dataset = JPTNERDataset(test_sentences, tokenizer, label_to_id, two_pass=True, max_length=128)
    two_pass_loader = DataLoader(two_pass_dataset, batch_size=16, shuffle=False, collate_fn=collate_jpt_batch)

    single_pass_dataset = JPTNERDataset(test_sentences, tokenizer, label_to_id, two_pass=False, max_length=128)
    single_pass_loader = DataLoader(single_pass_dataset, batch_size=16, shuffle=False, collate_fn=collate_jpt_batch)

    gold_sequences = [tags for _, tags in test_sentences]
    domain_results = {
        "domain": domain,
        "num_sentences": len(test_sentences),
        "target_entity_types": domain_types,
        "models": {}
    }

    # 1. Baseline Evaluation
    if "baseline" in models:
        base_model = models["baseline"]
        base_model.eval()
        base_preds = []
        # Baseline uses its own trained label map (from CoNLL-2003)
        base_id_to_label = models["baseline_id_to_label"]
        with torch.no_grad():
            for batch in single_pass_loader:
                out = base_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                preds = decode_predictions_to_bio(out["logits"], batch["eval_mask"].to(device), base_id_to_label)
                base_preds.extend(preds)

        base_metrics = compute_entity_metrics(gold_sequences, base_preds)
        domain_results["models"]["baseline"] = {
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "f1": base_metrics["f1"],
            "total_gold": base_metrics["total_gold"],
            "total_pred": base_metrics["total_pred"],
            "total_correct": base_metrics["total_correct"]
        }

    # 2. Paper-Inspired JPT (Canonical Single Definition)
    if "jpt" in models:
        jpt_model = models["jpt"]
        jpt_model.eval()
        backbone = models["backbone"]
        def_embs = create_definition_embeddings(
            type_keys, CANONICAL_DEFINITIONS, tokenizer, backbone, device=device, is_multi=False
        )
        jpt_preds = []
        with torch.no_grad():
            for batch in two_pass_loader:
                out = jpt_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    entity_embeddings=def_embs
                )
                preds = decode_predictions_to_bio(out["logits"], batch["eval_mask"].to(device), id_to_label)
                jpt_preds.extend(preds)

        jpt_metrics = compute_entity_metrics(gold_sequences, jpt_preds)
        domain_results["models"]["jpt"] = {
            "precision": jpt_metrics["precision"],
            "recall": jpt_metrics["recall"],
            "f1": jpt_metrics["f1"],
            "total_gold": jpt_metrics["total_gold"],
            "total_pred": jpt_metrics["total_pred"],
            "total_correct": jpt_metrics["total_correct"]
        }

    # 3. Multi-Def JPT (Paraphrase Aggregated Definitions)
    if "multi_def_jpt" in models:
        multidef_model = models["multi_def_jpt"]
        multidef_model.eval()
        backbone = models["backbone"]
        multi_embs = create_definition_embeddings(
            type_keys, PARAPHRASED_DEFINITIONS, tokenizer, backbone, device=device, is_multi=True
        )
        multidef_preds = []
        with torch.no_grad():
            for batch in two_pass_loader:
                out = multidef_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    entity_embeddings=multi_embs
                )
                preds = decode_predictions_to_bio(out["logits"], batch["eval_mask"].to(device), id_to_label)
                multidef_preds.extend(preds)

        multidef_metrics = compute_entity_metrics(gold_sequences, multidef_preds)
        domain_results["models"]["multi_def_jpt"] = {
            "precision": multidef_metrics["precision"],
            "recall": multidef_metrics["recall"],
            "f1": multidef_metrics["f1"],
            "total_gold": multidef_metrics["total_gold"],
            "total_pred": multidef_metrics["total_pred"],
            "total_correct": multidef_metrics["total_correct"]
        }

    return domain_results


def run_crossner_benchmark():
    print("=" * 95)
    print("ZERO-SHOT CROSS-DOMAIN EVALUATION BENCHMARK (CROSSNER)")
    print("=" * 95)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    config = AutoConfig.from_pretrained("distilgpt2")

    weights_path = os.path.join(PROJECT_ROOT, "distilgpt2_model.safetensors")
    causal_model = AutoModelForCausalLM.from_config(config)
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        causal_model.load_state_dict(state_dict, strict=False)
        causal_model.tie_weights()
    causal_model.eval()
    backbone = causal_model.transformer

    # Load trained checkpoints
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    models = {"backbone": backbone}

    # 1. Baseline Checkpoint
    base_ckpt_path = os.path.join(checkpoints_dir, "baseline_model.pt")
    if os.path.exists(base_ckpt_path):
        base_chk = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        base_labels = base_chk["labels"]
        base_model = BaselineTokenClassifier(backbone, num_labels=len(base_labels))
        base_model.load_state_dict(base_chk["state_dict"], strict=False)
        models["baseline"] = base_model
        models["baseline_id_to_label"] = base_chk["id_to_label"]

    # 2. JPT Checkpoint
    jpt_ckpt_path = os.path.join(checkpoints_dir, "jpt_model.pt")
    if os.path.exists(jpt_ckpt_path):
        jpt_chk = torch.load(jpt_ckpt_path, map_location=device, weights_only=False)
        jpt_labels = jpt_chk["labels"]
        jpt_model = JustPassTwiceNER(
            backbone=backbone,
            projection_dim=128,
            entity_emb_dim=config.hidden_size,
            num_entity_types=len(jpt_labels)
        )
        jpt_model.load_state_dict(jpt_chk["state_dict"], strict=False)
        models["jpt"] = jpt_model

    # 3. Multi-Def JPT Checkpoint
    multidef_ckpt_path = os.path.join(checkpoints_dir, "multi_def_jpt_model.pt")
    if os.path.exists(multidef_ckpt_path):
        multidef_chk = torch.load(multidef_ckpt_path, map_location=device, weights_only=False)
        multidef_labels = multidef_chk["labels"]
        multidef_model = MultiDefJustPassTwiceNER(
            backbone=backbone,
            projection_dim=128,
            entity_emb_dim=config.hidden_size,
            num_entity_types=len(multidef_labels),
            aggregation="mean"
        )
        multidef_model.load_state_dict(multidef_chk["state_dict"], strict=False)
        models["multi_def_jpt"] = multidef_model

    domains = ["ai", "literature", "music", "politics", "science"]
    all_domain_results = {}

    for domain in domains:
        print(f"\nEvaluating Domain: {domain.upper()} (Zero-Shot Transfer)...")
        res = evaluate_domain_zero_shot(domain, models, tokenizer, device=device, max_sentences=120)
        all_domain_results[domain] = res
        print(f"  Types: {res['target_entity_types']}")
        for m_name, m_res in res["models"].items():
            print(f"  [{m_name:<15}] Precision: {m_res['precision']:>5.2f}% | Recall: {m_res['recall']:>5.2f}% | F1: {m_res['f1']:>5.2f}% (Found: {m_res['total_correct']}/{m_res['total_gold']})")

    # Save to outputs/crossner_results.json
    out_path = os.path.join(PROJECT_ROOT, "outputs", "crossner_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_domain_results, f, indent=2)
    print(f"\nSaved cross-domain results to {out_path}")

    # Summary table
    print("\n" + "=" * 95)
    print("CROSS-DOMAIN ZERO-SHOT F1 BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"{'Domain':<15} | {'Baseline F1':<15} | {'JPT (Paper) F1':<18} | {'Multi-Def JPT F1':<18}")
    print("-" * 95)
    for dom in domains:
        m = all_domain_results[dom]["models"]
        b_f1 = f"{m.get('baseline', {}).get('f1', 0.0):.2f}%"
        j_f1 = f"{m.get('jpt', {}).get('f1', 0.0):.2f}%"
        md_f1 = f"{m.get('multi_def_jpt', {}).get('f1', 0.0):.2f}%"
        print(f"{dom.upper():<15} | {b_f1:<15} | {j_f1:<18} | {md_f1:<18}")
    print("=" * 95)


if __name__ == "__main__":
    run_crossner_benchmark()
