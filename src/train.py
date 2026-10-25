"""
Training pipeline for:
1. Conventional Baseline Token Classifier
2. Paper-Inspired Just Pass Twice (JPT) Model
3. Multi-Definition Aggregated JPT (Hypothesis Test)
"""

import os
import sys
import json
import time
import random
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.data import (
    read_conll_file,
    extract_entity_types_from_tags,
    CANONICAL_DEFINITIONS,
    PARAPHRASED_DEFINITIONS,
    JPTNERDataset,
    collate_jpt_batch,
    get_dataset_statistics
)
from src.model import (
    BaselineTokenClassifier,
    JustPassTwiceNER,
    MultiDefJustPassTwiceNER
)
from src.evaluate import (
    compute_entity_metrics,
    benchmark_model_speed
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_definition_embeddings(
    entity_types: list,
    definitions_dict: dict,
    tokenizer,
    backbone_model,
    device: str = "cpu",
    is_multi: bool = False
) -> torch.Tensor:
    """
    Encode entity definition strings using the backbone's token embeddings.
    For each definition, tokenizes the text and computes the mean pooled embedding.
    Returns:
    - If is_multi=False: tensor of shape [num_types, hidden_size]
    - If is_multi=True: tensor of shape [num_types, num_paraphrases, hidden_size]
    """
    backbone_model.eval()
    embed_layer = backbone_model.get_input_embeddings()
    hidden_size = backbone_model.config.hidden_size

    type_embeddings = []
    
    with torch.no_grad():
        for ent_type in entity_types:
            defs = definitions_dict.get(ent_type, ["An entity mention."])
            if isinstance(defs, str):
                defs = [defs]

            paraphrase_embs = []
            for d in defs:
                tokens = tokenizer.encode(" " + d, add_special_tokens=False)
                if not tokens:
                    tokens = [tokenizer.eos_token_id or 0]
                tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
                emb = embed_layer(tok_tensor).mean(dim=0)  # [hidden_size]
                paraphrase_embs.append(emb)

            if is_multi:
                # Stack all paraphrases
                stacked = torch.stack(paraphrase_embs)  # [num_paraphrases, hidden_size]
                type_embeddings.append(stacked)
            else:
                # Use canonical (first) definition
                type_embeddings.append(paraphrase_embs[0])

    if is_multi:
        # Pad to max number of paraphrases across types if unequal
        max_p = max(t.size(0) for t in type_embeddings)
        padded = []
        for t in type_embeddings:
            if t.size(0) < max_p:
                pad_t = t[-1:].repeat(max_p - t.size(0), 1)
                t = torch.cat([t, pad_t], dim=0)
            padded.append(t)
        return torch.stack(padded).to(device)  # [num_types, max_p, hidden_size]
    else:
        return torch.stack(type_embeddings).to(device)  # [num_types, hidden_size]


def decode_predictions_to_bio(
    logits: torch.Tensor,
    eval_mask: torch.Tensor,
    id_to_label: dict
) -> list:
    """Decode model logits to BIO tag sequences using eval_mask positions."""
    preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
    batch_bio = []

    for b in range(preds.size(0)):
        seq_bio = []
        for pos in range(preds.size(1)):
            if eval_mask[b, pos] == 1:
                label_id = preds[b, pos].item()
                seq_bio.append(id_to_label.get(label_id, "O"))
        batch_bio.append(seq_bio)

    return batch_bio


def train_single_model(
    model_type: str,
    train_sentences: list,
    val_sentences: list,
    test_sentences: list,
    config: dict,
    tokenizer,
    backbone_model,
    device: str = "cpu"
) -> dict:
    """Train and evaluate one model architecture."""
    print(f"\n--- Starting Training for Model Type: {model_type} ---")
    start_time = time.time()

    # Collect entity types
    all_tags = []
    for _, tags in train_sentences:
        all_tags.extend(tags)
    entity_types = extract_entity_types_from_tags(all_tags)
    
    # Label mapping: 0 -> O, followed by B- and I- tags for each entity type
    labels = ["O"]
    for t in entity_types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # For definition-based models, each label maps to an entity type definition
    # Index 0 is 'O', then B-T and I-T map to type T
    type_keys = ["O"]
    for lbl in labels[1:]:
        t = lbl.split("-", 1)[1]
        type_keys.append(t)

    is_two_pass = (model_type != "baseline")
    max_len = config["model"]["max_seq_length"]

    train_dataset = JPTNERDataset(train_sentences, tokenizer, label_to_id, two_pass=is_two_pass, max_length=max_len)
    val_dataset = JPTNERDataset(val_sentences, tokenizer, label_to_id, two_pass=is_two_pass, max_length=max_len)
    test_dataset = JPTNERDataset(test_sentences, tokenizer, label_to_id, two_pass=is_two_pass, max_length=max_len)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_jpt_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_jpt_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_jpt_batch)

    # Instantiate model
    proj_dim = config["model"]["projection_dim"]
    o_weight = config["model"]["o_class_weight"]
    hidden_size = backbone_model.config.hidden_size

    # Following paper Section 3.5: Freeze base language model backbone
    for p in backbone_model.parameters():
        p.requires_grad = False
    backbone_model.eval()

    if model_type == "baseline":
        model = BaselineTokenClassifier(backbone_model, num_labels=num_labels, dropout=config["model"]["dropout"])
        def_embs = None
    elif model_type == "jpt":
        model = JustPassTwiceNER(
            backbone=backbone_model,
            projection_dim=proj_dim,
            entity_emb_dim=hidden_size,
            num_entity_types=num_labels,
            dropout=config["model"]["dropout"],
            o_class_weight=o_weight
        )
        def_embs = create_definition_embeddings(type_keys, CANONICAL_DEFINITIONS, tokenizer, backbone_model, device=device, is_multi=False)
    elif model_type == "multi_def_jpt":
        model = MultiDefJustPassTwiceNER(
            backbone=backbone_model,
            projection_dim=proj_dim,
            entity_emb_dim=hidden_size,
            num_entity_types=num_labels,
            dropout=config["model"]["dropout"],
            o_class_weight=o_weight,
            aggregation="mean"
        )
        def_embs = create_definition_embeddings(type_keys, PARAPHRASED_DEFINITIONS, tokenizer, backbone_model, device=device, is_multi=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters for {model_type}: {num_trainable:,} (Backbone frozen as in JPT paper)", flush=True)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    best_val_f1 = -1.0
    best_model_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            if model_type == "baseline":
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, entity_embeddings=def_embs, labels=batch_labels)

            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Validation
        val_metrics, _, _ = evaluate_loader(model, val_loader, id_to_label, def_embs, model_type, device)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val P: {val_metrics['precision']:.1f}% | Val R: {val_metrics['recall']:.1f}% | Val F1: {val_metrics['f1']:.1f}%", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"]
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model for test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final Test Evaluation
    test_metrics, test_preds, test_golds = evaluate_loader(model, test_loader, id_to_label, def_embs, model_type, device)
    
    # Speed Benchmark
    sample_batch = next(iter(test_loader))
    sample_inputs = {"input_ids": sample_batch["input_ids"][:1], "attention_mask": sample_batch["attention_mask"][:1]}
    add_kwargs = {"entity_embeddings": def_embs} if model_type != "baseline" else {}
    speed_stats = benchmark_model_speed(model, sample_inputs, add_kwargs, num_runs=30, device=device)

    duration = time.time() - start_time
    print(f"Done {model_type} in {duration:.1f}s. Test F1: {test_metrics['f1']}%, Latency: {speed_stats['mean_latency_ms']} ms", flush=True)

    # Save checkpoint
    chk_dir = config["paths"]["checkpoints_dir"]
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, f"{model_type}_model.pt")
    torch.save({
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "type_keys": type_keys,
        "config": config,
        "test_metrics": test_metrics,
        "speed_stats": speed_stats
    }, chk_path)

    return {
        "model_type": model_type,
        "training_duration_sec": round(duration, 2),
        "test_metrics": test_metrics,
        "speed_stats": speed_stats,
        "history": history,
        "checkpoint_path": chk_path
    }


def evaluate_loader(
    model,
    loader: DataLoader,
    id_to_label: dict,
    def_embs,
    model_type: str,
    device: str
) -> tuple:
    """Run inference over a DataLoader and calculate exact chunk-level metrics."""
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            eval_mask = batch["eval_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            if model_type == "baseline":
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, entity_embeddings=def_embs)

            batch_bio = decode_predictions_to_bio(out["logits"], eval_mask, id_to_label)
            all_preds.extend(batch_bio)

            # Ground truth BIO
            for b in range(batch_labels.size(0)):
                seq_gold = []
                for pos in range(batch_labels.size(1)):
                    if eval_mask[b, pos] == 1:
                        lbl_id = batch_labels[b, pos].item()
                        seq_gold.append(id_to_label.get(lbl_id, "O"))
                all_golds.append(seq_gold)

    metrics = compute_entity_metrics(all_golds, all_preds)
    return metrics, all_preds, all_golds


if __name__ == "__main__":
    from src.run_experiments import main as run_all
    run_all()

