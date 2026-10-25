"""
Entity-level evaluation metrics (Precision, Recall, Micro-F1),
latency, and throughput benchmarking for Named Entity Recognition.
"""

from typing import List, Dict, Tuple, Set, Optional
import time
import json
import torch
import numpy as np


def extract_spans_from_bio(tags: List[str]) -> Set[Tuple[int, int, str]]:
    """
    Extract (start_idx, end_idx, entity_type) spans from a sequence of BIO tags.
    end_idx is inclusive.
    """
    spans = set()
    current_type = None
    start_idx = None

    for i, tag in enumerate(tags):
        tag = tag.upper().strip()
        if tag.startswith("B-"):
            if current_type is not None and start_idx is not None:
                spans.add((start_idx, i - 1, current_type))
            current_type = tag.split("-", 1)[1]
            start_idx = i
        elif tag.startswith("I-"):
            ent_type = tag.split("-", 1)[1]
            if current_type is not None and ent_type == current_type:
                # Continue existing span
                pass
            else:
                # Inconsistent I- tag without matching B-, start new span
                if current_type is not None and start_idx is not None:
                    spans.add((start_idx, i - 1, current_type))
                current_type = ent_type
                start_idx = i
        else:  # "O" or invalid
            if current_type is not None and start_idx is not None:
                spans.add((start_idx, i - 1, current_type))
            current_type = None
            start_idx = None

    if current_type is not None and start_idx is not None:
        spans.add((start_idx, len(tags) - 1, current_type))

    return spans


def compute_entity_metrics(
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]]
) -> Dict[str, float]:
    """
    Compute exact span-level Precision, Recall, and Micro-F1 across sentences.
    Also computes per-class performance metrics.
    """
    total_gold = 0
    total_pred = 0
    total_correct = 0

    per_type_gold = {}
    per_type_pred = {}
    per_type_correct = {}

    for true_tags, pred_tags in zip(true_sequences, pred_sequences):
        gold_spans = extract_spans_from_bio(true_tags)
        pred_spans = extract_spans_from_bio(pred_tags)

        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        correct_spans = gold_spans.intersection(pred_spans)
        total_correct += len(correct_spans)

        for _, _, ent_type in gold_spans:
            per_type_gold[ent_type] = per_type_gold.get(ent_type, 0) + 1
        for _, _, ent_type in pred_spans:
            per_type_pred[ent_type] = per_type_pred.get(ent_type, 0) + 1
        for _, _, ent_type in correct_spans:
            per_type_correct[ent_type] = per_type_correct.get(ent_type, 0) + 1

    precision = (total_correct / total_pred) if total_pred > 0 else 0.0
    recall = (total_correct / total_gold) if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-type breakdown
    all_types = sorted(list(set(list(per_type_gold.keys()) + list(per_type_pred.keys()))))
    type_metrics = {}
    for ent_type in all_types:
        g = per_type_gold.get(ent_type, 0)
        p = per_type_pred.get(ent_type, 0)
        c = per_type_correct.get(ent_type, 0)
        tp_prec = (c / p) if p > 0 else 0.0
        tp_rec = (c / g) if g > 0 else 0.0
        tp_f1 = (2 * tp_prec * tp_rec / (tp_prec + tp_rec)) if (tp_prec + tp_rec) > 0 else 0.0
        type_metrics[ent_type] = {
            "precision": round(tp_prec * 100, 2),
            "recall": round(tp_rec * 100, 2),
            "f1": round(tp_f1 * 100, 2),
            "support": g
        }

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "total_gold": total_gold,
        "total_pred": total_pred,
        "total_correct": total_correct,
        "by_type": type_metrics
    }


def benchmark_model_speed(
    model: torch.nn.Module,
    sample_inputs: Dict[str, torch.Tensor],
    additional_kwargs: Dict,
    num_runs: int = 50,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Accurately measure inference latency (ms per sample) and throughput (tokens/sec).
    """
    model.eval()
    model.to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(5):
            _ = model(
                input_ids=sample_inputs["input_ids"].to(device),
                attention_mask=sample_inputs["attention_mask"].to(device),
                **{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in additional_kwargs.items()}
            )

    latencies = []
    total_tokens = 0
    batch_size = sample_inputs["input_ids"].size(0)
    seq_len = sample_inputs["input_ids"].size(1)

    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(
                input_ids=sample_inputs["input_ids"].to(device),
                attention_mask=sample_inputs["attention_mask"].to(device),
                **{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in additional_kwargs.items()}
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
            total_tokens += (batch_size * seq_len)

    total_time_sec = sum(latencies) / 1000.0
    throughput = (total_tokens / total_time_sec) if total_time_sec > 0 else 0.0

    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "throughput_tokens_per_sec": round(float(throughput), 1),
        "num_benchmarked_runs": num_runs
    }


if __name__ == "__main__":
    import os
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
        print("=" * 85)
        print("EMPIRICAL EVALUATION RESULTS SUMMARY")
        print("=" * 85)
        print(f"{'Model Architecture':<30} | {'Precision':<10} | {'Recall':<10} | {'Micro F1':<10} | {'Latency':<10}")
        print("-" * 85)
        for m_key, m_val in data.items():
            print(f"{m_key:<30} | {m_val['test_precision']:<10.2f} | {m_val['test_recall']:<10.2f} | {m_val['test_f1']:<10.2f} | {m_val['speed']['mean_latency_ms']:<6.2f} ms")
        print("=" * 85)
    else:
        print(f"Results file not found at {results_path}. Please run src/train.py or src/run_experiments.py first.")

