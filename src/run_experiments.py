"""
Experiment execution script:
Trains and evaluates:
1. Conventional Baseline Token Classifier
2. Paper-Inspired JPT (Single Definition)
3. Proposed Multi-Def JPT (Paraphrase Aggregation)
Records exact entity-level precision, recall, F1, latency, throughput, and saves results.json.
"""

import os
import sys
import json
import time
import torch

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

from src.data import read_conll_file, get_dataset_statistics
from src.train import set_seed, train_single_model


def main():
    print("=" * 70)
    print("EXPERIMENT SUITE: JUST PASS TWICE (JPT) NAMED ENTITY RECOGNITION")
    print("Comparing: Baseline vs Paper-Inspired JPT vs Multi-Definition JPT")
    print("=" * 70)

    with open(os.path.join(PROJECT_ROOT, "config.json"), "r") as f:
        config = json.load(f)

    set_seed(config["training"]["random_seed"])
    device = "cpu"

    # Load shared backbone and tokenizer
    print("Loading DistilGPT-2 backbone...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    hf_config = AutoConfig.from_pretrained("distilgpt2")
    causal_model = AutoModelForCausalLM.from_config(hf_config)
    weights_path = os.path.join(PROJECT_ROOT, config["model"]["local_weights_path"])
    
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        causal_model.load_state_dict(state_dict, strict=False)
        causal_model.tie_weights()
    else:
        print("Warning: local weights not found, using initialized weights.")
        
    causal_model.eval()
    backbone = causal_model.transformer

    # Load dataset: CoNLL-2003
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw", "conll2003")
    train_file = os.path.join(data_dir, "train.txt")
    dev_file = os.path.join(data_dir, "dev.txt")
    test_file = os.path.join(data_dir, "test.txt")

    print(f"Reading dataset splits from: {data_dir}")
    train_all = read_conll_file(train_file)
    dev_all = read_conll_file(dev_file)
    test_all = read_conll_file(test_file)

    # Use controlled, reproducible subset for reliable CPU execution
    train_size = config["training"]["full_train_samples"]
    val_size = config["training"]["val_samples"]
    test_size = config["training"]["test_samples"]

    train_data = train_all[:train_size]
    val_data = dev_all[:val_size]
    test_data = test_all[:test_size]

    train_stats = get_dataset_statistics(train_data)
    test_stats = get_dataset_statistics(test_data)
    print(f"Train split: {len(train_data)} sentences, {train_stats['num_tokens']} tokens, {train_stats['total_mentions']} entity mentions")
    print(f"Val split:   {len(val_data)} sentences")
    print(f"Test split:  {len(test_data)} sentences, {test_stats['num_tokens']} tokens, {test_stats['total_mentions']} entity mentions")

    models_to_run = ["baseline", "jpt", "multi_def_jpt"]
    results = {}
    logs = {}

    os.makedirs(config["paths"]["outputs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)

    for m_type in models_to_run:
        set_seed(config["training"]["random_seed"])
        res = train_single_model(
            model_type=m_type,
            train_sentences=train_data,
            val_sentences=val_data,
            test_sentences=test_data,
            config=config,
            tokenizer=tokenizer,
            backbone_model=backbone,
            device=device
        )
        results[m_type] = {
            "model_type": m_type,
            "training_duration_sec": res["training_duration_sec"],
            "test_precision": res["test_metrics"]["precision"],
            "test_recall": res["test_metrics"]["recall"],
            "test_f1": res["test_metrics"]["f1"],
            "by_type": res["test_metrics"]["by_type"],
            "speed": res["speed_stats"],
            "checkpoint_path": res["checkpoint_path"]
        }
        logs[m_type] = res["history"]

    # Save to outputs/results.json and outputs/experiment_log.json
    results_path = os.path.join(PROJECT_ROOT, config["paths"]["results_file"])
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log_path = os.path.join(PROJECT_ROOT, config["paths"]["log_file"])
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENTAL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model Architecture':<30} | {'Precision':<10} | {'Recall':<10} | {'Micro F1':<10} | {'Latency':<12} | {'Throughput'}")
    print("-" * 90)
    for m_type, r in results.items():
        print(f"{m_type:<30} | {r['test_precision']:<10.2f} | {r['test_recall']:<10.2f} | {r['test_f1']:<10.2f} | {r['speed']['mean_latency_ms']:<7.2f} ms | {r['speed']['throughput_tokens_per_sec']} tok/s")
    print("=" * 70)
    print(f"Results successfully saved to: {results_path}")


if __name__ == "__main__":
    main()
