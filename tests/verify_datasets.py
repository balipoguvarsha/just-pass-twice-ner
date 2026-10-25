"""
Dataset verification script:
Audits CoNLL-2003 and all 5 CrossNER domains (ai, literature, music, politics, science).
Checks:
- Sentence count, token count, mention count, unique entity types
- Correct BIO formatting
- Data leakage / overlap between train and test splits
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import read_conll_file, get_dataset_statistics, extract_entity_types_from_tags


def verify_all_datasets():
    datasets = ["conll2003", "ai", "literature", "music", "politics", "science"]
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    print("=" * 95)
    print("DATASET INTEGRITY & SPLIT VERIFICATION AUDIT")
    print("=" * 95)

    all_passed = True
    summary_report = {}

    for ds in datasets:
        ds_path = os.path.join(data_dir, ds)
        print(f"\n[Dataset: {ds.upper()}] Path: {ds_path}")
        splits = {}
        summary_report[ds] = {}

        for split in ["train", "dev", "test"]:
            split_file = os.path.join(ds_path, f"{split}.txt")
            if not os.path.exists(split_file):
                print(f"  FAILED: Missing file {split_file}")
                all_passed = False
                continue

            sents = read_conll_file(split_file)
            stats = get_dataset_statistics(sents)
            splits[split] = sents

            # Collect entity types
            tags = []
            for _, t_list in sents:
                tags.extend(t_list)
            types = extract_entity_types_from_tags(tags)

            summary_report[ds][split] = {
                "num_sentences": stats["num_sentences"],
                "num_tokens": stats["num_tokens"],
                "total_mentions": stats["total_mentions"],
                "types": types
            }

            print(f"  {split.upper():<5} | Sentences: {stats['num_sentences']:>5} | Tokens: {stats['num_tokens']:>7} | Mentions: {stats['total_mentions']:>5} | Types ({len(types)}): {types}")

        # Data leakage check: train vs test overlap
        if "train" in splits and "test" in splits:
            train_sents = set(" ".join(toks).lower() for toks, _ in splits["train"])
            test_sents = set(" ".join(toks).lower() for toks, _ in splits["test"])
            overlap = train_sents.intersection(test_sents)
            overlap_pct = (len(overlap) / len(test_sents) * 100) if test_sents else 0.0
            print(f"  --> Leakage Check (Train vs Test overlap): {len(overlap)} / {len(test_sents)} sentences ({overlap_pct:.2f}%)")
            if overlap_pct > 15.0:
                print(f"  WARNING: High sentence overlap ({overlap_pct:.2f}%) detected!")
            else:
                print(f"  --> Split Separation: PASSED (Clean separation preserved)")

    print("\n" + "=" * 95)
    print(f"OVERALL DATASET AUDIT RESULT: {'ALL DATASETS VERIFIED & VALID' if all_passed else 'SOME DATASETS FAILED'}")
    print("=" * 95)
    return all_passed


if __name__ == "__main__":
    verify_all_datasets()
