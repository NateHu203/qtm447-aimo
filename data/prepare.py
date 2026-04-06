"""
Data preparation pipeline.

Steps:
  1. Download raw datasets (NuminaMath-CoT, AIME problems, AIMO3 split)
  2. Filter to olympiad-level problems
  3. Format each example as {"problem": ..., "solution": ..., "answer": ..., "source": ...}
  4. Save train/val splits as JSONL under data/processed/
"""

import json
import os
from datasets import load_dataset


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def format_example(problem: str, solution: str, answer: str, source: str) -> dict:
    return {
        "problem": problem,
        "solution": f"<think>\n{solution}\n</think>\n\\boxed{{{answer}}}",
        "answer": answer,
        "source": source,
    }


def load_numina_math():
    """Load NuminaMath-CoT from HuggingFace and filter to olympiad level."""
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    # TODO: filter by difficulty/source to keep olympiad-level only
    return ds


def save_jsonl(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # TODO: load and merge all data sources
    # TODO: train/val split (e.g. 95/5)
    # TODO: save_jsonl(train_data, f"{PROCESSED_DIR}/train.jsonl")
    # TODO: save_jsonl(val_data, f"{PROCESSED_DIR}/val.jsonl")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
