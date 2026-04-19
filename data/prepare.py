"""
Data preparation pipeline.

Steps:
  1. Load NuminaMath-CoT + NuminaMath-TIR from disk (run download_data.py first)
  2. Filter to olympiad/AIME-level problems with integer answers
  3. Format each example as {"problem", "solution", "answer", "source"}
  4. Deduplicate on problem text
  5. Save 95/5 train/val splits as JSONL under data/processed/
"""

import json
import os
import re
import random
from datasets import load_from_disk, Dataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

RANDOM_SEED = 42

# NuminaMath-CoT source tags to keep (olympiad + AIME level)
OLYMPIAD_SOURCES = {
    "amc_aime",
    "olympiad",
    "imo_shortlist",
    "cn_contest",
    "aops_forum",
}

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def is_integer_answer(answer: str) -> bool:
    try:
        int(answer.replace(",", "").strip())
        return True
    except ValueError:
        return False


def format_example(problem: str, solution: str, answer: str, source: str) -> dict:
    return {
        "problem": problem.strip(),
        "solution": f"<think>\n{solution.strip()}\n</think>\n\\boxed{{{answer}}}",
        "answer": answer.strip(),
        "source": source,
    }


def process_numina_cot(raw_path: str) -> list[dict]:
    print("Processing NuminaMath-CoT...")
    ds = load_from_disk(raw_path)

    # dataset may be a DatasetDict with a "train" split
    if hasattr(ds, "keys"):
        ds = ds["train"]

    examples = []
    skipped = 0

    for row in ds:
        source = (row.get("source") or "").lower()
        if not any(tag in source for tag in OLYMPIAD_SOURCES):
            skipped += 1
            continue

        problem = row.get("problem", "")
        solution = row.get("solution", "")

        answer = extract_boxed_answer(solution)
        if answer is None or not is_integer_answer(answer):
            skipped += 1
            continue

        examples.append(format_example(problem, solution, answer, source))

    print(f"  Kept {len(examples):,} | Skipped {skipped:,}")
    return examples


def process_numina_tir(raw_path: str) -> list[dict]:
    print("Processing NuminaMath-TIR...")
    ds = load_from_disk(raw_path)

    if hasattr(ds, "keys"):
        ds = ds["train"]

    examples = []
    skipped = 0

    for row in ds:
        source = (row.get("source") or "tir").lower()
        problem = row.get("problem", "")

        # TIR solutions include Python tool calls — keep them as-is
        solution = row.get("solution", "")

        answer = extract_boxed_answer(solution)
        if answer is None or not is_integer_answer(answer):
            skipped += 1
            continue

        examples.append(format_example(problem, solution, answer, f"tir_{source}"))

    print(f"  Kept {len(examples):,} | Skipped {skipped:,}")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for ex in examples:
        key = ex["problem"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    removed = len(examples) - len(unique)
    print(f"Deduplication: removed {removed:,} duplicates, {len(unique):,} remaining")
    return unique


def train_val_split(examples: list[dict], val_ratio: float = 0.05):
    random.seed(RANDOM_SEED)
    random.shuffle(examples)
    split = int(len(examples) * (1 - val_ratio))
    return examples[:split], examples[split:]


def save_jsonl(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data):,} examples → {path}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_examples = []

    cot_path = os.path.join(RAW_DIR, "numina_math_cot")
    if os.path.exists(cot_path):
        all_examples.extend(process_numina_cot(cot_path))
    else:
        print(f"Skipping NuminaMath-CoT (not found at {cot_path})")
        print("  Run: python scripts/download_data.py")

    tir_path = os.path.join(RAW_DIR, "numina_math_tir")
    if os.path.exists(tir_path):
        all_examples.extend(process_numina_tir(tir_path))
    else:
        print(f"Skipping NuminaMath-TIR (not found at {tir_path})")

    if not all_examples:
        print("No data found. Exiting.")
        return

    all_examples = deduplicate(all_examples)

    train_data, val_data = train_val_split(all_examples)
    print(f"Split: {len(train_data):,} train / {len(val_data):,} val")

    save_jsonl(train_data, os.path.join(PROCESSED_DIR, "train.jsonl"))
    save_jsonl(val_data, os.path.join(PROCESSED_DIR, "val.jsonl"))

    print("\nData preparation complete.")
    print(f"  Train: {os.path.join(PROCESSED_DIR, 'train.jsonl')}")
    print(f"  Val:   {os.path.join(PROCESSED_DIR, 'val.jsonl')}")


if __name__ == "__main__":
    main()
