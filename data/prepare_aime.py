"""
Prepare AIME 2023 + 2024 problems as a held-out OOD benchmark.

Outputs: data/processed/val_aime.jsonl with fields {problem, answer, source}

Usage:
    python data/prepare_aime.py
"""

import json
import os
import re
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path(__file__).parent / "processed" / "val_aime.jsonl"


def clean_problem(text: str) -> str:
    """Strip common boilerplate prefixes and whitespace."""
    text = text.strip()
    # Some sources prefix "Problem N." or numbering; keep the math content
    text = re.sub(r"^(Problem|Question)\s*\d*[.:]\s*", "", text)
    return text


def normalize_answer(ans) -> str:
    """AIME answers are integers 0–999."""
    s = str(ans).strip()
    # Strip leading zeros for comparison ("023" → "23"), but preserve as int
    try:
        return str(int(s))
    except ValueError:
        return s


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []

    # AIME 2024 — held-out test (30 problems)
    # Note: released Feb 2024; NuminaMath-CoT final update was Nov 2024,
    # so there's some contamination risk. Still useful as OOD baseline.
    ds_2024 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    for ex in ds_2024:
        records.append({
            "problem": clean_problem(ex["Problem"]),
            "answer": normalize_answer(ex["Answer"]),
            "source": "AIME_2024",
        })

    # AIME 2025 — clean OOD (30 problems, released after training data cutoff)
    ds_2025 = load_dataset("yentinglin/aime_2025", split="train")
    for ex in ds_2025:
        records.append({
            "problem": clean_problem(ex["problem"]),
            "answer": normalize_answer(ex["answer"]),
            "source": "AIME_2025",
        })

    with OUT_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} problems to {OUT_PATH}")
    sources = {}
    for r in records:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
