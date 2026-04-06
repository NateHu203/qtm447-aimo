"""
Download raw datasets from HuggingFace to data/raw/.

Usage:
    python scripts/download_data.py
"""

import os
from datasets import load_dataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_numina_math():
    print("Downloading NuminaMath-CoT...")
    ds = load_dataset("AI-MO/NuminaMath-CoT")
    ds.save_to_disk(os.path.join(RAW_DIR, "numina_math_cot"))
    print(f"  Saved {len(ds['train'])} train examples.")


def download_numina_tir():
    print("Downloading NuminaMath-TIR (tool-integrated reasoning)...")
    ds = load_dataset("AI-MO/NuminaMath-TIR")
    ds.save_to_disk(os.path.join(RAW_DIR, "numina_math_tir"))
    print(f"  Saved {len(ds['train'])} train examples.")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    download_numina_math()
    download_numina_tir()
    print("All datasets downloaded.")
