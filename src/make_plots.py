"""
Generate poster-ready plots from evaluation results.

Usage (run locally or in Colab):
    python src/make_plots.py

Outputs PNG files to results/plots/.
"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Emory colors for consistency with poster
EMORY_BLUE = "#012169"
EMORY_GOLD = "#B58500"
GRAY = "#555555"


def plot_eval_v1_vs_v2():
    """Bar chart: baseline and SFT under eval v1 (buggy) vs eval v2 (fixed)."""
    labels = ["Zero-shot\nbaseline", "SFT Round 1\n(LoRA r=64)"]
    eval_v1 = [36.0, 34.0]
    eval_v2 = [57.0, 67.0]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - width / 2, eval_v1, width, label="Eval v1 (original)", color=GRAY)
    b2 = ax.bar(x + width / 2, eval_v2, width, label="Eval v2 (corrected)", color=EMORY_BLUE)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.0f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Accuracy on val_200", fontsize=12)
    ax.set_title("Eval methodology changed the result entirely", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 80)
    ax.axhline(45, color="red", linestyle="--", alpha=0.4, label="Target (45%)")
    ax.legend(loc="upper left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "eval_v1_vs_v2.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_error_breakdown(errors_json="results/sft_round1_errors.json"):
    """Pie/bar chart of failure categories from analyze_errors.py output."""
    path = Path(errors_json)
    if not path.exists():
        print(f"Skipping error breakdown (file not found: {path})")
        return

    with open(path) as f:
        records = json.load(f)

    cats = Counter(r["category"] for r in records)
    wrong_cats = {k: v for k, v in cats.items() if k != "correct"}
    total = len(records)
    correct = cats.get("correct", 0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sorted_cats = sorted(wrong_cats.items(), key=lambda x: -x[1])
    labels = [c.replace("_", " ") for c, _ in sorted_cats]
    counts = [v for _, v in sorted_cats]

    bars = ax.barh(labels, counts, color=EMORY_GOLD)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{count} ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_xlabel(f"Count (out of {total} problems)", fontsize=11)
    ax.set_title(f"SFT failure breakdown  |  {correct}/{total} correct ({100 * correct / total:.1f}%)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    out = PLOTS_DIR / "error_breakdown.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_ood_comparison(val200_base=57.0, val200_sft=67.0, aime_base=None, aime_sft=None):
    """Bar chart: in-distribution vs OOD benchmark comparison."""
    if aime_base is None or aime_sft is None:
        print("Skipping OOD plot (AIME numbers not provided — edit this function after running AIME eval)")
        return

    benchmarks = ["val_200\n(in-distribution)", "AIME 2023+24\n(OOD)"]
    baseline = [val200_base, aime_base]
    sft = [val200_sft, aime_sft]

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - width / 2, baseline, width, label="Zero-shot baseline", color=GRAY)
    b2 = ax.bar(x + width / 2, sft, width, label="SFT Round 1", color=EMORY_BLUE)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.0f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Generalization: in-distribution vs held-out", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylim(0, max(max(baseline), max(sft)) + 15)
    ax.legend(loc="upper right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "ood_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_eval_v1_vs_v2()
    plot_error_breakdown()
    plot_ood_comparison()  # edit args after AIME eval
