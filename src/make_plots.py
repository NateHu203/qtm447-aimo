"""
Generate poster-ready plots from evaluation results.

Usage:
    python src/make_plots.py

Outputs PNG files to results/plots/.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EMORY_BLUE = "#012169"
EMORY_GOLD = "#B58500"
GRAY = "#555555"


def plot_main_results():
    """Baseline vs SFT on val_200 (in-distribution)."""
    labels = ["Zero-shot\nbaseline", "SFT Round 1\n(LoRA r=64)"]
    accuracy = [57.0, 67.0]
    colors = [GRAY, EMORY_BLUE]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(labels, accuracy, color=colors, width=0.45)

    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.0, f"{acc:.0f}%",
                ha="center", fontsize=14, fontweight="bold")

    ax.set_ylabel("Accuracy on val_200 (200 problems)", fontsize=12)
    ax.set_title("Main Results: In-Distribution Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "main_results.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_ood_comparison():
    """Baseline vs SFT across val_200 (in-dist), AIME 2024, AIME 2025."""
    benchmarks = ["val_200\n(in-dist)", "AIME 2024\n(OOD)", "AIME 2025\n(clean OOD)"]
    baseline = [57.0, 23.3, 6.7]
    sft = [67.0, 16.7, 6.7]

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    b1 = ax.bar(x - width / 2, baseline, width, label="Zero-shot baseline", color=GRAY)
    b2 = ax.bar(x + width / 2, sft, width, label="SFT Round 1", color=EMORY_BLUE)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("OOD Generalization: In-Distribution vs. Held-Out AIME",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "ood_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_error_breakdown(errors_json="results/sft_aime_errors_2.json", label="AIME"):
    """Horizontal bar chart of failure categories."""
    path = ROOT / errors_json
    if not path.exists():
        print(f"Skipping error breakdown (file not found: {path})")
        return

    with open(path) as f:
        records = json.load(f)

    cats = Counter(r["category"] for r in records)
    wrong_cats = {k: v for k, v in cats.items() if k != "correct"}
    total = len(records)
    correct = cats.get("correct", 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sorted_cats = sorted(wrong_cats.items(), key=lambda x: -x[1])
    labels = [c.replace("_", " ") for c, _ in sorted_cats]
    counts = [v for _, v in sorted_cats]

    bars = ax.barh(labels, counts, color=EMORY_GOLD)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{count} ({pct:.1f}%)", va="center", fontsize=11)

    ax.set_xlabel(f"Count (n = {total})", fontsize=11)
    ax.set_title(f"Error Analysis: SFT Failures on {label}   ({correct}/{total} correct, {100 * correct / total:.1f}%)",
                 fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    out = PLOTS_DIR / "error_breakdown.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_main_results()
    plot_ood_comparison()
    plot_error_breakdown()
