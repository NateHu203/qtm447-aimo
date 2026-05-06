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
    """Baseline vs Round 1 SFT vs Round 2 SFT on val_200 (in-distribution)."""
    labels = ["Zero-shot\nbaseline", "SFT Round 1\n(r=64, lr=2e-4)", "SFT Round 2\n(r=16, lr=1e-5)"]
    accuracy = [57.0, 67.0, 55.0]
    colors = [GRAY, EMORY_BLUE, EMORY_GOLD]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(labels, accuracy, color=colors, width=0.55)

    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.0, f"{acc:.1f}%",
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
    """Baseline vs Round 1 vs Round 2 across val_200, AIME 2024, AIME 2025."""
    benchmarks = ["val_200\n(in-dist)", "AIME 2024\n(OOD)", "AIME 2025\n(clean OOD)"]
    baseline = [57.0, 23.3, 6.7]
    round1 = [67.0, 16.7, 6.7]
    # AIME 2025 Round 2 is incomplete (2/30); use None to skip plotting that bar.
    round2 = [55.0, 3.3, None]

    x = np.arange(len(benchmarks))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    b1 = ax.bar(x - width, baseline, width, label="Zero-shot baseline", color=GRAY)
    b2 = ax.bar(x, round1, width, label="SFT Round 1 (r=64)", color=EMORY_BLUE)
    # Plot Round 2 bars only where data is complete
    r2_x = [x[i] + width for i in range(len(round2)) if round2[i] is not None]
    r2_vals = [v for v in round2 if v is not None]
    b3 = ax.bar(r2_x, r2_vals, width, label="SFT Round 2 (r=16)", color=EMORY_GOLD)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Mark the missing Round 2 AIME 2025 bar
    ax.text(x[2] + width, 2.0, "n/a", ha="center", fontsize=10,
            color=GRAY, style="italic")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("OOD Generalization: In-Distribution vs. Held-Out AIME",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=10)
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

    fig, ax = plt.subplots(figsize=(8, 3))
    sorted_cats = sorted(wrong_cats.items(), key=lambda x: -x[1])
    labels = [c.replace("_", " ") for c, _ in sorted_cats]
    counts = [v for _, v in sorted_cats]

    bars = ax.barh(labels, counts, color=EMORY_GOLD)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
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
