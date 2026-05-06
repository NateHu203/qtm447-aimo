# Experiments

Narrative records of each training/evaluation round. Each round is documented in the same format so they can be read together.

## Timeline

| Round | Status | Headline |
|---|---|---|
| [1](round_1.md) | Complete | LoRA SFT (r=64, lr=2e-4): +10 points over baseline on val_200 under corrected evaluation. Catastrophic forgetting on OOD AIME 2024 (−6.7 points). |
| [2](round_2.md) | Complete | Hyperparameter-only retrain (r=16, lr=1e-5, bf16). **Asymmetric OOD result:** AIME 2024 collapsed to 3.3% (vs baseline 23.3%) while AIME 2025 directionally improved to 10.0% (vs baseline 6.7%, n.s.). val_200 dropped to 55.0%. Round 2 is less specialized than Round 1 — less overfit to NuminaMath but more disruption of pre-trained capability. |

## Reading order

If you're trying to understand how the project evolved, read in order:

1. **[round_1.md](round_1.md)** — Initial training run; evaluation methodology diagnostic; primary findings.
2. **[round_2.md](round_2.md)** — Conservative-hyperparameter retrain motivated by Round 1's OOD analysis.

## Cross-cutting findings

- **Evaluation methodology can outweigh training methodology in measured effect.** Three eval bugs (`max_new_tokens` truncation, strict string equality, prompt-format mismatch) collectively flipped the measured SFT effect from −2 to +10 points — a 12-point swing from methodology alone. (See round_1.md §3.)
- **In-distribution gains do not imply out-of-distribution gains.** Round 1's +10 on val_200 coexisted with −6.7 on AIME 2024 (catastrophic forgetting of pre-existing capability) and 0.0 on AIME 2025 (floor effect). This motivated Round 2.
- **Hyperparameter conservatism alone does not prevent forgetting — and produces asymmetric OOD effects.** Round 2 (r=16, lr=1e-5, bf16) collapsed on AIME 2024 (where the base had pretraining-derived capability) but directionally improved on AIME 2025 (clean OOD, within noise). val_200 dropped to 55.0%. Both rounds train on plain text and evaluate on ChatML; Round 1 absorbed this mismatch with a larger adapter, Round 2 could not. Structural format fixes are likely required for any further gain. (See round_2.md §5.)
- **Diagnostic discipline saves compute.** A 15-minute free-tier inspection of 10 raw model outputs prevented an unnecessary multi-hour retraining run.

## Conventions used in each round

- **Sample sizes** are reported with Wilson 95% confidence intervals where they appear in tables.
- **Eval v1** refers to the original (buggy) protocol; **eval v2** refers to the corrected protocol used from Round 1's diagnostic onward (ChatML formatting, `max_new_tokens=2048`, numeric-tolerant comparison).
- **OOD benchmarks**: AIME 2024 (30 problems, possibly partially in pretraining data) and AIME 2025 (30 problems, guaranteed post-cutoff).
- **Compute**: Google Colab Pro A100 / T4 (with 4-bit quantization for inference on T4).
