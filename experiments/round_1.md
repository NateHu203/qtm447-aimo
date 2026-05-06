# Round 1 — LoRA SFT on Qwen2.5-Math-7B-Instruct

| | |
|---|---|
| **Status** | Complete |
| **Date(s)** | Training: 2026-04-19 to 2026-04-21 (resumed across sessions). Diagnostic + corrected eval: 2026-04-22 to 2026-04-23. |
| **Run name** | `qwen2.5-math-7b-lora-r64` |
| **W&B** | https://wandb.ai/xhu03204_1/huggingface/runs/6hpooror |
| **Adapter location** | `MyDrive/AIMO/checkpoints/lora-final` |
| **Headline** | Under corrected evaluation, the SFT model gains 10 points over the zero-shot baseline on val_200 (57% → 67%). The same fine-tuning degrades out-of-distribution capability on AIME 2024 (−6.7 points), exhibiting a catastrophic-forgetting signature. |

---

## 1. Setup

### Model
- **Base:** `Qwen/Qwen2.5-Math-7B-Instruct` (already RLHF-aligned by the Qwen team)
- **Adapter:** LoRA, r=64, α=128, dropout=0.05
- **Target modules:** all seven projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable params:** ~80M (vs. 7B frozen base)

### Data
- **Sources:** NuminaMath-CoT and NuminaMath-TIR
- **Filter:** olympiad-level sources, integer-valued answers
- **Split:** 72,283 train / 3,805 validation (95/5, seed=42)
- **Format used in training:** plain text — `"Problem: {problem}\n\nSolution: {solution}"` (does **not** match Qwen's post-training ChatML template)

### Training
- **Effective batch size:** 32 (per-device 4 × gradient accumulation 8)
- **Learning rate:** 2e-4, cosine schedule, 100 warmup steps
- **Precision:** fp16 with gradient checkpointing
- **Epochs:** 1 (2,259 optimization steps), ~9 hrs on a single A100
- **Loss:** standard token-level cross-entropy on next-token prediction, applied to the **full sequence** (problem + solution; no completion-only masking)
- **Checkpointing:** every 200 steps to Drive, last 3 retained
- **Config:** [`configs/sft_7b.yaml`](../configs/sft_7b.yaml)

### Evaluation (initial — "v1", later identified as buggy)
- **Generation:** plain-text template at inference, `max_new_tokens=512`, greedy, `n_samples=1`
- **Extraction:** regex `\\boxed\{([^}]+)\}`, last match
- **Comparison:** strict string equality after `.strip()`
- **Sample:** `val_200.jsonl` (200 problems, ±3.4% Wilson 95% CI)

### Evaluation (corrected — "v2", used from diagnostic onward)
- **Generation:** Qwen's ChatML template via `tokenizer.apply_chat_template`, `max_new_tokens=2048`, greedy
- **Extraction:** same regex
- **Comparison:** numeric-tolerant equality (`"5.0" ≡ "5"`)
- **Implementation:** [`src/evaluate.py`](../src/evaluate.py)

---

## 2. Results

### Headline: in-distribution accuracy

| Model | val_200 (n=200) | 95% CI |
|---|---|---|
| Zero-shot Qwen2.5-Math-7B-Instruct | 57.0% | [53.6, 60.4] |
| **SFT LoRA r=64 (this round)** | **67.0%** | [63.6, 70.4] |
| Δ | **+10.0** | (p < 0.01) |

### Out-of-distribution (AIME 2024 + 2025, n=60 total)

| Benchmark | n | Baseline | SFT | Δ |
|---|---|---|---|---|
| AIME 2024 (likely partial overlap with pretraining) | 30 | 23.3% | 16.7% | **−6.7** |
| AIME 2025 (post-pretraining cutoff, clean OOD) | 30 | 6.7% | 6.7% | 0.0 |
| AIME combined | 60 | 15.0% | 11.7% | −3.3 |

### Diagnostic finding (eval methodology)

Both models re-evaluated under identical conditions with each protocol:

| Model | Eval v1 (buggy) | Eval v2 (corrected) |
|---|---|---|
| Baseline | 36.0% | 57.0% |
| SFT | 34.0% | 67.0% |
| **SFT − Baseline** | **−2.0** | **+10.0** |

The same evaluation procedure changes that made the baseline go from 36% to 57% made the SFT model go from 34% to 67%. Three bugs in the v1 protocol were responsible:

1. **`max_new_tokens=512`** truncated olympiad solutions before reaching `\boxed{...}` — many correct solutions were 600–900 tokens.
2. **Strict string equality** failed on numerically equivalent outputs (e.g., `"5.0" != "5"`).
3. **Plain-text inference template** mismatched Qwen's post-training distribution; the model's `\boxed{}` convention was learned conditional on its ChatML chat template.

### Training-loss trajectory (W&B)

- Start: ~0.77; end: ~0.21 over 2,259 steps
- Eval loss (token-level cross-entropy on val): plateaued near 0.43 by step 1200
- Train/eval loss gap: ~2× — the model was fitting training surface form faster than it was generalizing

---

## 3. What we learned

**(a) Evaluation methodology can outweigh training methodology.** A 12-point swing in the measured SFT effect came from changing only the evaluation protocol: same model, same training, different way of scoring. Diagnostic inspection of ~10 raw outputs costs <1% of a full retraining run and reliably distinguishes real regressions from evaluation artifacts.

**(b) In-distribution gains do not imply OOD gains.** The +10 points on val_200 coexists with a 6.7-point degradation on AIME 2024, a benchmark on which the base model already had measurable competence. SFT erased some of that pre-existing capability. On AIME 2025, where neither model has meaningful capability, SFT neither helps nor hurts. This is the canonical signature of catastrophic forgetting under aggressive fine-tuning of an already-aligned base model.

**(c) Plausible mechanisms for (b).** All three are concrete and testable in Round 2:

- **Aggressive learning rate** (2e-4) on an already-instruct-tuned base overwrites the Qwen team's RLHF alignment (Dr. McAlister, personal communication).
- **Full-sequence loss** instead of completion-only masking spends roughly half the gradient budget teaching the model to predict problem statements rather than solutions ([`src/dataset.py`](../src/dataset.py) implements masking but [`src/train.py`](../src/train.py) does not invoke it).
- **Training-time format mismatch** (plain `"Problem: ... Solution:"` instead of Qwen's ChatML) trains the model in a context where its post-trained behavior, including its `\boxed{}` convention, does not activate.

**(d) Token-level loss is not capability.** The training loss dropped smoothly from 0.77 to 0.21, which we initially read as the model learning. It was learning the surface format of NuminaMath solutions, not olympiad math. The clean way to detect this is to re-evaluate under multiple conditions and look for distribution shift in the gap.

---

## 4. What's next

Round 2 (see [round_2.md](round_2.md)) tested the conservative-hyperparameter portion of the plausible-mechanisms list (smaller LoRA rank, lower learning rate, bf16) in isolation. The result was asymmetric: Round 2 collapsed AIME 2024 further (16.7% → 3.3%), directionally improved AIME 2025 (6.7% → 10.0%, within noise at n=30), and dropped val_200 to 55.0%. Hyperparameter conservatism alone did not close the OOD gap, and on benchmarks where the base model had pre-existing capability (val_200, AIME 2024) it disrupted that capability more than Round 1 did.

This points to the structural fixes (ChatML training format, completion-only loss masking) as the more likely lever for further gain. They are Round 3 priorities, with Round 2's hyperparameters retained — those hyperparameters are likely appropriate *given* the format is correct.

Beyond retraining, the most promising direction by literature precedent is tool-integrated reasoning (Python REPL at inference). NuminaMath-TIR is already in the training set; the missing piece is the inference-time tool-calling loop. This is documented as a future direction rather than executed here.

---

## 5. Reproducibility

- **Training:** `python src/train.py --config configs/sft_7b.yaml`
- **Corrected eval on val_200:** `python src/evaluate.py --model_path <adapter> --config configs/sft_7b.yaml --data_path data/processed/val_200.jsonl`
- **OOD eval on AIME:** same command with `--data_path data/processed/val_aime.jsonl` (built by `python data/prepare_aime.py`)
- **Error breakdown:** `python src/analyze_errors.py --model_path <adapter> --config configs/sft_7b.yaml --data_path <data> --output <out>`
- **Plots:** `python src/make_plots.py` (reads numbers from this document; chart data is in [results/plots/](../results/plots/))
- **Tables:** [results/tables.tex](../results/tables.tex) for LaTeX; numbers are also in this document.
