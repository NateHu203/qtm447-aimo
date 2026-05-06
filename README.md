# AIMO — Fine-tuning a 7B Math LLM for Olympiad Reasoning

QTM 447 final project. Authors: Nate Hu, Philip Wang, William Xu (Emory University).

We fine-tune `Qwen/Qwen2.5-Math-7B-Instruct` with LoRA on filtered NuminaMath-CoT olympiad problems, evaluate against held-out AIME benchmarks, and document a diagnostic protocol that distinguishes evaluation artifacts from training regressions.

---

## Headline results

Under **corrected evaluation** (ChatML format, `max_new_tokens=2048`, numeric-tolerant comparison):

| Benchmark | n | Baseline | Round 1 SFT (r=64) | Round 2 SFT (r=16) |
|---|---|---|---|---|
| `val_200` (in-distribution) | 200 | 57.0% | **67.0%** | 55.0% |
| AIME 2024 (partial pretraining overlap) | 30 | 23.3% | 16.7% | 3.3% |
| AIME 2025 (clean OOD) | 30 | 6.7% | 6.7% | 10.0% |

Round 1 (LoRA r=64, lr=2e-4) gains +10 in-distribution but loses 6.7 points on AIME 2024 — catastrophic forgetting on a benchmark where the base model already had pretraining-derived competence. Round 2 (r=16, lr=1e-5, bf16) tested whether conservative hyperparameters alone reduce the forgetting. The answer is asymmetric: Round 2 collapsed AIME 2024 further (3.3%) while directionally improving AIME 2025 (10.0%, within noise at n=30) and dropping val_200 to 55.0%. Round 2 is less specialized than Round 1, which means less overfitting to NuminaMath's style but also more disruption of pre-existing capability. The likely dominant cause across both rounds is the training/eval format mismatch (training uses plain text, evaluation uses ChatML); Round 1's larger adapter could absorb the mismatch by force-fitting, Round 2's could not. The structural format fix is the priority for any future round, not hyperparameter conservatism.

Under the original (buggy) evaluation, the Round 1 model appeared to regress by 2 points on val_200. Three evaluation bugs (token truncation, strict string equality, prompt-format mismatch) had silently inflated failure counts for both models. Diagnostic inspection of 10 raw outputs surfaced this in 15 minutes of free compute. A 12-point swing in measured effect came from the evaluation protocol alone.

For the full narrative, methodology, and what was learned, see [experiments/](experiments/).

---

## What's in this repository

```
qtm447-aimo/
├── README.md                       # this file
├── experiments/                    # narrative records of each round
│   ├── README.md                   # timeline + cross-cutting findings
│   ├── round_1.md                  # Round 1 SFT + diagnostic + OOD analysis
│   └── round_2.md                  # Round 2 hyperparameter-only retrain
├── configs/                        # training hyperparameter YAMLs
│   ├── sft_7b.yaml                 # Round 1 (lr=2e-4, r=64, fp16)
│   ├── sft_7b_v2.yaml              # Round 2 (lr=1e-5, r=16, bf16)
│   └── sft_7b_t4.yaml              # T4 fallback config (4-bit base)
├── data/
│   ├── prepare.py                  # NuminaMath-CoT/TIR filter pipeline
│   ├── prepare_aime.py             # AIME 2024+2025 OOD benchmark builder
│   ├── raw/                        # downloaded datasets (gitignored)
│   └── processed/                  # tokenized JSONL splits (gitignored)
├── src/
│   ├── model.py                    # base model + LoRA loader
│   ├── dataset.py                  # data loading + (unused) prompt-only mask
│   ├── train.py                    # SFTTrainer entry point
│   ├── evaluate.py                 # batched generation + boxed-answer extraction
│   ├── analyze_errors.py           # per-problem error categorization
│   ├── inspect_generations.py      # diagnostic: print N raw outputs (plain vs ChatML)
│   └── make_plots.py               # poster-ready PNGs from results JSON
├── scripts/
│   ├── colab_setup.sh              # Colab session bootstrap (mount, link, install)
│   └── download_data.py            # pull datasets from HuggingFace Hub
├── results/                        # data artifacts only
│   ├── plots/                      # PNGs for poster
│   ├── *_errors.json               # per-problem error analyses
│   └── tables.tex                  # LaTeX tables for write-up
├── pyproject.toml + uv.lock        # local dependencies (uv-managed)
├── requirements.txt                # snapshot for non-uv users
├── requirements-colab.txt          # Colab-specific (no torch/pandas — Colab pre-installs)
└── .env.example                    # secrets template
```

---

## Local setup (first time)

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12+, Git.

```bash
git clone https://github.com/natehu203/qtm447-aimo.git
cd qtm447-aimo
uv sync
cp .env.example .env
# Edit .env: HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN
```

Local machine is for editing, plotting, and lightweight inference only. Training runs on Colab.

---

## Colab setup (every session)

Colab's `/content/` is wiped between sessions; Drive (`MyDrive/AIMO/`) persists. The workflow re-clones the repo each session and pulls latest:

```python
# Cell 1: mount Drive (force-remount if it's stale from a prior session)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

```python
# Cell 2: load secrets from your Drive .env
import os
for line in open('/content/drive/MyDrive/AIMO/.env').read().strip().split('\n'):
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()
```

```python
# Cell 3: clone + setup
token = os.environ['GITHUB_TOKEN']
!git clone https://{token}@github.com/natehu203/qtm447-aimo.git /content/qtm447-aimo
%cd /content/qtm447-aimo
!cp /content/drive/MyDrive/AIMO/.env .env
!bash scripts/colab_setup.sh
```

After this, the `./checkpoints` and `./data/processed` paths inside the repo are symlinked to Drive, and dependencies are installed. Subsequent code edits flow via `git push` (local) → `git pull` (Colab terminal).

---

## Reproducing the key results

The intended workflow, in order:

### 1. Build the datasets (one-time)

```bash
python scripts/download_data.py        # pulls NuminaMath-CoT + TIR to data/raw/
python data/prepare.py                  # filters + splits → data/processed/{train,val,val_200}.jsonl
python data/prepare_aime.py             # AIME 2024+2025 → data/processed/val_aime.jsonl
```

### 2. Train (Round 1)

```bash
python src/train.py --config configs/sft_7b.yaml
```

W&B project: `xhu03204_1/huggingface`. ~9 hours on a single A100. Checkpoints save to Drive every 200 steps; a final adapter is saved to `./checkpoints/lora-final`.

### 3. Evaluate

```bash
# in-distribution
python src/evaluate.py \
  --model_path ./checkpoints/lora-final \
  --config configs/sft_7b.yaml \
  --data_path data/processed/val_200.jsonl \
  --quantize \
  --output results/sft_round1_val200.json

# OOD with per-problem error categorization
python src/analyze_errors.py \
  --model_path ./checkpoints/lora-final \
  --config configs/sft_7b.yaml \
  --data_path data/processed/val_aime.jsonl \
  --quantize \
  --output results/sft_round1_aime_errors.json
```

To reproduce the **baseline** numbers, pass `--model_path Qwen/Qwen2.5-Math-7B-Instruct` instead of an adapter path. The eval scripts auto-detect adapter directories vs. full-model HF IDs.

### 4. Diagnostic inspection (the cheapest, highest-value step)

Before retraining, print 10 raw model outputs side-by-side under both prompt formats:

```bash
python src/inspect_generations.py \
  --model_path ./checkpoints/lora-final \
  --config configs/sft_7b.yaml \
  --data_path data/processed/val_200.jsonl \
  --n 10 --quantize \
  --output results/diagnostic.json
```

This is what surfaced the eval bug. It costs nothing and is documented in [experiments/round_1.md §3](experiments/round_1.md).

### 5. Plots and tables

```bash
python src/make_plots.py
```

Outputs PNGs to [results/plots/](results/plots/) and reads from JSON files in [results/](results/). LaTeX tables live in [results/tables.tex](results/tables.tex).

---

## Compute & infrastructure notes

- **A100 vs T4.** Training requires A100; inference fits on T4 with 4-bit quantization (`--quantize` flag in `evaluate.py` / `analyze_errors.py`).
- **Drive caching.** `HF_HOME` and `HF_DATASETS_CACHE` should point inside `MyDrive/AIMO/` so model weights and tokenized datasets persist across Colab sessions. Configured in `.env` and `scripts/colab_setup.sh`.
- **Resume support.** Both `src/train.py` (auto-resumes from latest `checkpoint-N` subdirectory) and `src/analyze_errors.py` (skips problems already in the output `.jsonl`) recover from interrupted Colab sessions.
- **Run training from a Colab terminal**, not a notebook cell. The progress bar flood from `Trainer` can crash the notebook tab; a terminal handles it cleanly. Pipe output to a log on Drive.

---

## Where things came from

- **Base model:** [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) (Alibaba Cloud, 2024)
- **Training data:** [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) and [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (Numina, 2024)
- **OOD benchmarks:** [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) and [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)
- **Method (LoRA):** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
- **Competition:** [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) (Kaggle)
- **Methodological guidance** on chat templates, completion-only loss, and learning-rate selection: Dr. McAlister (Emory QTM 447), personal communication.
