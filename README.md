# AIMO — QTM 447 Final Project

Fine-tuning math reasoning LLMs for the [AI Mathematical Olympiad (AIMO3) Kaggle challenge](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3).

---

## Codebase Structure

```
qtm447-aimo/
├── configs/
│   ├── sft_7b.yaml           # Qwen2.5-Math-7B LoRA hyperparams
│   └── sft_35b_moe.yaml      # Qwen3-30B-A3B (MoE) hyperparams
├── data/
│   ├── raw/                  # downloaded datasets (gitignored)
│   ├── processed/            # tokenized JSONL splits (gitignored)
│   └── prepare.py            # full data pipeline
├── src/
│   ├── dataset.py            # MathDataset + prompt masking
│   ├── model.py              # model loading + LoRA config
│   ├── train.py              # SFTTrainer entry point
│   ├── tool_use.py           # Python REPL tool integration
│   └── evaluate.py           # exact match + majority vote scoring
├── scripts/
│   ├── colab_setup.sh        # Colab session bootstrap
│   └── download_data.py      # pull datasets from HuggingFace
├── inference/
│   └── kaggle_submission.py  # self-contained offline Kaggle inference
├── notebooks/
│   ├── 01_eda.ipynb          # data exploration
│   └── 02_error_analysis.ipynb
├── .env.example              # copy to .env and fill in tokens
├── pyproject.toml            # uv-managed dependencies
└── uv.lock
```

---

## Local Setup (First Time)

**Requirements:** [uv](https://docs.astral.sh/uv/), Python 3.12+, Git

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/qtm447-aimo.git
cd qtm447-aimo

# 2. Install dependencies
uv sync

# 3. Set up environment variables
cp .env.example .env
# Edit .env and fill in:
#   HF_TOKEN    — from huggingface.co/settings/tokens
#   WANDB_API_KEY — from wandb.ai/settings
```

To activate the virtual environment in your terminal:
```bash
source "/Users/nate/Desktop/QTM 447/.venv/bin/activate"
```

In Cursor/VSCode, select the interpreter at:
`/Users/nate/Desktop/QTM 447/.venv/bin/python`

---

## Google Colab Setup (First Time)

Run these cells at the top of your Colab notebook the **first time only**:

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Clone the repo
!git clone https://github.com/YOUR_USERNAME/qtm447-aimo.git
%cd qtm447-aimo

# Cell 3 — Bootstrap environment
!bash scripts/colab_setup.sh

# Cell 4 — Create your .env with tokens
%%writefile .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxx
```

---

## Daily Workflow

### On your local machine (Cursor)

```
Edit code  →  git add  →  git commit  →  git push
```

You do **not** need a GPU locally. Just write and test code structure, edit configs, update notebooks.

### On Colab (training / eval)

Every new session, run:

```python
# Cell 1 — Mount Drive + pull latest code
from google.colab import drive
drive.mount('/content/drive')

%cd /content/qtm447-aimo
!git pull origin main
!bash scripts/colab_setup.sh
```

Then run training:

```python
!python src/train.py --config configs/sft_7b.yaml
```

Checkpoints auto-save to `Google Drive/AIMO/checkpoints/` every 100 steps via the symlink created by `colab_setup.sh`.

### Syncing results back

After a training run, push logs or result files from Colab:

```bash
git add configs/ notebooks/
git commit -m "add eval results for run xyz"
git push origin main
```

Model weights (`.safetensors`, `.bin`) are gitignored — they stay on Drive and Kaggle Datasets only.

---

## Running Training

```bash
# 7B model (primary)
python src/train.py --config configs/sft_7b.yaml

# 35B MoE (stretch goal)
python src/train.py --config configs/sft_35b_moe.yaml
```

Monitor training at [wandb.ai](https://wandb.ai).

## Running Evaluation

```bash
# Greedy (pass@1)
python src/evaluate.py \
  --model_path ./checkpoints/lora-final \
  --config configs/sft_7b.yaml \
  --data_path data/processed/val.jsonl

# Majority vote over 8 samples (maj@8)
python src/evaluate.py \
  --model_path ./checkpoints/lora-final \
  --config configs/sft_7b.yaml \
  --data_path data/processed/val.jsonl \
  --n_samples 8
```

## Downloading Data

```bash
python scripts/download_data.py
# Then run the data pipeline:
python data/prepare.py
```

---

## Experiment Tracking

All training runs log to [Weights & Biases](https://wandb.ai). Each run is named in the config (e.g. `qwen2.5-math-7b-lora-r64`). The dashboard tracks loss curves, GPU usage, and hyperparams across runs — accessible by all teammates.

---

## Kaggle Submission

1. Merge LoRA adapter into the base model and upload to a Kaggle Dataset
2. In your Kaggle notebook, set `MODEL_PATH` in `inference/kaggle_submission.py` to point to the dataset
3. The submission script is fully offline-compatible (no internet required in Kaggle)

---

## Models

| Model | Active Params | Use case |
|---|---|---|
| `Qwen/Qwen2.5-Math-7B-Instruct` | 7B | Primary — fits on 1x A100 40GB with LoRA |
| `Qwen/Qwen3-30B-A3B` | 3B active / 30B total | Stretch — MoE, higher capacity |
