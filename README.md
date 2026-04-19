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

## Google Colab — First Time Only

> Only run this once. After the repo is cloned to `/content/qtm447-aimo`, future sessions just pull.

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. Read GitHub token from your Drive .env
token = open('/content/drive/MyDrive/AIMO/.env').read().split('GITHUB_TOKEN=')[1].split()[0]
```

```python
# 3. Clone the private repo using the token
!git clone https://{token}@github.com/YOUR_USERNAME/qtm447-aimo.git /content/qtm447-aimo
%cd /content/qtm447-aimo
```

```python
# 4. Bootstrap (installs deps, symlinks Drive folders)
!bash scripts/colab_setup.sh
```

```python
# 5. Copy secrets into project
!cp /content/drive/MyDrive/AIMO/.env /content/qtm447-aimo/.env
```

---

## Google Colab — Every Session After That

> `/content/` resets when a session ends. The repo is gone, but Drive (checkpoints, data) is not.

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. Re-clone and pull latest code
token = open('/content/drive/MyDrive/AIMO/.env').read().split('GITHUB_TOKEN=')[1].split()[0]
!git clone https://{token}@github.com/YOUR_USERNAME/qtm447-aimo.git /content/qtm447-aimo
%cd /content/qtm447-aimo
!git remote set-url origin https://{token}@github.com/YOUR_USERNAME/qtm447-aimo.git
```

```python
# 3. Bootstrap + restore secrets
!bash scripts/colab_setup.sh
!cp /content/drive/MyDrive/AIMO/.env .env
```

Then run training:

```python
!python src/train.py --config configs/sft_7b.yaml
```

Checkpoints auto-save to `MyDrive/AIMO/checkpoints/` every 100 steps via the symlink in `colab_setup.sh`.

---

## Daily Workflow

### On your local machine (Cursor)

```
Edit code  →  git add  →  git commit  →  git push
```

You do **not** need a GPU locally. Edit configs, write src/ files, update notebooks — then push.

### What lives where

| Location | What's stored | Persists? |
|---|---|---|
| `/content/qtm447-aimo/` | Code (re-cloned each session) | No — ephemeral |
| `MyDrive/AIMO/checkpoints/` | LoRA adapter weights | Yes — Drive |
| `MyDrive/AIMO/data/processed/` | Tokenized JSONL datasets | Yes — Drive |
| `MyDrive/AIMO/.env` | All secrets/tokens | Yes — Drive |
| GitHub | Source code | Yes — git |

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
