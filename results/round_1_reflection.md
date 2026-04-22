# Round 1: SFT Baseline — Results & Reflection

**Date:** 2026-04-21
**Run:** `qwen2.5-math-7b-lora-r64`
**W&B:** https://wandb.ai/xhu03204_1/huggingface/runs/6hpooror

---

## 1. Current setup

### Model
- Base: `Qwen/Qwen2.5-Math-7B-Instruct` (already instruct-tuned by Qwen team)
- Adapter: LoRA r=64, alpha=128, dropout=0.05
- Target modules: q/k/v/o_proj + gate/up/down_proj (all 7 projections)
- Trainable params: ~80M

### Data
- Source: NuminaMath-CoT + NuminaMath-TIR
- Filter: olympiad-level sources, integer answers only
- Split: 72,283 train / 3,805 val (95/5 from seed=42)
- Format: `"Problem: {problem}\n\nSolution: {solution}"` (plain text — **NOT** Qwen chat template)

### Training
- Effective batch size: 32 (4/device × 8 grad accum)
- LR: 2e-4, cosine schedule, 100 warmup steps
- Precision: fp16, gradient checkpointing
- Epochs: 1 (2,259 steps), ~9 hrs on A100
- Loss: computed on full sequence (problem + solution), **no completion-only masking** despite `dataset.py` having the logic
- Checkpoints every 200 steps to Drive, last 3 kept

### Evaluation
- Metric: exact-match on integer answer from `\boxed{...}`
- Eval set: `val_200.jsonl` (200 held-out problems)
- Prompt format: same plain-text template as training
- Generation: greedy, `max_new_tokens=512`, `n_samples=1`

---

## 2. Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Zero-shot Qwen2.5-Math-7B-Instruct | **36.0%** (72/200) | Week 1 baseline |
| SFT LoRA r=64 (round 1) | **34.0%** (68/200) | *Below baseline* |

### Training signal (W&B)
- Train loss: 0.77 → 0.21 (very aggressive drop)
- Eval loss: plateaued around 0.43 by step 1200
- Mean token accuracy: 84% → 87.6%
- Train/eval loss gap: 2× (0.21 vs 0.43) — overfitting

### Interpretation
Training loss said "the model is learning." Accuracy eval said "no, it regressed." Classic disconnect between token-level objective and task-level metric.

---

## 3. Problems identified

### P1 — Chat template mismatch (likely biggest)
Qwen2.5-Math-Instruct was post-trained with:
```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
```

We used `"Problem: ...\n\nSolution:"`. This breaks the model's instruction-following and un-teaches the `\boxed{}` convention our answer extractor depends on.

### P2 — Loss on full sequence (should be completion-only)
[src/dataset.py:40-43](src/dataset.py#L40-L43) has correct prompt-masking logic — **but it's never used**. The active training path in [src/train.py:29-32](src/train.py#L29-L32) builds a text-only `Dataset` and lets SFTTrainer tokenize everything. Roughly 40-50% of gradient signal is teaching the model to predict problem statements. Useless, and dilutes the solution-learning signal.

### P3 — Learning rate too high for instruct model
`lr=2e-4` is the standard for fine-tuning base models. For an already-instruct-tuned model, community convention is **5e-5 or lower** — you're preserving careful RLHF/DPO work, not overwriting from scratch. Our LR was overwriting Qwen's tuning.

### P4 — Data quality not verified
NuminaMath-CoT aggregates many sources with inconsistent solution quality. Our filter ("olympiad + integer answer") is structural, not qualitative. Many solutions likely:
- Don't end with `\boxed{}`
- Use informal reasoning the base model already surpasses
- Have arithmetic errors
Training on CoT *below* the base model's quality degrades it.

### P5 — Eval noise
- 200 samples → ±3.4% CI at 95%
- 34% vs 36% is statistically indistinguishable
- `n_samples=1` greedy + `max_new_tokens=512` — many solutions truncated before final answer

### P6 — Aggressive LoRA rank
r=64, alpha=128 → scaling factor 2.0 across all 7 projection layers = ~80M trainable params. High capacity to overfit 72K examples, especially in combination with P3.

---

## 4. Next steps (ranked by cost)

### Step 1: Diagnose before deciding (free, ~30 min A100)
Print raw generations from the SFT checkpoint alongside expected answers. Determine:
- Is the model emitting `\boxed{}`?
- Getting cut off at 512 tokens?
- Producing coherent reasoning vs. gibberish?

This tells us whether the model regressed or just wasn't evaluated properly.

### Step 2: Proper eval (free, ~30 min A100)
Re-run eval with:
- `max_new_tokens=2048`
- `n_samples=8` with majority voting
- Apply Qwen chat template at inference

This alone may close the gap or reveal the model is actually fine.

### Step 3: Round 2 retrain (if Step 1-2 confirm regression, ~2-3 hrs A100)
Fix the structural issues, not hyperparameters:
1. **Use Qwen chat template in training data** (`tokenizer.apply_chat_template`)
2. **Completion-only loss** (wire in existing masking or use `DataCollatorForCompletionOnlyLM`)
3. **LR: 5e-5** (down from 2e-4)
4. **LoRA r=16 or r=32** (down from 64)
5. **Early stop** at step 500-1000 (eval loss plateaued at 1200 anyway)
6. Save config as `configs/sft_7b_v2.yaml` for reproducibility

### Step 4: Data quality (optional, if Round 2 still underwhelms)
- Filter NuminaMath to solutions containing `\boxed{}`
- Verify extracted answer matches the dataset's `answer` field
- Consider curated alternatives: Open-R1 / OpenMathReasoning

### Step 5: Tool-use / TIR (Week 3 plan)
Integrate Python REPL at inference. Historically the biggest olympiad-math gains come from tool use, not more SFT. NuminaMath-TIR is already filtered into our training set.

---

## 5. Budget discipline

Compute units are finite. Rule for Round 2+:
- **Diagnose → hypothesize → fix one thing → measure.**
- No "try different LR and see what happens" runs.
- Every re-train must target a named problem from this document.
- Re-read this file before committing to any new run.


