# Round 2 Plan

**Date drafted:** 2026-04-22
**Parent doc:** [round_1_reflection.md](round_1_reflection.md)
**Rule:** every change below must target a named problem (P1–P6) from Round 1. No scattershot tweaks.

---

## 1. Goals and success criteria

**Primary goal:** Beat the zero-shot baseline of **36.0%** on `val_200.jsonl` by a statistically meaningful margin (≥3.4% above baseline given the ±3.4% CI at 95% → target **≥40%**).

**Stretch goal:** 45%+ (original Week 1 target).

**Non-goals:**
- Maximum possible accuracy. This is a structural-correctness round.
- Trying multiple hyperparameter combinations. Change once, measure, then decide.

**What a success looks like at the metric level:**
- Training loss starts higher than Round 1 (≥0.5–0.7 at step 10) — because we're no longer measuring "how well it learned our made-up format"
- Train/eval loss gap closes (Round 1 had 2× gap = overfit surface form)
- Accuracy ≥ 40% on `val_200` with proper eval

---

## 2. Pre-training diagnostics (do first, ~30 min A100)

Before any retraining, confirm Round 1 hypotheses with free diagnostics. These determine whether we need Round 2 at all, and baseline the comparison.

### D1 — Print raw generations from Round 1 checkpoint (targets: P1, P5)
Write a small script (`src/inspect_generations.py`) that runs the Round 1 SFT model on 10 problems and dumps to JSON:
- prompt (both plain-text and ChatML versions)
- full generated text (no truncation)
- extracted `\boxed{}` match (or `None` if missing)
- gold answer

**Decision rule:**
- If outputs are coherent but no `\boxed{}` → confirms P1. Proceed with Round 2.
- If outputs get cut off at 512 tokens → confirms P5 is a factor. May not need retraining.
- If outputs are incoherent → confirms deeper issue. Proceed with Round 2.

### D2 — Re-eval Round 1 model with proper settings (targets: P5)
Run full 200-problem eval with:
- `max_new_tokens=2048`
- `n_samples=8` majority vote
- **Apply Qwen chat template at inference** (test both with and without to see the ChatML delta)

If accuracy jumps significantly (e.g., to 38%+), it confirms the Round 1 model wasn't as broken as it looked — we were just evaluating it wrong. This changes our Round 2 priority.

---

## 3. Training changes (mapped to Round 1 problems)

### C1 — Switch to ChatML format (targets: P1)
**File:** [src/dataset.py](src/dataset.py) and [src/train.py](src/train.py)

Replace the plain `"Problem: ... Solution:"` template with Qwen's ChatML via `tokenizer.apply_chat_template`.

```python
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def format_example(ex, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["problem"]},
        {"role": "assistant", "content": ex["solution"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)
```

Use the same system prompt + messages structure in [src/evaluate.py](src/evaluate.py) — train/eval must match exactly.

### C2 — Completion-only loss via `DataCollatorForCompletionOnlyLM` (targets: P2)
**File:** [src/train.py](src/train.py)

Pass a response template to the collator so only assistant tokens contribute to loss:

```python
from trl import DataCollatorForCompletionOnlyLM

response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)
# pass collator=... to SFTTrainer
```

**Verify it works:** before training, call `print(tokenizer.decode(labels[labels != -100]))` on one example. Should only show the assistant turn content.

### C3 — Lower LR, smaller LoRA rank (targets: P3, P6)
**File:** [configs/sft_7b_v2.yaml](configs/sft_7b_v2.yaml) (new)

| Param | Round 1 | Round 2 | Why |
|---|---|---|---|
| `learning_rate` | 2e-4 | **1e-5** | Preserve Qwen's RLHF tuning (per Dr. McAlister) |
| `lora.r` | 64 | **16** | Reduce overfit capacity; faster training |
| `lora.alpha` | 128 | **32** | Keep alpha = 2r convention |

### C4 — Curated data + multiple epochs (targets: P4)
**File:** [data/prepare.py](data/prepare.py) — add filtering pass

- **Length filter:** drop examples where formatted sequence > 2048 tokens (counts both train and eval efficiency gains)
- **Quality filter:** keep only examples where `solution` contains `\boxed{<answer>}` and the boxed value matches `ex["answer"]`
- **Target:** ~20–30K curated examples, 2–3 epochs on that subset

This is more compute-efficient than 1 epoch over 72K noisy examples.

### C5 — Efficiency improvements (new, from professor's reply)
**Files:** [src/model.py](src/model.py), [src/train.py](src/train.py), [configs/sft_7b_v2.yaml](configs/sft_7b_v2.yaml)

| Change | File | Where |
|---|---|---|
| `bf16=True` (was fp16) | config | training.bf16 |
| `attn_implementation="flash_attention_2"` with `sdpa` fallback | model.py | `AutoModelForCausalLM.from_pretrained` |
| `packing=True` | train.py | `SFTConfig` |
| `optim="paged_adamw_8bit"` | train.py | `SFTConfig` |
| QLoRA (4-bit base) | model.py | reuse existing `BitsAndBytesConfig` logic |
| Rebalance `batch × grad_accum` after memory frees up | config | target: batch 8 × accum 4, not 4 × 8 |

**Note:** Flash Attention 2 requires `pip install flash-attn`. If it fails to install on Colab, fall back to `sdpa`.

---

## 4. Evaluation changes (targets: P5)

**File:** [src/evaluate.py](src/evaluate.py)

| Param | Round 1 | Round 2 | Why |
|---|---|---|---|
| Prompt format | plain text | **ChatML** | Match training (and match how Qwen base was post-trained) |
| `max_new_tokens` | 512 | **2048** | Olympiad solutions often > 512 tokens |
| `n_samples` | 1 | **8** | Majority voting; reduces single-sample variance |
| Temperature | 1.0 (greedy) | **0.7** (for n_samples > 1) | Standard for majority voting |
| Eval set | `val_200.jsonl` | `val_200.jsonl` | Keep consistent for apples-to-apples |

**Also:** consider evaluating on the **full `val.jsonl` (3,805 problems)** at the end for a tighter confidence interval. This is expensive but the final number deserves it.

---

## 5. Implementation checklist (order of operations)

Execute in this order — each step is a gate for the next:

- [ ] **D1**: Write `src/inspect_generations.py`, dump 10 Round 1 outputs, read them
- [ ] **D2**: Re-eval Round 1 model with ChatML + n_samples=8 + max_new_tokens=2048
- [ ] **Decision gate**: if D2 accuracy jumps to 38%+, reconsider whether Round 2 is needed at all
- [ ] **C1**: Implement ChatML formatting in `dataset.py` and `train.py`; verify with a print
- [ ] **C2**: Wire in `DataCollatorForCompletionOnlyLM`; verify label mask with a print
- [ ] **C3**: Create `configs/sft_7b_v2.yaml` with new LR, rank, alpha
- [ ] **C4**: Update `data/prepare.py` with length + quality filters; save to `data/processed/train_v2.jsonl`
- [ ] **C5**: Add efficiency flags (bf16, flash attn, packing, paged optim, QLoRA)
- [ ] **Code review**: feed `src/train.py` + `src/model.py` + `configs/sft_7b_v2.yaml` to Claude for a setup audit (padding side, dataloader workers, attention masks, etc.)
- [ ] **Smoke test**: run training for 50 steps on a 500-example subset, verify loss decreases and checkpoints save
- [ ] **Full Round 2 run**: 2–3 epochs on curated data
- [ ] **Eval**: run new `evaluate.py` on `val_200.jsonl`, compare to Round 1 and baseline
- [ ] **Write up**: create `results/round_2_reflection.md` in the same format as Round 1

---

## 6. Compute budget

Estimated hours on A100 (Colab Pro):

| Phase | Hours |
|---|---|
| D1 diagnostics | 0.2 |
| D2 proper eval | 0.3 |
| Smoke test | 0.2 |
| Full Round 2 training (20-25K × 3 epochs with efficiency wins) | 3–4 |
| Final eval on `val_200` | 0.3 |
| Optional: full `val.jsonl` (3805 problems) | 2–3 |

**Total: ~7–9 hours A100.** Budget accordingly.

---

## 7. Risk and rollback

### Known risks
- **QLoRA + Flash Attention interaction**: occasionally breaks on specific `bitsandbytes` versions. If smoke test fails, drop QLoRA first (keep flash attn), then drop flash attn to `sdpa` if still broken.
- **Packing + completion-only loss**: needs verification that the response template is found within packed sequences. Check label masks in smoke test.
- **bf16 on T4**: Not supported. If compute falls back to T4, revert to fp16 in config.

### Rollback
If Round 2 accuracy is ≤ Round 1 (34%), the culprit is almost certainly one of the efficiency flags, not the structural fixes. Disable them in this order:
1. Turn off packing
2. Revert flash attn to sdpa
3. Revert to fp16
4. Turn off QLoRA

Only after those → revisit the structural changes (C1–C4).

---

## 8. What we don't touch

Explicitly unchanged from Round 1:
- Base model choice (Qwen2.5-Math-7B-Instruct)
- Gradient checkpointing (still needed for memory)
- Cosine LR schedule with 100 warmup steps
- Eval / save frequency (every 200 steps)
- Target modules (all 7 projections — just lower rank)
- Drive-based checkpoint / HF cache layout

Not changing these keeps the experiment interpretable.
