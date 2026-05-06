# Round 2 — Conservative-Hyperparameter Retrain

| | |
|---|---|
| **Status** | Complete on val_200 and AIME 2024; AIME 2025 partial (2/30) |
| **Date(s)** | Plan drafted: 2026-04-22. Plan suspended: 2026-04-23 (after Round 1 diagnostic). Hyperparameter-only retrain executed: 2026-04-23 to 2026-04-24. Evaluation: 2026-05-03 to 2026-05-05. |
| **Run name** | `qwen2.5-math-7b-lora-r16_v2` |
| **Adapter location** | `MyDrive/AIMO/checkpoints_v2/lora-final` |
| **Config** | [`configs/sft_7b_v2.yaml`](../configs/sft_7b_v2.yaml) |
| **Headline** | Conservative hyperparameters alone **did not** prevent catastrophic forgetting — they made it worse. Round 2 is below baseline on every benchmark we have full data for. The negative result rules out the "hyperparameter-aggressiveness" hypothesis from Round 1 and points toward training-format mismatch as the dominant cause. |

---

## 1. Origin of this round

Round 1 finished with two findings (see [round_1.md](round_1.md)):

1. **+10 points on `val_200` under corrected evaluation** — the model genuinely improved in-distribution.
2. **−6.7 points on AIME 2024**, exhibiting catastrophic forgetting consistent with aggressive fine-tuning overwriting the base model's RLHF-tuned capability.

A full Round 2 plan was originally drafted to address all three plausible mechanisms behind (2) — chat template, loss masking, and aggressive hyperparameters. After the diagnostic showed the in-distribution gain was real, that full plan was reframed as overkill: changing five things at once would not isolate which fix mattered.

This Round 2 narrows the change to **hyperparameters only** so that any change in OOD behavior is attributable cleanly. Chat-template training and completion-only loss masking are tracked as Round 3 priorities if Round 2 does not close the OOD gap.

---

## 2. What changed (Round 1 → Round 2)

| Hyperparameter | Round 1 | Round 2 | Rationale |
|---|---|---|---|
| LoRA rank `r` | 64 | **16** | Smaller adapter capacity; less room to overfit `val_200`'s distribution |
| LoRA `alpha` | 128 | **32** | Maintain the `alpha = 2r` convention |
| Learning rate | 2e-4 | **1e-5** | Conservative LR preserves more of Qwen's RLHF alignment |
| Precision | fp16 | **bf16** | Wider exponent range, fewer NaN/inf risks during LoRA training |
| Output dir | `MyDrive/AIMO/checkpoints` | `MyDrive/AIMO/checkpoints_v2` | Keep Round 1 artifact reachable |

**Unchanged from Round 1 (deliberate):**

- Same base model (`Qwen/Qwen2.5-Math-7B-Instruct`)
- Same training data (72,283 examples, same filter, same split)
- **Same training prompt format — plain `"Problem: ... Solution:"` (NOT ChatML)**
- **Same loss — full sequence (NOT completion-only)**
- Same effective batch size (32 = 4 × 8)
- Same gradient checkpointing, same cosine schedule, 100 warmup steps, 1 epoch

This isolation is intentional. If Round 2 closes the AIME 2024 gap, the fix is hyperparameter conservatism. If it does not, the residual cause must lie in the training format or loss configuration — and Round 3 has clear priorities.

---

## 3. Why we did **not** include the other two fixes

The original Round 2 plan, drafted before Round 1's eval was corrected, included:

- **ChatML formatting in training data**, via `tokenizer.apply_chat_template`. Would require modifying [`src/dataset.py`](../src/dataset.py) and [`src/train.py`](../src/train.py).
- **Completion-only loss masking**, via `DataCollatorForCompletionOnlyLM`. Would require wiring the existing masking logic into the SFT training path.

These changes are well-motivated and recommended by Dr. McAlister. They are not in this round only because:

1. Each is a structural change to the training code, not a config edit.
2. Bundling them with hyperparameter changes would prevent attributing any observed effect to a specific cause.
3. If hyperparameter conservatism alone closes the AIME 2024 gap, the other two changes become lower-priority follow-ups rather than mandatory fixes.

Round 2's result rules out (3): the gap got worse, not better. Round 3 should implement the structural fixes.

---

## 4. Results

| Benchmark | n | Baseline | Round 1 SFT | **Round 2 SFT** | Δ vs Baseline | Δ vs Round 1 |
|---|---|---|---|---|---|---|
| `val_200` (in-distribution) | 200 | 57.0% | 67.0% | **55.0%** | −2.0 | −12.0 |
| AIME 2024 | 30 | 23.3% | 16.7% | **3.3%** | −20.0 | −13.4 |
| AIME 2025 | _2/30 partial_ | 6.7% | 6.7% | _0/2_ | _insufficient_ | |

Wilson 95% confidence intervals:

- Round 2 `val_200`: 110/200 = 55.0% [48.0, 61.7] — overlaps with baseline's [53.6, 60.4]; clearly below Round 1's [63.6, 70.4]
- Round 2 AIME_2024: 1/30 = 3.3% [0.6, 16.7] — barely doesn't overlap with baseline's [11.8, 40.9]; clearly below Round 1's [7.3, 33.6]

### Failure-mode breakdown

| Category (val_200, n=200) | Count | % |
|---|---|---|
| `correct` | 110 | 55.0% |
| `wrong_numeric` | 61 | 30.5% |
| `wrong_nonnumeric` | 24 | 12.0% |
| `truncated_no_box` | 2 | 1.0% |
| `no_box` | 3 | 1.5% |

| Category (AIME 2024, n=30) | Count | % |
|---|---|---|
| `correct` | 1 | 3.3% |
| `wrong_numeric` | 27 | 90.0% |
| `wrong_nonnumeric` | 1 | 3.3% |
| `truncated_no_box` | 1 | 3.3% |

The `wrong_nonnumeric` rate is notably higher on Round 2 than Round 1 (12% vs much lower on val_200; ~3% on AIME). This category includes outputs whose `\boxed{...}` content failed numeric parsing — for example, the model emitting `\boxed{xx}` placeholders or raw expressions. This suggests Round 2 has partially lost the formatting discipline that Round 1 retained.

---

## 5. What we learned

**(a) Hyperparameter conservatism alone is insufficient.** The most direct way to read the result: smaller LoRA, lower LR, and bf16 — applied in isolation — did not preserve OOD capability. They reduced it further. The "aggressive hyperparameters caused forgetting" hypothesis from Round 1 is *not* the dominant explanation.

**(b) The likely dominant cause is the training/eval format mismatch.** Both rounds train on plain `"Problem: ... Solution:"` text and evaluate on Qwen's ChatML template. Round 1 (r=64, lr=2e-4) had enough adapter capacity and gradient signal to absorb this mismatch — it learned olympiad-specific reasoning despite training in the wrong format, and got +10 points in-distribution. Round 2 (r=16, lr=1e-5) had less capacity and a slower learning signal; it could not overcome the format mismatch, so it ended up perturbing the base model's alignment without compensating in-distribution gain. Worst of both worlds.

**(c) The capacity/signal/mismatch trade-off is a real phenomenon.** Concretely: a larger adapter trained at a higher LR can succeed *despite* a format mismatch by force-fitting; a smaller adapter at a lower LR cannot, even though the smaller adapter is "safer" in principle. Conservatism is not free — it requires the rest of the pipeline (especially the format) to actually be correct.

**(d) `wrong_nonnumeric` rate is a useful signal of format degradation.** Round 2's elevated rate of malformed `\boxed{...}` outputs suggests the model has partially regressed on its base ChatML/`\boxed{}` convention. This is consistent with (b): less successful in-distribution learning, more disruption of base behavior.

---

## 6. Implications for any future round

If a Round 3 is run, the priorities are clear:

1. **ChatML training format** via `tokenizer.apply_chat_template` in [`src/dataset.py`](../src/dataset.py) and [`src/train.py`](../src/train.py). Match training to inference distribution.
2. **Completion-only loss masking** via `DataCollatorForCompletionOnlyLM` (or by wiring in the existing prompt-mask logic in [`src/dataset.py`](../src/dataset.py)). Stop spending gradient budget on predicting problem statements.
3. **Reuse Round 2's hyperparameters** (r=16, lr=1e-5, bf16) — they are appropriate *given* the format is correct. Round 2's failure was the missing structural prerequisite, not the hyperparameters themselves.

Predicted outcome of Round 3: matches or exceeds Round 1's `val_200` while reducing AIME 2024 forgetting. This is testable but not executed here.

---

## 7. AIME 2025 — what's left to finish

The AIME analysis run got 30 of 30 AIME_2024 problems but only 2 of 30 AIME_2025 problems before stopping. To finish:

```bash
python src/analyze_errors.py \
    --model_path /content/drive/MyDrive/AIMO/checkpoints_v2/lora-final \
    --config configs/sft_7b_v2.yaml \
    --data_path data/processed/val_aime.jsonl \
    --quantize \
    --output /content/drive/MyDrive/AIMO/results/sft_v2_aime_errors.json
```

The resume support in [`src/analyze_errors.py`](../src/analyze_errors.py) skips problems already in the `.jsonl` (32 done), so this run only needs to handle the remaining 28 AIME_2025 problems. The qualitative read on the partial 2/30 is that Round 2 is at or below the AIME 2025 floor (where neither baseline nor Round 1 had measurable capability), so even completing the eval is unlikely to change the headline interpretation. The completed number is needed for the writeup table, not for the conclusion.

---

## 8. Reproducibility

- **Training:** `python src/train.py --config configs/sft_7b_v2.yaml`
- **Evaluation (val_200):** see §7 (replace `val_aime.jsonl` with `val_200.jsonl`)
- **Evaluation (AIME):** see §7
- **Adapter contents (verified):** `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `README.md`, `tokenizer_config.json`, `tokenizer.json`, `training_args.bin`
- **Per-problem outputs:** [`results/sft_v2_val200_errors.json`](../results/sft_v2_val200_errors.json), [`results/sft_v2_aime_errors.json.jsonl`](../results/sft_v2_aime_errors.json.jsonl)
