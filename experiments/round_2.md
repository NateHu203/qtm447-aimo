# Round 2 — Conservative-Hyperparameter Retrain

| | |
|---|---|
| **Status** | Complete |
| **Date(s)** | Plan drafted: 2026-04-22. Plan suspended: 2026-04-23 (after Round 1 diagnostic). Hyperparameter-only retrain executed: 2026-04-23 to 2026-04-24. Evaluation: 2026-05-03 to 2026-05-05. |
| **Run name** | `qwen2.5-math-7b-lora-r16_v2` |
| **Adapter location** | `MyDrive/AIMO/checkpoints_v2/lora-final` |
| **Config** | [`configs/sft_7b_v2.yaml`](../configs/sft_7b_v2.yaml) |
| **Headline** | Conservative hyperparameters alone produced an **asymmetric** OOD result, not uniform degradation. AIME 2024 collapsed (16.7% → 3.3%) while AIME 2025 improved directionally (6.7% → 10.0%, within noise at n=30). val_200 dropped to 55.0% — below baseline. The result rules out simple "hyperparameter conservatism" as a sufficient fix and points to training-format mismatch as the actual cause. |

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
| AIME 2024 (partial OOD) | 30 | 23.3% | 16.7% | **3.3%** | −20.0 | −13.4 |
| AIME 2025 (clean OOD) | 30 | 6.7% | 6.7% | **10.0%** | +3.3 (n.s.) | +3.3 (n.s.) |
| AIME combined | 60 | 15.0% | 11.7% | **6.7%** | −8.3 | −5.0 |

Wilson 95% confidence intervals:

- Round 2 `val_200`: 110/200 = 55.0% [48.0, 61.7] — overlaps with baseline's [53.6, 60.4]; clearly below Round 1's [63.6, 70.4]
- Round 2 AIME_2024: 1/30 = 3.3% [0.6, 16.7] — barely doesn't overlap with baseline's [11.8, 40.9]; clearly below Round 1's [7.3, 33.6]
- Round 2 AIME_2025: 3/30 = 10.0% [3.5, 25.6] — heavily overlaps with both baseline and Round 1 [1.6, 21.6]; the directional improvement is not statistically significant at this sample size

### Failure-mode breakdown

| Category (val_200, n=200) | Count | % |
|---|---|---|
| `correct` | 110 | 55.0% |
| `wrong_numeric` | 61 | 30.5% |
| `wrong_nonnumeric` | 24 | 12.0% |
| `no_box` | 3 | 1.5% |
| `truncated_no_box` | 2 | 1.0% |

| Category (AIME 2024, n=30) | Count | % |
|---|---|---|
| `correct` | 1 | 3.3% |
| `wrong_numeric` | 27 | 90.0% |
| `wrong_nonnumeric` | 1 | 3.3% |
| `truncated_no_box` | 1 | 3.3% |

| Category (AIME 2025, n=30) | Count | % |
|---|---|---|
| `correct` | 3 | 10.0% |
| `wrong_numeric` | 22 | 73.3% |
| `truncated_no_box` | 4 | 13.3% |
| `no_box` | 1 | 3.3% |

The `wrong_nonnumeric` rate is notably higher on Round 2 than Round 1 (12% on val_200; ~3% on AIME 2024; absent on AIME 2025). This category includes outputs whose `\boxed{...}` content failed numeric parsing — for example, the model emitting `\boxed{xx}` placeholders or raw expressions. This suggests Round 2 has partially lost the formatting discipline that Round 1 retained on in-distribution problems.

The `truncated_no_box` rate is higher on AIME 2025 (13.3%) than AIME 2024 (3.3%), suggesting Round 2 generates longer chains on the genuinely-novel benchmark — possibly a reasonable signal of effort, possibly just verbosity.

---

## 5. What we learned

**(a) Hyperparameter conservatism alone is insufficient.** Smaller LoRA, lower LR, and bf16 — applied in isolation — did not preserve OOD capability uniformly. The simple "aggressive hyperparameters caused forgetting" hypothesis from Round 1 is *not* a complete explanation.

**(b) The result is asymmetric across OOD benchmarks, and the asymmetry is informative.**

- **AIME 2024 collapsed** (16.7% → 3.3%). This is the benchmark where the base model already had measurable competence (23.3%) — likely from pretraining exposure. Round 2 disrupted that pre-existing capability more severely than Round 1 did, despite using "safer" hyperparameters.
- **AIME 2025 directionally improved** (6.7% → 10.0%, within noise at n=30). This is the genuinely-novel benchmark where neither model had meaningful capability. Here, Round 2's smaller adapter was less able to specialize on the wrong (NuminaMath plain-text) distribution and so apparently retained or marginally improved generalization.
- **`val_200` dropped** (67% → 55%). Smaller capacity + lower LR meant Round 2 simply learned less of NuminaMath's solution style — the very thing Round 1 successfully fit.

The combined picture: Round 2 is **less specialized** than Round 1. It overfits less but also learns less. Where the base model had pretraining-derived capability (AIME 2024, val_200), Round 2 disrupts more than it adds. Where the base had no capability (AIME 2025), the lower specialization may have been mildly beneficial — though the sample size precludes a confident claim.

**(c) The likely dominant cause is the training/eval format mismatch.** Both rounds train on plain `"Problem: ... Solution:"` text and evaluate on Qwen's ChatML template. Round 1 (r=64, lr=2e-4) had enough adapter capacity to absorb this mismatch and force-fit useful in-distribution behavior. Round 2 (r=16, lr=1e-5) lacked the capacity to overcome the wrong format — it perturbed the base model's alignment without compensating gain on familiar problems. Both rounds therefore agree that the *format* needs to be correct; they differ only in how that mismatch manifests when adapter capacity changes.

**(d) The capacity/signal/mismatch trade-off is real.** A larger adapter at a higher LR can succeed *despite* format mismatch by force-fitting. A smaller adapter at a lower LR cannot. Conservatism is not free — it requires the rest of the pipeline (especially the format) to be correct first.

**(e) `wrong_nonnumeric` and `truncated_no_box` signal qualitative shifts.** Round 2's elevated `wrong_nonnumeric` rate on `val_200` (12% vs Round 1's much lower) suggests partial regression on the base `\boxed{}` convention. The elevated `truncated_no_box` rate on AIME 2025 (13%) suggests Round 2 produces longer chains on novel problems — consistent with the picture of less-confident, less-specialized output.

---

## 6. Implications for any future round

If a Round 3 is run, the priorities are clear:

1. **ChatML training format** via `tokenizer.apply_chat_template` in [`src/dataset.py`](../src/dataset.py) and [`src/train.py`](../src/train.py). Match training to inference distribution.
2. **Completion-only loss masking** via `DataCollatorForCompletionOnlyLM` (or by wiring in the existing prompt-mask logic in [`src/dataset.py`](../src/dataset.py)). Stop spending gradient budget on predicting problem statements.
3. **Reuse Round 2's hyperparameters** (r=16, lr=1e-5, bf16) — they are appropriate *given* the format is correct. Round 2's failure was the missing structural prerequisite, not the hyperparameters themselves.

Predicted outcome of Round 3: matches or exceeds Round 1's `val_200` while reducing AIME 2024 forgetting. This is testable but not executed here.

---

## 7. Reproducibility

- **Training:** `python src/train.py --config configs/sft_7b_v2.yaml`
- **Evaluation (val_200):** see §7 (replace `val_aime.jsonl` with `val_200.jsonl`)
- **Evaluation (AIME):** see §7
- **Adapter contents (verified):** `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `README.md`, `tokenizer_config.json`, `tokenizer.json`, `training_args.bin`
- **Per-problem outputs:** [`results/sft_v2_val200_errors.json`](../results/sft_v2_val200_errors.json), [`results/sft_v2_aime_errors.json.jsonl`](../results/sft_v2_aime_errors.json.jsonl)
