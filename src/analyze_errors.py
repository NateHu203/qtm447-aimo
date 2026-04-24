"""
Error analysis: run full eval, save per-problem outputs, categorize failures.

Usage:
    python src/analyze_errors.py \\
        --model_path ./checkpoints/lora-final \\
        --config configs/sft_7b.yaml \\
        --data_path data/processed/val_200.jsonl \\
        --quantize \\
        --output /content/drive/MyDrive/AIMO/results/sft_round1_errors.json

Output: JSON with per-problem records including raw model output and failure category.
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import load_config, load_model_and_tokenizer
from evaluate import SYSTEM_PROMPT, extract_answer, answers_equal
from peft import PeftModel


def categorize_failure(raw_output, extracted, expected, max_new_tokens):
    """Bucket a wrong answer into one of a few failure modes."""
    if extracted is None:
        # No \boxed{} at all. Was it truncated mid-solution?
        # Heuristic: if output is >= ~75% of max_new_tokens worth of chars, likely truncated
        if len(raw_output) > max_new_tokens * 3:  # rough chars/token estimate
            return "truncated_no_box"
        return "no_box"
    # Has \boxed{}, but wrong
    try:
        float(extracted)
        float(str(expected))
        return "wrong_numeric"  # both are numeric but don't match
    except (ValueError, TypeError):
        return "wrong_nonnumeric"  # at least one non-numeric — likely extraction or format issue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_cfg = os.path.join(args.model_path, "adapter_config.json")
    is_adapter = os.path.exists(adapter_cfg)

    if not is_adapter:
        config["model"]["name"] = args.model_path

    base_model, tokenizer = load_model_and_tokenizer(config, quantize_4bit=args.quantize)
    model = PeftModel.from_pretrained(base_model, args.model_path) if is_adapter else base_model
    model.eval()
    tokenizer.padding_side = "left"

    with open(args.data_path) as f:
        examples = [json.loads(line) for line in f]

    # Resume support: skip problems already in the output file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    jsonl_path = args.output + ".jsonl"  # incremental per-problem log
    done_problems = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done_problems.add(json.loads(line)["problem"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_problems)} problems already completed, skipping those.")

    records = []
    # Load anything we've already written so the final summary is complete
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    jsonl_file = open(jsonl_path, "a")

    remaining = [ex for ex in examples if ex["problem"] not in done_problems]
    for batch_start in tqdm(range(0, len(remaining), args.batch_size), desc="Analyzing"):
        batch = remaining[batch_start : batch_start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["problem"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for ex, raw in zip(batch, decoded):
            extracted = extract_answer(raw)
            correct = answers_equal(extracted, ex["answer"])
            category = "correct" if correct else categorize_failure(
                raw, extracted, ex["answer"], args.max_new_tokens
            )
            record = {
                "problem": ex["problem"],
                "expected_answer": str(ex["answer"]),
                "extracted": extracted,
                "correct": correct,
                "category": category,
                "output_length_chars": len(raw),
                "raw_output": raw,
            }
            records.append(record)
            # Persist immediately so we never lose work on a crash
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()

    jsonl_file.close()

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    cats = Counter(r["category"] for r in records)
    n = len(records)
    correct = cats["correct"]
    print(f"\n{'=' * 60}")
    print(f"Total: {n}  |  Correct: {correct}/{n} = {correct / n:.1%}")
    print(f"\nFailure breakdown ({n - correct} wrong):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        if cat == "correct":
            continue
        print(f"  {cat:25s} {count:3d} ({count / n:.1%})")
    print(f"\nFull records: {args.output}")


if __name__ == "__main__":
    main()
