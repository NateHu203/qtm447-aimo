"""
Evaluation logic: exact integer match and majority voting.

Usage:
    python src/evaluate.py --model_path ./checkpoints/lora-final \
                           --config configs/sft_7b.yaml \
                           --data_path data/processed/val.jsonl \
                           --n_samples 8
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import load_config, load_model_and_tokenizer
from peft import PeftModel


BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_answer(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def answers_equal(pred: str | None, gold) -> bool:
    """Compare prediction and gold answer, tolerant to numeric equivalence ('5.0' == '5')."""
    if pred is None:
        return False
    pred_s = pred.strip()
    gold_s = str(gold).strip()
    if pred_s == gold_s:
        return True
    try:
        return float(pred_s) == float(gold_s)
    except (ValueError, TypeError):
        return False


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def build_prompt(tokenizer, problem: str, use_chatml: bool = True) -> str:
    if use_chatml:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    from dataset import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(problem=problem)


def evaluate(model, tokenizer, data_path: str, n_samples: int = 1, max_new_tokens: int = 512, batch_size: int = 8, use_chatml: bool = True):
    correct = 0
    total = 0

    with open(data_path) as f:
        examples = [json.loads(line) for line in f]

    tokenizer.padding_side = "left"  # required for batched generation

    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[batch_start : batch_start + batch_size]
        prompts = [build_prompt(tokenizer, ex["problem"], use_chatml) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        candidate_batches = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(n_samples > 1),
                    temperature=0.7 if n_samples > 1 else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            generated = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
            candidate_batches.append(generated)

        for i, ex in enumerate(batch):
            candidates = [extract_answer(candidate_batches[s][i]) for s in range(n_samples)]
            prediction = majority_vote(candidates) if n_samples > 1 else candidates[0]
            if answers_equal(prediction, ex["answer"]):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--quantize", action="store_true", help="Load in 4-bit (use on T4/low VRAM)")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    parser.add_argument("--plain_prompt", action="store_true", help="Use Round 1's plain 'Problem: ... Solution:' template instead of ChatML")
    args = parser.parse_args()

    config = load_config(args.config)
    # Override config model name if model_path looks like a HF model ID or local dir
    config["model"]["name"] = args.model_path
    base_model, tokenizer = load_model_and_tokenizer(config, quantize_4bit=args.quantize)

    # Only wrap with PEFT if the path has an adapter_config.json (i.e. it's a LoRA adapter)
    adapter_cfg = os.path.join(args.model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = base_model

    model.eval()

    accuracy = evaluate(
        model, tokenizer, args.data_path,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        use_chatml=not args.plain_prompt,
    )

    if args.output:
        import datetime
        result = {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "prompt_format": "plain" if args.plain_prompt else "chatml",
            "accuracy": accuracy,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
