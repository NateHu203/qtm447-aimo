"""
D1 diagnostic: print raw generations from the Round 1 SFT model.

Compares two prompt formats on the same problems:
  - "plain"  : the Round 1 training template ("Problem: ... Solution:")
  - "chatml" : Qwen's ChatML template with math system prompt

This tells us whether Round 1's regression is a template issue (ChatML recovers
accuracy) or a deeper reasoning regression (both formats fail).

Usage (T4 with 4-bit, free):
    python src/inspect_generations.py \\
        --model_path ./checkpoints/lora-final \\
        --config configs/sft_7b.yaml \\
        --data_path data/processed/val_200.jsonl \\
        --n 10 \\
        --quantize \\
        --output /content/drive/MyDrive/AIMO/results/d1_inspection.json
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import load_config, load_model_and_tokenizer
from peft import PeftModel


BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_answer(text):
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def build_plain_prompt(problem):
    return f"Problem: {problem}\n\nSolution:"


def build_chatml_prompt(tokenizer, problem):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt, max_new_tokens=2048):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["name"] = args.model_path
    base_model, tokenizer = load_model_and_tokenizer(config, quantize_4bit=args.quantize)

    adapter_cfg = os.path.join(args.model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = base_model
    model.eval()

    with open(args.data_path) as f:
        examples = [json.loads(line) for line in f][: args.n]

    results = []
    for i, ex in enumerate(examples):
        print(f"\n{'=' * 80}\nProblem {i + 1}/{args.n}")
        print(f"  {ex['problem'][:150]}{'...' if len(ex['problem']) > 150 else ''}")
        print(f"  Expected: {ex['answer']}")

        plain_out = generate(model, tokenizer, build_plain_prompt(ex["problem"]), args.max_new_tokens)
        chatml_out = generate(model, tokenizer, build_chatml_prompt(tokenizer, ex["problem"]), args.max_new_tokens)

        result = {
            "idx": i,
            "problem": ex["problem"],
            "expected_answer": str(ex["answer"]),
            "plain_output": plain_out,
            "plain_extracted": extract_answer(plain_out),
            "plain_length_chars": len(plain_out),
            "chatml_output": chatml_out,
            "chatml_extracted": extract_answer(chatml_out),
            "chatml_length_chars": len(chatml_out),
        }
        results.append(result)

        print(f"  Plain  : extracted={result['plain_extracted']!r:15s} | len={result['plain_length_chars']} chars")
        print(f"  ChatML : extracted={result['chatml_extracted']!r:15s} | len={result['chatml_length_chars']} chars")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    def summarize(key_ext, label):
        hits = sum(1 for r in results if r[key_ext] is not None)
        correct = sum(1 for r in results if r[key_ext] == r["expected_answer"])
        return f"  {label:8s}: {hits}/{args.n} produced \\boxed{{}}, {correct}/{args.n} correct"

    print(f"\n{'=' * 80}\nSummary ({args.n} problems):")
    print(summarize("plain_extracted", "Plain"))
    print(summarize("chatml_extracted", "ChatML"))
    print(f"\nFull outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
