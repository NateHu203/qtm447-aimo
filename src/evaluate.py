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
import re
from collections import Counter
from tqdm import tqdm

from model import load_config, load_model_and_tokenizer
from peft import PeftModel


BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_answer(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def evaluate(model, tokenizer, data_path: str, n_samples: int = 1, max_new_tokens: int = 1024):
    from dataset import PROMPT_TEMPLATE

    correct = 0
    total = 0

    with open(data_path) as f:
        examples = [json.loads(line) for line in f]

    for ex in tqdm(examples):
        prompt = PROMPT_TEMPLATE.format(problem=ex["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        candidate_answers = []
        for _ in range(n_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(n_samples > 1),
                temperature=0.7 if n_samples > 1 else 1.0,
            )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            candidate_answers.append(extract_answer(generated))

        prediction = majority_vote(candidate_answers) if n_samples > 1 else candidate_answers[0]
        if prediction is not None and prediction == str(ex["answer"]).strip():
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
    args = parser.parse_args()

    config = load_config(args.config)
    # Override config model name if model_path looks like a HF model ID or local dir
    config["model"]["name"] = args.model_path
    base_model, tokenizer = load_model_and_tokenizer(config, quantize_4bit=True)

    # Only wrap with PEFT if the path has an adapter_config.json (i.e. it's a LoRA adapter)
    adapter_cfg = os.path.join(args.model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = base_model

    model.eval()

    evaluate(model, tokenizer, args.data_path, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
