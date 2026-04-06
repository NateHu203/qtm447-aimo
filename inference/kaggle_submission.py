"""
Kaggle submission inference script.

This file must be self-contained — Kaggle runs offline.
All model weights must be added as a Kaggle Dataset and loaded from disk.

Expected Kaggle dataset structure:
  /kaggle/input/aimo-model/
      ├── base_model/          (merged model weights OR)
      └── lora-adapter/        (LoRA adapter weights, requires base model too)
"""

import os
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Paths inside the Kaggle environment
MODEL_PATH = "/kaggle/input/aimo-model/lora-merged"
TEST_CSV = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"
OUTPUT_CSV = "/kaggle/working/submission.csv"

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
PROMPT_TEMPLATE = "Problem: {problem}\n\nSolution:"


def extract_answer(text: str) -> int:
    matches = BOXED_RE.findall(text)
    if matches:
        try:
            return int(matches[-1].strip())
        except ValueError:
            pass
    # Fallback: find last integer in text
    numbers = re.findall(r"\b\d+\b", text)
    return int(numbers[-1]) if numbers else 0


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, problem: str, n_samples: int = 8) -> int:
    from collections import Counter

    prompt = PROMPT_TEMPLATE.format(problem=problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    answers = []

    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.append(extract_answer(text))

    # Majority vote
    return Counter(answers).most_common(1)[0][0]


def main():
    model, tokenizer = load_model()
    test_df = pd.read_csv(TEST_CSV)

    predictions = []
    for _, row in test_df.iterrows():
        pred = predict(model, tokenizer, row["problem"])
        predictions.append({"id": row["id"], "answer": pred})
        print(f"  id={row['id']}  answer={pred}")

    pd.DataFrame(predictions).to_csv(OUTPUT_CSV, index=False)
    print(f"Submission saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
