"""
PyTorch Dataset and collator for SFT training.

Each JSONL record is expected to have keys: problem, solution, answer, source.
The model is trained to predict `solution` given `problem`.
"""

import json
from torch.utils.data import Dataset


PROMPT_TEMPLATE = "Problem: {problem}\n\nSolution:"


class MathDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        with open(path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = PROMPT_TEMPLATE.format(problem=item["problem"])
        full_text = prompt + " " + item["solution"]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Mask prompt tokens in labels so loss is only on the solution
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = encoding["input_ids"].copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        encoding["labels"] = labels
        return encoding
