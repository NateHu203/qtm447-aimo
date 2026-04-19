"""
SFT training entry point.

Usage:
    python src/train.py --config configs/sft_7b.yaml
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, os.path.dirname(__file__))
from model import load_config, load_model_and_tokenizer, apply_lora
from dataset import PROMPT_TEMPLATE

load_dotenv()

def load_jsonl_as_hf_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            # Combine prompt + solution into a single "text" field
            text = PROMPT_TEMPLATE.format(problem=ex["problem"]) + " " + ex["solution"]
            records.append({"text": text, "answer": ex["answer"]})
    return Dataset.from_list(records)


def main(config_path: str):
    config = load_config(config_path)
    train_cfg = config["training"]

    output_dir = os.environ.get("CHECKPOINT_DIR", train_cfg["output_dir"])

    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)

    train_dataset = load_jsonl_as_hf_dataset(config["data"]["train_path"])
    val_dataset = load_jsonl_as_hf_dataset(config["data"]["val_path"])

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_steps=100,
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        eval_strategy="steps",
        eval_steps=train_cfg["save_steps"],
        report_to=train_cfg["report_to"],
        run_name=train_cfg["run_name"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        load_best_model_at_end=True,
        packing=False,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_len"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train(
        resume_from_checkpoint=output_dir if os.path.exists(output_dir) else None
    )

    model.save_pretrained(os.path.join(output_dir, "lora-final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora-final"))
    print(f"Saved adapter to {output_dir}/lora-final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
