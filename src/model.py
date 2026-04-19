"""
Model loading and LoRA configuration.
"""

import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
import yaml

load_dotenv()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict, quantize_4bit: bool = False):
    model_name = config["model"]["name"]
    torch_dtype = getattr(torch, config["model"].get("torch_dtype", "float16"))

    bnb_config = None
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_lora(model, config: dict):
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
