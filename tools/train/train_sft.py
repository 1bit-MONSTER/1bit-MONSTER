#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# train_sft.py — Supervised fine-tuning via Unsloth (Apache 2.0)
#
# Usage:
#   python3 train_sft.py --config configs/qwen3-0.6b-lora.yaml
#   python3 train_sft.py \
#     --model unsloth/Qwen3-0.6B \
#     --dataset alpaca \
#     --output ./outputs
#
# Dependencies:
#   pip install unsloth  (Apache 2.0 — no unsloth_cli, no studio)
#
# Output:
#   ./outputs/<run-name>/ — checkpoint dir with adapter weights
#   Run export.py to produce GGUF for 1bit's C++ binary.

import argparse
import json
import os
import sys
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Unsloth imports (all Apache 2.0) ─────────────────────────────────────────
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model: str = "unsloth/Qwen3-0.6B"
    hf_token: Optional[str] = None
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # None = auto-detect

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    max_seq_length: int = 8192
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 400
    learning_rate: float = 2e-4
    embedding_learning_rate: float = 1e-5
    warmup_steps: int = 5
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    lr_scheduler: str = "linear"
    seed: int = 3407

    # Dataset
    dataset: str = "yahma/alpaca-cleaned"
    dataset_text_field: str = "text"
    dataset_split: str = "train[:1000]"
    chat_template: Optional[str] = None  # e.g. "qwen-3", "llama-3"

    # Output
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    save_steps: int = 100
    logging_steps: int = 1
    report_to: str = "none"  # "wandb" | "none"

    def resolve(self):
        if self.run_name is None:
            base = Path(self.model).name
            self.run_name = f"{base}-lora-r{self.lora_r}"
        self.output_dir = str(Path(self.output_dir) / self.run_name)
        return self


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


# ── Dataset formatting ────────────────────────────────────────────────────────

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_alpaca(example):
    text = ALPACA_PROMPT.format(
        example.get("instruction", ""),
        example.get("input", ""),
        example.get("output", ""),
    )
    return {"text": text}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unsloth SFT training for 1bit.MONSTER")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--model", type=str, help="Override: model name/path")
    parser.add_argument("--dataset", type=str, help="Override: dataset name/path")
    parser.add_argument("--output", type=str, help="Override: output directory")
    parser.add_argument("--steps", type=int, help="Override: max training steps")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    # Load config
    cfg = TrainConfig()
    if args.config:
        cfg = load_config(args.config)

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.dataset:
        cfg.dataset = args.dataset
    if args.output:
        cfg.output_dir = args.output
    if args.steps:
        cfg.max_steps = args.steps

    cfg = cfg.resolve()

    print(f"[1bit/train] Config: {json.dumps(asdict(cfg), indent=2, default=str)}")
    if args.dry_run:
        return

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[1bit/train] Loading model: {cfg.model}")
    dtype_map = {"float16": "float16", "bfloat16": "bfloat16", "auto": None}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype_map.get(cfg.dtype) if cfg.dtype else None,
        load_in_4bit=cfg.load_in_4bit,
        token=cfg.hf_token,
    )

    # Optional chat template
    if cfg.chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)

    # ── LoRA ────────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.lora_target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    from datasets import load_dataset

    print(f"[1bit/train] Loading dataset: {cfg.dataset}")
    dataset = load_dataset(cfg.dataset, split=cfg.dataset_split)

    if "alpaca" in cfg.dataset.lower() or "yahma" in cfg.dataset.lower():
        dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    elif cfg.chat_template:
        dataset = dataset.map(
            lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
            remove_columns=dataset.column_names,
        )
    elif cfg.dataset_text_field not in dataset.column_names:
        print(f"[1bit/train] WARNING: dataset missing '{cfg.dataset_text_field}' field")
        print(f"  Available: {dataset.column_names}")

    # ── Trainer ─────────────────────────────────────────────────────────────
    use_bf16 = is_bfloat16_supported()
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=cfg.dataset_text_field,
        max_seq_length=cfg.max_seq_length,
        dataset_num_proc=2,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            embedding_learning_rate=cfg.embedding_learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=cfg.logging_steps,
            optim=cfg.optim,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler,
            seed=cfg.seed,
            output_dir=cfg.output_dir,
            save_steps=cfg.save_steps,
            report_to=cfg.report_to,
        ),
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"[1bit/train] Starting training (output: {cfg.output_dir})")
    trainer.train()

    # ── Save adapter ────────────────────────────────────────────────────────
    print(f"[1bit/train] Saving LoRA adapter to: {cfg.output_dir}")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("[1bit/train] Done. Run export.py to produce GGUF.")


if __name__ == "__main__":
    main()
