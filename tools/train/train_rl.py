#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# train_rl.py — Reinforcement learning (GRPO) via Unsloth (Apache 2.0)
#
# Usage:
#   python3 train_rl.py --config configs/qwen3-0.6b-grpo.yaml
#   python3 train_rl.py \
#     --model unsloth/Qwen3-0.6B \
#     --dataset gsm8k \
#     --output ./outputs
#
# Dependencies:
#   pip install unsloth  (Apache 2.0 — no unsloth_cli, no studio)
#
# Output:
#   ./outputs/<run-name>/ — checkpoint dir with RL-trained adapter
#   Run export.py to produce GGUF for 1bit's Rust binary.

import argparse
import json
import os
import sys
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

import torch

# ── Unsloth imports (all Apache 2.0) ─────────────────────────────────────────
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.models.rl import GRPOTrainer, GRPOConfig


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RLConfig:
    # Model
    model: str = "unsloth/Qwen3-0.6B"
    hf_token: Optional[str] = None
    load_in_4bit: bool = True

    # LoRA (GRPO can train with or without LoRA)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO
    max_seq_length: int = 8192
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    learning_rate: float = 5e-6
    warmup_steps: int = 20
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"

    # RL specifics
    num_generations: int = 4       # responses per prompt
    max_prompt_length: int = 512
    temperature: float = 0.7
    seed: int = 3407

    # Dataset
    dataset: str = "gsm8k"
    dataset_split: str = "train"

    # Output
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    save_steps: int = 200
    logging_steps: int = 1
    report_to: str = "none"  # "wandb" | "none"

    def resolve(self):
        if self.run_name is None:
            base = Path(self.model).name
            self.run_name = f"{base}-grpo-r{self.lora_r}"
        self.output_dir = str(Path(self.output_dir) / self.run_name)
        return self


def load_config(path: str) -> RLConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RLConfig(**raw)


# ── Reward functions ──────────────────────────────────────────────────────────

def reward_correctness(completions, answer, **kwargs) -> list[float]:
    """Binary reward: 1.0 if model answer matches ground truth number."""
    rewards = []
    for completion, ans in zip(completions, answer):
        # Extract last number from completion
        import re
        nums = re.findall(r"-?\d+\.?\d*", completion)
        correct = re.findall(r"-?\d+\.?\d*", str(ans))
        if nums and correct:
            rewards.append(1.0 if nums[-1] == correct[-1] else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_format(completions, **kwargs) -> list[float]:
    """Format reward: penalize if missing expected reasoning tags."""
    rewards = []
    for c in completions:
        has_reasoning = "\\n\\n" in c and len(c) > 50
        rewards.append(0.5 if has_reasoning else 0.0)
    return rewards


def reward_length(completions, **kwargs) -> list[float]:
    """Length penalty: prefer concise answers."""
    return [max(0.0, 1.0 - len(c) / 1024) for c in completions]


# ── Dataset ───────────────────────────────────────────────────────────────────

def prepare_dataset(dataset_name: str, split: str):
    """Load a standard reasoning dataset and format for GRPO."""
    from datasets import load_dataset

    if "gsm8k" in dataset_name.lower():
        ds = load_dataset("gsm8k", "main", split=split)
        return ds.map(lambda x: {
            "prompt": x["question"],
            "answer": x["answer"],
        })
    elif "math" in dataset_name.lower():
        ds = load_dataset("lighteval/MATH", split=split)
        return ds.map(lambda x: {
            "prompt": x["problem"],
            "answer": x["solution"],
        })
    else:
        ds = load_dataset(dataset_name, split=split)
        return ds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unsloth GRPO training for 1bit.systems")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--model", type=str, help="Override: model name/path")
    parser.add_argument("--dataset", type=str, help="Override: dataset name/path")
    parser.add_argument("--output", type=str, help="Override: output directory")
    parser.add_argument("--steps", type=int, help="Override: max training steps")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    # Load config
    cfg = RLConfig()
    if args.config:
        cfg = load_config(args.config)

    if args.model:
        cfg.model = args.model
    if args.dataset:
        cfg.dataset = args.dataset
    if args.output:
        cfg.output_dir = args.output
    if args.steps:
        cfg.max_steps = args.steps

    cfg = cfg.resolve()

    print(f"[1bit/rl] Config: {json.dumps(asdict(cfg), indent=2, default=str)}")
    if args.dry_run:
        return

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[1bit/rl] Loading model: {cfg.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
        token=cfg.hf_token,
    )

    if cfg.use_lora:
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
    dataset = prepare_dataset(cfg.dataset, cfg.dataset_split)

    # ── Trainer ─────────────────────────────────────────────────────────────
    use_bf16 = is_bfloat16_supported()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_correctness,
            reward_format,
            reward_length,
        ],
        args=GRPOConfig(
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            optim=cfg.optim,
            bf16=use_bf16,
            fp16=not use_bf16,
            num_generations=cfg.num_generations,
            max_prompt_length=cfg.max_prompt_length,
            temperature=cfg.temperature,
            seed=cfg.seed,
            output_dir=cfg.output_dir,
            save_steps=cfg.save_steps,
            logging_steps=cfg.logging_steps,
            report_to=cfg.report_to,
        ),
        train_dataset=dataset,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"[1bit/rl] Starting GRPO training (output: {cfg.output_dir})")
    trainer.train()

    # ── Save ────────────────────────────────────────────────────────────────
    print(f"[1bit/rl] Saving to: {cfg.output_dir}")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("[1bit/rl] Done. Run export.py to produce GGUF.")


if __name__ == "__main__":
    main()
