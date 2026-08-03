#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# export.py — Export trained model to GGUF for 1bit's C++ binary
#
# Usage:
#   python3 export.py ./outputs/qwen3-0.6b-lora-r16/checkpoint-400
#   python3 export.py ./outputs/ --format gguf --quantization q4_k_m
#   python3 export.py ./outputs/ --format merged-16bit
#
# Dependencies:
#   pip install unsloth  (Apache 2.0 — no unsloth_cli, no studio)
#
# Output:
#   ./export/<model-name>/ — GGUF file(s) loadable by 1bit's gguf_loader

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── Unsloth imports (all Apache 2.0) ─────────────────────────────────────────
from unsloth import FastLanguageModel, save_to_gguf, unsloth_save_model


EXPORT_FORMATS = ["gguf", "merged-16bit", "merged-4bit", "lora"]
GGUF_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in an outputs directory."""
    path = Path(output_dir)
    if not path.exists():
        return None

    # Direct checkpoint path
    if (path / "adapter_config.json").exists():
        return str(path)

    # Search for checkpoint dirs
    checkpoints = sorted(path.glob("checkpoint-*"))
    if checkpoints:
        return str(checkpoints[-1])

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to GGUF for 1bit.systems C++ binary"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint dir or outputs dir",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="gguf",
        choices=EXPORT_FORMATS,
        help="Export format",
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="q4_k_m",
        choices=GGUF_QUANTS,
        help="GGUF quantization method",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./export",
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name in output path",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Max sequence length for reload",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load checkpoint in 4-bit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported and exit",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = find_latest_checkpoint(args.checkpoint)
    if ckpt_path is None:
        print(f"[1bit/export] ERROR: No checkpoint found at: {args.checkpoint}")
        sys.exit(1)

    print(f"[1bit/export] Checkpoint: {ckpt_path}")

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = Path(ckpt_path).parent.name
    export_dir = Path(args.output_dir) / model_name
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1bit/export] Format: {args.format}")
    print(f"[1bit/export] Output: {export_dir}")

    if args.dry_run:
        print(f"[1bit/export] Dry run — would export {model_name} to {export_dir}")
        return

    # ── Load model with adapter ────────────────────────────────────────────
    print(f"[1bit/export] Loading model + adapter from: {ckpt_path}")

    # Load base model first, then PEFT adapter
    # We need the adapter_config to know the base model
    adapter_config_path = Path(ckpt_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path", "")
        print(f"[1bit/export] Base model: {base_model}")
    else:
        base_model = ""
        print(f"[1bit/export] WARNING: No adapter_config.json found, using checkpoint as base")

    # Determine dtype based on format
    dtype = None  # auto-detect
    if args.format == "merged-4bit":
        load_in_4bit = True
    else:
        load_in_4bit = args.load_in_4bit

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model or ckpt_path,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=None,
    )

    # Load LoRA adapter if checkpoint has one
    if adapter_config_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt_path)

    # ── Export ──────────────────────────────────────────────────────────────
    if args.format == "gguf":
        print(f"[1bit/export] Exporting to GGUF ({args.quantization})...")
        save_to_gguf(
            model_name,
            model,
            tokenizer,
            quantization_method=args.quantization,
            output_dir=str(export_dir),
        )
        # Find the produced GGUF file
        gguf_files = list(export_dir.glob("*.gguf"))
        if gguf_files:
            print(f"[1bit/export] ✅ GGUF: {gguf_files[0]}")
            print(f"[1bit/export] Load with: ./build/zaya_server --model {gguf_files[0]}")

    elif args.format in ("merged-16bit", "merged-4bit"):
        save_method = "merged_16bit" if args.format == "merged-16bit" else "merged_4bit"
        print(f"[1bit/export] Exporting {save_method}...")
        unsloth_save_model(
            model, tokenizer,
            save_method=save_method,
            output_dir=str(export_dir),
            save_tokenizer=True,
        )
        print(f"[1bit/export] ✅ Merged model: {export_dir}")

    elif args.format == "lora":
        print(f"[1bit/export] Copying LoRA adapter...")
        import shutil
        shutil.copytree(ckpt_path, str(export_dir / "adapter"), dirs_exist_ok=True)
        print(f"[1bit/export] ✅ LoRA adapter: {export_dir / 'adapter'}")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("─" * 50)
    print(f"  Model:     {model_name}")
    print(f"  Format:    {args.format}")
    print(f"  Output:    {export_dir}")
    if args.format == "gguf" and gguf_files:
        size_mb = gguf_files[0].stat().st_size / 1_000_000
        print(f"  Size:      {size_mb:.0f} MB")
    print("─" * 50)
    print("[1bit/export] Done.")


if __name__ == "__main__":
    main()
