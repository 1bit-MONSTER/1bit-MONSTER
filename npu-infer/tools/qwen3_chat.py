#!/usr/bin/env python3
"""qwen3_chat.py — drive the npu-infer engine with a real prompt.

Encodes the prompt with the Qwen3 tokenizer + ChatML template, runs the
engine binary (runtime_layers path), and decodes the generated tokens.

Usage:
  python3 qwen3_chat.py "The capital of France is"
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from tokenizers import Tokenizer

MODEL_DIR = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2"
ENGINE = os.environ.get(
    "NPU_ENGINE", "/home/bcloud/1bit-MONSTER/npu-infer/build/npu_infer")
MAX_NEW = int(os.environ.get("NPU_MAX_NEW", "16"))

tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
special = json.load(open(f"{MODEL_DIR}/tokenizer_config.json"))

IM_START = tok.token_to_id("<|im_start|>")
IM_END = tok.token_to_id("<|im_end|>")
assert IM_START is not None and IM_END is not None, "im tokens missing"


def encode_chat(prompt: str):
    # Qwen3 ChatML: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text).ids
    return ids


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    ids = encode_chat(prompt)
    print(f"prompt tokens: {len(ids)}", file=sys.stderr)
    env = dict(os.environ)
    env["NPU_MODEL_PATH"] = f"{MODEL_DIR}/model.q4nx"
    env["NPU_RUNTIME_LAYERS"] = "1"
    env["NPU_PROMPT_IDS"] = ",".join(str(i) for i in ids)
    out = subprocess.run([ENGINE], env=env, capture_output=True, text=True,
                         timeout=600)
    # the engine prints "token ids" in the final line
    m = re.search(r"(\d+(?: \d+)+)", out.stdout)
    gen_ids = [int(x) for x in m.group(1).split()] if m else []
    text = tok.decode(gen_ids)
    print(f"\nGENERATED ({len(gen_ids)} tok): {text!r}")
    if out.returncode != 0:
        print("engine stderr tail:", out.stderr[-800:], file=sys.stderr)


if __name__ == "__main__":
    main()
