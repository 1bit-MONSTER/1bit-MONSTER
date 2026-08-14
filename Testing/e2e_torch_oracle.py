#!/usr/bin/env python3
"""e2e_torch_oracle.py — independent next-token oracle for the safetensors
end-to-end check. Loads the checkpoint with torch + transformers (BF16 -> f32)
and reports the greedy argmax next token for the same seed tokens the C++
harness uses, plus an 8-step chain from token 5.

Run:
    python3 Testing/e2e_torch_oracle.py /tmp/onebit-e2e/smollm
"""
import sys
import torch

from transformers import AutoModelForCausalLM, AutoConfig

def main():
    if len(sys.argv) < 2:
        print("usage: e2e_torch_oracle.py <model_dir>")
        return 1
    d = sys.argv[1]

    cfg = AutoConfig.from_pretrained(d, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32, config=cfg)
    model.eval()

    def next_tok(tok: int) -> int:
        with torch.no_grad():
            out = model(torch.tensor([[tok]]))
            return int(out.logits[0, -1].argmax())

    for tok in (5, 42, 99, 1000, 4242, 31337):
        print(f"seed {tok:6d}: torch->{next_tok(tok)}")

    t = 5
    chain = []
    for _ in range(8):
        t = next_tok(t)
        chain.append(t)
    print("torch 8-step chain from 5:", chain)

if __name__ == "__main__":
    sys.exit(main())
