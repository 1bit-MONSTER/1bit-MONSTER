#!/usr/bin/env python3
"""make_mini_deepseek_v41.py — Phase-0 oracle fixtures for the V4/V4.1 arch work (research/ws13).

A tiny, seeded DeepSeek-V4 fixture with the HF oracle logits AND per-layer hidden
states, for two profiles:

  sliding      all layers `sliding_attention`
               → the engine PASSES today. Regression baseline for the modules that
                 exist (mHC, shared-KV MQA, sinks, hash/top-k MoE).

  compressed   1 sliding + 1 CSA + 1 HCA + 1 sliding
               → the engine FAILS today, by design: it does not implement the
                 compressors (CSA/HCA) or the Lightning Indexer. This is the gate
                 that can fail — the instrument the P1 work is measured against.

Why the reference is trustworthy: transformers 5.16.1 ships `DeepseekV4` **with**
`compressed_sparse_attention` / `heavily_compressed_attention` layers,
`compress_rates`, and `DeepseekV4CSACache` (which carries the `"indexer"` entries) —
so the oracle implements the very machinery the engine is missing. There is no
`DeepseekV41` class in transformers, so V4.1-only modules (engram, candidate
blocks, `ffn.gate.bias_vl`, the MTP rewrite) have **no** reference here; that is
recorded in research/ws13-arch-gap-closure/FINDINGS.md rather than papered over.

Usage:
    python3 Testing/make_mini_deepseek_v41.py /tmp/onebit-dsv4-csa --profile compressed
    python3 Testing/make_mini_deepseek_v41.py /tmp/onebit-dsv4-ssl --profile sliding

Env: needs torch + transformers with native DeepseekV4 (e.g. ~/ft-zaya/bin/python).
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file as st_save
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

# V4-Flash's own compress ratios (config.json: compress_ratios values {0, 4, 128}).
COMPRESS_RATES = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}

PROFILES = {
    "sliding": ["sliding_attention"] * 4,
    "compressed": ["sliding_attention", "compressed_sparse_attention",
                   "heavily_compressed_attention", "sliding_attention"],
}

# Default sliding windows per profile. The compressed profile needs a window
# much smaller than the prompt: with the rate at 4 the compressed entries carry
# the long-range context, so a model that ignores them cannot agree. (The first
# version of this fixture kept window=32 with a 5-token prompt and the ENGINE
# PASSED the compressed profile — the compressed entries were redundant with the
# sliding window and one 4-token entry was not enough to move top-20 logits.
# A gate that cannot fail is not a gate.)
DEFAULT_WINDOW = {"sliding": 32, "compressed": 4}

PROMPT = [5, 7, 9, 11, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="/tmp/onebit-dsv4-csa")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="compressed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-len", type=int, default=0,
                    help="0 = the original 5-token prompt; >5 = seeded random ids")
    ap.add_argument("--window", type=int, default=0, help="0 = profile default")
    ap.add_argument("--weights-dtype", choices=("float32", "bfloat16"), default="float32",
                    help="fixture weight storage. float32 for the per-layer gate: bf16 "
                         "rounding alone perturbs the layer-0 state by ~1e-4, which is "
                         "100x above the 1e-6 bound the gate wants to measure")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    window = args.window or DEFAULT_WINDOW[args.profile]

    cfg = DeepseekV4Config(
        vocab_size=1000, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
        head_dim=16, q_lora_rank=8, o_lora_rank=8,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        max_position_embeddings=256, sliding_window=window,
        # indexer dims: the Lightning Indexer needs its own heads/dim (HCA has none)
        index_n_heads=2, index_head_dim=8, index_topk=8,
        layer_types=PROFILES[args.profile],
        mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe"],
        compress_rates=COMPRESS_RATES,
    )
    model = DeepseekV4ForCausalLM(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())

    ids = torch.tensor([PROMPT])
    if args.prompt_len and args.prompt_len != len(PROMPT):
        g = torch.Generator().manual_seed(args.seed + 1)
        ids = torch.randint(0, 1000, (1, args.prompt_len), generator=g)
    prompt = ids[0].tolist()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    logits_last = out.logits[0, -1].float().cpu()
    np.save(os.path.join(args.outdir, "logits_last.npy"), logits_last.numpy())
    torch.save(logits_last, os.path.join(args.outdir, "logits_last.pt"))
    torch.save(ids, os.path.join(args.outdir, "ids.pt"))
    open(os.path.join(args.outdir, "ids.txt"), "w").write(" ".join(map(str, prompt)) + "\n")

    # ── Sensitivity: measured by the engine, not assumed ─────────────────────
    # A same-weights ablation is IMPOSSIBLE here and that is a finding: the
    # compressor's `position_bias` is shaped [compress_rate, dim], so the rate is
    # baked into the tensors (a real checkpoint's `compress_ratios` must be read
    # from its weights, never guessed). So sensitivity is established by the
    # engine instead: on this profile the engine (which ignores compressor
    # weights entirely) must DISAGREE, while it agrees 20/20 on the sliding
    # profile whose only difference is those two layer types. First-divergent
    # -layer attribution is P1.1's job and uses hidden_states.npz below.
    if "compressed_sparse_attention" in PROFILES[args.profile]:
        n_entries = len(prompt) // COMPRESS_RATES["compressed_sparse_attention"]
        print(f"  compressed entries the reference emits (CSA, rate "
              f"{COMPRESS_RATES['compressed_sparse_attention']}): "
              f"{n_entries} over {len(prompt)} tokens (window={window})")

    # Per-layer reference activations: the half of the instrument that lets P1
    # locate the FIRST diverging layer instead of only comparing final logits.
    hs = {f"hidden_{i}": h[0].float().cpu().numpy() for i, h in enumerate(out.hidden_states)}
    np.savez(os.path.join(args.outdir, "hidden_states.npz"), **hs)

    wdtype = torch.float32 if args.weights_dtype == "float32" else torch.bfloat16
    sd = {k: v.detach().to(wdtype).contiguous() for k, v in model.state_dict().items()}
    torch.save(sd, os.path.join(args.outdir, "model.pt"))
    st_save({k: v.float() for k, v in sd.items()}, os.path.join(args.outdir, "model.safetensors"))

    cfgd = cfg.to_dict()
    cfgd["_mini_fixture"] = True
    cfgd["_profile"] = args.profile
    cfgd["_weights_dtype"] = args.weights_dtype
    cfgd["_prompt_ids"] = prompt[:8] + (["..."] if len(prompt) > 8 else [])
    json.dump(cfgd, open(os.path.join(args.outdir, "config.json"), "w"), indent=2)

    # What the engine will trip over, named up front (the point of this profile).
    compressor = sorted({k.split(".", 2)[-1] for k in sd if ".compressor." in k})
    indexer = sorted({k.split(".", 2)[-1] for k in sd if ".indexer." in k})
    print(f"profile={args.profile} params={n_params} layers={len(PROFILES[args.profile])}")
    print(f"  layer_types={PROFILES[args.profile]}")
    print(f"  window={window} prompt_len={len(prompt)} weights={args.weights_dtype}")
    print(f"  compressor tensors: {compressor}")
    print(f"  indexer tensors:    {indexer}")
    print(f"  top1={int(logits_last.argmax())}  wrote {args.outdir}")


if __name__ == "__main__":
    main()
