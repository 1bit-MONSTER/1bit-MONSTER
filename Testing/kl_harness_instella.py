#!/usr/bin/env python3
"""kl_harness_instella.py — logits-level gate for Instella-MoE GGUF conversions.

Compares HF reference (bf16, CPU) vs llama.cpp GGUF on a fixed prompt set:
  - top-1 token agreement per generated position (catches graph/connectivity bugs)
  - mean KL(p_ref || p_target) per position, computed on the union of top-256 logits

Usage:
  python3 kl_harness_instella.py ref    --limit N --gen-tokens M [--ckpt HF_ID]
  python3 kl_harness_instella.py target --limit N --gen-tokens M --gguf PATH
  python3 kl_harness_instella.py compare [--limit N]

Outputs land in <out>/ref/ and <out>/target/ as per-prompt npz:
  tokens    int32 [M]        greedy generated ids
  top_ids   int32 [M,256]    top-256 token ids per position
  top_vals  float32 [M,256]  corresponding logits

Fail: mean top-1 agreement < 0.97 or mean KL > 0.10. This is the gate that a
short ppl eval cannot be: stock deepseek_v3 conversion of Instella (missing
gate_proj + wrong FarSkip residual stream) passes a 708-token ppl check and
fails this one hard. See docs/plans/instella-moe-16b-1bp.md.
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

CKPT = "amd/Instella-MoE-16B-A3B-Think"
OUT = Path(os.environ.get("KL_OUT", "instella-kl"))
GEN_TOKENS = 32
TOP_K = 256
PASS_AGREE = 0.97
PASS_KL = 0.10

PROMPTS = [
    # general knowledge / world facts
    "The capital of France is",
    "Explain how photosynthesis works in two sentences.",
    "What is the difference between a compiler and an interpreter?",
    "Who wrote 'One Hundred Years of Solitude' and when was it published?",
    "List the planets of the solar system in order from the sun.",
    # math / reasoning
    "If a train travels 240 km in 3 hours, what is its average speed in km/h?",
    "Solve for x: 3x + 7 = 22. Show your steps.",
    "What is 17 * 23? Answer with just the number.",
    "A bag contains 5 red and 7 blue marbles. Two are drawn without replacement. What is the probability both are red?",
    "The sum of two numbers is 30 and their difference is 6. What are the numbers?",
    "How many factors does 72 have? Explain briefly.",
    # code
    "Write a Python function that checks if a string is a palindrome.",
    "Write a bash one-liner that prints the 10 largest files in the current directory.",
    "What does this code print and why?\ndef f(x, y=[]):\n    y.append(x)\n    return y\nprint(f(1))\nprint(f(2))",
    "Write a SQL query to find duplicate emails in a users table.",
    "Explain the difference between a list and a tuple in Python.",
    # reasoning / logic
    "If all bloops are glorps and some glorps are zorps, can we conclude some bloops are zorps? Why or why not?",
    "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?",
    "A clock shows 3:15. What is the angle between the hour and minute hands?",
    "Three people check into a hotel room that costs $30. Each pays $10. Later the clerk realizes the room is $25 and gives $5 to the bellhop to return. The bellhop keeps $2 and gives each person $1 back. Now each person paid $9, totaling $27, plus the $2 the bellhop kept = $29. Where is the missing dollar?",
    "You have 8 coins, one is heavier than the rest. Using a balance scale, what is the minimum number of weighings to find it?",
    # instruction following / formatting
    "List three reasons to exercise, as a bulleted list.",
    "Summarize the water cycle in exactly three sentences.",
    "Write a haiku about autumn.",
    "Give me a JSON object with keys 'name', 'age', and 'city' for a fictional person.",
    "Translate 'the early bird catches the worm' into Spanish, then explain the meaning.",
    "Write a short email to a colleague declining a meeting invitation politely. Include a subject line.",
    # STEM / science
    "What is the difference between an atom and a molecule?",
    "Explain Newton's three laws of motion briefly.",
    "Why is the sky blue? Give a concise scientific explanation.",
    "What is the pH scale and what does pH 7 indicate?",
    "Explain what a black hole is in simple terms.",
    # few-shot / pattern
    "2, 4, 8, 16, ?  What comes next and why?\n1, 1, 2, 3, 5, 8, ?\n3, 9, 27, ?",
    "Complete the analogy: cat is to kitten as dog is to",
    "apple:fruit :: carrot:",
    # open-ended / creative
    "Write a one-paragraph short story about a robot that learns to paint.",
    "What would you do if you found a wallet on the street?",
    "Describe the taste of an apple to someone who has never eaten one.",
    # knowledge boundaries / honesty
    "What are some well-known limitations of large language models?",
    "Explain the difference between correlation and causation with an example.",
    # editing / language
    "Fix the grammar: 'Me and him went to the store yesterday.'",
    "What is the opposite of 'ubiquitous'?",
    "Define 'serendipity' and use it in a sentence.",
    # math word problems
    "John buys 3 apples at $0.50 each and 2 bananas at $0.30 each. How much does he spend in total?",
    "A rectangle has length 12 and width 5. What is its area and perimeter?",
    "If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets?",
    # code reasoning
    "What is the time complexity of binary search and why?",
    "Write a recursive function in Python to compute the nth Fibonacci number.",
    "Explain what a 'race condition' is in multithreading with a small example.",
]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- ref phase
def run_ref(limit, gen_tokens, ckpt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # bf16 refs diverge from fp32 by slope ~0.976 over 27 layers (measured) —
    # enough to fail the KL gate on its own noise. Default to fp32; override
    # with KL_REF_DTYPE=bf16 to reproduce the old behavior.
    dtype = torch.bfloat16 if os.environ.get("KL_REF_DTYPE", "fp32") == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=dtype, trust_remote_code=True
    )
    model.eval()

    prompts = PROMPTS[:limit]
    (OUT / "ref").mkdir(parents=True, exist_ok=True)
    (OUT / "prompts.jsonl").write_text(
        "\n".join(json.dumps({"i": i, "prompt": p}) for i, p in enumerate(prompts))
    )

    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt")
        n_prompt = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )
        seq = out.sequences[0].tolist()
        gen = seq[n_prompt:]
        scores = torch.stack(out.logits).squeeze(1).float()  # [M, vocab]
        top_vals, top_ids = torch.topk(scores, TOP_K, dim=-1)
        np.savez(
            OUT / "ref" / f"{i:03d}.npz",
            tokens=np.asarray(gen, dtype=np.int32),
            top_ids=top_ids.numpy().astype(np.int32),
            top_vals=top_vals.numpy().astype(np.float32),
            prompt_ids=np.asarray(ids["input_ids"][0].tolist(), dtype=np.int32),
        )
        log(f"ref {i:03d}: prompt={n_prompt} gen={len(gen)} first={gen[:8]}")


# ------------------------------------------------------------- target phase
def run_target(limit, gen_tokens, gguf, teacher_forced=False):
    from llama_cpp import Llama

    prompts = [json.loads(l)["prompt"] for l in (OUT / "prompts.jsonl").read_text().splitlines()[:limit]]
    (OUT / "target").mkdir(parents=True, exist_ok=True)

    llm = Llama(model_path=gguf, n_ctx=8192, n_gpu_layers=0, verbose=False, logits_all=True)
    for i, p in enumerate(prompts):
        ids = llm.tokenize(p.encode(), add_bos=True)
        ref_ids = np.load(OUT / "ref" / f"{i:03d}.npz")["prompt_ids"]
        if len(ids) != len(ref_ids) or list(ids) != list(ref_ids):
            log(f"target {i:03d}: TOKENIZE MISMATCH (llama {len(ids)} vs hf {len(ref_ids)}) — skipping")
            continue
        # teacher-forced: feed HF's greedy tokens so both models see identical
        # contexts; isolates graph fidelity from near-tie sampling noise.
        forced = np.load(OUT / "ref" / f"{i:03d}.npz")["tokens"] if teacher_forced else None
        llm.reset()
        llm.eval(ids)
        gen, top_ids, top_vals = [], [], []
        for j in range(gen_tokens):
            # NOTE: use llm._scores (sliced to n_tokens), not llm.scores (raw n_ctx buffer)
            scores = np.asarray(llm._scores[-1], dtype=np.float32)
            idx = np.argpartition(-scores, TOP_K)[:TOP_K]
            idx = idx[np.argsort(-scores[idx])]
            nxt = int(forced[j]) if forced is not None else int(idx[0])
            gen.append(nxt)
            top_ids.append(idx.astype(np.int32))
            top_vals.append(scores[idx].astype(np.float32))
            llm.eval([nxt])
        np.savez(
            OUT / "target" / f"{i:03d}.npz",
            tokens=np.asarray(gen, dtype=np.int32),
            top_ids=np.asarray(top_ids, dtype=np.int32),
            top_vals=np.asarray(top_vals, dtype=np.float32),
            prompt_ids=np.asarray(ids, dtype=np.int32),
        )
        log(f"target {i:03d}: gen={len(gen)} first={gen[:8]}")


# ------------------------------------------------------------- compare phase
def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def run_compare(limit):
    prompts = [json.loads(l) for l in (OUT / "prompts.jsonl").read_text().splitlines()]
    prompts = prompts[:limit]
    tot_agree, tot_pos, kl_sum, kl_pos = 0, 0, 0.0, 0
    worst = []
    for i, meta in enumerate(prompts):
        r = np.load(OUT / "ref" / f"{i:03d}.npz")
        t = np.load(OUT / "target" / f"{i:03d}.npz")
        agree = (r["tokens"] == t["tokens"]).sum()
        n = len(r["tokens"])
        per_pos_kl = []
        for j in range(n):
            union = np.union1d(r["top_ids"][j], t["top_ids"][j])
            p = _softmax(np.asarray([r["top_vals"][j][list(r["top_ids"][j]) == u][0] if (r["top_ids"][j] == u).any() else -1e9 for u in union]))
            q = _softmax(np.asarray([t["top_vals"][j][list(t["top_ids"][j]) == u][0] if (t["top_ids"][j] == u).any() else -1e9 for u in union]))
            per_pos_kl.append(float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum()))
        kl = float(np.mean(per_pos_kl))
        tot_agree += agree; tot_pos += n; kl_sum += kl * n; kl_pos += n
        worst.append((kl, agree / n, i, meta["prompt"][:60]))
        log(f"cmp {i:03d}: agree={agree}/{n} ({agree/n:.3f}) mean_kl={kl:.4f}")

    agree_rate = tot_agree / tot_pos
    mean_kl = kl_sum / kl_pos
    worst.sort(reverse=True)
    log("=" * 60)
    log(f"AGGREGATE: top-1 agreement {agree_rate:.4f}  mean KL {mean_kl:.4f}  over {tot_pos} positions")
    log(f"worst 3 prompts: " + "; ".join(f"#{i} kl={k:.3f} agr={a:.3f} '{p}'" for k, a, i, p in worst[:3]))
    ok = agree_rate >= PASS_AGREE and mean_kl <= PASS_KL
    log(f"GATE: {'PASS' if ok else 'FAIL'}  (need agree>={PASS_AGREE} kl<={PASS_KL})")
    return 0 if ok else 1


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "compare"
    args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[2:] if "=" in a)
    limit = int(args.get("limit", len(PROMPTS)))
    gen = int(args.get("gen-tokens", GEN_TOKENS))
    if phase == "ref":
        return run_ref(limit, gen, args.get("ckpt", CKPT))
    if phase == "target":
        gguf = args.get("gguf")
        if not gguf:
            log("target requires --gguf=PATH")
            return 1
        return run_target(limit, gen, gguf, teacher_forced="--teacher-forced" in sys.argv)
    if phase == "compare":
        return run_compare(limit)
    log(f"unknown phase {phase}; use ref|target|compare")
    return 1


if __name__ == "__main__":
    sys.exit(main())
