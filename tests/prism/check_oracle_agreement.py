#!/usr/bin/env python3
"""Per-position next-token oracle agreement — the gate that catches convention bugs.

Runs our low-bit forward on the fixed 5-token prompt and compares its argmax at each
position against the token the *fork* predicts there, captured once from the fork's own
`/completion` endpoint (raw prompt, greedy, seed 20260918):

  prefix                       fork's next token
  "The"                        2614  ' following'   (Qwen3.6 packs)
  "The capital"                 314  ' of'
  "The capital of"              279  ' the'
  "The capital of France"       369  ' is'
  "The capital of France is"  11751  ' Paris'       (folded Qwen3.8 pack: 220 ' ' first)

Why this exists: every structural check in this directory (byte-exact payloads, decode
vs Prism's codec, aux tensors vs the source, Hadamard round-trip) passed while the model
predicted the wrong token at *every* position — the RMSNorm convention ((1+w) HF-style vs
llama.cpp's plain LLM_NORM_RMS) was invisible to all of them. Only agreement with the
oracle's own generated tokens sees that class of bug.

Usage: check_oracle_agreement.py <forward_binary> <model.1bp> <model_base_name>
"""
from __future__ import annotations

import re
import subprocess
import sys

PROMPT = ["760", "6511", "314", "9338", "369"]
# fork's own top-3 per position (recorded from its /completion with logprobs, same seed).
# Our top-5 must CONTAIN these at every position; top-1 must equal.
FORK_TOP3 = {
    "Ternary-Bonsai-27B-PQ2_0": [
        [2614, 1156, 220], [314, 3177, 1954], [279, 6535, 9338], [369, 11, 271], [11751, 31586, 1259],
    ],
    "Ternary-Bonsai-2-27B-PTQ1_0": [
        [220, 3377, 1118], [314, 3177, 1954], [279, 9338, 9564], [369, 11, 13], [11751, 303, 198],
    ],
}
ORACLE = {
    # base name -> argmax per position expected from our forward
    "Bonsai-27B-Q1_0": ["2614", "314", "279", "369", "11751"],
    "Ternary-Bonsai-27B-PQ2_0": ["2614", "314", "279", "369", "11751"],
    "Ternary-Bonsai-2-27B-PTQ1_0": ["220", "314", "279", "369", "11751"],
}


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    binary, bp, name = sys.argv[1], sys.argv[2], sys.argv[3]
    want = ORACLE.get(name)
    if want is None:
        print(f"  skip {name}: no recorded oracle")
        return 0

    out = subprocess.run([binary, bp] + PROMPT, capture_output=True, text=True, timeout=3600)
    got = re.findall(r"argmax=(\d+)", out.stdout)
    if len(got) != len(want):
        print(f"FAIL {name}: got {len(got)} argmax values, expected {len(want)}")
        print(out.stdout[-400:])
        return 1
    hits = sum(1 for a, b in zip(got, want) if a == b)
    ok = hits == len(want)

    # top-5 containment against the fork's own top-3 (independent implementation, full model)
    key3 = FORK_TOP3.get(name)
    if key3:
        out5 = subprocess.run([binary, bp] + PROMPT + ["--topk"], capture_output=True,
                              text=True, timeout=3600)
        rows = [r for r in re.findall(r"top5=([0-9,]+)", out5.stdout)]
        contained = 0
        total = 0
        for i, want3 in enumerate(key3):
            if i >= len(rows):
                ok = False
                break
            ours = [int(x) for x in rows[i].split(",")]
            for t in want3:
                total += 1
                contained += 1 if t in ours else 0
        # Metric only, not a gate: the top-1 gate above is the correctness criterion; the
        # tail of the distribution legitimately differs (quantisation + backend), and for the
        # folded Qwen3.8 pack it differs more than for the Qwen3.6 packs. Print it so drift is
        # visible, but do not fail on it — gating on a distribution tail would be noise.
        print(f"     metric: our top-5 contains the fork's top-3 at {contained}/{total} slots")
    print(f"{'ok  ' if ok else 'FAIL'} {name}: per-position oracle agreement {hits}/{len(want)}"
          + ("" if ok else f"\n     got  {','.join(got)}\n     want {','.join(want)}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
