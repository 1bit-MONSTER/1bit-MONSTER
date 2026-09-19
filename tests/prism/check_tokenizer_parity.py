#!/usr/bin/env python3
"""check_tokenizer_parity.py — does OUR .htok tokenizer agree with the oracle fork's tokenizer?

P6 requires the shared token stream to be identical across columns. The chosen construction is
that one tokenizer (the oracle's) produces the ids and every column consumes them, so our engine
never re-tokenizes. This script verifies — rather than assumes — whether our own tokenizer would
have agreed, over a sample of whole sequences (one non-empty line each).

Usage: check_tokenizer_parity.py <model.htok> <gguf> <text.txt> [n_lines=24]
       environment: LLAMA_TOKENIZE (path to the fork's llama-tokenize)

Exit 0 always (this is an evidence producer); the caller records the agreement rate.
"""
import os
import re
import subprocess
import sys

HTOK = sys.argv[1]
GGUF = sys.argv[2]
TEXT = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 24
LT = os.environ.get("LLAMA_TOKENIZE", "/home/bcloud/prism/llama.cpp/build/bin/llama-tokenize")
OURS = os.environ.get("TOKENIZE_HTOK", "/tmp/tokenize_htok")

lines = [l for l in open(TEXT, encoding="utf-8").read().split("\n") if l.strip()][:N]
sample = "/tmp/parity_sample.txt"
open(sample, "w", encoding="utf-8").write("\n".join(lines) + "\n")

ours = {}
out = subprocess.run([OURS, HTOK, sample], capture_output=True, text=True).stdout
for row in out.splitlines():
    parts = row.split("\t")
    if len(parts) < 3:
        continue
    ids = [int(x) for x in parts[2].split()]
    ours[int(parts[0])] = ids

def fork_ids(text):
    r = subprocess.run([LT, "-m", GGUF, "-p", text, "--ids", "--no-bos"],
                       capture_output=True, text=True)
    m = re.search(r"\[([0-9,\s]*)\]", r.stdout)
    return [int(x) for x in m.group(1).split(",")] if m else None

agree = 0
mism = []
for i, ln in enumerate(lines):
    a = ours.get(i)
    b = fork_ids(ln)
    if a is not None and b is not None and a == b:
        agree += 1
    else:
        d = -1
        if a is not None and b is not None:
            for k in range(min(len(a), len(b))):
                if a[k] != b[k]:
                    d = k
                    break
        mism.append((i, len(a) if a else None, len(b) if b else None, d))

print(f"tokenizer parity: {agree}/{len(lines)} sequences identical")
for i, na, nb, d in mism:
    print(f"  mismatch seq={i} ours_n={na} fork_n={nb} first_diff_at={d}")
    print(f"    ours: {ours.get(i, [])[:16]}")
    print(f"    fork: {fork_ids(lines[i])[:16]}")
print("VERDICT: " + ("AGREE" if agree == len(lines) else "DIVERGE"))
