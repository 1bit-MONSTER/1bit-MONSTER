#!/usr/bin/env python3
"""Print token strings for ids from a Prism GGUF (no gguf-py needed).

`llama-tokenize` in the fork tokenises but cannot detokenise, and stock gguf-py
refuses these files (private type ids 142/143), so this reads
`tokenizer.ggml.tokens` with gguf_header and prints the requested ids. Byte-level
BPE means the strings are escaped (e.g. 220 -> 'Ġ'), which is fine for judging
whether a forward pass produced sensible text.

Usage: python3 dump_prism_vocab.py <model.gguf> <id> [id ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "docs" / "research" / "prism-bonsai-27b"))

import gguf_header as gh  # noqa: E402


def load_tokens(path: str) -> list[str]:
    r = gh.Reader(path)
    r.raw(4)
    r.unpack("I")
    r.unpack("Q")
    (nkv,) = r.unpack("Q")
    toks = None
    for _ in range(nkv):
        k = r.string()
        (t,) = r.unpack("I")
        if k == "tokenizer.ggml.tokens" and t == gh.GGUF_TYPE_ARRAY:
            (et,) = r.unpack("I")
            (n,) = r.unpack("Q")
            toks = [r.string() for _ in range(n)]
        else:
            gh.read_value(r, t)
    if toks is None:
        raise SystemExit("no tokenizer.ggml.tokens in this GGUF")
    return toks


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    toks = load_tokens(sys.argv[1])
    print(f"vocab={len(toks)}")
    for a in sys.argv[2:]:
        i = int(a)
        print(f"{i}\t{toks[i]!r}" if 0 <= i < len(toks) else f"{i}\t<out of range>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
