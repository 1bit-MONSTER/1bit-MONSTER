#!/usr/bin/env python3
"""Convert a HuggingFace tokenizer.json to the .htok v2 format zaya reads.

The .htok v2 binary layout (matches src/tokenizer.cpp rcpp_tokenizer_load):
    magic "HTOK" | u32 version=2 | u32 vocab | u32 merges | u32 bos | u32 eos
    vocab: u16 len + bytes (id order)
    merges: u32 a, u32 b, u32 merged  (rank = insertion order, lower = priority)
    u32 num_special + u32 special ids (tokens containing < > |, longest first)

Usage: hf_tokenizer_to_htok.py tokenizer.json out.htok
"""
import json
import struct
import sys

KNOWN_BOS = {"<|begin_of_text|>", "<s>", "<|im_start|>"}
KNOWN_EOS = {"<|end_of_text|>", "</s>", "<|im_end|>"}


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} tokenizer.json out.htok", file=sys.stderr)
        return 1
    tok = json.load(open(sys.argv[1], encoding="utf-8"))
    vocab = tok["model"]["vocab"]  # token -> id
    merges = tok["model"]["merges"]  # "a b", priority order
    id_to_tok = [""] * len(vocab)
    for t, i in vocab.items():
        id_to_tok[i] = t

    triples, skipped = [], 0
    for m in merges:
        a, b = m.split(" ", 1)
        ia, ib, im = vocab.get(a), vocab.get(b), vocab.get(a + b)
        if ia is None or ib is None or im is None:
            skipped += 1
            continue
        triples.append((ia, ib, im))

    # Same special-token heuristic as build_htok_from_gguf (zaya_server.cpp):
    # any vocab token containing '<', '>' or '|', longest first.
    specials = sorted(
        (i for i, t in enumerate(id_to_tok) if any(c in t for c in "<>|")),
        key=lambda i: -len(id_to_tok[i]),
    )

    bos = eos = 0
    for t in tok.get("added_tokens", []):
        if not t.get("special"):
            continue
        if t["content"] in KNOWN_BOS:
            bos = t["id"]
        elif t["content"] in KNOWN_EOS:
            eos = t["id"]
    if bos == 0:  # fall back to vocab scan
        for i, t in enumerate(id_to_tok):
            if t in KNOWN_BOS:
                bos = i
            elif t in KNOWN_EOS:
                eos = i

    with open(sys.argv[2], "wb") as f:
        f.write(b"HTOK")
        f.write(struct.pack("<IIIII", 2, len(id_to_tok), len(triples), bos, eos))
        for t in id_to_tok:
            b = t.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        for a, b, m in triples:
            f.write(struct.pack("<III", a, b, m))
        f.write(struct.pack("<I", len(specials)))
        for s in specials:
            f.write(struct.pack("<I", s))
    print(
        f"wrote {sys.argv[2]}: {len(id_to_tok)} tokens, {len(triples)} merges "
        f"({skipped} skipped), {len(specials)} specials, BOS={bos} EOS={eos}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
