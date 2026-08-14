#!/usr/bin/env python3
"""Export a HF tokenizer.json (BPE) to the engine .htok binary format.
Format (from src/tokenizer.cpp rcpp_tokenizer_load):
  magic 'HTOK' | u32 version(=2) | u32 vocab | u32 merges | u32 bos | u32 eos
  then per vocab entry: u16 len + bytes
  then per merge (rank = insertion order): u32 a, u32 b, u32 merged
"""
import json, struct, sys

tok_path, out_path = sys.argv[1], sys.argv[2]
t = json.load(open(tok_path))
vocab = t["model"]["vocab"]  # token -> id
merges = t["model"]["merges"]  # "a b" list, ordered by rank
# qwen3's tokenizer.json: merges are in rank order; map tokens to ids via the
# added_tokens + vocab (added tokens come first in HF vocab usually, but the
# vocab dict is authoritative for the ids)
id_of = {tok: i for tok, i in vocab.items()}
for at in t.get("added_tokens", []):
    id_of[at["content"]] = at["id"]
merges_bin = []
for m in merges:
    if isinstance(m, list):
        a, b = m[0], m[1]
    else:
        a, b = m.split(" ")
    if a not in id_of or b not in id_of:
        continue
    merged = a + b
    mid = id_of.get(merged, 0)  # rank still from insertion order
    merges_bin.append((id_of[a], id_of[b], mid))
bos = 0; eos = 0
for at in t.get("added_tokens", []):
    if at.get("special") and at["content"] in ("<|endoftext|>", "<|im_end|>"):
        eos = at["id"]
with open(out_path, "wb") as f:
    f.write(b"HTOK")
    f.write(struct.pack("<IIIII", 2, len(vocab), len(merges_bin), bos, eos))
    for tok, i in vocab.items():
        b = tok.encode("utf-8")
        f.write(struct.pack("<H", len(b)) + b)
    for a, b, mid in merges_bin:
        f.write(struct.pack("<III", a, b, mid))
print(f"wrote {out_path}: vocab={len(vocab)} merges={len(merges_bin)} bos={bos} eos={eos}")
