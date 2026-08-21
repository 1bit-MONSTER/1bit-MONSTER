#!/usr/bin/env python3
"""tokenizer_json_to_htok.py — convert HF tokenizer.json to .htok v2.

Writes the binary format read by rcpp_tokenizer_load (src/tokenizer.cpp):
  "HTOK" u32 version(2) u32 vocab_size u32 num_merges u32 bos u32 eos
  vocab[].: u16 len + bytes            (id = position; empty bytes = unused id)
  merges[]: u32 a u32 b u32 merged     (rank = insertion order)
  u32 num_special + u32 special_ids[]

Usage:
  python3 tokenizer_json_to_htok.py tokenizer.json out.htok
"""
import argparse, json, struct, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tokenizer_json')
    ap.add_argument('out')
    args = ap.parse_args()

    t = json.load(open(args.tokenizer_json))
    m = t['model']
    vocab = m['vocab']                      # str -> id
    merges = m['merges']                    # ["a b", ...]
    added = {a['content']: a['id'] for a in t.get('added_tokens', [])}

    id_to_str = {}
    for s, i in vocab.items():
        id_to_str[i] = s
    for s, i in added.items():
        id_to_str[i] = s
    if not id_to_str:
        print('error: empty vocab', file=sys.stderr)
        sys.exit(1)

    vocab_size = max(id_to_str) + 1
    # Empty ids beyond the tokenizer's own range (e.g. Qwen3 reserves 293
    # slots past its 22 declared specials) get empty strings — the loader
    # skips them for encode but keeps id numbering aligned with the model.
    out = [id_to_str.get(i, '') for i in range(vocab_size)]
    assert len(out) == vocab_size

    def id_of(s):
        i = vocab.get(s)
        if i is None:
            i = added.get(s)
        if i is None:
            print(f'error: merge component {s!r} not in vocab', file=sys.stderr)
            sys.exit(1)
        return i

    merge_triples = []
    for ent in merges:
        if isinstance(ent, list):
            a, b = ent[0], ent[1]
        else:
            a, b = ent.split()
        merged = a + b
        merge_triples.append((id_of(a), id_of(b), id_of(merged)))

    bos = vocab.get('<|endoftext|>')
    if bos is None:
        bos = added.get('<|endoftext|>', 0)
    eos = vocab.get('<|im_end|>')
    if eos is None:
        eos = added.get('<|im_end|>', bos or 0)

    special_ids = sorted(added.values())

    with open(args.out, 'wb') as f:
        f.write(b'HTOK')
        f.write(struct.pack('<5I', 2, vocab_size, len(merge_triples), bos, eos))
        for s in out:
            b = s.encode('utf-8')
            if len(b) > 65535:
                print(f'error: token too long ({len(b)} B): {s[:40]!r}',
                      file=sys.stderr)
                sys.exit(1)
            f.write(struct.pack('<H', len(b)))
            f.write(b)
        for a, b, merged in merge_triples:
            f.write(struct.pack('<3I', a, b, merged))
        f.write(struct.pack('<I', len(special_ids)))
        for sid in special_ids:
            f.write(struct.pack('<I', sid))
    print(f'wrote {args.out}: vocab={vocab_size} merges={len(merge_triples)} '
          f'specials={len(special_ids)} bos={bos} eos={eos}')


if __name__ == '__main__':
    main()
