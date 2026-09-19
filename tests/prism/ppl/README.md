# tests/prism/ppl — evidence for the P6 three-substrate perplexity comparison

See `docs/research/prism-bonsai-27b/P6-three-substrate-ppl.md` for the report.

Files:
- `slice200.txt` — the corpus slice: first 52 486 bytes of wikitext-2-raw `wiki.test.raw`
  (`cmp`-verified prefix). md5 `88510b18f6ed7ece89a20c59c9bda6a2`.
- `slice200.ids.txt` — the canonical token-id stream, dumped by the oracle fork's own perplexity
  tokenizer (diagnostic `P6_DUMP_TOKENS`). 12 599 ids, md5 `62cd3e0de33b83123e1b9ece75e57580`,
  fnv1a64 `867c8618e1944fda`, `add_bos=0 add_eos=0`. Every column scores exactly this file.
- `tokenizer_parity.txt` — our `.htok` tokenizer vs the fork's `llama-tokenize` over 24 whole
  sequences (result: 24/24 identical, VERDICT AGREE).
- `hrx_blocker.txt` — why column 2 (HRX/Loom) could not run: the pinned hrx-b66 binary fails
  closed on the Prism GGUF during graph reservation (unsupported op `SCALE`).
- `fork_<pack>.ppl.txt` — per-pack excerpts from the fork oracle runs (stream hash, chunking,
  per-chunk cumulative PPL, final estimate).

The corpus is a wikitext-2 substitute for the wikitext-103 set named in the objective; no
wikitext-103 text file exists on this box.
