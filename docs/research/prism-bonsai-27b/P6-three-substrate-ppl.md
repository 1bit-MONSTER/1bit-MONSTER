# P6 — three-substrate perplexity on ONE token-id stream

Recorded 2026-09-19. Why this file exists: the 53-gate suite proves fidelity-to-oracle, not text
quality. This is the one thing it cannot test. Three columns score the same Prism ML Bonsai 27B
packs over the same token-id stream — our HIP engine, the HRX/Loom substrate, and the Prism
llama.cpp fork (oracle). Primary figure is PPL; throughput is not claimed here.

## The stream, and why it is the whole point

A PPL comparison across two tokenizations is void, so the stream is produced ONCE — by the oracle
fork's own perplexity tokenizer — and every column scores exactly that file.

* Corpus: `tests/prism/ppl/slice200.txt` — the first 52486 bytes of wikitext-2-raw `wiki.test.raw`
  (`cmp`-verified prefix), md5 `88510b18f6ed7ece89a20c59c9bda6a2`
  `[3-packs | wikitext-2-raw slice200 | corpus | strixhalo-unknown | - | wiki.test.raw | 2026-09-19]`.
  **No wikitext-103 text file exists on this box.** The objective names the wikitext-103 test set;
  the path the old JSON references (`/home/bcloud/halo-ai/datasets/wikitext-103-test.txt`) is
  absent and a filesystem search found no 103 text. A wikitext-2 slice is used instead, and the
  substitution is stated here rather than hidden.
* Canonical ids: `tests/prism/ppl/slice200.ids.txt` — 12599 ids, md5
  `62cd3e0de33b83123e1b9ece75e57580`, fnv1a64 `867c8618e1944fda`, `add_bos=0 add_eos=0`
  `[3-packs | wikitext-2-raw slice200 | oracle tokenizer dump | strixhalo-unknown | 12599 | wiki.test.raw slice200 | 2026-09-19]`.
  Dumped by the fork via its `P6_DUMP_TOKENS` diagnostic.
* Tokenizer parity: our `.htok` encoder vs the fork's `llama-tokenize`, **24/24 whole sequences
  identical** (`tests/prism/ppl/tokenizer_parity.txt`, VERDICT AGREE)
  `[3-packs | .htok vs llama-tokenize | tokenizer parity | strixhalo-unknown | - | 24 sequences | 2026-09-19]`.
  The stream is therefore identical across columns **by construction**, and that parity is
  measured, not assumed.

## Protocol (mirrors the fork's `llama-perplexity` default)

`n_ctx=2048`, `first=n_ctx/2=1024`, non-overlapping chunks, `n_chunk = 12599/2048 = 6`; within each
chunk the model is fed positions 0..2046 and scored on positions 1024..2046 (1023 per chunk, 6138
total); KV reset at every chunk; BOS substitution not applied (the vocab's `add_bos` is 0). Our
harness `tests/prism/test_prism_ppl_ids.hip` was written to this exact window.

## Results — PPL per column, same stream, same window

| pack | our engine (HIP PrismEngine) | fork oracle (Prism llama.cpp) | ours − fork |
|---|---:|---:|---:|
| Bonsai-27B-Q1_0 | 10.9750 | 10.9992 | −0.024 |
| Ternary-Bonsai-2-27B-PTQ1_0 | 8.7168 | 7.5887 | +1.128 |
| Ternary-Bonsai-27B-PQ2_0 | 9.8819 | 9.8897 | −0.008 |

Tags for the table, in row order — our engine
`[<pack> | <format> | HIP PrismEngine | strixhalo-unknown | 6138 | wiki.test.raw slice200 | 2026-09-19]`,
fork oracle
`[<pack> | <format> | Prism llama.cpp fork | strixhalo-unknown | 6138 | wiki.test.raw slice200 | 2026-09-19]`,
delta
`[<pack> | <format> | derived | strixhalo-unknown | 6138 | wiki.test.raw slice200 | 2026-09-19]`,
with `<pack>/<format>` = `Bonsai-27B-Q1_0/Q1_0`, `Ternary-Bonsai-2-27B-PTQ1_0/PTQ1_0`,
`Ternary-Bonsai-27B-PQ2_0/PQ2_0`.

Per-chunk ours and the fork's per-chunk cumulative values, and the full raw logs, are kept in
`tests/prism/ppl/` (`our_engine_ppl.txt`, `fork_*.ppl.txt`).

## Column 2 — HRX/Loom: NOT PRODUCED (blocked, evidence kept)

```
llama-perplexity -m <Prism GGUF> --device HRX0 -c 2048 -f slice200.txt
```

on the pinned hrx-b66 (`/home/bcloud/hrx-slice/hrx-llamacpp/out/llama-hrx-b66`, gfx1151) aborts
during graph reservation:

```
pre-allocated tensor (cache_r_l0 (reshaped) (view) (view)) in a buffer (HRX0)
that cannot run the operation (SCALE)
```

Full backtrace in `tests/prism/ppl/hrx_blocker.txt`. This is a hard runtime blocker (the substrate
fails closed on an unsupported op for this GGUF), not a measurement we declined to make. No HRX/Loom
runtime is linked into our engine — it is an alternate substrate measured, or here blocked, on its
own.

## Findings

* **Q1_0 and PQ2_0 reproduce the oracle's real-text quality to within 0.22% and 0.08%.** Fidelity
  and quality agree for these two packs.
* **PTQ1_0 does not: our PPL is 14.9% higher than the oracle's** (8.7168 vs 7.5887). The protocol
  is identical across all three packs, so this is pack-specific engine behavior, not a harness
  difference. PTQ1_0 is the pack whose GEMV dot is the approximate dp4a path (its float-tile
  predecessor was exact per the P3 record); an approximate dot showing up as real-text quality loss
  is the leading candidate, but the mechanism is **not isolated here** and is recorded as an open
  question, not a finding. That one pack can drift this far while the other two agree is exactly the
  thing this goal was built to surface.
* The duplicate-runner incident of 01:21 is kept visible: `our_engine_ppl.CONTAMINATED.txt` holds the
  aborted first attempt, and `run_ppl_ids.sh` now holds an flock so a second runner exits instead of
  contending. The reported numbers are from the single clean run at 01:22:26Z.

## Provenance

* Our engine: `/tmp/ppl_ids_v2 <pack> /home/bcloud/models/prism/eval/slice200.ids.txt 2048 0`,
  driven by `tests/prism/ppl/run_ppl_ids.sh`; source `tests/prism/test_prism_ppl_ids.hip`; packs
  `~/models/prism/1bp/*.1bp`.
* Fork oracle: `llama-perplexity` in `/home/bcloud/prism/llama.cpp` (build `build/bin`, P6 dump
  patch in `tools/perplexity/perplexity.cpp`), over `slice200.txt`.
* Tokenizer parity: `tests/prism/check_tokenizer_parity.py` + `tests/prism/tokenize_htok.cpp`.
* No throughput claim is made; no quiet-window triad is claimed in this report.
