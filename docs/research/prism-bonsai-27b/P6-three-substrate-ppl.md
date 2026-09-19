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


---

## Vulkan cross-check (goal `mu7ukkbl`) — is the PPL a substrate artifact?

Operator question: the section-3/5 numbers "seem really low". This checks whether the value is an
artifact of the CPU oracle or of our HIP engine.

### (a) The oracle's own Vulkan backend, same stream, same protocol

```
/home/bcloud/prism/llama.cpp/build/bin/llama-perplexity \
  -m <gguf> -f ~/models/prism/eval/slice200.txt -c 2048 -t 16 -ngl 99
```

Offload was confirmed in the run log, not assumed:
`llama_prepare_model_devices: using device Vulkan0 (AMD Radeon 8060S Graphics (RADV STRIX_HALO))`
and `load_tensors: layer N assigned to device Vulkan0`. The stream hash logged by the fork was the
same 12 599 ids / `fnv1a64=867c8618e1944fda` for every pack.

| Pack | Fork CPU | Fork **Vulkan** (`-ngl 99`) | Δ |
|---|---:|---:|---:|
| Bonsai-27B-Q1_0 | 10.9992 | **10.9992** | 0.0000 [Bonsai-27B-Q1_0 | Q1_0 | fork Vulkan vs CPU | strixhalo-unknown | 6138 | slice200 | 2026-09-19] |
| Ternary-Bonsai-2-27B-PTQ1_0 | 7.5887 | **7.5887** | 0.0000 [Ternary-Bonsai-2-27B-PTQ1_0 | PTQ1_0 | fork Vulkan vs CPU | strixhalo-unknown | 6138 | slice200 | 2026-09-19] |
| Ternary-Bonsai-27B-PQ2_0 | 9.8897 | **9.8897** | 0.0000 [Ternary-Bonsai-27B-PQ2_0 | PQ2_0 | fork Vulkan vs CPU | strixhalo-unknown | 6138 | slice200 | 2026-09-19] |

**Result: the oracle is backend-invariant on this stream.** Vulkan offload reproduces the CPU number
to four decimals on all three packs. So the absolute value is a property of the model + stream, not
of the CPU path. (Raw logs: `tests/prism/ppl/fork_vulkan*.txt`.)

### (b) Our engine's own Vulkan path — NOT PRODUCIBLE (blocker with evidence)

Asked to "use the engine", we checked whether our Vulkan/ZINC path can drive the same stream
end-to-end. It cannot, today:

* The **only** Prism Vulkan asset in the repo is `kernels/vulkan/dmmv_prism.comp` — a block-decoding
  **GEMV** — plus the standalone `tests/test_vulkan_prism.cpp` harness. `kernels/vulkan/` otherwise
  holds `dmmv_q1_bonsai.comp`, `dmmv_tq2_bonsai.comp`, `matmul_fp32.comp`, `zaya_cca_attn.comp`:
  there is no Prism attention, GDN recurrence, RMSNorm, RoPE or 64-layer forward in Vulkan.
* The engine's Vulkan backends contain **zero** Prism references:
  `backend_vulkan.cpp` (Zaya1-8B backend) = 0, `backend_vulkan_hpp.cpp` = 0,
  `backend_zinc.cpp` (ZINC) = 0, `backend_ggml_vulkan.cpp` = 0 — and the latter is a **stub**
  because `third_party/llama.cpp` is not checked out (no `ggml.h`).
* ZINC on Prism is blocked independently (P5 record: segfault in `zinc/src/model/loader.zig:666`
  uploading a 13.9 MB tensor; no Zig toolchain in tree) `[Prism ML Bonsai 27B | ZINC | blocked | strixhalo-unknown | - | - | 2026-09-19]`.
* What the Vulkan path **can** do is verified, and is not nothing: the GEMV harness builds and runs
  on gfx1151 and is per-element correct for all three packs — Q1_0 `max_abs_err=0.000000` at
  M=16/K=256 and `0.000005` at the model shape M=17408/K=5120; PQ2_0 and PTQ1_0 likewise
  `[3-packs | Q1_0+PQ2_0+PTQ1_0 | Vulkan dmmv_prism.comp vs CPU ref | strixhalo-unknown | - | synthetic | 2026-09-19]`.
  These are tiny/warm figures, not a throughput claim.

**Conclusion.** A Vulkan PPL from *our* engine would require authoring the Prism forward
(attention/GDN/norms/rope) as Vulkan kernels plus a host driver — a port, not a configuration. It is
recorded as a blocker with the inventory above, not silently dropped. The only full-forward Vulkan
substrate available today is the oracle fork's own backend, which is column (a) and is a measurement
oracle, never a dependency.

### (c) What the cross-check settles

1. **The PPL values are not a substrate artifact.** The oracle is identical on CPU and Vulkan, and
   our HIP engine matches it on Q1_0 (−0.22%) and PQ2_0 (−0.08%).
2. **PTQ1_0's +14.9% gap is localized to our HIP PTQ1_0 path.** Fork Vulkan = fork CPU = 7.5887,
   while our HIP engine = 8.7168 `[Ternary-Bonsai-2-27B-PTQ1_0 | PTQ1_0 | HIP engine vs both oracle backends | strixhalo-unknown | 6138 | slice200 | 2026-09-19]`.
   Because the oracle reproduces itself exactly across two backends, the discrepancy cannot be
   attributed to the evaluation; it is our engine's PTQ1_0 decode/dot (leading candidate: the
   approximate dp4a path). The mechanism remains open; this cross-check rules out the measurement.
3. **No throughput/quiet claim is made here.** The GEMV numbers above are tiny-shape and warm.
