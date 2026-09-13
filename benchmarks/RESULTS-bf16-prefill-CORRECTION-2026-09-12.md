# CORRECTION — the "native @1k prefill beats FLM by 40%" claim is withdrawn

**Date:** 2026-09-12 (dsh agent, resumed session)
**Affects:** `benchmarks/RESULTS-npu-prefill-parity-2026-09-12.md`,
`engine/npu/generators/FK3-STATUS-2026-09-12.md`, commit messages on
`goal/runlist-decode-wire` (`7e4a86835`, `64914b1a9`, `d5dabcc92`).

## What was claimed

> Native bf16 prefill (`NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=1024`)
> reaches **2087.7 tok/s @1024 tokens** on Qwen3-0.6B (+40% over FLM's published
> 1494), verified because the `boot=` token equals the `NPU_ATTN_CPU=1`
> reference (44402 @1024).

## Why it is wrong

1. **The bf16 prefill layer body only computes 256 rows.** Every GEMM in that
   path goes through `Bf16Mm::ensure_a()` (`npu_engine_bf16_mm.h`), which stages
   exactly two 128-row halves — `A[0..127]` into `a_cache0` and `A[128..255]`
   into `a_cache1` — and `gemm_wait()` copies back `128 * N`. The host loops are
   likewise written as exactly two batches (`int h0 = npt < 128 ? npt : 128;` and
   `for (int pi = 128; pi < npt; pi++)`). Rows ≥ 256 are never computed and keep
   stale data from the previous layer.
   So `NPU_PREFILL_MAX=1024` did **not** prefetch 1024 tokens; it ran a 256-token
   pipeline and reported the wall time as if it had done 1024.

2. **The correctness gate used was self-referential.** It compared
   bf16-NPU-attention against `NPU_ATTN_CPU=1` — but that CPU variant is the
   *same broken bf16 pipeline* with only the attention swapped. Two variants of
   the same wrong path agreed, and that agreement was read as verification.

3. **The attention ELF never loaded.** `Bf16Mm::init` looked for
   `attn_mha_1024_nh16.elf` only in FLM's per-model xclbin dir
   (`/home/bcloud/amd-oss/fastflowlm/src/xclbins/<model>/`), which contains only
   `mm/dequant/attn/layer.xclbin`. `strace` confirms `ENOENT` on every run, so
   every "1024-token" measurement actually used the embedded captured ELF.

## The decisive measurement

Same prompt, Qwen3-0.6B, three independent paths, boot token only:

| prompt tokens | `NPU_RUNLIST=1` (int8, byte-exact vs FLM) | `NPU_FLM_PREFILL=1` (FLM's own prefill) | bf16 native (`NPU_PREFILL_BF16=1`) |
|---|---|---|---|
| 256 | **1614** | **1614** | **1614** ✅ |
| 512 | **220** | **220** | 132352 ❌ |
| 1024 | **25** | **25** | 44402 ❌ |

(`[1] <tok>` printed by `npu_runlist_bridge.cpp` *is* the prefill's boot token.)
The two trusted paths agree at every length; the bf16 path agrees only at 256.

A length sweep of bf16-native vs the (also-bf16) CPU-attention variant:

| npt | 256 | 320 | 384 | 512 | 640 | 768 | 896 | 1024 | 1280 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

The passes at 384/512/768/896 are absent and the passes at ≥1024 are coincidences
of a weak argmax gate — the hidden states are not close. Dumping the full
`[token][H]` block per layer (`NPU_DUMP_HIDDEN=1`, now full-block) and comparing
bf16-NPU vs bf16-CPU at npt=512 shows a max abs difference of ~1.4 at layer 0
rising to ~1.9 at layer 1 on hidden values of magnitude ~5 — whole-percent
divergence at token 1, i.e. **not** bf16 rounding (ULP ≈ 0.03 there).

## What survives

- **@256 native bf16 prefill is correct** (boot=1614 on all three paths) and
  measures **~400–413 ms for 256 tokens ≈ 620–668 tok/s**.
- **Decode (~78 tok/s vs FLM on-box 73.58)** is unaffected — it runs on the
  int8/runlist path, which is the byte-exact one.
- The generator finding stands: `qwen3_npu_sequence::gen_mha_engine_seq` is
  exported and callable, and `~/npu-build/mha/gen_attn_chunk 0 1024 …` reproduces
  the committed `engine/npu/xclbins/attn_mha_1024_nh16.elf` **byte-identically**
  (sha256 `6ece6c33…`). But that ELF is ~1500× slower than the embedded captured
  ELF (225241 ms vs 147 ms attention for a 28-layer npt=1024 run), so it is now
  **opt-in** behind `NPU_ATTN_ELF_1024_USE`.
- The in-tree `attn_mha_256_nh16.elf` (26 928 B) is **FLM's captured runtime
  kernel**, not generator output — `gen_mha_engine_seq(0,256)` produces 95 936 B
  and a different hash.

## Changes made in this session

- `npu_engine_universal.cpp`: `NPU_PREFILL_MAX > 256` now **warns and caps to
  256** instead of silently producing a wrong token. @1024 prints
  `bf16 prefill: npt 1024 -> 256 (cap)` and returns boot=1614.
- `npu_engine_bf16_mm.h`: the long-context ELF is searched in
  `$NPU_ATTN_ELF_1024`, `<xclbin_dir>`, `$NPU_XCLBIN_DIR`, and
  `engine/npu/xclbins/` (it was previously unfindable), is reported when missing,
  and is used **only** with `NPU_ATTN_ELF_1024_USE`.
- `npu_engine_universal.cpp`: `NPU_DUMP_HIDDEN` now writes the whole
  `[token][H]` block per layer, not just token 0.

## To actually reach @1k

The 256-row ceiling is the whole gap; there is no throughput problem at 256.
Required work:

1. Generalize the bf16 layer body from two hardcoded 128-row batches to
   `for (blk = 0; blk < npt; blk += 128)` with an A/C row offset. `ensure_a()`
   currently takes rows only from the base of `A`, so either add a row-offset
   parameter to `gemm_launch`/`ensure_a` or pass `A + blk*K` (its cache-key check
   makes that correct but re-stages every block).
2. Run the attention ELF **once per 256-query chunk**, accumulating KV, the way
   FLM's own runtime does — not once for the whole prompt.
3. Re-gate against `NPU_RUNLIST=1` / `NPU_FLM_PREFILL=1` boot tokens, **not**
   against another bf16 variant.
