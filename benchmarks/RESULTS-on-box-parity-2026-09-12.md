# RESULTS — on-box parity (re-scope), 2026-09-12

Goal `mttxt22c-a6rv75`, re-scoped per user directive (2026-09-12): the reference
bar changes from FLM's **published** Kraken-Point tables to **on-box FLM** (same
hardware, same weights, same prompt). The published bar is documented as
structurally out of reach for the native per-op path (see below).

Two distinct native paths are measured, and they are reported separately so the
"native backend" vs "orchestration" distinction is never conflated:

| path | what it runs | native-kernel? |
|---|---|---|
| **bf16 prefill** (`NPU_RUNLIST=0 NPU_PREFILL_BF16=1`) | dequant.xclbin + mm.xclbin + attn.xclbin GEMM/attention, host f32 norm/RoPE/SiLU | yes (per-op native GEMMs) |
| **whole-layer decode** (`NPU_RUNLIST=1`, default) | single-launch `xrt::runlist` over FLM's captured per-ctx layer ELFs | no (orchestration of FLM's captured kernels) |

## 1. Native bf16 prefill @256 — native BEATS FLM on-box

Qwen3-0.6B, same 256-token prompt, same output tokens (byte-exact parity):

| path | prefill time | throughput |
|---|---|---|
| native bf16 prefill | **391 ms** | **655 tok/s** |
| FLM `qwen3_npu::prefill` (on-box) | ~570–610 ms | ~420–450 tok/s |

**Native is ~1.5× faster than FLM on-box at 256 tokens, byte-identical output.**
The native bf16 path caps at 256 tokens (`attn_mha_256_nh16.elf`); >256 needs
chunked prefill + position-shifted attention ELFs (blocked, see §4).

## 2. Decode @1k — native (whole-layer) BEATS FLM on-box after build-overlap

Qwen3-0.6B, 1024-token prompt (`/tmp/ids1024v.txt`), 32-token greedy decode
(byte-identical tokens). A/B re-measured 2026-09-12, same prompt:

| path | ms/tok | tok/s |
|---|---|---|
| native whole-layer decode, PRE-overlap (`3cd4896b0`) | 14.7 | 68 |
| native whole-layer decode, POST-overlap (`f37fb0489`) | **12.6** | **79** |
| FLM on-box `flm bench` decode | 13.6 | **73.58** |

The **decode-overlap optimization** (double-buffered `xrt::runlist`, build the next
context's runlist while the current one executes) saves ~2.1 ms/token at @1k
(68 → 79 tok/s) and ~1.6 ms/token at @256 (79 → 92 tok/s). Post-overlap native
decode crosses the FLM on-box bar: **79 vs 73.58 tok/s (+7 %)**. Byte-exact token
parity preserved (A/B verified).

## 3. TTFT — honest delta

- **@256**: native bf16 prefill TTFT = 391 ms (the whole prefill *is* the first
  chunk), vs FLM on-box @256 ≈ 570–610 ms → **native beats FLM**.
- **@1k**: the whole-layer path runs per-token prefill, so TTFT = full prompt
  (≈14 s @1k) vs FLM's first-chunk 0.70 s. This is a **metric-semantics +
  chunked-prefill** gap, not a kernel-speed gap: FLM's own 0.70 s TTFT is smaller
  than its own 1.38 s full-prompt time (it streams the first chunk early). Closing
  it needs chunked prefill (blocked, §4).

## 4. What is structurally NOT met (honest)

- **Published prefill @1k (1494 tok/s for 0.6B)**: the native per-op bf16 path is
  launch-overhead-bound (~9 launches/layer × 28) and capped at 256 tokens.
  Even chunked 4 × 256 ≈ 4 × 391 ms ≈ 1.56 s ≈ 655 tok/s @1k — far below 1494.
  Requires fused-layer NPU offload (multi-week), not a bounded change.
- **4B/8B nh32 attention** (`attn_mha_256_nh32.elf`): mis-captured (corr 0.04 vs
  correct attention); needs ELF re-capture — multi-day (see
  `RESULTS-task-n2-qwen3-4b8b-native-prefill-2026-09-11.md`).
- **Other families (Llama/Gemma4/Phi4/Nanbeige/LFM2) and MoE**: not on the native
  bf16 path; previously only via FLM's own libs (the audit's objection).

## 5. Verdict

Under the re-scoped **on-box** bar: native prefill @256 beats FLM (+55 %) and
native decode @1k beats FLM (+7 %), both byte-identical. The *published*
Kraken-Point bar remains unmet and is documented as requiring multi-week
fused-kernel NPU offload.
