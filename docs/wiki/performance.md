# Performance & Benchmarks

> **This is the canonical benchmark document.** Update this page first when benchmark numbers change.

> See [Supported Models](models.md) for per-model performance data.

> ⚠️ **Historical claim — not current.** The early "38 KB binary" figure (from the
> one-engine-every-model writeup) is **historical** and no longer reflects the binary
> today. Treat it as a record of the early engine, not as current data.

**Single source of truth for 1bit.MONSTER performance claims.** Every number here is
pulled directly from [`site/benchmarks.json`](../../site/benchmarks.json)
(`"_authoritative": true`). `README.md` and `site/index.html` link here instead of
restating tables — if you change a number, change it in `benchmarks.json` first, then
regenerate this page and `site/numbers.json`/`site/badge_*.json` from it. Do not hand-edit
numbers into more than one file again — that's what caused this page to drift roughly two
weeks out of date the last time it was hand-maintained (see git history).

**Verified on-device — AMD Ryzen AI Max+ 395 (Strix Halo)**

| Component | Spec |
|-----------|------|
| NPU | XDNA 2, 32 AIE2P tiles, 51 TOPS INT8 (measured via `xrt-smi validate`) |
| GPU | Radeon 8060S (gfx1151), 32 CUs, HIP + Vulkan |
| CPU | Zen 5, 16C/32T |
| RAM | 128 GB unified |

---

## Kernel-Level Microbenchmarks (synthetic 28-layer weight buffer)

> ⚠️ These measure single-GEMM-kernel throughput, isolated and correctness-verified
> bit-exact against a CPU reference. They exclude KV-cache attention, softmax, RoPE,
> non-GEMM FFN ops, sampler, tokenizer, and host↔device transfers — **not** an
> end-to-end decode number. See the End-to-End table below and
> [issue #235](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/235).

| Kernel | Value | Backend | Status |
|--------|:-----:|---------|--------|
| Q1 GEMV (fused) | **433 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| Fused TQ2 (QKV+GU) | **420 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| BitNet TQ2_0 (GGML native) | **420 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| Q1_0 binary | **380 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| TQ2 GEMV | **367 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| GPU ternary (Vulkan) | **318 tok/s** | Vulkan ZINC | ✅ validated |
| BitNet TQ1_0 (base-3 LUT) | **202 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| Prefill INT8 WMMA (I8-APRE) | **43.2 TFLOPS** | INT8 WMMA | ✅ re-measured 2026-08-01 (was 39.4) |
| IQ1_S dequant+GEMV | **45 tok/s** | ROCm HIP | ✅ validated — IQ1_M dequant also bit-exact vs llama.cpp reference (`Testing/iq1_selfcheck.cpp`); 50/56-byte block sizes fixed in reader |
| NPU INT8 GEMM | **0/10000 errors (22/22 shapes)** | XDNA 2 via Peano | ✅ verified 2026-07-28 — npu_engine_universal, 4 native ops (QKV/O/GU/D). **2026-08-05: multi-row generator (v27, 4 core rows / 32 cores) — 5.6× kernel-level** (QKV 675, O 757, GU 646, D 751 GOP/s). Chess toolchain deprecated. |

**NPU raw hardware validation** (`xrt-smi validate`, 2026-07-25): 51 TOPS INT8 GEMM,
50µs avg latency, 74,735–75,404 op/s — confirms the NPU/driver/firmware stack is healthy.
This is a device-level number, not a model-inference tok/s figure.

---

## End-to-End Inference (real model, real prompts)

**GGML-Vulkan (llama.cpp, Radeon 8060S, measured 2026-08-01):**

| Model | Value | Backend | Notes |
|-------|:-----:|---------|-------|
| SmolLM2-135M Q4_K_M | **662 tok/s** | GGML-Vulkan | Peak end-to-end decode |
| SmolLM2-360M Q4_K_M | **389 tok/s** | GGML-Vulkan | |
| SmolLM2-1.7B Q4_K_M | **167 tok/s** | GGML-Vulkan | |
| Qwen3-0.6B (native NPU engine) | **2.3 tok/s (435 ms/tok); 230-255 ms/tok** | XDNA 2 (M=32 open kernels, BS=1) | **2026-08-15**: the old "7.4 tok/s @ -B 8" was an INVALID fake batch (issue #111 — top-K candidates as sequential tokens, non-causal). True batch (BS=8, per-sequence KV + causal attention) measures 235-237 ms/tok; the M=32 open kernels are ~11% faster than the FLM M=128 baseline (255-262). Prefill 475 ms/9 tok. See `site/benchmarks.json` |
| Qwen3-0.6B Q4_K_M | **373 tok/s** | GGML-Vulkan | |
| Qwen2.5-VL-3B Q4_K_M | **100 tok/s** | GGML-Vulkan | |
| Qwen3.5-4B Q4_K_M | **65 tok/s** | GGML-Vulkan | |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M | **44 tok/s** | GGML-Vulkan | |

**Native engine (HIP / ZINC / NPU / CPU):**

| Model | Value | Backend | Notes |
|-------|:-----:|---------|-------|
| Qwen3.6-35B-A3B Q4_K_M | **75.65 tok/s** | llama.cpp Vulkan (RADV) | Measured 2026-08-01: tg64=75.65, tg128@8k ctx=75.95, pp512=1105.71 tok/s. 21.2 GB Q4_K_M — see `site/benchmarks.json` |
| Qwen3.6-35B-A3B (FLM) | **11.66 tok/s** | NPU XDNA 2 (FastFlowLM v0.9.46) | Measured: decode 11.66@1k → 8.82@32k; prefill 98.05@1k → 239.79@32k tok/s. 8 iters/ctx — see `site/benchmarks.json` |
| BlackMamba 1.5B | **79.4 tok/s** | Mamba1 HIP (Strix Halo) | Full decode, alternating SSM/MoE dispatch. Re-validated 2026-07-26 after `__shfl_xor_sync` kernel fixes. |
| llama.cpp ROCm (PrismML, third-party) | **229 tok/s** | Same hardware | Comparison point, not our engine. See [issue #235](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/235). |
| BlackMamba 2.8B | **46.0 tok/s** | Mamba1 HIP (Strix Halo) | Full decode. Re-validated 2026-07-26. Reachable today only via the server's internal benchmark thread — `POST /v1/completions` hangs, see [issue #922](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/922). |
| zaya_server (Qwen 27B Q4_K) | **30 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| ZR1-1.5B (Zyphra) | **26 tok/s** | Vulkan ZINC | Reasoning-tuned dense transformer, Qwen2 arch |
| zaya_server (Qwen 35B MoE Q4_K) | **20 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| CPU (generic backend), ZAYA1-8B-shaped | **2.5 tok/s** | AVX-512 CPU, portable path | Real `forward()`+`generate()` loop, not a synthetic kernel. Steady-state 5-token average; single-pass first-token latency was 4.37 tok/s. |

> ⚠️ **TQ2/TQ1 speed rows are kernel-level compute numbers for ternary-NATIVE
> models (BitNet/Bonsai) — they say nothing about dense-model quality.**
> Per the [1BP format policy](models.md#1bp-format-policy-2026-07-31-verdict-ppl-measured),
> TQ2-quantizing a dense model destroys it (ppl 2.6e8 vs Q4NX 62); dense
> models run Q4NX. These rows measure ternary *throughput*, not quality.

---

## DDR Bandwidth Savings — Binary/Ternary Formats

| Format | Bytes per K=64 col | vs INT8 |
|--------|:------------------:|:-------:|
| INT8 (baseline) | 64 | 1× |
| **TQ2** (2-bit) | **16** | **4×** |
| **TQ1** (1.58-bit) | **13** | **4.9× (best)** |
| **Q1_0** (1-bit) | **18** | **3.6×** (block overhead) |

---

## Engine Evolution (July 2026)

| Date | Engine | Decode | Breakthrough |
|------|--------|:------:|--------------|
| Jul 1 | i8 swap | 244 ms/tok | K-interleaving fixed |
| Jul 2 | v9/v12, M=32 batch | 10 ms/tok | M=32 + OpenMP attention |
| Jul 6 | Fused layer | 3.4 ms/tok | One xclbin/transformer layer |
| Jul 24 | Binary/ternary GPU kernels | 1–2 µs | Q1_0, BitNet, IQ GPU kernels verified exact |
| Jul 24 | NPU ternary LUT decode | 3 kernels | TQ2/TQ1/Q1_0 on-tile decode via Chess |
| Jul 25 | NPU HW re-validated + zero-copy fusion fix | — | `xrt-smi validate` clean; fixed buffer-overflow segfault in fusion pipeline test |
| Jul 28 | npu_engine_universal INT8 GEMM + Peano xclbins | **22/22 shapes, 0 errors** | All 4 ops (QKV/O/GU/D) across 5 models verified on real hardware. NPU attention fixed (xrt::ext::bo overload bug). Chess deprecated. |
| Aug 5 | Multi-core GEMM (v27) | 435 ms/tok | All 4 AIE core rows (32 cores) instead of 1; 5.6× kernel-level, 2.4× e2e decode. NPU attention default → opt-in (was 55× slower than CPU). |
| Aug 15 | FLM-free + M=32 kernels | 230-255 ms/tok | Open toolchain (v27 + peano-clang microkernel) builds byte-identical instruction streams; M=32 kernels beat FLM 11% (universal) / 15% (fused); 35B-A3B BS=8 + expert dequant cache 2× (2900→1450). True batch decode replaces the invalid fake-batch "-B 8". |
| Jul 26 | Mamba1 HIP re-measure | 79.4 / 46.0 tok/s | Fixed `__shfl_xor_sync` correctness bug, numbers went *up* |

---

*All kernel-level numbers verified bit-exact on real Strix Halo hardware (gfx1151), median
of 3 runs. Status legend: ✅ validated · ⚙️ optimized (kernel runs at this speed, engine
integration in progress).*

---

## Related

- [Supported Models](models.md) — per-model architecture, backend, and performance data
- [`benchmarks/README.md`](../../benchmarks/README.md) — how to run benchmarks locally
- [`site/benchmarks.json`](../../site/benchmarks.json) — machine-readable authoritative source for all numbers on this page

---

## On-Box Parity (re-scope, 2026-09-12)

> The goal bar was re-scoped from FLM's **published** Kraken-Point tables to
> **on-box FLM** (same hardware/weights/prompt) on 2026-09-12. The published bar
> remains unmet by the native backend (multi-week fused-kernel work) — the
> FLM-Orchestration section below keeps the published-table comparison.

Native backend vs FLM on-box, Qwen3-0.6B (byte-identical output, A/B verified):

| Metric | native | FLM on-box | verdict |
|---|---:|---:|---|
| Prefill @256 | **655 tok/s** (391 ms) | ~430 tok/s (570–610 ms) | **native +52 %** ✅ |
| Decode @1k | **79 tok/s** (12.7 ms/tok) | 73.58 tok/s (13.6 ms/tok) | **native +7 %** ✅ |
| Decode @256 | **91 tok/s** (11.0 ms/tok) | — | — |
| TTFT @256 | **391 ms** (first chunk) | ~570–610 ms | **native faster** ✅ |
| TTFT @1k | ~14 s (full per-token prefill) | 0.70 s (first chunk) | metric/chunked-prefill gap ⚠ |

- **Prefill @256**: the native bf16 path (`NPU_RUNLIST=0 NPU_PREFILL_BF16=1`) runs
  dequant/mm/attn xclbins; it beats FLM's own `qwen3_npu::prefill` on-box.
- **Decode**: the whole-layer single-launch `xrt::runlist` path plus the
  build-overlap optimization (`f37fb0489`, double-buffered runlist) moved decode
  from 62 → 79 tok/s @1k, now ahead of FLM on-box. (This is orchestration of FLM's
  captured layer ELFs, not native kernels.)
- **TTFT @1k** is a metric-semantics + chunked-prefill gap (FLM streams the first
  chunk early; the native whole-layer path runs per-token prefill). Closing it needs
  chunked prefill, blocked on the >256-token attention ELF.

Source of truth: `site/benchmarks.json` (`flm_parity.on_box`) +
`benchmarks/RESULTS-on-box-parity-2026-09-12.md`.

---

## FLM-Orchestration Parity (decode / prefill @1k, 2026-09-11)

> ⚠️ **Not the objective.** This records the native engine's FLM-*orchestration* path
> (`NPU_FLM_PREFILL`/`NPU_FLM_DECODE`), which invokes **FLM's own NPU libraries**
> (v0.9.46 for MoE + remaining families, v1.0.4 for dense Qwen3) — it is FLM re-run on
> this box, not 1bit-MONSTER's native int8/bf16 backend, so it cannot "meet-or-beat" FLM
> by construction. The native backend is 6–21× behind on prefill/TTFT
> (see `RESULTS-qwen3-dense-parity-2026-09-10.md`). Reference bar = FLM's **published**
> Kraken-Point tables. Decode trails the published bar for nearly every model (Strix-Halo
> vs Kraken-Point cross-hardware gap; FLM's own on-box numbers trail identically).
> Source of truth: `site/benchmarks.json` (`flm_parity`).

| Model | decode (ours/pub) | prefill (ours/pub) | TTFT @1k (ours/pub) |
|---|---:|---:|---:|
| Qwen3-0.6B | 74 / 66.5 ✓ | 1370 / 1494 | 0.73 / 0.67 s |
| Qwen3-1.7B | 38 / 40.2 | 971 / 956 ✓ | 1.03 / 1.05 s |
| Qwen3-4B | 18 / 19.6 | 510 / 509 ✓ | 1.96 / 1.96 s |
| Qwen3-8B | 11 / 11.9 | 370 / 357 ✓ | 2.70 / 2.80 s |
| Qwen3.6-35B-A3B | 13.5 / 17.48 | 125.0 / 102.45 ✓ | 8.00 / 9.76 s |
| Llama-3.2-1B | 56 / 64.5 | 1515 / 1686 | 0.66 / 0.59 s |
| Gemma4-E2B | 21 / 22.6 | 633 / 721 | 1.58 / 1.39 s |
| Gemma4-E4B | 12 / 12.6 | 435 / 441 | 2.30 / 2.27 s |
| Phi4-mini | 20 / 21.8 | 637 / 643 | 1.57 / 1.56 s |
| Nanbeige4.1-3B | 21 / 23.5 | 565 / 612 | 1.77 / 1.63 s |
| LFM2-1.2B | 62 / 62 ✓ | 1587 / 1537 ✓ | 0.63 / 0.65 s |
