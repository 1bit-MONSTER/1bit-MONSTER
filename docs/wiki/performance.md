# Performance & Benchmarks

> **This is the canonical benchmark document.** Update this page first when benchmark numbers change.

> See [Supported Models](models.md) for per-model performance data.

> ⚠️ **Historical claim — not current.** The early "38 KB binary" figure (from the
> one-engine-every-model writeup) is **historical** and no longer reflects the binary
> today. Treat it as a record of the early engine, not as current data.

**Single source of truth for 1bit.systems performance claims.** Every number here is
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
> [issue #235](https://github.com/1bit-systems/1bit-systems/issues/235).

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
| llama.cpp ROCm (PrismML, third-party) | **229 tok/s** | Same hardware | Comparison point, not our engine. See [issue #235](https://github.com/1bit-systems/1bit-systems/issues/235). |
| BlackMamba 2.8B | **46.0 tok/s** | Mamba1 HIP (Strix Halo) | Full decode. Re-validated 2026-07-26. Reachable today only via the server's internal benchmark thread — `POST /v1/completions` hangs, see [issue #922](https://github.com/1bit-systems/1bit-systems/issues/922). |
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
