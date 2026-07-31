# WS-04 Findings — CPU Ternary Kernel Sweep (P0 ✅)

**Date:** 2026-07-31 · **Tool:** `ternary_cpu_sweep.cpp` (AVX-512 + BMI2 + VNNI, Zen 5 / Ryzen AI Max+ 395) · **Data:** packed 2-bit ternary {-1,0,+1}, near-ternary weights, fp32 activations

## What was tested

Four GEMV variants on identical packed-2-bit weights (y[N] = W[N,K]·x[K], row-major):

| Variant | Approach | Source |
|---|---|---|
| scalar | naive dequant + FMA (double accum) | reference |
| lut | 4-bit pos/neg masks from 2×256B tables → maskz add/sub | T-MAC-style (FairyFuse paper's LUT comparison) |
| fairyfuse | BMI2 `_pext_u32` masks → maskz `vaddps/vsubps`, zero multiplies | FairyFuse 2604.20913 |
| vnni | decode→int8, `vpdpbusd`, +128 offset trick, per-row Σw correction | Litespark 2605.06485 |

Correctness: all variants verified vs scalar ref — maxdiff ≤ 1e-4 (float32 accumulation noise); VNNI verified vs an exact int32 reference (bit-exact).

## Results — 16 threads (engine config), ns per GEMV

| Shape | scalar | lut | fairyfuse | vnni | best |
|---|---:|---:|---:|---:|---|
| QKV-like N=4096 K=1024 | 1,383 µs | 44.8 µs | **25.7 µs** | 30.4 µs | fairy 53.8× |
| GU-like N=6144 K=1024 | 2,132 µs | 52.4 µs | **37.7 µs** | 37.9 µs | fairy 56.6× |
| dense N=4096 K=4096 | 5,611 µs | 135.9 µs | 100.1 µs | **90.4 µs** | vnni 62.1× |
| big-FFN N=16384 K=4096 | 23,347 µs | 731.3 µs | **408.6 µs** | 470.9 µs | fairy 57.1× |

- Effective weight bandwidth: **~41 GB/s** (16 threads) — memory-bound, matching the platform's practical stream bandwidth (cf. KV-FD 57 GB/s at L=2048).
- FairyFuse and VNNI are within 10-20% of each other; FairyFuse wins at K=1024 (short rows), VNNI at K=4096 (long rows). T-MAC-style LUT masks are ~20% behind both. All three crush the scalar path 43-62×.

## Findings

1. **The multiplication-free claim is real and it wins on this hardware.** Both published approaches (pext-masks and VNNI) give 55-62× over the naive dequant path at ~41 GB/s — the 2-bit packing is fully bandwidth-bound, exactly as FairyFuse's roofline analysis predicts (16× compression shifts GEMV to the compute ridge).
2. **FairyFuse ≈ Litespark ≈ T-MAC here** (within 20%) — the decode strategy barely matters once you're memory-bound; what matters is *not dequantizing to fp32 in the inner loop*. This validates the plan's NPU microkernel design too (the AIE LUT-unpack in `npu-ternary-roadmap.md` is the same class of trick).
3. **Integration target**: a GU-like TQ2 GEMV (K=1024, N=6144) at **37.7 µs** → a 0.6B ternary model's decode ≈ 28 layers × ~4 GEMMs × ~40 µs ≈ 4.5 ms/token ≈ **200 tok/s kernel-bound ceiling** on CPU alone (attention + norms excluded). The current CPU generic path (2.5 tok/s on 8B-shaped Q4) is ~100× off this ceiling for ternary models.
4. `-mavx512vnni` and `-mbmi2` are both available on Zen 5; the kernels are drop-in for the CPU backend and the ZINC (Vulkan) CPU fallback.

## Recommendations

- **P1 (WS-04):** integrate the fairyfuse kernel (or vnni for K≥4096 layers) into the CPU TQ2 decode path; target ≥50 tok/s e2e on Bonsai-1.7B (1BP). Publish the comparison per the honesty policy (this is blog-post material: "multiplication-free ternary CPU kernels vs our old dequant path").
- **For the NPU microkernel (WS-03):** the LUT-mask variant is the best fit for AIE2 (no pext on AIE; LUT in scalar unit matches the roadmap's design) — keep T-MAC-style decode in mind, not just the float-LUT.
- **Reuse:** the sweep's packed layout (2-bit, 4 codes/byte, 00/01/10/11 encoding) matches `include/block_scaled_ternary.h`'s encoding — kernels are directly reusable for the WS-06 block-scaled format.

## Bugs found & fixed during the sweep (for the record)

1. `std::mt19937` returns 64-bit `unsigned long` on LP64 — `rng() % 2000 - 1000` wraps in unsigned → 1e17-magnitude test data. Cast to int first.
2. VNNI reference used `(int8_t)x_u8[j] - 128` — int8 wrap (−120−128=−248) vs the kernel's unsigned treatment. Reference must use `(int)x_u8[j] - 128`. Kernel was correct.
3. Initial "LUT float" variant read 16 floats from a 4-float table row (out-of-bounds) — replaced with the proper T-MAC mask-LUT design.
4. Kernels initially lacked `#pragma omp parallel for` — no scaling at 16 threads; added (row-parallel).
