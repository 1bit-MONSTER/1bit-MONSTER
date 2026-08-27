# Sherry — 1.25-bit sparse ternary format (3:4 N:M)

Clean-room decode and implementation of the Sherry 1.25-bpw weight format
(published paper: arXiv 2601.07892), integrated across every 1bit.MONSTER
backend. Single MIT license — provenance at the bottom.

## Format

- **Ternary, 3:4 N:M sparse**: every 4 consecutive weights along K form one
  group with exactly one forced-zero position; the other three are ±1.
- **5 bits per group (1.25 bpw)**: `[zero_pos:2 | signs:3]` — 4 zero positions
  × 2³ sign combos = 32 states, exactly saturating a 5-bit index. `signs` are
  MSB→LSB in positional order, skipping `zero_pos`, so `signs[i]` is the sign
  of the i-th surviving (non-zero) lane.
- **Row layout**: `K_in / 4` groups × 5 bits = `K_in × 5 / 32` bytes per row
  (K_in must be a multiple of 32 for byte alignment). Weight buffer total:
  `N_out × K_in × 5 / 32` bytes.
- **Compute model**: (1.25 bpw weights) × (fp16 or int8 acts) → fp16 out,
  pure signed-sum. The bare reference kernel carries no per-row scale (callers
  apply scales at their own level); the service-decode variant carries
  per-row fp32 scales and a 32-entry LUT unpack.

## Implementations in this repo

| Backend | File(s) | Notes |
|---|---|---|
| GPU (HIP) — spec reference | `src/sherry_gemv.hip` | Minimal 1:1 spec-accurate kernel, fp16 acts, no scales/LUT; differential-tested vs scalar ref |
| GPU (HIP) — service decode | `kernels/ternary_gemv_sherry.hip` | halo-1bit v3 packing, int8 acts + per-row fp32 scales, `__constant__` 32-entry LUT, fp16 out |
| GPU (HIP) — scalar ref | `src/sherry_gemv_scalar_ref.hip` | Bit-level reference for the differential test |
| C API | `include/rocm_cpp/sherry.h` | `sherry_ternary_gemv_launch` (librocm_cpp.so) |
| NPU (AIE2) | `engine/npu/kernel/mm_ternary_stq_aie2.cc` | STQ/Sherry 3:4 on AIE-ML (VEK280, aie2 target); 10 bytes per 64-weight column vs 16 for TQ2 (−1.6× DDR traffic) |
| NPU x86sim | `engine/npu/kernel/stq_aiesim/` | Single-tile ADF x86sim harness vs golden |
| Vulkan | `engine/fusion/shaders/vulkan/dmmv_stq1_0.spv` | Compiled STQ shader |
| Tooling | `tools/gguf_to_h1b.cpp`, `tools/convert_gguf_to_h1b.cpp`, `tools/convert_zaya_to_h1b.cpp` | Converters to the Sherry `.h1b` model container (see also requantizer notes in `sherry.h`) |
| Tests | `tests/test_sherry_gemv.cpp`, `tests/test_sherry_e2e.cpp`, `engine/npu/kernel/test_stq_gemv_ref.cc` | Differential (≤1 bf16 ULP vs scalar ref across 50 seeds) + one-token e2e on a real sherry-v4 `.h1b` |
| Bench | `tools/bench_sherry.cpp`, `benchmarks/sherry-ppl.sh`, `benchmarks/data/sherry-*.json` | Throughput / perplexity numbers |

## Validation

- **Differential test**: `sherry_gemv.hip` vs `sherry_gemv_scalar_ref.hip`,
  required ≤ 1 bf16 ULP across 50 seeds — rejects any packing or lane-order
  drift.
- **End-to-end**: `test_sherry_e2e` loads a real Sherry fp16 `.h1b` model and
  runs a one-token decode.

## Provenance & license

- The Sherry format is published (arXiv 2601.07892). Everything in this repo
  is an original **clean-room implementation** by 1bit.MONSTER contributors —
  no third-party code, so no third-party copyright attaches.
- The Sherry sources were initially under PolyForm Noncommercial 1.0.0
  (forward-only relicense 2026-04-26). On 2026-08-27 the copyright holder
  (bong-water-water-bong) relicensed them to **MIT** and the carve-out files
  (`LICENSE-SHERRY.md`, `SHERRY-FILES.txt`) were removed — the repo is now
  single-license MIT with no carve-outs (commits `a077ee6e` / `4b3755ee`,
  PR #1894). Pre-2026-04-26 commits retain MIT per the old carve-out's
  snapshot clause; provenance remains in git history.
