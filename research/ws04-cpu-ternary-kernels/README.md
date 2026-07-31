# WS-04 — CPU Ternary Kernel Sweep

**Status:** 🔲 not started
**Papers:** 2604.20913 (FairyFuse), 2605.06485 (Litespark), 2410.16144 (1-bit AI Infra / bitnet.cpp), 2604.18556 (GSQ, cross-ref)
**Owner:** cpu

## Goal

Benchmark every published ternary-CPU approach against our Q1 GEMV (417 tok/s) and fused TQ2 (415 tok/s) on the same box, same harness; adopt what wins.

## Theory

FairyFuse (2604.20913) is multiplication-free fused ternary inference on commodity CPUs — conditional add/sub/no-op replaces multiply; Litespark (2605.06485) is custom SIMD ternary kernels targeting the 1B+ underused PCs; bitnet.cpp (2410.16144) holds the original 2.37-6.17× x86 claim. All three claim to beat "dequantize-and-FP-multiply" systems. Our 417 tok/s Q1 GEMV is the number to beat.

## Tasks

### P0 (do now)
- [ ] Port FairyFuse + Litespark kernel variants as isolated microbenchmarks (same model, WS-00 harness)

### P1 (next)
- [ ] Publish the comparison (honesty policy → strong blog post); adopt the winner into the CPU backend

### P2 (if the bet pays off)
- [ ] Block-scaled ternary on CPU (`include/block_scaled_ternary.h`: 5 B/16 elems) — the ~0.3-0.5 ppl win over raw TQ2 from per-block scaling

## Validation

- tok/s table on identical hardware with honesty tags
- ppl deltas; binary size stays ~400 KB
