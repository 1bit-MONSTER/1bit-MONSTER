# WS-03 — Native Ternary AIE Microkernel

**Status:** 🔲 not started — design exists in `docs/research/npu-ternary-roadmap.md`
**Papers:** 2603.27462 (RSR-core), 2603.05168 (Sparse-BitNet), 2604.20913 (FairyFuse, cross-ref), 2607.13511 (ExTernD, cross-ref)
**Owner:** npu

## Goal

Land the true TQ2-on-AIE microkernel: raw 2-bit ternary weights on AIE with LUT-based scalar unpack + ping-pong L1 buffers. The 4× DDR-traffic win for batch=1 decode of 8B+ models.

## Theory

Our own roadmap proves the design: AIE2 has no sub-byte arithmetic, so the trick is `load_v` 128 ternary codes → `uint32_t LUT[256]` unpack (one load + extract per 4 codes) → `mac_8x8_8x8T` on INT8. At batch=1, decode of 8B+ models is DDR-bound at ~200 GB/s shared across 32 cores — 4× less traffic ≈ 4× faster decode. RSR-core (2603.27462) offers a provable alternative accumulation strategy; Sparse-BitNet (2603.05168) shows 1.58-bit weights tolerate N:M sparsity naturally — another ~2× on top.

## Tasks

### P0 (do now)
- [ ] Finish `mm_ternary_tq2.cc`: replace INT8-only body with LUT unpack + ping-pong buffers; build; verify `ternary_tq2_gemv` on hardware (`n1_core_tq2_placed.py` MLIR design is already valid)

### P1 (next)
- [ ] 2:4 structured sparsity on ternary weights (Sparse-BitNet); measure ppl + speed on Bonsai 1.7B/4B

### P2 (if the bet pays off)
- [ ] RSR redundant-segment accumulation vs LUT unpack, compared on the chess microkernel

## Validation

- Bonsai-4B batch=1 decode ≥ 2× bridge path
- DDR bytes/token measured; ppl vs FP16 before/after sparsity
