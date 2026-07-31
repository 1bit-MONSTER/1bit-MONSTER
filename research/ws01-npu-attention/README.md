# WS-01 — NPU Fused Attention

**Status:** 🔲 not started — depends on P0.5 (kill CPU attention stub)
**Papers:** 2607.09385 (STEEL), 2605.01910 (SANTA), 2504.19874 (TurboQuant, cross-ref)
**Owner:** npu

## Goal

Kill the 8.4 ms/layer CPU attention stub (issue #936, 235 ms/token total) → then move attention onto XDNA natively with STEEL's fused-attention structure.

## Theory

STEEL is the **first open-source FlashAttention targeting AMD XDNA NPUs** — sparse-aware fused attention under the explicit data-movement programming model, exactly what FLM's extracted `attn.xclbin` (opcode 6) uses. SANTA (2605.01910) offers a later-stage alternative for KV-bandwidth-bound long context: sample S value rows post-softmax, gather-and-add instead of MAC — unbiased with variance-reduced GPU-friendly variants.

## Tasks

### P0 (do now)
- [ ] GPU flash-decoding attention kernel replacing `attn_stub.cpp` → fused engine targets 40-60 tok/s

### P1 (next)
- [ ] STEEL-style fused attention on XDNA (`attn.xclbin` ABI, sparsity-aware); benchmark vs GPU path

### P2 (if the bet pays off)
- [ ] SANTA sampled-value attention for 32k+ contexts (KV-FD baseline: 57.1 GB/s at L=2048)

## Validation

- Attention < 1.0 ms/layer on NPU path
- Fused engine ≥ 40 tok/s (from 5.0), tagged `validated`
- Long-context (32k) decode tok/s recorded
