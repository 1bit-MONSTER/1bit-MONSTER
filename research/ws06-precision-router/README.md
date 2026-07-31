# WS-06 — Precision-Profile Router

**Status:** 🔲 not started — design docs exist: `docs/research/hybrid-w4a8-router.md`, `docs/research/block-scaled-ternary-format.md`
**Papers:** 2607.17733 (MXSens), 2606.13054 (TWLA), 2310.10537 (MX), 2509.01229 (LiquidGEMM), 2301.12017 (INT4-for-transformers), + NVFP4-vs-MXFP4 deep-read (2509.23202)
**Owner:** gpu/router

## Goal

Per-layer precision dispatch — ternary / block-ternary / FP8 — chosen by a calibration-generated profile. Expected: ~2× vs full FP8 throughput, ~0.5 ppl better than full ternary.

## Theory

INT4-for-transformers (2301.12017) showed decoder-only W4A4 accuracy loss concentrates in specific layers — middle FFN layers are robust at ternary, early/late attention and lm_head are sensitive. Our block-scaled ternary (16-elem blocks, FP8 E4M3 scale, 2.5 b/elem) is the NVFP4 lesson applied to ternary. MXSens (2607.17733) adds sensitivity-aware mixed precision with MXINT formats (rotation methods don't compose with MX — MR-GPTQ's block-wise rotation + fusion is the fix). LiquidGEMM (2509.01229) is the kernel-design reference: dequant overlap, not the MAC, is the bottleneck.

## Tasks

### P0 (do now)
- [ ] Per-layer precision profile for one model (Bonsai-4B or Qwen3-0.6B) via calibration — extend `tools/mr_gptq_rotate.py`

### P1 (next)
- [ ] Kernel dispatch by profile: `precision_profile` byte per sub-layer in `rcpp_bitnet_model_t`; `zaya_gpu_router.hip` dispatches ternary / block-ternary / INT8-FP8 variants
- [ ] WS-05 correction planes as the "sensitive layer" precision option (cheaper than FP8 for the same error class)

### P2 (if the bet pays off)
- [ ] MXSens-style sensitivity ranking to auto-generate profiles (no calibration search)

## Validation

- ppl and tok/s vs full-ternary and full-FP8 profiles, same model, WS-00 harness
