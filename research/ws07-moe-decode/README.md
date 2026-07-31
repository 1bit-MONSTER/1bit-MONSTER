# WS-07 — MoE Decode & Speculative Decoding

**Status:** 🔲 not started — spec engine currently 0.9 tok/s, 0% draft acceptance (issue #938)
**Papers:** 2607.24434 (DraftExpert), 2607.16184 (PagedWeight), 2606.10493 (CPU-GPU MoE SLO), 2607.25852 (AngelSpec), 2607.27269 (MLA-Draft-Functional), 2606.24031 (35B-on-6GB)
**Owner:** moe/spec

## Goal

MoE decode that's actually fast (BlackMamba, Qwen-35B-A3B, Kimi K3) and a spec-decode path with real acceptance.

## Theory

DraftExpert (2607.24434) is our exact deployment pattern: **experts staged on demand** (CPU→GPU, Flash→NPU), self-speculative decoding where draft-expert-set size trades acceptance against expert-loading cost, and block verification activates the union of target experts. 0% acceptance today is likely a draft-target agreement failure — MLA-Draft-Functional (2607.27269) shows exactly how MHA/GQA→MLA conversion kills agreement. CPU-GPU MoE SLO (2606.10493) quantifies the 4 local-MoE gaps and fixes them with stream-loading prefill (1,200 tok/s). PagedWeight (2607.16184) balances expert precision vs KV headroom at runtime — direct fit for the 63 GB iGPU.

## Tasks

### P0 (do now)
- [ ] Instrument draft/target agreement on the existing spec engine (BlackMamba 2.8B) — explain 0% acceptance before changing anything

### P1 (next)
- [ ] DraftExpert-style expert-aware draft sizing
- [ ] PagedWeight-style runtime expert quantization (Zaya/Qwen MoE on 63 GB iGPU)

### P2 (if the bet pays off)
- [ ] AngelSpec-style drafter selection (MTP vs block-parallel) per workload
- [ ] Stream-loading prefill for >12k-token prompts (TTFT < 30 s)

## Validation

- Acceptance > 30%; BlackMamba 2.8B e2e > 46.4 tok/s; Qwen-35B-A3B Q4_K > 20 tok/s; TTFT < 30 s at 32k prefill
