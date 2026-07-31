# WS-09 — Router Unification

**Status:** 🔲 not started — P0.2 (decision) is the entry gate
**Papers:** 2607.25498 (DOPS), 2607.27269 (cross-ref), 2607.16488 (AutoScale-NPU, P2)
**Owner:** router

## Goal

One router with live throughput-ranking + health-check fallback — the missing behavior from the stack audit — plus DOPS-style weight-layout awareness.

## Theory

DOPS (2607.25498) argues prefill/decode disaggregation alone is insufficient: end-to-end latency depends on workload shape, device contention, and persistent weight layout. Their Bifocal scheduler (dynamic operator-to-device placement) + Weight Layout Arbiter (which device holds which weight blocks) is the research version of our vision — and Strix Halo's unified memory makes layout arbitration a real lever. Currently three routers exist (`cascade`, `tools/token_router.cpp`, `unified-router.py`) with no live ranking; the audit's "4-tier priority" doc describes a router that was never built.

## Tasks

### P0 (do now)
- [ ] Land P0.2: one router (StrategyConfig enum already exists in `token-router`'s Rust: Passthrough, Cascade, SpecDecode, ContentRouter, Performance); retire the other two

### P1 (next)
- [ ] Live throughput probe + health-check fallback (rank-by-throughput, the missing behavior)
- [ ] DOPS-style weight-layout awareness (which weights live where for MoE staging — cross-ref WS-07)

### P2 (if the bet pays off)
- [ ] Workload-shape-aware dispatch (prefill vs decode vs batch)

## Validation

- Router overhead < 1% of token latency
- Failover behavior on induced backend failure
- MoE staging throughput recorded
