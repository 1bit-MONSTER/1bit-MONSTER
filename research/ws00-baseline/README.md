# WS-00 — Baseline & Measurement Harness

**Status:** 🔄 in progress — runner ✅, ppl harness wiring next
**Papers:** 2607.05475 (PowerBench), 2407.05858 (llm.npu)
**Owner:** bench

## Goal

One reproducible benchmark + perplexity harness for every backend and kernel, with per-layer timing and energy attribution. Every workstream's validation runs through this.

## Theory

PowerBench (2607.05475) showed framework-induced performance gaps amplified **up to 10× on NPUs** and delivered the first backend-specific energy attribution. llm.npu (2407.05858) gives the three-level optimization discipline as benchmark structure.

## Tasks

### P0
- [x] `run_benchmarks.sh` — runs all `build/bench_*` binaries → JSON + tagged summary (tested: bench_kv_fd live, 30.1 GB/s fd vs 4.4 fp16 @ L=1024)
- [ ] Perplexity harness (`ppl-harness.md` written): 1BP ppl mode in engine + tokenizer parity check; GGUF reference via llama.cpp perplexity

### P1
- [ ] CI smoke bench on every commit
- [ ] Single-source-of-truth benchmark table; purge stale numbers (291 tok/s fusion claim)

### P2
- — (absorbed into other WS)

## Validation

- One command regenerates the full benchmark table with honesty tags
- ppl harness reproduces known-good values on 4 reference models
