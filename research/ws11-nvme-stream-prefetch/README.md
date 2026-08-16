# WS-11 — NVMe→NPU Expert Streaming (the memory hierarchy bet)

**Status:** 🔲 proposal — design sketch, awaiting sign-off
**Papers:** Colibrì (JustVugg/colibri, NVMe/RAM/VRAM as one tier), 2312.11514 (LLM-in-a-flash: windowing + row/column bundling), MoE-Infinity (activation-aware expert cache + prefetch), 2411.01433 (HOBBIT mixed-precision offload), 2607.24434 (DraftExpert — cross-ref WS-07), 2607.16184 (PagedWeight), 2606.10493 (CPU-GPU MoE SLO: stream-loading prefill)
**Owner:** memory/npu
**Cross-refs:** WS-07 (MoE decode + expert staging), WS-09 (router unification — needs WS-11's routing-heat signal)

## Goal

Break the ~8B NPU parameter ceiling: run a 1-bit MoE (30B-total / ~3B-active class) on Strix Halo with **experts streamed from NVMe**, dense/shared weights resident in DRAM, and the hot-expert working set in NPU SRAM. Zero CPU copies of weight payloads — the only moves are NVMe→DRAM (readahead) and DRAM→SRAM (NPU DMA). Target: >20 tok/s decode, expert hit rate > 90%.

## Why this works on Strix Halo (the numbers)

- Dense models re-stream **all** weights every token: 8B × 0.2 B/param (1BP) ≈ 1.6 GB/token → the bandwidth wall that defines the 8B ceiling.
- MoE kills that: top-2 of 8 experts ≈ 1/4 of weights per token. A 30B MoE at 1BP ≈ 3.75 GB total, ~0.5 GB/token active — PCIe 4.0 x4 NVMe (~7 GB/s sequential) clears that at ~14–28 tok/s *before any optimization* (windowing, bundling, heat caching push it up).
- NPU side constraints (from WINDOWS-NPU-RE-NOTES): 64 MB NPU SRAM (≈ 320M 1BP params resident), firmware 64 MB aperture @ 0x4000000 (the DMA window), `reorder_cpy(quant_block_t, …)` = the existing block-level weight-DMA primitive, DPU kernel 0x100 + per-shape microcode = the existing expert-GEMM dispatch mechanism (28 shape variants today).
- **Key NPU-specific fact:** NPU DMA cannot page-fault. The OS page cache does NOT serve DMA — pages must be resident and IOMMU-mapped before the aperture touches them. So demand paging (the CPU/GPU trick) is unavailable; prefetch must be **explicit and ahead-of-need**. This is the whole bet.

## Architecture — three tiers, one scheduler

```
 NVMe (1BP file, mmap MAP_PRIVATE)          T0 — storage
   │  posix_fadvise(WILLNEED) + prefetch thread (touch-ahead in token order)
   ▼
 Pinned DRAM pool (XRT HOST_ONLY SharedBO,  T1 — staging
   NPU-owned, reuse engine/fusion/zero_copy/shared_bo.h)
   │  double-buffered aperture DMA, issued by scheduler
   ▼
 NPU SRAM 64 MB                               T2 — resident set
   = dense/shared layers (permanent) + top-k experts by routing heat
```

The zero-copy rule: the CPU touches T0→T1 (readahead thread — one copy, unavoidable, that's the storage tier); T1→T2 is NPU DMA through the aperture, driven by `reorder_cpy` block descriptors; the payload never passes through a CPU memcpy or a BO round-trip. One pinned SharedBO allocation, `mlock`-stable, reused for the model's lifetime.

## The scheduler (the new code)

Per layer, per token:
1. **Router emits expert ids** (WS-09's job — it already needs to rank; add the heat table here so one component owns both).
2. **Heat table**: per-expert EMA of routing counts (`h_t = α·hit_t + (1−α)·h_{t−1}`). SRAM resident set = dense/shared weights + top experts by heat (PagedWeight-style: also decide *precision* per expert — 1BP for cold, 2–4 bit for hot, if SRAM headroom allows).
3. **Prefetch queue**: scheduler issues `posix_fadvise` + readahead for the next-k experts' NVMe blocks (LLM-in-a-flash bundling: fetch whole expert row-blocks, not pages), then queues aperture DMA into the free half of a double-buffer while the DPU GEMMs on the busy half. `move_weights` (already in the FLM bridge instruction pipeline) is the submit primitive.
4. **Dispatch**: expert GEMM = DPU kernel 0x100 with per-shape params (the 28-shape mechanism already decoded — expert id → shape variant, no new kernel work).
5. **Self-speculative staging** (WS-07 cross-ref): draft-expert set = the SRAM-resident top-k; target experts verified in blocks — draft agreement directly pays for streaming cost.

## Known honest costs

- SNU measurement (2508.06978): SSD-offloaded MoE runs up to ~12.5× energy/token vs HBM. Works, but it's a throughput-vs-energy trade — Strix Halo's unified memory means our DRAM tier absorbs most hot traffic, NVMe only sees cold experts; measure real duty cycle before optimizing further.
- Cold-start random reads are the worst case (~11 GB/token in Colibrì's numbers for a 744B model; ours is ~0.5 GB/token at 30B — 20× smaller, fine).
- SSD wear is a non-issue (read-only), swap traffic is the enemy — keep the DRAM budget strictly below RAM (Colibrì lesson).

## Tasks

### P0 (do now)
- [ ] Instrument the current weight path end-to-end: where are the copies today (BO alloc per tensor? aperture bounce? per-GEMM re-submit)? Baseline: bytes/token and copies/token on Qwen3-0.6B q4nx.
- [ ] Measure achievable NVMe→SharedBO streaming bandwidth on this box (fio-style, not paper numbers).
- [ ] Land the readahead thread + `posix_fadvise` ahead of the existing decode loop; measure TTFT/tok/s delta.

### P1 (next)
- [ ] Heat table + LRU resident set; swap hot/cold experts across the double-buffer.
- [ ] Route expert GEMMs through the per-shape DPU dispatch (shape table from the 28-variant mechanism).
- [ ] WS-09 cross-ref: one router consuming heat + health, emitting expert ids + rank.

### P2 (if the bet pays off)
- [ ] DraftExpert self-speculative staging on the resident set (WS-07).
- [ ] PagedWeight precision tiering inside SRAM.
- [ ] Stream-loading prefill for long prompts (2606.10493).

## Validation

- Qwen-35B-A3B-class 1BP MoE on NPU: > 20 tok/s decode (same bar as WS-07), > 90% expert heat hit rate, router overhead < 1% of token latency
- Zero CPU copies of weight payloads (perf stat: bytes copied via CPU ≪ model bytes/token)
- Correctness: expert outputs bit-match the DRAM-resident reference (same weights, no precision change in the P0 path)
