# WS-01 Findings — CPU Attention Stub → GPU Flash-Decoding (P0.5 / P0 ✅)

**Date:** 2026-07-31 · **Tools:** `attn_swap_bench.hip` (isolated bench) + engine patch to `engine/npu/src/npu_engine_overlap.hip`

## What was done

1. **Isolated benchmark** at the overlap engine's exact config (NH=16, NKV=8, HD=128, fp16 K/V cache, seq 512-4096): CPU stub (`attn_stub.cpp`, the 8.4 ms/layer bottleneck from issue #936) vs the existing flash-decoding kernel (`src/kv_cache_attn_fd.hip`, split-KV two-pass, already validated in bench_kv_fd).
2. **Engine swap**: `npu_engine_overlap.hip` — extern decl + both decode call sites now call `rcpp_kv_cache_attn_decode_fd` (drop-in: same signature, fp16 Q/K/V/out). Build: `src/kv_cache_attn_fd.hip` replaces `attn_stub.cpp` in the hipcc line. Builds clean (warnings only).

## Results (Strix Halo, Radeon 8060S, ROCm 7.1, gfx1151)

| seq_len | CPU stub | GPU FD | speedup | max|diff| vs stub |
|---|---:|---:|---:|---:|
| 512 | 7.81 ms | 0.147 ms | **53×** | 0.00017 |
| 1024 | 23.66 ms | 0.202 ms | **117×** | 0.00006 |
| 2048 | 45.77 ms | 0.191 ms | **240×** | 0.00006 |
| 4096 | 108.95 ms | 0.439 ms | **248×** | 0.00006 |

- At the engine's operating point (context 1024): attention drops from 23.7 ms → 0.2 ms per layer. × 28 layers = −658 ms/token of serial time *if attention were the serial bottleneck* (the overlap engine pipelines, so realized gain is bounded by the pipeline; measured e2e pending NPU health).
- Numerics: max diff 6e-5 vs the fp32 stub — well inside fp16 tolerance; FD kernel already validated to <0.05 fp16 units in bench_kv_fd.
- FD throughput at L=4096: 38 GB/s vs the KV-FD baseline's 57.1 GB/s at L=2048 (same ballpark — the kernel is memory-bound as designed).

## Engine-level e2e: BLOCKED on pre-existing NPU hang (not the swap)

- Both the ORIGINAL binary (attn_stub) and the FD binary **hang identically at prefill** (9-token prefill, 4/4 attempts each).
- dmesg: `amdxdna ... IO_PAGE_FAULT` (DMA faults, domain=0x0003, 0x74da... addresses) — the NPU shim's DMA path is faulting on this box's current driver/firmware state.
- This is the documented "hang at boot-to-decode transition" (docs/research/npu/benchmarks-2026-07-25.md: "~1/3-1/2 of runs still hang"; issues #939/#940 family) — **pre-existing, unrelated to this change** (the swap only affects decode attention, which runs after prefill).
- Unblocking requires NPU driver/firmware work (sudo territory, P0.1/P0.3 cross-ref) — did not touch drivers uninvited.

## Status

- ✅ Kernel-level swap validated: 53-248× faster attention, numerics clean.
- ✅ Engine patched + builds (binary: `build/npu_engine_overlap_fd`).
- 🔲 E2E tok/s number: blocked on NPU IOMMU hang (P0.1). First action when NPU is healthy: `./build/npu_engine_overlap_fd models/qwen3_0_6b.q4nx 32` and record tok/s vs the 5.0 tok/s baseline.

## Build command (updated for the swap)

```bash
hipcc -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq \
  -D__HIP_PLATFORM_AMD__=1 -I engine/npu/src -I src -I include -I/opt/rocm/include \
  engine/npu/src/npu_engine_overlap.hip engine/npu/src/gpu_kernels_fused.hip \
  engine/npu/src/dequant_q4nx.cpp src/kv_cache_attn_fd.hip \   # ← was attn_stub.cpp
  --offload-arch=gfx1151 \
  -lxrt_coreutil -lxrt_core -luuid -lpthread -laiebu -lm -ldl -fopenmp \
  -o build/npu_engine_overlap_fd
```

## Notes / follow-ups

- The engine still does `hipStreamSynchronize` right after attention + per-layer K/V staging memcpys (the doc's "hipMemcpy per-GEMV sync overhead", issue #937) — next lever after P0.1 unblocks e2e.
- Prefill attention is still CPU (inline scores loop, "simpler for now") — worth the same FD treatment for prefill later (it's batched, different kernel shape).
- `attn_stub.cpp` remains in-tree for CPU-only builds — the FD kernel is HIP-only.
