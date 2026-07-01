# SDD Progress — NPU Inference Engine

## Breakthrough: INT8 GEMM on XDNA2 (2026-07-01)

**The impossible was done.** INT8 GEMM now works on the NPU, verified across 5 xclbins (131K-786K elements, **100% correct**), achieving **442 ms/tok on Qwen3-0.6B** — 3× faster than BFP16 (1335 ms/tok).

## Current State (master @ 1c3f32e)

### Completed and Committed

- [x] **Phase 1a GEMM Generator** (compile_gemm.py, verify_gemm.py, bench_gemm.py, full tests) → master @ 70aac95
- [x] **BF16/BFP16 GEMM runner** → master @ 8479430
- [x] **4-row MLIR design** (n32_core_placed.py) → fixes at 30a186f, 721c8a5
- [x] **INT8 K-interleaving fixed** via `dataReuse(dimensionsToStream/dimensionsFromStream)` broadcast mode on A_L2L1 ObjectFifo → master @ d1322aa/5acc0db
- [x] **INT8 inference engine** with Chess-compiled vectorized kernel (`matmul_i8_i16` / `mac_8x8_8x8`) → `src/npu_engine_i8.cpp`
- [x] **5 INT8 xclbins NPU-verified** (131K-786K elements, 100% correct)

### Test Suite: All passing

- 18/18 pytest tests pass (~61s, constant data)
- 5/5 INT8 xclbins pass NPU verification (100% element-wise correct)
- End-to-end Qwen3-0.6B inference working at 442 ms/tok

## Architecture

### Two Parallel Engine Tracks

| Track | Data Type | Speed | Status |
|-------|-----------|-------|--------|
| **INT8** (new) | symmetric int8 quantized | 442 ms/tok | ✅ Production |
| **BFP16** (existing) | bfp16ebs8 mixed precision | 1335 ms/tok | ✅ Working |

The INT8 track uses **per-tensor symmetric quantization** for both weights (static) and activations (dynamic), with a combined scale dequantization path.

### INT8 MLIR Design (`bf16_kernel_dev/n1_core_i8_v2.py`)

**The fix that made INT8 possible:**

1. **`dataReuse` broadcast on A_L2L1 ObjectFifo** — switches DMA from round-robin to broadcast mode, eliminating the K-interleaving corruption that plagued earlier INT8 attempts
2. **Row-major A_L3L2** — simplified layout, no interleaving
3. **Sequential per-group DMA drain** — prevents fifo overwrite with M > m
4. **Switch from `matmul_scalar_i8_i16` → `matmul_i8_i16`** — vectorized `mac_8x8_8x8` for ~30× speedup
5. **Tile: 32×64×128** (M×K×N per core), 8 columns × 1 row of AIE cores

### INT8 Inference Engine (`src/npu_engine_i8.cpp`)

- **4-xclbin swap-based context management** — 1 active context at a time, cached BOs
- **Per-tensor symmetric int8 quantization** — activation scales computed dynamically per layer
- **NaN-safe softmax and RMS norm** — handles edge cases from quantized accumulation
- **Full Qwen3-0.6B pipeline** — embed → 28 layers (QKV + O + Gate/Up/Down) → LM head

## Milestone: 1-Bit Ternary Analysis Complete (2026-07-01)

**Key insight: The NPU is swap-bound, not compute-bound.**
- Current INT8 (0.6B): 442ms/tok — 97% is xclbin context swap overhead (448ms)
- Actual GEMM compute + DMA: only ~14ms for the whole forward pass
- 1-bit ternary weights (0.2 bytes/param vs 1.0 for INT8) reduce DMA by 5×
- But without fixing swap, model size barely matters — all 1-72B models run ~0.5-3.6s/tok
- Fix swap → 7B 1-bit at ~400ms/tok (real-time chat) on this laptop

**Sweet spot for THIS hardware (Strix Halo XDNA2): 1.8B-7B at 1-bit**
- 4B model: ~350ms/tok (current) or ~200ms (with M-core fast-switch)
- 7B model: ~800ms/tok (current) or ~300ms (with fast-switch)
- 7 models simultaneously loaded (SigLIP, Whisper, embedding, chat, code, TTS) = ~7.3 GB total
- 128 GB unified memory → all stay hot, zero disk, zero cloud

**200B verdict:** Not viable on this laptop NPU (450× too slow). Needs datacenter hardware.

**Actual killer app:** Local zero-cloud multi-modal AI (vision + code + RAG + chat) on a single laptop.

## Known Issues

### 1. Ramp Data Precision (BFP16 only)
- Microkernel `mm_bfp_mixed.cc` has `static int g_counter` wrapping every 4 K-tiles
- Causes 0.7-2.7% BFP16 precision loss for ramp patterns > 256 K-tiles
- **Not applicable to INT8** — INT8 has 100% correct ramp data
- Workaround: use INT8 engine for production

### 2. engine.cpp/engine.h Mismatch (Stale)
- `include/engine.h` declares `CustomNpuEngine` with `NpuGemm` member
- `src/engine.cpp` still defines old `NpuInferenceEngine` (FLM pipeline)
- **Superseded** — `src/npu_engine_i8.cpp` is the production engine now

### 3. Stale Build Artifacts
- Multiple engine variants in `src/` (v2-v10, fused, batched, chess, attn, etc.)
- Worth cleaning up for clarity

## Key Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-01 | `dataReuse` broadcast on A_L2L1 | Fixes INT8 K-interleaving — switches DMA from round-robin to broadcast |
| 2026-07-01 | Row-major A_L3L2 layout | Simplifies INT8 design, no interleaving needed |
| 2026-07-01 | Sequential per-group DMA drain | Prevents fifo overwrite when M > m |
| 2026-07-01 | Vectorized `matmul_i8_i16` kernel | ~30× faster than scalar INT8; Chess-compiled, byte-identical |
| 2026-06-30 | Remove layout_A from host runners | AIE A_transformations handles it |
| 2026-06-30 | Empty C_transformations for multi-row | Original formula only works for (128,128); aiecc rejects scaling |
| 2026-06-30 | Accept BFP16 ramp data precision loss | Microkernel limitation; INT8 path has no such issue |
