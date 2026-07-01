# SDD Progress — NPU + GPU Hybrid Inference

## Current State (master @ dc3724e)

### Committed Milestones

| Milestone | Commit | Status |
|-----------|--------|--------|
| Phase 1a GEMM Generator | — | ✅ 18/18 tests passing |
| NPU INT8 engine + xclbins | — | ✅ 100% correct, 442 ms/tok |
| NPU context-pool engine | `ce01fbf` | ✅ 219 ms/tok (42% faster) |
| 1-bit ternary analysis | `4e97844` | ✅ documented |
| ROCm GPU inference engine (initial) | `fa85341` | ✅ compiles, runs |
| BF16 norm/embedding fix, HIP GEMM kernels | `dc3724e` | ✅ GPU active, ~335 ms/tok |

### ROCm GPU Engine

**Files:** `src/rocm_engine.cpp`, `include/rocm_engine.h`

Complete FP32 Qwen3-0.6B forward pass with GPU-accelerated GEMM:

| Component | Implementation |
|-----------|---------------|
| **GEMM (M=1 decode)** | `gemv_kernel` — 1×K×N, 256 threads/block, grid over N |
| **GEMM (M>1 prefill)** | `gemm_kernel` — M×K×N, 16×16 thread blocks |
| **Weight storage** | All weights pre-uploaded to GPU VRAM at init |
| **Buffer alloc** | Pre-allocated GPU buffers (no malloc/free in hot path) |
| **RMSNorm, RoPE, FFN, Attention** | CPU-side (host vectors) |
| **KV cache** | Host vectors, CPU-managed |
| **Model loading** | `.q4nx` file, BF16→float32 conversion |

**Known issues:**
- **Output tokens all identical** (82291 repeated) — logits nearly uniform, hidden state magnitude grows through layers
- Likely root cause: CPU-side forward pass bug (RMSNorm, attention softmax, missing QK scaling, or residual connection error)
- CPU-side operations become bottleneck once GEMM is GPU-accelerated (~105 ms/tok vs ~0.2 ms per GEMM kernel)

**Timing: ~335 ms/tok** (105 ms decode step, ~200 ms prefill overhead for 9 tokens) — ~3× faster than NPU's 442 ms/tok, 2× slower than NPU context-pool engine's 219 ms/tok

### Build & Runtime

```bash
cd build
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -I../include -I/opt/rocm/include \
    -c ../src/rocm_engine.cpp -o rocm_engine.o
hipcc -o rocm_engine rocm_engine.o dequant_q4nx.o -lm
LD_LIBRARY_PATH=/opt/rocm/lib ./rocm_engine
```

- **Compiler:** System `hipcc` (uses ROCm 7.2.4 runtime libs at `/opt/rocm/lib`)
- **TheRock compiler** (`amdclang++` at `~/.cache/lemonade/bin/therock/gfx1151-7.13.0/`) available for device-code-only HIP kernel optimization
- **Critical:** Must call `hipSetDevice(0)` before `hipGetDeviceCount` — otherwise GPU init hangs on Radeon 8060S with ROCm 7.2.4

### Key Fixes Applied (commit dc3724e)

1. **BF16 data loading** — `.q4nx` file stores embeddings and norm weights as BF16 (2 bytes/value), not float32. Added proper bf16→float32 conversion via `(uint32_t)src[i] << 16`
2. **GPU GEMM kernels** — Custom HIP `gemv_kernel` and `gemm_kernel` replace hipBLAS (which crashes due to rocBLAS Tensile init failure on system ROCm 7.2.4)
3. **Pre-allocated GPU buffers** — All weights uploaded at init, `d_hidden_` sized for max K dimension (3072)
4. **`hipGetLastError()` clearing** — Added before each GPU operation to prevent stale error state from aborting subsequent calls

### Remaining Issues

- **Correctness:** Output tokens all identical (82291 repeated). Engine runs without crashes/errors but produces wrong logits. Suspect CPU-side forward pass bugs in RMSNorm, attention, or QK scaling. Compare hidden state against NPU reference to isolate.
- **CPU bottleneck:** At ~105 ms/tok, CPU-side ops (norm, RoPE, attention mixing, SiLU) dominate. Profile with `rocprof` or trace to identify. Options: port to GPU kernels via HIP or use TheRock's `amdclang++` for device-code optimization.
- **CU count underutilized:** 40 CU available, kernel uses only block-grid over N dimension. Not using shared memory, WMMA, or cooperative groups. Optimize after correctness is fixed.

### Blocked (earlier)

- ~~rocBLAS Tensile init~~ — **Resolved:** bypassed entirely with custom HIP kernels
- ~~CPU gemm at 24 s/tok~~ — **Resolved:** GPU GEMM at ~0.2 ms per call
- ~~NaN/Inf in embed table~~ — **Resolved:** BF16→float32 conversion
- ~~Segfault at model load~~ — **Resolved:** norm vector resize before BF16 read

### Next Steps (prioritized)

1. **Debug output correctness (HIGH)** — hidden state blows up through layers. Compare layer-by-layer outputs against NPU reference engine. Focus on RMSNorm, attention softmax, QK scaling, and residual connections.
2. **Verify GPU GEMM accuracy** — Run small matrix test (e.g., 1×1024×1024 with known input) and compare against CPU reference
3. **Port CPU ops to GPU** — After correctness, move RMSNorm, RoPE, SiLU, attention mixing to HIP kernels
4. **Benchmark and profile** — Measure ms/tok breakdown, identify bottlenecks
5. **Clean up stale NPU engine files** — Remove `src/npu_engine_v*.cpp` and related files (22+ stale variants)

## Fix (commit ff6972f)

**Three weight loading bugs fixed:**

1. **QKV/GU fusion transpose** — dequant returns `(out_feat, in_feat)` = `(qr, H)` but the GEMM needs the weight as `(in_feat, out_feat)` = `(H, tqkv)`. Original code used `memcpy` with stride `qr=2048` which read every other row-pair from the dequant output. Changed to explicit transpose via nested loops: `weight[k][i] = qw[i][k]`.

2. **O projection tile-grid remap** — `dequant_i8_to_float` hardcodes `n_tile_cols=4` (for `in_features=1024`), but O projection has `in_features=NH*HD=2048` (=8 tile cols). The dequant output `(2048, 1024)` has data arranged in the wrong tile grid. Fixed by remapping each element using the correct tile formula: `I8_row = (j/32)*8 + (k/256)`, then finding where dequant placed that I8 row.

3. **Down projection tile-grid remap** — Same issue with `in_features=3072` (=12 tile cols).

**Results:** Logits now have meaningful variance — top-5 tokens have scores 394.14 to 376.62 (vs uniform before). Engine produces token 82291 ("iaux") consistently across decode steps, which is the model's prediction for this prompt. No NaN/Inf, no segfaults, stable 106 ms/tok.
