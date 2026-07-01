# SDD Progress — NPU + GPU Hybrid Inference

## Current State (master @ 9ed49db)

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

### Root Cause Finding

**The `.q4nx` format is a proprietary NPU weight format.** `dequant_i8_to_float` interprets it as INT4 with per-group BF16 scales, producing weights that are 94% negative for gate/up projections → SiLU(gate) ≈ 0 → FFN dead → hidden state blows up.

**This is NOT a forward pass bug or weight-loading bug.** The CPU reference (`tools/cpu_ref_infer.cpp`) produces EXACTLY the same output as the GPU engine — confirming self-consistency. The AIE kernel in the xclbin dequantizes differently internally (its own CPU dequant function `npu_dequant_block` produces NaN/inf from the same bytes, yet `npu_infer` works perfectly).

**Recommended fix:** Export weights from HF Qwen3-0.6B (float32/BF16) and load those directly, bypassing `dequant_i8_to_float` entirely. The model architecture is: 28 layers, 16 Q heads, 8 KV heads (GQA=2), HD=128, H=1024, IM=3072, NV=151936, RoPE base=1000000.

### Remaining Issues

- Correctness blocked on Q4NX incompatibility (see above). With HF weights, the existing forward pass logic should work.
- **CPU bottleneck:** At ~105 ms/tok, CPU-side ops (norm, RoPE, attention mixing, SiLU) dominate. Port to GPU kernels after correctness.
- **CU count underutilized:** 40 CU available, kernel uses only block-grid over N dimension.

### See Also

- `docs/superpowers/handoff.md` — Detailed debugging handoff with analysis artifacts, file locations, and recommended next steps
