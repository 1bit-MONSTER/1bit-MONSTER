# SDD Progress — NPU + GPU Hybrid Inference

## Current State (master @ fa85341)

### Completed and Committed

- [x] **Phase 1a GEMM Generator** — compile_gemm.py, verify_gemm.py, bench_gemm.py, full test suite (18/18)
- [x] **BF16/BFP16 GEMM runner** and 4-row MLIR design (n32_core_placed.py)
- [x] **INT8 GEMM on XDNA2 NPU** — 5 xclbins, 100% correct, 442 ms/tok
- [x] **INT8 inference engine** — npu_engine_i8.cpp with Chess-compiled vectorized kernel
- [x] **1-bit ternary analysis** — 1.8-7B sweet spot, 200B not viable on NPU
- [x] **ROCm GPU inference engine** — RocmInferenceEngine class for Radeon 8060S

### ROCm GPU Engine (NEW)

**File:** `src/rocm_engine.cpp`, `include/rocm_engine.h`

Complete FP32 Qwen3-0.6B forward pass with:
- hipBLAS GEMM (and raw HIP kernel fallback) for all matmuls
- RMSNorm, RoPE, SwiGLU FFN, GQA attention, KV cache, LM head
- ~24 s/tok (CPU gemm fallback — no GPU acceleration yet due to rocBLAS Tensile init issue)

**TheRock compiler:** ROCm 7.13.0 optimized clang++ at `~/.cache/lemonade/bin/therock/gfx1151-7.13.0/`
**System ROCm:** 7.2.4 at `/opt/rocm/`
**GPU:** Radeon 8060S (gfx1151, 40 CU, 62 GB VRAM)

### Blocked

- **rocBLAS Tensile initialization fails**: `unordered_map::at` — the system ROCm 7.2.4 has gfx1151 kernel files but no master `TensileLibrary.dat`. TheRock (7.13.0) has the data but its libs are incompatible with system driver.
- **GPU acceleration not active**: CPU gemm in the inference loop is 24 s/tok. Proper HIP/WMMA kernel or hipBLAS fix needed.

### Next Immediate Steps

1. **Fix rocBLAS Tensile** — either create a TensileLibrary.dat pointing at existing gfx1151 files, or switch to a direct HIP GEMM kernel (pre-allocated GPU buffers, no malloc/free per call)
2. **Add actual GPU GEMM** — replace `gemm_cpu` with a proper HIP kernel or working hipBLAS call
3. **Verify outputs match NPU** — compare logits between NPU and GPU engines
