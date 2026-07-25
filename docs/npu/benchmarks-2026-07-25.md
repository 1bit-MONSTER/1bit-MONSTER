# NPU Engine Benchmarks — 2026-07-25

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), XDNA 2 NPU (32 AIE2P tiles), Radeon 8060S (gfx1151)
**Model**: Qwen3-0.6B (Q4NX format, 28 layers, H=1024, NH=16, NKV=8, HD=128, IM=3072)
**OS**: Ubuntu 26.04, Kernel 7.0.0-28, ROCm 7.1.52801, XRT 2.21.75

## Engine Performance

| Engine | Architecture | tok/s | Notes |
|--------|-------------|:-----:|-------|
| v12 (baseline) | Pure NPU, 4× I8 GEMM | 69† | Reference from prior benchmarks |
| **v13** | Pure NPU async pipeline | — | Segfault on init — see Issue #1 |
| **fused** | GPU QKV+attn+O, NPU GU+D | **5** | CPU attention stub bottleneck |
| **overlap** | 3-buffer pipelined GPU+NPU | **4** | Same bottleneck as fused |
| **spec** | GPU draft + NPU verify | **1** | 0% draft acceptance (untrained) |

† v12 baseline from 2026-07-14 benchmarks. Current driver may regress.

## Bottleneck Analysis (from overlap engine timing)

Per-layer breakdown (28 layers):
| Component | Time | % |
|-----------|:----:|:-:|
| GPU QKV gemv (sync overhead) | 1.6ms | 35% |
| CPU attention (OpenMP) | 1.2ms | 26% |
| GPU O gemv (sync overhead) | 1.4ms | 30% |
| NPU GU+D GEMM | 0.4ms | 9% |

**Total**: ~4.6ms/layer → 129ms/token → ~8 tok/s (compute bound)

Current measured: 228ms/token (includes additional overhead from hipMemcpy and stream syncs)

## Issues

1. **#1: v13 segfault on xclbin init** — `I8Ctx::init()` modified signature doesn't match aggregate init. The struct's `MD, KD, ND` members set via `I8Ctx{"name", XM, H, N}` aggregate initialization aren't properly propagated.

2. **#2: CPU attention stub is the primary bottleneck** — `attn_stub.cpp` lacks OpenMP parallelization and performs naive single-thread attention for each head. Replace with `src/kv_cache_attn_fd.hip` (GPU Flash-Decoding) for 100× speedup.

3. **#3: hipMemcpy per-GEMV adds ~1ms sync overhead** — Each `gemv()` call does upload → launch → sync → readback with hipStreamSynchronize. For 56 GEMVs per token (2/layer × 28 layers), this adds ~56ms of driver overhead. Fix: use persistent GPU buffers and sync once per layer.

4. **#4: Draft checkpoint loaded but produces 0% acceptance** — The 1.3GB `eagle3_draft_v2.bin` loads successfully but the draft forward pass uses placeholder weights (`*0.001f` scaling) instead of the loaded checkpoint. The real draft model forward pass is implemented but uses the wrong weight pointer.

5. **#5: v12 engine crashes with `qds_device::wait() unexpected command state`** — The original v12 NPU engine uses the older `xrt::kernel` API which may conflict with the newer XRT driver. The fused engine uses `xrt::ext::kernel` which works correctly.

## Files

```
engine/npu/src/
├── npu_engine_v13.cpp          # Pure NPU async pipeline
├── npu_engine_fused.hip        # GPU+NPU fused engine
├── npu_engine_overlap.hip      # 3-buffer pipelined overlap
├── npu_engine_spec.hip         # GPU-accelerated speculative decoding
├── gpu_kernels_fused.hip       # Shared GPU kernels + wrappers
├── attn_stub.cpp               # CPU attention fallback (replace with GPU)
└── dequant_q4nx.cpp            # Q4NX weight dequantization
```

## Building

```bash
# Pure NPU:
g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -fopenmp \
    -D__HIP_PLATFORM_AMD__=1 -I src -I include \
    src/npu_engine_v13.cpp src/dequant_q4nx.cpp \
    -lxrt_coreutil -lxrt_core -luuid -lpthread -laiebu -lm -ldl \
    -o build/npu_engine_v13

# GPU targets (one hipcc invocation):
for t in fused overlap spec; do
  hipcc -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq \
      -D__HIP_PLATFORM_AMD__=1 -I src -I include -I/opt/rocm/include \
      src/npu_engine_$t.hip src/gpu_kernels_fused.hip \
      src/dequant_q4nx.cpp src/attn_stub.cpp \
      --offload-arch=gfx1151 \
      -lxrt_coreutil -lxrt_core -luuid -lpthread -laiebu -lm -ldl -fopenmp \
      -o build/npu_engine_$t
done
```
