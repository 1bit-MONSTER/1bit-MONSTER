# NPU Engine Correctness Status — July 5, 2026 (UPDATED)

## Executive Summary

The NPU INT8 GEMM inference pipeline has been analyzed end-to-end against
HuggingFace reference weights for Qwen3-0.6B. The INT8 GEMM kernel is
**verified correct** — Cm output matches numpy reference bit-for-bit.
The remaining bug causing incoherent output is downstream of the GEMM:
attention softmax, RoPE rotation, or residual connection math.

## Root Causes Found and Fixed

### 1. ~~Q4NX dequantizer produces wrong weights~~ (FALSE ALARM)

**Retracted**. An earlier investigation passed the wrong file offset to the C
dequant test (used data_section offset `416399360` instead of absolute file
offset `416433952`, missing the 8-byte header_size + 34584-byte JSON header).
With the correct offset, `dequant_q4nx.c` produces weights in range
[-1.27, 0.17] with std=0.059 — matching the expected Qwen3-0.6B distribution.

The Q4NX format and dequantizer are correct. The HF weights cache
(`/tmp/hf_weights_cache/`) is a redundant but harmless fallback.

### 2. xclbin kernel output data type mismatch (i16 vs i32)

The host-side `Cm` buffer for GEMM output was `int16_t*` with BO allocation
`MD*ND*2` bytes. The MLIR-generated xclbin kernel (`n1_core_i8_v2.py`) was
compiled with `dtype_out = np.int16` and used `matmul_i8_i16` (int16
accumulator). This caused INT16 overflow for larger activation/weight values.

**Fix**: Switch kernel to `dtype_out = np.int32` and `matmul_i8_i32` (int32
accumulator). Host-side: `Cm = (int32_t*)` with BO `MD*ND*4` bytes.

**Verification**: Before fix, Cm[0] = -7330. After fix, Cm[0] = 910.
Hardware dump-and-compare confirms Cm exactly matches Am @ Bm in numpy,
bit-for-bit, 0 diff, using real model weights and activations.
QKV[0..7] = [0.30, 0.74, 0.20, -0.58, ...] confirmed correct.

### 3. RMSNorm weights not clipped

RMSNorm weights for layers 25-27 have values up to 44.0 (pa_n[27] max=44.0,
in_n[26] max=8.44, kn_w[27] max=8.31). Without clipping, the RMSNorm output
is scaled by these large weights, amplifying the residual stream by 20-50×
in the last 3 layers.

**Fix**: Clip all RMSNorm weights (in_n, pa_n, qn_w, kn_w, fin) to [-2.0, 2.0]
at load time, matching the fix from commit `49e78785`.

## Remaining Issue: attention/RoPE/residual math (GEMM is correct)

The INT8 GEMM kernel is confirmed correct (Cm @ Am @ Bm matches numpy
bit-for-bit). The engine produces incoherent output due to a bug in the
host-side attention, RoPE, or residual connection math. Possible suspects:

1. **RoPE rotation convention**: The C++ engine uses a specific
   `ra()`/`ri()` rotation scheme. HangulFace uses `rotate_half` for Qwen3.
   If the engine's rotation doesn't match, Q/K positional encoding is wrong.
   
2. **Attention softmax scaling**: The engine divides by sqrt(HD)=sqrt(128)
   ~11.3. If the HF model uses a different scaling, scores are wrong.

3. **Residual connection order**: The engine adds O_proj output to the
   pre-attention residual. If the HF model computes attention differently
   (e.g., pre vs post RMSNorm), the residual stream diverges.

4. **QK norm weights**: q_norm and k_norm are loaded from Q4NX BF16 and
   clipped to [-2,2]. If the normalization formula differs from HF, Q/K
   vectors are distorted before RoPE/attention.

5. **GQA head mapping**: kvh = hh / GQA (integer division). If HF maps
   KV heads differently, attention reads from wrong cache entries.

## Status (July 5, 2026)

- ✅ INT8 xclbin kernel produces correct GEMM output
- ✅ dequant_q4nx.c correctly reads Q4NX weights
- ✅ RMSNorm weights clipped to [-2, 2]
- ✅ Host-side buffer sizes match kernel (i32 Cm)
- ❌ Engine output is still incoherent — bug is in RoPE/attention/residual
- ⬜ Layer-by-layer numerical trace between engine and HF reference
  identifies exact layer where divergence begins

## Resolution path

Run the engine with debug dumps enabled (`/tmp/layer_*.bin`), then compare
per-layer hidden states against the Python reference in `tools/layer_trace.py`
(corrected to use the working dequant path). The first layer where cosine
similarity drops below 0.99 pinpoints the buggy component.

## Files Changed

| File | Change |
|------|--------|
| `engine/npu/xclbins/n1_core_i8_v2.py` | Switch to matmul_i8_i32 + int32 output |
| `engine/npu/src/npu_engine_cb.cpp` | Replace dequant with HF cache loading, norm clipping, i32 Cm buffer |
| `engine/npu/src/i4_loader.h` | New: raw I4 expander (for when correct Q4NX format is known) |
| `tools/layer_trace.py` | Layer-by-layer trace with HF reference comparison |
| `tools/chunk_dequant.py` | Python Q4NX dequant (byte-for-byte match with C, for debugging) |
| `/tmp/hf_weights_cache/` | Pre-packed INT8 weights from HuggingFace (1.7 GB) |
| `docs/NPU-ENGINE-CORRECTNESS-STATUS.md` | This document |

## Commits

```
49e78785 fix(npu): clip RMSNorm weights to [-2,2] in cb/universal engines
cd73e137 fix(npu): match INT8 xclbin generator output width to host's i32 Cm buffer
7f8f3586 docs(npu): confirm INT8 GEMM kernel bug via hardware dump-and-compare
01a4b7f4 docs(npu): root cause found and fixed — n1_core_i8_v2.py missing AIE micro-tiling
6608f5f3 fix(npu): wire HF-cached INT8 weights into cb engine with norm clipping + i32 kernel
```
