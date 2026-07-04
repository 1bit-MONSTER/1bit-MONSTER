# v12 Correctness — Audited July 5, 2026

## Discovery (July 3, 2026)

The open-source C++ engine (v12) was benchmarked at 97 tok/s across all project docs,
but **never validated for output coherence**. Sanity-checking the actual output against
FLM for the same prompt revealed complete garbage — the speed numbers were real, the
output behind them never was.

## Bugs Found and Fixed (7 Rounds)

| Round | Bug | Fix |
|-------|-----|-----|
| 1 | Activation quant clipped outside [-5,5] | `dynamic_ascale()` — per-call amax |
| 2 | Weight matrices packed wrong orientation | `transpose_pack()` + correct `in_features` |
| 3 | LM head used embed table instead of untied weights | `lm_head_f32` separate from `emb_f32` |
| 4 | RoPE used wrong convention | half-split `rotate_half` |
| 5 | Causal mask off-by-one in prefill | V-loops use `sp+pi+1` |
| 6 | Spec decode KV cache not written for draft tokens | Fix in 3 spec engines |
| 7 | Hardcoded paths | `NPU_XCLBIN_DIR`/`NPU_MODEL_PATH` env vars |
| 8 | NaN on LM head softmax | NaN guards added |

## Current Status (July 5, 2026)

### ✅ Host-side math: CORRECT
The universal engine (`npu_engine_universal.cpp`) compiles and runs. The host-side math
(dequant → transpose → INT8 quant → dispatch) has been verified against the fused
xclbin reference path.

### ✅ Fused xclbin: VALIDATED
The torch2aie fused xclbin (single-layer dispatch) produces output matching the CPU
oracle within `max_abs=0.0078`. This confirms:
- Q4NX dequantization is correct
- NPU INT8 GEMM kernels produce correct output
- Weight packing for the fused format is correct

### ⚠️ Standalone INT8 xclbins: NOT RE-VERIFIED
The 4 standalone INT8 xclbins (QKV, O, GU, D) used by `npu_engine_universal.cpp`
are opaque MLIR-compiled binaries. After all host-side fixes, they have NOT been
re-verified for coherent end-to-end output. The `npu_engine_fused_i8.cpp` engine
(using the fused xclbin) IS verified — it runs at 4 tok/s with valid output.

### 2. Weight-packing transpose

`dequant_i8_to_float(_ex)` returns row-major `[out_features, in_features]` (verified
from the tile-index arithmetic in `dequant_q4nx.c`: `linear_idx = (tile_row*32+lr) *
out_cols + (tile_col*256+col)` — classic row-major with `out_features` as the
slower-varying dimension). The GEMM dispatch (`packB()`/`go()`) needs the transpose
of that — `[in_features, out_features]`, since it computes `A[tokens,in] @
B[in,out]`. The original packing loop read the dequantized buffer with an
`out_features`-sized stride as if it were already `[in,out]`, silently scrambling
every weight matrix (Q/K/V/O/Gate/Up/Down, all of them) while still producing
finite, non-NaN, plausible-magnitude numbers — exactly why "doesn't crash" was never
sufficient validation. Confirmed empirically: dequanted Q-weight values at several
`(out_idx, in_idx)` positions matched the real HuggingFace model's weights (within
expected 4-bit quantization noise, ~5-20% relative error) when read as `[out,in]`,
and did not match when read as `[in,out]`.

A related, separate bug: O-proj and Down-proj dequant calls used the default
`dequant_i8_to_float()` wrapper, which hardcodes `in_features=1024`. O-proj's real
`in_features` is 2048 (`NH*HD`) and Down-proj's is 3072 (`IM`) — using the wrong
value scrambles the *dequant tiling itself*, before the transpose issue even
applies. Fixed by calling `dequant_i8_to_float_ex()` with the correct value for
these two projections.

**Fix**: `transpose_pack()` helper, correctly transposing dequant's `[out,in]`
output into the `[in,out]` layout `packB()`/`go()` expect, plus correct
`in_features` for O/Down.

### 3. Activation quantization clipping

`go()`'s activation-to-INT8 quantization used a hardcoded scale of `5.0f/127.0f`,
assuming activations stay within `[-5,5]`. Measured actual post-RMSNorm activations
at layer 0: range `[-8.24, 7.01]` — any value past ±5 silently clips to ±127 in the
INT8 quantization. This happens at every GEMM call, every layer, compounding across
all 28 layers.

**Fix**: `dynamic_ascale()` — per-call amax-based scale, matching the same approach
`packB()` already uses for weights, instead of a fixed constant.

**Update (July 2026)**: `dynamic_ascale()` has now been applied to **ALL** engine
files — every `.cpp` file in `engine/npu/src/` that uses `go()` calls now computes
activation scale dynamically per-call. The fix covers 19 source files including the
production engines (`npu_engine_all.cpp`, `npu_engine_server.cpp`), the speculative
decode engines (`spec.cpp`, `spec_decode.cpp`, `spec_v2.cpp`), the universal merge
engine (`universal_v12merge.cpp`), the multi-token engine (`npu_engine_mt.cpp`),
and all vintage numbered engines (v2 through v12). The v4 engine's inline dequant
path (bypassing `go()`) was also fixed with per-buffer dynamic scales. The only
remaining references to `5.0f/127.0f` are in comments documenting the old approach.

**Update (July 2026)**: `dynamic_ascale()` has now been applied to **ALL** engine
files — every `.cpp` file in `engine/npu/src/` that uses `go()` calls now computes
activation scale dynamically per-call. The fix covers 19 source files including the
production engines (`npu_engine_all.cpp`, `npu_engine_server.cpp`), the speculative
decode engines (`spec.cpp`, `spec_decode.cpp`, `spec_v2.cpp`), the universal merge
engine (`universal_v12merge.cpp`), the multi-token engine (`npu_engine_mt.cpp`),
and all vintage numbered engines (v2 through v12). The v4 engine's inline dequant
path (bypassing `go()`) was also fixed with per-buffer dynamic scales. The only
remaining references to `5.0f/127.0f` are in comments documenting the old approach.

## What's Still Broken

With all three original fixes plus the RoPE convention fix applied, the host-side
math in all 4 dispatch copies now matches the HuggingFace Qwen3 reference:
- LM head → correctly uses untied lm_head.weight (not embed tokens)
- Weight packing → correctly transposes dequant [out,in] to [in,out] for NPU GEMM
- Activation scale → dynamic per-call amax, no hardcoded clipping
- RoPE → uses HuggingFace rotate_half convention (verified against HF modeling source)

**The remaining risk is in the compiled NPU kernel binaries (.xclbin files).**
These are opaque MLIR-compiled binaries; debugging them would need the AI Engine
Simulator, which is blocked on this machine (missing `aie2p_8x4_device.json` for
NPU2 — see `docs/FUSED-INTEGRATION-BLOCKER.md`'s aiesimulator section).

If the host-side math is correct (as we believe it now is), and output is still
incoherent, the bug must be in the xclbin kernels themselves — either the INT8
GEMM matrix multiply or the quantization/dequant logic inside the NPU compute tiles.
Verification approach: run `tools/layer_trace.py` to produce a Python reference
trace of any layer's intermediates, and diff the C++ engine's intermediates against
it. If the C++ intermediates match through the full layer, the xclbin kernels are
the only remaining variable.

## Status (July 4, 2026, after RoPE fix)

**Do not wire `1bit.engine` (or any v12 variant) into the production daemon** until
the xclbin kernels have been validated against a Python reference trace. FLM
proxy stays in production (`daemon/npu-gpu-cpud.py`, port 9090) until this is resolved.
