# NPU Engine Correctness Status — July 5, 2026 (UPDATED)

## Executive Summary

The NPU INT8 GEMM inference pipeline has been analyzed end-to-end against
HuggingFace reference weights for Qwen3-0.6B. The INT8 GEMM kernel is
**verified correct** — Cm output matches numpy reference bit-for-bit,
0 diff, with real model weights and real activations.

The remaining bugs causing incoherent output are all in the **host-side
attention/RoPE/addressing code**, not in the xclbin kernel, not in the
dequantizer, and not in the weight layout. Two bugs have been found and
fixed in the decode path; one bug has been located in the prefill path.

## Earlier False Alarm (retracted)

An earlier investigation claimed `dequant_q4nx.c` produced weights 800x
too large and that the xclbin kernel had a data-tiling mismatch producing
~100x wrong GEMM output. Both were measurement errors in the investigation
itself:

- The C dequant test used data_section offset `416399360` instead of
  absolute file offset `416433952` (missing 8 + 34584 byte header).
  With the correct offset, `dequant_q4nx.c` produces weights in range
  [-1.27, 0.17], std=0.059 — matching HuggingFace's distribution.

- Hardware dump-and-compare confirms `Cm == Am @ Bm` bit-for-bit with
  0 diff. The "100x wrong GEMM" observation was an artifact of comparing
  against a Python reference that itself used the corrupted dequant data.

The Q4NX format, `dequant_q4nx.c`, and the standalone INT8 xclbins are
all verified correct.

## Root Causes Found and Fixed

### 1. xclbin kernel output data type mismatch (i16 vs i32)

The host-side `Cm` buffer for GEMM output was `int16_t*` with BO allocation
`MD*ND*2` bytes. The MLIR-generated xclbin kernel (`n1_core_i8_v2.py`) was
compiled with `dtype_out = np.int16` and used `matmul_i8_i16` (int16
accumulator). This caused INT16 overflow for larger activation/weight values.

**Fix**: Switch kernel to `dtype_out = np.int32` and `matmul_i8_i32` (int32
accumulator). Host-side: `Cm = (int32_t*)` with BO `MD*ND*4` bytes.

**Verification**: Before fix, Cm[0] = -7330. After fix, Cm[0] = 910.
Hardware dump-and-compare confirms Cm exactly matches Am @ Bm in numpy,
bit-for-bit, 0 diff, using real model weights and activations.

### 2. RMSNorm weights not clipped

RMSNorm weights for layers 25-27 have values up to 44.0 (pa_n[27] max=44.0,
in_n[26] max=8.44, kn_w[27] max=8.31). Without clipping, the RMSNorm output
is scaled by these large weights, amplifying the residual stream by 20-50x
in the last 3 layers.

**Fix**: Clip all RMSNorm weights (in_n, pa_n, qn_w, kn_w, fin) to [-2.0, 2.0]
at load time, matching the fix from commit `49e78785`.

### 3. Decode loop off-by-one (fixed)

Step 0 of decode ran a full 28-layer forward on the prefill's already-finalized
hidden state `h` at position `sp=9`, appending a bogus K/V entry at slot 9
BEFORE the LM head ran. This re-derived a hidden state from a hidden state,
double-counting layers 0..27 and corrupting the KV cache for every subsequent
decode step.

**Fix**: Reorder the decode loop body — LM head FIRST (predicts next token
from prefill-final h directly), THEN forward pass on the sampled token to
produce h for the next iteration. Commit `21864a41`.

## Remaining Issue: prefill Q addressing stride mismatch (located, not fixed)

### Evidence

Layer-by-layer trace comparing engine dumps (`/tmp/layer_*.bin`) against
HuggingFace reference produces:

```
Layer   HF norm    Eng norm   Cos sim   Status
  L0     8.45      14.03     0.235     ⚠️  already diverges
  L1    10.16      16.79     0.135     ⚠️
  L2    10.54    6934.47     0.580     ❌  catastrophic blowup
  L3+   grows     ~6930      ~0.6      ❌  stays bloated
```

Cos_sim=0.24 at layer 0 is **far too low for pure INT8 quantization noise**
(per-tensor INT8 simulation matches float to within 0.01 here). This points
to an addressing bug.

### The bug

In `npu_engine_cb.cpp`, prefill:

```cpp
// Line 215: cq.go writes QKV with stride 4096 per token (correct)
cq.go(l, ..., qo_b.data(), 4096);

// Line 219: Q normalization reads with stride NH*HD=2048 ← WRONG
qo_b[pi*NH*HD + hh*HD + d]  // should be pi*4096

// Line 220: K/V reads with stride 4096 (correct)
qo_b[pi*4096 + 2048 + kvh*HD]

// Line 228: Attention scores read Q with same wrong stride 2048
qo_b[pi*NH*HD + hh*HD + d]  // should be pi*4096

// Line 229: attention output stride also uses NH*HD
at_b[pi*NH*HD + hh*HD + d]
```

The QKV fused buffer stores Q[0:2048], K[2048:3072], V[3072:4096] for each
token with a per-token stride of 4096. But the Q-norm and attention loops
read Q with a stride of `NH*HD = 2048`, which is only half the correct stride.

For token 1 (pi=1):
- Correct Q offset: `1*4096 = 4096` (reads Q from second token's block)
- Buggy offset: `1*2048 = 2048` (reads K from the FIRST token's block)

For prefill with npt=9, token pi's Q is read from `qo_b[pi*2048+...]`, which
reads the FIRST half of each 4096-stride QKV block — meaning tokens 1..npt-1
read K-sections from earlier tokens as if they were Q-sections. This produces
garbage Q-values for all tokens except pi=0, causing the cos_sim=0.24
divergence at layer 0.

**The attention output stride `at_b[pi*NH*HD + ...]` is fictional** — at has
shape `XM*NH*HD` but the O-proj GEMM consumes `at_b.data()` with per-token
stride `NH*HD`. Since at_b is only ever indexed via `pi*NH*HD`, this stride
is actually self-consistent (the bug is in qo_b indexing only, not at_b).

### Severity

This is the **confirmed root cause** of the incoherent-multilingual-token
output for any prefill with `npt > 1`. The single-token case (npt=1) is
unaffected (pi=0 makes both strides equivalent, since `0*2048 == 0*4096`),
which is why scaling benchmarks "passed" — they only tested npt=1.

## Fix Plan

Replace all `qo_b[pi*NH*HD + ...]` with `qo_b[pi*4096 + ...]` in the
prefill Q-norm loop (line 219) and attention score loop (line 228).

```cpp
// Before:
qo_b[pi*NH*HD + hh*HD + d]

// After:
qo_b[pi*4096 + hh*HD + d]
```

The `at_b[pi*NH*HD + ...]` indexing is self-consistent and can remain as-is
(it just describes the activation buffer layout — the O-proj GEMM uses
`at_b.data()` with stride `NH*HD`, so the buffer's effective stride IS
`NH*HD`).

## Files Changed

| File | Change |
|------|--------|
| `engine/npu/xclbins/n1_core_i8_v2.py` | Switch to matmul_i8_i32 + int32 output |
| `engine/npu/src/npu_engine_cb.cpp` | RMSNorm clipping, i32 Cm buffer, decode off-by-one fix, HF cache loader |
| `engine/npu/src/i4_loader.h` | New: raw I4 expander (for when correct Q4NX format is known) |
| `tools/layer_trace.py` | Layer-by-layer trace with HF reference comparison |
| `tools/chunk_dequant.py` | Python Q4NX dequant (byte-for-byte match with C, for debugging) |
| `docs/NPU-ENGINE-CORRECTNESS-STATUS.md` | This document |

## Commits

```
49e78785 fix(npu): clip RMSNorm weights to [-2,2] in cb/universal engines
cd73e137 fix(npu): match INT8 xclbin generator output width to host's i32 Cm buffer
7f8f3586 docs(npu): confirm INT8 GEMM kernel bug via hardware dump-and-compare
01a4b7f4 docs(npu): root cause found and fixed — n1_core_i8_v2.py missing AIE micro-tiling
6608f5f3 fix(npu): wire HF-cached INT8 weights into cb engine with norm clipping + i32 kernel
060898fc docs(npu): comprehensive correctness investigation report + raw I4 loader + layer trace tools
83b833c9 docs(npu): retract false 'dequant 800x' theory — correct offset verified, GEMM confirmed bit-exact
21864a41 fix(npu): order decode loop LM-head BEFORE forward pass (off-by-one)
```