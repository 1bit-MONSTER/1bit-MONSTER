# MAX Recipes → 1bit Engine Port Guide (gfx1151 / Strix Halo)

Extracted from MAX 26.5.0 production kernels (disassembled AMDGPU ELFs) and verified
on this box. Target: 1bit engine HIP/C++ backends. All instructions verified with
amdclang 23 + gfx1151 + llvm-objdump.

## Platform facts (verified)

| Fact | Value | Source |
|---|---|---|
| bf16 WMMA | `v_wmma_f32_16x16x16_bf16` — 16×16×16, A/B = 16×bf16 (8 VGPRs), C/D = 8×f32 | MAX gemm_kernel_rdna |
| fp8 WMMA | **does not exist on gfx1151** (`v_wmma_..._fp8` invalid; fp8 WMMA is RDNA4/gfx12) | llvm-mc |
| MFMA | CDNA-only — rocwmma.hpp targets MI-series, not RDNA3.5 | rocwmma headers |
| Wave32 | all MAX kernels `wavefront_size32`, `workgroup_processor_mode 1` | kernel metadata |
| LDS | no `shared_mem_bytes` quirk in C++ — hipModuleLaunchKernel reads the kernel descriptor | — |
| DPP | `v_add_f32_dpp`/`v_mov_b32_dpp` quad_perm — synthesis of warp reductions | MAX gemv/rmsnorm |

## Mechanism: inline asm (bulletproof, version-proof)

```c
// bf16 WMMA 16x16x16 (MAX prefill GEMM core)
typedef unsigned short __v16bf __attribute__((ext_vector_type(16)));
typedef float __v8f __attribute__((ext_vector_type(8)));
asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0"
             : "+v"(D) : "v"(A16), "v"(B16));   // A,B: 16xbf16; D: 8xf32

// DPP butterfly step (MAX decode GEMV / RMSNorm reduction)
asm volatile("v_add_f32_dpp %0, %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1"
             : "+v"(v) : "v"(v));
```
(__builtin_amdgcn_* names vary by clang; rocwmma is CDNA-only. Inline asm is the contract.)

## Recipe 1 — Prefill GEMM (`gemm_kernel_rdna_bfloat16`, 7520B)

Per k-iteration, per thread:
```
ds_load_b128 (prefetch next A-slice from LDS)      ; software pipeline
v_wmma_f32_16x16x16_bf16 D0, A1, B1, D0
  s_waitcnt lgkmcnt(4)                              ; graded waits overlap the 4 WMMAs
v_wmma_f32_16x16x16_bf16 D1, A1, B2, D1
  s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_bf16 D2, A2, B1, D2
v_wmma_f32_16x16x16_bf16 D3, A2, B2, D3
s_barrier ×2 per k-iteration                        ; LDS double-buffer swap
```
Structure: **2×2 WMMA accumulator tile per lane, LDS double-buffered, graded lgkmcnt pipeline**.

## Recipe 2 — Decode GEMV (`gemv_kernel_vector_bf16_..._8_`, 760B)

Per lane, per 8-element chunk:
```
global_load_b128 (A row slice)  global_load_b128 (W col slice)   ; 8xbf16 each
v_cvt bf16→f32 (8 lanes)
v_dual_fmac_f32 ×4  (+1 scalar)                                   ; 8 FMAs, dual-issue
```
After the K loop — warp reduction (no LDS!):
```
v_add_f32_dpp quad_perm:[1,0,3,2]   ; butterfly step 1
v_add_f32_dpp quad_perm:[2,3,0,1]   ; step 2
v_add_f32_dpp row_half_mirror       ; step 3
v_add_f32_dpp row_ror:8             ; step 4
ds_bpermute_b32 (gather final to all lanes)
```
Note: decode uses cvt+FMA, NOT v_dot — 8-wide, dual-issued.

## Recipe 3 — RMSNorm (`rms_norm_gpu_warp_tiling`, 760B)

Same 4-step DPP butterfly, 128-bit IO, and the precision knob:
```
inv_rms:  MAX GPU uses v_rsq_f32 (fast, ~1-2 ULP)      ← for the engine: rsq
          MAX CPU uses IEEE vsqrtss+vdivss             ← Mojo std.math also IEEE
          (measured: exp() fuses to v_exp_f32 with 1/ln2 const in Mojo stdlib)
```

## Recipe 4 — FlashAttention (`mha_depth128_bf16_nqh16_nkvh8`, 41KB)

```
QK^T via v_wmma_f32_16x16x16_bf16 (128× WMMA in kernel)
online softmax: v_dual_max (running max) + rescale v_dual_mul + 66× v_exp_f32
PV  via WMMA
causal mask: v_dual_sub + v_dual_cndmask
GQA: 16 Q-heads / 8 KV-heads; 116× ds_load_b128 tile staging; 14 barriers
```

## Recipe 5 — Fused fp8 (the 1bit advantage, PROVEN)

MAX does **two passes**: `linalg_fp8_quantization_naive` (fp8 bytes → bit decode
v_and_b16/v_lshlrev + v_mul_f32 scale → bf16) then the bf16 WMMA GEMM.
**Proven in Mojo on this box: fuse it — one kernel, fp8 bytes in → decode →
scale → WMMA staging.** fp8 e4m3→bf16 is mantissa<<4, exp+120 — near-free.
fp8's value on gfx1151 = half weight bandwidth (decode path is bandwidth-bound),
math is bf16 WMMA either way.

## Verification workflow

Mojo prototype → `mojo build --target-accelerator gfx1151 --emit asm` → read the
.amdgcn → port to HIP inline asm → verify with llvm-objdump + a CPU reference.
Tool: `/opt/rocm-therock/.../_rocm_sdk_core/lib/llvm/bin/llvm-objdump --triple=amdgcn-amd-amdhsa-gfx1151`.
