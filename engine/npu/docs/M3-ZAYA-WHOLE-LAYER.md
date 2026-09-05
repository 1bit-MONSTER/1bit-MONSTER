# M3 — Zaya whole-layer on-device decode: measured foundation (2026-09-03)

Status: FOUNDATION MEASURED. Decision record + continuation spec.
Worktree: ~/wt/zaya-m1 (merged main f193319b + local profiler instrumentation).

## Why this supersedes the issue premise (silicon-measured, strixhalo, Zaya1-8B)

Phase profile of the merged-main standalone decode (NPU_FUSED=1, int8 embed
logits, 16 gen tokens, pos>0 stats):

    attn(10 layers) = 10.4 ms/tok   (~0.5 ms/layer, CPU GQA scan)
    fused MoE       = 124.0 ms/tok  (6.2 ms/layer  <- 79% of decode)
    norm+logits     = ~22 ms/tok (7.6 with NPU_EMB_INT8)
    total           = 156 ms/tok (6.4 tok/s)

Fused MoE layer split: pre(router+amax+hdr)=1.61 ms, P1(GUSILU GU+silu->h2,
final_i8_MOE_GUSILU_zaya.xclbin)=3.07 ms, mid=0 (no host silu at pos>0),
P2(D, final_i8_MOE_D_zaya_m8h2.xclbin)=1.52 ms. Total 6.20 ms/layer.

Variant matrix (NPU_FUSED=1): CPU attn 160.4 / +int8 logits 140.5 /
NPU_ATTN=1 236.1 / +NPU_PROJ=1 294.0 ms/tok. Attention-on-NPU is a LOSS:
per-run round trips (quant->launch->wait->readback->dequant with host float
work between ~50 runs/token across >=3 xclbin/hw_context partitions) cannot
be runlist-batched (host in the middle). The #1776 "runlist" framing does not
apply to the standalone per-op path.

## The actual gap (architectural, measured)

Qwen3-0.6B npu-infer runtime (per-ctx whole-layer ELFs, runlist, weights+KV
resident): ~27 ms/tok for 28 layers ≈ 0.96 ms/layer.
Zaya standalone per-op m8 kernels: 7.8 ms/layer. => ~8x gap. The per-ctx
whole-layer-ELF substrate is the fix; Zaya kernels must be AUTHORED (no FLM
Zaya runtime exists to capture from - fastflowlm models:
qwen3/llama/gemma/phi4/... no zaya).

## Building blocks (silicon-verified, separate single-dispatch designs)

1. attn.xclbin: QK^T->softmax->PV, N=512 baked, on-core exp-LUT softmax.
   Host bridge npu_attn_ctx.h correct (corr>=0.999). MAX_SEQ generalizeable.
2. fused GU->SiLU->D: two verified artifacts on main:
   a. final_i8_MOE_GUSILU_zaya.xclbin (GU+silu->h2 writeback; P1) +
      final_i8_MOE_D_zaya_m8h2.xclbin (D phase; P2) — the CURRENT decode
      path (2 launches, h2 via DDR). Per-section GU scales + per-token
      header fold (update_fused_header); per-column D scales (fd_cs).
   b. final_cascade_fused.xclbin (single-launch zero-h2-DMA cascade,
      generator n1_core_fused_gu_silu_d_iron.py, build_iron_cascade.sh).
      SILICON-VERIFIED for the synthetic recipe (fused_ab_probe) and real
      Qwen3 weights (cascade_real_weight_probe, bad=0/8192 maxrel=0) but:
      * artifact on main is N_D=128 (probe geometry) - NOT decode-usable;
      * Zaya real-weight calibration NOT closed (probe is Qwen3-shaped);
      * uniform per-tensor weight scales (no per-section/per-column fold) =>
        accuracy vs the current per-column-scaled path is UNKNOWN.
   Combined-AB contract (Qwen geometry) in engine/fusion/zero_copy/
   npu_cascade_kernel.h (packB_gu_into deriv-inverse tile map, A-tile fill,
   B_d row-major, C2 MxN_D int32 row0=token, S dequant per layer).

## Zaya weight conventions (from decode, validated at corr>=0.998)

gu per expert: gate rows [0,n_ff) then up rows [n_ff,2n_ff), each [n_ff x H]
row-major [p][j]. Decode interleaves: col 2p=gate[p], 2p+1=up[p] (guI
[j][2p/2p+1]). D: dn [expert][out=H][in=n_ff]; D input = silu'd GU pairs
over n_ff. Cascade geometry for Zaya: H=2048, n_ff=2048, N_GU=4096, n_cg=4,
n_k=32 (K_GU=H=2048), K_D=2048, N_D=H=2048, M=8.

## Next experiments (in priority order)

E1 (cheap, host-only): quantify the single-launch gain ceiling — time the h2
DDR round trip by running P1+P2 with h2_bo mapped to a same-location BO vs
fresh; expect <=1.2 ms/layer saving (decision: cascade swap economics).
E2 (xclbin build, ~30 min each on strixhalo iron toolchain): rebuild the
iron cascade at Zaya decode geometry N_D=2048 (build_iron_cascade.sh
N_D=2048 ROWS=4) -> real-weight Zaya probe (port cascade_real_weight_probe
geometry + real l==1 weights) -> close corr/S vs the current per-column path.
E3 (the M3 kernel authoring, weeks): whole-layer per-ctx ELF = co-place
attn + fused MoE in one hw_context + fold host ops on-device
(residual/norm, cca_prep conv_qk/means/L2/RoPE, silu, router), KV device
BO, per-ctx KV-length TXN; then RuntimeLayerEngine-for-Zaya forward() with
one xrt::runlist/token (copy npu-infer runtime_layer.cpp pattern) + parity
vs the CPU-float decode + byte-determinism gate.


## E2 PROGRESS (2026-09-03, executed)
Decode-geometry single-launch cascade BUILT and preserved:
  engine/npu/xclbins/final_cascade_fused_zaya_nd2048.xclbin (243968 B)
  engine/npu/xclbins/insts_cascade_fused_zaya_nd2048.txt
  (build: N_D=2048 ROWS=4 nd_row=512, ~90 s via build_iron_cascade.sh on
   strixhalo iron toolchain; md5 differs from the committed N_D=128 probe
   artifact - confirmed a new build).


## E2 SESSION LOG (2026-09-03, executed — first silicon gate ATTEMPTED)
Done:
- N_D=2048 ROWS=4 single-launch cascade BUILT (243968 B) and preserved as
  final_cascade_fused_zaya_nd2048.xclbin + insts (nd2048).
- Qwen3 control: cascade_real_weight_probe vs Qwen3-0.6B.1bp on THIS box =
  EXACT MATCH (bad=0/8192, maxrel=0) => cascade mechanism, ABI (groups
  1/3/4/5, ins[2] cmd count), and NpuCascadeKernel caller convention are
  all valid on the current driver/kernel.
- E2 hook (NPU_CASCADE_TEST=1) added in zaya_decode.cpp fused l1/pos0 diag:
  packs REAL Zaya layer-1 router-chosen expert (e=7) gate/up/down into the
  combined-AB + B_d (uniform scales), single-launch go(), compares vs the
  decode CPU float ref (cpu_out) + 2-launch (moe_out).
Findings (the open item):
- Cascade launches, full C2 produced (2044/2048 nonzero) BUT corr(rawC2,
  cpu_out) = 0.02 (no correlation); S rel-std 2031 (no scalar S). Block
  corr per 512 and M-row scan (ROWS=4 chunk layout m=0..7) all |corr|<0.05
  => NOT a C2 readback row/chunk permutation.
- ROWS=1 N_D=2048 build FAILS aiecc resource allocation (64 KB C2 = whole
  L1; BUG-009 family) => ROWS=4 is the only N_D=2048 geometry.
- Suspects (unresolved): (a) GU deriv-inverse tile pack for Zaya n_cg=4
  differs from the Qwen-verified n_cg=6 path in a non-row-permutation way;
  (b) the D-phase k-slice / h2 pairing indexing for the Zaya geometry; (c)
  ROWS=4 kernel internals. The Qwen closure was pinned with one-hot layout
  probes (cascade_real_weight_probe modes lay/guread/bread/h2r) - porting
  those to Zaya geometry is the next discriminator.
Next step (E2b): port the qwen probe one-hot layout modes to Zaya geometry
(H=2048, n_ff=2048, n_cg=4, n_k=32) against final_cascade_fused_zaya_nd2048
and pin (i) the GU B-tile read map, (ii) the A-tile read map, (iii) the
silu/D pair indexing, (iv) the ROWS=4 C2 arrangement. Then close corr.

## E2 next-session recipe (real-weight Zaya cascade probe)
Goal: close cascade numerics on REAL Zaya weights (the Qwen3 closure exists;
Zaya does not) and measure single-launch vs 2-launch on a real layer.
1. Probe binary in engine/npu/tests/ (pattern: cascade_real_weight_probe.cpp
   + npu_cascade_kernel.h). Geometry: H=2048 n_ff=2048 N_GU=4096 n_cg=4
   n_k=32 K_D=2048 N_D=2048 M=8.
2. Weight source decision (pick + document): reconstruct TRUE float gu/dn
   for a MoE layer (l=1) expert e from the q4nx manifest (the loader in
   zaya_decode.cpp GETI8/read path - note it feeds the CPU ref at corr
   0.9985; TRUE fp32 exists only for l==1 under FUSED_I4). Verify against
   the decode CPU reference (expert_ffn) so pack/kernel errors are isolated
   from weight-source errors.
3. Pack: uniform per-tensor scales FIRST (the verified cascade convention);
   GU gate/up pair-interleave per the decode (col 2p=gate,2p+1=up from
   guI[j][2p/2p+1]); B_d = dn_T[n_ff x H].
4. Compare: (a) CPU float expert_ffn, (b) current decode 2-launch GUSILU
   output, (c) cascade C2 x S. Report corr(cascade,float), corr(cascade,
   2-launch), per-column S spread (is one scalar S/layer enough? - if spread
   >~2% the cascade needs a per-section fold like update_fused_header).
5. Gate: corr >= 0.999 vs the decode reference at l==1 e==router-chosen.
6. If closed: decode integration = per-(layer,expert) cascade AB BOs packed
   at startup (like fgu_bo), per-token A fill + single launch replacing
   P1+P2+h2, S per (layer,expert); measure ms/layer + token parity vs the
   GUSILU 2-launch decode (expect <=1.2 ms/layer saving per E1 ceiling).

## Environment

- runlist-capable XRT stack: /usr/local/xrt-runlist/lib (LD_LIBRARY_PATH).
- baseline binary: engine/npu/build/npu_engine_m1 (built from merged main;
  contains the [M1prof] phase profiler; do not ship — revert instrumentation
  or keep behind NPU_PROFILE=1 if promoted).
- decode cmd: NPU_FUSED=1 NPU_N_GEN=16 ./engine/npu/build/npu_engine_m1
  ~/models/zaya1-8b-fresh.q4nx (stderr has [perf]/[M1prof]).

---

## int4 vectorized mmul — root cause + verified fix (2026-09-05)

**Symptom:** the vectorized int4 fused GU xclbin (no `I4_SCALAR_C1`) gave
`[MoE L1 fused dbg] corr=-0.055` (wrong) but good perf (P1 GU ~11 ms,
0.6→3.0 tok/s). The scalar int4 was correct (0.9993) but 17× slow.

**Root cause (corrects earlier mis-diagnosis):** the mmul C-store was NOT the
bug. The aie2p backend mis-compiles the *register-array (V[4]) + nested
inner-loop* form (measured "blocks 4-15 unwritten + block-2 row-corruption",
see `cascade_d_i8_i32_slice`). The int4 dequant used exactly that shape:
`int8_t Bb[4][64]` + a nested `jt` loop + `load_v(Bb[jt])`. The `B''` thus
arrived scrambled at the mmul even though the scalar dequant math was correct.

**Fix (`mm_kernel_reference.cc`, `matmul_i8_i32_i4`):** restructured the `B''`
dequant into four named `aie::vector<int8,64>` registers (`B0..B3`, unrolled
`jt`, lambda `deq_b`), feeding them directly to the mmul — no array, no nested
loop. The backend spill fix (llvm-aie `a36c62b9d`: ACC1024 fp-acc → cmh) is
also required so the vectorized mmul path is selectable at all.

**Verified on strixhalo (reproducible ×2):**
| path | corr (`[MoE L1 fused dbg]`) | GU ms/layer | tok/s |
|---|---|---|---|
| int8 single-launch (production) | 0.9985 | ~9 | 9.5–10.3 |
| int4 scalar (correct, old) | 0.9993 | 35.3 | 0.6 |
| **int4 vectorized (fixed)** | **0.999336** | **9.68** | **3.1–3.3** |

- `[C2gate] corr=1.0` (byte-identical), `maxdiff=0.0231`, `npu rms=0.1919`
  vs `cpu 0.1930`.
- Build: `generators/build_fixed_zaya_i4_vec.sh` →
  `xclbins/final_i8_MOE_GUSILU_i4_zaya_VECFIX.xclbin` (production
  `final_i8_MOE_GUSILU_i4_zaya.xclbin` untouched).

**Honest framing:** int4 is an *accuracy*-over-speed trade. The vectorized fix
turns int4 from "correct but 17× slow" into "correct and ~3.3× slower than
int8" (0.9993 vs 0.9985 accuracy, 3.1 vs 10.3 tok/s). It does not beat int8 on
throughput; it buys ~1e-3 of accuracy at ~3× the cost.

---

## cascade (issue #2078 single-launch) — M=1 decode + config deadlock (2026-09-05)

**Design verified working:** the iron cascade (`build_iron_cascade.sh`,
`n1_core_fused_gu_silu_d_iron.py`) runs `final_cascade_fused_zaya_nd2048`
correctly at **M=8 all-ones**: `launch state=4, 7 ms, bad=0/20480` for
**N_D=2560** (N_D_row=640) and the generator-verified N_D=3840. So the
cascade DESIGN and the device are valid — it is NOT fundamentally broken.

**A-layout bug FIXED (`npu_cascade_kernel.h::go`):** for decode (one token)
the A-tiles must be **replicated** across the 8 mmul rows (the mmul reads
A as row-major `(8 rows, K=c)`). The old write `A[i*64+c] = h[ki*64 + i*8 +
c%8]` SPREAD h across rows and used only 8 of the 64 K values (`c%8`),
scrambling the GU C1 → the M=1 `corr=0.12` garbage. Fixed to
`A[i*64+c] = q127(h[ki*64 + c], a_is)` (replicated, full 64-K chunk).

**Config-specific deadlock (blocker for the user's D=2048):** N_D=2048 is
forced into `rows=4` (the array has only 4 compute rows; rows=1/2 give
N_D_row=2048/1024 which exceed the shim BD ≤1023 limit), yielding
**N_D_row=512**, which DEADLOCKS (`launch state=8`, 7–8 s timeout), while
N_D_row=640 (N_D=2560) and N_D_row=960 (N_D=3840) work. rows=8 avoids the
deadlock but is invalid (only 4 array rows) → wrong C2 (bad=8192). The
deadlock is documented in the generator ("launch deadlock state=8 timeout";
memtile-split lock protocol) and is a power-of-2/memtile-boundary quirk at
N_D_row=512. Only N_D=2560/3840 were silicon-verified; N_D=2048 was not.

## cascade N_D=2048 deadlock — FIXED (2026-09-05, RE toolchain = Peano a36c62b9d)
The N_D_row=512 deadlock (`state=8`) was a **Peano-version bug**, not the design or
BD-pool. Building the cascade with the **newer open-source Peano `a36c62b9d`
(clang-22, license-free; `llvm-aie-src/install_aie`)** instead of the old
`c9c5ecb7` (clang-21, iron venv) FIXES it. Verified (reproducible):
- `final_cascade_fused_zaya_nd2048_RE.xclbin` (a36c62b9d, rows=4): **M=8 all-ones
  `state=4, 6-7ms, bad=0/16384`** (was `state=8` deadlock on c9c5ecb7).
- N_D=2560 (a36c62b9d) also `bad=0/20480`.
Build: `generators/build_re_test.sh` (P=`/home/bcloud/llvm-aie-src/install_aie`,
peano flow `--no-xchesscc`). The upstream mlir-aie `c80b88c` BD-field fixes are
NOT needed and conflict with the local NPU2-40 patches.

## cascade M=1 decode — A-layout FIXED (zero-pad), corr 0.578 (open)
The A-layout bug: the design is an **M=8 batch**; for decode (one token) the
token's h belongs in **row 0**, rows 1-7 **ZERO-PADDED** (AM=1), not replicated
or spread. `go()` now writes `A[i*64+c] = (i==0 ? q127(h[ki*64+c]) : 0)`. M=1
decode `corr` went -0.021 → **0.578**. Still <1.0 — the cascade's real-weight
GU→SiLU→D / per-column scale convention for the MoE needs the calibrated S (the
probe passes S=1.0) — open.
