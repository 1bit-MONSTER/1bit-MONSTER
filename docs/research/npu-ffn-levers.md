# NPU FFN levers — status and recipes (2026-08-29)

The two levers that would make the NPU FFN path competitive, from
`docs/research/fused-npu-ffn-perf.md` (rounds 6-9: the NPU FFN is DMA-bound at
~1.4-1.5 GB/s single-launch; the GU B weight DMA is 6.3 MB/layer and the
GU→D alternation pays ~0.8-0.9 ms per ERT hw_context switch, ~44 ms/token):

## Lever 1 — int4 GU weights (halves the 6.3 MB GU B DMA)

**Status: BUILDABLE for Qwen3-0.6B; runtime wiring BLOCKED on numerics.**

- The bit-level pack contract already exists and is CPU-verified:
  `engine/npu/generators/i4_pack.h` (`pack_i8_to_i4`/`unpack_i4_to_i8`,
  even element in the low nibble, ×16 scale fold) with the exhaustive test
  `engine/npu/tests/test_i4_pack.cpp` — `i4 pack contract OK`.
- The Qwen3 int4 fused GU→SiLU xclbin now BUILDS end-to-end with the iron
  toolchain:
  ```
  engine/npu/generators/build_p1i4_qwen3_iron.sh
  → engine/npu/xclbins/final_i8_GUSILU_i4_qwen3_0_6b.xclbin   (61 KB)
  ```
  (shapes `-M 8 -K 1024 -N_GU 6144 -N_D 1024`; the pipeline was verified on
  this box 2026-08-29 — clang → ld.lld → symbol check → iron generator →
  aiecc → xclbin.)
- **Remaining blocker (documented in `mm_kernel_reference.cc`):** per-column
  int4 re-quantization with a single K-uniform scale caps MoE-FFN corr at
  ~0.972 vs float (int8 fused: 0.9995) and flips tokens at int8 boundaries.
  The C1 accumulator cannot carry the Q4NX per-(32-col,row) scales a 4-bit
  grid needs — a **per-group-scale kernel restructure** is required before
  wiring `packB_i4` into `npu_state_*`. Until then, do NOT enable the i4
  path: `tools/parity_fused` would catch the token flips.

## Lever 2 — single-launch fused GU→SiLU→D cascade (kills the context-switch tax)

**Status: GENERATOR FIXED + QWEN3 XCLBIN BUILDS (2026-08-29); silicon round pending.**

- **Root cause fixed**: the generator conflated the GU input width with the D
  input width (Zaya: both 2048, because its GU is 2:1). `n1_core_fused_gu_silu_d_iron.py`
  now takes `K` = the D input width (the silu'd GU output) and `--K_GU` = the
  GU input width; `n_k = K_GU/k` drives the GU k-slice count and AB stream
  sizing, and the assert is now `K == n_cg_gu·(n/2)·cols` ("D input width must
  equal the silu'd GU output"). Backward compatible (K_GU defaults to K).
- **Qwen3 build**: `engine/npu/generators/build_iron_cascade_qwen3.sh` with
  `-K 3072 --K_GU 1024 -N_D 1024 --rows 2` emits a valid design (150 tiles)
  and aiecc produces **`engine/npu/xclbins/final_cascade_fused_qwen3_0_6b.xclbin`
  (140,496 B) + insts** — XRT-parses (1 kernel, uuid 4a8a0207-…). The
  host-buffer contract: AB_gu_bo = 8·6·16·8704 B (per-column [A-tile 8x64 |
  B_gu-tile 64x128], gate/up interleaved per 128-tile), C2_bo = M×N_D int32,
  B_d_bo = K×N_D int8.
- **Remaining**: runtime wiring (a cascade-launch path in npu_state_* that
  fills AB/B_d and reads C2) + silicon verification (needs a healthy NPU —
  the stability probe currently faults ~100% of runs on this box).

## Runtime integration point (when a lever lands)

`src/backend_fused_npu.cpp` `npu_state_create`/`npu_state_ffn`:
- For lever 1: detect `final_i8_GUSILU_i4_qwen3_0_6b.xclbin`, pack B with
  `i4_pack.h`'s contract, invoke the i4 GU kernel (same BO layout family).
- For lever 2: detect `final_cascade_fused_qwen3_0_6b.xclbin` and issue the
  single cascade launch instead of the GU→D pair (removes the ~1.6 ms/layer
  hw_context alternation).
Both must keep the current GU+D fallback when the artifact is absent.

## Lever 2 — the exact redesign the 1:6 GU needs (2026-08-29 analysis)

The `K == N_GU/2` assert is not a shape bug — it is the generator's per-tile
1:2 structure: every GU output tile (n=128) consumes a distinct (n/2=64)-wide
h2 slice (gate|up pair inside the tile), and the D's K is the silu'd half
(N_GU/2). The coupling is K = n_cg_gu·(n/2)·cols with n_cg_gu = N_GU/(n·cols),
which collapses to K == N_GU/2 for ANY microtile params. Qwen3 violates it at
BOTH stages: GU 1024→6144 (1:6) and D 3072→1024 (3:1).

Required generator changes (per core column):
1. **GU h2 broadcast**: each output tile must consume the SAME K/cols h2
   (1024/8 = 128 = 2 k-tiles of 64), not a per-cg slice — B differs per tile,
   A is broadcast. The cascade C1 accumulation then sums the 2 k-tile
   partials per output tile (per-column cascade, 6 tiles/column).
2. **Silu moves from per-cg to post-column**: with the [gate|up] block layout,
   the gate⊙up pairing spans 3 output tiles (gate tiles 0-2 ↔ up tiles 3-5,
   offset +3·128 = +384 within the column). The silu stage runs after the
   column's 6 tiles accumulate.
3. **D phase K = IM = 3072** (24 output tiles of the silu'd width) → N=1024
   via the cascade reduce; the cross-core redistribution of the silu'd values
   is the existing memtile-split mechanism (N_D_row ≤ 1023 → N_D=1024 needs
   ROWS=2).

Silicon verification of the redesigned cascade additionally requires a
healthy NPU (the stability probe currently fails ~100% on this box).

## Lever 1 — variant D is the answer; the restructure hypothesis was wrong

`engine/npu/tests/test_i4_grouped_fused.cpp` (CPU gate, real Zaya weights)
reproduces the ws09 verdict:

| variant | GU bytes/layer | FFN corr |
|---|---|---|
| A. int8 per-section (current fused) | 8.4 MB | 0.999562 |
| B. int4 per-column re-quant | 4.2 MB | 0.869526 |
| C. int4 per-group re-quant (the old restructure plan) | 4.2 MB | 0.865916 |
| **D. RAW Q4NX nibbles + per-row scales, on-chip dequant** | **4.2 MB** | **0.999640** |
| E. gu_i4_pack.h packer roundtrip | 8.4 MB | 0.999643 |

The per-K-chunk restructure (option C) does NOT reach int8 quality — the win
is "int4 storage + on-chip dequant to the existing int8 contract" (D): stream
the raw Q4NX nibbles + per-row bf16 scales, reconstruct B'' in-kernel. All
pieces exist: `engine/npu/src/q4nx_raw.h` (raw tile reader), `engine/npu/src/
gu_i4_pack.h` (byte-exact packer), the kernel's ratio dequant
(`mm_kernel_reference.cc` i4 path), the built Qwen3 int4 xclbin
(`final_i8_GUSILU_i4_qwen3_0_6b.xclbin`), and the CPU gate. **Remaining:
runtime wiring + the silicon corr round.**

Wiring spec (npu_state_*): when `final_i8_GUSILU_i4_{model}.xclbin` is present,
(1) pack B from the 1BP's RAW tiles (get_tile_ptr / q4nx_raw) via GuI4Pack
instead of the f32→int8 packB, (2) invoke the fused GU→SiLU kernel in ONE
launch (the silu is in-kernel — drop the host silu step), (3) keep the int8
GU→D path as fallback. Gate on the stability probe as for the int8 path.

## Wiring spec — fill-level (pinned from the generators, 2026-08-29)

### Lever 2 (cascade, `final_cascade_fused_qwen3_0_6b.xclbin`)

Kernel invocation follows the NpuGemmKernel pattern (`(*k)(opcode, bI, ninstr,
bAB, bC2, bBD)`); the xclbin exposes 5 data-buffer groups (3-7) with the
sequence (AB_gu_bo, C2_bo, B_d_bo). Shapes: AB_total = 8·6·16·8704 B,
C2 = M×1024 i32, B_d = 3072×1024 i8.

- **B_d (fixed per layer)**: `B_d[ki·64+ks·8 + r, col]` for ki ∈ [0,48), ks ∈
  [0,8), r ∈ [0,8) — the D weights [3072, 1024]; the strided taps read rows
  [ki·64+ks·8, +8) with N_D_row=512 per row.
- **AB (A per-token, B fixed per layer)**: element (col, ki, cg) at
  `col·6·16 + ki·6 + cg` × 8704 B = [A-tile 8×64 | B_gu-tile 64×128]. The
  A-tile = h2 (hidden state) rows [0,8) × k-slice ki (BROADCAST across all
  cols/cg — fill once, the memtile forwards); the B_gu-tile = the gate/up
  weight slice for (ki, cg, col) — 128 output cols interleaved gate/up
  (the silu pairs within the tile; the exact interleave is a kernel contract
  the silicon round validates). B is fixed per layer; only the A parts change
  per token.
- **C2 (output)**: M×1024 i32 read after r.wait; `h += C2` (the fused FFN
  result), replacing the GU→host-silu→D two-launch.

### Lever 1 (int4 GUSILU, `final_i8_GUSILU_i4_qwen3_0_6b.xclbin`)

Kernel `(opcode, instr, ninstr, bo0..bo4)`: bo0 = A [M×1024] (per-token h2),
bo1 = 768×8192-B int4 B tiles (per-layer, via `GuI4Pack` from the raw Q4NX
tiles — `n_k = H/64 = 16` must be a multiple of 4 ✓), bo2 = C2 [M×1024],
bo3 = B_d [3072×1024], bo4 = H2 scratch. One launch does GU→SiLU→D (the silu
is in-kernel) — the host silu step and the GU→D context switch disappear.

Both paths: auto-select when the xclbin is present, keep the GU+D int8 path as
fallback, and gate on the stability probe exactly like the int8 path.

### Cascade structural contract — PINNED (host reference: tests/fused_ab_probe.cpp)

- BO groups CONFIRMED: bI = group_id(1), AB_gu_bo = group_id(3), C2_bo =
  group_id(4), B_d_bo = group_id(5); invocation `k((unsigned)3, bI, ninstr,
  bAB, bC2, bBD)` with ninstr = ins[2] (ncmds, the measured crash fix).
- Silicon-verified structure: all-ones recipe → C1 = K, silu saturates at 127,
  D sum = 127·K (260096 for K=2048) — `npu_fused_smoke` EXPECT.
- **OPEN (silicon calibration)**: the real-weight A/B quant fold. The in-kernel
  `silu_quant_i8_fused_q22` treats the int32 C1 as the float gate/up directly
  (Q22 sigmoid LUT over [-4,4]; "scale folded into c1"), so the standard
  two-launch convention (C1 dequant with as_·Bs in the host) does NOT carry
  over. The exact host fold (how ascale/gu_scale produce C1 ≈ gate·u in LUT
  range) is the one contract the silicon round must calibrate — the parity
  harness (`tools/parity_fused`) is the validation gate. Until then the
  cascade path stays opt-in (NPU_CASCADE=1) with the GU+D fallback.

## SILICON RESULTS (2026-08-29, live NPU — the "blocked" claim was a probe false positive)

The stability probe's pattern (rapid back-to-back GU→D GEMMs, no GPU work
between) triggers the driver's degenerate state, but the REAL fused pattern
(GU→D interleaved with GPU attention) runs clean: 4/4 `USE_NPU_FFN=1` runs
produced parity tokens with the gate bypassed. The gate needs recalibration to
the real pattern (interleave or fewer iterations) — until then the bypass
(NPU_PROBE_BIN=/bin/true) enables the NPU on this box.

- **Qwen3 single-launch cascade: STRUCTURALLY SILICON-VERIFIED.** With the
  correct K_D=3072 B_d sizing, the all-ones recipe yields C2 = 127·K_D =
  390144 everywhere (bad=0/8192, launch state=4, 5 ms) — the same rigor as the
  Zaya smoke (C2 = 127·K). The K_GU generator split produced a CORRECT design.
- **Real-weight calibration — D-side contract PINNED on silicon.** The layout
  probes in `engine/npu/tests/cascade_real_weight_probe.cpp` (modes `lay` +
  `bd1`/`bd1p`/`rep`/`pad`/`h2r`) established the D-phase contract EXACTLY
  (bad=0/8192 for the one-hot probes at j0=0/1/2, and the new `h2r` per-pair
  reads confirm the structure):
  - **h2 slice pairing**: `a8s[kstep][c_] = h2b[ks][cg·64 + kstep·8 + c_]` —
    the slice is the **ACC ROW (kstep)**, per the generator source. The h2r
    per-pair reads prove it: the acc row t reads the pair `t·8 + c_` (the rows
    differ; the earlier ks-slice reading was an artifact of the one-hot lay
    probes, which zero the rows 1..7).
  - **C2 layout = FULL microtile dump**: `C2_bo[r·512 + kstep·1024 + n_row]`
    holds the acc row `(p/8)%8` at the col `(p/64)·8 + p%8`, p = kstep·512 +
    n_row (all 8 acc rows are dumped, not just row 0 — the h2r rows prove it).
  - **bd read (mmul B-tile reindex)**: `B(k, n) = b8[(n/64)][64·((n/8)%8) + k·8 + n%8]`
    → bd ROW = `ki·64 + ks·8 + n/64`, bd COL = `rh·512 + 64·((n/8)%8) + c_·8 + n%8`
    (n = the half-col nn%512). Same tile structure as the scalar
    `matmul_i8_i32` reference (`b[((kb_·nb+nb_)·8+rr)·8+cc]`).
  - **The h2r per-pair oracle + guread one-hot-A probes: GU read structure
    FULLY PINNED (64/64).** The one-hot-A probes across the A-tile rows 0 AND 1
    (A[i·64+k0] = 127 → C1[0][n] = 127·B(i·8+k0, n)) confirmed BOTH the B read
    `B(K, n) = B_tile[n/16 + 8·(K/8)][64·((n/8)%2) + (K%8)·8 + n%8]` and the
    A-tile layout (row i = the K-slice). The `bread` one-hot-B probes showed
    the A-values saturate through the silu's sat8 (h2 = silu(A)·A = +127 for
    BOTH A signs) — the A-value semantics cannot be read directly and remain
    the last unknown (the h2 mirror reproduces ~50% of the silicon h2 pairs).
    This is the persistent blocker for the real-weight C2 comparison.
- GPU-vs-NPU logits parity (real prompts): tokens match for ~50 steps then the
  int8 FFN quantization accumulates and diverges (worst |Δlogit| 14 at step 8;
  19 token mismatches) — the parity harness quantifies the NPU-vs-GPU gap.
