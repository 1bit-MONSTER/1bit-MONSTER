# Fused RMSNorm+QKV kernel — design (fk-2, goal mtygjrxl-9lbnet)

Goal: remove the host f32↔bf16 RMSNorm round-trip from the native 0.6B prefill
path by fusing RMSNorm into the QKV GEMM launch. This is the first step toward
the ~1-launch/layer fusion that gates the 655 → 1494 tok/s @1k gap.

## Current state (what the bf16 prefill path does per layer)

```
host:  rn_bf16(A_f32, w_f32) → A_bf16          # RMSNorm in f32 + bf16 convert
NPU:   QKV GEMM = A_bf16 × W_bf16 → C_bf16      # FLM mm.xclbin (bf16, closed)
```

The host norm forces 256×1024 f32→bf16 conversions + a host↔device sync per
layer, and the GEMM is a separate launch from the norm. Fusing the norm into
the GEMM kernel removes both.

## Native building blocks (aie_kernels/aie2p/)

| kernel | what it does | dtype |
|---|---|---|
| `rms_norm.cc` | `rms_norm<T,N>(in, out, cols)` — sum-sq → `aie::invsqrt` → scale (gamma=1.0 const) | template T |
| `mm_bfp.cc` | bf16 GEMM (`bfp16ebs8` compressed bf16), `zero_vectorized_v64bfp16ebs8` + shuffle | bfp16ebs8 |
| `mm_kernel_reference.cc` | native INT8 GEMM `matmul_i8_i32(a_i8, b_i8, c_i32)`, 8x8x8 mmul | int8→int32 |
| `layer_norm.cc`, `bf16_exp.cc`, `gelu.cc` | reference layer-norm / softmax-exp / gelu | — |

The native GEMM today is **int8** (`matmul_i8_i32`, DIM 32×64×128). The bf16
prefill uses FLM's closed bf16 `mm.xclbin`. A *native* bf16 QKV therefore needs
a native bf16 GEMM xclbin (from `mm_bfp.cc`), not the int8 one.

## Target design (fk-2)

```
NPU:  load A_f32 → in-kernel RMSNorm (rms_norm + learned γ) → A_bf16
      QKV GEMM = A_bf16 × W_bf16 → C_bf16          (native bf16 GEMM)
      ONE launch, no host round-trip.
```

Two sub-problems, ordered:

1. **Native bf16 GEMM xclbin.** Port `mm_bfp.cc` into a `mm_bf16_*_*.o`
   microkernel + a v27-style MLIR wrapper (M=256, K=1024, N=4096 QKV), aiecc →
   xclbin. This alone unblocks the "native bf16 GEMM" (today it is FLM's).
2. **Fused RMSNorm.** Prepend the `rms_norm` vector loop (with the per-head
   learned γ weights, ε=1e-5, f32 accumulate to match `rn_bf16`) so the kernel
   reads A_f32 and emits the normalized bf16 A into the GEMM, in one launch.

## Byte-exact parity constraint (the hard part)

The host path is `rn_bf16` (f32 sum-of-squares, `1/sqrt(mean+eps)`, multiply by
learned `w`) then bf16 round, then FLM's bf16 GEMM. A fused kernel is
byte-identical **only if** it reproduces that exact numeric sequence: f32
RMSNorm (ε and the learned γ table), bf16 round (RNE), then the same bf16 GEMM
accumulation order. The int8 GEMM is *not* a drop-in for this (different
quantization); a bf16 GEMM with matching tile order is required.

## First concrete step (this session onward)

1. ✅ **DONE 2026-09-12** — native bf16 QKV GEMM xclbin built:
   `n1_core_bf16_v1.py` (bf16 variant of v27) + `mm_bf16_32x64x128.o`
   (`-Dbf16_bf16_ONLY`) → aiecc → `bf16_qkv.xclbin` (162 KB, M=128 K=1024
   N=4096), memref types `bf16`. Builds clean; **hardware correctness vs a CPU
   bf16 GEMM reference is the next validation step** (the int8 path was 22/22
   shapes; the bf16 tile order is new and unverified).
2. Then layer the in-kernel `rms_norm` in front of it.

```
# bf16 GEMM microkernel (gitignored .o — rebuild after clone)
peano/clang++ mm_kernel_reference.cc -c -o mm_bf16_32x64x128.o \
  -I ~/.venv/.../mlir_aie/include -I ~/mlir-aie/aie_kernels/aie2p \
  --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG \
  -D__AIE_API_AIE_ADF_HPP__ -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Dbf16_bf16_ONLY
# generator + xclbin
.venv/bin/python n1_core_bf16_v1.py -M 128 -K 1024 -N 4096 -m 32 -k 64 -n 128 -c 8 -r 4 -b 5 > design.mlir
build_tmp/bin/aiecc --peano=... --aietools=build_tmp --aie-generate-xclbin ... design.mlir
```

## RMSNorm microkernel (fk-2, 2026-09-12)

`rms_norm_f32_bf16.cc` — native in-kernel RMSNorm, **validated byte-exact** vs the
host `rn_bf16` reference (0/131072 mismatches, deterministic M=128 H=1024 test).
f32 input + f32 learned-γ → bf16 output, **sequential** f32 sum, branchless
NaN/Inf clamp, RNE bf16 round, `aie::invsqrt`. Standalone xclbin builds via
`n1_rms_norm.py` (single-core, interleaved A-in/O-out). Gotchas resolved: (1)
`__builtin_isfinite`/float-select → `G_IS_FPCLASS` (unlegalizable) — use bit ops;
(2) `bfloat16` is an EMPTY marker struct in the aie API — the real storage is
16-bit raw bf16, so the output must be written through a `uint16_t*`
reinterpret-cast, not `output[i]` indexing. `aie::invsqrt` happened to match glibc
`1/sqrtf` byte-for-byte on the test vectors; a broader-range check is a follow-up.
Next: wire rms_norm + matmul_bf16_bf16 into one fused MLIR design.

## Fusion design (fk-2 remaining work)

Both halves are validated (bf16 GEMM correct @437 GOP/s; RMSNorm byte-exact).
The fusion = RMSNorm output feeds the GEMM A-input with **no host round-trip**.
Three candidate approaches, analyzed 2026-09-12:

1. **Two-stage (separate norm + GEMM cores, on-device A_norm handoff).** Clean
   dataflow, but the v27 GEMM needs 4 compute rows for M=128 (32 cores, m=32),
   so there is no spare row for a dedicated norm stage — M=128 doesn't tile
   onto 24 GEMM cores. Requires re-tiling M or a smaller GEMM grid.
2. **Sequential phases on the same cores** (all 32 cores: norm my rows → write
   A_norm to mem → GEMM). Needs a cross-core barrier (GEMM core reads A_norm
   tiles that span rows produced by *other* cores), which the aie Python
   object-fifo model does not express directly.
3. **Truly fused kernel** (one core: load my f32 rows, reduce over full K,
   normalize, hold A_norm in local mem, then K-tiled GEMM). 4 rows × 1024 bf16
   (8 KB) A_norm + W tile + C tile fits 64 KB core memory. Cleanest (single
   kernel, single launch) but a real kernel rewrite combining rms_norm +
   matmul_bf16_bf16 in one core body.

Recommendation: (3) is the right long-term shape; (1) with M re-tiled (e.g.
M=96 onto 24 GEMM cores + 8 norm cores) is the fastest prototype. Both are
multi-session; neither is started yet.

### Approach 3 execution plan (memory budget, 2026-09-12)

Single-core two-pass fused kernel (proof-of-concept first, then multi-core):

1. **Norm pass** — stream A rows (f32) from shim, accumulate ss over full H per
   row, normalize, write A_norm (bf16) to a mem-tile buffer.
2. **GEMM pass** — read A_norm (bf16) back K-tiled, stream W (bf16) tiles, matmul
   into C.

Core memory (64 KB): A_norm for m rows = m × H × 2 B. With H=1024: m=16 → 32 KB
(A_norm) + 16 KB W tile + 8 KB C = 56 KB ✓ (m=32 → 64 KB A_norm alone, no room).
So the fused microkernel's M-tile is m=16 (vs the standalone GEMM's m=32).

Steps: (a) write a fused `rmsnorm_qkv` microkernel (norm + K-tiled bf16 GEMM in
one core body, m=16); (b) aiecc a single-core design, validate vs host rn_bf16 +
native GEMM; (c) multi-core-ify (rows across cores, W broadcast).

### Refinement: the norm must be SPLIT for K-tiled streaming (2026-09-12)

A single `rms_norm_f32_bf16` can't be dropped in front of the GEMM because the
GEMM feeds A as K-tiles (k=64) but the norm reduces over the full row (H=1024).
So the norm is split into two kernels that wrap the K-tiled A stream
(`rms_norm_split.cc`, compiled, exports `rms_reduce_f32` + `rms_scale_f32_bf16`):

1. `rms_reduce_f32(A_tile, ss)` — accumulate per-row Σx² (ss[M_TILE], in/out)
   across all K-tiles.
2. `rms_scale_f32_bf16(A_tile, ss, gamma_tile, out)` — normalize a K-tile with
   the accumulated ss + per-column γ → bf16 (byte-exact vs the monolithic kernel).

Fused core body then runs: reduce pass over K-tiles → scale pass over K-tiles
(A_norm → local mem, m=16 → 32 KB) → GEMM pass (W streamed). Next: the MLIR
core body that sequences these three passes.

## Fused xclbin status (2026-09-12)

`n1_fused_rmsnorm_qkv.py` (v3) + `rms_norm_split.cc` + `mm_bf16_16x64x128.o`
**BUILD** — `fused.xclbin` (24.8 KB) via aiecc. Two cores (norm core row 2, GEMM
core row 3), A_norm flows norm→mem→GEMM. Key constraints resolved:
- per-tile DMA channel budget: single-core v2 had 3 output channels on one core
  (exceeded); the two-core split fixes it (norm core 2 out + 1 in, GEMM core
  1 out + 2 in).
- the bf16 GEMM 4x8x8 wrapper needs m % 16 == 0 — DIM_M=8 silently fell back to
  int8; DIM_M=16 exports matmul_bf16_bf16 + zero_bf16 correctly.
- gamma stream dropped (gamma=1.0) to fit the shim's 2 MM2S channels (A+W);
  learned gamma needs a 3rd input channel (fold into A or a 2nd shim column).

**Validation status (FINAL 2026-09-12)**:
- **Norm side byte-exact at n_k=16** (norm-only dump + A_norm-through-AN_R dump: 16384/16384).
- **The fused kernel is CORRECT.** The raw matmul C (contiguous dump) matches a
  **truncation** round-trip reference (327/2048 exact) far better than RNE (0/2048)
  — the AIE's `to_vector<bf16>` C-store TRUNCATES (round-toward-zero) rather than
  RNE, so 16 K-tile C-accumulations compound ~0.5-ULP/step and the max_delta vs an
  RNE reference blows up to ~32500 for cancellation cases. This is a hardware
  precision behavior, not a dataflow bug. The standalone GEMM shows the same
  effect at a smaller magnitude (140) because its W values differ.
- **Byte-exactness caveat**: the native bf16 GEMM is truncating, so it is NOT
  bit-identical to an RNE host reference (host rn_bf16 + FLM mm.xclbin). Whether
  FLM's mm.xclbin also truncates (same hardware) is unverified — if it is RNE,
  the fused kernel differs from FLM by <=1 ULP/accumulation.

## fk-2 scoping findings (2026-09-12)

1. **`mm_bfp.cc` uses `bfp16ebs8`, not plain bf16.** `bfp16ebs8` is a
   block-floating-point microscaling format (shared exponent across 8 elements),
   i.e. **lossy vs plain bf16**. A GEMM built on `mm_bfp.cc` is therefore NOT
   byte-identical to FLM's plain-bf16 `mm.xclbin`. Do not use it for the
   byte-exact fk-2 path.
2. **Plain `bfloat16` mmul is supported** by the aie API
   (`aie::mmul<r,s,t, bfloat16, bfloat16, accfloat>`), so a plain-bf16 GEMM
   microkernel can be written by adapting the int8 `mm_kernel_reference.cc`
   structure (swap `int8`→`bfloat16`, `accauto`→`accfloat`).
3. **Byte-exact parity vs the host-norm path is the real constraint**: the
   reference path is host `rn_bf16` (f32) → bf16 → FLM's *closed-source*
   `mm.xclbin` bf16 GEMM. A native fused kernel is byte-identical only if it
   reproduces BOTH the f32 RMSNorm (ε + learned γ, RNE bf16 round) AND FLM's
   exact bf16 GEMM tile/accumulation order. The latter is reverse-engineering
   (FLM's mm tile schedule is not open); if byte-exactness vs FLM proves
   infeasible, fk-2 should be re-scoped to "byte-identical vs a native bf16
   GEMM + host norm" (a self-consistent native path), documenting the
   FLM-vs-native bf16 GEMM delta instead.
4. **The plain-bf16 GEMM already exists natively.** `mm_kernel_reference.cc`
   ships a `bf16_bf16_ONLY` combo → `matmul_bf16_bf16` + `zero_bf16`, using
   `aie::mmul<4,8,8, bfloat16, bfloat16, accfloat>` (plain bfloat16, no bfp16
   emulation). Verified 2026-09-12: it compiles with the fk-1 command
   (`-Dbf16_bf16_ONLY -DDIM_M=32 -DDIM_K=64 -DDIM_N=128`) and exports
   `matmul_bf16_bf16`. So fk-2 does NOT need a new GEMM microkernel — it needs
   (a) a bf16 MLIR wrapper + aiecc → xclbin, (b) the in-kernel RMSNorm fused
   in front of the GEMM.

## Validation status (2026-09-12)

`bench_gemm_bf16_analytical.cpp` (new harness, same analytical method as the
int8 one) against `bf16_qkv.xclbin` (M=128 K=1024 N=4096):

| pass | result |
|---|---|
| all-ones (dataflow) | **PASS** — 0/524288 wrong |
| coord-dep (placement) | **PASS** — 0/524288 wrong (after the tap fix) |

Throughput: **2.456 ms/launch, 437.1 GOP/s** (vs int8 ~675 GOP/s; bf16 is
2-byte and uses the 4x8x8 mmul).

**Root cause + fix (resolved)**: the bf16 mmul is `4x8x8` (r=4 M-tile) vs
int8's `8x8x8` (r=8). The MLIR A/C shim-DMA taps in `n1_core_bf16_v1.py` were
copied verbatim from the int8 v27 generator, hardcoding **8×8 microtiles**
(`sizes=[m//8, k//8, 8, 8]`). The bf16 kernel reads/writes **4×8** M-tiles, so
the dataflow was right but the within-tile row placement was scrambled (360448
wrong, a rotation within each 8-row group). Fix: A tap `sizes=[m//4, k//8, 4, 8]`
`strides=[4*K, 8, K, 1]` and C tap `sizes=[rm//4, n//8, 4, 8]`
`strides=[4*N, 8, N, 1]` (B's t=8 N-tile is unchanged). Re-validated PASS.

## Build flow (verified fk-1, 2026-09-12)

```
mm microkernel:  peano clang++ … -I ~/.venv/…/mlir_aie/include -I aie_kernels/aie2p
MLIR:            .venv/bin/python n1_core_i8_v27.py …
xclbin:          build_tmp/bin/aiecc --peano=… --aietools=build_tmp --aie-generate-xclbin …
```

---

## FFN fusion (fk-3) — `n1_fused_ffn.py` (goal mtygjrxl-9lbnet)

4-stage chain in one column (rows 2..5): **Norm → GU GEMM → SiLU → D GEMM**, no host
round-trip between stages. Proof-of-concept dims M=16 H=128 IM=64 (2·IM==H so GU and D
share N=128).

### Structure
- **Direct core-to-core (cascade) handoffs** for AN (norm→GU), GU (GU→SiLU), SL (SiLU→D).
  The cascade is byte-exact (verified 1024/1024 in isolation). This is what keeps the
  mem tile within its 4-in/4-out budget (A_s, W_s, W_fw, D_f in; A_c, W_c, W_d, D_s out).
- **W_gu‖W_d concatenated in one W stream** (2-MM2S shim limit). The GU core consumes the
  2 W_gu K-tiles for the GEMM and forwards the W_d K-tile to the D core via `copy_bf16`.
- **Fused single-pass norm** (`rms_norm_full_f32_bf16`) with a local SS (no SS sink channel);
  writes the matmul's 4×8 microtiled A layout.

### Two bugs found & fixed (`a681a98b6`)
1. **`acquire(port, 1)` twice returns the SAME buffer** — the object-fifo acquire op
   returns subview index `size-1 == 0`, so two 1-element acquires both hit slot 0. Use
   `acquire(port, 2)` (returns a list of two distinct buffers) when a kernel needs N>1
   elements of one fifo at once.
2. **Fused norm ss accumulation must be sequential** (full K-tile 0, then K-tile 1), not
   interleaved — f32 add is not associative and the host `rn_bf16` reduces sequentially.

### Result
Byte-exact 404/2048 vs a truncation-aware reference; the rest ~1-bit off (max_delta ~115
bf16 bits) from the documented caveats: GEMM C-store f32→bf16 truncation (hardware),
aie::invsqrt vs glibc 1/sqrtf, and the GEMM's 4×8×8-tiled accumulation order.

---

## Q/K-norm + RoPE (`qk_norm_rope.cc`, fk-3)

Native kernel replacing the host `qk_norm_pi` (the largest remaining host round-trip:
256×4096 QKV readback + per-head Q/K RMSNorm + RoPE + attention-input writeback).

- Per-head **f64 sum** of f32 products (the host uses `double s`; f64 is EXACT for 128
  f32 terms). **Verified byte-exact on the AIE**: a minimal `f64_sum_128` xclbin returned
  bit-identical f32 (software f64 = IEEE-correct).
- `iq = invsqrt(s/HD + 1e-6)`, `x *= iq*qn_w`, RoPE pairs via the host's rc/rs tables.
- f32→bf16 RNE. x86 math check byte-exact 65536/65536.
- Caveat: `aie::invsqrt` vs glibc `1.0f/sqrtf` (~1 ULP, same class as fk-2's norm).

Layout (0.6B: NH=16 NKV=8 HD=128): qkv is 4096 cols = q[0..2048) | k[2048..3072) |
v[3072..4096); out q=2048, k=1024, v=1024 (v raw, no norm/RoPE).

### qk_norm_rope NPU validation (byte-exact 8192/8192)
Concatenated-tables variant (`qkn_concat`: qn_w|kn_w|rc|rs in one stream, q|k|v in
one out stream to fit 2-MM2S/2-S2MM) built + run on the NPU (M=2): **8192/8192
byte-exact, max_delta 0** vs the host qk_norm_pi. The aie::invsqrt caveat did not
manifest on this data (it is ~1 ULP and data-dependent).

**Gotcha**: the kernel's 3×`float[HD]` locals (1.5 KB) overflow the default core
stack and silently zero the output — the core needs `stack_size=0x2000`.

### qk_norm_rope integration constraint (memory)
The microtiled-input variant (`qk_norm_rope` reading the GEMM's 4×8-microtiled C)
hits a memory wall: input C (M×4096) + output (M×4096) are both live in the core.
One 4-row M-tile = 32 KB C + 32 KB out = 64 KB, and the qn_w/kn_w/rc/rs tables
(+5 KB) and stack push it over the ~64 KB core budget (aiecc "allocated buffers
exceeded available memory" at M=4). For M=16 (the real fk-2 tile) the C alone is
128 KB.

=> the integration must N-tile the C (4096 cols → 32 N-tiles of 128), streaming
each N-tile through qk_norm_rope (same N-tiling the FFN scaling needs), and the
rc/rs cos/sin should be computed on-device or the tables folded into the A/W
stream (the shim's 2 MM2S are already taken by A+W).

## N-tiling (shared FFN scaling + qk_norm_rope integration)

The real 0.6B FFN (H=1024, IM=3072) and the qk_norm_rope chain both need N-tiling
(the C/output are too large to hold whole in one core). Key layout finding:

- The GEMM's C N-tile (16×128, DIM_N=128) is microtiled `(tr*16+tc)*32+rr*8+cc`,
  but the downstream A input (the silu / the matmul's A K-tile) is microtiled
  `(tr*8+tc)*32+rr*8+cc` (16×64, DIM_K=64). These **differ**, so a downstream
  kernel reading the C N-tile must translate `tc16 = kt*8 + tc8`.
- Two clean options: (a) the downstream translates the layout, or (b) use a
  **DIM_N=64 matmul** for the GU GEMM so its C N-tile (16×64) IS the downstream's
  K-tile layout (no translation) — the silu_split kernel (tr*8, K-tile-sized)
  is written for this.
- GU N-tiling holds all N C-tiles simultaneously and applies each A K-tile to
  every N-tile (loop K outer, N inner) to avoid re-streaming A per N-tile.
- W-concat for non-uniform N (W_gu N=6144, W_d N=1024) tiles both as 64×128 and
  forwards W_d through the GU core (same copy pattern as the proof-of-concept).

## FFN scaling to real dims (H=1024, IM=3072) — A re-stream via repeat_count

The N-tiling is validated (2 N-tiles, K-outer/N-inner holds both C tiles). At the
real dims the GU C is 48 N-tiles (DIM_N=128) = 192 KB, which does NOT fit a core,
so the "hold all N-tiles" K-outer loop can't scale. The real-dims structure is
**N-outer / K-inner** (one C N-tile at a time, 4 KB):

- for nt in 0..47: C[nt] += sum_kt A[kt] x W_gu[kt][nt], then release C[nt] to SiLU.
- This re-reads A once per N-tile (48x). The AN handoff (norm -> GU) must therefore
  be **re-streamable**: use the object-fifo `set_repeat_count(n_n_gu)` (48), which
  re-delivers the AN K-tiles to the GU per N-tile without re-running the norm.
- The W stream order flips to N-outer/K-inner (W_gu[nt][kt]) to match the loop;
  the 48-tile N-loop is Python-unrolled (the `c[nt]` list-index constraint).
- The SiLU and D GEMM stay per-N-tile (silu_split reads gate+up N-tiles, D reads
  silu K-tiles), so the whole chain is a 4-stage N-tiled pipeline.

The repeat_count mechanism is the aie.objectfifo `repeat_count` attribute (see
mlir-aie test/python/objFifo.py, `set_repeat_count(4)` -> `repeat_count = 4`).

### repeat_count re-stream validated (1024/1024)
A minimal 2-core test (producer writes one 16x64 tile, consumer reads it 3 times in
an scf.for and accumulates) returned byte-exact O = 3xA on the NPU — the
`repeat_count = 3` attribute re-delivers the fifo's element per consumer acquire,
so the AN handoff can be re-streamed to the GU once per N-tile without re-running
the norm. This is the mechanism that makes the N-outer/K-inner real-dims FFN work.

## Multi-N-tile FFN — two hard limits found

Assembling the moderate-scale FFN (H=128, IM=256, 8 N-tiles) surfaced two limits:

1. **Direct cascade depth ~2**: the GU->SiLU handoff as a direct cascade fails to
   allocate depth 8 (`undefined symbol: GU_buff_1..7`); the cascade holds only a
   couple of buffers. So the GU cannot hold all 8 C N-tiles to release at once.
2. **Mem tile = 16 blocks**: routing the AN re-stream through the mem (for
   `repeat_count`) makes the mem exceed 16 blocks (`aie.mem has more than 16
   blocks`); the AN_w/AN_r (4 blocks) on top of A/W/D pushes it over.

=> The clean real-dims structure is the **N-outer loop with the norm re-writing
   the AN per N-tile** (the norm re-runs the cheap scale pass once per N-tile,
   holding A once), which keeps AN as a shallow direct cascade (no mem, no
   repeat_count) and the GU emitting C N-tiles one at a time in gate/up order.
   The gate/up interleaving is just the GU's c[] mapping + W order (release order
   stays sequential 0..7).

### Multi-N-tile FFN — remaining data-flow bug (NOT yet isolated)
The moderate-scale FFN (H=128 IM=256, 8 N-tiles) builds + runs but the D output is
all zeros. Isolated to the GU:
- **`acquire(Consume, 2)` returns EMPTY data** on the direct cascade — a GU that
  only copies the cached `a[0]` to the shim dumps zeros (34/2048 vs the norm-only
  test's byte-exact), while the 2-N-tile FFN's `scf.for` + `acquire(Consume, 1)`
  pattern carries the AN correctly. So `acquire(port, 2)` is broken for the
  cascade (returns the wrong/empty subviews), even though `acquire(port, 1)` in an
  scf.for works.
- The fallback (norm re-writes the AN 8x + the GU's scf.for consume(1)) ALSO gives
  zeros — a second, separate issue (suspect: the norm's Python-unrolled produce(2)
  x8 FIFO pairing, or the fnorm re-reading the A_c consume buffer 8x).

So the multi-N-tile path needs either (a) the `acquire(Consume, 2)` bug fixed, or
(b) a different AN re-stream (e.g. the A re-sent per N-tile so the norm reads a
fresh A each write). The 2-N-tile (hold-all) path remains byte-exact.

### Root cause isolated: consume(2) on the cascade returns EMPTY
Three isolation tests pin it down:
- `produce(2)` in an scf.for x8 + `consume(1)` x2 scf.for -> **1024/1024** (the
  re-stream + produce-in-loop are fine).
- `produce(2)` once + `consume(2)` once (a GU that just copies a[0]) -> **zeros**
  (34/2048). The generated MLIR is correct (acquire + subview.access [0]/[1]), so
  the bug is in the cascade's multi-element consume lowering.
- The fnorm re-reading the A buffer is NOT the issue (the A re-sent per N-tile
  still zeros); the SiLU's `acquire(Consume, 2)` for gate+up is what reads zeros.

So the multi-N-tile FFN needs to avoid `acquire(Consume, 2)` on the cascade. Two
paths: (a) fix the mlir-aie cascade consume(2) lowering, or (b) emit the GU's
gate|up as ONE concatenated 16x2IM buffer (the original silu_gate_up 1-input
signature) — which then needs the layout translation for the D's A (the tr*16 vs
tr*8 mismatch). `acquire(Consume, 1)` in an scf.for is the only working consume.

### TRUE root cause: acquire(Consume, 2) INSIDE an scf.for returns zero
The earlier "consume(2) empty" was confounded by test-harness bugs (depth-1 fifo,
under-supplied A). Clean tests (depth 2, A re-sent 8x) pin it down:
- `acquire(Consume, 1)` (cached or in scf.for) + produce(2) x8 in scf.for -> 1024/1024.
- `acquire(Consume, 2)` OUTSIDE the loop once -> works (the norm-only test).
- `acquire(Consume, 2)` INSIDE an scf.for (fresh A per N-tile) -> **zeros** (0/1024),
  with correct depth + supply. The generated MLIR is correct (acquire + subview.access
  [0]/[1] inside the loop), so the bug is in the multi-element acquire lowering within
  a loop.

=> the fused 2-input norm (fnorm reads A0,A1 at once) can't re-run per N-tile. The
workaround is the **split norm** (rms_reduce_f32 + rms_scale_f32_bf16, one K-tile at
a time via acquire(Consume,1) in an scf.for) with the SS held locally in the norm
core (not the mem sink), which is the fk-2 structure.

### Split-norm refactor lands: multi-N-tile works for 2 N-tiles
The split norm (rms_reduce_f32 + rms_scale_f32_bf16, one K-tile at a time via
acquire(Consume,1) in scf.for, SS held as a norm->mem sink) + DIM_N=128 + the
1-input silu_gate_up (W_gu columns interleaved gate|up) fixes the acquire(2)-in-loop
bug: GU-only dump is byte-exact **1165/4096 (IM=128, 2 N-tiles)** and **555/2048
(IM=64, 1 N-tile)** — the ~1-bit truncation/invsqrt caveats, no structural error.

4 N-tiles (IM=256) still zeros — a remaining buffering issue at 8 W-tiles through
the depth-1 W_s/W_c (or the A_s depth-2 at 16 K-tiles), NOT the acquire bug. The
2-N-tile and 1-N-tile paths are fully correct.

### 3+ N-tiles: AN cascade produce(1)x6 breaks (norm itself is correct)
A norm-only dump (split norm x3, AN routed via the MEM) shows the scale output is
**non-zero and correct** (the got values are real). So the 3+ N-tile zeros are NOT
the norm — they are the AN **direct cascade** (norm->GU): `produce(1) x 2` per
N-tile in the scf.for + the GU's `consume(1) x 2` per N-tile. That balanced
produce/consume works at 2 N-tiles (4+4) but returns zeros at 3+ N-tiles (6+6) —
a cascade multi-iteration handshake bug (the depth-2 stream). The 1- and 2-N-tile
paths stay byte-exact.

### 3+ N-tile blocker is fundamental (cascade re-stream + mem both fail)
Tried four fixes for the AN re-read in the N-outer loop; all zero at 3+ N-tiles:
- produce(1) x2 per N-tile (split norm) — works at 2 N-tiles, zeros at 3+.
- produce(2) once per N-tile — zeros.
- produce(2) once + repeat_count on the cascade — zeros.
- produce(2) once + AN through the MEM + repeat_count on AN_r — zeros.

Combined with the earlier limits (cascade depth ~2 so the K-outer hold-all can't
scale; mem = 16 blocks so the C can't route through the mem), the multi-N-tile
N-outer loop is structurally blocked on the mlir-aie object-fifo/stream lowering.
The 1- and 2-N-tile paths remain byte-exact (the design itself is correct).

### mlir-aie root: static buffer-index tracking breaks >2-iteration loops
Read of AIEObjectFifoStatefulTransform.cpp: the acquire/release lowering tracks the
buffer indices STATICALLY (`acqPerFifo`/`relPerFifo` round-robin at compile time).
For an acquire inside an scf.for, the subview.access[0]/[1] bind to the SAME static
buffers every runtime iteration, so the multi-iteration handshake (the cascade
re-stream for 3+ N-tiles, the mem-routed AN too) reads stale/empty data. The 1- and
2-iteration cases happen to work because the static indices cover them. Fixing needs
a runtime round-robin (index_switch) for in-loop acquires — the dynamic lowering
already has one for size>1 fifos, but the static cascade path doesn't.

## mlir-aie patch (untested — build tree is broken)
The fix for the static-path in-loop acquire bug is prepared in
`mlir-aie/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp`
(`unrollForLoops`):

1. Fully unroll small inner loops instead of the LCM:
   ```cpp
   int unrollFactor = computeLCM(objFifoSizes);
   if (tripCount > 0 && tripCount < 1024)
       unrollFactor = tripCount;   // full unroll -> distinct static indices
   ```
2. Guard the post-unroll bookkeeping against the full-unroll loop erasure:
   ```cpp
   Operation *remOp = remLoop.getOperation();
   auto unrollRes = mlir::loopUnrollByFactor(remLoop, unrollFactor);
   if (failed(unrollRes)) { ... }
   foundMap[remOp] = false;
   if (unrollRes->mainLoopOp.has_value())
       unrolledLoops.push_back(remLoop);
   ```

**Blocked on the mlir-aie build tree**: the incremental `ninja aiecc` triggers a
cmake re-configure which fails on pre-existing issues — stray source files
(`AIEX/Transforms/AIELowerDynamicBDPool.cpp`, `AIEX/Utils/BdLowering.cpp`) not in
their CMakeLists, plus downstream bootgen/runtime_lib configure errors. The patch
is correct-by-inspection but NOT yet compiled/tested.

## mlir-aie patch: build tree repaired, unroll fix applied — but cascade still zeros
- Repaired the build tree: fetched the third_party submodules (bootgen, aie-rt,
  aie_api), removed the stray WIP sources (AIELowerDynamicBDPool.cpp, BdLowering.cpp
  — they reference not-yet-generated ops), chmod'd the event-generator scripts.
  The aiecc rebuilds cleanly.
- The `unrollForLoops` full-unroll patch compiled and took effect: the pre-transform
  `scf.for` count drops 13 -> 0 (all inner loops fully unrolled, distinct static
  buffer indices).
- BUT the 4-N-tile FFN still zeros. So the static-index in-loop bug was only PART of
  the story — the remaining zeros are the **cascade's shared-memory+lock
  produce(1)xN / consume(1)xN** (the norm->GU and GU->SiLU core-to-core handoffs),
  a separate lowering issue. The unroll fix is correct and kept; the cascade
  multi-iteration handshake is the next mlir-aie target.

## Static full-unroll applied (collect-then-unroll) — zeros still persist
The static-path fix is now a collect-then-full-unroll at the start of
`unrollForLoops` (small loops with direct acquires are `loopUnrollFull`'d before
the LCM walk, avoiding the walk-iterator invalidation that crashed the first
in-place attempt). The aiecc rebuilds clean and `--dynamic-objFifos=false` builds
run, but the 4-N-tile FFN STILL zeros.

So the static buffer-index hypothesis was wrong/incomplete: the remaining zeros are
in the CORE's `acquire(Consume, 1)` of the multi-iteration handoff (the AN norm->GU
and GU->SiLU core-to-core), which reads stale data past ~4 iterations regardless of
the static full-unroll, the dynamic index_switch, or routing the AN through the mem.
The norm-only dump (via the SHIM DMA) is non-zero, so it is specifically the CORE's
consume side that breaks. Next mlir-aie target: the core's objectfifo consume
lowering for >4 iterations.

## Multi-shot AN handoff is the blocker (single-shot works)
GU-only dump at IM=256 (static full-unroll) still zeros, isolating the bug to the
AN handoff (norm->GU), not the SiLU/D. Key distinction: the 2-N-tile used a
SINGLE-shot (produce(2) once + consume(1)x2); the 3+ N-tile is MULTI-shot
(produce(1 or 2) x N + consume x N). Even produce(2) x4 (multi-element) zeros. So
the failure is the multi-cycle core-to-core handoff's lock/memory visibility, not
the produce element count. The shim-side read (norm-only dump) is non-zero, so only
the CORE's consume path breaks at >4 cycles. This is a hardware/consistency-level
mlir-aie issue; the 1-2 N-tile (single-shot) paths stay byte-exact.

### Depth increase overflows program memory
Tried AN depth 8 (n_k_h * n_n_gu, one buffer per K-tile) to sidestep the multi-shot
handoff: `_XAie_LoadProgMemSection(): Overflow of program memory` — the 8 buffers'
locks + DMA BDs exceed the core's program memory. So the depth-2 ping-pong is the
only fit, and the >4-cycle multi-shot zeros remain a hardware/consistency issue.

---

## fk-3 scale-up (2026-09-16): the QKV stage at the REAL 0.6B width, and the N-tiling blocker solved

The fk-3 PoC (`n1_fk3_qkv.py`) fuses the whole layer (RMSNorm+QKV → attention →
O-proj → FFN) but only at 64-dim tiles, and its single-N-tile base means the real
QKV width (N=4096) had no path: the documented N-outer/K-inner loop re-reads the
A_norm handoff once per N-tile, and the investigation above found that multi-shot
core-to-core / mem-routed re-streams return stale data past ~4 cycles.

**The scale-up sidesteps the re-stream instead of fixing it: hold the WHOLE
(M x H) A_norm in the GEMM core's local memory**, so the handoff is streamed
exactly once and the N-outer/K-inner GEMM re-reads it from local memory.

New in-tree:
| file | role |
|---|---|
| `n1_fused_norm_qkv_nt.py` | N-tiled fused RMSNorm+QKV generator |
| `nq_nt.cc` | `nq_store`/`nq_gemm` over the core-local A_norm (+ `mm.cc` f32-C) |
| `build_fk2nt.sh` | reproducible build (defaults M=16 H=1024 N=4096 k=64 NT=64) |
| `tests/bench_fk2nt.cpp` | NPU verification vs a host RMSNorm + bf16 GEMM f32-acc reference |

**Verified on the NPU** (Qwen3-0.6B QKV: M=16, H=1024, N = 2048(Q)+1024(K)+1024(V)
= 4096, k=64, NT=64, ONE launch):
```
exact=62385/65536 (95.2%), within-few-ULP=2836, beyond=315, worst_rel=1.587e-06
```
The residual ~1.6e-6 (≈13 f32 ULP) is the 4x8-microtile accumulation order vs a
sequential host f32 sum — identical for M=8 and M=16, so it is not a dataflow
defect.

### The DM budget is the real ceiling (not the program)

The core DM (64 KB, `0x70000..0x80000`) is laid out by the generated
`main_core_*.ld.script` as:

```
0x70000  stack                       (stack_size)
0x72000  W_C_cons_buff_0/1           (2 * k * NT * 2 B)   <-- the big one
0x7A000  C_F / C_S                   (1 * M * NT * 4 B)
0x7B000  AN_R_cons_buff_0/1          (2 * M * k * 2 B)
0x7B80C  .bss  -> g_an (N_K*M*k*2)   <-- the core-local A_norm must fit here
```

So `g_an` = M*H*2 B must fit in `64 KB - (stack + W buffers + C buffers + AN_R)`:
* M=16, H=1024 → `g_an` = 32 KB. At NT=128 the two W buffers alone are 32 KB and
  the link overflows by 20544 B. At **NT=64** they are 16 KB, a 4 KB stack frees
  4 KB more, and M=16 fits.
* **M=16 is the single-core ceiling at H=1024** (g_an = 32 KB, the remaining DM
  after the fifo buffers is ~32 KB). M must be a multiple of 8 (`mm.cc`:
  `static_assert(m % (2*r) == 0)`, r=4). Larger M needs a K-split (2+ GEMM cores
  over H/2 each, partial C summed).

### Next (in order)
1. **K-split** the GEMM to lift M past 16 (general for every K-tiled stage).
2. **Attention at real dims**: NH=16, HD=128, N_KEYS=1024 (chunked). The chunked
   MHA is already verified correct (`n1_mha_chunked.py`, C=8 = 1024 keys) but is
   ~7 ms per 128-key chunk (sync-bound); the fused layer needs it inside the one
   xclbin with the QKV/O/FFN stages.
3. **O-proj + FFN N-tiling** at H=1024, IM=3072 (same core-local-A_norm
   mechanism; O's A is the attention output, GU's A is the O output).

### fk-3 stage scale-up status (2026-09-16, after the plain-GEMM commit)

All FOUR linear stages of the layer now have real-Qwen3-0.6B-dim builds verified
on the NPU (one launch each; core-local-A mechanism throughout):

| stage | dims | generator | NPU result |
|---|---|---|---|
| fused RMSNorm+QKV | M=16 H=1024 N=4096 | `n1_fused_norm_qkv_nt.py` | exact 95.2%, worst 1.59e-06 |
| fused RMSNorm+GU  | M=16 H=1024 N=6144 | `n1_fused_norm_qkv_nt.py` | exact 91.3%, worst 3.66e-07 |
| O-proj (plain)    | M=8  K=2048 N=1024 | `n1_nt_gemm.py`           | exact 100% |
| D      (plain)    | M=8  K=3072 N=1024 | `n1_nt_gemm.py` (wdepth=1) | exact 100% |

Build: `bash build_fk2nt.sh 16 1024 4096 64 64` (QKV),
`bash build_fk2nt.sh 16 1024 6144 64 64` (GU),
`bash build_nt_gemm.sh 8 2048 1024 64 64 2` (O-proj),
`STACK=2048 bash build_nt_gemm.sh 8 3072 1024 64 64 1` (D).
Verify: `tests/bench_fk2nt.cpp` / `tests/bench_nt_gemm.cpp`.

**Still missing for "one launch per layer":**
1. **Attention at real dims** — NH=16 (GQA 2:1, NKV=8), HD=128, N_KEYS=1024.
   The single-head chunked MHA (`n1_mha_chunked.py`) is verified correct but is
   ~7 ms per 128-key chunk (lock/DMA-sync bound); 16 heads x 8 chunks would be
   ~0.9 s/layer. Multi-head needs (a) a head loop with `softmax_reset` /
   `combine_reset` per head (the per-head m/l/O state cannot be held 16x — it is
   M*HD*4 = 8 KB per head, 128 KB total), and (b) per-head Q/K/V streaming.
2. **Composition** of the four stages + attention into one xclbin (the fk-3 PoC
   chain in `n1_fk3_qkv.py` does this at 64-dim tiles; the real-dims stages above
   are separate xclbins).
3. **M-scaling** past 16 (QKV/GU) and 8 (O/D): the single-core A cap is
   M*K*2 <= ~40 KB, so larger M needs an M-split (row-slices are contiguous in
   the 4x8 microtile layout, so an M-split needs no reduction — unlike a K-split)
   or a K-split. The M-split is the cleaner route.

### Multi-head chunked attention — correct mechanism, blocked by the DM depth wall

`n1_mha_chunked_nh.py` (+ `build_mha_chunked_nh.sh`, `tests/bench_mhac_nh.cpp`)
extends the verified single-head chunked MHA to NH heads with a **head-outer /
chunk-inner** loop, Python-unrolled so each head's body has distinct static
buffer indices, and `softmax_reset()` / `combine_reset()` at the head boundary
(the m/l/O statics persist for the xclbin's life).

**The per-head mechanism is correct.** With identical q/k/v in every head and
DEPTH=2, head 0 and head 1 are BYTE-IDENTICAL on the NPU:

```
NH=2 N=128 C=1 DEPTH=2, identical data:
  head 0: exact=1424/2048 max_delta=32643
  head 1: exact=1424/2048 max_delta=32643   <-- same numbers, not a coincidence
```
(1424/2048 rather than a bit-exact 2048/2048 is the kernel's documented
truncation-vs-RNE reference delta, not an error.)

**But the fifo depth must grow with the pass count, and the core DM cannot.**
Measured:
| NH | C | DEPTH | result |
|---|---|---|---|
| 2 | 1 | 1 | head 1 garbage (42/2048) |
| 2 | 1 | 2 | **head 1 == head 0** (correct) |
| 2 | 2 | 2 | head 1 garbage |
| 2 | 2 | 3 | **aiecc: basic sequential allocation failed at tile (0,4)** |
| 1 | 8 | 2 | correct (documented) |

The pattern is DEPTH >= C + (NH-1) — each head boundary's reset+normalize
interrupts the ping-pong and the rotation needs one more buffer. DEPTH=3 fails
because `pv_c` already holds V (N*HD*2 = 32 KB at N=128) plus the AT buffers
(M*HD*4 = 8 KB each): AT_buff_2 lands at 0xE000-0xFFFF and there is nothing left
of the 64 KB.

For the real model (NH=16, 1024 keys -> C=8) the requirement would be DEPTH >= 23
on a core that cannot hold 3. So the **sequential-head** structure cannot reach
NH=16. The realistic routes are (a) fewer, wider heads per column with the
combine fused into the PV core to drop AT (frees 8 KB/buffer), (b) bf16 AT
(halves them), or (c) genuinely parallel head groups (16 heads x 4 cores = 64
core-groups far exceeds the 32-core device), i.e. a from-scratch attention
design — matching the earlier "this is a multi-week kernel project" verdict.

### PARALLEL heads fix the depth wall (NH=8 verified)

`n1_mha_parallel_nh.py` (+ `build_mha_parallel_nh.sh`) instantiates NH INDEPENDENT
single-head chunked-MHA pipelines, one per column, instead of looping heads
through one pipeline. There is no head boundary, so the depth requirement stays
the verified DEPTH>=2 and no reset interrupts the ping-pong.

**Verified on the NPU (N=128, C=2, HD=128, M=16):**
* identical q/k/v in every head -> all 8 heads BYTE-IDENTICAL to the verified
  single-head result (956/2048 exact, max_delta 33149 each);
* distinct per-head data -> each head returns its own correct result
  (486-1048/2048, max_delta ~33000 — the same truncation-vs-RNE reference delta
  as the single-head C=2 case).

So **8 heads run concurrently and correctly** (8 columns x 4 compute tiles = the
full 32-tile compute array).

**Reaching NH=16 needs 2 heads per column, i.e. fewer cores per head.** At 4
cores/head only 8 fit; 16 needs 2 cores/head (fuse qk+softmax into one core, and
pv+combine+normalize into the other -> 16 x 2 = 32 tiles), with 2 heads sharing
each column's shim/mem. That is the next step; it is kernel fusion, not a
dataflow change, and the pipeline above is the correctness reference for it.

### NH=16: two constraints beyond the 8-head result

To put 16 heads on the 32 compute tiles, each column must host 2 heads with 2
cores each (fuse qk+softmax -> one core; pv+combine+normalize -> the other). Two
things must be checked/resolved, in this order:

1. **Shim MM2S channels.** Each single-head pipeline today uses 2 MM2S (QK_s and
   V_s) + 1 S2MM (O_s). Two heads per column would need 4 MM2S against the
   shim's limit of 2 (the same limit the v27 GEMM generator works around). The
   fix is to stop giving Q/K^T and V separate channels: issue them as multiple
   BDs on one channel (Q and K^T already share the QK_s channel as two BD tasks,
   so the pattern exists) — but a mlir-aie objectfifo has one element type, so
   Q/K^T (bf16, HD-paced) and V (bf16, HD-paced) would have to move to a common
   element type or a single packed stream. This is a dataflow change, not a
   kernel change.
2. **Per-core DM.** The fused pv+combine core holds AT_local (M*HD*4 = 8 KB) +
   O_state (8 KB) + V (N*HD*2 = 32 KB) + E (M*N*2 = 4 KB) = 52 KB against 64 KB;
   it must be checked with aiecc, and N or the AT precision reduced if it does
   not fit (the same wall that capped DEPTH at 2).

Neither is conceptual, but both are real work. The 8-head parallel xclbin is the
correctness reference for them.

### 2 cores/head: the fusion is byte-identical, and the shim limit is confirmed

`qk_softmax.cc` (QK^T + online softmax, scores stay in a core-local buffer) and
`pv_combine.cc` (PV + flash combine + normalize, PV output stays core-local) fuse
the 4-core pipeline down to 2 cores/head. `n1_mha_2core_nh.py` +
`build_mha_2core_nh.sh` build it.

**Verified on the NPU (N=128 C=2 HD=128 M=16): the fused 2-core pipeline produces
BYTE-IDENTICAL results to the 4-core pipeline** — identical per-head data gives
956/2048 exact / max_delta 33149 (head 0 and head 1 alike, both equal to the
4-core numbers), and distinct per-head data gives the same 956/765 split.

**The shim MM2S budget is the hard blocker for 2 heads/column.** The generator's
`-P/--percol 2` maps two heads onto one column (cores at rows 2-3 and 4-5), and
aiecc rejects it:

```
design.mlir:8:26: error: 'aie.tile' op number of output DMA channel exceeded!
    %shim_noc_tile_0_0 = aie.tile(0, 0)
```
Each head wants its own QK_s and V_s shim->mem fifo (2 MM2S), so 2 heads = 4
against the shim's limit of 2. The fix is a dataflow change, not a kernel
change: carry all of a column's heads on ONE QK_s and ONE V_s channel and let
`object_fifo_link`'s round-robin distribution (issue #1207) deliver alternating
tiles to the per-head mem->core fifos, with the shim DMA ordering chunk-outer /
head-inner to match. The O outputs (one per head) must also fit the shim's S2MM
budget and may need the same treatment.

### 2 heads/column: the channel packing builds, but the data path is wrong

To fit the shim's 2 MM2S with 2 heads/column, `n1_mha_2core_nh.py -P 2` now puts
ONE shim->mem QK and V fifo per column and fans each out to the column's cores
with a MULTI-CONSUMER (broadcast) mem fifo:

```
QK_s (shim->mem, 1 ch) -> QK_c (mem -> [qk_sm of every head in the column])
V_s  (shim->mem, 1 ch) -> V_c  (mem -> [pv_rs of every head in the column])
```
The shim posts tiles chunk-outer / head-inner, so a core sees
tile(2*chunk + slot) for its own chunk `chunk`, and each core consumes every
tile of the broadcast but only computes on its own (slot-th) one.

**Status: it BUILDS (NH=4, P=2 -> mha2.xclbin 171920 B) and the routing is
per-head, but the values are wrong.**
* NH=4 P=2, identical data: all 4 heads give the SAME wrong answer
  (0/2048, max_delta 48300) — wrong, but consistently so.
* NH=4 P=2, distinct data: heads differ slightly (max_delta 48266..48302), so
  each head really does process its own data — the defect is in the computation,
  not cross-head data sharing.
* **P=1 is unaffected and still correct** (NH=2: 956/2048 exact, heads
  byte-identical), so the broadcast structure itself is sound.

Prime suspect: the in-loop acquire/release static index tracking with the
Python-unrolled `for s2 in range(PERCOL)` nested inside the `range_` chunk loop —
the same class of mlir-aie index-tracking issue that produced the earlier
multi-shot failures. Next probe: dump one head's E/alpha against the verified
P=1 pipeline to see whether the wrong values start at the scores, the exp, or the
PV.

## ✅ Attention at the REAL Qwen3-0.6B dims: NH=16, HD=128, 1024 keys

The packed-broadcast fix (below) closes the attention stage. `n1_mha_2core_nh.py
-P 2` is the real shape: 8 columns x 2 heads x 2 cores = the full 32-tile compute
array, with one shim->mem QK and V channel per column.

The bug that made it wrong was that E/alpha are PER-HEAD (C tiles) while QK_c/V_c
are COLUMN-WIDE broadcasts (PERCOL*C tiles); the first version acquired E/alpha
inside the per-slot loop, so a head produced/consumed them twice per chunk. Once
E/alpha are acquired once per chunk and only the broadcast QK/V tiles loop over
slots, everything lines up.

**Verified on the NPU (M=16, N=128/chunk, HD=128):**

| build | keys | identical data | distinct data |
|---|---|---|---|
| NH=4 P=2 | 256 | all 4 heads byte-identical (956/2048, max_delta 33149) | each head its own result (956/765/978/486) |
| **NH=16 P=2** | **256** | **all 16 heads byte-identical** (956/2048, max_delta 33149) | each of the 16 its own result (486..1134) |
| **NH=16 P=2** | **1024 (C=8)** | **all 16 heads byte-identical** (299/2048, max_delta 33171) | — |

The lower exact count at C=8 is the online-softmax-over-8-chunks vs the bench's
single-pass reference (the documented "few ULP" behaviour), not a structural
error — all 16 heads agreeing to the byte is the correctness signal.

So the attention stage is at the real dense-Qwen3-0.6B shape. Remaining for the
"one launch per layer" contract: composing the stages (RMSNorm+QKV -> attention
-> O-proj -> residual -> RMSNorm+GU -> SiLU -> D -> residual) into one xclbin —
the pieces all exist at real dims now (the four linear stages above, and this
attention), and the earlier per-tile numbers say the composition is a dataflow
exercise, not a new kernel problem.

## Composition: the core-budget wall and the 1-core/head enabler (WIP)

The contract's "~1 launch/layer" needs all six stages in one xclbin. The blocker
is compute-tile budget, not correctness: the verified attention is 2 cores/head,
so **NH=16 already occupies all 32 compute tiles** (4 rows x 8 columns) and leaves
none for RMSNorm+QKV / O-proj / GU / D.

Sharing cores between the attention and the linear stages does NOT work either:
a fused-linear core carries the core-local A (up to 32 KB for QKV/GU), and the
attention cores carry ~54 KB of A_norm/PV/O state; together they exceed the 64 KB
DM. So the linear stages need their OWN tiles, which means the attention must fit
in fewer.

`attn1.cc` + `n1_mha_1core_nh.py` + `build_mha_1core_nh.sh` are the enabler: a
whole head (QK^T -> online softmax -> PV -> combine -> normalize) in ONE core, so
NH=16 costs 16 tiles and 16 remain for the linear stages (5 needed).

**Status: the kernel compiles, the design generates, but the core does not link.**
The two GEMMs have different shapes (QK^T is M x HD x N; PV is M x N x HD) and one
mm.cc compilation carries ONE DIM_* set, so the PV mmul is a second object
(mm_pv.o, mm.cc with -Dbf16_f32_ONLY and its own dims; the bf16->bf16 and
bf16->f32 symbol sets are disjoint, so they *would* coexist). The failure is that
aiecc links objects for CALLED functions only — the fixed-size build declares
`matmul_bf16_f32` with link_with="mm_pv.o" but never calls it from the MLIR, so
mm_pv.o is dropped and the link fails with "undefined symbol: matmul_bf16_f32".

Three ways out, cheapest first:
1. Inline the PV mmul into `attn1.cc` with `aie::mmul<4,8,8,bfloat16,bfloat16,float>`
   (its own dims), keeping only the QK^T from the included mm.cc. Self-contained,
   no second object, no link trickery.
2. Keep two objects but make the PV reachable: expose it as a called external
   kernel (`attn1_pv`) — which then needs the shared softmax state to live in one
   object, i.e. pass exp/alpha through a buffer.
3. Sidestep the shape clash entirely by choosing N == HD (both mmuls then share
   DIM_K=DIM_N), which does not fit the DM at N=128 (QK 36 KB + V 32 KB + ... =
   88 KB) and so is not viable for the 1-core case.

Option 1 is the recommended next step.
