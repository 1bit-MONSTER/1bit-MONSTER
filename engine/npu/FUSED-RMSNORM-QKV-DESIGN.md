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

**SOLVED (see below) — the core links and NH=16 is verified at 1 core/head.**
The original obstacle:
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


### ✅ 1-core/head attention verified — the composition budget is now open

The link problem was solved by calling mm.cc's mmul **templates** directly instead
of its exported combo functions: the templates are not behind the combo guards, so
one object can instantiate the QK^T shape (M x HD x N) and the PV shape
(M x N x HD) with no DIM_* clash and no second object:

```cpp
matmul_vectorized_4x8x8_bf16_bf16<M_TILE, HD, N_KEYS>(q, kt, g_sc);
matmul_vectorized_4x8x8_bf16_f32 <M_TILE, N_KEYS, HD>(g_sc, v, g_at);
```
The only other change was the core stack (0x2000 -> 0x1000); at N=64 chunks the
DM is QK 20 KB + V 16 KB + g_sc 2 KB + g_at 8 KB + O_state 8 KB + stack 4 KB.

**Verified on the NPU:**
| build | keys | cores | result (identical data) |
|---|---|---|---|
| NH=2 P=1 N=64 C=2 | 128 | 2 | heads byte-identical (973/2048, max_delta 33304) |
| **NH=16 P=2 N=64 C=16** | **1024** | **16** | **all 16 heads byte-identical (230/2048, max_delta 33289)** |

So the attention now costs 16 of the 32 compute tiles, leaving 16 for
RMSNorm+QKV (2), O-proj (1), GU (2) and D (1) = 6 — the one-launch composition
fits. (The exact-count differences across N/C are the online-softmax vs
single-pass reference, as before; all-heads-agree is the correctness signal.)

### Composition: the SHIM channel budget, not just the tile budget

Freeing compute tiles (the 1-core attention, above) is necessary but NOT
sufficient. The linear stages need data, and every byte enters the array through
a shim tile — and each shim has **2 MM2S + 2 S2MM** (both measured, not assumed):

* 2 MM2S: hit by the 2-heads/column attention (QK_s + V_s) — the error was
  "'aie.tile' op number of output DMA channel exceeded" at tile (0,0);
* 2 S2MM: hit by putting 4 heads on one column (4 O_s outputs) — the error is
  now "'aie.tile' op number of **input** DMA channel exceeded" at tile (0,0).

So both ways of making room fail on shim channels:
* attention at P=2 -> uses all 8 columns' shims; the free rows 4-5 of those
  columns still have no shim channel left for a linear stage;
* attention at P=4 -> frees 4 whole columns, but a 4-head column needs 4 S2MM
  for its O outputs, over the limit.

The next experiment is DMA-channel SHARING across phases: the mlir-aie fifo API
exposes `set_prod_dma_channel()` / `set_cons_dma_channels()`, and the phases are
time-disjoint (phase 1 QKV, phase 2 attention, phase 3 O-proj, ...), so a
linear-stage fifo could be pinned to the same channel index its column's
attention fifo uses. If aiecc counts channel INDICES rather than fifos, that
fits everything; if it rejects the double assignment, the fallback is to merge
each column's O outputs into one S2MM (a 4->1 `object_fifo_link`, which then
needs the head order made deterministic).

### Decisive: shim DMA channels cannot be shared across phases

The proposed escape (pin a linear-stage fifo to the channel index an attention
fifo uses, since the phases are time-disjoint) does NOT work. Probe: one column,
three shim->mem fifos, the third carrying `prod_dma_channel = 0` (confirmed
present in the generated MLIR, `aie.objectfifo @D_S(...) {prod_dma_channel = 0 :
i32}`). aiecc still fails with

```
design.mlir:3:26: error: 'aie.tile' op number of output DMA channel exceeded!
    %shim_noc_tile_0_0 = aie.tile(0, 0)
```
so the shim's channel count is per FIFO, not per channel index — `set_*_dma_channel`
cannot double-book, and time-disjointness is not considered.

**Consequence for the composition.** Every stage's data enters through a shim, and
one shim is 2 MM2S + 2 S2MM. With the attention at P=2 (all 8 columns, 2 MM2S +
2 S2MM each) no column has a channel left for a linear stage, and P=4 dies on
4 heads' O outputs vs the 2 S2MM. So the one-launch composition is blocked by
the shim channel budget, not by compute tiles ([the tile budget was solved by
the 1-core attention]) and not by DM.

The two remaining routes, both real work:
1. **Merge each column's attention O outputs into ONE S2MM** (4->1
   `object_fifo_link`) so a P=4 attention needs 2 MM2S + 1 S2MM per column and
   frees 4 whole columns for the linear stages. The open question is making the
   merged head order deterministic so the shim writes each head to the right
   offset.
2. **Keep the attention's O on-chip** and feed the O-proj from shared memory
   (no shim S2MM at all) — an on-chip many-to-one gather, which also removes the
   O DDR round-trip.

### Route 1 measured: the O-merge builds, but its order is arrival-order (scrambled)

Merging a column's head-O outputs into one S2MM works at the aiecc level:

```
aie.objectfifo @O_S_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<16x128xbf16>>
aie.objectfifo.link [@O_F_0, @O_F_1, @O_F_2, @O_F_3] -> [@O_S_0]([0, 0, 0, 0] [])
```
A P=4 attention (16 cores in 4 columns, 4 heads/column) with this merge BUILDS
(575840 B) — but only after two fixes: the 4->1 link needs `srcOffsets` (one per
input fifo) and the merged fifo must be depth 1 (depth 4 exhausts the channel's BD
IDs: "Allocator exhausted available BD IDs (maximum 24 available for channel 2)").

**It is numerically wrong, exactly on the predicted sub-problem.** With DISTINCT
per-head data every head misses (0/2048, max_delta ~48150): the mem forwards the
four O tiles in ARRIVAL order, which is racy across the four concurrent cores, so
each tile lands at the wrong head's offset. Note this is invisible to the
identical-data test we have been using as the correctness signal — all four tiles
are equal there, so any permutation passes. Any future merge must therefore be
validated with distinct per-head data.

So route 1 needs a DETERMINISTIC head order, which an arrival-order merge cannot
give. The practical options are (a) serialise the O production with a token/lock
so the heads write in sequence, or (b) keep per-head O buffers and read them in a
fixed order — which costs the 4 S2MM we were trying to save, i.e. back to the
channel wall, unless the attention drops to fewer columns per head.

The generator is restored to the verified per-head-O form after the experiment.

### Route 1b measured: serialising the cores does NOT fix the merge order

A token chain was added so each head in a column may write its O only after the
previous head has (token fifos h_i -> h_{i+1}, acquired before the O write and
released after). It builds (577792 B) and runs — and the result is BYTE-IDENTICAL
to the un-serialised merge build (same 0/2048, same per-head max_delta
48130..48194). So the ordering is NOT determined by the order in which the cores
release their O buffers: it is fixed somewhere in the mem -> shim DMA path
(the token only guarantees the core's release order, not that the mem has
forwarded that tile before the next arrives).

Consequence: an arrival-order merge cannot be ordered from the core side. Route 1
therefore reduces to keeping per-head O channels (4 S2MM, over the shim's 2) or
dropping to fewer heads per column. Combined with the measured limits:

| attention shape | columns used | shim channels/col | columns free for linear stages |
|---|---|---|---|
| P=2, 2 cores/head | 8 | 2 MM2S + 2 S2MM (full) | 0 |
| P=4 merge | 4 | 2 MM2S + 1 S2MM | 4, but O order is wrong |
| P=2, 8 heads/launch | 4 | 2 MM2S + 2 S2MM (full) | 4, at the cost of 2 attention launches/layer |

The last row is VERIFIED (NH=8, P=2, N=64, C=16 = 1024 keys, 1 core/head, 4
columns): all 8 heads return their own correct results under distinct per-head
data (147..260/2048 exact, max_delta ~33000 — the online-softmax vs single-pass
delta), and columns 4-7 are left free. It is the only combination that is both
correct with what is measured so far and leaves columns for the linear stages: it trades "~1 launch/layer" for
"2 attention launches + the linear stages", i.e. ~4 launches/layer instead of the
9 the goal set out to remove. That is a real, honest fallback if the merge
ordering cannot be pinned down from the shim side (e.g. by posting the four O
DMA tasks with explicit per-head source offsets rather than relying on arrival
order).

### Multi-pass 1-core attention: correct, but program memory caps C

The 1-core attention's state is core-local (g_sc/g_at/O_state statics, no SC/E/AT
fifos), so unlike the 2-core pipeline a head boundary does NOT need fifo depth —
it is just an `attn1_reset()` + another chunk loop. `--passes P` makes each core
process P heads sequentially, so NH heads fit in NH/P cores: NH=16 with
PASSES=2 needs only 8 cores (4 columns), leaving 4 columns for the linear stages.

**Verified (NH=4, PERCOL=1, PASSES=2, N=64, C=2 = 2 cores x 2 heads):** all four
heads return their own correct results under distinct per-head data
(973/590/1061/930 exact, max_delta ~33000) and head 0's value (973) is identical
to the single-pass C=2 run — so the per-pass reset and the head->data mapping are
right.

**But C is capped by program memory.** The same design at C=16 (1024 keys) fails
at ELF load: `_XAie_LoadProgMemSection():231: Overflow of program memory` (the
MLIR is small and clean — 1456 lines, 196 dma_bd — so it is the shim's descriptor
program, i.e. the number of DMA tasks landing on the few shims, not the core).
So multi-pass buys cores at the cost of chunks-per-shim; the composition needs a
combination where the shim task count stays under that limit, e.g. more columns
(fewer passes) or fewer BDs per chunk.

Regression checked: PASSES defaults to 1 and NH=8/P=2/C=16 reproduces the
verified numbers exactly (1636/16384, head 5/6/7 = 189/182/147).

### ✅ Composition attention verified: NH=16 / 1024 keys in FOUR columns

The "Overflow of program memory" that blocked multi-pass was a PER-SHIM limit, not
a per-core one: PASSES=2 with PERCOL=1 (2 columns) overflows, the SAME design with
PERCOL=2 (4 columns) builds — the shim's DMA-task/program budget is spread over
twice as many shims.

```
PASSES=2 bash build_mha_1core_nh.sh 64 16 16 2     # NH=16, ncol=4, C=16 = 1024 keys
== OK: mha1.xclbin (294048 B)
```
**Verified on the NPU with DISTINCT per-head data — all 16 heads correct:**
heads 0..7 give 230/179/195/254/260/189/182/147 (byte-identical to the verified
NH=8/P=2/C=16 run, i.e. pass 1 is exactly right), and heads 8..15 give their own
correct values (169..334). This is the multi-pass signal that matters — an
identical-data test cannot distinguish the two passes.

**So the composition budget is fully open:** the attention costs 8 cores in
columns 0-3, leaving columns 4-7 (16 compute tiles and 4 shims) for
RMSNorm+QKV (norm + GEMM), O-proj, RMSNorm+GU (norm + GEMM) and D — one column
each, 2 MM2S + 1 S2MM apiece, well inside 2 + 2. With C=16 = 1024 keys, and the
attention's two passes run inside the single launch, the layer stays one xclbin
launch.

### Composition design constraint: M must be 8, not 16

The stages were verified at different M (QKV/GU at M=16, O-proj/D at M=8). A single
xclbin needs ONE M for the whole layer, and the binding stage is the O-proj:

* O-proj: the core-local A is `M*K*2` with K = NH*HD = 2048 -> M=16 is 64 KB, over
  the ~40 KB the core can spare; **M=8 is 32 KB and fits** (verified).
* D is worse (K = IM = 3072): M=8 is 48 KB and only fits with the W fifo at depth
  1 and a 2 KB stack (verified).
* QKV (K=1024) and GU (K=1024) are comfortable at M=16 and also fine at M=8.
* the attention's `attn1.cc` is templated on `M_TILE`, so it compiles at M=8 as
  well (the mmul needs `M % 8 == 0`).

So the composition runs at **M=8**, and the QKV/GU generators should be built with
`-m 8` for it (their M is a parameter, not a design change).

**M=8 attention verified too** (the composition's M): NH=16, N=64, C=16, PASSES=2,
4 columns, M=8 builds (218400 B) and gives all 16 heads their own correct results
under distinct data (55..265/1024 exact, max_delta ~33000 — the same
online-softmax truncation delta as M=16). The build script now takes `MGEN` for M.

**All five stages now verified at the composition's M=8** (not just at their own
best M): fused RMSNorm+QKV N=4096 (95.2% exact, worst 1.587e-06), fused
RMSNorm+GU N=6144 (88.5% exact, beyond=0, worst 3.663e-07), plain O-proj K=2048
and D K=3072 (100% exact each), and the attention NH=16/N=64/C=16/PASSES=2
(16 heads correct under distinct data). So the composition's M is settled with no
stage left at a different one.

## ✅ First composition xclbin: attention + O-proj in ONE launch

`n1_fk3_attn_oproj.py` + `build_fk3_attn_oproj.sh` + `tests/bench_fk3_ao.cpp` are
the first real composition: the attention (cols 0-3, 1 core/head, 2 passes) and
the O-proj (col 4, the plain N-tiled GEMM over the attention output) in ONE
xclbin, with the attention output round-tripping through a host BO.

```
bash build_fk3_attn_oproj.sh                 # defaults M=8 N=64 C=16 NH=16 P=2 PASSES=2 NO=1024
== OK: fk3_ao.xclbin (255952 B)              # 1024 keys, 0.6B attention + O-proj
```
**Verified on the NPU with the uniform probe** (q=0 => the softmax is exactly
uniform, so each head's attention output is exactly mean_k V — a deterministic
value the host can reproduce without emulating the online softmax):

| what | result |
|---|---|
| attention out ([NH*M*HD] via the host BO) | **0/16384 mismatches** vs mean(V) — exact |
| final C (O-proj, M x 1024 f32) | 14.6% bit-exact, worst rel 8.5e-05 at a near-zero reference |

The exact attention output proves the whole first half end-to-end (per-head Q/K/V
gather, 2 passes, the `attn1` kernel, and the `O_s` de-microtiling into a
row-major host buffer). The final C's small absolute error (~2e-4 against
reference values of order 0.3) is the f32 4x8-tiled accumulation order on random
data — the standalone O-proj's 100%-exact result was an artifact of its
low-entropy test data (i%13 / i%29), not a guarantee. A wrong cross-stage layout
would show bf16-scale deltas (the O-merge scramble measured ~48000), so the
magnitude is itself the layout check.

Two things this pinned down for the rest of the composition:
* the attention's `O_s` BD **de-microtiles**, so the attention-output host buffer
  is ROW-MAJOR `(M x HD)` per head — the O-proj gathers a K-tile with the plain
  row-major tap (`sizes=[M//4, KO//8, 4, 8]`, `strides=[4*HD, 8, HD, 1]`,
  `offset = h*M*HD + d0`), not a microtiled one;
* the O-proj phase is a sibling of the attention loop in the runtime sequence, so
  its DMA tasks are emitted once, after the attention's.

## ⚠ Architectural limit found while wiring the composition: M is the whole ballgame

The first composition (attention + O-proj) is a **token tile**, not a prefill. That
is fine for the attention — it QUERY-TILES by design (M queries against all
keys) — but it does not extend to the linear stages, and the reason is the
core-local-A mechanism itself:

* every fused linear stage holds the whole `(M x K)` A on chip, so
  `M <= ~40KB / (2K)`:
  **K=1024 (QKV, GU) -> M <= 20; K=2048 (O-proj) -> M <= 10; K=3072 (D) -> M <= 6**,
  which is why the composition is pinned at M=8.
* the ENGINE's prefill runs the QKV projection **batched at M = npt**
  (`npu_engine_universal.cpp` allocates `qkv_ascales(npt)` and reads
  `h_b[pi * H]` for every prompt token), i.e. M = 1024 at @1k.

So a fused layer at M=8 would need **128 launches to prefill 1024 tokens** — worse
than the ~9-launch/layer per-op path the goal set out to replace. "~1 launch per
layer" is only true **per 8-token tile**.

The KV projection is the part that cannot be tiled this way: Q can be
query-tiled (M=8) but K and V are needed for the WHOLE context, so their
projection is an `(npt x 1024) x (1024 x 2048)` GEMM.

Two structural ways out, neither built:
1. **A_norm in DDR, N-outer/K-inner re-reading the A_norm K-tiles from the shim**
   for each N-tile. The on-chip version of this re-stream is the documented
   multi-shot zeroing blocker, but a *shim* re-delivery from DDR is a different
   path (`dma_bd` tasks again, not an objectfifo handshake) and may not hit it.
   **This is the decisive next experiment**: one N-tiled fused RMSNorm+GEMM with
   `n_n > 2` N-tiles, A_norm read from DDR each N-tile, at M=8 — does it stay
   correct? If yes, the linear stages scale to any M and the composition becomes a
   real prefill path; if it zeroes like the on-chip handoff, the fused-layer
   approach cannot do long-context prefill at all.
2. **Keep the KV projection as the engine's existing per-op large-M GEMM** and use
   the fused layer only for the query-tiled parts. That is a hybrid, and it is
   what the numbers in FK3-STATUS already pointed at.

### ✅✅ DECISIVE: the M cap is gone — shim re-delivery works

The experiment that decides the whole architecture (see the section above):
`n1_nt_gemm.py --reread-a` runs the O-proj shape with **no core-local A at all** —
each N-tile accumulates `A(kt) x W(kt,nt)` over all K with BOTH A and W
re-delivered by the shim from DDR for every N-tile (16 N-tiles x 32 K-tiles =
512 A re-deliveries of the same region).

**Result on the NPU: exact = 8192/8192 (100.0%)**, identical to the
core-local-A build.

So the documented "multi-shot re-stream returns stale/zero data" blocker is
specific to the **on-chip objectfifo handoff** (core->core / mem-routed, the
AN/AT/SC class of fifos). A **shim -> core** re-delivery from DDR — the same
`dma_bd` tasks issued once per N-tile — is perfectly repeatable.

**Consequence: the linear stages can process ANY M.** The core-local-A mechanism
(`M*K*2 <= ~40 KB`) was only ever a way to avoid the on-chip re-stream; with the
shim path proven, the fused RMSNorm+QKV / GU / O-proj / D can run at
M = npt (the prompt length), which is exactly what the engine's prefill needs
(`qkv_ascales(npt)`). The cost is DMA bandwidth: A is re-read `n_n` times per
launch (for the QKV shape, 32 N-tiles -> 32 reads of the A_norm), which is a
throughput question, not a correctness one.

This also means the composition no longer has to be pinned at M=8 — the M from
the attention (which query-tiles naturally) and the M of the linear stages can
both be the real prefill length.

**And it scales: M=128 verified.** The same re-read build at the engine's prefill
tile width `XM=128` (O-proj shape K=2048, N=1024) is **exact = 131072/131072
(100.0%)**. It needs `NT=32` and `wdepth=1` because the MEM tile's buffer budget,
not the core's, becomes the limit at M=128: the A tile alone is 2 x (128*64*2) =
32 KB and the f32 C tile (128 x NT x 4) is 32 KB at NT=64 — 80 KB against 64 KB.
At NT=32/depth 1 it is 16 + 4 + 16 = 36 KB and fits.

So the fused linear stages are no longer an 8-token tile: they run at the real
prefill M, with A re-read from DDR per N-tile (a DMA-bandwidth cost, not a
correctness one).

**All four linear shapes verified at M=128 with the re-read path** (NT=32,
wdepth=1 for the MEM budget): O-proj K=2048 -> 131072/131072 exact, D K=3072 ->
131072/131072 exact, plus QKV/GU are the same shape class as O-proj/… and their
non-re-read builds already cover K=1024. So the fused linear stages run at the
real prefill M, and the remaining work is the composition at that M (the
attention already query-tiles there).

### Normed-stage re-read: builds, but A_norm never reaches DDR (OPEN)

`n1_fused_norm_gemm_rr.py` + `build_fused_norm_gemm_rr.sh` apply the proven shim
re-read to a NORMED stage: col 0 runs the norm (A f32 K-tiles -> reduce+scale ->
A_norm bf16 K-tiles) and sends A_norm to DDR; col 1 re-reads it per N-tile and
multiplies by W. Two columns because one column's shim would need 3 MM2S (A,
A_norm-in, W) against the limit of 2. It builds at the prefill shape
(`M=128 H=1024 N=4096 k=32 NT=32` -> 20474 B) — note the norm's A tile is
`(M+1) x k` f32, so k must drop to 32 at M=128 for the MEM tile's budget.

**But it is not correct yet: A_norm is ALL ZERO in DDR** (0/131072 nonzero), and
C is all zero downstream. The DMA tasks are generated correctly
(`dma_configure_task_for @AN_S` with the verbatim `sizes=[1,1,128,32]
strides=[1,1,32,1]`, 32 of them; 64 `@A_S` tasks = 2 reps x 32 K-tiles), and the
run completes — so the norm core consumes A (the A_S awaits would otherwise
stall) but its A_norm does not arrive. The difference from the verified path is
the LINK TARGET: the verified `AN_W -> AN_R` forwards core->mem->CORE, this one
forwards core->mem->SHIM. So the suspicion is the mem->shim direction of the link
(and/or the AN_S depth), not the kernel.

Next probes, cheapest first: (1) AN_S/AN_W depth 2 -> n_k; (2) replace the
mem->shim link with the verified mem->core link plus a dummy consumer core, to
see whether the norm output appears at all; (3) if the link is the problem, have
the norm core write A_norm through a DIFFERENT mem tile (or straight into the
N-tile loop's fifo) so no mem->shim forwarding is needed.

### ✅ RESOLVED — the normed re-read works; it was the DRAIN ORDER, not the link

The all-zero A_norm was a **shim BD-lifetime** bug, not a mem->shim link problem.
The A_norm drain tasks must be:
* **armed concurrently with the scale pass** (and NOT before it — awaiting a drain
  task before the A that feeds it deadlocks, since the core has produced nothing
  yet), and
* **windowed** (started-but-not-awaited, at most ~8 in flight — arming all 32 at
  once trips `Too many simultaneously active buffer descriptors on tile (0,0),
  which supports up to 16`).

With the drain armed per scale-pass tile inside an 8-deep window:

```
A_norm nonzero = 128923/131072, C nonzero = 524288/524288
fused RMSNorm+QKV  M=128 H=1024 N=4096 (k=32 NT=32):
  exact = 498761/524288 (95.1%), within-few-ULP = 23950, worst_rel = 4.11e-06
```
i.e. the real QKV projection at the engine's prefill M, correct to the same
tiled-accumulation ULP as the M=16 build (95.2%). **The linear stages now run at
prefill scale with the fused RMSNorm, and every architectural blocker on the
fused layer is closed.**

### bf16 output: the QKV -> attention interface, verified bit-exact

The QKV projection's consumer is the attention, whose mmuls are bf16, but the
re-read GEMM accumulates in f32 over K — so the output must be converted once at
the end. `nq_nt.cc` gained a core-local f32 accumulator with an RNE store
(`nq_acc_zero` / `nq_acc_mac` / `nq_acc_store_bf16`): the accumulator stays in the
core (which matters because the MEM tile's buffers bind first at prefill M) and C
leaves as bf16, so there is no f32 round-trip through DDR just to convert.

```
BF16OUT=1 bash build_fused_norm_gemm_rr.sh 128 1024 4096 32 32 1
tests/bench_ngrr_bf16.cpp:
  fused RMSNorm+QKV bf16-out M=128 H=1024 N=4096: exact = 524288/524288 (100.0%)
```
Bit-exact against a host reference that rounds the f32 accumulation to bf16 with
the same RNE rule.

**So every interface the full-layer composition needs now exists and is
verified:** normed linear stages at prefill M (f32 or bf16 out), the attention at
NH=16/1024 keys, the cross-stage layout rules (O_s de-microtiles => row-major
buffers; A_norm is microtiled => verbatim copies), and the shim BD-lifetime rules.

## The full-layer assembly plan (and the column budget that shapes it)

Every piece is verified; this is the map for `n1_fk3_layer.py`.

**The stages need 10 columns but the array has 8**, so phases MUST share columns
(the cores are statically placed, but a core's body is a program: it can run
phase 1's loop then phase 4's, with per-phase fifos):

| column | phase 1 | phase 3 | phase 4 | phase 7 |
|---|---|---|---|---|
| 0-3 | attention (NH=16, 2 passes, query-tiled) | | | |
| 4 | norm col (A -> A_norm to DDR) | — | norm col (post-O residual) | — |
| 5 | GEMM col (QKV, N=4096) | — | GEMM col (GU, N=6144) | — |
| 6 | — | O-proj (K=2048) | — | D (K=3072) |

Both GEMM columns use the SAME tile shapes ((M,k) A, (k,NT) W, (M,NT) C), so one
set of fifos per column serves both phases; only the DDR offsets (the BD
`offset=`) and the runtime sequence's task order change.

Phase order (one launch): `norm+QKV -> attention (x2 passes) -> O-proj ->
residual -> norm+GU -> SiLU -> D -> residual`.

Two things this plan exposes, both budget rather than architecture:
* the re-read makes the DMA-task count large — QKV alone is `n_n * n_k` A tasks
  (128 x 32 = 4096 at NT=32), and the whole layer is ~5x that, so the shim's
  BD/program budget (the limit that already bit at PASSES=2/C=16) must be
  re-checked per column;
* the residual adds and SiLU are elementwise, so they are cheap to place — but
  the residual add is on the critical path between O-proj and the GU norm, and
  the cleanest form is to FUSE it into the O-proj's store (the core adds the
  saved layer input, delivered by one more shim fifo) rather than add another
  round-trip through DDR.

`residual_add.cc` (new) provides the bf16 and f32 residual adds; `silu_split.cc`
from the fk-3 PoC is the SiLU.

### QKV output -> attention taps (the last unwritten interface)

The QKV GEMM produces ONE bf16 buffer `QKV (Mtot, 4096)` row-major, blocks
`Q = [0,2048)`, `K = [2048,3072)`, `V = [3072,4096)` (Qwen3-0.6B: 16 q heads /
8 kv heads x HD=128). Every tap the attention needs is expressible from it,
because a BD's `sizes`/`strides` need not be in source order — the destination is
filled in `sizes` order, so the dim order IS the layout permutation.

For chunk `ch` (chunk tokens `[ch*N, (ch+1)*N)`), head `h`, kv head `kh = h/2`:

* **Q** (row-major (N,HD) in the mem buffer — source is already row-major, so no
  permutation): `offset = ch*N*4096 + h*128`, `sizes=[1,1,N,HD]`,
  `strides=[1,1,4096,1]`.
* **V** (row-major (N,HD)): `offset = ch*N*4096 + 3072 + kh*128`,
  `sizes=[N/8,HD/8,8,8]`, `strides=[8*4096,8,4096,1]`.
* **K^T** (the buffer holds K TRANSPOSED as (HD,N) row-major — that is what
  `attn1.cc` was verified against): the source is (N,HD), so this needs a real
  permutation. With dim order `(d, j/8, j%8)` the destination is
  `d*N + (j/8)*8 + (j%8)` = `d*N + j`, i.e. exactly K^T row-major, so
  `sizes=[HD, N/8, 8]`, `strides=[1, 8*4096, 4096]`,
  `offset = ch*N*4096 + 2048 + kh*128`.

(Note the K^T dim order — sizes `[HD,N/8,8]`, not `[HD/8,N/8,8,8]` — is what makes
the transpose come out right; the naive 4D form yields `a*8N+b*64+c*8+r`, which
is not `d*N+j`.)

So the whole QKV -> attention handoff needs **no extra kernel and no extra
buffer**: it is three BD parameterisations of the one QKV output.

## ONE-LAUNCH ATTENTION BLOCK: 3 of 4 stages bit-exact

`n1_fk3_layer.py` + `build_fk3_layer.sh` compose the whole attention block into a
single xclbin (307 KB, 535 KB of instructions): RMSNorm+QKV (cols 5,6, re-read,
bf16 out) -> attention (cols 0-3, Q/K^T/V gathered out of the QKV buffer) ->
O-proj (col 4). `tests/bench_fk3_layer.cpp`, M=16 H=1024 NH=16 HD=128 NO=1024:

```
  QKV      exact=65536/65536 (100.0%)   <- bit-exact
  attn     exact=295/32768     (0.9%)   <- the K^T tap (see below)
  O-proj   exact=0/16384       (0.0%)   (inherits the wrong attention input)
  O-proj*  exact=16384/16384 (100.0%)   <- bit-exact when fed the DEVICE's own O_all
```
`O-proj*` is the isolation check: recomputing the O-proj from the device's own
attention output shows the O-proj itself is perfect, so exactly ONE stage is
wrong and it names itself.

### The `aie.dma_bd` legality rule (this is what all the tap constraints come from)

`lib/Dialect/AIEX/IR/AIEXDialect.cpp` (~line 236) checks every BD:

```cpp
for (int i = 0; i < 4; i++) {
  if (i == 0 && inputStrides[i] == 1) continue;   // <- applied to the REVERSED array
  if (inputStrides[i] * elemWidth % addressGranularity != 0) error("Stride i ...");
}
```
i.e. **the LAST stride may be anything; every other stride must satisfy
`stride * elemWidth % 256 == 0`** — for bf16 that means *every stride except the
last must be EVEN*. This single rule explains why:
* f32 taps happily carry `strides=[1,1,H,1]` (1 element = 4 bytes divides 4) but
  the same shape in bf16 fails;
* leading size-1 dims are NOT exempt (they still trip the check), so they should
  just be dropped;
* and a BD's `sizes` order IS the layout permutation of the destination (the
  destination is filled contiguously in `sizes` order, fastest last).

### The Q tap must be MICROTILED (correcting an earlier note)

`mm.cc` reads the A operand with `load_v<MMUL::size_A>(pA)` in 32-element (4x8)
steps, so A must be in the mmul's 4x8 MICROTILED layout — NOT row-major. The Q tap
is therefore `sizes=[N/4,HD/8,4,8], strides=[4*NQKV,8,NQKV,1]` (all strides even
except the last, so it is BD-legal). Getting this wrong is not subtle: with a
row-major Q the attention scored 0.9% exact, with the microtiled Q it scores
90.5%. The earlier claim that the QKV->attention handoff needed no kernel was
wrong about Q; only K/V are plain gather taps.

### Consequence: a BD CANNOT transpose for the attention

The verified attention expects the K buffer blocked as `[d/8][j/8][d%8][j%8]`.
Producing that from K row-major needs dims `(d/8, j/8, d%8, j%8)`, whose source
strides are `[8, 8*NQKV, 1, NQKV]` — the stride-1 dim (`d%8`, the only contiguous
axis of a row-major K) sits at index 2, and it can only be legal as the LAST
entry, which would force the destination's fastest axis to be `d` (i.e. K
row-major). **So no choice of `sizes`/`strides` can produce K^T**: the transpose
has to happen in a core (either a transposing pass into the attention's K buffer,
or a `-DK_ROW_MAJOR` mode in `attn1.cc` that reads K row-major and lays it into
the blocked order locally). The current generator uses the legal-but-swapped
`sizes=[HD/8,N/8,8,8], strides=[8,8*NQKV,NQKV,1]`, which is why only the
attention fails.

### FINAL state of the one-launch attention block (M=16, H=1024, NH=16, HD=128, NO=1024)

```
  QKV      exact=65536/65536 (100.0%)  <=2ulp=100.0%   BIT-EXACT
  attn     exact=29642/32768  (90.5%)  <=1ulp=93.2% <=2ulp=94.4%   all 16 heads 82-100%
  O-proj   exact=6607/16384   (40.3%)  (inherits the attention's residual)
  O-proj*  exact=16384/16384 (100.0%)  BIT-EXACT when fed the device's own O_all
```
Three of the four stages are bit-exact, and the attention is structurally correct
with a residual explained by `attn1` accumulating its SCORES in bf16
(`matmul_vectorized_4x8x8_bf16_bf16`) while the host reference sums them in f32 —
so the two disagree by a few ULP in the scores, which `exp` then amplifies. The
sampled outputs agree to 4 bf16 ULP (dev `0.58594 -0.00230 0.08643` vs ref
`0.57812 -0.00574 0.08398`).

So the whole attention block — fused RMSNorm+QKV -> attention -> O-proj — is now
ONE launch, with the same arithmetic as the per-op path.

## + FFN GU: two norms, two GEMMs and the attention in ONE launch

The GU projection (the FFN's gate|up, N=6144) was added WITHOUT a new column: the
FFN's RMSNorm shares the QKV norm column and the GU GEMM shares the QKV GEMM
column, because the two phases never overlap — so the runtime sequence simply
points the SAME fifos at different DDR buffers and weights:

```
  QKV      exact=65536/65536 (100.0%)   BIT-EXACT
  attn     exact=29642/32768  (90.5%)
  O-proj*  exact=16384/16384 (100.0%)   BIT-EXACT (isolated)
  GU       exact=98304/98304 (100.0%)   BIT-EXACT
```
`build_fk3_layer.sh 16 1024 16 128 1024 2 2 64 64 64` -> 348 KB xclbin, 1.06 MB of
instructions, i.e. two fused RMSNorm+GEMM stages, the whole 16-head attention and
the O-proj in a single launch.

**Shim channel budget (the constraint that shaped this):** the norm column needs
A and A2 (2 MM2S) and would naively need AN and AN2 (2 S2MM) — and 2+2 trips
`'aie.tile' op number of output DMA channel exceeded`. Sharing ONE A_norm output
fifo between the two norm phases (the sequence points its drain at AN or at AN2)
brings it back to 2 MM2S + 1 S2MM. Likewise the single GEMM column runs QKV then
GU with one ANR/W/C fifo set, since the tile shapes are identical.

Left for the full layer: SiLU (`silu_split.cc`), the D projection, and the two
residual adds (`residual_add.cc`) — the SiLU can take over the FFN norm column as
a later phase, and D can become a third phase of the GEMM column (K=3072, N=1024).

## + SiLU (BIT-EXACT), and the D phase is a DISCRIMINATED bug

`silu_split.cc` now runs on its own column (2 MM2S: gate, up + 1 S2MM) over the
GU output: gate tile `nt` at `offset=nt*NT`, up tile `nt` at `offset=NI+nt*NT`,
both de-microtiled from the row-major C2 into the mmul's (M,NT) layout.

```
  ND=0 build (no D phase):   QKV bit-exact | GU bit-exact | SiLU BIT-EXACT 49152/49152
                             attn 90.5% | O-proj* bit-exact
```
The bench mirrors `sigmoid_fast` exactly, so SiLU is bit-exact, not just close.

**The D phase silently kills the tail of the sequence.** Adding it makes BOTH the
D output and the attention come back all-zero, while everything before it (QKV,
GU, SiLU) stays perfect — i.e. the sequence stops partway, with no error, no
timeout and no XRT failure.

This is NOT a task/instruction-count ceiling, and that was worth proving:

| build | tasks | insts | attention |
|---|---|---|---|
| QKV+GU+attn+O-proj | 6480 | 1,062,736 | 90.5% |
| + SiLU, ND=0 | 6624 | 1,086,352 | 90.5% |
| + SiLU, ND=64 | 6721 | 1,102,260 | **0%** |
| + SiLU, ND=1024 | 8176 | 1,340,880 | **0%** |
| + SiLU, ND=0, **bigger GU** (N2=8192) | **7728** | **1,267,408** | 90.5% |

The last row is larger than every failing row and still works, so the ceiling
hypothesis is dead: the **presence of the D phase** is what breaks it, at any
size (ND=64, a single N-tile, is enough). Its emitted BDs are correct
(`sizes=[4,8,4,8] strides=[12288,8,3072,1]` = `[4*NI,8,NI,1]`, reading the
row-major silu buffer), so the defect is structural — most likely the GEMM core's
THIRD phase in one core body. Next step: give D its own core (or its own column)
instead of a third phase, and if that fixes it, bisect what about the third phase
breaks the sequence.

### Tried and FAILED: moving D onto the O-proj core (recorded so it is not repeated)

The obvious fix — take the D phase out of the GEMM core's body and append it to
the O-proj core (col 4, whose fifos have identical shapes and whose shim is only
at 2 MM2S + 1 S2MM, so no new channels) — made things WORSE, not better: with the
D seq block moved to the end, **every** stage came back all-zero, including QKV
and GU, i.e. the sequence now stalls at its very first phase rather than partway.
So the failure mode is sensitive to how the DMA-task order and the cores' acquire
order interleave, and "which core runs the phase" is not the whole story. Reverted.

Two things worth carrying forward: (a) an extra phase in a core body is not
obviously the culprit, since the O-proj core with two phases also stalls; and
(b) the symptom moved from "tail silently dropped" to "nothing at all", which
means the first thing to check next is not D but the seq<->core ordering
invariant (every phase's task block must match its core's acquire sequence
exactly, and adding a phase anywhere perturbs it). The committed, verified
configuration is `ND=0`:
`ND=0 bash build_fk3_layer.sh 16 1024 16 128 1024 2 2 64 64 64`.

## ROOT CAUSE of the D failure: the core's A:W:C ratio is a COMPILE-TIME constant

**A core's fifo consumption pattern is fixed by its body: per N-tile it acquires
`n_k` A tiles, `n_k` W tiles, produces 1 C tile.** Every phase the runtime
sequence streams into those fifos must use the SAME `n_k`, or the stream desyncs
and the launch stalls **silently** — no error, no timeout, buffers just stay zero.

The committed `gemm_body` contained ONLY the QKV phase: the string patches that
were supposed to append the GU and D phases never matched (the anchor had moved
when the fifos/cores were reordered), so they were silent no-ops. That produced a
beautifully misleading pair of symptoms:

* **GU "worked"** — by accident. GU's K is also H (= 1024), so it also needs 16
  K-tiles per N-tile; the core's infinite outer loop simply consumed the QKV
  stream (64 N-tiles) and then kept going through the GU stream, and because the
  ratio matched, every result landed correctly.
* **D broke everything after it** — D's K is the FFN intermediate (3072), i.e.
  **48** K-tiles per N-tile against the core's 16. The sequence posts 48:48:1
  while the core consumes 16:16:1, so they desync and the sequence stalls.

It also explains why moving D onto the O-proj core failed: that core's ratio is
32 (KO_TOT/KO), which does not match 48 either.

Fix: write the GEMM core's phases out explicitly (64x16, 96x16, 16x48) and — the
important hygiene lesson — **verify the phases are actually in the emitted MLIR**
rather than trusting a patch to have applied:

```
GEMM core region: 74 lines, 7 scf.for:
  0..4294967295 | 64 x 16 | 96 x 16 | 16 x 48     (outer, QKV, GU, D)
```

## FULL LAYER COMPUTE IN ONE LAUNCH — five of six stages BIT-EXACT

M=16 H=1024 NH=16 HD=128 NO=1024, `bash build_fk3_layer.sh 16 1024 16 128 1024 2 2 64 64 64`:

```
  QKV      bit-exact 65536/65536
  attn     90.5% exact, 94.4% <=2ULP, every head 82-100%
  O-proj   (inherits the attention's residual)
  O-proj*  bit-exact 16384/16384   (isolated: recomputed from the device's O_all)
  GU       bit-exact 98304/98304
  SiLU     bit-exact 49152/49152
  D        bit-exact 16384/16384
```
That is RMSNorm+QKV -> attention -> O-proj -> RMSNorm+GU -> SiLU -> D, the whole
transformer layer minus the two residual adds, in ONE launch, with the same
arithmetic as the per-op path.

## The residual adds: the design fork, worked out (read this before coding)

Everything except the two residual adds now runs in one launch. The residuals hit
a genuine constraint, and the constraint is TYPE CONSISTENCY, not capacity:

* the norm kernels are `rms_reduce_f32(float* A, float* ss)` and
  `rms_scale_f32_bf16(float* A, float* ss, bfloat16* out)` — they take **f32** A;
* every elementwise-friendly fifo on the array is **bf16** (`(M,NT)` tiles);
* all 8 columns are full at 2 MM2S + 2 S2MM (attention 0-3, O-proj 4, norm 5,
  GEMM 6, SiLU 7), so a new stage cannot get its own fifos, and the A:W:C ratio
  rule means it cannot silently piggyback either.

So a residual stage can only live on a column whose fifos already have the right
TYPE and the 2-in/1-out pattern — which today means col 7 (bf16, the SiLU's
fifos) — and that forces the residual stream to be **bf16**, which then forces the
FFN norm's input to be bf16, which the norm kernels do not accept. That is the
whole difficulty, and it has three clean resolutions:

**(A) Make the hidden state bf16 throughout (recommended).** Add bf16 norm
variants (`rms_reduce_bf16`, `rms_scale_bf16_bf16`) and make the layer input
`(M+1, H)` bf16. Then names everything elementwise is bf16, both residuals fit on
col 7 sharing the SiLU core's fifos and its exact 2-in/1-out pattern (the core
body just gains two more phases — which is safe now that we know to verify the
phases land in the MLIR), and — the real prize — **residual1 can be fused into the
O-proj for free** by extending its K with an identity block: `A = [O_all | x]`,
`W = [W_O | I]`, so `C = O_all*W_O + x = attn_out + x` with no extra stage at all
(K becomes 2048+1024 = 3072, i.e. n_ko 32 -> 48, which is a ratio change the O-proj
core body simply declares). That leaves only residual2 needing a stage.

**(B) Keep f32 and fuse both.** Same identity-block trick, but the extra K-tile
source must be the same type as the GEMM's A tiles (bf16 O_all vs f32 x) — so this
needs a bf16 copy of x anyway, which is route (A) plus work.

**(C) Keep f32 and push the add into the norm.** New variants
`rms_reduce_add_f32(x, o, ss)` / `rms_scale_add_f32_bf16(x, o, ss, out)` reading x
from the A fifo and o from the A2 fifo (col 5 already has exactly those two f32
inputs, and the O-proj would just need an f32 store, `nq_acc_store_f32`). This
materialises no h, so it solves residual1 with zero new fifos — but residual2
(h + D) still needs h materialised, so it needs a column anyway.

**Decision needed:** whether the fused layer's interface is f32 or bf16 — i.e.
whether the engine's `h_b` hidden state handed to the fused layer is f32. If bf16
is acceptable, route (A) is strictly the simplest and the fastest (one fewer pass
over the hidden state, and residual1 costs nothing).

### RESOLVED: keep f32 (the engine's `h_b` is `std::vector<float>`), and both residuals cost NO extra stage and NO extra fifo

Checked the engine rather than guessing: `npu_engine_universal.cpp:2629` declares
`std::vector<float> h_b(XM*H)`, so the layer interface is f32 and route (A) would
mean changing the engine for no benefit. Folding that in gives a design that
sidesteps the fork entirely — it uses only fifos that already exist, on columns
that are already full:

1. **residual1, without materialising h.** Add `rms_reduce_add_f32(x, o, ss)` /
   `rms_scale_add_f32_bf16(x, o, ss, out)`: col 5 already has TWO f32 (M+1,k)
   inputs (A and A2), and the norm already wants exactly 2-in/1-out. Point A at
   the layer input x and A2 at the O-proj output o (the O-proj then needs an f32
   store, `nq_acc_store_f32`, so A2 carries o). The FFN norm then reads
   `sum((x+o)^2)` and emits `(x+o)*inv*gamma` directly — h is never written.
2. **h as bf16, for free.** The same norm core gains one more phase writing
   `h = x+o` as bf16 into a new `H_BF (M,H)` buffer, reusing the bf16 A_norm
   output fifo with different offsets — still no new fifos.
3. **residual2, fused into D.** Extend the D GEMM's K with an identity block:
   `A_D = [silu | H_BF]`, `W_D = [W_D | I]`, K = 3072+1024 = 4096, so
   `C_D = silu*W_D + h = silu*W_D + x + o = the layer output`. n_k_d goes 48 -> 64,
   which is only a declared ratio in the core body — and now we know to verify it
   lands in the MLIR.

Net: the two residual adds disappear as stages. They cost one kernel variant, one
f32 store variant, one extra norm phase, and one identity block in W_D.

## RESIDUAL 1 IS IN: the layer runs in the REAL order, in one launch

The sequence now runs the layer in its true data order (the earlier composition had
the FFN norm fed from a host buffer, which was a staging approximation):

```
  QKV norm -> QKV GEMM -> attention -> O-proj -> FFN add-norm -> h = x+o -> GU -> SiLU -> D
```

* the O-proj now stores **f32** (`nq_acc_store_f32`) into **A2's rows 0..M-1**, so
  A2 carries `o`; row M stays host-provided and holds the FFN's gamma;
* the FFN norm is the **add-aware** one: it streams x (from A) and o (from A2)
  together and reduces `(x+o)^2`, emitting `(x+o)*inv*gamma` — **h is never
  materialised**;
* a third phase of the same norm core writes `h = x+o` as bf16 into H_BF,
  microtiled per K-tile exactly like A_norm (which is what the D GEMM's identity
  block will read).

Verified at M=16 H=1024 NH=16 HD=128 NO=1024, ONE launch:

```
  QKV      bit-exact 65536/65536
  attn     90.5% exact, 94.4% <=2ULP, every head 82-100%
  o (f32, in A2)  matches the reference exactly at sampled points; worst_rel 1.1e-04
                  (38.6% bit-exact is just f32 accumulation order)
  h = x+o  H_BF 16383/16384 = 100.0%
  GU       bit-exact 98304/98304      <- through the ADD-AWARE norm, i.e. residual 1
  SiLU     bit-exact 49152/49152
  D        bit-exact 16384/16384
```
GU/SiLU/D being bit-exact is the actual proof that residual 1 is correct: they are
computed from the add-aware norm's output, so `(x+o)` must be right for them to be.

Two traps worth keeping: (a) a dropped or misplaced string patch had silently left
the core bodies without their later phases, so every phase insertion now ASSERTS
its anchor matched and the emitted MLIR is checked (GEMM core: 7 loops
[64x16, 96x16, 16x48]; norm core: 6 loops with A x5, A2 x3, AN x3); (b) H_BF (like
A_norm) is NOT row-major — it is microtiled per K-tile with the blocks
concatenated, so a row-major reference reports ~0.5% "exact" on data that is
actually perfect.

## ✅ fk-3 MILESTONE: the WHOLE LAYER, BOTH RESIDUALS FUSED, IN ONE LAUNCH

`n1_fk3_layer.py`, M=16 H=1024 NH=16 HD=128 NO=1024, one xclbin, one runtime
sequence, ~0.9 launches per layer instead of the per-op path's ~9:

```
  QKV          bit-exact 65536/65536
  attention    90.5% exact, 94.4% <=2 ULP, every head 82-100%
  o   (f32)    correct; exact at sampled points, worst_rel 1.1e-04 (f32 sum order)
  h = x + o    H_BF 16383/16384 = 100.0%          <- RESIDUAL 1
  GU           bit-exact 98304/98304
  SiLU         bit-exact 49152/49152
  D            bit-exact 16383/16384 = 100.0%      <- RESIDUAL 2 FUSED IN
```

Residual 2 costs **nothing** — no stage, no fifo, no column: the D GEMM's K is
simply extended by H and its weight becomes `[W_D ; I]`, so
`C_D = silu*W_D + h = the layer output`. The sequence is the layer's true data
order:

```
QKV norm -> QKV GEMM -> attention -> O-proj(f32, into A2) ->
FFN add-norm (x+o, h never materialised) -> h=x+o (bf16, H_BF) -> GU -> SiLU -> D
```

Every arithmetic stage is now bit-exact against a host reference except the
attention, whose residual is its own bf16 score accumulation (`attn1` accumulates
scores in bf16; the reference sums in f32).

That is the goal's core claim delivered: the fused layer is ONE launch, and the
per-op launch overhead that capped native prefill is gone.

## Scaling past M=16: what binds, and what prefill scale will need

The verified composition is at **M=16**. Probing upward, both failure modes are
resource budgets, and they bind at different M:

```
M=32  ld.lld: section '.bss' will not fit in region 'data': overflowed by 14656 bytes
M=64  'aie.tile' op allocated buffers exceeded available memory
```

* **M=32 is the attention core's own DM.** `attn1.cc`'s `.bss` scales with M_TILE
  and N_KEYS — `O_state[M_TILE*HD]` f32 is 16 KB at M=32 (8 KB at M=16) and
  `g_kt[HD*N_KEYS]` doubles too — and at M=32 that overflows the core's data
  region. So **the attention is what caps the layer's M**, not the GEMMs.
* **M=64 is the MEM tile** (the same budget that already forced k=32 for the norm
  at M=128 in the standalone build).

So prefill scale (the goal's M=128) needs the two M's DECOUPLED rather than one
shared M:

1. **Query-tile the attention**: keep `M_attn = 16` and have each attention core
   loop over `M_layer/16` query blocks, re-reading K/V for each block. The QK
   buffer then holds `(16, HD) + (HD, N)` rather than `(M, HD) + (HD, N)`, which is
   what keeps it inside the mem budget at large M.
2. **Run the norm/GEMMs at `M_layer = 128`** — the shim re-read path already
   handles that (it is 100% exact at M=128), but the norm's A tile is (M+1,k) f32
   = 33 KB at k=64, so it needs **k=32** there, exactly as the standalone build
   found.
3. The attention's O_s output and the O-proj's A gather would then be indexed per
   query block rather than per whole row.

None of that is architectural — every piece is verified, and the two budgets above
say precisely which sizing knob each one needs.

### Tried and FAILED: eliminating attn1's `g_at` to raise the M cap

`g_at[M_TILE*HD]` f32 is redundant — the alpha rescale can happen BEFORE the PV
mmul, which then accumulates straight into `O_state`, saving `M*HD*4` bytes
(1 KB per query row, i.e. the dominant static). Implemented it (drop g_at, rescale
O_state in place in the mmul's microtiled layout, PV into O_state, finalize reads
microtiled):

```
[AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory
XAie_LoadElf failed with XAIE_INVALID_ELF
```

So the attention core is near its **program** memory limit as well as its data
limit: trading the static for a slightly larger kernel body overflows the
instruction store. Reverted. Raising M therefore cannot come from shrinking
attn1's data alone — it needs the query-tiling (M_attn=16 with the outer query
loop) so that `O_state`/`g_at` are sized by the QUERY TILE rather than by the
layer's M, which is the plan recorded above anyway.

## Query-tiling the attention: the refactor is IN and verified (and what M=32 hits)

`n1_fk3_layer.py` now takes `-MA` (attention query tile) and `-NC` (attention key
chunk), and the attention core gained an OUTER query-block loop:

```
for _p in range_(PASSES):            # head (per pass)
    for _qb in range_(n_qb):         # QUERY TILE - sizes attn1's statics
        reset()
        for _ in range_(C):          # KEY CHUNKS - online softmax accumulates
            chunk_f(qk, v)
        fin(o)
```
`attn1.cc` is compiled with `-DM_TILE=$MA -DN_KEYS=$NC`, so its `O_state`/`g_at`
are sized by the QUERY TILE rather than by the layer's M — which is the whole
point of the exercise. The QK/V fifo types are now `(MA*HD + HD*NC)` / `(NC*HD)`,
and the sequence posts one Q tile (from query block `qb`) plus a K^T/V chunk
(from key chunk `ch`) per (qb, ch), with the O store per query block.

**At M=16 with MA=NC=16 (n_qb=1, C=1) the build is byte-identical in behaviour**
— same xclbin size, same every-stage result — so the refactor is a verified
no-op there and a safe base to scale from.

**M=32 still fails, and it is NOT the volume or the tiling:**
```
M=16: 8768 DMA tasks -> builds and verifies
M=32 (MA=16,NC=16): 8928 tasks -> _XAie_LoadProgMemSection(): Overflow of program memory
M=32 (MA=16,NC=32): 8832 tasks -> same overflow
```
i.e. only **+64 tasks** over a configuration that works, and *fewer* tasks in the
NC=32 case that also fails — so it is not the task count and not the key-chunking.
The only fifos whose counts change are the ATTENTION columns' shims (each +16
tasks: QK 8->32, V 4->16, O 2->4), while the heavy shim 6 (7344 tasks) is
M-independent and unchanged. That points at a per-shim program-memory accounting
issue specific to those columns rather than a real capacity wall — the next step is
to identify the failing TILE (the error names none) and bisect the attention tap
count, since a shim that carries 7344 tasks elsewhere plainly has room for 32.

### Fixed: the layer's M was leaking into attn1's COMPILE — but it is not what breaks M=32

`build_fk3_layer.sh` passed `-DDIM_M=$M -DDIM_N=$N` to `attn1.cc`, i.e. the
LAYER's M, even though the kernel's real dims are the attention TILES. That is why
the attention cores' ELFs grew with M (21368 -> 21904 B) and why they sit at the
edge of the program memory. Now compiled with `-DDIM_M=$MA -DDIM_N=$NC`, so
`attn1.o` is **byte-identical for any M** (md5 aea65eac... at both M=16 and M=32) —
correct on its own terms, and it removes a real M-dependency from the kernel.

M=32 nevertheless still fails, so the trigger is NOT attn1's compile:

```
M=32, -DDIM_*=M : attn1.o 14368 B -> core ELF 21904 B -> Overflow of program memory
M=32, -DDIM_*=tile: attn1.o 14268 B (== M=16's) -> core ELF 21904 B -> still fails
```
The attention core's ELF is 21904 B at M=32 and 21368 B at M=16 **even with an
identical object file**, so the difference is in the link/BD accounting rather than
in the kernel's code — while a core elsewhere in the array carries 7344 DMA tasks
without complaint. Every core's ELF grows slightly with M (+64 B for the norm and
GEMM cores, the statics that are legitimately sized by M), so the attention cores
were already the tallest pole and any M-driven growth tips them over. Next: either
identify the failing tile directly (the error names none) or take a few hundred
bytes out of attn1's program — note the earlier `g_at` removal, which saves DATA,
made the program LARGER and failed.

## THE M-CAP IS A 16 KB PROGRAM (.text) LIMIT — the attention core sits 80 B under it

Found the real wall by reading the ELF sections rather than guessing:

```
main_core_3_2.elf   .text            limit = 0x4000 = 16384 B (AIE2P core program memory)
  M=16 (WORKS)      0x3fb0 = 16304 B   headroom   80 B
  M=32 (FAILS)      0x41c0 = 16832 B   OVER by    448 B
```
So the attention core is at **99.5% of its program memory**, and ANY M-driven
growth tips it over — which is exactly why:
* `_XAie_LoadProgMemSection():231: Overflow of program memory` appears at M=32
  even though the DMA-task count barely moves (+64) and `attn1.o` is byte-identical;
* the `g_at` elimination failed — it saves DATA but makes the program bigger;
* every core's ELF grows slightly with M (legitimately M-sized statics), the
  attention cores were already the tallest pole, and they go over first.

`.bss` is a separate 0x5280 = 21,120 B and is not the constraint here.

**First fix landed — and it improved accuracy as well as size.** The 8-term
DOUBLE-precision `exp2_soft` Horner was the largest single block in `.text`;
replacing it with a 4-term float polynomial (≈1e-6 relative on |f|<=0.5, orders of
magnitude below the bf16 the scores are stored in) gives

```
.text   16304 B -> 16208 B   (headroom 80 -> 176 B)
attn    90.5% -> 92.2% exact, <=2ULP 94.4% -> 95.3%, mean ULP 41.2 -> 31.3
H_BF    16383/16384 -> 16384/16384 (100%)
```
i.e. smaller AND better — the old double Horner was not buying accuracy.

M=32 still needs roughly another 450 bytes out of `.text`. The remaining
candidates, largest first: the two inlined 4x8x8 mmul instantiations (QK bf16->bf16,
PV bf16->f32), the softmax's per-element microtile index arithmetic
(`/8`,`%8`,`*32` recomputed in the inner loops — strength-reducible), and the
`-DK_ROW_MAJOR` K-layout conversion loop.

## M=32 NOW BUILDS (the all-float attention freed 5 KB), and the qb loop has a desync

Removing soft-double from attn1 took `.text` from 16,208 B to **10,976 B** — 5,232
bytes of program memory back — so the attention core is no longer the tallest pole
and **M=32 builds** (M=32 attention core: `.text` 11,504 B, headroom 4,880 B).
At M=32 (MA=16, NC=16 => n_qb=2, C=2) the layer's non-attention stages are as
good as at M=16 — QKV 131072/131072 bit-exact, GU 196608/196608 bit-exact, SiLU
98304/98304 bit-exact, H_BF 32762/32768, o (f32) worst_rel 1.1e-05, D 99.2%.

The ATTENTION, however, is only 23.4% and the per-head breakdown localises it
exactly:

```
heads  0-7: 45-49% exact      (pass 0)
heads 8-15:  0%     exact      (pass 1)
```
~46% is half of the 92% those heads get at M=16, i.e. **pass 0's FIRST query block
is right and everything after it is wrong** — a desync that starts at `qb=1` and
never recovers. It is NOT the key chunking (NC=32 => C=1 gives byte-identical
numbers) and NOT the emitted structure: the attention core's loops are exactly
`outer -> p(2) -> qb(2) -> ch(2)` with `reset()` inside qb and `fin()` per qb, and
the sequence posts Q/K/V per (p, qb, ch) with the O store per (p, qb). So the next
step is to trace the fifo accounting across the qb boundary specifically — the
first thing to check is whether the QK/V fifo's write pointer still advances as the
taps assume once a second query block re-sends the same K/V rows.

## Note for fk-4: the driver's per-SUBMISSION TDR interacts with fusion

@agent-afbeb7 traced the shared `ERT_CMD_STATE_TIMEOUT` to the amdxdna driver's
`timeout_in_sec` TDR (it was 2 s; 83 dmesg "Firmware timeout state capture" dumps
all at ctx_pc 0x28b05db8/0x28b060ad/0x28b06005, matching the native Llama runlist,
the FLM oracle and the 35B runtime), and raising it to 15 s made the Llama runlist
complete alone at 82.9 ms/tok.

That matters for the fused direction in a specific way: TDR fires on a
SUBMISSION's latency, not on a launch's total work. The per-op prefill path makes
~9 submissions per layer, so it exposes ~9 opportunities per layer to sit behind
other hwctx traffic; the fused layer makes ONE, which is larger but far less
latency-exposed. Measured here, one fused layer is ~tens of ms of device time, so
it is nowhere near even the old 2 s cliff — and the fused path is therefore the
structural mitigation for the ERT class rather than a victim of it. Worth
re-confirming at prefill M, where the single launch gets much bigger.

### The qb desync: everything the sequence emits is CORRECT, and the 2nd block ignores its Q

Three checks narrowed this a long way, and the last one is the decisive negative.

**The emitted taps are all correct.** Dumped from the M=32 MLIR (M=32, HD=128,
NQKV=4096, KOFF=2048):
```
QK_S_0 taps:  Q off=0     K off=2048      (p0 qb0 ch0, head 0)
              Q off=128   K off=2048      (p0 qb0 ch0, head 1)
              Q off=0     K off=67584     (p0 qb0 ch1)      67584 = 16*4096 + 2048
              Q off=65536 K off=2048      (p0 qb1 ch0)      65536 = 16*4096
O_S_0_0 taps: off=0 2048 32768 34816      (= h*M*HD + qb*MA*HD for h=0 and h=8)
```
i.e. query block, key chunk, head and output row block all index exactly right.

**The core structure is right** — `outer -> p(2) -> qb(2) -> ch(2)`, with
`attn1_reset()` at the qb level, `attn1_chunk()` inside ch, and
`attn1_finalize()` per qb.

**The plumbing is consistent** — core qk acquires 16 == QK taps/2, v 16 == V taps
16, O 4 == O taps 4.

**And yet: posting query block 0's Q for BOTH blocks changes nothing.**

```
normal : attn 23.4%  halves-identical 0.0%   heads 0-7 ~46%, 8-15 0%
probe  : attn 23.4%  halves-identical 0.0%   heads 0-7 ~46%, 8-15 0%   (Q always qb=0)
```
If the second block were computing with the wrong Q we would at least see the two
halves converge; instead the result is completely insensitive to the second
block's Q. Together with "the halves are never identical", that says the second
query block's output is not a function of its posted Q at all — consistent with its
O item never being produced/drained as intended (every non-QB tap would then keep
working, which is exactly what we see: QKV/GU/SiLU/D all still verify).

Next hypotheses, in order: (1) the O_F/O_S handshake when a core produces MORE
THAN ONE O per loop body (at M=16 it produced one per pass; now it produces
qb-times that), i.e. give each query block its own O_F/O_S fifo pair; (2) the
QK/V broadcast fifo's item accounting across the qb boundary.

## ✅ THE M-SCALING IS FIXED: 16 -> 32 -> 64, and the qb desync was an O-FIFO DEPTH

The second query block's "insensitive to its own Q" behaviour was the tell: the
attention core produces **one O per QUERY BLOCK per pass**, but `O_F`/`O_S` were
created with **depth 1**. With more than one produce outstanding the sequence
desynced — and because only the O path desynced, every non-attention stage stayed
bit-exact, which is exactly what the symptom looked like.

```
O_F/O_S depth 1  -> M=32: attn 23.4%, heads 0-7 ~46%, 8-15 0%
O_F/O_S depth 2  -> M=32: attn 92.6% exact, 95.7% <=2ULP, every head 88-97%
```
Two more fixes came with it:

* `OUT_ty` was still `(M, HD)` — the LAYER's M — instead of the attention's
  `(MA, HD)`; that wasted 2x the mem-tile budget for the O fifos and would have
  blocked scaling.
* depth `n_qb` (rather than a flat 2) overflows the mem tile at n_qb=4, so 2 is
  the right value: enough for the core to stay one query block ahead.

**Verified at M=64** (MA=16 => n_qb=4, NC=16 => C=4 key chunks, and norm k=32 to
fit the `(M+1,k)` f32 A tile in the mem tile):

```
  QKV      bit-exact 262144/262144
  attn     92.4% exact, 95.8% <=2ULP, every head 88-96%
  o (f32)  correct, worst_rel 2.8e-05
  h = x+o  H_BF 65529/65536 = 100%          <- residual 1
  GU       97.7% exact, 99.8% <=2ULP
  SiLU     96.9% exact, 99.3% <=2ULP
  D        99.3% exact, 99.9% <=1ULP        <- residual 2 fused, the LAYER OUTPUT
```
So the fused layer now runs at 4x the M it did this morning, in ONE launch, with
the same arithmetic. Build at M=64:
`MA=16 NC=16 bash build_fk3_layer.sh 64 1024 16 128 1024 2 2 32 64 32`
(note `k=32 KO=32` — the norm's A tile is what bounds M now, not the attention).

## ✅✅ PREFILL M=128: THE WHOLE LAYER, ONE LAUNCH, 0.6B-CORRECT

`bash build_fk3_layer.sh 128 1024 16 128 1024 2 2 16 32 16` with
`MA=16 NC=16 NDEP=1`. One xclbin, one runtime sequence, 11.5 MB of instructions:

```
  QKV      bit-exact 524288/524288
  attn     92.6% exact, 95.9% <=2 ULP, mean ULP 34.25, every head 88-96%
  o (f32)  correct, worst_rel 2.83e-05
  h = x+o  H_BF 131058/131072 = 100.0%      <- RESIDUAL 1
  GU       97.8% exact, 99.8% <=2 ULP
  SiLU     97.1% exact, 99.3% <=2 ULP
  D        99.5% exact, 99.9% <=1 ULP       <- RESIDUAL 2 FUSED = THE LAYER OUTPUT
```
That is the goal's target M for dense Qwen3-0.6B prefill, with the same arithmetic
as the per-op path, in a single launch.

**The three budgets that decide what M fits** (each was found the hard way):

| knob | bound by | M=128 value |
|---|---|---|
| `NT` | the GEMM core's f32 accumulator `DIM_M*DIM_N*4` of **.bss in a ~20 KB core data region** (a link failure: `.bss will not fit in region 'data'`) | 32 |
| `k` | the norm's `(M+1,k)` f32 A tile in the **64 KB mem tile** (4 such fifos) | 16 |
| `NDEP` | the same mem tile, for the norm and SiLU (their tiles span the whole M) | 1 |
| `MA`/`NC` | attn1's statics in the **16 KB program memory** | 16 / 16 |

Two dead-code removals made this possible at all: `nq_nt.cc`'s core-local A_norm
plus `nq_store`/`nq_gemm` are never called by the re-read design, and they were
costing `N_K*DIM_M*DIM_K*2` bytes of `.bss` **in every GEMM core** — deleting them
freed 8 KB and is what let `NT=32` fit; and the O-proj's C/W tiles had to stop
being hardcoded 64 wide so `NT` could actually be reduced.

The result is the milestone the objective asked for: the per-op path's ~9 launches
per layer are now ONE, at prefill M, with token-equivalent arithmetic.

## Engine integration: the surface, and the one real obstacle

Reconnaissance for wiring `n1_fk3_layer` in as the dense-Qwen3 prefill path.

**The surface.** The native bf16 prefill is driven from
`engine/npu/src/npu_runlist_bridge.cpp` (`npu_bf16_prefill_init`, 496 lines) plus
`npu_engine_bf16_mm_bridge`, and the engine calls it at
`npu_engine_universal.cpp:4495` behind an auto-select at :795 ("dense Qwen3,
bf16 prefill + runlist decode"). Note `flm_prefill_bridge.cpp` is NOT the target —
it drives FLM's own mm/attn xclbins; the native per-op path is the one that caps at
~655 tok/s.

The bridge already exposes exactly the shape a fused layer wants:
```c
int npu_bf16_prefill_init(model_path, H, NC, NH, NKV, IM, NV, HD);
int npu_bf16_pack_layer(int layer, uint8_t* bo, int* offs);   // offs[6] = {q,k,v,o,gu,d}
int npu_bf16_layer_bo_bytes();
```
so a drop-in is: pack layer L's weights, call the fused kernel once with the
17 buffers n1_fk3_layer expects, and let the engine keep its decode path.

**The obstacle — and it is a real one.** `npu_bf16_pack_layer` packs the weights as
**Q4NX quantized tiles** (`woff = tile*5120`, dequantized on the NPU by mm.xclbin),
whereas `n1_fk3_layer` consumes **bf16 matrices** in specific layouts: W row-major
`(H, NQKV)`, W_GU `(H, N2)`, W_O `(KO_TOT, NO)`, and W_D as `[W_D ; I]`
(`(NI+H, ND)`). Three ways to bridge it:

1. **Host-side dequant once per layer** into those four bf16 matrices, then call the
   fused kernel. Simplest, and per-layer dequant of ~33 MB is affordable — but it
   must produce the SAME bf16 values as the engine's on-NPU dequant, or token parity
   fails for a reason that has nothing to do with the fusion. This is the part to
   validate FIRST, in isolation, before any timing is believed.
2. Teach nq_nt.cc to consume Q4NX tiles directly (no dequant pass). Best for speed,
   most kernel work, and it changes the verified bit-exactness surface.
3. Extend the fused kernel to take the Q4NX tile handles the engine already packs.

Recommended order: (1) to establish token parity, then (2) once parity is pinned.
**Validate the dequant parity first** — it is the one thing that can make a fused
layer look numerically wrong when it is fine.

Also note for the integration: the engine now takes an exclusive flock on
`/tmp/1bit-npu-device.lock` (afbeb7's repair) — my harnesses must take the same
lock, and the driver TDR is now a durable 15 s.

### CORRECTION: the weight bridge already exists — the "obstacle" is mostly solved

The Q4NX-vs-bf16 gap I flagged above is narrower than I thought, because
`npu_engine_bf16_mm_bridge.cpp` already exposes exactly the conversion needed:

```c
// Dequantize a Q4NX layer-BO projection -> bf16 W (D_in x D_out, row-major).
// q4nx_weight_offset is in Q4NX BYTES (tile*5120, see npu_pack_layer_bo).
extern "C" void bf16mm_dequant(uint16_t* wout, const uint8_t* q4nx,
                               int D_in, int D_out, int q4nx_weight_offset);
```
It runs `g_mm.run_dequant`, i.e. the SAME on-NPU dequant the engine's own prefill
uses. So dequant parity is guaranteed **by construction** rather than something to
chase — the one risk I said to validate first is already eliminated, and it also
means my layout requirement is met directly: each projection comes out as bf16
row-major `(D_in, D_out)`, which is what `n1_fk3_layer` consumes.

| fused buffer | source | shape |
|---|---|---|
| `W` (QKV) | `bf16mm_dequant(bo, H, NQKV, offs[0])` | (1024, 4096) |
| `W_O` | `offs[3]` | (2048, 1024) |
| `W2` (GU) | `offs[4]` | (1024, 6144) |
| `W_D` | `offs[5]` + an appended identity block | (4096, 1024) = [W_D ; I] |

So the integration is: per layer, `npu_bf16_pack_layer(layer, bo, offs)` once, four
`bf16mm_dequant` calls, and ONE fused-kernel launch with the 17 buffers. The only
non-mechanical piece is appending the identity block to W_D (the residual-2 fusion).

That makes the remaining fk-3 work: build the bf16 matrices per layer, launch the
fused kernel, and compare tokens against the existing per-op prefill.

### The integration is now EXECUTABLE (all three prerequisites verified present)

1. **Kernel dimensioned for the real model.** 0.6B is H=1024, NH=16, NKV=8,
   HD=128, IM=3072, so NQKV=4096, N2=6144, NI=3072, NO=1024 — *exactly* the shape
   the M=128 build already runs and verifies. No dimension changes needed.
2. **Weights.** `~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx` is present;
   `npu_bf16_pack_layer` + `bf16mm_dequant` turn it into the four bf16 row-major
   matrices, via the engine's own on-NPU dequant (parity by construction).
3. **Measurement vehicle.** `benchmarks/flm_parity.sh` already takes `--engine`, so
   the A/B is a flag:
   ```
   benchmarks/flm_parity.sh --model qwen3_0_6b --flm-tag qwen3:0.6b \
     --engine engine/npu/build/npu_engine_qwen3_0_6b \
     --q4nx ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
     --tokenizer ~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json ...
   ```

So the remaining fk-3 work is exactly:

* a per-layer driver: `npu_bf16_pack_layer(L, bo, offs)` once, four
  `bf16mm_dequant` calls (appending `I` to W_D for the residual-2 fusion), then ONE
  fused launch with the 17 buffers;
* gate it behind an env flag (e.g. `NPU_FK3=1`) so the per-op path stays selectable
  for the A/B;
* take the engine's `/tmp/1bit-npu-device.lock` flock before any accel0 work;
* run `flm_parity.sh` both ways and compare TOKENS, not timings — that is the fk-3
  contract, and only after tokens match does the fk-4 tok/s number mean anything.

## ⚠ THE FUSED LAYER IS NOT YET A VALID MODEL LAYER: no causal mask, no RoPE

Checked before wiring it in, and it would have made token parity fail for reasons
that have nothing to do with the fusion:

```
attn1.cc:  no causal mask    (no mask/position logic anywhere)
attn1.cc:  no RoPE           (no cos/sin/theta)
engine:    DOES apply RoPE   (partial rotary: rope_dim = round(HD*partial_rotary_factor),
                              tables built by ri2_build/ra2 at npu_engine_universal.cpp:362+)
```
So everything verified so far is a **fusion** result — correct as a composition, and
bit-exact or near-bit-exact against a reference that has the same structure — but
the layer is not yet a correct *transformer* layer:

* **Causality**: my attention lets every query attend to every key, including future
  ones. For a fresh prefill of M tokens the mask is mandatory. The reference in
  `bench_fk3_layer.cpp` is also unmasked, which is exactly why it agreed — the
  agreement was real, but it is agreement about the wrong function.
* **RoPE**: Q/K go into the attention unrotated. Qwen3 uses partial rotary, so the
  engine rotates only the first `rope_dim` dims of each head, with a precomputed
  table per slot.

**Where both belong, and why it is cheap.** Both are contained `attn1.cc` changes,
and the query-tiled + key-chunked structure I already built makes them natural,
because it is the only place that knows the GLOBAL positions:

* *causal*: in the score loop, masked entries become -inf before the max, so the
  online softmax handles them for free; for query block `qb` and key chunk `ch` the
  mask is the usual `key_pos > query_pos`.
* *RoPE*: applied on load of the Q tile and the K chunk, using the same
  `rope_dim`/theta the engine uses, at the global positions `qb*MA + row` and
  `ch*NC + col`. Partial rotary means only the first `rope_dim` of each head's HD
  dims rotate.

Neither changes the fusion architecture, the buffers, the tariffs, or the launch
count — the sequence and the buffers stay exactly as verified. But until they land,
**token parity is not a test of the fusion**, and I should not run it as one.

Ordering for fk-3 now: (a) causal mask, (b) partial RoPE, (c) then the engine driver
and token parity; and the bench's reference must be updated in step with (a)/(b) or
it will keep agreeing about the unmasked, unrotated function.

## ✅ CAUSAL MASK LANDED (and it improved accuracy)

`attn1.cc` is now causal, which it has to be for a prefill of fresh tokens. Two
static counters give the kernel its GLOBAL positions without any index arithmetic
in the generator's DSL (which has none): the sequence drives every core in
(pass -> query block -> key chunk) order, so `attn1_reset()` advances `g_qb` and
starts `g_ch`, and `attn1_chunk()` masks scores with `k0 + c > q0 + r` before the
online softmax — masked entries become bf16 -inf, so the max and exp loops are
untouched and `l_state` never sees them.

```
            attn exact   <=2ULP   mean ULP
M=16        92.2% -> 96.6%   95.3% -> 98.3%   31.3 -> 20.6
M=32        92.6% -> 94.5%   95.8% -> 97.0%   40.6 -> 25.3
```
It *improves* agreement rather than degrading it, which makes sense: each query now
sums over fewer keys, so the accumulation error drops. The bench's reference was
given the same mask in the same commit — without that it would have kept agreeing
about the unmasked function.

One bug worth recording, because the shape of it recurs: `g_qb` was incremented in
`attn1_reset()` but initialised to 0, so the FIRST query block came out as 1.
At `N_QB == 1` the modulo wrapped it back to 0 and hid the bug; at `N_QB > 1` every
query block was off by one and the mask cut the wrong rows (30.5% instead of 94.5%).
Initialising it to `-1` is the fix. Any counter that is "advanced at the start of a
unit" must start one below its first value.

Still outstanding for a valid model layer: **partial RoPE**, which the engine
applies (`rope_dim = round(HD*partial_rotary_factor)`) and the kernel does not.

## M=128 CAUSAL — the prefill layer, one launch, current state

`MA=16 NC=16 NDEP=1 bash build_fk3_layer.sh 128 1024 16 128 1024 2 2 16 32 16`

```
  QKV      BIT-EXACT 524288/524288
  attn     93.2% exact, 95.4% <=1ULP, 96.2% <=2ULP, mean ULP 41.2, every head 89-96%
  o (f32)  correct, worst_rel 1.75e-04
  h = x+o  H_BF 131059/131072 = 100.0%          <- RESIDUAL 1
  GU       98.5% exact, 99.8% <=2ULP
  SiLU     97.8% exact, 99.3% <=2ULP
  D        99.6% exact, 99.9% <=1ULP, mean ULP 0.26   <- RESIDUAL 2 FUSED = OUTPUT
```
Causality cost nothing and gained a little (attention 92.6% -> 93.2%). Two stages are
exactly 100%: the fused RMSNorm+QKV, and residual 1 (`h = x+o`).

Honest reading of the rest: the attention and everything downstream of it agree to
a few ULP rather than bit-exactly, and the reason is understood — `attn1` accumulates
its SCORES in bf16 (`matmul_vectorized_4x8x8_bf16_bf16`) while the host reference sums
them in f32, and `exp` amplifies that into the output. The layer output lands at
99.6% bit-exact / 99.9% within 1 ULP with a mean error of 0.26 ULP. That is the
fused layer matching the reference as closely as its own arithmetic allows — not a
defect to chase, but also not yet "bit-identical to the native path", which is what
token parity will actually adjudicate.

Remaining for a valid model layer: **partial RoPE**.

## RoPE: attempted, reverted, and the constraint is now precise

Everyone should know this before trying it: **there is no trig for aie2p.**

* bare `sin`/`cos`/`cosf`/`sinf` **fail to link** (`undefined symbol: sin`);
* `aie::sin` / `aie::cos` exist but their scalar overloads are gated on
  `arch::is(arch::AIE)` — the compiler says so explicitly — so they are AIE1-only;
* the official `aie2p` kernel `mlir-aie/aie_kernels/aie2p/rope.cc` therefore takes a
  **LUT** (`const T* lut`) instead of computing anything.

**And that LUT has nowhere to go here.** The attention columns are already at the
shim limit (2 MM2S: QK and V; 1 S2MM: O), so a table fifo would need a third MM2S
that does not exist.

Convention, which is the other trap: the ENGINE is **half-split**
(`ri2_build`/`ra2`: pairs `(d, d + rope_dim/2)`, `x[d]=x[d]*c - x[d+hd2]*s`,
`x[d+hd2]=x[d+hd2]*c + x[d]*s`, `f_d = 1/theta^(d/hd2)`), whereas the official
`rope.cc` is **interleaved** (even/odd pairs via `filter_even`/`filter_odd`). So
`rope.cc` cannot be dropped in either — it would silently compute a different RoPE.

Parameters for 0.6B are simple, at least: `config.json` has **no**
`partial_rotary_factor`, so it is FULL rotary, `rope_dim = HD = 128`, `theta = 1e6`.

So there are exactly two viable routes, and both are real work:
1. **in-kernel polynomial sin/cos** with coarse range reduction (a = pos*f_d reaches
   ~4096 rad at d=0). Precision only needs to beat the bf16 rounding of Q/K
   (~1e-3 relative is comfortably below it), so a 4-term polynomial is enough — and
   `pos` is already available as `q0+r` / `k0+j` from the causal-mask counters.
2. **give the attention a table**: one extra shim channel, which means reducing the
   attention to fewer columns or folding RoPE into a stage that still has one.

Cost note for route 1: it is `(MA + NC) * rope_dim/2` trig calls per chunk, i.e. 32*64
= 2048 per chunk here, which is comparable to the chunk's own MACs — so it needs the
cheap polynomial, not a libm call, even if libm existed. This is the one remaining
piece between the current fused layer and a valid model layer.

### RoPE, second attempt: the polynomial works, but it does not FIT

Built exactly the design above — in-kernel `fast_sincos` (Cody-Waite reduction + a
4-term minimax, ~1e-3 rad), the frequency table by recurrence from a build-time
constant (no libm `pow` either — that also fails to link), the engine's half-split
convention, Q rotated in place before the mmul and K rotated inside the existing
layout conversion. It compiles and links, and the rotation order was verified in
the source before building.

Then it fails on program memory:

```
attn core .text   without RoPE : 11504 B   (headroom  4880 B of 16384)
attn core .text   with RoPE    : 24768 B   (OVER by   8384 B)
after a noinline refactor      : 24640 B   (still over)
```
So RoPE costs ~13 KB of `.text` — the index arithmetic and bf16 gather/scatter for
the microtiled Q and the blocked K, times two call sites, times whatever the
compiler unrolls. Making the two block rotations `noinline` recovered only 128 B,
so it is not the unrolled loops. **The attention core cannot host RoPE.**

Where it CAN go, and this is the useful part: `.text` per core in this build is
attn 11504 B, **norm 12336 B (4048 free), GEMM 4192 B (12192 free)**. The QKV GEMM
core is the obvious host — it holds ~12 KB of headroom, it already knows each row's
GLOBAL position (within its C tile the row index IS the position, since the tile
spans all of M), and it knows the channel from the N-tile index, so it can tell a Q
column from a K column and rotate the right ones before storing. The frequency table
and `fast_sincos` are then instantiated once, in the core with room.

That is now the concrete plan for RoPE: move it from attn1 to the QKV GEMM core's C
store. Reverted for now — the verified causal mask is intact.

### …and the QKV-GEMM placement does not work either: a RoPE pair spans two N-tiles

Correcting the paragraph above before anyone acts on it. Moving RoPE into the QKV
GEMM's C store looked ideal (12 KB of program headroom, the row index IS the global
position, the channel is known from the N-tile index) — but the rotation needs the
pair `(d, d + hd2)`, which is **64 channels apart**, while a C tile covers only
`NT` adjacent channels. With `NT = 64` the two halves of every pair land in
DIFFERENT N-tiles, i.e. in two separate stores, so the store never sees both.

So RoPE needs a stage that holds a **full head (all 128 channels) at once** *and*
has program headroom — and on this build nothing has both:

| core | .text | headroom | sees a full head? |
|---|---|---|---|
| attn1 | 11504 B | 4880 B | yes (that is where it belongs) |
| QKV GEMM | 4192 B | 12192 B | no — N-tiled, pair split across tiles |
| norm / SiLU | ~12 KB / ~7.5 KB | small | no |

Making the GEMM accumulator `NT = 128` would fix the visibility but the f32
accumulator is then `M*128*4` = 64 KB against a ~20 KB core data region, which is
the same wall that already forced `NT = 32` at M=128.

So the honest options for RoPE are:
1. **a dedicated RoPE stage** — a core that reads Q and K back, rotates, writes:
   correct and modular, but it costs a column and shim channels, both of which are
   exhausted;
2. **shrink attn1** enough to fit ~13 KB of RoPE — the kernel is already lean;
3. **keep RoPE on the host** — pragmatic, and it is what the objective wanted moved
   in-kernel, so this is a real (if temporary) concession;
4. **restructure the QKV N-tiling** so a head is contiguous in one tile, which
   costs the accumulator budget and therefore M.

This is now a genuine open design decision rather than a missing implementation.

### RoPE option 1 is ELIMINATED: the attention cannot run in 2 columns

Tested `PERCOL=4` (which would put 8 attention cores into 2 columns and free two for
a RoPE stage):

```
design.mlir:54:26: error: 'aie.tile' op number of input DMA channel exceeded!
```
The MEM tile cannot take four consumer channels per column, so 4 cores per column is
not buildable at all. Option 1 is out, and with it the cheapest in-kernel route.

That leaves, honestly:

| route | cost |
|---|---|
| keep RoPE in attn1 | needs ~13 KB of program text the core does not have (4880 B free) |
| RoPE in the QKV GEMM store | the pair (d, d+hd2) spans two N-tiles; NT=128 would fix it and cost the accumulator budget (hence M) |
| dedicated RoPE column | no column left, and the attention cannot shrink to two (above) |
| **split the launch**: QKV launch -> host RoPE -> attention+O+FFN launch | **2 launches per layer instead of 1**, RoPE stays on the host |
| host RoPE inside one launch | impossible: Q and K are produced and consumed inside the same launch |

The split route is worth stating plainly because it still delivers the objective's
actual goal: the per-op path's ~9 launches/layer become **2**, the fixed per-launch
overhead drops by ~4.5x, and RoPE on M=128 tokens x 16 heads x 128 dims is
microseconds of host work. What it does NOT deliver is the literal "~1 fused layer
launch" wording, and it leaves RoPE as a host op, which the objective asked to move
in-kernel.

This is a genuine fork between the objective's wording and the array's resource
limits, so it is the user's call rather than something to silently pick.

## DECIDED: split into 2 launches, host RoPE — and launch A already exists

User decision. The fused layer becomes two launches with the host rotating Q/K in
between, and the priority is token parity + fk-4 numbers.

**Launch A already exists and is already verified.** `n1_fused_norm_gemm_rr.py` /
`build_fused_norm_gemm_rr.sh` is exactly "fused RMSNorm + QKV, bf16 out", verified
bit-exact at M=128, N=4096. Its C store de-microtiles with
`sizes=[M/4, NT/8, 4, 8], strides=[4*N, 8, N, 1]`, i.e. it emits **row-major
(M, NQKV)** — precisely the layout launch B reads. So no new kernel work:

```
launch A : bash engine/npu/generators/build_fused_norm_gemm_rr.sh 128 1024 4096 32 32 1
           -> bit-exact QKV, row-major (M, 4096) bf16
host     : rotate Q (cols [h*HD, h*HD+HD)) and K (cols KOFF + kh*HD) in place,
           full rotary, theta 1e6, half-split pairs — ~microseconds at M=128
launch B : n1_fk3_layer.py with -NOQKV : attention + O-proj + FFN norm + GU + SiLU + D
```
`-NOQKV` only needs to drop TWO phases, not any hardware: the FFN norm already
shares the norm column and the GU/D already share the GEMM column, so B keeps every
fifo and core it has today and merely skips the QKV norm phase and the QKV GEMM
phase — in BOTH the core bodies and the sequence, because a core's fifo ratio is a
compile-time constant and the two must stay in step.

That is the whole change, and it is small — which is the point of having chosen this
route.

### Launch B is built: `-NOQKV` verified structurally

`NOQKV=1 bash build_fk3_layer.sh 16 1024 16 128 1024 2 2 64 64 64` →
315 KB xclbin, 1.08 MB of instructions (down from 1.44 MB). Checked in the emitted
MLIR, which is the only thing that proves it:

```
norm core loops : outer, 16, 16, 16            (was 5 loops - QKV norm phase gone)
GEMM core loops : outer, 96x16, 16x64          (was 64x16 + 96x16 + 16x64)
seq A_S taps    : 48   (was 80 - only the FFN add-norm's A reads remain)
seq QKV_S stores: 112  = GU 96 + D 16          (was 176 - the QKV's 64 gone)
```
So the two phases are dropped from BOTH the core bodies and the sequence, which is
the only way it can work: a core's fifo ratio is a compile-time constant, so the
body and the stream have to move together. `QKV` is now an input to launch B rather
than an output — the host passes launch A's buffer straight in, so the sequence
signature is unchanged.

Remaining for the 2-launch design: the host RoPE pass between A and B (rotate Q at
columns `[h*HD, h*HD+HD)` and K at `KOFF + kh*HD`, full rotary, theta 1e6,
half-split pairs), then the engine driver.

### Host RoPE: implemented and verified (`engine/npu/src/npu_fk3_rope.h`)

`fk3::rope_qk_bf16(qkv, M, NH, NKV, HD, theta, pos0)` rotates Q and K in place in a
row-major `(M, NQKV)` bf16 buffer — Q at columns `h*HD`, K at `KOFF + kh*HD`,
V untouched — using the engine's **half-split** convention, in f32 working precision
with an RNE bf16 store.

`engine/npu/tests/test_fk3_rope.cpp` checks it against an independent transcription
of the engine's `ra2` loop on a small full-rotary case:

```
rope_qk vs independent reference: 0/160 differ -> MATCH
V untouched: yes
```
So the pass between launch A and launch B is done and self-checked. It is
microseconds at M=128 (M*24 heads*HD/2 rotations, all host-side).
