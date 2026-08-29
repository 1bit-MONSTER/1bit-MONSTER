# Single-launch fused decode: core-stream h2 relay (issue #1775 — the real fix)

## Why every other approach fails (verified on strixhalo)

| Approach | Result |
|---|---|
| npu.sync / npu.dma_wait / dma_wait(H2_S) barriers | built + tested on two fresh boots — TCT fires at DMA-accept, not cross-column DDR visibility |
| per-MoE-layer h2 BOs (f8cd4187) | fixes the layer-3+ HANG; not the token flips |
| split two-launch with host h2 sync + drain-read (f4b74476) | works (#1767 partial-fusion milestone) but the shared two-launch path (NPU_FUSED=0) also flips tokens — the handoff race is the S2MM-writeback TCT≠visibility class affecting every NPU output readback |
| BD locks (get_lock_acq/rel, exposed on aiex.npu.writebd) | per-tile resources — the D-phase A2 MM2S on shim[0] cannot wait for shim[1..7]'s locks |
| mem-tile h2 relay (object fifos) | mem-tile DMA channel budget: 3 in / 3 out "at the measured limit" (n1_core_fused_gu_silu_d.py comment); the relay needs 4+ in — over budget |
| core DMA channel relay (A_m mem→core) | core budget 2 in / 2 out — A+A2+B = 3 in, exceeded (measured) |

## The design that fits the budgets: core-to-core STREAM relay

Streams are separate from the DMA channel budget (the AIE switch + core stream
ports). Keep h2 in CORE tile-local buffers (no DDR, no H2_c fifo writeback):

```
GU phase (per column c): A(residual shim0→cores, existing A_c) + B_gu → C1 →
  silu → h2 chunk kept in core[c] tile-local h2buf[c]   (8×256 int8 = 2 KB)
Relay (core-stream, via switchbox config + cascade-style core kernels):
  core[c] sends h2buf[0..c] to core[c+1]; each core keeps received chunks in
  its own h2buf — after the chain every core holds the FULL h2 (8×2048 = 16 KB)
D phase: A2 read directly from the core's own h2buf (no fifo) + B_d → C2
  (existing C2_c/C2_s writeback to DDR, host readback as today)
```

Channel/buffer budget:
- core DMA: A_c + B (2 in), C2 (1 out) — H2 uses the CORE STREAM port, not DMA ✓
- core L1: existing ~43 KB (batch-2 depths) + h2buf 16 KB + relay staging 2 KB ≈ 61 KB < 64 KB — MUST re-verify with aiecc (the old design already shrank depths to 2; expect similar)
- relay traffic: ~144 KB/layer through 128-bit core streams ≈ 10 µs — negligible vs ~300 µs/layer

## Implementation requirements (why this is multi-session)

1. Core-to-core stream ops + switchbox config in the generator. The old dialect
   API (aie.dialects.aie) exposes `switchbox` but the core-stream acquire/produce
   + routing must be verified; the iron API (aie.iron, CascadeFlow) is the
   supported path for worker-to-worker streams — likely a rewrite of the
   generator into the iron paradigm.
2. The relay chain kernel: each core passes chunks downstream while keeping
   copies (a core loop over its stream port + tile-local stores).
3. aiecc build + hardware iteration on strixhalo (clone ~/1bit-fused-verify has
   the toolchain, models, all xclbin variants, the built binary).
4. Determinism gate: 8+ runs of the same binary/model/prompt must produce
   identical tokens; L1 h2 fingerprint sum=1037 must hold.

## Interim (already committed, reliable)

- f8cd4187 per-MoE-layer h2 BOs — kills the hang mode.
- f4b74476 split GU→SiLU + D launches with host h2 sync — correct output, the
  #1767 partial-fusion milestone; the fallback if the stream relay stalls.

## Implementation attempt result (stream relay)

The core-stream relay v2 generator (n1_core_fused_gu_silu_d_v2.py) was
implemented and generates valid MLIR (42 stream ops + 8 flows), but BOTH
the strixhalo mlir-aie and the dump's npu2_40_toolchain REJECT it at the
aiecc verifier stage:

    error: 'aie.put_stream' op expects parent op 'aie.core'

The verifier requires `aie.put_stream`/`aie.get_stream` to have `aie.core`
as their IMMEDIATE parent — stream ops inside `scf.for` loops are not
accepted. The relay needs 128 stream beats per 512 B chunk × ~56 chunk
transfers per core per layer; without loops that is ~7k unrolled stream
ops per core (AIE core code-size limit exceeded). The iron API
(CascadeFlow, aie.iron) is the remaining path — worker-to-worker streams
are a different (dataflow-level) lowering that may route around the
verifier, but it is a complete paradigm rewrite of the generator
(multi-day, uncertain).

CONCLUSION: the single-launch h2 broadcast cannot be built with the
available toolchains' core-stream abstraction. The definitive fix options
are (a) iron-API generator rewrite, (b) a toolchain whose verifier allows
stream ops in loops, or (c) a hardware mechanism outside these
abstractions. The committed two-launch split (f4b74476) remains the
reliable interim; note the shared-path token flips (two-launch included)
still need the S2MM-visibility investigation.

## iron-API rewrite (complete generator rewrite — issue #1775)

n1_core_fused_gu_silu_d_iron.py: the ENTIRE fused GU→SiLU→D generator
rewritten in aie.iron:
- 8 CoreWorkers (one per column), fused core fn: GU mmul -> on-core SiLU ->
  h2 held in a CORE-LOCAL buffer -> D GEMM as a CASCADE REDUCE (partial
  C2 per k-slice summed down the cascade; core 7 writes the output).
- No h2 DDR round-trip -> the cross-shim S2MM->MM2S visibility race is
  structurally eliminated.
- Dataflow: A broadcast fifo (shim0 -> 8 cores), per-column B fifos, per-
  column C2 writeback fifos, 7 CascadeFlow edges (core c -> c+1).
- Generates valid MLIR (972 lines: 8 aie.core, 7 aie.cascade_flow, 24
  kernel calls).

Open item blocking the build+verify: the cascade-reduce kernels
(cascade_d_first/mid/last, ~40 lines each: partial = A2@B; get_scd/add/
put_mcd) need the AIE2P cascade intrinsics, which the installed Vitis
2025.2 aie2p headers do NOT expose (put_mcd/get_scd exist only for aie2
in adf/stream/me/accessors.h). Options: a Vitis/newer-toolchain aie2p
cascade intrinsic, or lowering the cascade to the switch-stream path once
a stream-loop-capable toolchain exists.

RESOLVED 2026-08-27: the AIE2P cascade intrinsics ARE available in the
PEANO toolchain (llvm-aie clang resource dir aie2p/aie2p_streams.h:
get_scd_v16acc32 / put_mcd(v16int32) -> __builtin_aie2p_scd_read_acc32 /
__builtin_aie2p_mcd_write_vec, 512-bit words = v16int32; the Vitis
adf/stream/me accessors only carry the __AIE_ARCH__<20 variants, so
chess has no aie2p cascade — the cascade kernels are peano-only).

Implemented + BUILDING: cascade_d_first/mid/last_i8_i32 in
engine/npu/generators/mm_kernel_reference.cc (guarded #ifndef __chess__;
one k-slice per call, a2s(8x64) @ b(64x128) via the m8 mmul pattern,
then the 8x128 int32 tile is cascade-merged in 16 x 512-bit chunks with
the same block-major C layout as matmul_vectorized_8x8x8_i8_i32_m8, so
the host D-gate validates the output unchanged). Generator
n1_core_fused_gu_silu_d_iron.py fixed for the build: C2 writeback is
TAIL-ONLY (col 7; first/mid get a scratch Buffer — draining all 8 raced
one output BO), runtime fills added (per (cg,ki) task groups, 1 A
broadcast + 8 per-column B tiles per group, awaited/freed per group so
the shim BD pool is not exhausted; 4-D sizes/strides taps — the 1-D tap
form made aiecc emit repeat_count = element_bytes-1 and the npu-insts
pass blew the [0:255] push_queue repeat range), gs header tiles ride the
B stream, and the python late-binding closure bug (c2scr captured the
last loop iteration's Buffer) fixed by passing it via fn_args. aiecc
(iron venv, --no-xchesscc peano) now builds final.xclbin + insts.txt
with configure_cascade(%tile_c_2, West, East) on cores 0-6 and (West,
South) on core 7, and the core ELFs carry the vmov mcd/scd instructions.
NEXT (hardware round on strixhalo): host packer for the B_gu/B_d layout
+ C2 gate vs CPU, cascade lane-order check, then the fused decode run.

## SILICON STATUS 2026-08-27 (strixhalo, /dev/accel0)

The cascade chain is now PROVEN on hardware in isolation (minimal 8-core
cascade design: h2=42 const, B=ones, C2 = 42·K·8 everywhere, validated
block-by-block). Key findings:

1. **AIE2P cascade instrs are real + the chain works**: get_scd/put_mcd
   (aie2p_streams.h, peano) lower to `vmov mcd/scd`, the configure_cascade
   West→East wiring is honored, and a single cascade pass sums 8 cores'
   partials correctly (C2 = 21504 = 42·64·8 on every one of 16 blocks).
2. **aie2p codegen bug FOUND + FIXED**: the cascade-reduce kernel's
   register-array form (`aie::vector<int32,64> V[4]` + 4x-mmul j-group +
   nested inner get_scd/put_mcd) is mis-compiled by the peano aie2p backend —
   blocks 4-15 never written, block 2 row-corrupted (deterministic). The
   one-mmul-per-block form (no V[4] array, A pointer reset per block, no
   nested cascade in the inner loop) is correct on silicon.
3. **Multi-call cascade deadlock (REMAINING BLOCKER)**: calling the cascade
   kernel MORE THAN ONCE per core (the generator's per-k-slice D loop = 64
   calls) deadlocks (n=128 → hang; n=64 → all-zeros). The aie2p cascade is a
   CONTINUOUS stream, not per-call: the get/put count drifts across kernel
   call boundaries. FIX (next): restructure so each core does ONE cascade
   call per D col-group (processing its whole K-slice internally: loc_c =
   h2_slice_c(8×256) @ B_d-region, summed by the chain over 2 cg2 = 2 calls),
   instead of 64 per-ki calls.
8. **Host-facing C2 layout**: the drain writes the tail's column block 7 tiles
   at C2_bo[ (cg2·1024+896)·4 .. ] — the host must read row r cols [896,1024)
   and [1920,2048); cols 0-6's tiles are never written (tail-only writeback).

## TWO-PHASE CASCADE — SILICON-VERIFIED (2026-08-27, updated)

The multi-call deadlock is FIXED by restructuring the D reduce to TWO PHASES
(the aie2p cascade is a continuous stream — per-k-slice kernel calls deadlock):

1. **Accumulate** (per col-group): each core computes its OWN partial
   `c2scr = Σ_ki (a2s_ki @ b_ki)` with the PROVEN `matmul_i8_i32` over the
   streamed B k-slices (B is chunked via the fifo — it does not fit L1). No
   cascade — the mmul is a plain accumulation.
2. **One cascade pass** (per col-group): `cascade_reduce_first/mid/last_i32`
   merge the 8 cores' accumulated partials through the hardware cascade in a
   SINGLE continuous pass; the tail writes the sum to C2.

VERIFIED ON SILICON:
- Minimal two-phase (8 cores, h2=42, B=ones): C2 = 21504 = 8·(42·64) on EVERY
  element, all 16 blocks — the cascade_reduce chain is correct.
- Full design (no-gu, h2=42 const): C2 = 626304 / 653184 per D col-group
  (row- and col-uniform, deterministic) — the pipeline runs end-to-end and
  writes C2. (The values differ per col-group because B_d differs per column
  block — correct; the exact magnitude vs the CPU reference is the remaining
  B_d tile-delivery detail.)
- CPU gate test_cascade_reduce.cpp validates the protocol (chain==ref,
  tail accumulate, drain→row-major).

NEXT (data-path debugging): (a) the B_d tile delivery reads ~1864-1944 of the
2048 K-values (not all) — #1868-class truncated-tile issue to pin; (b) the
real GU phase (silu→h2) in this design produces 0 — the A/gs delivery or the
GU→h2 scatter needs isolating (the GU was proven in the earlier fused
milestones; the iron design's A broadcast + gs tiles are the new pieces).

## FINAL STATUS 2026-08-27 (two-phase cascade verified, GU-path remaining)

The two-phase cascade is SILICON-VERIFIED:
- Minimal two-phase: C2 = 21504 = 8·(42·64) exact on all 16 blocks
- mmul accumulate (32-call, core-local Buffer): 2048 exact
- A broadcast, multi-tile fills, task-group batching: all verified
- Design builds end-to-end (final_cascade_fused.xclbin, 90688 B)

REMAINING (needs the aie2p simulator — blocked by the xbridge
duplicate-.LBB-local wall on the peano cascade object, OR a different GU
scatter approach): the real GU phase (silu→h2b→D) in the iron design yields
C2=0 despite the GU consume completing; the h2b scatter / GU mmul / silu
feedback in this Single-launch design is not yet correct on hardware. Fixing
it requires per-tile visibility (the sim) or splitting the GU and D into
separate launches (the proven p1/p2 two-launch split — the reliable interim).

## GU-path hang ROOT CAUSE ISOLATED (2026-08-28)

The real GU phase in the iron design previously gave C2=0. Systematic silicon
bisect (minimal designs, C-output pattern-prefill to detect core hangs):
- **GU 1-cg (A broadcast 4D-tap + 32x mmul accumulate + gs tile + silu): WORKS**
  (c1b = 2048, correct).
- **GU 2-cg (2x the above in a cg loop): HANGS** — C1 untouched (pattern).
- **GU 2-cg WITHOUT silu (64x mmul, no gs): WORKS** (c1b = 1856, nonzero).
- **GU 2-cg + acquire gs tile (no silu): HANGS.**

CONCLUSION: the hang is the **B-fifo gs tile acquired between col-groups** in a
MULTI-cg loop — a B-only tile with no matching A-tile breaks the A-broadcast /
B-fifo sync (the cores pause A consumption at the gs; the broadcast handshake
stalls). Single-cg (1 gs at the end of the only cg) works; multi-cg (gs between
cgs) hangs. This is a toolchain/iron-objectfifo behavior, not the cascade or the
mmul. The p1 generator avoids it by acquiring the gs ONCE per launch (the fold
rides in C1 per the int4 path) — the iron int8 silu reads gs[0]/gs[4] per cg,
so it acquires a gs tile per cg -> multi-cg -> sync stall.

FIX (next): either (a) read the silu scales from a CORE-LOCAL section header
(no per-cg gs fifo acquire), or (b) reshape the GU B stream so the gs tiles are
not interleaved B-only acquisitions in a multi-cg loop, or (c) use the proven
p1/p2 two-launch split (reliable interim). The CASCADE two-phase (the novel
part) is unaffected and silicon-verified.

## GU-path hang CONFIRMED: the FLOAT silu miscompiles on aie2p (#1836)

Definitive silicon bisect (2-cg GU, C-output pattern-prefill, silent-core
detection):
- 2-cg GU NO silu kernel (h2s=127 via python): **WORKS** (c1b=2048).
- 2-cg GU WITH `silu_quant_i8_fused` (float silu_lut): **HANGS**.

So the hang is the **float silu** (silu_lut's float loop / float sigmoid LUT)
which the peano aie2p backend MISCOMPILES (#1836) — it faults/infinite-loops,
stalling the core. The prior "gs-tile acquire" and "2-cg structure" theories
were red herrings; the gs acquire and multi-cg loop are fine.

THE FIX: use the **q22 int32 fixed-point silu** (`silu_pair_q22` +
`silu_sigmoid_q22`, the #1844/v59 aie2p-correct path). This is what the PROVEN
p1/p2 fused split uses: the int4 matmul (matmul_i8_i32_i4) computes the gate/up
dot products AND stashes the per-tile fold metadata (foldG/boundG/boundU/Q/
shG/shU) in C1 rows 1-4 (gu_i4_pack.h computes the fold from the scales);
silu_quant_i8_fused_i4 reads it and computes h2 via silu_pair_q22. The int8
float silu (silu_quant_i8_fused) has NO aie2p-safe fixed-point version in the
iron design.

RECOMMENDED: port the GU to the int4 path (matmul_i8_i32_i4 +
silu_quant_i8_fused_i4 + gu_i4_pack.h host packer) — silicon-verified
(corr 0.999563, halves GU DMA) and avoids the float miscompile. The two-phase
cascade (novel) is unaffected and already silicon-verified.

## BREAKTHROUGH + remaining constraint (2026-08-28, cascade single-launch)

**THE GU HANG IS FIXED — the float silu was the culprit.** Confirmed on
silicon: 2-cg GU with h2s set directly (no silu kernel) WORKS; with
`silu_quant_i8_fused` (float silu_lut) it HANGS (aie2p float miscompile #1836).
**Fix**: `silu_quant_i8_fused_q22` — a pure-int32 silu using the `silu_sigmoid_q22`
LUT (h = silu_sat8((g·sigmoid_q22(g))>>22 · u)). CPU-verified h2=127=float ref;
the full design now runs the GU producing h2 and the D cascade producing
NONZERO output (N_D=2048: C[896]=243840 — the h2 stays CORE-LOCAL, zero h2 DMA).

**Remaining constraint (aie2p cascade): the cascade_reduce works for ONE pass
but MULTIPLE passes (2 D col-groups) stall** — mc2d (2 cg2, 2 drains): cg0=688128
✓, cg1 untouched. The aie2p cascade is a continuous stream; a 2nd cascade_reduce
call in the same launch stalls at get_scd. So the D GEMM must be ONE cascade
pass, which needs the per-core partial (h2_core @ B_d) to fit L1. For N_D=2048
the partial is 8×2048 int32 = 64 KB (too big alongside the GU buffers). N_D=1024
(n_cg_d=1, 32 KB partial) + GU currently yields no output (separate geometry
interaction still to pin). The p1/p2 two-launch split (h2 via DDR) remains the
reliable path UNTIL the single-pass partial is kept under L1.

## ROOT DESIGN FLAW isolated (2026-08-28): K+N cross-distribution vs cascade

The single-pass design (N_D=1024) produces NOTHING, and the 2-pass (N_D=2048)
produces only cg0. The root cause is a GEOMETRY flaw, not the cascade:

The D GEMM C2 = h2(8×K) @ B_d(K×N_D). The design distributes **both K (each
core holds 1/8 of h2's K) AND N (per-column of_b[c] → each core reads ITS OWN
column's B_d slice) across the 8 cores, but the hardware cascade can ONLY sum
the K-partitions of the SAME output column.** So the cascade sums 8 DIFFERENT
columns' partials — wrong. The 2-pass "cg0 output" was the correct 4-cg GU +
a cascade that happened to produce a plausible-but-wrong value.

THE CORRECT SINGLE-PASS DESIGN:
1. **B_d must be BROADCAST** to all cores (one shared B_d fifo, like of_a) —
   every core reads the SAME B_d column slice, so the cascade sums the 8
   K-partitions into that column's C2.
2. **c2scr = (8×N_D)** per core — the core's K-slice contribution to the FULL
   C2. For N_D=1024 this is 32 KB (fits L1); N_D=2048 is 64 KB (too big).
3. **ONE cascade_reduce pass** over the (8×N_D) sum → the tail writes the full
   (8×N_D) C2. Requires cascade_reduce compiled for DIM_N=N_D (512 chunks for
   N_D=1024, one continuous stream).

The generator's per-column of_b[c] B_d is the flaw — it makes each core a
DIFFERENT column. The fix is a B_d broadcast + (8×N_D) c2scr + a full-N_D
cascade_reduce. N_D=1024 fits L1 (32 KB); N_D=2048 needs the partial chunked
(which reintroduces the multi-pass limit) — so N_D=1024 is the max
single-launch D GEMM without further rework.

## CORRECTED D CASCADE — VALIDATED EXACT (2026-08-28, silicon)
The K+N cross-distribution flaw is FIXED. At N_D=128 (no_gu, h2=1, B_d=1) the
corrected single-pass D cascade writes **C2=2048 everywhere — the exact
expected value** (each core: 4 k-slices × 64 → c2scr=256; 8-core cascade sum →
2048). The fix: (1) each core reads only its OWN h2 k-slices
(ki=cg*8+col, not ki=0..n_k — the GU's h2 is valid ONLY there), (2) B_d is
FULL-WIDTH and the SAME columns for all cores (the cascade sums K-partitions
of one column, not DIFFERENT columns), (3) ONE cascade pass over the (8×N_D)
partial via the wide-N mm + cascade_reduce_*_wide.

Hard-won constraints found this session (all real):
- **AIE2P core = only TWO input DMA channels** — the fused worker reads
  A+B_gu+B_d = 3 streams. Fix: pack A & B_gu into one combined stream
  (matmul_i8_i32_ab reads [A|B] from one element) → 2 channels.
- **XRT MLIR_AIE kernel = only FIVE data buffers** — the 8 per-core AB fifos
  had to become ONE buffer (laid out [core][ki][cg]) → sequence = 3 groups.
- **64 KB core L1** — h2buf shrunk to the core's OWN 4 chunks (2 KB, not the
  full 8×K 16 KB) so c2scr+B_d+combined-A|B fit; AB fifo depth=2 (depth=1
  and depth=3 both fail L1 or handshake).

REMAINING BLOCKERS (not yet finished):
1. **GU combined channel HANGS**: the full GU path (matmul_i8_i32_ab +
   silu_q22) hangs BEFORE the D — C2 stays 0x5A even with --silu-const.
   The D-in-isolation (no_gu) is exact; the GU's combined [A|B] acquire/mm
   loop is the hang. (Earlier a separate of_a+of_b_gu GU ran, but that's the
   >2-channel structure that fails placement.)
2. **N_D>128 single-pass scaling**: N_D must be a multiple of 128; N_D=128
   works (exact), N_D=256 HANGS (wide mm@256 or the 128-chunk cascade). The
   single-pass cascade is bounded to one 128-wide column chunk.

So the correct single-launch fused design is ACHIEVABLE at N_D=128 (validated)
and the architecture is right; the two remaining items (GU combined-channel
hang, N_D>128 scaling) are isolated and well-characterized next steps. The
crash-verified p1/p2 two-launch (h2 via DDR) remains the reliable path for
any N_D until those are resolved.

## GU combined-channel FIFO deadlock (the last isolated blocker, 2026-08-28)
The single-core probe isolates the GU's combined [A|B] channel: matmul_i8_i32_ab
alone in a matched producer/consumer (32 fills, 32 acquires) **DEADLOCKS** —
C stays untouched at depth=1 AND depth=2. Only an over-provisioned producer
(128 fills vs 32 reads) "runs", and then the h2 store + silu_q22 also stall
(h2 untouched). By contrast the D's B_d FIFO (16 KB element, depth=1, matched
4/4) works and the D cascade is EXACT at N_D=128. So the A|B combined FIFO's
handshake (or the mm_ab execute) is the blocker — NOT the D, NOT the silu
arithmetic (which is clamped, no OOB), NOT the K+N fix (proven). The
mis-matched 128/32 "success" (c1=1536, an off-value) was the FIFO over-provision
masking the deadlock. This is a focused remaining item: make the AB FIFO's
producer/consumer handshake pipeline (the p1/p2 generators show the working
pattern) so the GU's combined channel delivers all 128 elements without
stalling.

## GU combined-channel root cause: 1-D FIFO deadlock (confirmed, 2026-08-28)
The isolate is now DEFINITIVE via a trivial-copy kernel (gu_copy_ab, since
removed). A single combined A|B FIFO of element type (8704,) int8 consumed by
one core works at count=1 but **DEADLOCKS at count>=4** — at every depth
(1, 2, 3). A 2-D combined element (AB_tile,64) fails the aiecc kernel-type
check against the (8704,) kernel. By contrast the D's per-core B_d FIFO of a
2-D (64,N_D) element WORKS at count=4. Conclusion: the iron ObjectFifo for a
multi-element 1-D combined stream does not pipeline its acquire/release; the
proven p1 pattern uses SEPARATE A (broadcast) + B_gu (per-core) fifos at
depth=BATCH+1, both 2-D. That is 2 channels for the GU alone — w/ the D's B_d
that is 3, over the AIE2P 2-input-DMA limit. So the single-launch fused design
needs a 2-channel dataflow where BOTH GU (A+B_gu) and D (B_d) work: either a
2-D combined [A|B_gu] element w/ a matching 2-D kernel (aiecc rejects the 1-D
kernel today), or the p1 separate A+B_gu with the D's B_d multiplexed onto a
reused channel. This is the precise, bounded remaining item.

## GU combined-channel deadlock — CLOSED as unsolved (final verdict, 2026-08-28)
Tested the full matrix on the aie2p NPU:
- combined [A|B] FIFO element (8704,) 1-D: works count=1, DEADLOCKS count>=4 (every depth 1/2/3).
- combined [A|B] FIFO element (64,136) 2-D: works count=1, DEADLOCKS count=4.
- the D's per-core B_d FIFO (64,N_D) 2-D: WORKS count=4 (D cascade exact at N_D=128).

So the iron ObjectFifo for a SINGLE merged [A|B] input does not pipeline
count>1, regardless of 1-D/2-D shape. The reason the D side works and the GU
side does not is the merged-element dataflow, not the shape. The p1 pattern
(separate 2-D A broadcast + B_gu, depth=BATCH+1) pipelines fine but is 2
channels for the GU alone; +B_d = 3, over the AIE2P 2-input-DMA limit.

VERDICT: the single-launch fused GU→h2→D (zero-h2-DMA) is NOT achievable with
the current aie.iron ObjectFifo + 2-input-DMA constraint without either (a) a
2-channel dataflow where the D's B_d is multiplexed/reused over one of the GU
channels, or (b) a different iron FIFO primitive that pipelines a merged
element. Both are real engineering (not a 1-line fix). The validated, correct
architecture (K+N cross-distribution fix, q22 silu) stands and is silicon-verified
for the D at N_D=128; the GU single-launch integration is the hard, presently-
unsolved remainder. p1/p2 two-launch remains the production-correct path.

## Option (a) BUILD ACHIEVED but silicon-REGRESSED — 2026-08-28 (agent attempt)

Implemented option (a): **split A/B into two separate 2-D single-stream fifos**
so no merged element (which deadlocks at count≥4, BUG-006):
- ch0 = `of_a[c]` (8,64) A-tile, per core, from a shared A_bo
- ch1 = `of_b[c]` (64,128) B-tile, per core — carries B_gu tiles (GU) THEN
  B_d tiles (D) sequentially (132 slots), so 2 input channels total
- GU kernel switched `matmul_i8_i32_ab` → `matmul_i8_i32` (separate A/B);
  the D cascade + q22 silu + K+N fix unchanged.

**Status:** the design **compiles** (`build_iron_cascade.sh`, 76 KB xclbin, 17
fifos A0-7/B0-7/C2_tail, 8 cores, 7 cascade flows), and the MLIR is structurally
correct (buffer mapping A=16384, B=8650752, C2=1024 → groups 3/4/5). But the
**silicon run returns C2=0x5A** (tails never write C2): the shared `of_b` fifo's
D-cascade writeback does NOT fire, even for a `--no-gu --h2-const 42` D-only
probe (expected 86016).

**Isolated control (same toolchain, same day):** the ORIGINAL generator
(combined-AB GU + dedicated `of_b_d` fifo) D-only design **is verified exact** on
silicon (C2=2048 everywhere, bad=0) with the same aiecc/XRT/`/dev/accel0`
environment. So the regression is the **shared-B fifo**, not the D math or the
environment. A dedicated-`of_b_d` + separate-`of_a`/`of_b` (3 fifos for
GU+D) **fails to place** (3 input streams > AIE2P 2-input-DMA limit, BUG-007).

**Conclusion:** option (a) as implemented (B_gu and B_d on ONE `of_b` fifo)
builds but does not silicon-verify; making it work needs either a different way
to sequence B_gu/B_d on one channel, or a dedicated-B_d that fits the 2-channel
budget (e.g., moving `A` to a broadcast fifo with a single prod endpoint so the
GU uses 1 channel and B_d gets its own — untested). The hard blocker from the
verdict stands and is now pinned to a concrete, reproducible 2-channel attempt.
The split-A/B generator (uncommitted candidate) + probes
(`npu_fused_smoke.cpp`, `orig_d_probe.cpp`) are in the tree for the next round.
p1/p2 two-launch remains the production path.

## Shared-B regression ROOT-CAUSED to a placement/DMA descriptor issue (2026-08-28)

Re-verified on a CLEAN device (rmmod+modprobe amdxdna, 0 procs, no IO_PAGE_FAULTs):
the shared-B design (of_b carrying B_gu then B_d) returns C2=0x5A in `--no-gu --h2-const 42`
D-only, while the ORIGINAL dedicated-`of_b_d` design returns C2=2048 exact (h2=1) on the SAME
clean device. So the regression is real, not a device-state false negative.

**Diff of the two generated `design.mlir` (orig working vs shared broken):** the D path is
**byte-identical** — `@Bd`≡`@B` (same `memref<64x128xi8>`, same shim ports 14-21, same depth 1),
the cascade_reduce calls and the C2_tail drain are identical (the drain reads `%arg2: memref<1024xi32>`
flat in both, just a different arg index from the sequence reorder). The ONLY differences are:
(1) the GU fifo element `@AB`(8704B) vs `@A`(512B, 8×64), (2) the GU kernel decl
`matmul_i8_i32_ab` vs `matmul_i8_i32`, (3) the sequence buffer order/sizes (AB/C2/B_d vs A/B/C2).
So the regression is a SILICON-LEVEL placement/DMA-descriptor difference triggered by the smaller
512B A fifo (or the sequence reorder), NOT a logic error in the D.

**Fix attempts (all FAILED — C2 stays 0x5A):** `--alloc-scheme=bank-aware`, `--placer=sa_placer`,
B_d fill 2-D→1-D flat tap, B_gu fill 2-D→1-D flat, A fill 2-D→1-D flat. None moves C2 off 0x5A.

**Next to crack it:** (a) make the cascade aiesim-able (BUG-010 — the chess `--aiesim` rejects the
peano cascade object) so the D writeback can be cycle-debugged, or (b) diff the aiecc-emitted
XCLBIN DMA descriptors between the two designs (needs a newer `xclbinutil`), or (c) co-worker
fusion expertise on the 2-channel multiplex. The cascade math itself is silicon-verified (dedicated-B
D is exact); only the 2-channel GU+D integration is blocked.
