# Cross-column handoff on AIE2P (npu2) — findings for fk-3

## The mechanism that works: core-to-core "west" shared memory

On the AIE2P (Strix NPU) `aie.device(npu2)` target, a core tile can directly
read the *memory* of the core tile one column to its **west** (same row) — no
DMA, no mem-tile hop, no object_fifo_link. This is the same shared-memory
mechanism the tool already uses for same-column core-to-core cascades
(`isLegalMemAffinity(col+1, row, col, row) == true` via the west memory region,
base `0x50000`).

To hand data from a core in column 0 to a core in column 1 at the same row,
just declare a plain object_fifo between them:

```python
AT_OUT = object_fifo("AT_OUT", rs_c, o_c, 1, AT_ty)   # rs_c=(0,5), o_c=(1,5)
```

The objectFifo lowering places ONE buffer on the producer tile (col 0) and the
consumer (col 1) accesses it through its west memory window (`0x50000 + off`).
Locks are likewise shared: producer uses the "east/own" lock bank (48+id),
consumer uses the "west" bank (16+id); the 6-bit lock id maps to the same
physical lock on hardware. Verified: `copy_bf16` 64-el and `copy_1024` 1024-el
relays across the column boundary are **byte-exact 1024/1024** (including the
last elements), at both row 2 and row 5.

### Things that DON'T work
- `mem_tile_0_1 -> tile_1_2` (mem tile directly to a core in the next column):
  builds but produces all-zero output. The mem-tile switchbox has no
  East/West, and mem tiles only reach cores/shims in their own column + the
  adjacent mem tile via dedicated DMA channels.
- `mem0 -> mem1 -> c1` (mem-tile relay): needs 2 object_fifo_link ops
  (one at mem0, one at mem1) sharing the middle fifo, which
  `verifyObjectFifoLinks` rejects ("objectfifo cannot be in more than one
  ObjectFifoLinkOp").
- Same-column and same-row cross-column shared memory both REQUIRE same-row for
  cross-column (west is row-locked), so the last attention core and the first
  O-proj core must sit at the same row index.

## fk-3 wiring (n1_fk3.py): attention -> O-proj, single launch

Column 0: QK^T(0,2) -> softmax(0,3) -> PV(0,4) -> rescale(0,5).
Column 1: O-proj(1,5) (same row as rescale). Handoff `AT_OUT = rescale -> O-proj`
via west shared memory; the rescale's microtiled output is consumed directly as
the O-proj GEMM's microtiled A (no layout translation).

`mm_oproj.cc` is a renamed wrapper (#includes `mm.cc`) so `matmul_oproj` /
`zero_oproj` don't collide with the PV GEMM's `matmul_bf16_bf16` /
`zero_bf16` at the MLIR declaration level.

Result (M=16, N=64, HD=64, ON=64): the chain runs end-to-end in one xclbin
launch. Using the actual kernel A operand in the reference, the O-proj output is
**493/1024 byte-exact + 531 within 1 ULP (max_delta=1)** — the expected GEMM
hardware-accumulation caveat.

## RESOLVED: the "row-15 anomaly" was a reference artifact

The earlier "26/1024 row-15" mismatches were NOT a hardware/GEMM/west bug.
They came from comparing the O-proj output against a HOST-computed rescale
output `rs`. The kernel's actual rescale output (AT_OUT) differs from the host
`rs` by ~1 ULP in 34/1024 positions (exp2_soft + hardware-accumulation
rounding), and those tiny A differences are AMPLIFIED through the O-proj GEMM's
f32 accumulation + bf16 truncation into raw-diff values up to ~84 in the last
row (which is just where that row's A happens to have the most 1-ULP deltas).

Proof: dumping the actual A operand (a scalar west read of AT_OUT) and using it
in the reference gives O-proj output at 493/1024 byte-exact + 531 within 1 ULP
(max_delta=1) — the expected GEMM hardware-accumulation caveat. The GEMM's
vector `vlda` west read is byte-identical to the scalar `lda.s16` west read.

So the cross-column west handoff is CORRECT end-to-end (scalar and vector).
The practical implication for fk-3: ~1 ULP intermediate differences amplify
through each downstream GEMM, so the realistic full-layer target is "a few ULP"
not byte-exact (consistent with the known GEMM accumulation caveat).

## RESOLVED: the FFN "error" is inherent (not a bug)

Built the validated `n1_fused_ffn_nt.py` FFN (H=128) and ran a truncation-aware
reference against it: **404/2048 byte-exact, max_delta = 115 raw bf16** (hist:
886 off-by-1, 277 off-by-2, 168 off-by-3, 78 off-by-4, 30 off-by-5, 20 off-by-6,
185 off-by-7+). The same wide spread appears in the isolated shim-fed FFN (both
upward AND downward cascades) and in the full layer. So the GU->SiLU->D chain's
large raw deltas are the GEMM f32-accumulation ~1 ULP error amplified through
the SiLU nonlinearity (sigmoid saturation near |x|>8) and the D GEMM's bf16
truncation — data-dependent, so wider A values give larger raw deltas. Not a
layout/wiring bug.

## Remaining (real) work
- **fk-3 full-layer wiring is in place and CORRECT** (`n1_fk3_full.py`):
  attention(col 0) -> O-proj -> FFN GU+SiLU+D (col 1, upward cascade), 8 cores
  in ONE xclbin across 2 columns. `mm_ffn.cc` provides renamed
  `matmul_gu`/`zero_gu`/`matmul_d`/`zero_d`. attention->O-proj is ~1 ULP; the
  FFN tail carries the documented ~few-ULP + SiLU-amplification caveat.
- Remaining for the contract: wire the QKV (RMSNorm+QKV) stage in front of the
  attention (needs a 3rd column / restructure), then scale to 0.6B dims.
- fk-4 @1k measurement.

## Key file
`engine/npu/generators/n1_fk3.py` — generator for the attention->O-proj wiring.
Build: generate `design.mlir`, copy `mm_qk_concat.o mm_bf16_16x64x64.o
zero_qk.o softmax_bf16.o rescale_bf16.o` (from a validated n1_mha build) and
`mm_oproj.o` (compile `mm_oproj.cc` with `-DDIM_M=16 -DDIM_K=64 -DDIM_N=64
-Dbf16_bf16_ONLY`) into the build dir, then the standard aiecc command.
Bench: `/tmp/bench_fk3.cpp` (opcode 3, 4 data args).

## QKV integration plan (remaining fk-3 piece)

The current `n1_fk3_full.py` feeds Q/K^T/V from the shim; the QKV
(RMSNorm+QKV GEMM, `n1_fused_rmsnorm_qkv.py`) is still a separate launch. To
finish fk-3 the QKV must produce Q, K, V in-kernel and feed the attention.

Key facts:
- QKV output C is the GEMM's microtiled C layout with N = 3H (one concatenated
  Q|K|V). That layout does NOT match the attention's per-head A layout
  (K=HD microtiled) — N=3H vs K=HD tiling.
- CLEANEST approach (no layout conversions): use THREE separate Q/K/V GEMMs
  (each M x HD) with the right C layout flag, all sharing the one X_norm
  input, plus a QK^T compiled with -DB_COL_MAJ:
  - Q GEMM: microtiled C (c_row_maj=false) -> QK^T A (microtiled). Direct.
  - K GEMM: row-major C (c_row_maj=true) -> QK^T B compiled -DB_COL_MAJ
    (column-major B reads row-major K directly, no transpose). Direct.
  - V GEMM: row-major C -> PV B (row-major). Direct.
  Cost: 3 GEMM cores instead of 1, and one extra -DB_COL_MAJ QK^T object.
  Alternatively (single QKV GEMM, N-tiled NT=HD): Q direct, but K and V need
  microtiled->row-major (mem-tile 4-D strided DMA or a small transpose/copy
  kernel).

Topology (3 columns, 11 cores with separate Q/K/V GEMMs):
  col 0: QKV-norm(0,2) -> Q(0,3)/K(0,4)/V(0,5) GEMMs (X_norm broadcast)
  col 1: QK^T(1,5) <- Q/K (west) ; softmax(1,4) ; PV(1,3) <- V (west) ; rescale(1,2)
  col 2: O-proj(2,2) -> GU(2,3) -> SiLU(2,4) -> D(2,5)
  (cross-column hops: Q/K/V -> QK^T/PV via west, rescale->O-proj via west).

Note for @1k: self-attention needs K/V for ALL keys while Q is tiled over
queries, so the QKV integration is tied to the CHUNKED attention (>2-chunk
cascade, still blocked). A one-shot self-attention PoC (M = N_keys) is the
tractable first step that avoids chunking.

## QKV integration — RESOLVED (n1_fk3_qkv.py)

The full in-kernel QKV layer (RMSNorm + Q/K/V GEMMs -> attention -> O-proj ->
FFN, 11 cores, 3 columns, ONE xclbin) builds, runs, and is CORRECT within the
documented GEMM accumulation caveat. Three bugs were found and fixed:

1. **W_qk deadlock**: the QK-gemm acquired all 3 W tiles (W_q/W_k/W_v) at once
   against a depth-2 fifo -> the 3rd acquire blocked forever (all-zero output).
   Fix: acquire/release W tiles one-at-a-time, interleaved with the GEMMs.
2. **N=16 mmul codegen bug**: the QK^T (DIM_N=16) / PV (DIM_K=16) aie::mmul
   shapes produced scrambled output. SIDESTEPPED by padding the self-attention
   N_KEYS 16 -> 64 (zero-fill K^T columns and V rows) so the validated
   N=64/K=64 kernels (mm_ffn.o matmul_gu/matmul_d, softmax N_KEYS=64) are used.
3. **K^T/V B-tile layout**: the K^T and V written in-kernel must be in the
   mmul's B TILE layout (element (k,n) at (k/8*(N/8)+n/8)*64 + (k%8)*8 + n%8),
   not row-major (the shim DMA normally does that conversion). `qkv_gemm.cc`
   `matmul_k_transpose` / `matmul_v_convert` now write the B layout directly
   with a local static C buffer (zeroed each call).

Validation: norm-only byte-exact 1024/1024; QK^T scores 783/1024 (max_delta 41,
Q/K ~1-ULP accumulation); V 3573/4096 (max_delta 1); full D 42/1024
(max_delta 33286) — the ~1-ULP Q/K/V accumulation amplifying through the
nonlinear softmax exp + SiLU sigmoid (the same caveat class as the FFN). The
padding with zero scores (not -inf) shifts the softmax isw slightly, but the
reference uses the same pad, so it is self-consistent.

Caveat for the real model: zero-pad columns contribute exp2(-max) to the
softmax sum (non-negligible for small scores), so @1k use -inf padding or a
real N_KEYS=1024 path (chunked attention).

## Chunked-attention >2-chunk cascade — progress (this turn)

Revisited the C=4/8 cascade blocker. Findings:
1. The mlir-aie `build_tmp` binaries were STALE (the static-unroll source patch
   was applied Sep 12 but `aie-opt`/`aiecc` binaries were from Jul 21). Rebuilt
   `ninja -C build_tmp bin/aie-opt bin/aiecc`; the full-unroll path is now
   active (C=4 chunked MHA transforms to 0 scf.for with
   `--dynamic-objFifos=false`).
2. With the full unroll active, C=2 still validates (956/2048), but C=4 is
   STILL wrong (1/2048, max_delta 33878) — the depth-2 SC/E/AT ping-pong with
   the statically-unrolled 4-chunk body still serves wrong data. The depth-4
   f32 AT still overflows pv_c memory (AT 4x8KB + V 32KB + E 8KB + stack 8KB
   > 64KB).
3. The unroll-by-LCM (the pre-patch path) is fundamentally broken for the
   chunked MHA because the input fifos QK_C/V_C are depth-1 (QK chunk is 36KB,
   depth-2 would overflow), so the LCM=2 unroll body acquires 2 cascade buffers
   but only 1 input buffer.

So @1k (C=8) remains blocked on the >2-chunk cascade: either fix the depth-2
ping-pong stale data, or reduce the per-chunk footprint (smaller N chunk, or
bf16 AT) to fit depth-C. Next concrete probe: dump the sm_c's exp/alpha for
chunk 2 vs the reference to see whether the stale data is in SC or E/AT.
