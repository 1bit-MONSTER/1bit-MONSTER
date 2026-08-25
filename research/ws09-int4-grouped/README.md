# ws09 — int4 GU weights for the fused decode path (issue #1769)

**Status: design decision made (2026-08-23), backed by a CPU gate on the real
zaya1-8b.q4nx weights. Next: kernel round on strixhalo.**

## The question

The fused decode's dominant cost is weight DMA (~12.6 MB/layer GU int8, ~60 ms/tok
DMA-bound). Halving it via int4 packed weights (like Q4NX, 0.5 B/value) was the
goal. PR #1813 proved the unpack stage (`unpack_i4_b`, `vldb.unpack`) works
bit-exactly on hardware, but per-column int4 RE-quantization of the Q4NX weights
capped MoE-FFN corr at 0.972 — tokens flip at position 0.

The issue's proposed fix was a **per-group-scale kernel restructure** (per-K-chunk
accumulation with per-chunk scale dequant). This workbench tested that hypothesis
on CPU with the real weights — and found a better answer.

## The finding (measured, real weights, real layer-1 activation)

`engine/npu/tests/test_i4_grouped_fused.cpp` — three GU quantization variants,
byte-identical downstream (SiLU LUT + qn_s + int8 D), corr vs float FFN:

| variant | GU bytes/layer | GU corr | h2 corr | FFN corr |
|---|---|---|---|---|
| A. int8 per-section (current fused) | 8.4 MB | 0.9991 | 0.9980 | 0.9978 |
| B. int4 per-column re-quant (PR #1813) | 4.2 MB | 0.9874 | 0.8155 | 0.8174 |
| C. int4 per-(32-row,32-col)-group re-quant | 4.2 MB | 0.9890 | 0.8258 | 0.8283 |
| **D. raw Q4NX nibbles + per-row scales, on-chip dequant** | **4.2 MB** | **0.9999** | **0.9998** | **0.9996** |

Consistent across all sampled (layer, expert) pairs: D = **0.9995–0.9997** FFN corr,
B/C = 0.82–0.88. Two conclusions:

1. **The per-K-chunk restructure (option C) does NOT reach int8 quality.** Even
   with group scales, re-quantizing to a 16-level grid loses too much; the SiLU
   stage amplifies ~1% GU error to ~18% h2 error (small-gate × large-up elements).
   Building that restructure would have been wasted days.
2. **The win is "int4 storage + on-chip dequant to the existing int8 contract"**
   (D): stream the raw Q4NX nibbles + per-row bf16 scales, reconstruct
   `B'' = round(q4(i,j)·s[i][j/32] / S_col[j])` in-kernel, and feed the unchanged
   int8 mmul. Zero re-quantization — the reconstruction is EXACT (Zaya mins = 0),
   and S_col (per-column int8 scale) is finer than the current per-section pack,
   so D is actually *more* accurate than the current fused path at half the bytes.

## Why D works (and B/C can't)

- Q4NX stores per-(row, 32-col-group) bf16 scales (`scales[lr*8+g]`, g = col/32).
  The small weights stay on-grid within their group; a single per-column scale
  over K=2048 (B) or a 32-row block (C) cannot carry that, so most small weights
  quantize to zero and the SiLU·up amplification destroys the FFN output.
- D consumes the file's own q4+s, so `W = q4·s` exactly (mins=0). Re-encoding to
  int8 with a per-column S_col is the same math as the host int8 pack (corr 0.9995)
  — the mmul never sees int4, only the DMA does.

## Kernel design (next: hardware round on strixhalo)

The existing fused kernel structure is UNCHANGED — only the B-load stage changes:

```
// per B tile (64 K-rows x 128 cols), int4-packed:
load nibbles [64x128/2 bytes] + per-row scales [64 x 4 bf16]   // +512 B/tile metadata
for each (i, j):
    q4  = unpack nibble (vldb.unpack, sign=1)                  // existing unpack_i4_b
    w   = q4 << 4 folded * s[i][j/32]  (or float: q4 * s)
    B'' = sat8(round(w / S_col[j]))                            // per-column scale
    mmul consumes B'' as today                                 // int8 x int8 -> int32
```

- **Scale metadata**: per B tile, 64 rows × 4 col-groups = 256 bf16 (512 B), streamed
  alongside the tile (the gs-header tap pattern already exists). S_col: per column
  (128 per tile) bf16 — also streamed. Total metadata ≈ +12.5% of the int4 tile.
- **Register pressure**: no change (B'' lives in the load pipeline; the mmul
  accumulators are untouched).
- **Rounding contract**: `round(w/S_col)` must match the host pack bit-exactly
  (int4→int8 boundary). Prefer integer arithmetic: `(q4·s_bf16) → float → roundf`
  with a documented rounding; the CPU gate pins it.
- **Host changes** (`npu_engine_i8ctx_inc.h`): pack the GU B-tiles from the raw
  Q4NX (nibbles + per-row scales) instead of the dequant-float re-quant; stream
  S_col per tile; keep the row-major shadow for the amax pass (can now be the
  exact reconstructed W, not a re-quant).
- **Kernel changes** (`mm_kernel_reference.cc` + `n1_core_fused_gu_silu_d*.py` +
  `build_zaya_fused.sh`): replace the B-tile load with nibble+scale loads and the
  dequant stage; update the #1777 DMA-signature regexes (B-tile stride 8192→4096
  for int4 tiles + the scale-tile offsets).

## What the DMA win is

GU weights halve: 12.6 → 6.3 MB/layer streamed per token. At the measured 4–10 GB/s
coherent host-DMA, that's ~30 ms/tok saved ≈ the 6.2 → 8–10 tok/s target from the
issue. D weights (2048×2048 int8) could follow the same trick later (their corr
propagates linearly — the h2 amplification is GU-specific).

## Host packer (validated, byte-pinned)

`engine/npu/src/gu_i4_pack.h` (`pack_gu_fused_i4`) + `q4nx_raw.h`. BO regions
per expert (K = H = 2048, N = 2·n_ff = 4096 interleaved):

- **Region A — nibbles [K·N/2 = 4 MB]**: tiles (ki·32 + nt)·4096 B; tile byte
  s4 = i0·512 + i1·32 + i2·4 + i3/2, row = ki·64+i0·8+i2, col = nt·128+i1·8+i3,
  even element (i3 even) in the LOW nibble (i4_pack.h contract).
- **Region B — row scales [(K/32)·N·2 = 512 KB]**: per (K-colgroup i/32, col j)
  bf16 = scl[gate/up row of j][i/32] — the exact per-element scale, stored as
  bf16 (what the kernel sees). Index `row_scales[i/32][j]`.
- **Region C — S_col [N·2 = 8 KB]**: per-column int8 scale amax/127 (bf16).
- **Region D — gs header** (existing, per-token ag/qn_s fold, unchanged).

Total ~4.52 MB vs 8.4 MB int8 (−46%). The CPU gate's variant E emulates the
kernel dequant from THESE regions and verifies **byte-identity vs the packer's
B_shadow: 8,388,608/8,388,608 exact** on every sampled (layer, expert), with
FFN corr 0.9995–0.9997 (matches variant D). The packer layout is the pinned
contract for the kernel round.

## Files

- `engine/npu/src/q4nx_raw.h` — raw Q4NX accessor (nibbles + per-row scales + zp).
- `engine/npu/src/gu_i4_pack.h` — the int4 fused GU packer (regions A/B/C).
- `engine/npu/tests/test_i4_grouped_fused.cpp` — the CPU gate (variants A–E,
  real activation; variant E = packer roundtrip + byte-identity). Build:
  `g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src ... -o /tmp/t && /tmp/t zaya1-8b.q4nx [layer] [expert] [activation.bin]`
- `engine/npu/tools/zaya_cpu_runner.cpp` — `NPU_DUMP_MOE_INPUT`/`NPU_DUMP_LAYER`
  hooks to dump real MoE inputs for the gate.
- Existing (already merged): `i4_pack.h` (#1793), `unpack_i4_b` (#1813).

## Kernel interface (for the strixhalo round) — v2, per-tile contiguous

**Region B is now per-tile contiguous** (validated: byte-identity 8,388,608/8,388,608,
BO writer exact): tile (ki, nt) at `(ki·32+nt)·512`, `[group_in_tile 2][col-in-tile 128]`
bf16. The kernel reads ONE linear 4864 B tile = `[nibbles 4096][row scales 512][S_col 256]`
(gu_i4_bo_size now returns K·N/2 + n_tiles·512 + N·2 = 4 MB + 512 KB + 8 KB).

**Key simplification — per (8,8) mmul chunk**: nibble bytes are contiguous
(`byte_off = i·512 + jt·32 + i2·4 + i3/2` → chunk = 32 B), and within a chunk
all 8 rows share one K-group → the row scales reduce to 8 values (one per col)
and the dequant becomes:

```
ratio[c] = s[group][c] / (16 · S_col[c])     // 8 precomputed bf16 per chunk
B''[r][c] = sat8(round( (q4<<4)[r][c] · ratio[c] ))     // ONE bf16 mult/element
```

64 contiguous bytes per chunk (32 nibbles + 16 row scales + 16 S_col) — a
single linear DMA, and the dequant is a bf16 vector multiply with a
stride-8 ratio broadcast (cheap vs the mmul MACs; keeps the kernel DMA-bound).

1. **B-tile loads**: linear 4864 B per tile; kernel dequantizes per (8,8) chunk
   as above (reuse `unpack_i4_b`'s vldb.unpack for the nibble→q4<<4 stage).
2. **S_col/ratio**: computed host-side per tile into region B' — the packer
   writes `ratio` instead of `s` (bf16) so the kernel has zero divisions.
3. **silu stage**: per-column S'[j] read (per-token fold, region D'), replacing
   the gs[0]/gs[4] per-section reads — the only arithmetic change to
   `silu_quant_i8_fused`.
4. **amax pass unchanged**: `host_h2_amax_qn_s` with `guB = pack.B_shadow`,
   `guGs = pack.scol`.
5. **#1777 signatures**: B-tile stride 8192 → 4864 + the fold-header tap.

## Kernel-round status (2026-08-24) — scalar dequant works, bf16 vector is the path

- **Scalar dequant kernels exist and compile** (aiecc aie2p): #1822's separate
  pass (`i4_dequant_kernel.cc`) and the inline variant `matmul_i8_i32_i4`
  (PR #1824, same canonical arithmetic). Both are CORRECT but ~6–8 ms/launch
  scalar compute — that eats the DMA win, so vectorization is mandatory.
- **Critical toolchain finding (#1822)**: fp32 vector FMUL is NOT legalizable
  by peano (`G_FMUL on <N x s32>` fails). The vectorized dequant must use
  **bf16 arithmetic**.
- **Verified on strixhalo (2026-08-24)**: `aie::vector<bfloat16,64> va*vb`
  COMPILES clean through peano; `aie::to_fixed<int8>(v, shift)` exists
  (bf16→fixed with rounding+saturation); `aie::to_float<float>(v)` compiles at
  the call site (the fp32-mult blocker is the FMUL, not the conversion).
- **Remaining vector-API details to pin**: int8→bf16 conversion (no
  `from_vector`; `cast_to`/`to_float`+bf16 paths), the stride-8 per-column
  ratio broadcast (`insert` signature), and rounding parity of `to_fixed`
  vs the gate's `roundf` (the gate pins `w16 = q4<<4` exact,
  `ratio = (s*0.0625f)/S_col`, `round(w16*ratio)` — a bf16 path must match
  within ±1 int8 at round boundaries; corr gate tolerates that).
- **Generator**: `n1_core_fused_gu_silu_d_p1_i4.py` (WIP) wires the per-tile
  4864-B taps (nibbles + s + S_col, ONE linear chunk per (64,128) tile) →
  `matmul_i8_i32_i4` (inline unpack+dequant+mmul) + the per-column fold silu
  `silu_quant_i8_fused_i4`. Next: bf16-vectorize the dequant, build the
  xclbin, NPU corr + tok/s.

## CPU contract fixes (2026-08-24, after #1824) — gates GREEN

- **`B_shadow` byte-identity is now kernel-exact**: `B_shadow` is computed with
  the bf16-rounded S_col read from the tile (`matmul_i8_i32_i4` taps
  `scp = tile + 4608 + col*2`), NOT the full-precision float — the float
  version caused 292,796/8,388,608 ±1 flips at round boundaries. The kernel
  arithmetic (`ratio = (bf16(s)*0.0625f)/bf16(S_col)`) is pinned by
  `test_i4_dequant.cpp`: **8,388,608/8,388,608 byte-identity** on real
  zaya1-8b.q4nx weights, layers 1/3/5/9, multiple experts.
- **`test_i4_dequant.cpp` + `test_i4_grouped_fused.cpp`** consume the v2
  per-tile 4864-B chunk layout (packer `tiles`, `TILE_TOTAL`) exactly as the
  kernel does; variant E FFN corr 0.9991–0.9997, `GATE PASSED`.
- **`silu_quant_i8_fused_i4`** added (per-column fold S'[2p]/S'[2p+1] instead
  of the section header gs[0]/gs[4]) — parity-pinned against the CPU gate.
- **Host wiring**: `update_fused_header_i4` writes the per-token fold at
  `gu_i4_bo_size` (after the per-tile chunks, matching the generator's gs
  tap), and `zaya_decode.cpp` selects the `final_i8_MOE_GUSILU_i4_zaya`
  xclbin under `NPU_FUSED_I4=1`. `build_zaya_fused.sh` gained an int4 mode
  (`NPU_FUSED_I4=1`) with the 4864-byte B-tile #1777 signature checks.

## Risks / next steps

- **aiecc lowering** of the scale-multiply + requant in the B path (the #1813
  `vldb.unpack` proved the nibble load; the per-element float mult + round is new).
- **Rounding parity** host-vs-kernel at the int8 boundary (CPU gate pins it).
- Verify on strixhalo: corr gate vs the host emulation, then tok/s.

## v59 silu kernel round (2026-08-24, issue #1844) — the v50/v51 Q22 silu was broken; fixed + CPU-gated

**Blocker #1844 reproduced and root-caused on CPU** (test_i4_silu_q22.cpp —
the missing gate: the v50/v51 fixed-point silu was NEVER CPU-tested against
the float reference; it went straight to strixhalo and measured corr ≈ -0.02).

### Root cause of the corr ≈ -0.02 / "host h2=12 → NPU (0,0)" zero pairs

The v51 `silu_quant_i8_fused_i4` computed `gQ22 = c1*fold` and
`uQ22 = c1*fold` at a fixed Q22 with a per-tile fold from the tile MAX scale:

1. **int32 overflow**: `uQ22 = c1*fold` wraps for `|u| > 512`. The up
   pre-activations reach ±600-2000 (h2 = silu(g)·u saturates only at
   |h| ≥ 127.5, so |u| = |h|/silu(g) is unbounded for small gates). The
   wrapped `uQ22` produced garbage — including the exact reported pattern
   `host pair (-12,-7) → npu (0,0)` (reproduced in the gate: g=0.03,
   u≈800 → host h2=12, v51 h2=0).
2. **Fold-range zeroing**: the per-tile Q was chosen from the tile MAX scale,
   so the "small weights majority" columns (S' a decade+ below the max)
   rounded to `fold = 0` → `gQ = 0` → `h2 = 0` for every pair of those
   columns, whatever the true gate.

### The fix (v59) — overflow-free, precision-bounded, pure int32

- **Per-tile fold Q from the tile MIN scale** (`Q = 22 - s`,
  `s = max(0, 15+ceil(log2(minS')))`): the smallest fold keeps ≥ ~64 bits,
  the fold range inside a tile no longer rounds to zero.
- **Exact per-column c1 bounds, host-precomputed** (no kernel division):
  `boundG[j] = (2^31-1)/|fold[j]|` and
  `boundU[j] = 4·((2^31-1)/|fold[j]|)+3` (the +3/×4 is the (c1u>>2)
  pre-shift: `uQ = (c1u>>2)·fold` covers |u| ≤ 2^(33-Q)). The kernel clamps
  `c1` to these → every product is overflow-free by construction.
- **Saturating hQ16 product**: `h·2^16 = siluF·uF` with a per-pair
  power-of-2 clamp of |uF| against |siluF|; clamped pairs always have
  h ≥ 2^14 → the sat8 output is exactly the float reference's.
- **Full-precision silu for small gates**: `siluQ` uses the split
  `(gQ>>11)·(σQ22>>11)` only for |gQ| ≥ 2048; the small-|gQ| branch
  `(gQ·(σQ22>>11))>>11` keeps the gate precision (the old code tested
  `gQ < 2048`, wrongly routing large NEGATIVE gQ into the small branch whose
  `gQ·(σQ22>>11)` overflowed — Python's unbounded ints masked it; the C++
  gate caught it).
- **Shared dual-compiled arithmetic**: `silu_pair_q22` + the Q22 σ LUT moved
  into `silu_quant.h` (kernel + CPU gate include the same header — no drift,
  and it removes the #1845 duplicate-LUT hazard).

### CPU gate (`engine/npu/tests/test_i4_silu_q22.cpp`, runs on x86, no NPU)

- `g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src \
  engine/npu/tests/test_i4_silu_q22.cpp -o /tmp/t && /tmp/t`
- Emulates the kernel bit-exactly vs the float silu_quant reference on the
  realistic envelope (gate g ~ N(0,1.2), up u ~ ±10^U(-0.5,2.8), S' log-
  uniform over ~5 decades, c1 = g/S') + an adversarial corner sweep at the
  old overflow boundary (|u| ~ 550-2000, tiny gates).
- Results: synthetic corr **0.999970**, 99.07% within ±1, worst |Δh2| = 5;
  adversarial corr **0.999956**, 98.0% within ±1, worst 4. (v51 measured:
  corr ~ 0.75 on the same set, worst 254.)

### Pad layout (per 8192-B tile, written per token by `update_fused_header_i4`)

```
[6144, 6148)  Q        int32 per-tile fold Q
[6656, 7168)  foldG    128 int32 = round(S'*2^Q)
[7168, 7680)  boundG   128 int32 = (2^31-1)/|foldG|
[7680, 8192)  boundU   128 int32 = 4*((2^31-1)/|foldG|)+3
```
stashed by the last `matmul_i8_i32_i4` at 0x6000 (Gg_0) for the silu; the
v48 section-scale floats that used to live at [7168,7184) were vestigial
(nothing read them; the int8-fallback silu reads the B-stream gs object).

### Remaining for the strixhalo kernel round

1. Build `n1_core_fused_gu_silu_d_p1_i4.py` with the v59 kernel, run the
   on-NPU corr gate vs the host emulation (expect the h2 corr ≈ 0.9995-0.9999
   range of the ws09 design instead of -0.02), then tok/s.
2. The dequant (`matmul_i8_i32_i4`) is still the scalar variant — the bf16
   vectorization (README §kernel-round status) is the perf follow-up.

### v59 build/test hardening (follow-ups from the kernel round)

- **`gu_i4_pack.h`**: the pad metadata writer is now the shared
  `write_silu_pad_meta` (Q/foldG/boundG/boundU + the bf16 fold) — the
  packer's first-launch init (ag=1, qn_s=1) and `update_fused_header_i4`
  (per token) call the SAME function, so the host math cannot drift. The
  first launch before any header update now dequantizes sanely (Q=22,
  folds = S_col).
- **`test_i4_silu_q22.cpp`** gained the real-data mode: it loads a real
  zaya1-8b.q4nx (manifest + `read_q4nx_raw`), slices an expert, packs via
  `pack_gu_fused_i4` (kernel-exact B_shadow + S_col), computes the int8 GU
  GEMM c1, the per-token qn_s, and gates the kernel-vs-float silu on the
  real weight/scale distribution:
  `g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src
   engine/npu/tests/test_i4_silu_q22.cpp -o /tmp/t
   && /tmp/t zaya1-8b.q4nx [layer] [expert] [activation.bin]`
- **`build_p1i4.sh`**: post-build symbol check on the merged kernel object
  (`matmul_i8_i32_i4`, `silu_quant_i8_fused_i4`, `unpack_i4_b`, `zero_i32`
  must all be present — a stale/partial object fails loudly, issue #1841);
  duplicate-definition check (the kernel .cc must not define
  `silu_sigmoid_q22`, issue #1845); and the #1842 build-time address check
  that parses `input_with_addresses.mlir` for Gg_0 @ 0x6000, C1 @ 0xE000 and
  the H2 fifo @ 0xF000 (warns if the aiecc version emits no address map).
- The stale on-disk `generators/mm_32x64x128.o` (only `zero_i32`) was
  removed — it is `.gitignore`d and the build regenerates it in the workdir.
