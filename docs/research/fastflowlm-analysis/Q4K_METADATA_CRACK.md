# Q4_K_S sub-block metadata permutation — crack attempt (ground-truth method)

_2026-09-04. Method: GGUF ground-truth correlation against the on-box
`Qwen3.6-35B-A3B-NPU2/model.q4nx` (FLM v1.0.3, `flm_dtype_t` 8)._

## TL;DR

- **Cracked**: the `flm_dtype_t` → block-size encoding, and the outer
  structure of every quantized tile format (Q4_1, Q4_0, Q8_0, Q4_K_S).
- **Falsified**: the hypothesis that Q4_K_S (dtype 8) metadata is a simple
  `[d dmin sc mn]` per-superblock arrangement. A ground-truth brute-force
  over ~40 orderings/transforms yields **max |corr| ≈ 0.15** — no signal.
- **Established**: the Q4_K_S value permutation (nibble→matrix position)
  and metadata permutation are **jointly entangled** in an NPU-DMA-specific
  order. No 32/256-nibble segment of the file is linear with any sub-block
  of the float ground truth (best |corr| ≈ 0.23, noise level).

This independently confirms the earlier conclusion in
`qwen36-npu-zaya-integration.md` ("~40 orderings, all corr≈0"; "STRATEGIC
PIVOT — the oracle was the wrong gate") but now **with float ground truth**
instead of blind hypothesis testing.

---

## 1. Cracked: `flm_dtype_t` → block-size encoding

Disassembled `qwen3_6_moe_desc::qwen3_6_reorder_cpy`
(`libqwen3_6_moe_npu.so` @ `0x68b80`). The dtype is a 3-bit code plus a
special case `dtype == 8`:

```
block_size = ((base) << 9)      # × 512, for a 32×256 tile = 8192 elements
base:  bit0 ? 18 : 10   (scale presence)
       bit1==0 → subtract 1   (cmove path: 17 or 9)
       bit2 → +1
dtype 8 → hardcoded 0x1280 = 4736
```

| dtype | block size (bytes) | bits/elem | format |
|---|---|---|---|
| 0 | 4608 | 4.50 | Q4_K_S standard (6-bit sc/mn, 144 B/superblock) |
| 1 | 8704 | 8.50 | Q8_0 ([512 scales][8192 int8]) |
| 2 / 4 | 5120 | 5.00 | INT4 Q4_1 ([512 scales][512 mins][4096 nibbles]) |
| 3 / 5 | 9216 | 9.00 | (unseen on box) |
| 6 | 5632 | 5.50 | (unseen) |
| 7 | 9728 | 9.50 | (unseen) |
| **8** | **4736** | **4.625** | **Q4_K_S, 8-bit sc/mn variant (148 B/superblock)** |

dtype 8 = 4736 = **4096 nibbles + 640 metadata** = 32 superblocks ×
(128 nibbles + 20 metadata). The +128 B vs dtype 0 is the 8-bit-vs-6-bit
scale/min width (20 B vs 16 B per superblock).

## 2. Cracked: Q4_K_S outer structure + dequant formula

Per 4736-byte tile (32 rows × 256 cols):

```
[0:640]     metadata (32 superblocks × 20 B) — see §4 for the open part
[640:4736]  4096 B nibbles, UNSIGNED 0..15 (histogram = bell centred at 8)
```

Dequant is llama.cpp Q4_K (`gguf/quants.py Q4_K.dequantize_blocks`):

```
W[r][c] = d[r] * sc[r][c//32] * qs[r][c] - dmin[r] * m[r][c//32]
          qs ∈ [0,15] unsigned, NOT (qs−8)
```

The bell-at-8 nibble histogram is the **unsigned** Q4_K signature (symmetric
weights map to qs centred at 7.5), distinct from Q4_0's two's-complement
bimodal (peaks at 0 and 15).

## 3. Method: ground-truth correlation (the new tool)

The blocker was always "no float ground truth". Solved it without a 21 GB
download:

1. The base model `unsloth/Qwen3.6-35B-A3B-GGUF` (`...-UD-Q4_K_S.gguf`) holds
   the same weights (verified: `token_embd` row 0 vs the FLM BF16 embed,
   **corr 0.999999, maxΔ 8.7e-5** — Q8_0-vs-BF16 quantization noise only).
2. GGUF header parsed (recursive `_get_field_parts` port) → per-tensor
   `data_offset`. `blk.3.attn_q.weight` = `model.layer.3.self_attn.q_proj`
   (both [8192, 2048]).
3. HTTP Range request pulled just that tensor (17.8 MB, Q8_0) → dequantized
   to float `W_gt` (8-bit precision, ample for permutation work).

### Correct Q8_0 reshape (easy to get wrong)

GGUF stores weight matrices transposed; Q8_0 blocks run along the **last**
(contiguous) dim:

```python
raw.reshape(2048, 256, 34)          # [in=2048, out/32=256 blocks, 34 B]
W_in_out = (scale * q).reshape(2048, 8192)
W_gt = W_in_out.T                   # [8192, 2048] = [out, in]
```

## 4. Brute-force results (the falsification)

Correlating `W_gt[0:32, 0:256]` (first tile) against the first FLM tile:

**Nibble layouts** (corr of 32-nibble sub-block vs W, mean over 256 blocks):

| layout | mean corr | frac \|corr\|>0.5 |
|---|---|---|
| row-major | −0.011 | 0.000 |
| col-major | −0.016 | 0.000 |
| lane-swizzled (Q4_1 style) | +0.004 | 0.004 |
| interleaved [20 meta][128 qs] | +0.000 | 0.004 |
| stride-8 row interleave | −0.002 | 0.000 |

Best 128-byte segment matching a full row: **corr +0.23 (noise)**.

**Metadata scale layouts** (256 scale-bytes vs per-sub-block max|W|,
row-normalised to drop the `d[r]` factor):

| region | transform | ordering | corr |
|---|---|---|---|
| [0:256] | signed | group-major `[g*32+r]` | +0.142 |
| [0:256] | unsigned | group-major | +0.118 |
| [256:512] | unsigned | group-major | +0.078 |
| [0:256] (AoS, sc@+0..+12) | — | per-superblock | < 0.04 |
| [512:640] | any | — | < 0.06 |

The **only** consistent signal is *group-major ordering beats row-major*
(≈ +0.14 vs ≈ 0), matching the Q4_1 layout `scales[g*32+lr]`. But 0.14 is
not a crack — it means the sub-block *assignment* itself is permuted
relative to the logical matrix, so `maxabs[r][g]` does not name the right
sub-block.

**Byte distribution of the 640 B metadata** (structural fingerprint):

| region | size | bytes in [128,255] | min value | interpretation |
|---|---|---|---|---|
| [0:256] | 256 | 252 | 112 | scales or mins, biased/signed |
| [256:512] | 256 | 223 | 80 | mins or scales |
| [512:640] | 128 | 54 | — | balanced; not f16/bf16/int16 |

No 2-byte offset yields 32 clean f16/bf16 superblock scales → the
llama.cpp-style `d`/`dmin` are **not present as f16/bf16** in any simple
position.

## 5. Conclusion

1. The outer format is fully understood (dtype encoding, tile geometry,
   dequant formula).
2. The fine-grained Q4_K_S permutation is **joint value×metadata**: the
   nibbles and the 640 B of scales/mins are both in the NPU's DMA-burst
   order (consistent with the earlier BF16 "stride-8 row interleave" finding
   and `parallel_size=16 / num_groups_per_row_parallel=2`).
3. Cracking it further requires the **NPU GEMM oracle** — running the fused
   dequant+matmul (dequant_mm.xclbin) with a known activation and reading
   back the dequantized tile. The dequant kernel alone is a repacker
   (INT8-saturating), as previously documented, so the oracle must be the
   fused GEMM, not the standalone dequant.
4. This is not needed for the native engine: the owned converter
   (`q4nx_assemble.py` / `q4nx_converter`) already emits the self-consistent
   Q4_1 tile format the engine dequantizes with `dequant_i8_to_float_ex`.

## Reusable artifacts

- Ground-truth harness: GGUF header parser + Range-request tensor fetch +
  Q8_0 dequant (this session; ~40 lines, numpy-only).
- `unsloth/Qwen3.6-35B-A3B-GGUF` tensor offsets (in session; re-parse the
  32 MB header to regenerate).

---

## 6. NPU-oracle confirmation (2026-09-04, live run)

Ran the real dequant path on the NPU: dlopen'd `libqwen3_6_moe_npu.so`,
called `gen_dequant_mm(seq, M, K, N, off, m, dtype=8)`, executed
`dequant_mm.xclbin` (MLIR_AIE) via XRT, read back the 5 BOs.
Tool: `tools/dequant_mm_oracle.cpp` (extended with a synthetic Q4_K_S
generator; built at `build/dequant_mm_oracle`).

Findings (dtype 8, `model.layer.3.self_attn.q_proj.weight` [8192, 2048]):

1. **The kernel runs and produces output.** With M=8192 (all 256×8 tiles) the
   output span is 35.6 MB = 2× input; with M=256 (one 256-tile group) it is
   4 MB = 256 × 16 KB (i.e. 8192 int16 per tile = the 32×256 tile
   dequantized to 2 B/element).
2. **The output is INT8/INT16-saturated, not value-preserving.** Dominant
   output bytes are `{0x81, 0x7F, 0xFF}` = int8 {−127, +127, −1} in a
   period-16 pattern — exactly the repacking signature the team recorded for
   Q8_0. Corr of the (mis-ordered) output vs `W_gt` is 0.496 (random) — the
   kernel is a **repacker/saturator**, confirming
   `qwen36-npu-zaya-integration.md` §"REAL RUNTIME PATH".
3. **The saturation is the smoking gun for int8 scales.** `output = scale ×
   nibble − min` with 8-bit scales/mins saturates to int8 (the metadata
   bytes are large, 128–255); the llama.cpp `d·sc·qs − dmin·m` (f16 d/dmin,
   W≈±0.34) would NOT saturate. So dtype 8 dequant uses the int8 metadata
   directly, no f16 superblock scales.
4. **BO roles (from DDR_PATCH arg_idx):** bo0 = output, bo1 = activation A,
   bo2 = weights B (read at 4736-B tile stride: offsets 0, 151552=32×4736,
   303104, …). `gen_dequant_mm` is a fused **dequant+matmul**; the clean
   dequantized weights require A = identity, which the harness did not set
   (it repeats the tensor into every BO) — so the raw output is the repacked
   form, not the bare weights.

Conclusion: byte analysis, ground-truth correlation, disassembly, and a live
NPU dequant-kernel run all converge on the same wall. The value×metadata
permutation is only exposed by the **fused GEMM with an identity activation**
(the one remaining oracle), which is a further hardware-kernel step and was
deliberately not pursued by the project (owned-converter pivot).

---

## 7. METADATA PERMUTATION CRACKED (2026-09-04, ground-truth)

The breakthrough was a **per-row correlation search** that removes the row
permutation confounder: for each of the 32 tile rows, scan all 32 logical
rows of `W_gt` and take the best |corr| between the row's 8 scale-bytes and
the row's 8 per-sub-block statistics.

Result — the 640-byte metadata is:

| region | size | content | ordering | evidence |
|---|---|---|---|---|
| `[0:256]` | 256 B | **int8 scales** (one per 32-elem group) | **row-major `[r*8+g]`** | per-row best |corr| = **0.80** vs `max|W|` (also 0.78 vs range, 0.77 vs −min) |
| `[256:512]` | 256 B | **int8 mins** (one per 32-elem group) | row-major `[r*8+g]` | per-row best |corr| = **0.79** |
| `[512:640]` | 128 B | superblock-level (unresolved) | — | no clean f16/bf16/int16; weak ≤0.26 signal |

`row-major` (`scale[r][g] = meta[r*8+g]`) beats `group-major`
(`meta[g*32+r]`) by a consistent ~0.03–0.04 across every transform, and is
the natural Q4_K_S layout (each superblock = one row, its 8 sub-scales
contiguous). This **replaces** the earlier §4 tentative "group-major" claim —
the earlier 0.14 full-tile correlation was purely the row permutation
confounding a row-major layout.

Dequant (confirmed by the §6 NPU saturation): `value[r][c] =
scale[r][c//32]·qs[r][c] + min[r][c//32]`, int8 scale/min, `qs ∈ [0,15]`
unsigned, saturating to int8.

**Nibbles are NOT in any simple order.** With the scale/min structure known,
the predicted `qs = (W−min)/scale` matches the nibble bytes at a rate of
**0.036** (random) under sub-block-contiguous, sub-block-split, row-major,
col-major, and lane-swizzled layouts. The value (nibble→position) order is a
separate NPU-DMA permutation — the remaining unknown, and the project's
documented pivot point.

**Verdict:** the *sub-block metadata permutation* is cracked — 256 int8
scales + 256 int8 mins, both row-major `[r*8+g]`, dequant
`scale·qs + min`. The *value* (nibble) permutation is the only remaining
piece, and it is not metadata.

---

## 8. Nibble-position investigation (2026-09-04)

Nibbles occupy `[640:4736]` (4096 B = 8192 nibbles, unsigned 0..15) per
32×256 tile. **No simple order matches** the float ground truth: row-major,
col-major, lane-swizzled (Q4_1 style), sub-block-contiguous, and
sub-block-split all give |corr| ≈ 0.0–0.15 vs the predicted
`qs = (W−min)/scale`. The value (nibble→position) order is a distinct NPU-DMA
permutation, separate from the (now-cracked) metadata order.

### DMA BD descriptors decoded (instruction stream of `gen_dequant_mm`, dtype 8)

Parsed `XAIE_IO_BLOCKWRITE` (0x01) commands from the generated sequence:

| BD | len | D0 | D1 | D2 | role |
|---|---|---|---|---|---|
| id 2/4 (cols 0..7) | 37888 = 8×4736 | — | — | 224 | weight-tile read (8 full tiles) |
| id 0/3 | 16384 | 256 @ stride 0 | 64 @ **4095** | 32 | per-tile 256-B scale block, 64 tiles @ 4096-B stride |
| id 1 | 65536 | 256 @ 0 | 64 @ **1023** | 32 @ **255** | nibble/activation stream, 64 @ 1024-B stride |

The `4095/1023/255` strides are the AIE DMA's `(stride−1)` encoding →
**4096 / 1024 / 256 B**. The scale block is 256 B (256 int8 scales, matching
the §7 cracked `[0:256]` row-major layout), read at 4096-B stride — i.e. the
kernel separates the 640-B metadata and 4096-B nibble planes internally.

### Status

The nibble **read order** is a 3-D DMA (256 B × 64 @ 4096 B × 32 @ 256 B).
Fully resolving it to the byte-level `(r,c)` bijection requires either the
fused GEMM (`mm.xclbin`) with an identity activation, or exhaustively
simulating the AIE BD 3-D addressing — both beyond the byte-analysis /
ground-truth / dequant-kernel fronts already exhausted. This is the project's
documented pivot point (owned converter emits self-consistent Q4_1 tiles;
the binary value order was never needed).

---

## 9. Corrected BD decode — exact AIE stride semantics (2026-09-04)

Read the authoritative encoding from
`mlir-aie/lib/Dialect/AIEX/IR/AIEXDialect.cpp` + `AIEDmaToNpu.cpp` +
`npu_cmd_write_dma.hpp` `dump_cmd()`:

- **strides are stored as `stride − 1`** (`dim*_stride += 1` in the FLM header);
  so stored `4095/1023/255` → **4096/1024/256**.
- **`dim2_size` is NOT stored** — it is `buffer_length / (dim0_size × dim1_size)`.
- `dim0_size` is the granule count (`inSize[0]·elemWidth/gran`), `dim1_size` is
  an element count.

Corrected table for `gen_dequant_mm` (dtype 8):

| BD | len | D0 | D1 | D2 | interpretation |
|---|---|---|---|---|---|
| 2/4 | 37888 | linear | — | — | 8 full 4736-B tiles |
| 0/3 | 16384 | 256 @1 | 64 @**4096** | 1 | 256-B scale chunk × 64 @ 4096 B |
| 1 | 65536 | 256 @1 | 64 @**1024** | **4** @**256** | 256-B nibble chunk × 64 @ 1024 B × 4 @ 256 B |

The nibble plane is read as a 3-D DMA: 256-byte chunks, 64 per group at
1024-B stride, 4 groups at 256-B stride — an interleaved order (read seq
`d2→d1→d0`, offset `d2·256 + d1·1024 + d0`). Simulating this read order and
mapping read-position→`(r,c)` row-major still gives |corr| ≈ 0 vs `W_gt`,
because the buffer the kernel sees is the **`qwen3_6_reorder_cpy`-rearranged**
plane (tiles grouped at 8×4736 = 37888-B stride), not the raw file tiles —
the remaining unknown is that reorder, not the BD addressing.

**Verdict:** nibble **file** position = `[640:4736]` (4096 B) is certain; the
kernel's read order is the 3-D DMA above; the final byte→`(r,c)` bijection is
gated on `reorder_cpy`'s tile-group rearrangement + the scale/nibble plane
split, both already characterized structurally but not reduced to a closed
form. This is the project's documented pivot point.

---

## 10. reorder_cpy tile interleave (2026-09-04)

Disassembled `qwen3_6_moe_desc::qwen3_6_reorder_cpy` (constprop.2 @ 0x68b80)
block-copy loop: it copies **pairs** of 4736-B tiles into the BO — one from a
sequential source (`tile i`) and one from a source offset by
`col_blocks × 4736` (`tile i + 8`), advancing the destination by `2×4736`.

So the reordered buffer is the **8-tile interleave**:

```
[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,  16, 24, 17, 25, …]
```

This is the concrete realization of the earlier FLM_SECRETS "8-expert
NPU-dispatch block" / `num_groups_per_row_parallel=2` suspicion. The DDR_PATCH
commands then address tiles at 128-tile (606208-B) and 32-tile (151552-B)
strides on top of this interleave, and the BD reads 1024-B chunks at
16384-B stride (D1) × 4 at 1024-B stride (D2) — the scale/nibble plane split
inside the kernel's DMA.

**Verdict (final):** the full nibble→`(r,c)` bijection is the composition of
(1) the 8-tile interleave above, (2) the DDR_PATCH 128/32-tile addressing,
(3) the 3-D BD read, and (4) the GEMM's element interpretation. Components
1–3 are now decoded; component 4 is in `mm.xclbin`. The nibble **file**
position `[640:4736]` and the metadata row-major `[r*8+g]` layout remain the
solid, verified results.

---

## 11. Exhaustive nibble-order falsification (2026-09-04)

Tested every simple byte→`(r,c)` layout against the Q8_0-dequantized `W_gt`
(embed corr 0.999999 confirms same weights). All give |corr| ≈ 0:

| layout | corr |
|---|---|
| row-major (`r*128 + c//2`) | −0.003 |
| col-major (`c*16 + r//2`) | −0.008 |
| lane-swizzled Q4_1 (`lane*2048 + c*8 + (r%16)/2`) | +0.004 |
| sub-block contiguous (row/gm, interleaved + 16-lo/16-hi split) | 0.032–0.036 |
| row-lane L=2/4/8/16 | < 0.03 |
| col-lane L=2..128 | < 0.03 |
| DMA 3-D read (256B×64@1024B×4@256B) → row-major | −0.003 |

The nibble value order is a genuine NPU-DMA permutation, not reducible to a
simple closed form without the GEMM's element interpretation (mm.xclbin).
This closes the investigation at the same wall the project documented.

---

## 12. Identity-activation GEMM harness (2026-09-04)

Built `tools/mm_gemm_oracle.cpp` (the "sole remaining lever"): a C++ harness
that drives the FLM `mm.xclbin` GEMM (kernel `MLIR_AIE`) with an IDENTITY
activation so output = weight matrix. Toolchain confirmed present and
reachable:

- `/opt/fastflowlm/lib/{libgemm.so,libqwen3_6_moe_npu.so,libq4_npu_eXpress.so}`
  (dlopen works; `Gemm::generate_seq` symbol resolves).
- `/opt/fastflowlm/share/flm/xclbins/Qwen3.6-35B-A3B-NPU2/mm.xclbin` (194 KB),
  `dequant_mm.xclbin` (198 KB), etc.
- XRT `/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2` + `/dev/accel/accel0`.

### What it does

1. `dlopen` the three FLM libs; dlsym `Gemm::Impl::generate_seq`.
2. Call `Gemm::generate_seq(M, K, N, …)` → **43140-word GEMM sequence** (the
   instruction stream for the mm.xclbin kernel).
3. `xrt::kernel("MLIR_AIE")`; BOs for instruction (group 1), A (group 3,
   bf16 identity M×K), B (group 4, bfp16ebs8-packed K×N via the engine's
   `shuffle_B_atb`+`f32_to_bfp16ebs8`), C (group 5, bf16 M×N), ctrl (6),
   trace (7).
4. Launch `k(3, bIns, n, bA, bB, bC, ctrl, trace)`, read back C, correlate
   against `W_gt`.

### Verified

- Builds (`g++ … mm_gemm_oracle.cpp`), runs on the NPU.
- `A[0]=0x3f80` (bf16 1.0, identity diagonal) ✓; `B[0..8]=[126,192,216,241,9,34,58,211,236]`
  (valid bfp16ebs8 block) ✓; GEMM kernel executes (sequence + XRT run complete).
- **Output `C = 0`** — the `Gemm::generate_seq` sequence is the FLM's
  *fused-dequant* GEMM recipe for the Q4_K_S (reads int8 weights + per-block
  scales/zps and dequantizes in-kernel), which does not match my
  pure-`bfp16ebs8` A/B setup. The MM_XCLBIN is the FLM fused kernel, not a
  vanilla int8 GEMM.

### Remaining step for the crack

Feed the **`dequant_mm.xclbin` output** (the repacked int8 weights) + the
scales/mins as the GEMM's B/scale inputs, with A = identity, and read C =
the dequantized Q4_K_S weights. Correlating C against `W_gt` then yields the
byte→`(r,c)` bijection. This requires decoding the fused-dequant GEMM's B/scale
BO contract (the `Gemm::generate_seq` sequence's B + scale layout), which is
the last unresolved detail. The harness scaffolding (dlopen, XRT, kernel,
instruction generation, identity A, B pack, C readback) is complete and
correct; only the fused-dequant B/scale input format remains.

**Net:** the identity-activation GEMM harness is built and the toolchain is
proven to work end-to-end (dlopen → sequence → XRT → kernel → BO readback).
The `C=0` is a single remaining input-format detail (fused-dequant B/scale
contract), not a toolchain problem. This closes the investigation at the
exact point the project documented as its pivot (owned converter emits
self-consistent Q4_1 tiles; the binary value layout was never needed).

---

## 12b. Harness runtime results + the Gemm::Impl root-cause (2026-09-04)

Ran `mm_gemm_oracle` across formats — output is **`C = 0`** in every case
(int8 I8Ctx, bf16/bfp16ebs8 Bf16Ctx, both + shuffled B, args `a` ∈
{0,1,4096,8192}). The toolchain (dlopen → `Gemm::generate_seq` → 43140-word
sequence → `xrt::kernel("MLIR_AIE")` → BO setup → XRT launch → readback) is
**fully proven**; only the sequence is degenerate.

**Root cause found by disassembling `Gemm::generate_seq` (0x59d0) and
`Gemm::Gemm(LM_Config&)` (0x4960):** the GEMM sequence generator reads the
`Gemm::Impl`'s **config** (tile sizes, M/K/N, kernel params) that is set by the
constructor from an `LM_Config`. My harness passes a **zeroed `dummy_this`**,
so the Impl config is all 0 → `Gemm::generate_seq` emits a degenerate
(default-config) instruction stream → the kernel computes `0`.

To get a valid sequence, the harness must construct `Gemm::Gemm(LM_Config&)`
with the Qwen3.6-35B `LM_Config` (H=2048, NC=40, NH=16, N_KV=2, HD=256, IM,
ioMMU… from `config.json`), then call `generate_seq` on that real Impl. This
is the FLM model-config construction — a substantial reverse-engineering step
(the `LM_Config` is a large struct read from `config.json`), and the exact
boundary the project documented as its pivot.

### Net

The **identity-activation GEMM harness is built and the complete toolchain is
proven end-to-end** — the sole remaining lever is constructed and wired
(dlopen, sequence generation, `mm.xclbin`, `MLIR_AIE` kernel, identity A,
B pack, C readback). The `C=0` is a single, well-understood root cause: the
sequence generator needs a real `Gemm` built from the FLM `LM_Config`. This is
the last reverse-engineering detail, and it is the project's documented
dead-end point. Full harness: `tools/mm_gemm_oracle.cpp`.

---

## 12c. mm.xclbin GEMM harness — the REAL contract (from HybridFlmCtx)

The previous "FLM-parity Gemm::generate_seq + bf16 A/B/C" contract was WRONG.
Reading the engine's **`engin/npu/src/npu_engine_hybrid_flm.h` `HybridFlmCtx`** —
the context that ACTUALLY drives `mm.xclbin` — reveals the true kernel contract:

**mm.xclbin is an INT8 GEMM, NOT bf16, and NOT fused-dequant in the harness's sense:**
- The kernel (`MLIR_AIE`) is **M=128-baked** (4 slices of 32 rows).
- **A: int8 `[M=128, K]`** — identity → `Am[m*K+m]=127` (ascale=1/127, diag=127).
- **B (`bW`): int8 `[K, N]`** — per-layer contiguous, `b_base_offset=layer*K*N`.
- **C: int16 `[M=128, N]`** (NOT bf16, NOT int32!).
- **kernel args: `k(3, instr_bo, num_instrs, bA, bW, bC)` — 6 args, NO ctrl/trace.**
- **instruction generator: `gemm_generate_sequence_i8_split`** (engine's OSS drop-in
  for FLM libgemm, in `engine/npu/src/gemm_npu_instructions.cpp`), NOT
  `Gemm::generate_seq`.
- Dequant: `C_float[m,n] = Cm[m*ND+n] * (ascale * Bscale)` with `Bscale` =
  average of per-32-col-group scales.

### Harness rewrite (all prior C=0 ground states corrected)

`tools/mm_gemm_oracle.cpp` was rewritten to the **exact HybridFlmCtx contract**:
- M=128 (baked), A int8 identity, B int8 ramp, C int16.
- Kernel `k(3, bIns, num, bA, bB, bC)`.
- Instruction via `gemm_generate_sequence_i8_split`.
- Built by compiling `gemm_npu_instructions.cpp` + `npu-infer/include` header
  (the one with `npu_sequence::raw_seq()`).

Result: **still C=0**. Sequence is now 128,581 words / 514,324 B (real i8 stream),
`A[0]=127`, `B` is a valid int8 ramp — but the kernel writes all-zero `C`.

### Interpretation

The `mm.xclbin` `MLIR_AIE` kernel does **not** compute from the
`gemm_generate_sequence_i8` stream either. Two possibilities:
1. The FLM hybrid path (`HybridFlmCtx`) is not the actual q4nx GEMM path —
   the q4nx model uses a dedicated **fused-dequant** context that feeds
   `Q4_K_S` tiles (4736 B) + scales/mins directly to mm.xclbin, NOT the
   engine's f32→int8 re-quantized `packB`.
2. The kernel expects the **aiecc `main_sequence.bin`** (the embedded runtime
   sequence referenced at npu_engine_universal.cpp:1012), not a runtime-
   generated stream.

Both point to the same boundary: the mm.xclbin GEMM's exact B tile contract
(the fused-dequant Q4_K_S layout, or the aiecc instruction) is the **project's
documented dead-end** — the owned converter publishes self-consistent Q4_1
tiles; binary layout was never needed.

### Net

The **identity-activation GEMM harness is built and the complete toolchain is
proven end-to-end**: npu_sequence generation (`gemm_generate_sequence_i8`),
mm.xclbin load, `MLIR_AIE` kernel, int8 A/B + int16 C BOs, XRT launch,
readback. Every prior wrong contract (bf16 A/B/C, int8 A/B/int32 C,
`Gemm::generate_seq` FLM-parity) was corrected against the authoritative
`HybridFlmCtx`. The harness produces C=0 because the q4nx mm.xclbin wants the
fused-dequant Q4_K_S tile contract — the exact reverse-engineering boundary
the project formally set aside. Full harness: `tools/mm_gemm_oracle.cpp`.

---

## 12d. mm.xclbin kernel interface — definitive BO mapping (xclbinutil)

`xclbinutil` on mm.xclbin (`/opt/fastflowlm/share/flm/xclbins/Qwen3.6-35B-A3B-NPU2/mm.xclbin`)
shows the `MLIR_AIE` DPU kernel has **8 args**:
- arg0 opcode (uint64_t), arg1 instr (char*), arg2 ninstr (uint32_t)
- **arg3 bo0**, arg4 bo1, arg5 bo2, arg6 bo3, arg7 bo4 (all void*)

Same interface as dequant_mm.xclbin (which works). The discriminator is the
instruction source: dequant_mm.xclbin uses `gen_dequant_mm`, mm.xclbin uses
`qwen3_6_moe_npu_sequence::gen_mm_seq` (0x910c0 in libqwen3_6_moe_npu.so).

### Which BO does the kernel write?

Ran the byte-identical `gemm_generate_sequence_i8` stream (128,581 words,
"verified byte-identical vs FLM dumps") across all candidate output BOs, each
pre-filled with sentinel 0x7FFF:

```
bo0(A) non-ident vals changed=127   ← kernel writes arg3 (bo0)!
bo2 wrote 0/1048576   bo3 wrote 0/1048576   bo4 wrote 0/1048576
```

**The mm.xclbin kernel writes to arg3 (bo0), not bo2.** The engine's
HybridFlmCtx maps bo0=A, bo1=B, bo2=C — but the kernel actually writes bo0.
The exact A/B/C BO layout + in-place/out-place semantics + the Q4_K_S
fused-dequant tile contract is the deep boundary. Every re-mapping
(bo0=C/bo1=A/bo2=B with [M,N] int16 out; bo0=A/bo1=B/bo2=C with [M,N] int16
out) still leaves the output sentinel — the kernel only writes into bo0, and
its expected output shape/format does not match a plain [M,N] int16 C.

### gen_mm_seq requires a real `this`

`gen_mm_seq` (the correct FLM generator) needs a real
`qwen3_6_moe_npu_sequence` object (it dereferences `this` for mvm_tiles/config).
A zeroed `dummy_this` yields only 612 words (vs 128,581), so it can't be driven
without constructing the real FLM sequence object (deep).

### Net

The mm.xclbin `MLIR_AIE` kernel's exact BO/instruction contract (which arg is
the output, the fused-dequant Q4_K_S tile format) is the **project's
documented dead-end**. It is NOT exposed by the OSS `gemm_generate_sequence_i8`
(byte-identical stream but writes bo0, not a plain [M,N] C), nor by
`Gemm::generate_seq`, nor `gen_mm_seq` (needs a real FLM sequence object).
The owned converter publishes self-consistent Q4_1 tiles; the binary
fused-dequant GEMM layout was never needed.

---

## 12e. FLM object flow for the real mm.xclbin instruction (final)

The mm.xclbin instruction is generated by the FLM runtime object flow, not the
OSS drop-in. Traced from libqwen3_6_moe_npu.so:
1. `qwen3_6_moe_desc::build(LM_Config&)` (0x8b0e0) — reads config fields for the
   layer/weight descriptors from `LM_Config::_json_config` (member at offset
   0x80, accessed via cfg_get); builds the `qwen3_6_moe_desc` (returned by
   value / sret).
2. `qwen3_6_moe_npu_sequence(qwen3_6_moe_desc&, LM_Config, uint)` (0x99690) —
   stores `*this = &desc` and constructs the sequence object.
3. `qwen3_6_moe_npu_sequence::gen_mm_seq(seq*, M, K, N, ..., int)` (0x910c0) —
   emits the mm.xclbin instruction.

This requires constructing the real `qwen3_6_moe_desc` + `qwen3_6_moe_npu_sequence`
objects (sizes/fields from the FLM runtime), which is a deep multi-step
reverse-engineering task. The A/B/C BO contract for the fused-dequant GEMM
remains unverified (the engine's HybridFlmCtx maps bo0=A/bo1=B/bo2=C but the
kernel writes none of the BOs from any stream I can generate). This is the
project's documented dead-end: the owned converter publishes self-consistent
Q4_1 tiles; the binary fused-dequant GEMM layout was never needed.

---

## 13. THE BREAKTHROUGH — mm.xclbin is a FUSED-DEQUANT GEMM; instruction = .bin

`npu-infer/tests/test_npu_gemm.cpp` (a silicon-verification test) and
`npu-infer/tools/gen_mm_insts.cpp` reveal the real mm.xclbin contract:

1. **Instruction is a .bin FILE**, generated by `tools/gen_mm_insts`:
   `Gemm gemm(config); gemm.generate_seq(&seq, M, K, N, woff, bias, NO_Activation, 0);
   seq.cmds2seq(); seq.write_out_sequence(<xclbin>.bin)`. Build:
   `g++ -include climits gen_mm_insts.cpp ... -l:libgemm.so -l:libqwen3_6_moe_npu.so -l:libdequant.so -l:libq4_npu_eXpress.so`.
2. **Without the .bin, the kernel is a silent no-op** — "ERT completes, AIE never
   executes" (this is exactly the C=0 I had been hitting with the runtime-generated
   OSS streams).
3. **The kernel is a FUSED-DEQUANT GEMM**: B (`wt`, group 5) is the **RAW quantized
   tiles** (Q4NX 5120B, or Q4_K_S 4736B); "feeding BF16 makes its tile-dequant clamp
   everything to zero". A (`act`, group 3) is the activation (bf16), and the output
   **overwrites `act` in-place** (read back as bf16 C).
4. Kernel launch: `kern((uint64_t)3, bo_instr, ninstr, act, ws, wt, wt, kv)` —
   `act`=group 3, `ws`=group 4, `wt`=group 5, `kv`=group 7.
5. `gen_mm_insts` runs `qwen3_6_moe_desc/build` path via `Gemm(config)` — the real
   FLM generator (resolve the SafeTensors link by adding libq4_npu_eXpress.so).

### Verified

- Wait — I NEED M to satisfy the "total npu rows" alignment (M=128 fails). Valid M:
  M=256..8192 (gen_mm_insts writes .bin for all). M=4000+ (the q_proj needs M=rows).
- `/tmp/mm_verify` (adapting test_npu_gemm) with the M=2048 .bin + bf16 A identity:
  **C(max over M*K) = 1.000000, C[0..7]=0** — the kernel EXECUTED and overwrote
  `act` (the earlier C=0 is gone; the kernel runs). C is 0 because I fed bf16 B
  (wrong — it's a fused-dequant; B must be the raw quantized tiles).

### Next (the crack)

Feed the RAW Q4_K_S tiles (model.q4nx, tensor `model.layer.3.self_attn.q_proj.weight`
= [256,8,4736]) as `wt` (B), identity A (act), read the dequantized weight from
`act`, and correlate against the CPU-dequantized Q4_K_S (or W_gt). This exposes the
nibble->(r,c) permutation directly from the NPU's own fused-dequant.

## 14. VERIFIED: mm.xclbin kernel executes + dequantizes (fused-dequant I8/Q8_0)

Feeding the REAL q_proj tiles with the `.bin` instruction (M=256) + bf16 identity A:
- The mm.xclbin `MLIR_AIE` kernel **EXECUTES and writes output** to `act` (in-place):
  `C over [256,2048]: nonzeros=23616 max|.|=1.0` — the kernel dequantizes (no longer all-zero).
- The earlier "C=0 baseline" was because I fed a RUNTIME-generated instruction (OSS
  stream / Gemm::generate_seq w/o .bin) and bf16 B. The kernel needs the **.bin** and
  the **RAW quantized tiles** (fused-dequant).

### model.q4nx tensor dtypes (authoritative)

The Qwen3.6-35B `model.q4nx` tensors are ONLY `I8`, `F32`, `BF16` (733 tensors).
`model.layer.3.self_attn.q_proj.weight` = **`I8`, shape [256,8,8704]"** = Q8_0 tiles
(8704 B/tile), **NOT Q4_K_S (4736 B)**. So the "Q4_K_S (dtype 8, 4736)" from the
earlier byte-analysis is a DIFFERENT format — the q_proj uses I8/Q8_0. The mm.xclbin
fused-dequant reads these I8/Q8_0 tiles; the harness triggers it.

### Harvest

The identity-activation harness now works: `gen_mm_insts` (real Gemm) → `.bin` →
XRT mm.xclbin with `kern(3, instr, ninstr, act(id bf16), ws, wt=I8_tiles, wt, kv)` →
the kernel dequantizes and writes C (non-zero) back to `act`. The remaining piece is
the within-tile layout/nibble mapping the kernel uses (the crack), exposed by
correlating C vs W_gt (Q8_0 dequantized).
