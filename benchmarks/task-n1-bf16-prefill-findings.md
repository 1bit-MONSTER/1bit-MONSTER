# task-n1 — dense Qwen3 bf16 prefill: correctness findings

Working reference: FLM `qwen3_npu::prefill` **@256 tokens boot = 72429** (captured
on-box; the earlier "760" bar was the 1024-token prompt — apples-to-oranges, since
the bridge's `NPU_PREFILL_BF16` path caps `npt` at 256).

## Q4_1 formula VERIFIED (breakthrough)

The tile layout is confirmed and the dequant formula is:

```
weight[i] = scale[g] * v[i] + min[g]        g = i/32 (group of 32)
scale[g]  = bf16 at tile[2g]            (tile[0:512)   — POSITIVE small
min[g]    = bf16 at tile[512+2g]        (tile[512:1024) — NEGATIVE small (the bias)
v[i]      = 4-bit nibble of tile[1024 + i/2]
```

Check on the real capture (`q_proj` tile 0, `scale[0]=0.0045776`, `min[0]=-0.0302734`):

| element | bridge W | implies v | tile nibble |
|---|---|---|---|
| W[0] | 0x3ae8 (0.00177) | **7.00** | tile data[0] low nibble = **7** ✓ |
| W[1] | 0xbcd2 (-0.0256) | 12.2 | data[1] low = 13 / high = 8 |

`W[0]` reproduces **exactly** with the low-nibble-first packing, so the
packer/dequant formula is right; subsequent elements diverge => the dequant's
**output layout is tiled/permuted**, not element-linear.

FLM's captured `arg4` first 8 imply `v ≈ 8.0 8.2 8.05 7.76 8.17 8.56 7.6 8.03`
— near-constant and *not* the tile-0 nibbles `[7,2,13,8,...]`, so **that BO is
almost certainly not the QKV weight** (or is pre-permuted). The W-injection test
(→ 48035) is therefore inconclusive about our dequant.

## ⚠️ CRITICAL: the benchmark prompt is out-of-vocab for the 0.6B

`/tmp/ids1024.txt` (and the 256-token prefix) contains token **248044**, which
exceeds the 0.6B vocab (**151936**) — it appears **24/256** times, and is the
*first* token. So the "boot @256 = 72429" bar is measured on a malformed input.

Effect discovered via `NPU_DUMP_L0`: with the malformed prompt the embedding
lookup for token 248044 is out of bounds and `bh[0]` becomes zero, which makes
`bA` row 0 zero (`0x8000/0x0` pattern), so **GEMM C row 0 = all zero**, and every
downstream layer-0 output for token 0 is zero. That "first-row-zero" is an
*artifact of the out-of-vocab token*, not a kernel bug — with a valid prompt
(the 248044 replaced by 151644) `bA` row 0 = 1004/1024 nonzero and `bC` row 0 =
4096/4096 nonzero.

### Corrected bars (valid prompt, `ids256_valid.txt`)

| | boot @256 |
|---|---|
| FLM `qwen3_npu::prefill` | **62865** |
| bridge bf16 prefill (`NPU_RUNLIST=0`) | **91364** |
| (malformed prompt, for reference) FLM / bridge | 72429 / 78471 |

Note `NPU_RUNLIST` must be 0 for the bridge to take the bf16 path on a valid
prompt — otherwise the RuntimeLayer whole-layer path runs first and returns 0.

## Current state

| build | boot @256 |
|---|---|
| G=8 host tile reorder (old) | 47358 (a prompt token → GEMM producing garbage) |
| **identity tile reorder (committed c845c6d77)** | **78471** (a real token) |
| identity + chunked GU mode=1/2 | 6983 |
| identity + full GU `D_out=3072` mode=1/2 | 78471 |
| FLM reference | 72429 |

So the identity reorder is a genuine step (the FLM loader does **not** apply the
`[0,4,1,5,2,6,3,7]` G=8 host reorder the packer used). The residual 78471→72429
gap is a single-token argmax flip.

## Captured FLM dequant args (interposer on `Dequant::generate_dequant_q4_1_seq`)

```
QKV: D_in=1024 D_out=4096 woff=0        mode=0
O:   D_in=2048 D_out=1024 woff=2621440  mode=0
GU:  D_in=1024 D_out=3072 woff=3932160  mode=2 (gate) + mode=1 (up)
D:   D_in=3072 D_out=1024 woff=7864320  mode=0
```
`woff` is bytes into a **contiguous per-layer Q4NX BO** (QKV@0, O@2621440,
GU@3932160, D@7864320 → 9 375 000 B total = matches `npu_layer_bo_bytes`).

## `model.q4nx` tile structure (verified by byte inspection)

Per 5120-B tile (e.g. `layers.0.self_attn.q_proj.weight`, shape `I8 [256, 5120]`):

```
[0,   512)  256 × bf16 scales   (hi byte 0x3b-0x3c, small positives)
[512,1024)  256 × bf16 mins     (hi byte 0xbc-0xbd, small negatives)  ← Q4_1 bias
[1024,5120) 4096 B packed 4-bit data
```
=> group = 32 weights, each with (scale, min). The packer's tiles are byte-exact
with the raw file at `8 + header_len + data_offset` (reorder aside).

## The open discrepancy

The FLM's GEMM weight BO (`arg4`, 10 MB = 5120×1024 bf16) has **65 239 distinct
bf16 values** in its first 8 MB; the bridge's dequant output has only **2 275**
(a coarse, 4-bit-like set). The FLM's distribution looks like `scale·q + min`
with finely varying min/scale; the bridge's looks like `scale·q` with ~128
scales/tile.

Hypotheses (next session, in order):
1. **Mins not applied / wrong offset** — check whether the bridge dequant reads
   the `[512,1024)` min block (the tile is passed raw, but the `Dequant`
   geometry may expect a different scale/min stride).
2. **Group-size mismatch** — 256 groups/tile assumed vs FLM's actual
   (disassemble `Dequant::Impl::generate_dequant_q4_1_seq` @ `0x6950` in
   `libdequant.so` for the group constants).
3. **`arg4` may not be the QKV weight at all** (5120 cols ≠ QKV's 4096) —
   confirm which run owns the 10-MB BO via the `RUNLIST_ADD` log.

### Extra evidence (W[0..8] element comparison)

```
FLM    : 0.006317 0.007263 0.006561 0.005249 0.007111 0.008911 0.004517 0.006500
bridge : 0.001770 0.025635 0.018799 0.006348 0.002869 0.017578 0.031128 0.010193
computed (scale[0]=0.00458, min[0]=0.0303, low-nibble-first): 0.0623 0.0394 0.0898 ...
```
The FLM values are ~smooth and smaller than `min[0]`; the bridge's include
values both above and below. `Dequant` disasm constants seen: `0xf` (4-bit
mask), `0x20` (32 = group), `0x10`, `0x18`, `0x30` — consistent with a 32-weight
group, so the tile/group geometry is right and the divergence is in the
element ordering / nibble order (bridge W[3]=0.006348 ≈ FLM W[0]=0.006317).

## W-injection diagnostic (BF16MM_W_FILE, committed 24d9a5b1c)

Replaces the first `run_dequant_dev` (the QKV) with a raw bf16 tensor read from a
file. Feeding FLM's captured `arg4` (10 MB, stride 4096 or 5120):

| QKV W source | boot @256 |
|---|---|
| bridge dequant (current) | 78471 |
| FLM's captured `arg4` | 48035 |
| FLM reference (full pipeline) | 72429 |

So the QKV weight alone does not reproduce the reference — **at least one more
stage diverges** (O/GU/D dequant, attention, or a host op: norm/RoPE/residual/
SiLU). Note the bridge's `w_dev` and FLM's `arg4` may also sit in different
tile layouts, so this test bounds but does not isolate the remaining bug.

## Layer-0 dump sanity (valid prompt)

With `ids256_valid.txt` + `NPU_RUNLIST=0` + `NPU_DUMP_L0=1`, all layer-0 stage
outputs are non-zero and plausible:

| dump | type | nonzero | range |
|---|---|---|---|
| `bf16_l0_bA` (input norm) | bf16 | 4036/4096 | — |
| `bf16_l0_bC` (QKV GEMM) | bf16 | 16384/16384 | — |
| `bf16_l0_qkv` (post q/k-norm+RoPE) | f32 | 4096/4096 | -99 … 71 |
| `bf16_l0_o` (O proj) | f32 | 1024/1024 | -1.64 … 1.42 |
| `bf16_l0_dw` (D proj) | f32 | 1024/1024 | -0.63 … 0.55 |
| `bf16_l0_hidden` (final hidden) | f32 | 1024/1024 | -376 … 207 |

The post-norm/RoPE QKV range (-99…71) and the final hidden range (-376…207) are
far larger than expected for a 0.6B model (weights ~O(1)) — the RoPE table or
the q/k-norm scaling is the prime suspect for the remaining 91364-vs-62865 gap.

### Per-segment analysis (layer 0, token 0, valid prompt)

| segment | bf16 raw (bC) std | post-norm+RoPE (bqo) std |
|---|---|---|
| Q (0:2048) | 0.786 | 1.975 |
| **K (2048:3072)** | 0.824 | **5.531** |
| V (3072:4096) | 0.362 | 0.362 (unchanged ✓) |

RMSNorm should give std ≈ |kn_w| ≈ O(1); the K std of **5.53** (× 6.7 over the
raw) means `kn_w[l][d]` (the `k_norm.weight`, loaded at `npu_engine_universal`
line 844 from `kn_off[l]`) is either mislocated or mis-scaled. Per-head K stds
range 2.2–9.7, vs Q's 0.5–2.5. **This is the strongest remaining lead** for the
91364-vs-62865 divergence.

## Repro

```
# FLM reference @256
head -256 /tmp/ids1024.txt > /tmp/ids256.txt
NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins NPU_FLM_PREFILL=1 \
  ./engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 1 /tmp/ids256.txt

# bridge bf16 prefill @256
NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins NPU_PREFILL_BF16=1 \
  ./engine/npu/build/npu_engine_qwen3_0_6b ... # same args

# weight dump
NPU_DUMP_L0=1  → /tmp/bf16_l0_Wqkv.bin (bridge dequant'd W)
CAP_DIR=/tmp/cap CAP_NO_SYNC=1 CAP_DUMP_BIG=1 CAP_SKIP_BIG=1 \
  LD_PRELOAD=npu-infer/tools/capture/cap_interposer.so ... NPU_FLM_PREFILL=1
  → /tmp/cap/preinsts_001_01_i4_*_10485760.bin (FLM's arg4 W)
```
