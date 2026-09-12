# task-n1 — dense Qwen3 bf16 prefill: correctness findings

Working reference: FLM `qwen3_npu::prefill` **@256 tokens boot = 72429** (captured
on-box; the earlier "760" bar was the 1024-token prompt — apples-to-oranges, since
the bridge's `NPU_PREFILL_BF16` path caps `npt` at 256).

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
