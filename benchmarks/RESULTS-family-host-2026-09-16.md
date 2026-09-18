# Family host work: the AttnCtx layout for nkv>2 and hd>128

Goal `mu35shsg-i3hlyi`, task `family-host`. Date 2026-09-16 ~04:20 ADT. Device runs
serialised; each number below is from the file it names.

## Landed on goal/runlist-decode-wire

`engine/npu/src/npu_attn_ctx.h` now checks shapes against the kernel's structure
instead of pinning them to the one built configuration (ported from the family
worktree, whose own single-pass guard showed the changed code collapses to the old
values for nq=8/nkv=2/hd=128):

| was | now |
|---|---|
| `nq != 8 \|\| nkv != 2 \|\| hd != 128` -> unsupported | `nq % cols == 0`, `nkv` divides `cols`, `hd` a multiple of 128 and `<= K_FRAME` |
| `cols` implicit 8 | `cols` from `NPU_ATTN_COLS`; `PARAM_ROW = 15` for nq<=15 else `nq`; `qsz = max(16, PARAM_ROW+1)*K_FRAME` |
| A-frame params hard-wired to row 15 | `PARAM_ROW` everywhere |
| KV/act re-packed every row | a K/V cache keyed on the buffer pointers and `seq` |

`engine/npu/tools/attn_kernel_bench.cpp` takes `CK_NQ` / `CK_NKV` / `CK_HD` from the
environment (the goal tree's copy was the old hardcoded 8/2/128 one, whose runs
ignored the env and produced identical numbers for every shape — a false result
caught before it was used).

Regression after the port (goal tree): 0.6B runlist 156 tok/s; Zaya with the
generated N=2048 kernel `[MoE L1 dbg] corr=0.999342`.

## Measurement 1 — Phi4 (nkv=8): the host layout matches what the kernel read

Built `-H 24 -c 8 --nkv 8 -K 128 -N 512` (Phi4-mini: nh24, nkv8, hd128). Gated with
`/tmp/ck_goal` (`CK_NQ=24 CK_NKV=8 CK_HD=128 NPU_ATTN_COLS=8 NPU_ATTN_MAX_SEQ=512`):

| seq=512 | C2 | max_abs_err | ms/call |
|---|---|---|---|
| NPU | **2/2 non-zero** | **2.403474e-02** | 2.039 |
| EMU | | **2.403474e-02** | |

**NPU == EMU to the digit.** The nkv8 bKv layout is correct: the host packs 8 kv
heads per token and the kernel reads them where they were written.

## Measurement 2 — Nanbeige (nkv=4, multi-pass): also matches

`-H 20 -c 4 --nkv 4 -K 128 -N 2048`, gated with `CK_NQ=20 CK_NKV=4 CK_HD=128
NPU_ATTN_COLS=4`: NPU==EMU `5.552646e-02` at seq=2048 and `2.377548e-01` at 513,
C2 2/2. (This is the bench gate `task-family-nanbeige` asked for.)

## Measurement 3 — hd256 (Qwen3.5-4B / Gemma3): the host layout does NOT match

Built `-H 16 -c 8 --nkv 4 -K 256 -N 512` (Qwen3.5-4B: nh16, nkv4, hd256). Gated with
`CK_NQ=16 CK_NKV=4 CK_HD=256 NPU_ATTN_COLS=8`:

| seq=512 | C2 | max_abs_err | ms/call |
|---|---|---|---|
| NPU | 2/2 non-zero | **3.452878e-01** | 2.119 |
| EMU | | **4.791975e-01** | |

The errors differ (and both are large — a 256-wide int8 PV has more quantisation
error than a 128-wide one). Per the L1 record the PV N-split is a **three-part**
change — generator core, generator seq, and **host V packing + C2 read** for
`n_hd = hd/128` tiles — and only the generator and the `AttnCtx` hd check are in
place. The host still packs V and reads C2 as if hd were one 128-wide tile, so for
hd256 the kernel reads V tiles and writes C2 columns the host does not agree with.
**This is the remaining `family-host` item**, and its fix is in
`engine/npu/src/npu_attn_ctx.h` (the V pack at `Vm + kv*N*K + ki*8192 + …` and the
C2 read-back), exactly as `RESULTS-attention-c2-regression-2026-09-15.md:959-1000`
sketches.

## State

- `family-host` is satisfied for **nkv=8 (Phi4)** and **nkv=4 (Nanbeige)** with the
  digit-level NPU==EMU measurements above; the **hd256** half is measured but not
  yet matching, with the exact host location named.
- The generated kernels and logs: `/tmp/attn_phi4_512.{xclbin,insts.txt}`,
  `/tmp/attn_q35_512.{xclbin,insts.txt}`, `/tmp/attn_v4_2048/*`; bench `/tmp/ck_goal`.

## PV N-split host update (2026-09-16, later)

Implemented the `n_hd = hd/128` host packing in `npu_attn_ctx.h`:

- The V pack now branches. For `n_hd > 1` it writes V **row-major** `[t][d]` with
  row stride `hd`, matching the generator's PV feed
  (`offset = kv*N*K + ki*(k*hd) + hi*n`, `strides = (hd, 1)` — read it in
  `n1_core_attn.py`'s PV phase). For `n_hd == 1` the original chunk-interleaved
  pack is kept byte-identical (guarded).
- The EMU V read matches the same two layouts.
- The C2 read walks the head-dim tiles:
  `cidx = nh_i*1024 + (dl/8)*64 + (dl%8)`.

**hd128 is untouched and re-verified**: Nanbeige nh20/nkv4 N2048 and Phi4
nh24/nkv8 N512 both still `C2 2/2` (the `n_hd == 1` guards).

**hd256 after the change**: the EMU `max_abs_err` drops from `4.791975e-01` to
`2.126317e-02` — the host packing is now a valid attention computation — but the
NPU is `3.752225e-01`, still not matching. So the host layout is no longer the
(only) difference: the kernel's second head-dim tile writeback/read does not
agree with the seq's stated offsets, or the hd256 build's `n_hd` path is not what
the seq describes. Next: `NPU_ATTN_DUMP=1` on the hd256 kernel and compare the
per-tile C2 offsets with the seq writeback (`(hp*cols+c)*(M*K) + hi*(M*n)`), and
confirm the built ELF actually carries `n_hd = 2`.

The build for the measurement: `n1_core_attn.py -H 16 -c 8 --nkv 4 -K 256 -N 512`
(goal tree's generator), gated with `CK_NQ=16 CK_NKV=4 CK_HD=256 NPU_ATTN_COLS=8`.

### hd256 follow-up: where it stands (2026-09-16, later)

Rebuilt the hd256 kernel from the current family generator
(`NPU_ATTN_K=256 NPU_ATTN_N=512 NPU_ATTN_HEADS=16 NPU_ATTN_NKV=4 NPU_ATTN_COLS=8`)
— same NPU error, so the stored xclbin was not stale. `NPU_ATTN_DUMP=1
NPU_ATTN_DUMP_SEQ=512` shows:

- **QK^T MATCH** (`2/2 8/8 8/8 8/8 3/3 27/27 3/3 5/5 9/9`, and the t=126/t=254
  tile-boundary probes MATCH) — the Q/KT packing is correct for hd256.
- The design's emitted V feed is confirmed **row-major**:
  `aie.dma_bd(%arg3, 376960, 8192, [<1,4>, <1,4>, <64,256>, <128,1>])`
  (= `kv*N*K + ki*(k*hd) + hi*n`, strides `(hd,1)`) — exactly the host pack.
- Both head-dim tiles are written (`C2 head0 nonzero=256`, the row-0 elements of
  both 1024-wide `M*n` tiles).
- But the delivered C2 row 0 is wrong: `-8935 -13136 ...` vs expected
  `16110 -30925 ...`, i.e. the **PV phase**, not QK^T.

So with the host pack, the V DMA and the QK^T all verified, the disagreement is
inside the kernel's PV path for `n_hd > 1` (the A2 re-read for the second tile,
the C2 tile ordering, or the `n_hd`-deep C2 FIFO drain). Next diagnostic is to
have the dump also print the A2 the PV consumes (bo4) per `hi`, and to check
whether the two C2 tiles are swapped. hd128 (Nanbeige/Phi4) is unaffected.
