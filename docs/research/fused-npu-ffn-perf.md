# Fused backend NPU-FFN performance analysis (2026-08-29)

**Box:** Strix Halo (gfx1151, AI MAX+ 395) · **Toolchain:** TheRock
**Model:** Qwen3-0.6B-1BP (H=1024, NH=16, NKV=8, HD=128, IM=3072, NC=28)

## 2026-08-29 (round 12) — async VK dispatch (objective item 1)

The VK path's 56 per-layer submit+waitIdle (attention + FFN each layer) are
collapsed into ONE command buffer per token: `VkAttention::record_forward`
records the embed + all 28 layers' 4-stage attention + 5-stage on-pages FFN
(253 stages, per-layer push constants) and submits once.  The GPU stays
continuously busy end-to-end.  VK+GPU-FFN: 23.4 -> 22.1 ms (45 tok/s),
parity 15 13 15 15 ... (`f8b89744`).  The remaining VK-vs-HIP gap is the VK
gemv shader efficiency (~69 GB/s vs the HIP GEMVs' ~180) — the concurrent
agent's one-block-per-row FFN gemv pattern (676 -> 250 us/layer) is the
template for the qkv shader.

## 2026-08-29 (round 13) — coalesced VK gemv shaders: VK+GPU-FFN 22.1 -> 14.6 ms

The VK qkv shader was rewritten to ONE BLOCK PER OUTPUT ROW (4096 rows: Q
2048 + K/V 1024 each), 256 threads reading k=tid contiguous — the old
one-block-per-head 2-lane/row pattern read each thread's element from a
different row (stride H apart), capping at ~69 GB/s.  The per-head QK-norm +
RoPE + f16 KV-store moved to a new attn_qkns.comp (`9f8d18d7`).  VK+GPU-FFN
22.1 -> 14.6 ms (45 -> 68 tok/s), parity 15 13 15 15 ... .

PARITY LESSON: a one-block-per-row rewrite of the post (wo) shader flipped
the VK+NPU tokens (15 13 -> 15 15) — the post's output h feeds the int8 NPU
FFN directly, and the accumulation-order change shifted h by ~1e-7 across an
int8 quantization boundary.  The q/k/v rewrites are safe (the f16 KV
quantization absorbs the differences).  The post was reverted (`55334169`).

## 2026-08-29 (round 13b) — batch gemv efficiency: 208 -> 229 tok/s

The batched gemv's per-layer cost was dominated by re-reading the W row (and
the full [B,N] x) once per batch — up to M*B*N of L2 traffic.  The W row in
shared (read once per block, reused across the B batches; bit-identical
accumulation order) gave 208 -> 223 tok/s (`99fe489a`); fusing the 3 qkv
launches and the w1/w2 launches into fused_qkv_batch_ws / fused_gu_batch_ws
(3 fewer launches/layer) gave 223 -> 229 tok/s (`3e1012c8`).  34.9 ms/batch,
parity 15 13 15 15 ... on all 8 sequences.

## 2026-08-29 (round 11) — the batch decode is now FULLY batched: 208 tok/s

The round-10 batched-decode fault was a VARIABLE SHADOWING bug: the batch
kernel copied the single's online-softmax body where the attention score is
named `s` — the batch's V_row offset `(size_t)s * seq_stride` then used the
FLOAT score (cast to size_t, ~7 for typical scores) instead of the batch
index, reading ~7 sequence-strides past the KV end at the last layer.  The
K_row sits before the score's declaration (correct), which is why every
isolated test passed.  Fixed by renaming the batch index to `sb` (`f1d24b50`).

Batch (B=8, GPU FFN) is now fully batched: attention GEMVs + elementwise
(rmsnorm/qk-norm+rope/f2h/silu/residual) + kv_store + decode + FFN + lm_head
all read the weight matrices once per batch.  **38.4 ms/batch = 208 tok/s
aggregate (18x the 11.4 ms single-stream)**, parity 15 13 15 15 ... on all 8
sequences.  Single-stream unchanged: HIP+GPU-FFN 11.4 ms (88 tok/s).

## 2026-08-29 (round 9) — the GPU FFN wins; the fused backend's real state of the art

The round-8 reframe held and was pushed: the GPU FFN (weights at DRAM
bandwidth) beats the NPU FFN by ~10x, so the GPU-FFN configs are now the
optimization target.  Commits `3fb0bc0b` (float4 GEMV), `d79c3777` (fused
QKV+GU), `eb9fc6b0` (elementwise fusions):

| Config (single-stream) | time | tok/s |
|---|---|---|
| **HIP attn + GPU FFN** | **14.4 ms** | **69** |
| VK attn + GPU FFN | 24.3 ms | 41 |
| HIP/VK attn + NPU FFN | 151-153 ms | 6.5 |

Batch (B=8, GPU FFN): ~113 ms/batch, **72 tok/s aggregate** (11x
single-stream).  Parity `15 13 15 15 ...` on all paths.

The float4 GEMV (21.2 -> 15.0 ms) was the big win — the v4 loads + double
accumulation (parity by construction).  The fused QKV/GU (15.0 -> 14.5) and
the elementwise fusions (14.5 -> 14.4) were smaller.  The per-layer ~0.36 ms
is now gemv-bound (the small shapes run at ~100 GB/s vs the 270 available —
the 1-iteration-per-thread N=1024 gemvs are reduction-dominated; warp-shuffle
and multi-row variants measured NO faster and were reverted).

## 2026-08-29 (round 8) — the FFN's real hidden cost: hw_context alternation

The m8 GU (2.06 ms) and D (0.93 ms) kernels are fast in isolation, but the
FFN's GU→D alternation runs them at 2.97 ms + 1.69 ms — **~0.8-0.9 ms per
context switch** (measured: GU-only with both contexts loaded = 2.15 ms;
alternating = 2.97; D 0.93 -> 1.69).  The FFN therefore pays ~1.6 ms/layer
of ERT hw_context alternation (two xclbins -> two contexts; the kernel name
"MLIR_AIE" is hardcoded in aiecc, so one xclbin cannot hold both kernels;
the zaya "fused" GUSILU is also two xclbins).  bI re-sync, BO flags, and BD
scheduling are all innocent (measured).  ~28 x 1.6 = ~44 ms of the 153.9 ms
single-stream token is this switch.

Eliminating it needs a SINGLE-LAUNCH fused GU→SiLU→D kernel (the zaya
cascade-grade workstream: the silu's gate×up needs per-core col pairing, so
the GU weights must interleave gate/up columns; the D's A then needs the
cross-core redistribution of the 3072 silu'd values).  Projection if it
lands: single-stream ~110 ms, batch ~175 ms/batch (46 tok/s at B=8).

## 2026-08-29 (round 7) — multi-sequence batch decode (FUSED_BATCH=N)

The batched FFN (`npu_state_ffn_batch`, committed `da33e028`) + batched
decode in the fused backend (`f1dd8b27` + `d8d4c628`):

- **Batched FFN**: `goB_rows` (per-row activation scales) + `npu_state_ffn_batch`
  process N rows in ONE GU + ONE D launch — the B weight DMA is read once.
  Measured 7.6x vs per-row calls, BIT-IDENTICAL per row (test_npu_ffn_batch).
- **forward_batch**: N sequences advance one token per call.  Per layer:
  batched attention (fused_gemv_batch_kernel reads each W row once for all
  N rows — qkv/wo), per-sequence elementwise kernels (rmsnorm/qk-norm/rope/
  kv-store/decode) on the stream, then ONE batched NPU FFN.  Batched lm_head
  reads the 622 MB vocab weight once.
- **Measured** (FUSED_BATCH=8 USE_NPU_FFN=1, 10 tokens): 222.6 ms/batch,
  **36 tok/s aggregate (5.5x single-stream)**, every sequence's stream is the
  parity stream `15 13 15 15 ...`.  Scaling: B=2 → 12, B=4 → 22, B=8 → 35
  tok/s aggregate.  The batched NPU FFN (~5 ms/layer) now dominates the
  batch; the D kernel's DMA (~1.7 GB/s vs the GU's 3.1) is the next target.

## 2026-08-29 (round 7) — M=8 vectorized FFN: the m1 scalar stream was COMPUTE-bound

The m1 "DMA wall" (~1.46 GB/s) was actually the scalar matmul throttling the
DMA: `matmul_scalar` consumes B at ~1 GB/s (1 MAC/cycle), so the fifo
backpressure stalls the shim.  The M=8 vectorized 8x8x8 mmul
(`M8_VECTORIZED`, same recipe as `build_zaya_m8.sh`) consumes B at ~8 GB/s,
so the launch drops to the DMA rate.  Committed `b1699d70`:

| xclbin family | GU r.wait (6.3 MB B) | note |
|---|---|---|
| M=128-baked (FLM parity) | 2.76 ms | vectorized but 4-slice stream |
| m1 scalar (`n1_core_i8_m1.py`) | 4.32 ms | compute-bound (1 GB/s) |
| **m8 vectorized (`v27 -M 8`)** | **2.06 ms (3.1 GB/s)** | fastest; oracle bit-identical |

Measured: FFN 8.46 -> 7.93 (m1) -> **4.42 ms/layer (m8)**; VK+NPU 272.7 ->
258.6 -> **153.1 ms**, HIP+NPU ~275 -> 256.6 -> **156.3 ms** (VK now ahead).
Token parity `15 13 15 15 ...` verified on VK + HIP clean runs (oracle cosine
0.997791, bit-identical to the 128-row baseline).

**Multi-sequence amortization (the server win)**: am=8 through the same m8
kernel runs **2045 us total for 8 rows** (256 us/row) — the B DMA is read once
for all rows.  A batched decode (8 sequences per launch) puts the per-layer
FFN at ~2.05 ms for all 8 sequences -> ~7.2 ms/sequence-token FFN, an 8x
aggregate throughput win for multi-token workloads.  Requires batched
attention (next step).

Tile-size experiments (n=256 GU, k=128 D) both produced numerically WRONG
output (the C writeback layout does not generalize beyond 64x128 tiles in
n1_core_i8_v27.py) — reverted; the committed m8 stays 64x128.

## 2026-08-29 (round 6) — true M=1 single-row FFN xclbins + DMA-wall map

### The M=1 win (committed `85021e4c`)

The shipped `final_i8_{GU,D}_qwen3_0_6b` xclbins bake a fixed M=128 AIE tile
stream — every decode launch runs a 128-row stream for 1 row of data.
`n1_core_i8_m1.py` emits a genuine single-core-row M=1 stream (linear 1-row
A/C DMA taps, same 8x8-microtile B gather).  Two correctness fixes were
required to land it:

- **The M=1 microkernel must index the microtiled L1 B layout.**  The DMA
  delivers B as [kb][nb][8][8] block-major (the only DMA-legal int8 form —
  the toolchain rejects byte-granular strides).  `matmul_scalar`'s row-major
  `b[i*colB+col]` indexing produced uncorrelated output (oracle cosine 0.04);
  the `DIM_M<16` alias now reindexes `[((kb*nb+nb)*8+r)*8+c]` — bit-identical
  values to the vectorized mmul accumulation (cosine 0.9978, same as the
  M=128 baseline, token parity `15 13 15 15 ...` on all four paths).
- **`cascade_d_first/mid/last_i8_i32` need `#if DIM_M == 8`** — the a2s@b
  cascade slice static_asserts `DIM_M == 8` and broke every non-8 build
  (M=1, M=16, M=128) since the cascade kernels landed.

Measured: FFN 8.46 → 7.93 ms/layer; VK+NPU 272.7 → 258.6 ms, HIP+NPU ~275 →
256.6 ms.  The m1 family is auto-selected by `npu_state_create` (MD=1) when
`final_i8_{GU,D}_qwen3_0_6b_m1.{xclbin,txt}` are present.

### The FFN wall is single-launch DMA-bound (~1.4-1.5 GB/s)

The M=1 stream did NOT deliver the ~50 µs/launch the older docs hoped for:
`r.wait` is still ~4.3 ms for the GU (6.3 MB B).  Per-step micro-benchmark of
`goB` (m1 GU): quantize 1 µs, bA sync 1 µs, bI sync 4 µs, launch 46 µs,
**r.wait 4320 µs**, bC sync 4 µs, dequant 2 µs.  The kernel time is the wall.

Exhaustive probes (all leave `r.wait` ~4.0-4.4 ms for the 6.3 MB GU B):

| Probe | Change | r.wait |
|---|---|---|
| baseline m1 (n=128, b=5) | — | 4.36 ms |
| n=256 tiles | half the BDs/commands (384 vs 768), same bytes | 4.32 ms |
| microtiled-packed B source | contiguous 64-byte block reads instead of strided 8-byte runs | 4.05 ms (and subtly wrong — cosine 0.9865; reverted) |
| 2x concurrent half-N kernels | 2 launches in flight | 1.11x vs serial (no bandwidth sharing) |
| CACHEABLE bB BO | vs HOST_ONLY | identical |

Conclusion: the single-launch NPU DMA path delivers ~1.4-1.5 GB/s regardless
of BD count, tile size, source layout, concurrency, or BO flags.  With int8
B weights the per-layer FFN floor is 9.4 MB (GU 6.3 + D 3.1) / 1.4 GB/s ≈
6.7 ms DMA + ~1.3 ms host glue ≈ 7.9 ms/layer — exactly where we are.  (The
28-independent-FFN pipeline's 3.76 ms/layer aggregate ~2.5 GB/s is the only
faster DMA regime, and it is unusable for single-stream autoregressive
decode.)  The only remaining byte-level lever is int4/ternary B (0.59x /
0.25x bytes → ~5.2 / ~3 ms per layer).

## Measured

| Item | Time | Note |
|---|---|---|
| `npu_state_ffn` serial, 1 layer | 8.46 ms | wall |
| 28 FFNs async-parallel (independent) | 3.76 ms/layer | **2.25x** |
| FFN while GPU does warm work | 8.5 → 9.9-10.9 ms/layer | GPU/NPU share DRAM |
| VK attention layer, isolated | 190-280 µs | GPU-only, warm |
| VK attention layer, in bench (after 8.7 ms FFN idle) | 1.2-1.7 ms | GPU cold-start |
| HIP attention + slot-copy sync, GPU-side | 25 µs | async stream |
| 4-stage VK batch, GPU-only (timestamped) | 260 µs warm / 1156 µs cold | cold = first after idle |

## Findings

1. **The NPU FFN is the wall** (~8.46 ms/layer → ~237 ms/token for 28 layers),
   identical in the HIP and VK paths. It dominates both (VK+NPU 299 ms,
   HIP+NPU 275 ms).

2. **The FFN is sync/launch-bound, not compute-bound**: 28 *independent* FFNs
   pipeline at 3.76 ms/layer (2.25x). The serial path's per-layer
   `r.wait()` + double BO sync (`bA`/`bI`/`bC`) blocks the host between
   layers. This speedup only applies to independent FFNs (multi-sequence /
   batch), NOT single-stream autoregressive decode where layers are strictly
   dependent (`h` is updated in place).

3. **GPU/NPU share DRAM on this APU**: any GPU activity concurrent with the
   NPU FFN slows it (8.5 → 9.9 ms even for a tiny 16-group warm dispatch).
   Keep-warm shaders are therefore counterproductive end-to-end.

4. **The VK-vs-HIP gap is GPU cold-start**: after each 8.7 ms FFN idle the
   GPU drops to a low power state; the first dispatch runs 4.4x slower
   (1156 µs vs 260 µs GPU). HIP avoids it because its attention kernels are
   async on a stream and its D2H slot copy keeps the pipeline flowing —
   total 25 µs/layer. VK pays ~1.2 ms/layer in the bench.

## What this means

- Per-token overlap (run FFN(l) with attention(l+1)) is impossible: strict
  data dependency, `h` in place.
- Cross-token overlap is impossible for single-stream autoregressive decode:
  token t+1 needs token t's lm_head output.
- Warm-keep during the FFN gap is counterproductive (DRAM contention).
- The remaining levers are shared (faster FFN benefits both paths equally):
  pipeline the FFN's per-layer syncs (`r.wait()` → async + fence), or use
  the 2.25x parallel-FFN path for multi-sequence workloads.

## Recommended next step (shared-path win)

Make `npu_state_ffn`'s kernel submission async (no per-goB `r.wait()`;
collect via a fence at the layer boundary). Expected: FFN 8.46 → ~4 ms/layer,
dropping BOTH VK+NPU and HIP+NPU by ~125 ms/token. That changes the absolute
numbers but not the VK-vs-HIP delta — the goal of "VK faster than HIP" needs
the attention-side cold-start solved, which the APU's shared-DRAM design
blocks for this workload.

## 2026-08-29 (round 5) — parallel GEMV shaders + quantize analysis

5. **Parallel GEMV shaders (committed `4d92eef5`)**: the qkv/post GEMVs were
   one-thread-per-row with serial inner loops (~12x slower than HIP's gemv).
   Rewrote with shared-memory reductions (2 lanes/row qkv, 8 lanes/row post):
   post 173→57 us; va_.layer in bench 1.2-1.7 ms → 0.86-0.93 ms (cold-start is
   multiplicative on shader work, so shrinking the work shrinks the cold
   penalty). VK+NPU 299 → 283-290 ms; gap to HIP narrowed 25 → ~13 ms.

6. **CPU quantize is a large FFN component**: the GU quantize alone is
   1.57 ms/layer (6.3M float→int8), plus ~0.8 ms for D — ~2.4 ms of the
   8.46 ms FFN wall is pure CPU serialization.  But for single-stream the
   serialization is unavoidable (GU needs attention output, D needs GU's silu),
   and moving quantize to the GPU would shrink the wall for BOTH paths equally
   (the AIE kernel time is ~3.76 ms/layer; CPU burn does not contend with the
   AIE — measured FFN unchanged with a CPU-burning thread).

7. **Final A/B (committed state)**: VK+NPU 287-289 ms vs HIP+NPU 275 ms
   (delta ~13 ms).  The residual gap is GPU cold-start after each 8.7 ms FFN
   gap: VK's per-layer submit+waitIdle pays ~0.86 ms/layer vs HIP's async
   stream at ~25 us/layer.  Both share the ~237 ms FFN wall.  VK's theoretical
   floor ≈ 282 ms vs HIP ≈ 258 ms — VK cannot beat HIP on this workload/hardware
   through shader or buffer changes; only a fundamentally different attention
   dispatch model (async stream, not per-layer waitIdle) could close it.

## 2026-08-29 (round 15) — VK fusion + batch root-cause (both candidates at parity limit)

**VK single-token (record_forward 9 -> 7 stages/layer, commit 32f8bd8c):**
Per-stage GPU timestamps (new `VkAttention::profile_forward`) show the old
"~43 us/dispatch floor" was a TIMESTAMP ARTIFACT — with barriers restored the
true per-stage costs are: attn_rms 2.9 us, attn_qkv ~96 (16 MB @ ~184 GB/s),
attn_qkns+decode 17 (fused), attn_post ~92 (8 MB @ ~87 GB/s — parity-locked
8-lane strided pattern), ffn_rms 2.9, ffn_gu ~141 (24 MB), ffn_silu+down+add
74 (fused).  The gemvs are DRAM-bandwidth-bound at ~170-184 GB/s — the old
"69 GB/s" was the pre-coalescing qkv (fixed in 9f8d18d7).  Fusing the two
pairs that win (qkns+decode, silu+down+add) saves ~0.2-0.3 ms; the
redundant-RMS fusions (rms+qkv, rms+gu) measured neutral-to-slower and were
dropped.  FAILED experiments (measured, reverted):
  * no inter-stage barriers: races through L1 — 2.9e-1 pages diff (AMD
    global writes are NOT coherent across dispatches; the HIP path needs no
    barriers only because it never re-reads same-stream scratch).
  * scoped VkBufferMemoryBarrier: same cost as the full barrier on
    device-local, 5 ms WORSE on the dma-buf pages (RADV).
  * coalesced float4 attn_post (HIP wo order): token stream flips
    `15 13 15 15` -> `16 17 17` — the post's accumulation order is locked by
    the int8 FFN boundaries.
VK remains 14.6-14.7 ms/token (HIP 11.5); the residual gap is the required
barriers (~2.1 ms) + the parity-locked post pattern (~1.3 ms).

**Batch decode (commit ad53584e):** the ~20 ms/batch "launch overhead" is
root-caused to the ws gemv kernels' per-(row,batch) sequential-tree
structure — NOT launch overhead.  Each (row,b) does a 256-thread tree with 10
syncs; B=8 rows x M blocks re-reads the full x from L2 (lm_head: 152K blocks
x 32 KB = 4.9 GB of L2 traffic).  The structure is parity-locked: the sr
single-reduction variant and three new kernels (x-staged-in-shared 48 KB LDS
-> 1 block/CU; x-in-registers 4x fewer blocks; coalesced float4) all measured
SLOWER or flipped tokens.  Safe wins, bit-identical end-to-end:
  * argmax_rows_kernel: lm_head's host scan of 8x152K logits -> GPU
    (first-max semantics), 4.9 MB D2H kept for API compat.
  * the ws kernels' sync before the sdata write and after the y write are
    provably redundant (per-thread sdata slots; the tree's own syncs order the
    reduction) — 10 -> 8 syncs per (row,batch).
forward_batch 25.7 -> 24.7 ms, lm_head 9.1 -> 8.1 ms; 230 -> 243-244 tok/s
aggregate.  Parity held on all six paths: HIP 11.5 ms, VK 14.6, batch 243
tok/s, VK+NPU 150 ms, batch NPU 44 tok/s, HIP+NPU unchanged.

Both user candidates are now at their parity-safe limits: the VK residual is
barrier cost + a parity-locked post; the batch residual is the parity-locked
ws reduction structure + the lm_head W stream.
