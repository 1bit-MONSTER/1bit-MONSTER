# Fused backend NPU-FFN performance analysis (2026-08-29)

**Box:** Strix Halo (gfx1151, AI MAX+ 395) · **Toolchain:** TheRock
**Model:** Qwen3-0.6B-1BP (H=1024, NH=16, NKV=8, HD=128, IM=3072, NC=28)

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
