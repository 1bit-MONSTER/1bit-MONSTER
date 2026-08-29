# Fused backend NPU-FFN performance analysis (2026-08-29)

**Box:** Strix Halo (gfx1151, AI MAX+ 395) · **Toolchain:** TheRock
**Model:** Qwen3-0.6B-1BP (H=1024, NH=16, NKV=8, HD=128, IM=3072, NC=28)

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
