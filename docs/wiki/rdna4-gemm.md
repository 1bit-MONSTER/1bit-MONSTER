# RDNA4 (gfx1201) GEMM & Memory Findings

> Measured on the RX 9070 XT (gfx1201, RDNA4) during the Mojo GPU-puzzles /
> rocBLAS race session, 2026-08-13. All numbers verified on-device; this page
> exists so the engine's GPU work (HIP GEMM kernels, WMMA prefill, ROCm
> tooling) doesn't re-derive or re-learn these the hard way.

## The corrected bandwidth picture

- **Streaming copy (1 GiB read + 1 GiB write): 537–571 GB/s** — right at the
  GDDR6 spec (256-bit @ 20 Gbps → 640 GB/s theoretical). An earlier claim of
  "2.28 TB/s" was a 4× buffer-size unit error; do not trust any bandwidth
  number without a full-coverage checksum.
- A trivial copy kernel beats torch's `copy_` (267 GB/s) by ~2× — torch's
  copy kernel is weak; raw HIP/Mojo streaming is the reference.
- Random (uncoalesced) gather: ~10 GB/s — access pattern dominates; sanity
  check for any kernel that looks "too fast".

## Why rocBLAS sgemm is fast (9.1 ms @ 4096³ fp32 = 15 TFLOPS)

Full decomposition in [`mojo-bench/ROCBLAS_ANALYSIS.md`](../../../mojo-bench/ROCBLAS_ANALYSIS.md).
The short version:

- **Tile geometry cuts DRAM traffic, not bandwidth.** rocBLAS's gfx1201 F32
  kernels are `Cijk_Ailk_Bljk_MT128x64x8` (128×64×8 macro tile, GSU1 = no
  split-K, single-buffered). Asymmetric 128×64 tiles re-read A 64× but B only
  32× → **6.06 GiB** traffic vs 8 GiB for square 64×64 tiling.
- **L3 residency**: the 64 MiB Infinity Cache holds a B column strip (1 MiB)
  while the 32 M-blocks that share it sweep it → B's DRAM traffic collapses
  to ~64 MiB → ~4.1 GiB total → 487–715 GB/s effective, consistent with the
  ~550 GB/s DRAM wall. rocBLAS does not run memory faster; it moves less data.
- **K-slice 8 with tiny LDS** gives rocBLAS full occupancy; their kernel
  tolerates 512 barrier-pairs because warp parallelism hides them.

## Lessons for the engine's HIP GEMM kernels

Measured on the 9070 XT with hand-written tiled GEMMs (Mojo, but the lessons
are architecture-level):

- **K-slice sweep is monotonic: bigger wins.** 128×64 tiles at K=8/16/32/64
  → 27.3/22.3/19.2/17.2 ms. Barrier count dominates over occupancy for
  hand-rolled kernels — do NOT chase occupancy with tiny K-slices unless the
  staging is pipelined (async copies).
- **Warp-broadcast A reads are free; B reads are the conflict surface.** With
  the warp spanning the N direction, A reads (`shared_a[row, k]` — same k for
  the whole warp) are broadcasts; only B needs padding. Swapping the block
  dims so warps span M turned a 29 ms kernel into 17.2 ms.
- **Shared row stride 64 = 32-way conflicts on row-strided reads.** Any tile
  row stride divisible by 32 banks aliases every row to the same bank. Pad by
  1 unless reads are true broadcasts.
- **Staging must be warp-coalesced on global reads**: consecutive threads →
  consecutive columns of one row. A stride-2-row staging pattern cost 10×.
- Both square 64×64 (8 GiB) and asymmetric 128×64 (6 GiB) kernels land at the
  same 17.2 ms — hand-rolled kernels here are **latency-bound**, not
  bandwidth-bound; the traffic advantage only pays off once latency is hidden.

## WMMA on gfx1201 (fp16/bf16 — there is no fp32 MMA on RDNA4)

- **fp16 WMMA works: 16×16×16 with 16-element A/B fragments and an 8-element
  fp32 accumulator** (`llvm.amdgcn.wmma.f32.16x16x16.f16`). The gfx12 native
  8-element fragment encoding has **no** implementation in the Mojo/MAX
  stdlib; the 16-element RDNA3 encoding was made selectable on gfx1201 in
  upstream fix [modular/modular#6722](https://github.com/modular/modular/issues/6722)
  (nightlies after 2026-07-24).
- Mojo's higher-level `TensorCore` API still crashes codegen on the
  accumulator store for this shape on gfx1201 — [modular/modular#6889](https://github.com/modular/modular/issues/6889).
  In HIP, rocWMMA works fine (verified upstream, `mma_sync` 16×16×16 fp16).
- rocBLAS fp16 (tensor cores): **1.4 ms / 100 TFLOPS** @ 4096³ — the card's
  real compute ceiling; fp32 CUDA-core path tops out ~15 TFLOPS (rocBLAS) /
  8 TFLOPS (hand-rolled).

## ROCm toolchain notes (TheRock 7.14 nightly stack)

- torch rocm-nightly wheels load their kernels from `torch/.kpack/torch_<arch>.kpack`,
  shipped by the per-arch **`amd-torch-device-gfx1201`** package — without it
  every kernel fails with `hipErrorInvalidImage`.
- The 7.14 sdk's `librocprofiler-register.so.0` is internally broken
  (unresolvable `rocprofiler_configure*` → fatal at dlopen → poisons HIP/HSA
  init in-process). Workaround: no-op stub with the same SONAME; the full
  idempotent fix script is
  [`mojo-gpu-puzzles/scripts/setup_rocm_sdk_fixes.sh`](../../../Code/mojo-gpu-puzzles/scripts/setup_rocm_sdk_fixes.sh).
- In-process mixing of sdk-7.14 and system-7.2 HIP/HSA libs breaks device
  init (`Failed to initialize HSA runtime`): keep one consistent pair
  (unversioned `libamdhip64.so`/`libhsa-runtime64.so` symlinks in the sdk lib
  dir + `LD_LIBRARY_PATH`).
- `rocm-smi` on stock Ubuntu lives in `/opt/rocm/bin` (not on PATH); `mclk`
  readings on gfx12 are unreliable — trust measured bandwidth, not clock
  levels.
