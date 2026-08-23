# FLM-parity plan — close the Zaya NPU decode gap to FastFlowLM

**Status:** 4.2 tok/s (232 ms/tok) · **Target:** FLM-class ~11–20 tok/s · **Gap:** ~3–5×

## Measured baseline (strixhalo, Zaya1-8B q4nx, NPU FFN ∥ CPU attention)

| Phase | ms/tok (M=128) | ms/tok (M=16) |
|-------|---------------:|--------------:|
| MoE (40 NPU launches: 20×GU + 20×D) | ~200 | ~126 |
| CCA attention (CPU, parallelized) | ~11 | ~11 |
| lm_head matvec (memory-bound floor) | ~17 | ~17 |
| residual/router/silu | ~5 | ~8 |

## FLM's numbers (the target)

From `benchmarks/RESULTS-qwen3.6-35b-a3b-npu-flm-2026-07-30.md` and the README:

- Qwen3.6-35B-A3B (35B MoE): **11.7 tok/s** decode (FLM v0.9.46).
- Qwen3-4B (dense): **20.8 tok/s** decode.
- Zaya1-8B has *less* active work/token than either — so FLM-class throughput
  is the right bar, not a stretch goal.

## Root cause (documented in `engine/npu/AIE2P-FACTS.md` §3b)

1. **M=128-baked kernel.** The FLM `mm.xclbin` is baked for XM=128 (4 slices ×
   32 rows). Decode is M=1, so each launch executes a fixed 128-row stream for
   1 row of data. `regen_insts(M<XM)` **deadlocks** (REG_M can't resize the
   baked tiling) — M=1 and M=8 both hang.
2. **No fusion.** GU and D are two separate launches; the SiLU runs on CPU
   between them. FLM fuses the whole layer (norm → attention → FFN → SiLU-in-
   kernel) into one dispatch via `xrt::runlist`.
3. **Attention on CPU.** FLM runs QKᵀ + online-softmax + PV on-NPU (the MHA
   engine); we run CCA attention on CPU.
4. **No in-kernel activation.** FLM's GEMM takes the activation via RTP reg
   `0x100c` (0=none, 1=GeLU, 2=SiLU). Our `gemm_generate_sequence_i8` has the
   `activation` parameter but `(void)`s it.

## The fix (in order of impact / effort)

### 1. Small-M xclbins (biggest, ~50µs M=1 streams)

Build per-shape xclbins with a baked small M so decode launches run M=1 (or
M=8) streams instead of M=128. Concretely:

1. Compile the microkernel with `-DDIM_M=1` (the `#if DIM_M < 16` scalar
   `matmul_i8_i32` alias in `mm_kernel_reference.cc` already supports this).
2. Generate MLIR for M=1 (single core-row topology, `-r 1`).
3. Emit a matching M=1 instruction stream — requires un-hardcoding the 4-slice
   loop in `gemm_generate_sequence_i8` (currently `(void)M`, always 4 slices).
4. aiecc build + NPU-verify (corr vs CPU reference + token check).

Shapes needed for Zaya: GU (K=2048, N=4096) and D (K=2048, N=2048).

### 2. Fused GU→SiLU→D (one launch per MoE layer)

Wire the `activation` RTP (`0x100c = 2`) into the kernel + instruction
generator so the SiLU runs on-NPU, eliminating the GU→CPU→D round-trip and
halving the MoE launches (40 → 20).

### 3. CCA attention on NPU (longest)

Port the CCA prep (conv_qk, qk_means, L2, RoPE) + GQA attention to NPU, fusing
the attention layer into the same dispatch. This removes the ~11 ms CPU
attention and the CPU↔NPU activation round-trips.

### 4. `xrt::runlist` batching

Once per-layer is a single stream, batch the whole token's layer streams into
one runlist dispatch (FLM-style). Note the CCA recurrence (conv_state/vrec) is
token-sequential, so this batches *within* a token, not across tokens.

## Fused GU→SiLU→D design (option 1)

The MoE FFN per layer is `D @ (silu(gate) ⊙ up)` where `[gate|up] = GU @ h`.
Fusing GU and D into one launch saves the D launch's fixed overhead (~0.85 ms)
and the intermediate C writeback — ~27 ms/tok (6.2 → ~7.5 tok/s).

**Cross-tile SiLU is the crux.** GU output `[2*n_ff]` is tiled across 8 cols:
gate lands on cols 0–3, up on cols 4–7, so `silu(gate[i])·up[i]` needs gate from
tile i and up from tile i+4 — cross-tile. Fix: **interleave gate/up rows in the
packed weights** (row 2i = gate[i], row 2i+1 = up[i]) so each tile holds the
(gate, up) pair it needs and the SiLU is tile-local.

**Sigmoid on-NPU.** No HW sigmoid. Since the D GEMM quantizes to int8 anyway,
a coarse fixed-point sigmoid (polynomial or 256-entry LUT indexed by the top
bits of the int32 gate) is sufficient. RTP `0x100c` is FLM's activation selector
(0=none, 1=GeLU, 2=SiLU).

**Stages:**
1. Interleave gate/up in `packB_into` (host-side, needs the fused kernel to
   consume it).
2. New kernel: GEMM1 (GU, mmul) → SiLU (fixed-point sigmoid + gate·up) →
   GEMM2 (D, mmul), intermediate held in tile SRAM.
3. New MLIR generator chaining the two GEMMs + activation on one core row.
4. aiecc build + NPU-verify (tokens + corr vs the CPU-SiLU path).

Note: the fused SiLU is fixed-point, so it will NOT be bit-identical to the
float CPU SiLU — the bar here is token parity / corr ~0.999, not bit-exact.

### Fused contract — implemented & CPU-verified (2026-08-21, issue #1759)

The exact kernel arithmetic is pinned in `engine/npu/generators/silu_quant.h`
(dual-compiled: AIE kernel + host reference; no libm) and validated on REAL
zaya1-8b.q4nx data by `engine/npu/tests/test_fused_silu.cpp`:

- **Per-token qn_s is required.** A fixed scale (qn0) caps corr at ~0.96–0.99
  because |h2/ag| spans ~16× across tokens (p50≈90 → p99≈1472 on real data).
  The fused kernel gets qn_s = 127/max|h2| from a **host amax pass**: the host
  recomputes the GU GEMM from the same int8 inputs (integer accumulation is
  order-independent → bit-identical c1 to the NPU; ~8.4M MACs ≈ 0.1–0.3 ms
  AVX2 per layer) and folds qn_s into the per-token gs' header.
- **Measured: fused corr 0.99931–0.99958 vs the float reference — statistically
  identical to the two-launch NPU path (0.99932–0.99958); argmax parity 5/5
  tokens; maxdiff/p50/p95/p99 match the two-launch path.**
- LUT: 256-entry sigmoid over [-4, 4] (real gate_f ∈ [-3.4, 3.4]).
- Kernel topology (n1_core_fused_gu_silu_d.py): single core row, M=8 1x4 mmul;
  C1 held in tile SRAM (produce-only fifo); h2 → DDR (2 KB) → D-phase A
  broadcast; one merged B stream per column [gs tile | GU 128 | D 64].
- **UNVERIFIED (needs strixhalo)**: aiecc build + NPU-verify per
  `build_zaya_fused.sh`'s checklist — the produce-only C1 fifo lowering and the
  shim S2MM channel pressure are the two build-time risks (Design J — C1 DDR
  round trip — is the documented fallback).

## Key finding (2026-08-21): compute is solved; the wall is launch overhead

Measured decode launch wait by M:

| M | moe (40 launches) | per-launch |
|---|------------------:|-----------:|
| 128 | 200 ms | 5.0 ms |
| 16 | 126 ms | 3.15 ms |
| 8 | 126 ms | 3.15 ms |

M=16 → M=8 is a no-op: the GEMM compute is now negligible (~0.1-0.3 ms/launch),
and **~2.9 ms/launch of fixed overhead + weight DMA dominates** (40 x 2.9 = 116
ms). Reducing M further (vectorized M=1 GEMV) saves only ~5-10 ms. The levers
are therefore, in order:

1. **Fuse GU+D** — one launch per MoE layer (40 → 20), amortizing the pipeline
   fill + TCT sync + intermediate C writeback. Needs in-kernel SiLU (RTP 0x100c).
2. **Faster weight DMA** — 12.6 MB/layer streamed per token from host-resident
   BOs; DEVICE-only (uncached) BOs may cut the DMA latency.
3. **CCA attention on NPU + runlist** — the FLM architectural gap.

| Milestone | tok/s | status |
|-----------|------:|--------|
| baseline (M=128, cache, CPU scalar) | 2.2 | done |
| + CPU parallelization (lm_head + attention OpenMP) | 3.9 | done |
| + resident-expert BOs (no per-token memcpy/sync) | 4.3 | done |
| + M=16 decode xclbins (`build_zaya_m16.sh`) | **6.2** | **done** |
| + vectorized M=8 xclbin (1x4 mmul, `build_zaya_m8.sh`) | 6.3 | **done — bit-perfect; confirms compute is NOT the bottleneck** |
| + M=1 scalar decode xclbin | — | **dead end** (scalar kernel reads B row-major vs microtiled B-DMA → corr~0; also slower than M=16) |
| + fused GU+D (on-NPU SiLU, RTP 0x100c) — halves the 40 launches | ~7.5 | **NPU-VERIFIED (strixhalo): host-vs-NPU corr 1.000000 (bit-exact), MoE L1 corr 0.999528 vs float (= two-launch quality), tokens match the CPU float reference; AVX2 host amax 3.8→0.45 ms; tile-contiguous weight pack + linear B taps cut the weight-DMA wait 5.1→2.1 ms/L — measured 11.0 tok/s (91 ms/tok, 4 tokens) at short seq, EXCEEDING the ~7.5 target** |
| + attention on NPU + runlist | ~15–20 | later |
| + attention on NPU + runlist | ~15–20 | later |

## Owner / log

- `engine/npu/AIE2P-FACTS.md` — the M=128-bake + regen deadlock facts.
- `fastflowlm_analysis/FLM_SECRETS.md` — FLM's fused-layer + RTP 0x100c
  activation + runlist architecture.
