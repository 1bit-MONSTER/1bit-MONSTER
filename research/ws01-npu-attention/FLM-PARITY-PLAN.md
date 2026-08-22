# FLM-parity plan — close the Zaya NPU decode gap to FastFlowLM

**Status:** 4.2 tok/s (232 ms/tok) · **Target:** FLM-class ~11–20 tok/s · **Gap:** ~3–5×

## Measured baseline (strixhalo, Zaya1-8B q4nx, NPU FFN ∥ CPU attention)

| Phase | ms/tok |
|-------|-------:|
| MoE (40 NPU launches: 20×GU + 20×D) | ~200 |
| CCA attention (CPU, parallelized) | ~11 |
| lm_head matvec (memory-bound floor) | ~17 |
| residual/router/silu | ~5 |

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

## Expected trajectory

| Milestone | tok/s |
|-----------|------:|
| now (resident experts + CPU parallelization) | 4.2 |
| + small-M xclbins (M=1 decode) | ~7–10 |
| + fused GU+D (on-NPU SiLU) | ~10–12 |
| + attention on NPU + runlist | ~15–20 |

## Owner / log

- `engine/npu/AIE2P-FACTS.md` — the M=128-bake + regen deadlock facts.
- `fastflowlm_analysis/FLM_SECRETS.md` — FLM's fused-layer + RTP 0x100c
  activation + runlist architecture.
