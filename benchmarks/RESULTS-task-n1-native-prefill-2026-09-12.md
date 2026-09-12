# RESULTS — task-n1 native bf16 prefill: baseline + parity re-check (2026-09-12)

Goal `mttxt22c-a6rv75`, task-n1. This session re-established the honest native
bf16 prefill baseline (`NPU_PREFILL_BF16=1`, `NPU_RUNLIST=0` — the hand-rolled
bf16 path, **not** `NPU_FLM_PREFILL=1` FLM orchestration) and re-verified the
contract's gate ("token parity vs FLM preserved").

## Build fix (committed)

HEAD `057ca994b` did **not compile**: `runtime_layer.cpp` defined
`RuntimeLayerEngine::prefill_batch` but `runtime_layer.h` never declared it.
Committed the declaration as `4129c4521`.

## Baseline (Qwen3-0.6B, 256-token prefill, 3 runs)

```
Prefill: 696ms (3 ms/tok) [GEMM 140ms, attn 116ms, conv+other 694ms]
Prefill: 674ms (3 ms/tok) [GEMM 134ms, attn 116ms, conv+other 672ms]
Prefill: 689ms (3 ms/tok) [GEMM 140ms, attn 117ms, conv+other 686ms]
```

→ **~370 tok/s** (686 ms / 256 tok). `conv+other` is the `tc` full-layer
counter, so the real split is GEMM ≈ 140 ms, attention ≈ 116 ms, and
**host-side math (f32↔bf16 conversions + RMSNorm/RoPE/SiLU/residuals) ≈ 430 ms**
— the dominant cost, matching the task contract's premise ("~380 tok/s").

## Token parity — BROKEN at HEAD

| path | boot token (9-tok default prompt) |
|---|---|
| FLM reference (`NPU_FLM_PREFILL=1`) | **151667** (deterministic, 2/2) |
| native bf16 + NPU attention | **39982** (deterministic, 3/3) |
| native bf16 + CPU attention (`NPU_ATTN_CPU=1`) | **24370** |

The native bf16 path does **not** reproduce FLM's boot token, and the NPU vs
CPU attention paths disagree with each other. This is consistent with the
prior-session findings (`f20f0f87b` "attention is a real divergence — NPU
91364 / CPU 79362 / FLM 62865"; `0a61a9efa` "RuntimeLayer whole-layer path also
disagrees with FLM"). The Sep-11 doc's "boot=151667 = FLM ✓" claim is not
reproducible at HEAD; the divergence predates this session's commits.

## OpenMP host-math parallelization — reverted (regression)

Attempt: `#pragma omp parallel for schedule(static)` on the per-token host
loops (input-norm rn_bf16, qk_norm, bActQ convert, O/GU/D residuals + SiLU) —
embarrassingly parallel over tokens, parity-preserving by construction.

Result: prefill **1085 ms vs 690 ms baseline (57% slower)**. Cause: fork/join
overhead × (28 layers × ~10 regions) on small per-token loops + CPU thread
contention while the NPU GEMM runs concurrently. Reverted.

## Honest status vs task-n1 contract

- **Prefill throughput**: ~370 tok/s vs FLM published 1494 @1k (native is ~4×
  short; ~2× short of FLM's on-box ~1400). The gap is the ~430 ms of host math
  that only NPU-offload of RMSNorm/RoPE/SiLU (new xclbins) can remove.
- **Token parity**: NOT preserved (39982 vs 151667) — a correctness gate that
  must be fixed before any throughput claim is meaningful.

## The "1494 @1k" bar is UNMEASURABLE on the native path — hard 256-token cap

`npu_engine_universal.cpp` truncates the native bf16 prefill input to 256
(`if(getenv("NPU_PREFILL_BF16")){ if(input_tok_file && npt > 256) npt = 256; }`)
because the attention ELF is the fixed `attn_mha_256_nh16.elf` (256-token MHA).
Verified: a 1024-token valid input prints `=== Prefill 256 ===` and runs only
256 tokens. So the contract's "1494 @1k" reference cannot even be reproduced on
this path today — closing it requires the task-n2 work (position-shifted
attention ELFs + chunked prefill) as a prerequisite, not just faster GEMM/norm.
