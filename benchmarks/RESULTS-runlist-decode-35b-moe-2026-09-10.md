# RESULTS — runlist decode extension to Qwen3.6-35B-A3B (MoE), 2026-09-10

Goal `mtusoiy1-cfdhqr`: extend the dense-Qwen3 single-launch whole-layer
per-ctx ELF runlist path (1 submit/token, 88 tok/s on Qwen3-0.6B) to the
35B-A3B MoE, lifting its ~0.7 tok/s native decode.

## Correctness (task-4) — PASSED

6 structural bugs fixed (see npu-infer/docs/35b-forward-integration.md R102-R110):
QKV K/V pointer-stride, reference emb reshape, reference l2norm per-row,
reference residual raw-vs-normed, rope_theta 1e7, STD k-norm per KV head.
Weights byte-identical (max|d|=0.0); CPU attention+MoE match to float noise.
Token 4329 is the self-consistent int8 output (float64 parity needs bf16 O/QKV).

## Measurement (task-5, this box, Strix Halo NPU)

Prompt [151644] (BOS), greedy, single-token sequential decode, 16 tokens.

| Path | launches/token | decode | per-layer breakdown |
|---|---|---|---|
| CPU MoE (default) | **80** (40 QKV + 40 O) | **1751 ms/tok (0.57 tok/s)** | QKV 9.1 + attn 4.7 + O 5.8 + FFN 26.6 ms |
| NPU MoE fused (`NPU_MOE=1 NPU_MOE_FUSED=1`, M=1) | 160 (QKV/O + MOE_GUSGU/MOE_DSD ×2) | **3082 ms/tok (0.32 tok/s)** | FFN 62.3 ms/layer (expert pack-miss ~64 ms dominates) |

Bottleneck: the MoE FFN (26.6 ms/layer CPU = 58% of the 46 ms/layer). The
expert Q4NX dequant + float32 FFN runs on the CPU; the NPU path is *slower* at
M=1 because the per-expert dequant+repack (cache miss) costs ~64 ms/expert.

## Gap vs the runlist class

| | Qwen3-0.6B (dense, goal mtunui03) | Qwen3.6-35B-A3B (this goal) |
|---|---|---|
| runlist whole-layer | 1 submit/token, 11.4 ms/tok (88 tok/s) | **infeasible**: per-ctx ELF lane NaN (R59), cross-xclbin runlist batching blocked by differing ERT group_ids (R98) |
| best working path | — | 80 launches/token, 1751 ms/tok (0.57 tok/s) CPU MoE |

The single-launch runlist path could not be extended to the 35B MoE: the
per-ctx layer ELFs produce NaNs, and the MoE xclbins use different ERT
group_ids so one `xrt::runlist` cannot batch across them. The engine's
`NPU_MOE` fused path (v28 MOE_GUSGU/MOE_DSD) is the on-box 2-launch/layer
mechanism but loses to the CPU FFN at M=1 decode.
