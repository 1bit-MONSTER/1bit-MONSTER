# RESULTS — task-n1: dense Qwen3 native bf16 prefill parity (baseline)

Goal `mttxt22c-a6rv75`, task-n1. Native-engine bf16 prefill (`NPU_PREFILL_BF16=1`
+ `NPU_RUNLIST=0`; dequant.xclbin + mm.xclbin + attn.xclbin driven by the native
engine's own loop, NOT `NPU_FLM_PREFILL`). This is the real native-path prefill
(host-side norms/RoPE/SiLU + FLM's NPU xclbins as the GEMM/attn kernels).

## Baseline (2026-09-11, 256-token batch, Qwen3-0.6B)

| metric | value |
|---|---|
| prefill total | **~535 ms / 256 tok = ~479 tok/s** (QKV 3→1 GEMM + scratch/A reuse + GU 12→2 GEMMs + attn f32-rt skip) |
| — QKV GEMMs (tg) | 52 ms (was 97 — folded Q/K/V into one N=4096 GEMM) |
| — attention + host norm/RoPE (ta) | 137 ms |
| — O/GU/D GEMMs + f32↔bf16 conversions + SiLU (tc−tg−ta) | 463 ms |
| token parity (9-tok default prompt) | boot=151667 = FLM ✓ |
| target (FLM published prefill @1k) | 1494 tok/s → 256 tok ≈ 171 ms |

Gap: **~3.1×** (~535 vs 171 ms). The bf16 GEMM path is token-correct; the gap is
throughput, not correctness.

## Per-layer cost (28 layers → 24.9 ms/layer)

- QKV GEMMs 3.5 ms, attention 4.9 ms, **other 16.5 ms**.
- "other" is dominated by: GU GEMMs (12 × N=512 — gate/up interleaved in the packed
  BO forces 512-row chunking), D GEMM (K=3072), O GEMM (N=1024), and ~15 f32↔bf16
  conversion/readback passes + host SiLU/norms.

## Levers (in order of size, from session-2 notes + this baseline)

1. ~~**GU chunking** (12 GEMMs/layer)~~ — **DONE**: host-dequant each interleaved
   chunk, concatenate into contiguous gate/up, upload via `bf16mm_upload_w`;
   the GU FFN is now 2 GEMMs (N=IM) instead of 12 (N=512).
2. **QKV combine** (3→1 GEMM): `Wqkv` is already dequant'd contiguously
   [q 2048 | k 1024 | v 1024] = N=4096; a single N=4096 GEMM would fold Q/K/V
   (subject to the mm.xclbin C-write capacity / tiling).
3. **f32↔bf16 round-trips** (~15 passes/layer): the bf16 GEMM output → f32 (host
   math) → bf16 (next GEMM) chain is architectural; overlap or OpenMP the passes.
4. **Host norm/RoPE** (137 ms/layer path): move q_norm/k_norm/RoPE onto the NPU or
   fuse with the attention pre-processing.

## Env to reproduce

```
NPU_RUNLIST=0 NPU_PREFILL_BF16=1 \
  engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 1 /tmp/toks256.txt
```


## Architectural finding (why the last ~3× is hard)

After the 5 optimizations above (367 → ~479 tok/s), the per-layer cost is now
evenly split three ways: GEMMs (QKV+O+gate+up+D, ~7.5 ms), attention (host
q/k-norm + RoPE + attn.xclbin, ~5 ms), and host math (f32↔bf16 conversions +
SiLU + RMSNorm + residual, ~6 ms).

FLM's prefill (1269 tok/s on-box / 1494 published) runs the whole layer as ONE
fused NPU sequence (`qwen3_npu_sequence::gen_layer_seq`) with no host round-trips
between projections. The native bf16 path does ~7 kernel round-trips per layer
(each with sync + launch + readback) plus host math in between. That per-op
round-trip structure has a floor of roughly 2–3× FLM's fused throughput, so the
remaining gap is architectural, not a few more micro-optimizations. Closing it
fully means fusing the layer into a single sequence — which is exactly FLM's own
`gen_layer_seq` orchestration (the path the audit already flagged as not-native).
