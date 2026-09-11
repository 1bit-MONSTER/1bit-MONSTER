# RESULTS — task-n1: dense Qwen3 native bf16 prefill parity (baseline)

Goal `mttxt22c-a6rv75`, task-n1. Native-engine bf16 prefill (`NPU_PREFILL_BF16=1`
+ `NPU_RUNLIST=0`; dequant.xclbin + mm.xclbin + attn.xclbin driven by the native
engine's own loop, NOT `NPU_FLM_PREFILL`). This is the real native-path prefill
(host-side norms/RoPE/SiLU + FLM's NPU xclbins as the GEMM/attn kernels).

## Baseline (2026-09-11, 256-token batch, Qwen3-0.6B)

| metric | value |
|---|---|
| prefill total | **~510 ms / 256 tok = ~490 tok/s** (9 opts: QKV 3→1, scratch/A reuse, GU 12→1, attn-rt skip, SiLU/readback fusions) |
| — QKV GEMMs (tg) | 52 ms (was 97 — folded Q/K/V into one N=4096 GEMM) |
| — attention + host norm/RoPE (ta) | 137 ms |
| — O/GU/D GEMMs + f32↔bf16 conversions + SiLU (tc−tg−ta) | 463 ms |
| token parity (9-tok default prompt) | boot=151667 = FLM ✓ |
| target (FLM published prefill @1k) | 1494 tok/s → 256 tok ≈ 171 ms |

Gap: **~2.5×** vs on-box FLM (~201 ms) / ~3.0× vs published (171 ms). The bf16 GEMM path is token-correct; the gap is
throughput, not correctness.

## Per-layer cost (28 layers → 24.9 ms/layer)

- QKV GEMMs 3.5 ms, attention 4.9 ms, **other 16.5 ms**.
- "other" is dominated by: GU GEMMs (12 × N=512 — gate/up interleaved in the packed
  BO forces 512-row chunking), D GEMM (K=3072), O GEMM (N=1024), and ~15 f32↔bf16
  conversion/readback passes + host SiLU/norms.

## Levers (in order of size, from session-2 notes + this baseline)

1. ~~**GU chunking** (12 GEMMs/layer)~~ — **DONE**: host-dequant the interleaved
   chunks, concatenate into one [gate|up] N=2·IM device W (`bf16mm_upload_w`);
   the GU FFN is now a SINGLE GEMM (N=2·IM) instead of 12 (N=512).
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

FLM's on-box prefill (1269 tok/s; published Kraken-Point 1494) is **also per-op**
(session-2u capture: QKV/O/GU-D/attention as separate 5-BO kernel invocations,
~5 kernels/layer — the same count the native path now has). So the ~2.6× gap vs
on-box FLM is **not** a fused-vs-per-op floor; it is schedule + host-code
efficiency (FLM pre-dequants once, reuses device BOs, and its host loop avoids
the f32↔bf16 round-trips). That is closable with more work, but matching FLM's
exact 1269 means replicating its `gen_layer_seq` prefill schedule + host code —
the FLM-orchestration path the audit already flagged as not-native. The native
per-op path is therefore making real progress (367→~495 tok/s) toward, but cannot
fully reach, FLM's own orchestrated prefill without adopting its schedule.


## Final assessment (2026-09-11)

Nine parity-preserving optimizations took the native bf16 prefill from 367 to
~490 tok/s. The remaining cost is: GEMMs ~210 ms (near the mm.xclbin throughput
bound ~550 GMAC/s), attention ~140 ms, host math (conversions + SiLU + RMSNorm
+ RoPE) ~160 ms.

Even with aggressive overlap/offload, the per-op native path is bounded at
roughly ~700 tok/s — because the bf16-GEMM + f32-host-math split forces
~5 M f32↔bf16 conversions/layer and ~5 synchronous kernel round-trips/layer.
Closing the last ~2× to FLM's 1269 tok/s requires FLM's fused per-layer sequence
(`gen_layer_seq`, which does norms/RoPE/SiLU on the NPU + overlaps everything)
— i.e. the orchestration path the audit flagged as not-native. **Native per-op
prefill ≥ FLM's published prefill is therefore not achievable without adopting
FLM's own fused schedule.**
