# Qwen3.5-4B GDN QKV truncation — root-caused and fixed (2026-09-14)

## The bug

`npu_engine_universal.cpp` sized the QKV GEMM output buffer and the
`finish_async_rows` read-back width with `qkv_n = cfg.qkv_total`, which is the
**plain** q+k+v layout (`NH·HD + 2·NKV·HD`). Qwen3.5/3.6 do not use that layout:

| layer kind | packed QKV layout | width |
|---|---|---|
| GDN (linear_attention) | q[KD] + k[KD] + v[VD] | 2·KD+VD = **8192** |
| full-attention (fused) | q[NH·HD] + gate[NH·HD] | 2·NH·HD = **8192** |
| plain q+k+v (`qkv_total`) | q[NH·HD] + k[NKV·HD] + v[NKV·HD] | **6144** |

The pack path already widens `cq.ND` to 8192 (commit 6d03d0529 fixed the pack
overrun), but `qkv_n` stayed at 6144. The NPU GEMM produced all 8192 rows, the
read-back kept only the first 6144, and **the second half of the GDN value
projection (v[2048:4096]) was silently dropped** — `gdn_attn_step` then read
zeros where the v-tail belonged. First symptom: Qwen3.5-4B boot **163554** vs
FLM's **16** on the 16-token probe.

## The fix

`qkv_n` is now widened for GDN/MoE models:

```cpp
int qkv_n = cfg.qkv_total;
if (cfg.has_gated_delta_net || cfg.has_moe) {
    if (max_gdn_conv_dim > qkv_n) qkv_n = max_gdn_conv_dim;  // GDN q+k+v
    int fused_w = 2 * NH * HD;                               // fused q+gate
    if (fused_w > qkv_n) qkv_n = fused_w;
}
```

This covers both GDN and fused full-attn (both 8192 here) and also fixes the
same truncation in the Qwen3.6-35B-A3B MoE GDN path (`qkv_total` there is 5120).

## Verification (byte-level, layer 0)

Golden references `tools/q35_l0_golden.py` / `tools/q35_l0_conv_check.py`
replicate the engine's dequant (4736-byte I8 and 8704-byte Q8_0) + GDN math in
float64 and diff each intermediate against the `NPU_DUMP_L0` dumps:

| intermediate | result |
|---|---|
| qkv (QKV GEMM out) | **corr 0.9999** (int8 quantization residual only) |
| conv1d + silu | **corr 1.0000** |
| q/k repeat + L2-norm | **corr 1.0000** |
| alpha/beta → g, β | **corr 1.0000** |
| z-gate | **corr 1.0000** |

Result: Qwen3.5-4B boot moved **163554 → 50** (FLM reference 16).

## Remaining gap: 50 vs 16

Layer 0 is now byte-correct through the gated-RMSNorm input. The **embedding
is also verified byte-correct** (Q8_0 lm_head dequant == engine `emb_f32`, corr
1.0000 for token 16). The residual boot gap is therefore **downstream of layer
0**: the O/out_proj GEMM, the MLP gate/up/down GEMMs, the full-attention layers
(`std_attn_step`, the most recently rewritten code), or accumulated int8-GEMM
quantization drift across the 32 layers.

`NPU_DUMP_LOGITS=1 NPU_GREEDY=1` shows the native argmax is **token 50 (logit
24.07)** while FLM's reference token 16 sits at **logit −6.13** — a 30-logit
margin, i.e. a **second structural bug, not quantization drift** (drift would
be a near-tie). Next probe: a full-model single-token golden (embedding → 32
layers → final norm → lm_head) to bisect which layer first diverges from FLM.

## Performance (unchanged)

Correctness is only half the bar. Qwen3.5-4B's GDN SSM is still CPU-only:
prefill **16.3 s/tok**, decode **2 tok/s**, vs FLM's published **322 tok/s**
prefill and **15.9 tok/s** decode @1k (`amd-oss/fastflowlm/docs/docs/benchmarks/
qwen3.5_results.md`, Kraken Point). Meeting that requires a fused NPU GDN kernel
(conv1d + delta-rule recurrence + gated norm in one launch) — the multi-day
path-1 effort already scoped in `docs/research/qwen36-gdn-float32-repair-plan.md`.
