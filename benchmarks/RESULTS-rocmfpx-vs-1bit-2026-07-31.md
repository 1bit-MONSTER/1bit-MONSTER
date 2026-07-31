# ROCmFPX vs 1bit-systems — Qwen3.6-35B-A3B head-to-head (Strix Halo)

_2026-07-31, strixhalo (Ryzen AI MAX+ 395, gfx1151, 122 GB UMA). ROCmFPX fork
@ b41ce12 built with Vulkan + HIP (rocm-therock amdclang++, gfx1151)._

## Method

- Quantized `unsloth/Qwen3.6-35B-A3B-GGUF` BF16 (69 GB) → 4 formats with the
  fork's own `llama-quantize`.
- `llama-bench -ngl 99 -t 8 -r 2 -p 512 -n 128` (pp512+tg128, ~640 ctx — the
  README's shape) on Vulkan0 and ROCm0.
- Cross-check: 1bit-systems' own Q4_K_M baseline file (llama.cpp 5f55650a7,
  Vulkan): 75.65 t/s decode — this fork measures 72.15 t/s on the same file
  (~5% build-to-build delta, so compare percentages, not absolutes).

## Decode (tg128, tok/s)

| Quant | size | Vulkan0 | vs Q4_K_M | ROCm0 | vs Q4_K_M |
|---|---:|---:|---:|---:|---:|
| Q4_K_M (re-quant) | 19.70 GiB | 72.15 | 1.00× | 59.21 | 1.00× |
| Q4_0_ROCMFP4_FAST | 17.22 GiB | 80.07 | **+11.0%** | 55.53* | −6.2% |
| Q4_0_ROCMFP4_STRIX_LEAN | 17.31 GiB | 78.85 | +9.3% | 66.82 | +12.9% |
| Q2_0_ROCMFPX (FP2) | 10.40 GiB | 92.16 | **+27.7%** | 78.59 | **+32.7%** |

_*noisy run (±16.8); FAST on HIP needs a rerun before trusting the sign._

Prefill (pp512): Vulkan FP2 1061 t/s, FP4_FAST 988, STRIX 973, Q4_K_M 876 —
FP2 leads prefill too (+21%).

## Context-length caveat (found while cross-checking)

At **8k context** (pp8192+tg128) the FP4 family LOSES to Q4_K_M on Vulkan
(Q4_K_M 71.8, FP4_FAST 61.6, STRIX 55.2, FP2 64.2). The FP4 decode kernels
degrade with long context; FP2 stays competitive. The README's numbers are
short-context only. (The 1bit repo's own 75.65 t/s was also short-context.)

## Mining-informed read (see docs/research/rocmfpx-format-mining.md)

- **FP2's no-zero S40 codebook {-4,-1,+1,+4} + dual UE4M3 scales = the
  consistent winner on both backends** (+28-33% decode, +21% prefill, 47%
  smaller than Q4_K_M). This is the strongest argument for a no-zero 2-bit
  variant in the 1BP converter (currently TQ2 = {-s,0,+s}).
- FP4's win is real but backend/shape-sensitive — the 12→10 codebook retune
  (Qwen3 sampling) and imatrix-weighted MSE scale search are cheap wins to
  port to gguf_to_onebp.
- Tensor routing (STRIX_LEAN: attn K/V dual-scale + Q5_K embd) ≈ what the
  q4nx NPU layout already does (attn=Q8_0, experts=INT4) — validates the
  direction.

## Files

- `/home/bcloud/ROCmFPX/build-strix/bin` — built tools
- `/home/bcloud/models-bf16/BF16` — BF16 source (69 GB)
- `/home/bcloud/models-bf16/quant/*.gguf` — q4km / rocmfp4_fast / rocmfp4_strix / rocmfp2
- Raw log: `/tmp/rocmfpx_bench.log`
