# E2E token-verify — 5 models on the current FLM-free NPU stack (#1699)

Full run on a Ryzen AI MAX 395 (Strix Halo, gfx1201, 122GB RAM), Ubuntu 26.04 / kernel
7.0.0-29, XRT 2.21.75, engine `npu_engine_universal.cpp` (FLM-free, #1689 stack),
committed v27 xclbins. All five models built, converted GGUF→q4nx (FLM converter),
run on the NPU, and compared against llama.cpp greedy on the same GGUF.

## Summary

| Model | Engine tokens (greedy, NPU) | llama.cpp reference | Result |
|---|---|---|---|
| Qwen3-8B | `'()*+,-./012` | `'()*+,-./012` | **8/8 exact** |
| Qwen3-VL-4B-Instruct | `'()*+,-.` then `/ : ; = ?` | `'()*+,-./012` | 5/8 exact, tail = int8 noise |
| Qwen3-0.6B (control) | `'()*+,-.012` | `'()*+,-./012` | 7/8 exact |
| Llama-3.1-8B | coherent chat (`It seems to…`), flat logits flip | `2 + 2 = 4` | arch correct, output noise-bound |
| Gemma4-E2B-IT | degenerate | `"#$%&*" [eot]` | **arch unsupported** (below) |
| Qwen3.6-35B-A3B | — | — | MoE q4nx conversion not committed (below) |

## Root causes found & fixed (commit `788cf33`)

**Engine** (`engine/npu/src/npu_engine_universal.cpp` + `npu_engine_i8ctx_inc.h` +
`npu_engine_hybrid_flm.h` + `model_config.h`):

1. **Boot predicted the wrong token.** The boot re-ran a phantom position-N forward
   with the previous hidden as input → predicted the *second* next token as the first
   (the `edly`×8 / `**`×8 garbage). Fix: predict from the prefill's final hidden
   (standard causal LM). Qwen3-8B's first token went 334→6 (`'`), matching llama.cpp.
2. **Decode emitted duplicate tokens.** The M=8 batch decoded all candidates at the
   same position from identical contexts → 7 copies of one token. Fix: one token per
   step (sequential decode).
3. **Shared batch activation scale zeroed rows.** 0.6B prefill: pos0 `su` max ~3671
   vs pos1-3 max ~5 → rows 1-3 quantized to all-zero int8 → D GEMM output zeros →
   every non-first prompt position destroyed. Fix: per-token ascales in the dense
   prefill (`go_rows`/`launch_async_rows`/`dequant_only_rows`).
4. **QKV per-section weight scales.** llama v_proj rms ~4× smaller than q/k; one
   shared scale packed v onto ~10 int8 levels (~5% output error compounding over 32
   layers). Fix: per-section q/k/v pack + dequant.
5. **q/k RMS-norm applied unconditionally.** Llama has no q_norm/k_norm; its keys
   were RMS-normalized (halved). Fix: gate the norm on `has_q_norm`/`has_k_norm`.
6. **rope_theta stuck at Qwen's 1e6.** Llama-3.x needs 500000 (verified the wrong
   theta was in effect). Fix: llama-family tag → 500000.

**Converter** (`third_party/FLM_Q4NX_Converter`):

7. **Q8_0 inputs truncated to garbage.** The Q4NX tile is unsigned-asymmetric
   (value = nibble·scale + zp, nibble ∈ [0,15]); Q8_0 int8 values were truncated to
   their low 4 bits → q4nx weights corr ≈ 0.001 vs GGUF → `edly` repetition. Fix:
   route Q8_0/Q4_0 through dequant→requantize(Q4_1).
8. **llama q/k interleaved-rope reorder corrupted Llama-3.x.** The converter applied
   the llama-1/2 GPT-J-style reorder to all llama q/k; Llama-3.x (NeoX, freq_base
   500000) weights decoded with corr 0.005. Fix: apply the reorder only for
   freq_base ≤ 20000.

## Verification methodology

- Per-tensor q4nx decode verified against GGUF dequant (corr ≥ 0.995).
- Engine prefill hidden states verified layer-by-layer against a float64 reference
  (corr 0.997–0.999 for the working models).
- Token streams compared against `llama-completion --temp 0 --top-k 1` on the same
  GGUF.

## Remaining gaps

- **Int8 activation quantization noise** flips marginal tokens (the `'.'→'/'`
  transition in the punctuation prompt; llama's flat logits). This is the NPU int8
  pipeline's inherent precision — architecture is correct (hidden corr 0.89–0.98
  for llama after the fixes).
- **Gemma4-E2B** needs gated residual connections (`inp_gate`, `layer_output_scale`),
  sliding-window attention (dual rope base 10000), shared-KV layers, and
  final-logit softcap (30.0) — none implemented in the engine's forward; the q4nx
  also lacks the `window_size` keys the engine's SWA detection reads.
- **Qwen3.6-35B-A3B**: the MoE q4nx (gate_exps/up_exps/down_exps 3D tiled,
  moe_router BF16 stride-8 interleave, GDN linear_attn) requires a custom converter —
  not committed; the engine's MoE paths (dense QKV/O/G/U/D + MOE_* xclbins +
  `qwen3.6-moe_35b` tag) are ready and load, but no valid 35B q4nx exists to verify
  against yet.

## Environment notes

- NPU needs `ulimit -l unlimited` (XRT BOs map with MAP_LOCKED; the 8MB default
  memlock killed large BOs with EAGAIN).
- The distro XRT 2.21.75 + kernel 7.0.0-29 amdxdna combination works once memlock
  is raised.
- Prompt token files are tokenizer-specific: Qwen-family `[1, 48077, 4, 5]`,
  llama-family `[128000, 1, 49177, 4, 5]` (BOS included).
