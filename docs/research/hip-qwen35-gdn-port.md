# feat(gpu): qwen3_5_moe (35B-A3B) on the HIP backend — port map (#1831)

**Status:** scoped / M1 plan. Issue #1831. Branch `feat/hip-qwen35-gdn`.

## Goal

`backend_hip_1bp` currently hard-rejects every `qwen3_5_moe` model at init
(missing `attn_v/ffn_gate/ffn_up/ffn_down` dense tensors, arch check #1624).
Goal: run the 35B-A3B (GatedDeltaNet + gated GQA + gated MoE) on the HIP GPU
path behind the `RCPP_ARCH_QWEN35` gate (= 21, `include/rocm_cpp/bitnet_model.h`;
GGUF arch strings that map to it: `interns2preview`/`kimik2`/`quark`/
`qwen3_5dllm`).

## Ground truth collected (2026-09-05)

- **Arch gate exists:** `cfg.arch == RCPP_ARCH_QWEN35`; `model_router` already
  routes the family (CPU + NPU lanes exist); the HIP lane is the gap.
- **CPU reference (math, exact):** `src/qwen3next_engine.cpp` — per-layer
  `linear_attention | full_attention` dispatch, GatedDeltaNet delta rule
  (conv1d → silu → q/k l2norm → `state *= exp(g)` with `g = -exp(A)·softplus(a+dt)`,
  `beta = sigmoid(b)`, group RMSNorm (direct weight, norm-before-gate) × silu(z),
  out_proj), full-attention `q_proj [2*hd]` q+gate split + q/k RMSNorm + partial
  rope × sigmoid(gate), MoE (softmax router top-k + norm_topk, gated experts,
  shared expert + shared_expert_gate sigmoid). Validated corr 0.9997 vs numpy.
  **Caveat:** consumes HF safetensors naming (`model.layers.N.linear_attn.*`,
  `mlp.experts.N.*`, `lm_head.weight`) + `config.json` `layer_types`.
- **NPU reference (schema family):** `engine/npu/src/npu_engine_universal.cpp`
  consumes 1BP-side names `model.layer.N.linear_attn.ssm_a` /
  `.ssm_alpha_proj.weight` / `.ssm_beta_proj.weight` / `.ssm_conv1d.weight` /
  `.ssm_dt.bias` / `.ssm_norm.weight` / `.ssm_out_proj.weight`, MoE
  `model.layer.N.mlp.gate_exps_proj.weight` + `share_*_exps_proj` +
  `shared_expert_gate.weight` (+ JSON sidecar manifest). GDN `ssm_a`/`dt_bias`
  are `[32]` (linear value heads), `ssm_norm [128]` (value head dim).
- **Issue #1831's GGUF tensor list (community GGUF):** fused
  `attn_qkv.weight`; `ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt.bias,
  ssm_norm, ssm_out, attn_gate`; MoE `ffn_gate_exps / ffn_up_exps /
  ffn_down_exps.weight`, `ffn_gate_inp`, shared `ffn_gate_shexp/up_shexp/
  down_shexp`; `attn_output_gate: true`.

## Schema hazard (why M1 is plan-first, not blind code)

Three naming dialects exist for the SAME arch family and the hip backend's
GGUF-direct mode must match the writer that produced the target file:
llama.cpp-style GGUF (`blk.N.*`), HF safetensors (`model.layers.N.*`), 1BP/NPU
(`model.layer.N.*`, no `.weight` on `ssm_a`). The community 35B GGUF's exact
kv-key names (head counts, head_dim, linear heads, `layer_types` encoding,
expert counts) are NOT confirmed in any checked-out source; the producing
converter lives in the ROCmFP4/llama.cpp fork. Guessing keys would produce a
loader that silently finds nothing — the failure mode #1627/#1624 exist to stop.
M1 therefore = this plan + a schema-capture procedure; M2 = code once the
capture is in hand.

## Milestones

- **M1 (this PR):** port map + capture procedure. No engine behavior change.
- **M2 — load path** (`backend_hip_1bp.cpp` init, GGUF-direct only):
  1. Gate on `cfg.arch == RCPP_ARCH_QWEN35` before the dense #1624 check.
  2. Probe actual tensors via `GgufReader::tensor_names()` / `tensor_info()`
     (shape[0] fastest-varying) and metadata via `kv_keys()` suffix matching —
     classify subfamily by presence rules (fused `attn_qkv` vs split q/k/v;
     `ssm_*` vs `linear_attn.*`; `ffn_*_exps` vs `mlp.experts.N.*`).
  3. Parse dims from shapes + metadata (no guessing): H, full-attn NH/NKV/HD,
     linear NK/NV/KHD/VHD, conv kernel, `decoder_sparse_step`, expert NE/topk/
     MIE/SIE, `norm_topk_prob`, eps, rope_theta, partial_rotary.
  4. Fused `attn_qkv` → split q/k/v rows per the row layout the converter used
     (recorded from capture); validate each slice size before accept.
  5. Refuse decode loudly until M3/M4 kernels land (no silent garbage).
- **M3 — GDN layer decode:** conv1d + delta-rule recurrent kernels with
  per-layer recurrent state (port the math from `qwen3next_engine.cpp`
  `linear_attn()`; device state layout mirrors the CPU engine).
- **M4 — full-attn + MoE decode:** `q_proj[2*hd]` gate split + partial rope +
  qk-norm path; MoE router softmax top-k + experts (grouped or top-k gemv loop
  over existing h1bp kernels) + shared expert + output gates.
- **M5 — validation on strixhalo:** decode-vs-CPU corr gate (run the 35B on
  `qwen3next`-class CPU reference vs the HIP path over the same prompts;
  tokens 1000/1001 methodology; H1BP_DUMP bisection per layer).

## Acceptance gate (M5)

- Logits corr ≥ 0.99 vs CPU reference over ≥ 100 tokens (multi-prompt), on
  strixhalo TheRock (gfx1151); byte-parity for the deterministic layers.
- Dense models: zero behavior change (existing CI + regression).

## Blockers to lift before M2

1. **Schema capture** of the actual target file (35B-A3B community GGUF and/or
   official `model.q4nx`) — kv keys + tensor list + the fused-QKV row layout.
   Procedure below.
2. **Router wiring:** confirm `model_router` offers a HIP candidate for the
   qwen35moe arch once the backend can load (CPU/NPU lanes exist; HIP lane add).
3. **Model on strixhalo:** a 35B GGUF/q4nx staged at `~/models/` for the M2/M5
   loops (visible set today is GLM-4.7/MiniCPM/Qwen3-0.6B/zaya + qwen3next-80b).

## Schema-capture procedure (run on strixhalo against the target file)

```
# kv keys + tensor table via the GGUF reader this backend uses:
# add a debug env (H1BP_SCHEMA_DUMP=<path>) that prints, for the opened GGUF:
#   kv_keys() sorted; per tensor: name, shape (GGUF order), dtype, numel
# then: grep -nE "attn_qkv|ssm_|linear_attn|ffn_gate|ffn_up|ffn_down|exps|gate_inp|shared|output_norm|token_embd|output\b|layer_types|full_attention" 
```
Expected output drives M2's exact name table + fused-QKV slice geometry.

## Captured schema (authoritative — 2026-09-05)

Captured from the header of the real target file (`unsloth/Qwen3.6-35B-A3B-GGUF`
`Qwen3.6-35B-A3B-Q8_0.gguf`, arch token **`qwen35moe`**, GGUF v3, 733 tensors,
40 layers × 19 tensors). This is the M2 loader table.

KV (arch `qwen35moe.*`):

| Key | Value |
|-----|-------|
| embedding_length | 2048 (H) |
| block_count | 40 |
| attention.head_count / head_count_kv | 16 / 2 |
| attention.key_length / value_length | 256 / 256 |
| attention.layer_norm_rms_epsilon | 1e-6 |
| rope.dimension_count / dimension_sections | 64 / [11, 11, 10, 0] |
| rope.freq_base | 1e7 |
| ssm.conv_kernel / group_count | 4 / 16 |
| ssm.inner_size / state_size | 4096 / 128 |
| ssm.time_step_rank | 32 |
| expert_count / expert_used_count | 256 / 8 |
| expert_feed_forward_length / expert_shared_… | 512 / 512 |
| full_attention_interval | 4 (every 4th layer full MHA) |
| context_length | 262144 |

Per-layer tensors (GGUF-native shape, shape[0] fastest; row-major = reversed):

| blk.N. name | gguf shape | row-major | dtype |
|-------------|-----------|----------|-------|
| attn_norm.weight | (2048,) | (2048,) | F32 |
| attn_qkv.weight | (2048, 8192) | (8192, 2048) | Q8_K |
| attn_gate.weight | (2048, 4096) | (4096, 2048) | Q8_K |
| post_attention_norm.weight | (2048,) | — | F32 |
| ssm_a | (32,) | — | F32 |
| ssm_alpha.weight / ssm_beta.weight | (2048, 32) | (32, 2048) | Q8_K |
| ssm_conv1d.weight | (4, 8192) | (8192, 4) | F32 |
| ssm_dt.bias | (32,) | — | F32 |
| ssm_norm.weight | (128,) | — | F32 |
| ssm_out.weight | (4096, 2048) | (2048, 4096) | Q8_K |
| ffn_gate_inp.weight | (2048, 256) | (256, 2048) | F32 |
| ffn_gate_exps / ffn_up_exps | (2048, 512, 256) | (256, 512, 2048) | Q8_K |
| ffn_down_exps | (512, 2048, 256) | (256, 2048, 512) | Q8_K |
| ffn_gate_shexp / ffn_up_shexp | (2048, 512) | (512, 2048) | Q8_K |
| ffn_down_shexp | (512, 2048) | (2048, 512) | Q8_K |
| ffn_gate_inp_shexp.weight | (2048,) | — | F32 |

Globals: `token_embd.weight` + `output.weight` both (248320, 2048) Q8_K
(probably tied — compare data offsets at load); `output_norm.weight` (2048) F32.
Vocab 248320.

Open questions for M3 (kernel math, NOT M2): attn_qkv 8192-row slice semantics
(full-attn q/k/v vs GDN q/q-halves), attn_gate 4096-row role, conv1d over which
8192 slice, ssm group/head mapping (32 vs group_count 16 × state 128), and the
fused-expert tensor row indexing (256×512×2048).

## References

- Issue #1831 (parent), #1830 (NPU-universal 35B decode bug), #1624/#1627
  (loud-rejection precedent), #2104 (dtype-dispatch rule), #1942/#2080
  (cascade program), #2115-2117 (llama.cpp fork hrx/qwen35 lane).
