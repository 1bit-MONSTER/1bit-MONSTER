# HIP qwen35moe decode — M3 math spec (#1831)

Authoritative source: ggml-org/llama.cpp `src/models/qwen35moe.cpp` (master,
fetched 2026-09-05) + the Q8_0 header capture in this port map. This resolves
the §captured-schema "open questions" with the exact per-op math the HIP
kernels must reproduce. Remaining empirical unknowns (marked ⚠) get locked via
llama.cpp debug-tensor capture on strixhalo before kernel implementation —
never by guessing.

## Hyperparams (from GGUF KV, all confirmed by the model code)

| Symbol | Value | Source |
|--------|-------|--------|
| H (n_embd) | 2048 | embedding_length |
| L | 40 (10 full-attn @ (il+1)%4==0, 30 GDN) | block_count; `is_recr` |
| full-attn: n_head 16, n_head_kv 2, n_embd_head_k/v 256 | 16/2/256 | head_count(_kv), key/value_length |
| GDN: n_k_heads 16 (=ssm.group_count), head_k_dim 128 (=ssm.state_size) | — | code: `n_k_heads = ssm_n_group` |
| GDN: n_v_heads 32 (=ssm.dt_rank), head_v_dim 128 (= d_inner/n_v_heads) | — | code |
| key_dim = 16·128 = 2048; value_dim = 32·128 = 4096 | — | code |
| conv: d_conv 4, channels = value_dim + 2·key_dim = 8192 | — | code `conv_channels` |
| ssm: d_inner 4096, d_state 128, dt_rank 32, n_group 16 | — | KV |
| MoE: n_expert 256, n_expert_used 8, n_ff_exp 512, shexp n_ff 512 | — | KV |
| rope: freq_base 1e7, sections [11,11,10,0] (dimension_count 64 = block) | — | KV |
| norm eps 1e-6 | — | KV |
| vocab 248320; output.weight present (may equal token_embd) | — | capture |

## Layer kinds

`is_recr(il)` = GDN linear attention when (il+1)%4 != 0 (layers 0,1,2,4…38 —
30 layers); full attention otherwise (3,7,…,39 — 10 layers). Both kinds run
the SAME MoE block after `post_attention_norm` (no dense-FFN layers).

## Full-attention layer (10×)

1. `attn_norm` RMSNorm(x).
2. `Qcur_full = attn_q @ x` — rows = n_embd_head·n_head·2 = **8192**: per head a
   contiguous [q_h (256) | gate_h (256)] pair (code views q at offset 0,
   gate at +256, head stride 512 within the 8192).
3. Q view → per-head RMSNorm with `attn_q_norm` (256) on each head's 256 (not 1+w).
4. `Kcur = attn_k @ x` (512 rows = 2·256), reshape per head → RMSNorm with
   `attn_k_norm`; `Vcur = attn_v @ x` (512).
5. ⚠ IMRoPE: `ggml_rope_multi(n_rot, sections, …)` on Q and K — sections
   [11,11,10,0] over 64-dim blocks (dimension_count 64); exact bit layout to
   capture (llama.cpp rope op vs the repo's simple h1bp_rope differ — new kernel).
6. Scaled attention (scale 1/√256), result × sigmoid(gate) per head, `attn_output
   (attn_o @ ·)` rows 4096 → H.
7. Residual.

## GDN layer (30×) — per-token decode

1. `attn_norm` RMSNorm(x).
2. `qkv_mixed = attn_qkv @ x` — rows **8192** laid out [q 2048 | k 2048 | v 4096]
   (code view offsets: q@0, k@2048, v@4096). ⚠ intra-block interleave of the
   16/32 heads (per-head dims 128) to capture empirically.
3. `z = attn_gate @ x` — rows 4096 (value_dim) ⚠ (z per v-head dims 128).
4. `alpha = ssm_alpha @ x` (32 rows); `alpha = softplus(alpha + ssm_dt.bias)`;
   `gate = ssm_a · alpha` (elementwise, ssm_a stored as −A per repo convention —
   check llama.cpp: uses ssm_a directly; ⚠ sign convention vs repo #1460 note).
5. `beta = sigmoid(ssm_beta @ x)` (32 rows).
6. `conv_in = conv_state(qkv_mixed)`; `conv_out = silu(conv1d(·))` — conv kernel
   4 over 8192 channels with per-channel 3-deep state.
7. q/k: per-head L2-norm over head_k_dim (128) after conv; v = conv v-part.
   When n_k_heads(16) != n_v_heads(32): q/k repeat ×2 to 32 heads (⚠ repeat
   layout: llama `ggml_repeat_4d` per head index 0/1 mapping — capture).
8. Delta rule over per-head recurrent state [head_v_dim 128 × head_v_dim 128 ×
   32 heads × S]: out[t] = Σ_h q[t]ᵀ·(state·v?) — exact llm_build_delta_net_base
   `build_recurrent_attn` recurrence: state ← state·exp(gate_h)·? + k·(v −
   state·k)ᵀ·beta … (copy op-for-op from llama's GDN kernel in M3; state tensor
   shape (128,128,32,S) from `build_rs`, memory S = n_embd_s()).
9. `attn_out = ssm_norm RMSNorm(output) × silu(z_2d)` per v-head dim (norm-
   before-gate; z reshaped [head_v_dim 128, n_v_heads 32]).
10. `cur = ssm_out @ attn_out` — ssm_out rows 2048 (out H) × 4096 in. Residual.

## MoE block (every layer)

1. Router: `logits = ffn_gate_inp @ x` (256); softmax; top-8; ⚠ `norm_topk_prob`
   renormalize (config default true for qwen3.5 per qwen3next CPU engine; confirm
   in llama hparams — `expert_weights_scale`/gating handled in build_moe_ffn).
2. Experts: fused 3-D tensors — gate/up `{n_embd, n_ff_exp, n_expert}` = expert e
   slice rows e·512…(e+1)·512 (cols H); down `{n_ff_exp, n_embd, n_expert}` = e
   slice rows e·2048…(e+1)·2048 (cols 512). ⚠ gguf-py shape (2048,512,256) row-
   major (256,512,2048): expert-major with per-expert [512×2048] — capture to
   confirm slice layout before the gemv indexing.
3. Shared expert (always present, 1): silu gated FFN (shexp gate/up 512×2048,
   down 2048×512) × sigmoid(shared_gate) where shared_gate = `ffn_gate_inp_shexp
   @ x` (scalar, 2048→1). Sum: moe_out + gated_shexp.
4. Residual (add to pre-post-norm stream), next layer.

## LM head

`output_norm` RMSNorm → `output.weight @ ·` (248320×2048; tied to token_embd if
offsets equal — check at load) → logits.

## M3 kernel inventory (HIP, behind the qwen35 gate)

1. IMRoPE kernel (sections, interleaved 64-blocks) — full-attn Q/K.
2. GDN conv1d+silu kernel (8192 ch × 4, per-channel state) + q/k/v slice views.
3. GDN delta-rule recurrent kernel (state 128×128×32×S; per-head q/k l2-norm,
   gate/beta combine) — mirror llama's `build_recurrent_attn` op-for-op.
4. GDN z-gate norm kernel (group RMSNorm × silu) + ssm_out gemv (existing h1bp
   gemv reuse).
5. MoE: router softmax+top8(+renorm) kernel; per-expert gemv loop over existing
   h1bp gemv/tq2nz kernels w/ fused-expert slice offsets; shared-expert ffn +
   sigmoid gate.
6. Full-attn layer path: reuse existing backend q/k/v gemv + per-head rmsnorm +
   attn + o_proj, replacing rope with IMRoPE and adding the q/gate split +
   sigmoid gate multiply.

## M3 empirical capture (before kernels — locks ⚠ items)

On strixhalo with the staged Q8_0 file + llama.cpp: run a 2-4 token decode with
`GGML_DEBUG`/node-dump of the qwen35moe graph intermediates (qkv_mixed, conv
output, q/k views, z, gate, router logits, per-expert slices), or add a small
llama.cpp debug harness. Deliverable: exact flat offsets for the fused tensors +
rope layout + repeat mapping + expert slice indexing — the loader/kernels are
written to the capture, not to inference.

## Validation gate (unchanged from M1 doc)

Decode-vs-CPU corr ≥ 0.99 over ≥100 tokens on strixhalo; byte-parity for
deterministic stages; dense models untouched.
