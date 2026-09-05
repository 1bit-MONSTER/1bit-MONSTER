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

## M3 empirical capture — findings (2026-09-05, strixhalo)

### llama.cpp CPU decode is BROKEN for qwen35moe (upstream finding)

Both `ggml-org/llama.cpp` **master 6a1a922d** and **v0.4.0** crash / corrupt the
heap running the Q8_0 GGUF on CPU (strixhalo): `malloc(): invalid size
(unsorted)` abort; gdb backtrace shows the poisoned heap surfacing on the first
post-compute malloc; a pristine (unpatched, no M3CAP) run degenerates into a
multi-GB `> ` REPL loop after eval. The GGUF itself is structurally sound
(gguf-py full read, 733 tensors, data readable; earlier crashes were against
the pre-completion file that was 15 MB short of the HF listing). Working
hypothesis: the fused-GPU strided q/k/v VIEW extraction (qwen35moe.cpp, nb1 =
full 8192-elt row between heads) reads OOB when the CPU non-fused path consumes
the same tensors. Implication for #1831: **llama.cpp CPU cannot serve as the
corr-gate golden** until fixed upstream (or validated on a CUDA host) — the
in-repo CPU reference (below) replaces it.

### Layout lock from source (replaces the ⚠ items; run-free, authoritative)

Read op-for-op against v0.4.0 `src/models/qwen35moe.cpp` +
`src/models/delta-net-base.cpp` + `ggml/src/ggml-cuda/gated_delta_net.cu`:

- **GDN recurrence (per token, decode):** state `S[i][col]` per head h, stored
transposed `M[col][i]` (S_v×S_v = 128×128, 32 v-heads); per token:
  `kv[col] = Σ_i S[i][col]·k[i]`; `delta[col] = (v[col] − exp(g_h)·kv[col])·β[col]`;
  `S[i][col] ← exp(g_h)·S[i][col] + k[i]·delta[col]`; `out[col] = (1/√128)·Σ_i S[i][col]·q[i]`.
  g_h = ssm_a[head]·softplus(alpha[head] + dt_bias[head]); β = sigmoid(beta[head]).
- **k-repeat 16→32:** q/k heads map `h % 16` (tiled ggml_repeat); CPU non-fused
decode keeps H_k=16 with `h%16` semantics via broadcast.
- **MoE expert indexing (locked):** fused 3-D GGUF tensors are expert-major
  contiguous: gate/up expert e block = floats `e·(512·2048)` (512 out × 2048 in);
  down expert e = `e·(2048·512)`; consumed via `ggml_mul_mat_id` semantics
  (W[k,n,ne]; out rows = n per expert). Router: softmax over 256 → top-8.
- **Full-attn:** `attn_q` rows = per-head [q 256 | gate 256] pairs (head stride
  512); q/k per-head RMSNorm; IMRoPE `ggml_rope_multi` sections [11,11,10,0] over
  64-dim blocks; `attn × sigmoid(gate)`; `attn_output` 4096→2048.
- **GDN fused-row channel order (remaining crux, hypothesis H):** attn_qkv rows
  (8192, = conv channels) = HF qkv_proj verbatim order: `[q 2048 | k 2048 | v
  4096]`, heads contiguous within each block (head dim 128 consecutive per
  head). Verified empirically by the in-repo CPU reference (below) whose
  per-layer dumps must be internally consistent + corr-matched against the
  qwen3next-engine-validated math family — before any HIP kernel uses it.

### Empirical graph capture — ALL layout items now confirmed (2026-09-05)

Root cause of the llama CPU crash found: the **fused** GDN path corrupts the
heap on CPU. `Q35_NOFUSE=1` (env override added locally, llama-context.cpp
fused_gdn_ar/ch gating) forces the non-fused CPU path → llama.cpp decodes the
35B cleanly (Prompt 33.1 t/s / Gen 16.8 t/s) → **the llama CPU golden is
restored for the corr gate**; the in-repo CPU reference is no longer required
(the crash itself is a genuine upstream CPU-fused-op bug worth reporting).

M3CAP node dump (6188 nodes, 2-token prefill) confirmed from live tensors:
- `attn_qkv @ x` → (8192,T); SSM_CONV (8192,T) → silu → q/k/v extracted as
  contiguous channel views at byte offsets **0 / 8192 (2048 fl) / 16384
  (4096 fl)** — fused order [q | k | v], heads contiguous per block, per-head
  dim 128; q/k per-head L2-norm (L2_NORM (128,16/32)); state (128,128,32);
  GDN recurrence via chunked ops (CUMSUM/TRI/DIAG/SOLVE_TRI) on prefill and
  the per-token delta rule on decode (ggml GATED_DELTA_NET, matched op-for-op
  to gated_delta_net.cu).
- Full-attn: Qcur (8192,T) → per-head views; ROPE on (256,16) Q and (256,2) K
  = classic rotary over **first 64 dims per 256 head** (rope.dimension_count
  64; sections [11,11,10,0] collapse to uniform theta 1e7 when all bases
  equal — matches partial_rotary 0.25) → reuse the dense backend rope kernel
  with n_rot=64.
- MoE: router MUL_MAT (256,T); experts via **MUL_MAT_ID** (40 layers × 3) on
  the fused 3-D tensors, per-expert out rows 512 (gate/up) / 2048 (down) —
  expert-major contiguous blocks confirmed; shared expert = plain MUL_MATs
  (512/T, 2048/T) + sigmoid scalar gate.

### Reference plan (replaces llama.cpp CPU for the corr gate)

In-repo CPU reference `tools/qwen35moe_cpu_ref.cpp` (GgufReader f32 path):
implements the exact math above for the qwen35moe GGUF schema; validated by the
math-family parity trick — the SAME GDN/MoE math compiled against the
qwen3next-80B q4nx GGUF (this repo's own converted file on strixhalo) must
match the repo's validated `qwen3next_engine.cpp` outputs (corr 0.9997 vs HF),
proving the math; then the 35B GGUF run is the trustworthy golden for the HIP
kernels (decode-vs-golden corr ≥ 0.99).

## Validation gate (unchanged from M1 doc)

Decode-vs-CPU corr ≥ 0.99 over ≥100 tokens on strixhalo; byte-parity for
deterministic stages; dense models untouched.
