# Zamba2 on Vulkan — Research & Port Plan

Status: research complete + CPU path fixed & verified (2026-08-03).
Goal: run the Zyphra Zamba2 family (1.2B/2.7B/7B) on the AMD Radeon 8060S (gfx1151)
via the ZINC Vulkan compute path, with ROCm-class throughput.

**2026-08-03 verification result:** the CPU reference (`backend_zamba2`)
now runs the EchoLabs33 1.2B q4_0 GGUF end-to-end and matches an
independent numpy implementation of modeling_zamba2.py logit-for-logit
(top-3 identical: 1302/12.072, 8888/11.581, 110/11.425; all 38 layers' hidden
states agree to ~1e-5). Verified with `tests/smoke_zamba2.cpp` +
`tools/zamba2_ref.py`. The Vulkan port can validate against the CPU reference
token-for-token (the pre-fix CPU path could not run at all — it crashed).

Fixes landed in the CPU path (all validated):
1. A-convention double-exp (#1460) — CPU + HIP + mamba1 engines
2. double-softplus in `selective_scan_step` (caller pre-softplussed)
3. `attn_head_dim` never read from GGUF KV (defaulted 80, real 128)
4. hybrid layer rewritten to the reference structure: concat(hidden, embedding)
   → post_attention_norm(2*d_model) → MHA (scale sqrt(2/head_dim)) → o_proj
   → ffn_norm → SiLU FFN → ssm_mix → mamba decoder (was: mamba-first with
   wrong norms, no concat, wrong scale, OOB attn_out write)
5. README ZINC column for zamba2/zaya: no code path (#1461)

Note: the in-flight Vulkan port (`backend_zamba2_vulkan.cpp`, agent-cf)
mirrors the OLD buggy A convention in `mamba2_scan.comp`
(`A_bar = exp(dt_sp * -exp(A_log))`) — must switch to the direct-A form
(`exp(dt_sp * A)`) before it can validate against this fixed reference.

## 1. Why this is not a "retry" — what exists today

| Model | Today's backend | Code |
|---|---|---|
| Zamba2 1.2B/2.7B/7B | HIP (ROCm) + CPU | `src/zamba2_engine_hip.hip`, `src/mamba2_kernels.hip`, `src/backend_zamba2.cpp` |
| ZR1-1.5B (dense) | ZINC Vulkan ✅ | `src/backend_zinc.cpp` |
| ZAYA1-8B | HIP + Vulkan (CCA) | `src/zaya_moe_launcher.hip`, `src/backend_vulkan.cpp` |

The ZINC C++ backend (`src/backend_zinc.cpp`) allowlists only `llama/mistral/qwen2`
and refuses everything else ("would produce silently-wrong output") — its shader
set is the dense stack only. The README "ZINC ✅" on the Zamba2 rows has no code
path behind it. **Zamba2 on Vulkan is net-new work**, not a rerun.

## 2. Architecture ground truth (from code, cross-checked CPU vs HIP)

Both `src/mamba2_kernels.cpp` (CPU reference) and `src/mamba2_kernels.hip`
(ROCm) implement the same Mamba2 SSD math. This is the spec to port:

Per Mamba2 layer, per token:

```
1. normed     = RMSNorm(input, input_norm_w, eps=1e-5)        # + residual later
2. in_proj    = in_proj_w @ normed                            # [d_in_proj]
               split: z [d_inner], xBC [conv_dim], dt [n_head]
               conv_dim = d_inner + 2*n_group*d_state
               d_in_proj = d_inner + conv_dim + n_head
3. xBC_conv   = conv1d(xBC) + conv1d_b                        # d_conv=4 taps
4. xBC_act    = silu(xBC_conv)
               split: x [d_inner], B [n_group*d_state], C [n_group*d_state]
5. scan per head h (heads_per_group = n_head/n_group, group g = h / heads_per_group):
     dt_sp     = softplus(dt[h] + dt_bias[h])      # stable: x>20 → x
     A_bar     = exp(dt_sp * (-exp(A_log[h])))     # A = -exp(A_log)
     for hd in 0..head_dim-1:
       state[s] = A_bar * state[s] + dt_sp * B[g,s] * x[hd]     # s in 0..d_state
       y[hd]    = Σ_s C[g,s]*state[s] + D[h] * x[hd]
6. y_inner   = GroupRMSNorm(y, norm_w)             # group_size = d_inner/n_group, eps 1e-6
7. y_inner   = y_inner * silu(z)
8. out       = out_proj_w @ y_inner                 # residual: out += input
```

Hybrid layer (ids [6,12,18,24,30,36,42,47,51] on 2.7B): input norm → Mamba2
decoder (same math, with its own `ssm_*` weights + `attn_norm`) → residual →
`ssm_mix` linear → shared attention (MHA 32q=32kv; RoPE per §3 — 1.2B/7B
only, 2.7B has none) → `post_attention_norm` → SiLU FFN
(`ffn_gate`/`ffn_up`/`ffn_down`) → `ffn_norm` → residual. The GGUF converter
duplicates shared-block weights per hybrid layer.

Key detail: the per-head scan shares ONE d_state vector across all head_dim
elements of a head (state is NOT per-dimension). head_dim = d_inner/n_head.

## 3. Model configs (verified from HF config.json, 2026)

| | 1.2B-v2 | 2.7B-v2 (in-code default) | 7B-v2 |
|---|---|---|---|
| n_layers / n_hybrid | 38 / 6 (every 6th) | 54 / 9 | 81 / 13 |
| d_model | 2048 | 2560 | 3584 |
| d_inner | 4096 | 5120 | 7168 |
| n_head (ssm) | 64 | 80 | 112 |
| head_dim | 64 | 64 | 64 |
| d_state | **128** | **64** | **64** |
| n_group | 1 | 1 | **2** |
| d_conv | 4 | 4 | 4 |
| chunk_size | 256 | 256 | 256 |
| attn heads | 32×128 MHA | 32×160 MHA | 32×224 MHA |
| shared attn blocks | 1 | 2 (ABAB) | 2 (ABAB) |
| RoPE in attn | yes | **no** | yes |
| LoRA adapters | attn+MLP, rank 128 | MLP only, rank 128 | MLP only, rank 128 |

Notes: attention is **MHA (32 q = 32 kv), not GQA**; 2.7B attention has no
RoPE (`use_mem_rope=false`) — verify against the GGUF's KV and the in-repo
`apply_rope` caller. All configs read from GGUF KV at load time
(`gguf_zamba2_loader.cpp`) — the engine must stay config-driven, not hardcoded
(the HIP kernels already are). ⚠️ The loader never reads LoRA adapter tensors —
unknown whether the target GGUFs carry them; if they do and the HIP path
ignores them, that's a silent correctness gap in the existing backends too.

## 4. GGUF layout (llama.cpp convention, from `src/gguf_zamba2_loader.cpp`)

Per layer: `blk.N.attn_norm.weight`, `blk.N.ssm_in.weight` (transposed to
[d_in_proj, d_model]), `blk.N.ssm_conv1d.weight` [d_conv, conv_dim],
`blk.N.ssm_conv1d.bias`, `blk.N.ssm_dt.bias`, `blk.N.ssm_a` (A_log),
`blk.N.ssm_d` (D), `blk.N.ssm_norm.weight`, `blk.N.ssm_out.weight`
(transposed). Hybrid adds: `blk.N.ssm_mix.weight`, `blk.N.attn_q/k/v.weight`,
`blk.N.attn_output.weight`, `blk.N.post_attention_norm.weight`,
`blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`, `blk.N.ffn_down.weight`,
`blk.N.ffn_norm.weight`. Globals: `token_embd.weight` (tied LM head, transposed
to [vocab, d_model]), `output_norm.weight`.

Quant: the loader dequantizes F32/F16/Q4_0/Q8_0/Q4_K/Q6_K → fp32 in host RAM,
uploads fp32. Q4_0/Q8_0 GGUFs exist on HF (EchoLabs33). The Vulkan path can
start fp32-only (weights already fp32 in VRAM) and reuse the existing
`dmmv_q4k`/`dmmv_q8_0` shaders later for quantized resident weights.

## 5. Reusable Vulkan assets (this is the big unlock)

`engine/gpu/src/shaders/*.comp` (GLSL → glslc → SPIR-V, compiled copies in
`engine/fusion/shaders/vulkan/*.spv`, dispatched by the zig ZINC runtime in
`/home/bcloud/zinc`) already contain:

- **`ssm_conv1d.comp`** — Mamba2's exact conv1d+SiLU with a circular state
  buffer (per-layer `state_offset` counter, single state write per channel per
  token, d_conv taps, f32/f16 kernel select). **Directly reusable for Zamba2**
  with one gap: it has **no bias binding** (Mamba2 needs `ssm_conv1d.bias`).
  Also expects kernel layout [conv_channels, d_conv] — GGUF gives [d_conv,
  conv_dim], so transpose at load or index strided.
- **`ssm_delta_net.comp`** — the structural template for the scan: per-head
  recurrent state in device memory, workgroup = 64 threads = 1 wave64,
  **token-loop folded inside the shader** (A3: state read once at t=0, written
  once at t=n_tok-1 — prefill collapses to ONE launch, recurrence preserved =
  bit-identical to per-token decode). Subgroup ops (`GL_KHR_shader_subgroup_*`)
  for the dot products. Qwen3.5 Gate-Delta update rule differs from Mamba2
  (decay*gate vs A_bar, beta-gated write vs dt*B*x), but the dispatch shape,
  state lifecycle and wave64 pattern port directly.
- Dense ops the Mamba2 layer needs: `dmmv_f32.comp` (in_proj/out_proj/ssm_mix),
  `rms_norm*.comp` (input/final norms), `flash_attn.comp` (hybrid attention),
  `silu_mul.comp` (z-gate), `rope_fused.comp`, `mul_elementwise.comp`
  (residuals), `dmmv_q4k*/q8_0*` (quantized weights later).

The C++ `zinc_cpp` engine (`engine/gpu/zinc_cpp`) is the simpler 3-binding
dispatch API (`dispatch(shader, push, in, out, weights, gx, gy, gz)` +
PushConstants M/N/K/scale/eps/token/layer/head/pos) and is what
`backend_zinc.cpp` uses. The zig runtime has the richer descriptor model.

## 6. Gap analysis — what must be written

| Missing | Notes |
|---|---|
| `mamba2_scan.comp` | THE kernel. See §7 spec. Closest prior art: ggml-vulkan `ssm_scan.comp` (llama.cpp — the only existing Mamba2-style scan on Vulkan: subgroup-per-head, fp32, grouped) and in-house `ssm_delta_net.comp` (token-fold + wave64 pattern). |
| group RMSNorm | `rms_norm.comp` is full-dim; needs per-group reduction (d_inner/n_group) |
| conv1d bias | extend `ssm_conv1d.comp` with a bias binding (or fold bias into the scan kernel's state? No — bias is applied pre-silu; add binding) |
| Mamba2 state lifecycle | persistent per-layer buffers: conv state `[n_layers][d_conv-1][conv_dim]`, SSM state `[n_layers][n_head][d_state]`, per-layer `state_offset` counters, KV cache `[n_hybrid][2][max_seq][n_kv][hd]` |
| Binding model | scan needs x, B, C, dt, dt_bias, A_log, D, state-in/out, y ≈ 9 buffers — beyond zinc_cpp's 3-binding `dispatch()`. Either extend `dispatch` to N bindings or give the scan its own descriptor set. |
| Prefill orchestration | dense prefill (in_proj GEMV over T tokens = GEMM; `mul_mm_*` shaders exist) + token-folded scan + conv1d over T (needs `ssm_conv1d_batched`-style or t-loop) |
| Backend wiring | new `zamba2_vulkan` backend (mirror `backend_zamba2.cpp`/`backend_zinc.cpp`), route `{"zamba2_vulkan", "zamba2_gpu", "cpu_generic"}` in `model_router.cpp`, factory entry, README ZINC-column fix |

## 7. `mamba2_scan.comp` spec (decode L=1)

Follow the HIP kernel's proven structure, mapped to wave64. The recurrence is
purely discrete — **no ZOH discretization** (verified against the mamba_ssm
reference `mamba2.py::step`):

```
per token, per head h:
  dt_sp  = softplus(dt[h] + dt_bias[h])          # stable: x>20 → x
  dA     = exp(dt_sp * A[h])                     # A[h] < 0 (see A-convention note)
  dBx    = dt_sp * B[g,s] * x[hd]
  state[s] = dA * state[s] + dBx                 # per d_state element
  y[hd]    = Σ_s C[g,s]*state[s] + D[h]*x[hd]
```

dispatch: workgroups = n_head, local_size = 64 (1 wave64)
per workgroup (head h):
  lane s (< 64) owns state[s]                    # d_state = 64 → 1:1 lane map (2.7B/7B)
  group g = h / heads_per_group
  for hd in 0..head_dim-1:
    xv = x[head*head_dim + hd]
    state = dA * state + dt_sp * B[g*d_state + s] * xv
    dot = subgroupAdd(C[g*d_state + s] * state)
    if lane == 0: y[head*head_dim + hd] = dot + D[h]*xv
  write state back once
```

- **d_state=64 → one wave64, no shared-memory reduce** (the HIP kernel needed
  LDS only because it used 2×wave32). If the pipeline reports wave32, fall
  back to LDS reduce — keep the `ssm_delta_net.comp` precedent (pins wave64
  via `local_size_x=64` + subgroup ops).
- **d_state=128 (1.2B)**: 128 lanes or 2 waves + LDS reduce — port the HIP
  kernel's `SCAN_DSTATE_THREADS=64` cross-warp LDS path verbatim.

Prefill (L>1): fold the token loop inside the shader (delta-net A3 pattern) —
state stays in registers across t, B/C/dt/x read at per-token offsets, state
written once. Correct-by-construction and state-identical to decode; the
chunked parallel scan (mamba2 reference, chunk_size=256) is a later
optimization, not a correctness requirement. (llama.cpp's HIP path also falls
back to sequential scan for prefill — sequential is not a cop-out.)

⚠️ **A-convention — RESOLVED (2026-08-03, issue #1460)**: llama.cpp's
conversion stores `ssm.a` already-negated (A = -exp(A_log)). Verified with
`tools/dump_ssm_a.py` on the EchoLabs33 1.2B q4_0 GGUF: `blk.0.ssm_a` ∈
[-15.19, -0.42], all 64 values negative → **the in-repo HIP/CPU engines
re-applying `-exp(A_log)` are WRONG (double exp)** — A_bar ≈ 1.0, SSM state
never decays, output garbage. Use stored A directly: `dA = exp(dt_sp * A)`.
Same convention applies here.

## 8. Validation plan (house style, per #844 precedent)

1. Unit: `mamba2_scan.comp` vs `selective_scan_step()` (`mamba2_kernels.cpp`)
   on random inputs, tol 1e-5 (fp32 both sides; identical op order where cheap).
2. End-to-end: Zamba2-1.2B-v2 Q4_0 GGUF (EchoLabs33, `scripts/download_zamba2.sh`),
   token-for-token vs `backend_zamba2` CPU path; then vs `zamba2_gpu` HIP for
   cross-backend agreement.
3. Prefill/decode seam: run prefill(64) → decode(1) and compare state vectors
   against pure decode from reset — must match exactly (recurrence preserved).
4. Benchmark decode tok/s on 8060S; compare vs 30 tok/s (README claim) and the
   HIP numbers. Target: beat CPU (≈2.5 tok/s class) first, then chase HIP.

## 9. Risks / open questions

- **Quantized weights**: fp32 weights in VRAM cost 2-4× GGUF size (1.2B Q4_0
  ≈ 1.1 GB file → 2.4 GB fp32). Start fp32 (correctness first); the
  `dmmv_q4k`/`dmmv_q8_0` shaders are the upgrade path, and the scan's B/C/x/dt
  are activations anyway (always fp32).
- **Subgroup size**: RDNA4 runs wave32 by default; the delta-net shaders
  assume wave64 (`local_size_x = 64`, "1 full wave"). Verify actual subgroup
  size at pipeline creation (`VkPhysicalDeviceSubgroupProperties`) and pin it.
- **State compatibility between prefill and decode** is the classic SSM bug
  class — the A3 fold avoids it; do not "optimize" prefill into a different
  recurrence order without a state-equivalence check (llama.cpp shipped this
  bug class: #18631, #18606, #20570).
- **A-convention / double-exp risk** — **CONFIRMED BUG, issue #1460** (§7);
  affects the existing HIP + CPU paths, not just the Vulkan port.
- **conv1d circular buffer vs GGUF weight layout**: [d_conv, conv_dim] in GGUF
  vs [conv_dim, d_conv] expected by `ssm_conv1d.comp` — transpose at load.
- **`ssm_conv1d.comp` bias**: needs a binding addition; verify whether the
  batched variant has it.
- **LoRA adapters**: RESOLVED for 1.2B — the EchoLabs33 q4_0 GGUF contains
  zero adapter/lora tensors (405 tensors checked), so the loader's silence is
  consistent with the file. 2.7B/7B files unverified.
- **F32 tensors hold f16-rounded values** in the EchoLabs33 files (100% of
  `attn_norm` weights are f16-exact) — the converter wrote f16 precision into
  F32 tensors. No action needed for the Vulkan port; just don't expect more
  than f16 precision from weights.
- llama.cpp has no Zamba2 (hybrid) support on master — open PR #21412; the
  EchoLabs33 GGUFs depend on it. Their tensor layout is the one our loader
  already targets.

## 10. Suggested phasing

1. **P1 decode-only correctness**: `mamba2_scan.comp` + group-RMSNorm + conv1d
   bias, `backend_zamba2_vulkan.cpp` behind a `ZAMBA2_VK=1` env gate, validation
   vs CPU (32 tokens, 2 models). Reuses `dmmv_f32`, `rms_norm`, `silu_mul`,
   `flash_attn` for the hybrid path.
2. **P2 prefill**: token-folded scan + batched conv1d + GEMM prefill, state-seam
   validation.
3. **P3 perf**: wave tuning, shader fusion (in_proj is 3 disjoint GEMVs of one
   matrix — one dispatch already), quantized weight path, README + router
   integration + claim correction.

## Prior art worth reading before writing the scan shader

- llama.cpp ggml-vulkan `ssm_scan.comp` — the only existing Mamba2-style scan
  on Vulkan: subgroup-per-head, fp32, grouped (llama.cpp master, mamba2 arch)
- llama.cpp `ggml/src/ggml-cuda/ssm-scan.cu` — asserts f32-only scan; HIP
  prefill falls back to sequential scan (our plan matches)
- mamba_ssm Triton reference: `ssd_combined.py`, `ssd_chunk_state.py`,
  `ssd_chunk_scan.py` — chunked scan (chunk_size=256), for the P3 optimization
- state-spaces/mamba `mamba2.py::step` — the authoritative per-step recurrence
- in-house `ssm_delta_net.comp` — token-fold + wave64 + state-lifecycle pattern

## Sources (local)

- `src/mamba2_kernels.cpp` — CPU reference (scan step, layer forward)
- `src/mamba2_kernels.hip` — ROCm kernels (tuned GEMV/conv1d/scan/group-norm)
- `src/zamba2_engine_hip.hip` — HIP engine orchestration + state layout
- `src/gguf_zamba2_loader.cpp` — GGUF tensor names, transposes, quant support
- `src/backend_zamba2.cpp`, `src/model_router.cpp`, `src/backend_manager.cpp:1376`
- `engine/gpu/src/shaders/ssm_conv1d.comp`, `ssm_delta_net.comp` — reusable GLSL
- `engine/gpu/zinc_cpp` — C++ Vulkan dispatch API (`compute_engine.h`)
- `/home/bcloud/zinc/src/compute/forward_cuda.zig` + effort docs — delta-net
  token-fold + chunked scan precedent, prefill/decode state compatibility rules
