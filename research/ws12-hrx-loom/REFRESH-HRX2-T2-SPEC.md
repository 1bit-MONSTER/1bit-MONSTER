# T2 delta spec — Zaya arch registration on C (AMD 6319038, round-28-zaya-port)

**Branch:** `round-28-zaya-port` (T1 committed fac6f64). **Source:** A = ~/hrx-ws/hrx-v2-src zaya files.
**C structure:** class-per-arch (`llama_model_mapping` factory in llama-model.cpp; per-arch tensor
loading in src/models/<arch>.cpp with `create_tensor(tn(LLM_TENSOR_*, ...))`; tensor names =
global `LLM_TENSOR_NAMES` map in llama-arch.cpp; hparams per-arch in llama-model.cpp cases at
~296/2617-style).

## Step 0 — C field presence (measured)
Present in C llama_layer: attn_norm, wq/wk/wo, ffn_gate(+_b), ffn_gate_inp(+_b),
ffn_gate_up_exps, ffn_down_exps, ssm_conv1d(+_b). **MISSING (must add to llama_layer):**
post_attn_norm, res_scale_hs(+_b), res_scale_res(+_b), res_scale_hs_mlp(+_b),
res_scale_res_mlp(+_b), cca_conv_grp(+_b), cca_k_scale, cca_val_proj1, cca_val_proj2,
zaya_router_mlp2(+_b), zaya_router_mlp4, zaya_router_biases, zaya_router_eda_scale.
**MISSING in llama_model:** input_hidden_states_scale(+_bias), zaya_res_scale_hs/final pairs,
zaya_input_hs_scale/bias.
(For field semantics/names copy A `src/llama-model.h` lines 313-364 + 550-556.)

## Step 1 — registration skeleton (compile gate)
1. `src/llama-arch.h`: `LLM_ARCH_ZAYA` before `LLM_ARCH_UNKNOWN` (line ~152).
2. `src/llama-arch.cpp`: `{ LLM_ARCH_ZAYA, "zaya" }` in LLM_ARCH_NAMES (~line 135).
3. New LLM_TENSOR_* entries (append before LLM_TENSOR_COUNT): ZAYA_ROUTER_MLP2,
   ZAYA_ROUTER_MLP2_B, ZAYA_ROUTER_MLP4, ZAYA_ROUTER_BIASES, ZAYA_ROUTER_EDA_SCALE,
   CCA_CONV_GRP, CCA_CONV_GRP_B, CCA_K_SCALE, CCA_VAL_PROJ1, CCA_VAL_PROJ2,
   POST_ATTN_NORM, RES_SCALE_HS, RES_SCALE_HS_B, RES_SCALE_RES, RES_SCALE_RES_B,
   RES_SCALE_HS_MLP, RES_SCALE_HS_MLP_B, RES_SCALE_RES_MLP, RES_SCALE_RES_MLP_B,
   INPUT_HIDDEN_STATES_SCALE, INPUT_HIDDEN_STATES_BIAS, RES_SCALE_HS_FINAL,
   RES_SCALE_RES_FINAL (check A's exact enum names first).
4. `LLM_TENSOR_NAMES` map entries with A's gguf format strings:
   "blk.%d.cca_val_proj1", "blk.%d.zaya_router_mlp2", "blk.%d.zaya_router_eda",
   "blk.%d.res_scale_hs", "blk.%d.ssm_convNd"→existing SSM_CONV1D name etc (read A's
   llama-arch.cpp NAMES entries for zaya).
5. `llama-model.cpp`: factory `case LLM_ARCH_ZAYA: return new llama_model_zaya(params);`
   + declare `class llama_model_zaya : public llama_model` in llama-model.h (pattern =
   nearest arch class; methods: load hparams, load tensors, graph entry via arch hook).
6. Arch-flag helpers: A had zaya listed in `llm_arch_is_recurrent`-style cases (A arch.cpp
   2950 area) → mirror in C's equivalents (`llm_arch_is_recurrent` etc.).

## Step 2 — model class impl (new `src/models/zaya.cpp`), adapted from A
1. hparams: read zaya.* KVs (A pattern) incl ssm.conv_kernel, expert top-1, n_embd_s.
2. Tensor load: translate A's llama-model.cpp LLM_ARCH_ZAYA load case (gguf tensor name →
   layer/model field): token_embd, output_norm, input_hidden_states_scale/bias, per layer:
   attn_norm, post_attn_norm, res_scale_* ×6(+b), wq/wk/wo (attn_q/attn_k/attn_output),
   cca_val_proj1/2 (v current/delayed), ssm_convNd→ssm_conv1d(+b), cca_conv_grp(+b),
   cca_k_scale, qk_norm temp (skip? A had qk_norm.temp BF16 [2] → field), MoE/odd-layer
   router stack: ffn_gate(+b), ffn_gate_inp(+b), zaya_router_mlp2(+b), zaya_router_mlp4,
   zaya_router_biases, zaya_router_eda_scale, ffn_norm, ffn_gate_up_exps, ffn_down_exps,
   post_mlp residual scales. TYPE42 accept: tensors of GGML_TYPE_Q4NX → HRX buffer (T3 gate;
   for T2 land: allow CPU-host load w/ type traits; device claim added in T3).
3. Graph: port A src/models/zaya.cpp llm_build_zaya into C's class graph hook (uses
   llama_memory_recurrent for conv state; flash-attn disabled path; CCA attention built as
   ops; MoE via ffn_gate_up_exps stacked + top-1 router).
4. Recurrent/state wiring per C's recurrent-model pattern (find C arch using
   llama_memory_recurrent e.g. QWEN3NEXT/KIMI as the template for state buffers + caches).

## Gate
G1: arch recognized ("zaya" loads past arch check). G2: hparams parse OK. G3: tensors create
(load completes, type42 on host). G4: decode graph builds + runs single-seq (numeric dprobe:
NaN=0) — CPU-path decode first acceptable (slow), HRX acceleration = T3.

## Notes / risk
- A's GGUF naming mixes engine-era + ssm-era conventions; use A's loader mapping as ground
  truth for zaya-q4nx.gguf (cca_val_projN = v current/delayed; ssm_convNd = conv_qk 2-tap).
- 17-slot router (16 experts + skip) = zaya_router_biases [17] etc. — MoE graph must honor skip.
- Model alternates? NO — HF ZayaDecoderLayer has BOTH CCA-attn + MoE per layer (A commit
  2cf3109b "BOTH CCA attention and MoE in every layer") — graph runs both every layer.
- Do NOT attempt device offload of Q4NX until T3; G3-G4 on CPU only.

---

# T3 addendum — type42-on-HRX + decode kernels (hook surface, measured)

**C's MUL_MAT dispatch** (`dispatch_registration/common/dispatch-mul-mat-weight-format.h`):
weight formats = Q4K/Q6K/Q8_0/Q8_1/F16/BF16/F32 via `common_mul_mat_format_for_type()`. Q4NX
hooks = new enum + `case GGML_TYPE_Q4NX` (format: 5120-B tile, blck 8192, 32×256 BF16-scaled
int4 rows — no upstream analog). Dispatch entry points to extend: dispatch-mul-mat.cpp /
dispatch-mul-mat-id.cpp / dispatch-common (block bytes, k, row geometry per format).
Flash-attn/gated/rope dispatches unaffected (CPU/other path).

**Kernel side**: C compiles a kernel-corpus at build time (`.loom` sources + manifests, merge
mode #93). Its qwen corpus has NO generic mul_mat kernels visible in `kernels/qwen/` — the mm
family ships via `loom-libs` (per earlier corpus dir list). To port the Q4NX decode path you
need a q4nx-tile MUL_MAT loom kernel + manifest entry, mirroring C's existing Q4K/BF16 mm
kernels.
**A's sources to port from** (`~/hrx-ws/hrx-v2-src/ggml/src/ggml-hrx2/`, 44 .loom kernels):
`mul_mat_q4_k_swiglu_f32.loom`, `mul_mat_f16_f32_batched.loom`, `mul_mat_id_q4_k_f32.loom`,
`mul_mat_q8_0_f32.loom` … — but A's catalog/route/JIT system differs from C's corpus/registry;
the loom kernel math transfers, the dispatch glue does not (rewrite against C's registry).

**Sequencing**: T3 cannot be end-to-end-validated until T2 lands (no zaya-q4nx load → no
matmul to route). Recommended: T2 Step1-2 (CPU decode gate G4) → T3 weight-format + kernel +
registry → HRX device decode → multi-seq. A's validated q4nx decode semantics (dequant =
q*scale+zp with clamps, validated bit-exact vs C ref this session) are the numeric oracle.

---

# T3 final framing (after G4)

The q4nx weights are **tile-framed** (`[8192, 256]` in-file = 256 tiles; block = 8192
logical elements / 5120 B) while the graph expects logical `{k, n}` (e.g. attn_q
`{2048, 1024}`). Flat `dequantize_row_q4nx` output ≠ f32-file logical order for the
quantized set (measured earlier — permutation differs per tensor family). A solved this
with a custom op (`ggml_mul_mat_q4nx` routing `a->type == GGML_TYPE_Q4NX`, tile-major
semantics) + HRX2 loom kernels (`mul_mat_q4nx_*`, r16x8t family) whose dispatch knows the
tile↔logical mapping. So **T3 = port A's tile-mapping MUL_MAT semantics**, one of:
1. CPU: ggml-cpu MUL_MAT case for GGML_TYPE_Q4NX that dequantizes per tile into the
   logical layout (needs A's tile→logical index map, derivable from A's kernel or a
   one-time empirical fit vs zaya-f32.gguf on the 280 tensors — the f32 twin is the oracle).
2. HRX: C's `CommonMulMatWeightFormat` Q4NX case + dispatch glue + a q4nx MUL_MAT loom
   kernel in C's kernel-corpus (kernel math from A's 44 .loom sources; dispatch rewritten
   against C's registry).
Numeric gates: decode zaya-q4nx-c43.gguf clean (dprobe: NaN=0, argmax sane vs f32-gguf
reference), then multi-seq npl 1-8 on the refreshed base (stale was segfaulting there).
