# feat(engine): baretorch — cs_lrad chunked-state linear-recurrent — port map (#1907)

**Status:** M1 port map + M2/M3 CPU engine landed (2026-09-05, commits 6c47e693 +
6b48a73c on `feat/baretorch-cslrad-port`, PR #2123). The baretorch CPU engine
(`src/baretorch_engine.cpp`, `baretorch_cpu`) now runs the real 500M-Base:
engine logits vs torch chunked-prefill reference = max abs err 2.7e-4 / corr 1.0
over 96 positions x 49152 vocab (Base, F32) and max err 1.5e-4 (SFT, BF16 —
reader converts BF16→F32); discovery → route → engine init → generate
verified end-to-end. Remaining: M4 (GGUF/quant) + M5 (GPU), per milestones below.
Census coverage stays 100%. Issue #1907. Branch `feat/baretorch-cslrad-port`.

## Goal

`backend_generic` (safetensors + GGUF) currently **refuses** every `baretorch`
model at load with a clean diagnostic (`RCPP_ARCH_BARETORCH = 990` registry
token, landed in #1974 / 708bcb46). That refusal is deliberate and correct:
cs_lrad is a genuinely new hybrid — 18× chunked-state linear-recurrent
layers interleaved 3:1 with 6× standard GQA transformer layers — and aliasing
it to llama/qwen/muse/lfm2 would silently mis-execute (the #1624/#1627
loud-rejection precedent). Goal: run `model-rampage/BareTorch-500M-{Base,SFT}`
on the engine behind the `RCPP_ARCH_BARETORCH` gate with a real cs_lrad
chunked-state forward.

Census state (checked 2026-09-05 on `main`): the registry token already
counts baretorch as COVERED —
`total=405884 with_arch=321611 registry_covered=321611 (100.00%)` via
`python3 Testing/census_coverage.py`. So the census gate is satisfied; the
remaining gap is **engine execution**, not coverage.

## Ground truth collected (2026-09-05)

- **Arch gate exists:** `RCPP_ARCH_BARETORCH = 990`
  (`include/rocm_cpp/bitnet_model.h`); `rcpp_arch_from_string` maps
  `baretorch` and `cs_lrad` to it; the generic loader refuses cleanly in
  `backend_generic.cpp` (load_gguf + safetensors paths) with a precise
  "BARETORCH — cs_lrad registry token, engine support XL (issue #1907)"
  message. `model_router` has no baretorch route yet (falls through to the
  generic lane → refusal).
- **Target checkpoints** (real, live on HF 2026-09-05):
  `model-rampage/BareTorch-500M-Base` (F32 weights) and
  `model-rampage/BareTorch-500M-SFT` (config declares bf16). Both ship the
  **full reference modeling source inside the HF repo** under `baretorch/`
  (`modeling/cs_lrad.py`, `modeling/transformer.py`,
  `integration/modeling_baretorch.py`, `integration/configuration_baretorch.py`)
  — this is the ground-truth math, no guessing needed.
- **Config decode** (`config.json`, Base):

  | key | value | engine meaning |
  |-----|-------|----------------|
  | architectures | `["BareTorchForCausalLM"]` | discovery class |
  | model_type | `baretorch` | → `RCPP_ARCH_BARETORCH` |
  | d_model | 1152 | hidden |
  | num_layers | 24 | layers |
  | num_heads | 16 | query heads |
  | num_kv_heads | 4 | KV heads (transformer layers only) |
  | head_dim | 72 (= 1152/16) | per-head dim |
  | d_ff | 4032 (= 3.5 × d_model) | MLP hidden |
  | chunk_size (C) | 32 | cs_lrad chunk length |
  | rank (r) | 8 | cs_lrad low-rank state |
  | vocab_size | 49152 | SmolLM2 tokenizer vocab |
  | max_seq_len | 2048 | context (README claims 32768) |
  | layer_types | 24 entries: `[cs_lrad×3, transformer]` × 6 | **per-layer kind schedule** |
  | dtype | float32 (Base) / bfloat16 (SFT) | weights |
  | rms norm eps | 1e-6 (reference code default) | norms |
  | dropout / use_cache | 0.0 / false | inference |

  **Layer schedule (verified against real tensor names):** transformer layers
  are exactly **3, 7, 11, 15, 19, 23** (indices ≡ 3 mod 4); all others
  (18×) are cs_lrad.

- **Tensor inventory (345 tensors, captured from the real
  `model.safetensors` header, 2026-09-05):** all F32, `nn.Linear` orientation
  `[out, in]`, per-layer names `model.layers.N.*`, globals
  `model.token_embedding.weight`, `model.final_norm.weight`, `lm_head.weight`.
  `lm_head` is **NOT tied** to the embedding (verified unequal — both
  [49152,1152] are stored separately).

  **cs_lrad layer (16 tensors — layers 0,1,2,4,5,6,8,…21,22):**

  | tensor | shape |
  |--------|-------|
  | ln1.weight / ln2.weight | [1152] (RMSNorm) |
  | attn.W_q.weight / W_k.weight / W_v.weight | [1152,1152] (all 16 heads — **no GQA** here) |
  | attn.W_u.weight / W_r.weight | [128,1152] (= 16 heads × rank 8) |
  | attn.W_gate.weight + bias | [16,1152] + [16] |
  | attn.W_beta_gate.weight + bias | [16,1152] + [16] |
  | attn.W_swish_gate.weight | [1152,1152] |
  | attn.W_out.weight | [1152,1152] |
  | mlp.w1.weight / w2.weight | [4032,1152] |
  | mlp.w3.weight | [1152,4032] |

  **transformer layer (9 tensors — layers 3,7,11,15,19,23):**

  | tensor | shape |
  |--------|-------|
  | ln1.weight / ln2.weight | [1152] (RMSNorm) |
  | attn.W_q.weight | [1152,1152] (16 heads × 72) |
  | attn.W_k.weight / W_v.weight | [288,1152] (4 KV heads × 72) |
  | attn.W_out.weight | [1152,1152] |
  | mlp.w1.weight / w2.weight | [4032,1152] |
  | mlp.w3.weight | [1152,4032] |

  Total: 18×16 + 6×9 + 3 = 345. ~593M params total (incl. untied lm_head +
  embedding).

- **Reference math (from the shipped `baretorch/` source):**

  *cs_lrad layer (LRADDecoderBlock)* — Pre-norm RMSNorm(1e-6) → cs_lrad
  attention → residual → RMSNorm → GatedMLP (silu(w1)·w2 → w3) → residual.
  The cs_lrad attention (`LowRankAssociativeDeltaEngine`) has **two code
  paths**:
  - **Chunked forward** (L > 1, no past): silu(W_q)·silu(W_k) products over
    the chunk with decay links `M_links = exp(clamp(Λ_i − Λ_j, ≤0))` where
    `Λ = cumsum(log gate)` inside each 32-token chunk; `gate =
    clamp(sigmoid(W_gate x), 1e-3, 0.999)`, `β = sigmoid(W_beta_gate x)`;
    `Y_local = chunked QK^T·M_links·V`; cross-chunk decay scan
    (`M_chunks` over chunk-log-decays) accumulates the rank-r state
    `S_hist = Σ M_chunks · (U·β)^T V`; `Y_global = R·exp(Λ)·S_hist`;
    output gated by `silu(W_swish_gate x)` before `W_out`. Per-layer cache
    = the final low-rank state `S` of shape **[B, 16, r=8, 72]**.
  - **step_inference** (L == 1 or past present): delta-rule update
    `S = gate·S + (U·β)^T V`, output `Q(K^T V) + R·S` — an O(1)-state decode.

  *transformer layer (TransformerDecoderBlock)* — standard pre-norm GQA:
  16 Q / 4 KV heads × 72, **full-head-dim RoPE** (RotaryEmbedding dim =
  head_dim = 72, base 10000, standard rotate-half; no partial rotary), RMSNorm
  (1e-6), GatedMLP 3.5×. KV cache per layer: **[B, 4, L, 72] × 2**.

  **Hybrid cache semantics:** 18 cs_lrad layers carry the [B,16,8,72] state;
  6 transformer layers carry KV. Both are per-layer — the engine's per-layer
  loop must dispatch on `layer_types[i]`, exactly like the CPU reference in
  `src/qwen3next_engine.cpp` dispatches `linear_attention | full_attention`.

## Critical finding — chunked prefill ≠ step_inference in the published reference

Validated numerically on the real 500M-Base weights on strixhalo
(torch 2.13 CPU, float64, 2026-09-05):

1. **The two code paths in the shipped reference disagree.** Running the
   SAME tokens through (a) chunked prefill (`forward`) vs (b) token-by-token
   `step_inference` with the cache gives max logits err ~14–27 even in
   **float64**, diverging **from position 0** of a fresh single chunk
   (pos0 err 13.87, pos31 err 23.99; per-layer state handoff differs by
   0.78 on the first cs_lrad layer's S). This is a structural difference in
   the published reference (the delta-rule step path is not a numerically
   faithful decode of the chunked path) — **not** float noise.
2. **The served reference never uses step_inference.** The released config
   sets `use_cache: false`, and an instrumented `generate()` run shows the
   model is re-prefilled over the full growing context every step (12 calls,
   input lens 160 → 171, `past_key_values` never passed). So what the model
   card/pipeline actually serves is **chunked prefill of the whole context
   per token** — i.e. per-position logits == chunked math.
3. **Chunked prefill is length-stable (causal).** The same token position
   yields the same logits whether prefilled at total length 64, 96 or 128
   (pos-40 err ≤ 4e-6 across 64/96/128). So an engine whose decode
   reproduces the chunked-math logits per position will match the served
   reference exactly; an engine that ports `step_inference` as its decode
   kernel will NOT match (that path is inconsistent with the served model).

**Consequence for the engine (M3+):** the engine must implement the
**chunked** forward as the correctness target (per-position logits equal the
full-context chunked prefill), with an incremental decode that carries the
per-layer rank-8 state and re-computes the current chunk exactly — the
canonical chunked-state decode. Do **not** port the reference's
`step_inference` math as the decode contract without first fixing/reporting
the upstream inconsistency (candidate: the delta-rule step path drops the
within-chunk decay-link QK attention that chunked prefill computes).

## Milestones

- **M1 (this PR):** port map + captured schema + the prefill/decode
  consistency finding above. No engine behavior change (refusal stays).
- **M2 — structural loader** (generic safetensors path):
  1. Gate on `cfg.arch == RCPP_ARCH_BARETORCH` before the generic refusal.
  2. Validate the real structure against this captured schema (345 tensors;
     16 vs 9 per layer by `layer_types[i]`; shapes above) via the existing
     SafetensorsWeightReader index — cheap, no data read.
  3. Decode `config.json` `layer_types` into a per-layer kind array on
     ModelConfig (`d_model/num_heads/num_kv_heads/chunk_size/rank/…`).
  4. Refuse decode loudly until M3 lands (no silent garbage) — same pattern
     as #2121's M2 for qwen35moe.
- **M3 — cs_lrad chunked-state forward (CPU/generic):** port the chunked
  math (per-chunk decay links + cross-chunk rank-8 state scan + swish gate)
  and the GQA transformer layers with per-layer `layer_types` dispatch;
  hybrid cache (S per cs_lrad layer, KV per transformer layer); decode =
  chunked-consistent incremental.
- **M4 — GGUF/quantized path:** GGUF tensor mapping (see mapping sketch
  below) + quant dtypes once the F32 path is exact. (No public GGUF for
  baretorch exists yet — model ships safetensors only; 500M F32 = 2.37 GB,
  SFT config bf16.)
- **M5 — validation on strixhalo:** decode-vs-torch-reference corr gate over
  the real 500M (chunked per-position logits as reference; tokens 1000/1001
  methodology; per-layer bisection if needed). Golden artifacts from the
  reference run (golden.pt: ids + logits) already staged in
  `/home/bcloud/baretorch-ref/`.

## Acceptance gate (M5)

- Logits corr ≥ 0.99 vs the torch chunked-prefill reference over ≥ 100 tokens
  (multi-prompt) on the real 500M-Base/SFT; byte/near-bit parity for the
  deterministic transformer layers.
- Existing models: zero behavior change (refusal stays until M3; census
  coverage stays 100%).

## GGUF mapping sketch (for M4)

llama.cpp-style naming for this hybrid (writer-side proposal; must be
confirmed against the actual converter before M4 code, per the #1831
schema-hazard lesson):

- globals `token_embd.weight` (49152×1152), `output.weight` (untied, 49152×1152), `output_norm.weight` (1152)
- cs_lrad kind (per blk.N): `attn_norm.weight` (1152), `attn_q/k/v.weight`
  (1152² / 1152² / 1152²), `attn_u.weight`/`attn_r.weight` (128×1152),
  `attn_gate.weight`+`attn_gate.bias`, `attn_beta_gate.weight`+bias,
  `attn_swish_gate.weight` (1152²), `attn_output.weight` (1152²),
  `post_attn_norm.weight`, `ffn_gate/up/down.weight`
- transformer kind (layers ≡ 3 mod 4): `attn_q.weight` (1152²),
  `attn_k/v.weight` (288×1152), `attn_output.weight`, norms + ffn as above
- per-layer `layer_types` must be encoded in GGUF metadata (`*.layer_types` /
  custom key) since 3:1 is a property of the schedule, not of any single
  tensor.

## Open questions for M3 (math, not M1)

- Confirm the intended **decode semantics** with the upstream authors: the
  published `step_inference` does not reproduce the served chunked-prefill
  logits (float64, from position 0). Engine decode target = chunked logits;
  file an upstream note if the delta-rule step path is claimed to be
  equivalent.
- RoPE is full-head-dim (72) on transformer layers only; cs_lrad layers have
  no RoPE and no GQA. Verify against the reference at M3 (rope base 10000).
- RMSNorm eps 1e-6 and weight+1 conventions: baretorch stores plain RMSNorm
  weights (verified non-ones); no layer-norm/1P quirks.

## References

- Issue #1907 (this), #1974 (registry token + refusal, merged 708bcb46),
  #1906 (first watcher-run aliases), #2121 (sibling XL port map pattern:
  qwen35moe-on-HIP, M1 doc + M2 structural loader), #1624/#1627 (loud
  rejection precedent), #1831 (qwen35moe XL parent).
- Reference source shipped in the model repo (`baretorch/` on HF) —
  captured at `/home/bcloud/baretorch-ref/` on strixhalo with the real
  weights + golden artifacts.
