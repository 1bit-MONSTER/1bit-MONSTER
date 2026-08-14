---
name: model-bringup
description: Add a new HF model family to the 1bit-systems MONSTER engine (arch mapping -> quirk code -> fixture -> honest validation gate). Use when the user asks to bring up, validate, or add a new model family/architecture to the decode-loop engine (backend_generic.cpp + safetensors/GGUF loaders + Testing/bringup_runner.sh).
---

# Model Bring-Up (MONSTER family add)

The MONSTER engine (`src/backend_generic.cpp` + the bespoke backends) loads HF
checkpoints natively (safetensors or GGUF). A "family" = one `rcpp_arch_t`
token covering every HF `architectures` string of that class. Adding a family
is a guided process — this skill is that guide. It was written after 8
families were validated in one session (gptoss, step1, deepseek-mla, bloom,
whisper + the Phase 3 runner), including three dead-code-backend rewrites.

## The one-sentence process

**Add a family = manifest entry + arch mapping + quirk code + a gate command.
No runner surgery.**

## Step 1 — Recon (before writing code)

1. **The reference**: find the authoritative `modeling_<arch>.py` (HF repo
   raw, or the installed transformers 5.14 copy). Read the attention forward
   and the config keys. This is the ground truth — not llama.cpp, not your
   first guess.
2. **The checkpoint**: pick the SMALLEST real checkpoint of the class (the
   census often lists a tiny pretrain run). Download to `/tmp/onebit-e2e/<fam>/`.
3. **The tensor names**: dump the safetensors keys (`safe_open` + print) or
   the GGUF names (a small `GgufReader` probe). The engine's loader matches
   EXACT names — the HF repo sometimes stores them flat (`h.0.*`) vs the
   `transformer.h.0.*` prefix in the modeling code.
4. **The oracle**: decide BEFORE coding —
   - torch (transformers 5.14 has the arch): `Testing/e2e_torch_oracle_<fam>.py`
     (argmax chain + N generated tokens + full logits).
   - llama.cpp (`llama-completion -no-cnv --temp 0` for text; the
     `--save-all-logits` perplexity file for per-position logits; llama.cpp
     archs live in `src/models/`).
   - a numpy port of the modeling code (when torch 5.x dropped the arch or
     the model is memory-infeasible: gptoss 20B, deepseek).

## Step 2 — The mapping + config decode

- `include/rocm_cpp/bitnet_model.h`: `RCPP_ARCH_<FAM>` token + the
  `rcpp_arch_from_string` entry (suffix-stripped arch string, e.g. `step1moe`
  from `Step1MoEForCausalLM`).
- `src/safetensors_reader.cpp`: the config keys the family uses
  (`n_embed`/`n_head`/`n_layer`/`layer_norm_epsilon` for bloom; `experts_per_token`
  for gptoss; `num_attention_groups` for step1). The reader's fallbacks only
  cover common keys — new families need their own.
- `Testing/arch_mapping_selfcheck.cpp` + `Testing/rotation_table_selfcheck.cpp`:
  one `check(...)` each. `run_all.sh` must stay 7/7.

## Step 3 — The quirk code (the real work)

Per-family quirks land in `backend_generic.cpp`'s per-arch branch or the
bespoke backend. The recurring gotchas (each cost a debugging saga):

1. **GGUF 2D weights are `[out][in]`** (element (i,j) at i*K+j, K=in). The
   generic loader's `matmul` convention. Verify against llama.cpp's
   `src/models/<arch>.cpp` `create_tensor` shapes when unsure.
2. **Fused qkv is usually HEAD-INTERLEAVED** (`[h0(q,k,v), h1(q,k,v), ...]` —
   gpt-neox, bloom, gptoss) — check `modeling_*.py`'s `_reshape`/`view`
   before assuming a `[q|k|v]` row split. Bloom's row split was the 4.6x
   hidden-state error.
3. **The exps/large-tensor layout**: GGUF 3D expert tensors are `[A, B, NE]`
   with shape[0] (A) INNERMOST and blocks of 32 along A; expert OUTERMOST.
   A circular verification trap: your loader and the reader's `get_tensor_f32`
   share the same wrong flat interpretation — a standalone dequant + a
   non-circular numpy full-model comparison catches it.
4. **Rope pairing**: llama.cpp `LLAMA_ROPE_TYPE_NORM` = ADJACENT pairs
   (2i, 2i+1); NEOX = chunk (i, i+half). DeepSeek2 and Step1/Bloom are in
   the NORM list; HF's interleave-view makes the math equivalent. The
   `ggml_rope_ext` mode is set per-arch in `llama_model_rope_type`.
5. **ALiBi**: two variants — Step1 is `-slope*sqrt(distance)`, Bloom is
   `-slope*distance` (linear). Same slope table (2^(-8(h+1)/n) then
   2^(-4(2h+1)/n)). `slope*pos_k` is softmax-equivalent to `-slope*(q-k)`.
6. **The residual is sacred**: pre-norm transformers keep `x` as the residual
   and norm into a SEPARATE buffer. The whisper/deepseek dead-code backends
   clobbered `x` with the normed input — repeated-blank or exploded outputs.
7. **Buffer sizes**: attention helpers that write N rows need N-sized outputs
   (the whisper `self_attn` wrote N rows into a single-token buffer — ASan).
8. **The GGUF tensor-name mapping**: modern conversions use
   `model.encoder.layers.N.self_attn.q_proj` etc. — old backends expect
   legacy names. Watch for substring collisions: `attn_ln.` matches inside
   `cross_attn_ln.` — order the replacements cross-first.
9. **Embedding norms**: some families norm the token embedding
   (`word_embeddings_layernorm` for bloom) — the engine has `cfg.embed_ln`.
10. **Memory**: Q8→f16 for big MoE experts (29GB vs 63GB f32) is the
    established tradeoff; keep blocks packed and dequant per-selected-expert
    for the >50GB cases (gptoss).

## Step 4 — The honest validation (NO SECRETS)

The manifest `Testing/models_manifest.json` tiers are the ledger:

| Tier | Meaning |
|------|---------|
| `full` | 20/20 generated tokens identical to torch (f32) |
| `full-llamacpp` | 20/20 vs llama.cpp Q8 (near-ties documented) |
| `numpy-exact-20of20` | 20/20 vs a numpy port of the authoritative modeling code |
| `numpy-exact[-near-tie/-degenerate/-oracle-wrong]` | engine ≡ numpy top-8 EXACT but no clean 20/20 (near-ties / flat logits / llama.cpp oracle wrong) |
| `documented-limitation` | refuses loudly or only partially works (openelm, whisper pre-port) |

Never inflate a tier. The ledger explicitly calls out: the census coverage
is a SAMPLED upper bound; near-tie flips from f16 storage are real; a
family whose argmaxes diverge on weak-text positions is NOT "20/20".

## Step 5 — The gate

Add a `gate` field to the manifest entry: the exact command that exits 0 on
success (a 20-token sequence match, a transcript match, a chain signature).
`Testing/bringup_runner.sh` dispatches it automatically — a new family is
manifest + gate, no runner edits. `run_all.sh` stays 7/7.

## Step 6 — Docs

- `docs/plans/monster-500-build.md`: a Phase 2 entry (quirks found, honest
  caveats, the gate command).
- `docs/wiki/models.md`: the coverage table + remaining classes.
- The wiki: one observation with the specific bugs found.

## The dead-code-backend warning

Bespoke backends (`src/deepseek.cpp`, `src/whisper.cpp`, `src/backend_mamba1.cpp`)
can be UNWIRED and UNVALIDATED — written for a different GGUF variant, never
run. Before trusting one: compile a harness, run it, and compare against the
reference at the FIRST step. Expect the full bug list above. The port pass
is the real work; budget like a new family.
