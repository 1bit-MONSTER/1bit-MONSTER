# PLAN: MONSTER 500-Model Build — execution plan

**Date:** 2026-08-14 · **Status:** ready to execute
**Vision/strategy:** `docs/plans/monster-500-models.md` · **Master log:** `docs/research/onebit-modular-research.md`
**Strategy:** `docs/plans/monster-500-models.md` · **Master log:** `docs/research/onebit-modular-research.md`

## The math (how 500 is reached)

Models are data. 500 models = ~50 HF architecture classes mapping onto validated layouts.
Current: **24 arch tokens, 48 arch checks, 35 rotation checks, 14 families validated** — 13 torch-full 20/20 (llama, qwen2, qwen3, gemma, granite-MoE, mistral, phi, olmo, gpt2, falcon, opt, gptj, gptneo), exaone Q8-oracle 20/20, 4 numpy-exact (internlm2, minicpm, gptneox, codegen), **gptoss numpy-exact 20/20** (2026-08-14, the memory-blocked family now runs via packed-MXFP4 per-expert dequant — see Phase 2 #7).

Unlock table (each ✅ family adds every HF checkpoint of that class):

| # | Family | HF arch strings | Unlocks (approx) | Engine work |
|---|--------|----------------|------------------|-------------|
| 1 | llama-layout | llama, openelm, nemotron, minicpm, baichuan, exaone, solar, internlm, xverse, starcoder, stablelm, mosaic, mpt, dbrx, jamba, command-r | ~150+ checkpoints | ✅ done (validated) |
| 2 | qwen2 | qwen2, qwen, deepseek (V1) | ~60 | ✅ validated (QKV bias done) |
| 3 | qwen3 | qwen3 (+QK-norm) | ~50 | 🔲 validate generic path (QK-norm code exists) |
| 4 | mistral | mistral, pixtral | ~40 | ✅ validated |
| 5 | gemma | gemma2/3/4, granite, ovis | ~30 | ✅ gemma3 validated; granite MoE validated |
| 6 | phi | phi, phi3, phi4 (fused qkv/gate_up) | ~25 | ✅ phi-3 validated; phi-4 needs download |
| 7 | olmo | olmo, olmo2, olmoe, molmo | ~15 | 🔲 LayerNorm + no-RoPE (new paths) |
| 8 | falcon | falcon, falcon3 | ~15 | 🔲 parallel attn + MQA |
| 9 | gpt2 | gpt2 | ~30 (huge legacy tail) | 🔲 custom tensor map (no GQA, tied embed) |
| 10 | zamba/mamba | zamba2, zamba, mamba | ~10 | bespoke backends exist — keep, validate |
| 11 | deepseek MLA | deepseek2/3, deepseek_v4 | ~15 | bespoke (MLA) — keep, validate |
| 12 | kimi | kimi_k3, moonlight, kimi_vl | ~5 | bespoke — keep |
| 13 | whisper | whisper | ~5 | bespoke STT — keep |
| 14 | VLMs | qwen2vl, qwen3vl, smolvlm, llava | ~20 | bespoke VLM paths exist |

→ **Phase 1 (breadth) + 7 new validated families (rows 3,7,8,9 + validation of 10–13) ≈ 500.**

## Phase 1 — Arch-string breadth sweep (cheap, 1–2 days)

Enumerate every HF `architectures` string on the hub for dense causal-LM classes; add one-line
mappings in `rcpp_arch_from_string()` (`include/rocm_cpp/bitnet_model.h`) + an assertion each in
`Testing/arch_mapping_selfcheck.cpp`. Unknown strings keep failing loud (`RCPP_ARCH_UNKNOWN`).

**DONE 2026-08-14:** authoritative list extracted from llama.cpp `convert_hf_to_gguf/conversion/`
registry (~230 HF class names across 86 family files → `docs/plans/monster-500-build.md`). Batch 1
landed: smollm3, apertus, cohere, gptbigcode, internlm3 (→LLAMA); mixtral (→MISTRAL); qwen2moe
(→QWEN2); qwen3moe (→QWEN3); deepseekv2/v3 (→DEEPSEEK); deepseekv4 (→DEEPSEEK_V4). Arch self-check
30→41, rotation 17→28, `run_all.sh` 10/10 green.

- [ ] Source the authoritative arch-string list (HF `transformers/models/*/modeling_*.py` class names + GGUF `general.architecture` values in llama.cpp `convert.py`)
- [ ] Batch: grok (MQA — likely QWEN2-layout), smollm (llama), nemotron4, granite small variants, t5/bloom/bart (encoder-decoder → **DECISION: out of scope** or new path), gpt_neox, codegen, yoso, gptj, bloom
- [ ] Extend `Testing/arch_mapping_selfcheck.cpp`; keep `run_all.sh` 10/10 green
- [ ] `Testing/rotation_table_selfcheck.cpp`: every new mapping must set rotate/neox correctly

**Gate:** `./Testing/run_all.sh` → 10/10.

## Phase 2 — Family bring-up deck (the real work, ~1–2 days/family)

Per family, the pilot loop (see research §8–30): config.json quirks → loader + engine changes →
fixture → real-checkpoint e2e vs torch (20-token generation, full-logits compare).

Order by unlock size:

1. **qwen3** (biggest unlock, QK-norm code already in loader/engine) — download Qwen3-0.6B, e2e vs torch. Expect: QK-norm weight naming, rope_theta 1e6 default.
   **DONE 2026-08-14: 20/20 generated tokens identical to torch** (Qwen/Qwen3-0.6B from HF cache; config qk_norm=none for this checkpoint, rope_theta 1e6 handled). Wired into `run_all.sh` + `run_gen.sh` + `bringup_runner.sh`.
2. **olmo** — first no-RoPE family: LayerNorm (vs RMSNorm), learned positional embeddings. Loader: `model.norm` naming, no rope rotation. Engine: `neox_rope_=false` + no rope at all.
   **DONE 2026-08-14: 20/20 generated tokens identical to torch** (allenai/OLMo-1B-0724-hf). Actual quirks found (differ from doc): OLMo-1B-0724 uses **RoPE theta 10000** (computed, not learned positions — the 'no RoPE' doc line is the pre-0724 family), **LayerNorm with NO affine params** (OlmoLayerNorm: F.layer_norm(x, None, None, eps=1e-5) — no norm weights in safetensors NOR in llama.cpp GGUF), and **clip_qkv=8.0** clamp. Engine: new `layernorm()` path (mean/var centered, biased var, no weight) + clip_qkv clamp before rope, keyed on cfg.norm_is_layernorm/cfg.clip_qkv (ModelConfig). Loader: norm-weight requirement relaxed for OLMO on both safetensors + GGUF paths; OLMo GGUF q/k ARE pre-rotated (olmo.py calls LlamaModel.permute) so unrotate applies. GGUF-Q8 path 18/20 vs torch (2 Q8 near-tie flips, expected). Two real gotchas: (a) load_gguf copies hdr eps over loader-set eps at the END — OLMO flag block must run after; (b) `forward(int)` rejects token<0 — generation feeds argmax back, not -1.
3. **falcon** — parallel attn+FFN layout (q/k/v + dense before FFN), MQA (1 kv head). Tensor map: `transformer.h.N.self_attention.{query_key_value,dense}` + `mlp.dense_h_to_4h/4h_to_h`.
   **DONE 2026-08-14: 20/20 generated tokens identical to torch** (tiiuae/falcon-7b, 7.2GB). Actual quirks: nn.LayerNorm (weight+bias, eps 1e-5 — NOT RMSNorm), **parallel attn+FFN on the SAME norm output** (cfg.parallel_attn_ffn — engine saves the normed input and skips the pre-FFN re-norm; both outputs add to residual), fused MQA `query_key_value` [H+2·HD, H] (q rows [0,H), k/v rows [H,H+2·HD) — 1 kv head, head_dim 64, 71 query heads), **erf-gelu** (nn.GELU default — new gelu_erf, distinct from gelu_new), no-gate FFN (reuses gpt2 path), `multi_query:true` → kv=1 in safetensors_reader (new json_find_bool), no intermediate_size → 4×hidden FF fallback generalized, `transformer.`-prefixed names + `word_embeddings`/`ln_f.bias`. alibi=false (no alibi engine support needed). Oracle fixture needed auto_map stripped from config for the torch oracle (falcon IS built into transformers 5.14). Gotchas: FF dim heuristic (8192 wrong → 18176 via 4× fallback).
4. **gpt2** — custom tensor map (no GQA), tied embed, own norm naming (`ln_1/ln_2/ln_f`). Highest legacy-model unlock (~30).
   **DONE 2026-08-14: 20/20 generated tokens identical to torch** (openai-community/gpt2). The big one: **GPT-2 Conv1D stores weights [in,out] — every projection transposed at load**, and the fused `c_attn` [H,3H] splits along COLUMNS (not rows like phi's qkv_proj). Also landed: learned position embeddings (wpe table added to embedding, cfg.use_learned_pos), LayerNorm with weight+bias (cfg.norm_is_layernorm + new rms_attn_b/rms_ffn_b/bo/w1_b/w3_b LayerW slots + final_norm_bias), no-RoPE (cfg.no_rope), non-gated gelu FFN (w2==SIZE_MAX → gelu(c_fc x + b) → c_proj + b), GPT-2 config key support in safetensors_reader (n_embd/n_layer/n_head/n_positions/n_ctx/n_inner + 4×hidden FF default), and the flat tensor names (wte.weight/wpe.weight/ln_f.weight — no `transformer.` prefix in the converted file). New RCPP_ARCH_GPT2 token (23). GGUF loader parity NOT wired for gpt2 (llama.cpp gpt2 GGUF layout differs — torch gate is authoritative, wired via run_gen/bringup_runner).
5. **Validation batch (cheap, no new code):** baichuan, exaone, solar, internlm2, openelm, nemotron, minicpm — already mapped to LLAMA; each gets one real-checkpoint e2e entry.
   **DONE 2026-08-14:** exaone **20/20 vs llama.cpp** (reader needed `num_layers` fallback + GPT-2-style name branch: transformer.h.N.attn.attention.*, ln_1/ln_2, c_fc_0/c_fc_1 gate/up); internlm2 **engine ≡ authoritative modeling_internlm2.py numpy top-8 EXACT** — TWO real bugs found: (a) fused wqkv is HEAD-INTERLEAVED [group(q,q,k,v)] not [q|k|v], (b) MLP is `down = w2(silu(w1 x) * (w3 x))` — w2 is the DOWN projection (w1/w3 up), was assigned swapped; vs Q8 oracle: near-tie top-3 (the/a/Paris within 0.5 logits); minicpm **engine ≡ authoritative openbmb modeling numpy top-8 EXACT** — three quirks landed (scale_emb→embedding_multiplier 12, scale_depth→residual_multiplier 1.4/√40, logits÷9→logits_scaling) PLUS a real engine bug found: **the gated-dense FFN residual ignored residual_multiplier** (granite is MoE so never hit it — fixed both dense paths); llama.cpp's minicpm oracle is UNRELIABLE (treats it as plain llama, misses the quirks); openelm **documented limitation confirmed** (heterogeneous per-layer heads/ffn + fused qkv + per-head norms — refuses loudly, needs bespoke path). New gates: `Testing/e2e_gen_check_llamacpp.py` (llama.cpp oracle for archs torch 5.x dropped) + `Testing/e2e_numpy_ref.py` (authoritative numpy for oracle-unreliable families, internlm2/minicpm). bringup_runner now selects torch/llamacpp/numpy oracle per manifest.
6. **Bespoke re-validation:** zamba2, mamba, deepseek, kimi, whisper — existing backends; one e2e per family (SSM/MLA/STT paths need their own oracles; whisper via transcript compare).
7. **gpt-oss (OpenAI, MXFP4-packed MoE) — DONE 2026-08-14: engine 20/20 generated tokens == numpy reference** (port of authoritative modeling_gpt_oss.py) on the REAL openai/gpt-oss-20b checkpoint (13.4GB packed, in `/tmp/onebit-e2e/gptoss`), full logits max|diff| 6.7e-5, top-8 EXACT. The memory-block (dequantized ~105GB fp32) is sidestepped by keeping the MoE **packed U8 blocks+scales in RAM and dequantizing only the selected 4 experts per token** (~17GB total, no torch reference needed). Quirks landed: YARN rope (theta 150000, factor 32, beta 32/1, orig_max 4096 — the ramp runs over FREQ indices arange(dim//2), an easy 2x), attention sinks (per-head learned logit cat to scores before softmax, dropped after), router bias, MXFP4 decode (FP4 e2m1, value = FP4[nibble]·2^(scale−127), low nibble→even/high→odd, 32 vals/block), interleaved gate/up rows, gate=min(g,7)·sigmoid(1.702g), **gated=(up+1)·glu** (the `+1` was missing in the first pass), top-4 softmax gating (softmax over top-k only), untied lm_head, head_dim 64, odd-safe rope pairing. Validation harness: `Testing/e2e_numpy_ref_gptoss.py` (reference) + `e2e_seq_gen` (engine). Known gap: sliding-window (128) attention on sliding layers is a no-op < 128 tokens and unimplemented beyond (same class as the gemma3 SWA gap).
   Gate: `python3 Testing/e2e_numpy_ref_gptoss.py /tmp/onebit-e2e/gptoss /tmp/gptoss_ids.txt 20` — ref-gen must equal engine-gen.

Each family lands: mapping (+self-check) → quirk code → fixture → e2e entry in `Testing/run_all.sh`.

**Gate per family:** 20/20 generated tokens identical to torch (or family-appropriate oracle); full-logits corr documented in the ledger table (research §27 format).

## Phase 3 — Bring-up automation (the force multiplier)

Turn the pilot loop into a manifest-driven runner so families 20–50 are agent-portable:

- [x] `Testing/models_manifest.json`: per family {hf_arch_strings, mapping_target, quirk_flags (rotate/neox/bias/norms/moe/fused), e2e_model, oracle, budget} — **seed landed 2026-08-14** (11 families, validated + pending deck)
- [x] `Testing/bringup_runner.sh`: mapping gate + per-family generation gate (20/20 vs torch) — **skeleton landed 2026-08-14**, qwen3 gate passes
- [ ] Codify as repo skill (`skills/` — seed is `skills/1bit-writer`): "add a family" = edit manifest + quirk code + run bringup_runner
- [ ] CI hook: manifest additions gated on `run_all.sh`

**Gate:** a new family added with zero human archaeology — mapping + manifest + runner → green.

## Phase 4 — Catalog & the count

- [x] Publish arch→checkpoint table in `docs/wiki/models.md` — **DONE 2026-08-14**
- [x] Catalog sweep — **DONE**: live HF census (220k text-gen models sampled) → 11 validated families covered 193,318 checkpoints (88%); **+GPT-OSS 407 checkpoints validated 2026-08-14** (was the #2 uncovered class; packed-MXFP4 engine path needs no torch reference). Biggest remaining causal-LM classes: **Step1MoE 2,882** · DeepSeek-V3 · Bloom · Qwen2VL · Mamba. Encoder-decoders out of scope.
- [ ] Refresh HF 1BP catalog (37 → grow with new families) + NPU FLM map as xclbins land
- [x] Count claim lands in `docs/wiki/models.md` header ("193k / 88% of HF text-gen")

**Gate:** sweep script output == documented count; table matches registry.

## Risks / decisions

- **Encoder-decoder (t5, bloom, bart, m2m):** new engine paths (cross-attention). Recommend OUT of phase-1 scope; revisit when dense coverage is done.
- **Per-family quirks are the real work** — budget 1–2 days/family; the manifest runner (Phase 3) is what makes 50 families tractable, so start it early (parallel with Phase 2 #1).
- **MoE CPU validation** — gptoss MoE now validated via the packed per-expert numpy oracle (2026-08-14); other MoE (qwen3moe, deepseek) still deferred to the engine tokenizer (htok workstream) or per-family oracle. Step1MoE (2,882 checkpoints) is now the biggest remaining causal-LM class.
- **ggml_vulkan safetensors-first routing** (pilot #3 note): safetensors models hit GGUF-first routes and failover — fine by design, revisit if first-load latency matters.
- **gemma3 SWA masking >512 tokens** — known gap, affects long-context gemma only.

## First work items (next session)

1. Phase 1 batch: enumerate HF arch strings + bulk mappings (one sitting)
2. Phase 2 #1: qwen3 real-checkpoint e2e
3. Phase 3 seed: `models_manifest.json` schema + `bringup_runner.sh` skeleton
