# Architecture gaps — SIGNIFICANT arrivals the registry does not map

`Testing/hf_new_models.py` splits uncovered architecture classes into two lanes:

* **family variant** — a renamed/same-shape sibling, which the family-variant
  lane resolves with a `rcpp_arch_from_string` mapping (and may draft a PR for);
* **`!! SIGNIFICANT`** — a major-family or vision arrival, where an alias would
  **fake support**. Significant arrivals are deliberately excluded from the
  auto-PR path.

This file is the reviewed evidence for the significant lane: what each family
is, what it adds over the closest supported architecture, and what real support
requires. It exists so that (a) nobody later "fixes" the alert with an alias,
and (b) whoever implements the family starts from measurements instead of
re-deriving them.

Evidence below was taken from the **official HF configs** (fetched
2026-09-12), not from the model cards or the class name.

Measured reference — the shape the census compares against:

| engine arch | implementation | reference model |
|---|---|---|
| `RCPP_ARCH_DEEPSEEK_V4` | `src/deepseek_v4.cpp` (mHC residual, CSA+HCA hybrid attn, FP4 MoE) | `deepseek-ai/DeepSeek-V4-Flash` — 43 layers, H=4096, 256 experts |
| `RCPP_ARCH_MAMBA` | Mamba-1 + MoE (BlackMamba) | — |
| `RCPP_ARCH_FALCONH1` / `RCPP_ARCH_NEMOTRONH` / `RCPP_ARCH_GRANITEMOEHYBRID` | Mamba-**2** SSM + attention hybrids | — |
| `RCPP_ARCH_OLMO` | OLMo text decoder (`molmo` already maps here) | — |

---

## `deepseekv41` — DeepSeek V4.1

* **Class / model_type**: `DeepseekV41ForCausalLM` / `deepseek_v41` (text tower
  `deepseek_v41_text`); example `deepseek-ai/DeepSeek-V4.1-Flash`,
  `Solstice-AI/DeepSeek-V4.1-Flash-UNCENSORED-FP8`.
* **Shape**: **40 layers, H=5120, 64 heads, 1 kv head, 384 experts, vocab
  129280** — versus V4 Flash's 43 layers / H=4096 / 256 experts. It is a
  different model, not a revision label.
* **Keys V4 does not have**: `engram_*` (8: `engram_layer_ids`,
  `engram_num_embeddings`, `engram_vocab_size`, `engram_compressed_vocab_size`,
  `engram_max_ngram_size`, `engram_n_heads`, `engram_head_dim`,
  `engram_pad_token_id`), `candidate_block_size`, `candidate_source_layer_id`,
  `candidate_topk_blocks`, `index_source_layer_ids`, `kv_source_layer_ids`.
  (`dspark_*` is *not* V4.1-exclusive — `DeepSeek-V4-Flash-DSpark` carries it
  under `model_type: deepseek_v4`.) It is also a multimodal wrapper
  (`vision_config: deepseek_v41_vision`, `image_token_id`).
* **Why an alias is wrong**: `src/deepseek_v4.cpp` reads only
  hidden/layer/head/expert/`hc_*` scalars, so a V4.1 checkpoint would load
  through the V4 path and produce tokens while **silently ignoring the engram
  and candidate-block machinery** — the silent-substitution failure this repo
  bans, not a crash.
* **Real support needs**: shape-agnostic config + the engram embedding path +
  candidate-block sparse attention (and a scope decision for the vision tower),
  validated against reference logits from the HF implementation.

## `qwen3mamba3` — Qwen3 + Mamba-3 hybrid

* **Class / model_type**: `Qwen3Mamba3ForCausalLM` / `qwen3_mamba3`; example
  `arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1`.
* **Keys**: `layer_types` (attention/SSM mix) plus Mamba-**3** knobs —
  `mamba_mimo_rank`, `mamba_bc_norm_eps`, `mamba_bc_bias_init`,
  `mamba_rope_fraction`, `mamba_a_floor`, `mamba_chunk_size`, `mamba_d_head`,
  `mamba_d_state`, `mamba_expand`, `mamba_n_groups`, `mamba_n_heads`.
* **Why an alias is wrong**: the engine has Mamba-1 (`RCPP_ARCH_MAMBA`) and
  Mamba-2 hybrids (`FALCONH1`, `NEMOTRONH`, `GRANITEMOEHYBRID`). MIMO rank, BC
  normalisation, an `a_floor`, and a rope fraction inside the SSM are Mamba-3
  parameters with no Mamba-2 equivalent — mapping to a Mamba-2 path would
  compute a different recurrence.
* **Real support needs**: a Mamba-3 SSM kernel + the hybrid layer loop, gated
  on decode validation against the reference.

## `molmoact2` — MolmoAct2 (vision-language-action)

* **Class / model_type**: `MolmoAct2ForConditionalGeneration` / `molmoact2`;
  example `NUSMAGIC/MolmoAct2-MolmoAct2-SO100_101`.
* **Text tower**: `text_config: molmoact2_text` — 36 layers, H=2560, 32 heads,
  8 kv, vocab 154624 (`qk_norm_type`, `rope_scaling_layers`, `norm_after`).
* **Not a text-only decoder**: `action_expert_config`, `flow_matching_*` (6
  keys), `max_action_dim`, `max_action_horizon`, action/state/depth token
  ranges — the model's contract is action generation, with depth reasoning.
* **Why an alias is wrong**: the existing `molmo` → `RCPP_ARCH_OLMO` mapping is
  right for a Molmo *VLM*; here it would drop the entire action expert.
* **Real support needs**: a scope decision first (text tower only, or the action
  path), then the action-expert/flow-matching head if in scope.

## `cosmos3edge` — Cosmos3-Edge (video world model)

* **Class / model_type**: `Cosmos3EdgeForConditionalGeneration` /
  `cosmos3_edge`; example `ubr-physical-ai/Cosmos3-Edge-NF4-bnb`.
* **Text tower**: `cosmos3_edge_text` — 28 layers, H=2048, 16 heads, 8 kv,
  vocab 131072, `mlp_bias`, `pretraining_tp` (llama-shaped). Alongside
  `vision_config` (`cosmos3_edge_vision`, `num_channels` → video patch embed),
  `projector_config`, `video_token_id`.
* **Why an alias is wrong**: the text tower *looks* llama-shaped, but the model
  is a video world model; the engine has no video/image pipeline for it, so a
  mapping would advertise support for something that cannot be served.
* **Real support needs**: scope decision (is video in scope at all?) before any
  mapping.

---

## Unverifiable: gated configs

`WaveMatrix/Qwen3-VL-8B-Instruct-GPTQ-Int4` is counted as **unverifiable** on
every run: its `config.json` cannot be fetched anonymously, so the census
cannot classify it. `Testing/hf_new_models.py` sends no `Authorization` header.
Supplying a token (`HF_TOKEN` in the environment, or the standard
`~/.cache/huggingface/token`) would resolve gated repos instead of retrying
them forever.

## If a family *is* a variant

The lane split is the point: an uncovered class that is a renamed or
same-shape sibling belongs to `basic_uncovered` and is genuinely fixed by a
`rcpp_arch_from_string` mapping in `include/rocm_cpp/bitnet_model.h` + selfcheck
+ a `census_coverage.py` re-run. Verify shape equality (layers / hidden /
heads / experts / added config keys) before mapping — that is exactly the check
that moved the four families above into this file.
