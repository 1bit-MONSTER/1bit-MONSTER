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
re-deriving them. A later section, *Uncovered classes reviewed later*, applies
the same standard to `!! UNCOVERED` classes that turned out not to be aliases
either.

Evidence below was taken from the **official HF configs** (fetched
2026-09-12), not from the model cards or the class name.

**This is a snapshot.** New significant arrivals are expected as HF moves —
add them here when they are reviewed. Seen while writing this: `deepseekv41`
(4 models on 2026-09-11, 6 on 2026-09-12), the other three from the 2026-09-11
run, and new on 2026-09-12 `qwendriveforplanning` (e.g.
`Yuro1991/Qwen-Drive-1.0-4B`) — **not yet reviewed**, so its census line is
still the only record of it.

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

## Uncovered classes reviewed later — same standard, and still not aliases

The lane split above is the *watcher's*: a class is printed `!! SIGNIFICANT` when
`_is_significant()` calls it a major-family/vision arrival, and `!! UNCOVERED`
otherwise — and the alert's instruction for that second lane is "add the mapping
… alias if it is a known family". The three below arrived in the `!! UNCOVERED`
lane on 2026-09-13 and are **not** aliases either, for the same reason the
sections above exist: a mapping would claim support the engine has not
validated, and for `englishbase` it would be silently wrong rather than merely
unsupported.

Evidence below is from the official HF configs **and the models' own modeling
code**, fetched 2026-09-13.

### `englishbase` — `SlayerLab/fabryka-english-250m-e01-sft-v1`

* **Class / model_type**: `EnglishBaseForCausalLM` / `fabryka_english_base`.
* **Shape**: 36 layers, H=768, 6 heads / 2 kv, head_dim 128, intermediate 3040,
  vocab 32768, tied embeddings, rope theta 10000, `rms_norm_eps` 1e-6,
  `qk_norm: true` — llama-shaped on the surface, which is why the watcher filed
  it as a family variant rather than a significant arrival.
* **Decisive evidence — the model's own `model.py`**: when
  `recipe.activation == "relu2"` (this config's `recipe` says exactly that) the
  block's MLP is replaced by `ReluSquaredMLP`, which computes
  `down_proj(relu(up_proj(x)).square())` — **one up matrix, no gate, ReLU²** —
  and the same file describes its QK normalization as *parameter-free*.
  `hidden_act: "silu"` is only the default the Llama config class fills in; the
  recipe overrides the MLP at construction, so the two keys disagree and the
  code is what settles it.
* **Why an alias is wrong**: Llama's and Qwen3's MLP is gated SwiGLU
  (`gate_proj` + `up_proj` + `down_proj`, SiLU), and Qwen3's qk-norm carries
  learned `q_norm`/`k_norm` weights. A mapping would load and then compute a
  different function — silent substitution, not a crash.
* **Real support needs**: a two-matrix ReLU² MLP and a parameter-free QK-norm, or
  a loud refusal.

### `gdn2` — `Berlm/hyb16-gdn2-s1`

* **Class / model_type**: `GDN2ForCausalLM` / `gdn2`.
* **Shape**: 24 layers, H=1024, 6 heads, head_dim 128, `conv_size` 4, hybrid —
  full attention on layers 3/7/11/15/19/23 — dense SwiGLU (2816), vocab 32000,
  tied embeddings.
* **The config schema is `fla`'s** (flash-linear-attention), not HF's:
  `attn_mode: chunk`, `fuse_cross_entropy` / `fuse_norm` / `fuse_swiglu`,
  `expand_v`, `use_short_conv`, `allow_neg_eigval`, and an `ra_*` block
  (`ra_project`, `ra_gate_b`, `ra_theta_init`, `ra_lambda_max`, `ra_feedback`)
  that no engine path reads.
* **Nearest implemented family, and why that is still not a mapping**: the
  registry does have Gated-Delta paths — `gateddeltanet` / `deltanet` /
  `tinygdn` → `RCPP_ARCH_QWEN3NEXT`, and `RCPP_ARCH_QWEN35` is Qwen3.5's *dense*
  GDN — but the engine's reader (`src/qwen3_5.cpp`) keys on Qwen3.5's own field
  names (`linear_conv_kernel_dim`, `linear_num_key_heads`,
  `linear_value_head_dim`) and on tensor names with expected sizes
  (`linear_attn.conv1d.weight`). This config supplies neither. A matching
  mechanism is not a matching checkpoint; an alias here would be a guess about
  weight layout.
* **Real support needs**: an `fla` → engine config adapter, a tensor-layout check
  against one real checkpoint, and a decision on `ra_*`.

### `haiku` — `kerzgrr/Haiku-base`

* **Class / model_type**: `HaikuForCausalLM` / `haiku`.
* **Shape**: 36 layers, H=1024, 8 heads / 8 kv, `layer_types` = `kda` ×3 then
  `gated_mla` (`full_attention_interval: 4`), `mlp_activation: "situ_glu"`
  (`situ_gate_cap`, `situ_up_cap`), MLA ranks (`mla_q_lora_rank` 512,
  `mla_kv_lora_rank` 256, `mla_qk_nope_head_dim` / `mla_v_head_dim` 128),
  `partial_rotary_factor` 0.5, rope 1e6, and MTP heads (`mtp_num_heads`,
  `mtp_adapter_rank`, `mtp_loss_weight`).
* **Why an alias is wrong**: KDA and gated MLA *are* named in the registry —
  `RCPP_ARCH_KIMI_K3` is "Moonshot Kimi K3 — 2.8T MoE with KDA + Gated MLA +
  LatentMoE" — but that token is **routed, not implemented**: `include/kimi_k3.h`
  is included by nothing in the tree and its `load_from_gguf` prints
  `[KimiK3] Loading not yet implemented for Strix Halo targets.` and returns
  false; `src/model_router.cpp` has no branch for the arch, so it falls through
  to the generic `hip_gpu` → `cpu_generic` default; and `situ_glu` appears in no
  engine source — the only SITU in the tree is upstream llama.cpp's
  `kimi_k3_situ`. (That routing/validation split is deliberate and documented:
  issue #1635, closed as "not fixable on this box".) Mapping Haiku onto KIMI_K3
  would chain one unvalidated class onto another.
* **Real support needs**: a SITU-GLU MLP plus a real KDA / gated-MLA
  implementation — the K3 work itself, which `research/TRACKING.md` still
  carries as not started ("QK-Normed MLA absorption on Kimi K3 gated-MLA
  decode").

The common thread: all three are *shaped* like families the engine already
routes (`llama`, `gdn`, `kimi`), and each differs in a mechanism that changes
the computation — which is the one thing a `rcpp_arch_from_string` mapping
cannot express.

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
