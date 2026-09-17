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
`Yuro1991/Qwen-Drive-1.0-4B`) — reviewed 2026-09-13 (section below; a driving
planner, not a text decoder).

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

## `qwendriveforplanning` — Qwen-Drive (driving planner)

* **Class / model_type**: `QwenDriveForPlanning` / `qwen_drive`; example
  `Yuro1991/Qwen-Drive-1.0-4B`.
* **Not a text decoder**: the top-level config nests a `vlm_config`
  (`Qwen3_5ForConditionalGeneration` — the Qwen3.5 text tower plus vision) and an
  `expert_config`, and its own keys are trajectory planning: `trajectory_hz`,
  `trajectory_point_dim`, `num_future_points`, `num_history_points`,
  `trajectory_scale`, `num_inference_steps`, `noise_init_std`, `min_one_minus_t`,
  `max_reasoning_tokens`, plus `id2label`/`label2id` for two labels. Image
  geometry (`image_patch_size`, `image_spatial_merge_size`,
  `image_temporal_patch_size`, `current_image_pixels`, `history_image_pixels`)
  confirms a video/image input path.
* **Why an alias is wrong**: the familiar part is the Qwen3.5 tower; the model's
  contract is a **driving trajectory** — a sampled (denoising) plan over
  `num_inference_steps` with a noise seed and `min_one_minus_t`, not tokens. The
  engine has no head or pipeline for that, so a mapping would serve the tower and
  silently drop the planner, exactly as with `molmoact2` and `cosmos3edge` above.
* **Real support needs**: a scope decision first (is trajectory planning in scope
  at all?), then the expert/trajectory head.

---

## Uncovered classes reviewed later — same standard, and still not aliases

The lane split above is the *watcher's*: a class is printed `!! SIGNIFICANT` when
`_is_significant()` calls it a major-family/vision arrival, and `!! UNCOVERED`
otherwise — and the alert's instruction for that second lane is "add the mapping
… alias if it is a known family". The entries below arrived in the `!! UNCOVERED`
lane from 2026-09-13 on (`han2han` on 2026-09-14) and are **not** aliases either,
for the same reason the sections above exist: a mapping would claim support the
engine has not validated, and for `englishbase` it would be silently wrong rather
than merely unsupported.

Evidence below is from the official HF configs **and the models' own modeling
code**, fetched 2026-09-13/14.

### `englishbase` — `SlayerLab/fabryka-english-250m-e01-sft-v1`

**Implemented 2026-09-13 — no longer an uncovered class** (`RCPP_ARCH_ENGLISHBASE`
= 1006, generic backend): the two-matrix relu2 MLP rides the non-gated FFN branch
and the parameter-free QK-norm is the existing per-head path driven by an all-ones
`[head_dim]` weight. Validated token-for-token against the model's own code — 68/68
greedy tokens over three prompts — with `Testing/englishbase_parity_check.cpp`. The
review below is kept as the record of *why it was not an alias*, which is what
decided the shape of the implementation.

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

### `vapor` — `Neeze/Vapor-2B-V1.2`

* **Class / model_type**: `VaporForCausalLM` / `vapor`.
* **It looks like the cleanest alias of the five.** The config carries the LFM2
  schema *verbatim*: every `block_*` and `conv_*` key identical to LFM2-2.6B,
  **all 30 `layer_types` identical**, hidden 2048, 30 layers, 32 heads / 8 kv,
  intermediate 10752, `conv_L_cache` 3. What is left over is packaging (`dtype`,
  `auto_map`, the `tie_*` pair), config-driven (vocab 128000, rope theta 1e7) —
  and one thing that is neither: an explicit `rank_config`.
* **The checkpoint is what disqualifies it** (weights index read 2026-09-13):
  layer 0 stores the dense SwiGLU form — `feed_forward.w1/w2/w3` — and
  **layers 1–28 do not**. They store `feed_forward.A2`, `A_shared`, `B1`, `B2`,
  `B3` plus `side_A2`, `side_A_shared`, `side_B1..B3`: the `rank_config`'s
  `SharedSwiGLULinear` (r_shared 1396, r2 1116, side_rank 8) realised in the
  tensors rather than a training-time detail that materialises on export.
* **Why an alias is wrong**: the dense names the readers know (`w1`/`w3`, ~18
  hits each in this repo) describe layer 0 only; `A_shared`, `side_B1` and
  `feed_forward.A2` have **0 hits** anywhere in the tree. An LFM2 mapping would
  read the two dense layers it recognises and have nothing for the 28 factorized
  ones.
* **Real support needs**: a materialisation (or runtime) path for
  `SharedSwiGLULinear`, alongside the plain dense block, then decode validation.

### `fidel` — `4E-AI/Fidel1.1-1B`

* **Class / model_type**: `FidelForCausalLM` / `fidel`.
* **Not a text transformer at all.** Its 359 tensors live entirely under a
  `policy.` prefix and the stack is bespoke (from the safetensors header, read
  by range request — the file is 5.6 GB and was never downloaded):
  * `policy.model.layers.N.mixer.mamba.*` — 88 tensors: the per-layer mixer is
    **Mamba**, not attention;
  * `policy.model.layers.N.moe.shared_fc1_gate` / `moe.w_gate` — a MoE with
    shared experts;
  * `policy.model.layer_adapters.N.base.{up,down}` and `.delta.{up,down}` — the
    per-layer rank adapters the config's `ranks` block describes (base 512,
    delta 256);
  * `policy.model.prompt_hyper_conditioner.*` (78 tensors) and
    `policy.model.prompt_cross_conditioner.*`;
  * `policy.model.contextual_prompt_v7/v8/v9.*` and `policy.chat_prompt.*` —
    four separate prompt-attention blocks;
  * `policy.model.lm_head.base.weight` + `lm_head.lora_a` — the head itself is
    base plus LoRA.
  The header's own metadata agrees with the config:
  `execution_contract: fidel_step9051_legacy_reference_fp32`.
* **Why an alias is wrong**: no engine family has mamba mixers + MoE + rank
  adapters + prompt conditioners, and `layer_adapters`,
  `prompt_hyper_conditioner` and `mixer.mamba` each have **0 hits** in this repo.
  There is no partial match to argue about.
* **Real support needs**: a scope decision first — this is a policy model, not a
  chat LM — and then the adapter/conditioner machinery.

### `han2han` — `cadazar/han2han-it`

**Excluded from the census 2026-09-15, not mapped.** It belongs to the `#1676`
encoder-decoder lane in `Testing/census_coverage.py`'s `NON_TEXT_GEN`, beside
`t5`/`bart`/`m2m100`, so it no longer counts in the `with_arch` denominator and
the watcher no longer prints it as an uncovered class. The review below is the
evidence for that classification.

* **Class / model_type**: `Han2Han` / `han2han` (custom code — `han2han_config.py`,
  `modeling_han2han.py`, Flax-origin port).
* **A seq2seq encoder-decoder, not a causal decoder.** The config states it
  (`is_encoder_decoder: true`, `decoder_start_token_id` 9, 18 encoder + 18
  decoder layers at H=640, `d_ff` 2048, `decoder_cross_attention_types: ["mha"]`,
  `attn_window` 128), and the 825-tensor header (read 2026-09-15 by range
  request — the file was never downloaded) confirms it: `encoder.h.layers.N`
  has no cross-attention, while **every** `decoder.h.layers.N.crossattention.*`
  carries its own `query`/`key`/`value`/`c_proj` plus per-head `q_norm`/`k_norm`
  and sub-layer `attn_sub_norm`.
* **Its input machinery is unlike any engine family.** Three embedding tables
  per side — `wte` (subword), `wce` (char, `char_vocab_size` 5376), `wje` (jamo,
  `jamo_vocab_size` 4992) — feed `subword_proj` + `ln_emb`, with char/jamo
  bucket buffers `cbu`/`jbu`; the MLP is dense GEGLU (`wi_0`/`wi_1`/`wo`), so the
  config's `moe_num_experts: 8` is not in the weights at all (the shipped code
  raises `NotImplementedError` for sparse layers). `wce`, `wje`,
  `subword_proj`, `crossattention`, `cbu` and `jbu` have **0 hits** in the
  engine.
* **Why an exclusion is the right verdict, not a mapping**: the engine's only
  cross-attention kernel is whisper's (`src/whisper_hip.hip`), a speech path,
  and the plan that produced `#1676` declared encoder-decoders out of scope.
  Mapping `han2han` onto a decoder-only family would be a guess about a
  checkpoint half of whose layers attend to an encoder that does not exist in
  the engine.
* **Real support needs**: a whole encoder stack + decoder cross-attention — i.e.
  the scope decision `#1676` already made in the negative.

The common thread: each of these is *shaped* like a family the engine already
routes (`llama`, `gdn`, `kimi`, `lfm2`) — `fidel` only in the loosest sense, as a
Mamba/MoE hybrid — and each differs in a mechanism that changes the computation,
which is the one thing a `rcpp_arch_from_string` mapping cannot express. `vapor`
is the case worth remembering: from the config alone it looked like a one-line
alias, and only the tensors said otherwise.

**Reading a checkpoint without downloading it.** For a multi-GB
`model.safetensors`, the header is enough to settle this class of question and
costs one range request (follow redirects — `-L` — or you get the 302 body):

```sh
U=https://huggingface.co/<repo>/resolve/main/model.safetensors
H=$(curl -sSL -r 0-7 "$U" | python3 -c 'import sys,struct; print(struct.unpack("<Q", sys.stdin.buffer.read(8))[0])')
curl -sSL -r 8-$((8+H-1)) "$U" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d), list(d)[:20])'
```

`<repo>/raw/main/model.safetensors.index.json` answers the same question in one
request when the model is sharded.

---

## Reviewed 2026-09-17 — the next four un-reviewed classes by instance count

`Testing/census_coverage.py` reports 184 uncovered instances across 77 classes on
`main` at `d7e9aac7c`. The daily watcher only names classes inside its rolling
100-model window, so most of those 77 are never surfaced to anyone. These are the
four largest un-reviewed ones, in decreasing order, reviewed from their live
configs and HF metadata.

**A limit on all four**: the standard above is to decide an alias on the
checkpoint's *tensors*, and these verdicts rest on `config.json` plus HF metadata,
not on tensor headers. That is decisive for the two that are not text LMs at all
(a vocabulary of 82, and a `text-to-speech` pipeline tag); for the two real ones it
establishes "not a name we map", not "not an alias" — the tensor check is still
outstanding and is called out per entry.

### `blockmtp` / `looped_block_mtp` — `violetxi/hparam-92m-block-mtp-…` (12)

**Not a text LM. Not an alias, and not a coverage gap.**

* **Class / model_type**: `BlockMTPForCausalLM` / `looped_block_mtp`, 12 instances,
  a single aggregate entry.
* **What it is**: `violetxi/hparam-92m-block-mtp-truncated-peak_lr-1e-5-wu-0p05-s42`
  and its siblings — an experiment grid (`hparam-*`, `chess-*`, `...-4xh100`) over a
  looped / block multi-token-prediction transformer.
* **The decisive number is `vocab_size: 82`**, against 100k+ for every text family
  the engine serves. A model whose entire vocabulary is 82 tokens is not a chat LM
  and cannot be one; the class is an artefact of one uploader's sweep.
* **Verdict**: no implementation is warranted. This sits in the same category as the
  already-excluded experiment uploads (`helloworld`, `mbztestmodel`, `myfirstllm`),
  but it has not been added to an exclusion set here, because that changes the
  published denominator and is a policy call rather than a review outcome.

### `jarvistitanmoe` / `jarvis_titan_moe` — `dhanesh-hf/Jarvis-Titan-V12-MoE-14B` (9)

**A real causal text LM. Not a name the registry maps — and the tensor check is
outstanding.**

* **Class / model_type**: `JarvisTitanMoEForCausalLM` / `jarvis_titan_moe`.
* **Config**: `vocab_size` 152064, `hidden_size` 3584, `num_hidden_layers` 28, and
  MoE. A genuine text-generation family from a community uploader
  (`Jarvis-Titan-V10-SFT-Merged`, `V12-MoE-14B`, `V12-MoE-Adapted`).
* **Verdict**: this is a coverage gap worth implementing, not an artefact. Whether it
  is also an *alias* is untested: the vocabulary 152064 matches the Qwen2.5-class
  size, so the tensors — not the class name — are what would settle it.

### `canopy` — `canopylabs/orpheus-3b-0.1-ft` (7)

**Text-to-speech, not a causal text decoder. Not an alias, and not a coverage gap.**

* **Class**: `CanopyForCausalLM`, 7 instances.
* **What it is**: Canopy Labs **Orpheus**, whose HF `pipeline_tag` is
  `text-to-speech` and which is llama-architecture underneath. The name says
  `ForCausalLM`; the model is a speech synthesiser with a text backbone.
* **Verdict**: the engine has no TTS path — the `NON_TEXT_GEN` comment states the
  plan put TTS out of scope beside `parlertts` and `kosine`. Recording it here, not
  excluding it, for the same denominator reason as `blockmtp`.

### `zgcm` — `zgcagi/ZGCM-1-7B` (6)

**A real causal text LM. Not a name the registry maps — and the tensor check is
outstanding.**

* **Class / model_type**: `ZgcmForCausalLM` / `zgcm`.
* **Config**: `vocab_size` 155136, `hidden_size` 4096, `num_hidden_layers` 32 —
  a 7B-class text LM with a full text vocabulary, and a trainer's own architecture
  name (`ZGCM-1-7B`, plus `-Pretrain-Curriculum`, `-Midtrain-Staged-16K/256K`).
* **Verdict**: a coverage gap worth implementing. As with `jarvis_titan_moe`, the
  alias question needs the tensors.

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
