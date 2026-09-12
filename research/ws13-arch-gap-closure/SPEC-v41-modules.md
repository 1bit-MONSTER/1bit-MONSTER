# ws13 SPEC — the V4.1-only modules (P0.2)

**What this is:** the maths and name map for the modules **for which no `transformers` reference
exists** — `transformers` 5.16.1 has `DeepseekV4` but no `DeepseekV41` class, so `engram`,
candidate-block selection and DSpark have no executable oracle there. Their authority is DeepSeek's own
inference code, fetched 2026-09-12:

* `deepseek-ai/DeepSeek-V4.1-Flash/inference/engram.py` — 184 lines, the hashed n-gram memory.
* `deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py` — 1309 lines: `ModelArgs`, `Compressor` (:429),
  `Indexer` (:488), `select_candidate_blocks` (:583), `Attention` (:613), `Engram` (:328),
  `DSparkAttention` (:1032), `DSparkMarkovHead` (:1077), `DSparkConfidenceHead` (:1089),
  `DSparkBlock` (:1100).

The shared machinery (compressor, indexer, sliding/CSA/HCA attention) *does* have an executable oracle
in `transformers` — that is what the Phase-0 fixtures exercise — but the two disagree on names, so the
map below is required before either can be trusted.

## 1. Name map: published checkpoint ↔ transformers reference

| maths | checkpoint (`model.safetensors.index.json`) | transformers 5.16.1 (`DeepseekV4`) |
|---|---|---|
| compressor projection | `attn.compressor.wkv` | `self_attn.compressor.kv_proj` |
| compressor gate | `attn.compressor.wgate` | `self_attn.compressor.gate_proj` |
| compressor norm | `attn.compressor.norm` | `self_attn.compressor.kv_norm` |
| per-position pooling bias | `attn.compressor.ape` | `self_attn.compressor.position_bias` |
| indexer query projection | `attn.indexer.wq_b` | `self_attn.compressor.indexer.q_b_proj` |
| indexer score weights | `attn.indexer.weights_proj` | `self_attn.compressor.indexer.scorer.weights_proj` |
| indexer's own compressor | `attn.indexer.compressor.{wkv,wgate,norm,ape}` | nested as `compressor.indexer.{kv_proj,gate_proj,kv_norm,position_bias}` |

Read the maths from `model.py` (it matches the checkpoint naming) and validate against the executable
reference through this map. Shape evidence for the bias: `position_bias` is `[compress_rate, head_dim]`
(measured: `[4,32]` for CSA=4, `[128,16]` for HCA=128), which is also why the rate is baked into the
tensors.

## 2. Engram — hashed n-gram memory added to the residual stream

`Engram` (`model.py:328`), used only on the layers in `engram_layer_ids` (`Block.__init__`: `if
engram_layout is not None and layer_id in engram_layout.layer_ids`). It is **not** a new matmul kernel:
it is a hashing + gather + small linear, which is why this is the cheapest of the three to implement.

**Steps (`engram.py`), all CPU-reproducible:**

1. **Compressed token map.** `build_compressed_token_map(tokenizer)` normalises each token
   (NFKC → NFD → StripAccents → Lowercase → collapse whitespace → Strip, with a private-use sentinel so
   a token that is exactly one space survives) and maps every token id onto that smaller id space, so
   `" The"`, `"the"` and `"THE"` hash identically. **Trap:** `NgramHashState.__init__` asserts the
   resulting size equals `args.engram_compressed_vocab_size` — every hash multiplier derives from it, so
   a mismatch silently rehashes the whole table instead of failing.
2. **Per-(layer, n-gram, head) bucket space.** `EngramLayout` draws *disjoint* primes in order
   (`find_next_prime` upward from `engram_vocab_size - 1`, never reused); each (n-gram size, head) pair
   owns its own prime-sized range; `offsets` make the concatenated ranges disjoint.
3. **Hash multipliers.** `compute_hash_multipliers`: one per (layer, lookback) from a per-layer RNG
   (`np.random.default_rng(10007 * layer_id)`), forced odd (`value * 2 + 1`) and bounded so
   `token_id * multiplier` cannot overflow int64.
4. **The hash itself.** For each lookback `shift` in `0..max_ngram_size-1`, gather the token `shift`
   positions back; block the position if `pos < shift` or the source token is `DEAD`; replace blocked
   slots with `engram_pad_id` (default **2**, "matches training"). Then a running XOR of
   `token * multiplier` accumulates the 2-gram … `max_ngram_size`-gram hashes, each taken `% primes`,
   plus `offsets`. Output `[B, L, n_engram_layers, n_hash_cols]`.
5. **DEAD tokens and images.** `token_mask` marks image spans; a blocked n-gram can never span one, and
   the state cache carries this across the prefill/decode split (so a decode step sees the same hashes
   a full prefill would).
6. **Insertion.** `Engram.embed = ParallelEngramEmbedding(num_embeddings[layer_hash_index], head_dim)`
   (`model.py:296`) holds `weight` as **fp8 e4m3** with a per-block `scale`
   (`dim // block_size` columns) — i.e. the table is quantised and needs the same `ue8m0` block-scale
   handling as the experts. The retrieved rows are consumed by
   `wkv: Linear(n_hash_cols * head_dim → dim * (hc_mult + 1))`, with `q_weight` / `k_weight` as
   `[hc_mult, dim]` parameters (initialised to ones) and `clamp_value = 1e-6`; the `(hc_mult + 1)`
   output width is how the result lands in the **mHC residual streams** (`hc_mult`) plus one extra
   channel — an additive n-gram memory, not an attention.

Checkpoint tensors involved: `layers.N.engram.{embed.weight,embed.scale,q_weight,k_weight,wkv.weight,wkv.scale}`
(present on 2 layers in V4.1-Flash).

## 3. Candidate pre-filtering — a second selection level before the indexer

`select_candidate_blocks` (`model.py:583`) and the `Indexer` docstring ("rectified then combined by
`weights_proj`. With a candidate source this is the second of two levels; `select_candidate_blocks` is
the first") describe **two-stage sparse selection**:

* `candidate_source_layer` (`< 0` disables it; `Indexer.is_candidate_source` marks the producing layer,
  `uses_candidates = 0 <= candidate_source_layer < layer_id` the consumers),
* `candidate_topk_blocks`, `candidate_block_size` — the cheap block-level filter,
* then the indexer keeps its `index_topk` best compressed positions per query (`Indexer` docstring).

The wiring keys in the config are the *source layer* lists: `candidate_source_layer_id`,
`index_source_layer_ids`, `kv_source_layer_ids` (`Indexer.owns_k = layer_id in args.kv_source_layers`).
Consequence for the engine: the compressed-attention path is not "attend over compressed entries with
top-k" but a **two-level** scheme whose first level reads an earlier layer's state — a data dependency
the current single-layer, single-pass engine does not express, so P1's design must carry per-layer
source indices rather than a per-layer flag.

## 4. DSpark is the MTP/draft head — it does not belong on the generation path

`ModelArgs` comments say it plainly ("dspark draft head. Only the forward pass is implemented here —
nothing calls `forward_spec`"), and the classes are `DSparkAttention` / `DSparkMarkovHead` /
`DSparkConfidenceHead` / `DSparkBlock` with `dspark_block_size`, `dspark_noise_token_id`,
`dspark_target_layer_ids`, `dspark_markov_rank`, `dspark_n_routed_experts`,
`dspark_n_activated_experts`. These are the `mtp.*` tensor families seen in the inventory
(`markov_head`, `confidence_head`, `main_proj`, `main_norm`).

**Scope consequence:** plain autoregressive generation does not need DSpark/MTP weights loaded. That
removes `mtp.*` (and its `N`-layer stack) from the critical path — the workstream's P2 should say so
explicitly rather than loading everything a checkpoint contains.

## 5. What this changes in the plan

* **P2.1 splits.** Load-bearing for text generation: `engram` (2 layers), `ffn.gate.bias_vl`,
  candidate-block selection, the new shape. *Not* load-bearing: DSpark/MTP (draft head), the vision
  tower + `aligner` (still a scope decision).
* **P1's design constraint is now explicit**: two-level candidate selection with source-layer
  dependencies, and a compressor whose pooling bias is rate-shaped per layer — so the loader must carry
  `compress_ratios`, `*_source_layer_ids` and `index_topk` per layer, not infer them.
* **Engram needs no new kernel** (hashing + gather + one linear per engram layer), but its embedding
  table is fp8 with per-block scales, so it depends on the same block-scale machinery as the experts.
