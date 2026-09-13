# LFM2 native NPU path — recon + implementation spec (pi agent, 2026-09-13)

Goal `mttxt22c-a6rv75`. The user assigned LFM2. This is the recon that a native implementation
needs; nothing in `engine/npu/src` implements it yet.

## 1. What LFM2 is (from the bundle's own `config.json`)

`LFM2-1.2B-NPU2`: `model_type=lfm2`, `architectures=['Lfm2ForCausalLM']`,
**hybrid** — most layers are a gated short-convolution block, the rest are GQA attention.

| param | value |
|---|---|
| hidden_size (H) | 2048 |
| num_hidden_layers (NC) | 16 |
| num_attention_heads (NH) | 32 |
| num_key_value_heads (NKV) | 8 |
| head_dim (HD) | **64** |
| intermediate_size (IM) | 8192 |
| vocab_size (NV) | 65536 |
| rope_theta | 1000000.0 |
| norm_eps | **1e-05** |
| conv_L_cache | 3 |
| conv_bias | false |

Layer map (from the 149-entry q4nx tensor map):
- **conv layers**: 0,1,3,4,6,7,9,11,13,15 (10)
- **attention layers**: 2,5,8,10,12,14 (6)

Every layer: `input_layernorm.weight`, `post_attention_layernorm.weight`,
`mlp.{gate,up,down}_proj.weight`.
Conv layer additionally: `shortconv.in_proj.weight` (H→3H = 6144),
`shortconv.conv.weight` **[2048,3] BF16 depthwise**, `shortconv.out_proj.weight` (H→H).
Attention layer additionally: `self_attn.{q,k,v,o}_proj.weight` + `self_attn.{q,k}_norm.weight` [64].
Top level: `model.token_embd.weight` [65536,2048] BF16 (**not** `embed_tokens`),
`model.norm.weight`, `lm_head.weight` (I8; not tied).

The q4nx is **8 bytes of little-endian header (u64 = JSON byte length) then the JSON**, and
the JSON is a flat dict of tensor-name → info.

## 2. What the engine already does with LFM2 (verified)

- **Dims parse correctly.** `parse_q4nx_header` yields
  `H=2048 NC=16 NH=32 NKV=8 HD=64 IM=8192 NV=65536 GU_split=0 rope_theta=1000000` and the
  engine prints `NPU Engine Universal — lfm2_1_2b`. So the earlier claim that the parser
  "chokes" is not true of the engine's own parser.
- **Family detection already knows LFM2**: `NV == 65536 -> family = "lfm2"`, and
  `flm_prefill_bridge.cpp` instantiates `lfm2_npu` for it (so the **FLM-reference path
  works**). The build links `-llfm2_npu`.
- **Native path dies at weight init**: with no FLM-ref env, the engine looks for canonical
  Qwen3 tensor names, finds no QKV weights, and reports
  `cq before init: MD=128 KD=0 ND=0` → `No insts for QKV` → `FAIL QKV`.

## 3. Reference tokens + inputs (the gate)

FLM reference via `NPU_FLM_PREFILL=1` (FLM's own `lfm2_npu` lib):

| npt | FLM-ref boot | FLM-ref prefill |
|---|---|---|
| 256 | **708** | 444–525 ms |
| 1024 | **1443** | 720 ms (≈1428 tok/s) |

**The token ids must be LFM2-tokenized** — LFM2's vocab is 65536, so the Qwen3-tokenized
`/tmp/ids_*.txt` (ids up to 151936) are out of range and give `boot=0`. Generate with:

```
engine/npu/tokenizer/tokenize ~/.config/flm/models/LFM2-1.2B-NPU2/tokenizer.json \
  < benchmarks/prompts/reclaimer.txt | tr ',' ' ' > /tmp/lfm2_ids.txt
# -> /tmp/lfm2_256.txt, /tmp/lfm2_1024.txt (first 256 / 1024 ids)
```

## 4. Captured FLM LFM2 ELFs

`~/npu-build/caplfm2` (1.2 GB, 1182 files, 23+ ELFs), captured by LD_PRELOADing
`cap_interposer.so` onto the NPU_FLM_PREFILL path:

```
LD_PRELOAD=…/cap_interposer.so CAP_DIR=~/npu-build/caplfm2 CAP_NO_SYNC=1 CAP_SKIP_BIG=1 \
  NPU_FLM_PREFILL=1 ./engine/npu/build/npu_engine_qwen3_4b \
    ~/.config/flm/models/LFM2-1.2B-NPU2/model.q4nx 1 /tmp/lfm2_256.txt   # boot=708
```

Sizes: 68384, 75744, 182192, 15472, 5616, 20400, 26432, 5760, 12352, 33472, 8080, 7424, 24832.
The attention ELFs should be identifiable by a 256-vs-1024 differential (as done for Nanbeige);
the shortconv kernel is the one with no Qwen3 analogue.

## 5. What a native implementation needs

1. **Name map / weight loading** for `model.token_embd.weight`, `model.norm.weight`,
   `model.layers.N.input_layernorm.weight`, `post_attention_layernorm.weight`,
   `mlp.{gate,up,down}_proj.weight`, and per layer-type
   `shortconv.{in_proj,conv,out_proj}.weight` or `self_attn.{q,k,v,o}_proj.weight` +
   `self_attn.{q,k}_norm.weight`. Today every lookup is Qwen3-canonical
   (`model.embed_tokens.weight`, `self_attn.q_proj.weight`, …) — hence `KD=0`.
2. **Shortconv block** (the only genuinely new kernel):
   `x → in_proj (H→3H) → [B,C,x] split → depthwise k=3 with a 2-tap persistent cache
   (conv_L_cache=3) → C ⊙ x gate → out_proj (H→H)`. Precedent for a depthwise conv with
   persistent state exists in the Zaya CCA path (`conv_state`, 2-tap).
3. **Hybrid dispatch**: conv vs attention per layer. There is already a name-pattern
   dispatcher (`is_gdn_layer`, ~line 629) to model this on; the discriminator here is the
   presence of `shortconv.` vs `self_attn.` for the layer.
4. **Packing/BO layout** for whichever native path is used. The high-performance bf16
   prefill path packs a Qwen3-shaped layer BO (`npu_bf16_pack_layer`), which will not fit
   LFM2; the shortconv weights and the k=3 depthwise kernel need their own layout.
5. **rope_theta / norm_eps**: `1e6` matches the engine default and `read_config_rope_theta`
   now reads `config.json` anyway; `norm_eps=1e-5` differs from the engine's `EPS=1e-6f`
   constant and should be plumbed per-model.

## 6. Honest scope

This is a **family implementation** (new architecture), not a tuning task: a new weight/name
map, a new shortconv block + its device kernel/BO layout, and hybrid dispatch. The FLM-ref
path and the captured ELFs de-risk it, and the gate is unambiguous (708 @256, 1443 @1024).

## 7. Exact change list (so the work is mechanical)

**Loader — `npu-infer/src/model.c`** (SHARED; the dsh agent said it would stay out of
`engine/npu/**` only, so coordinate before editing this file):

1. **Embed alias** (~line 614). After
   `int idx_emb = find_tensor("model.embed_tokens.weight", tensors, num_tensors);` add
   `if (idx_emb < 0) idx_emb = find_tensor("model.token_embd.weight", tensors, num_tensors);`
   LFM2 names the embedding `model.token_embd.weight`.
2. **Shortconv tensors** — `model.layers.%d.shortconv.in_proj.weight` (H→3H),
   `model.layers.%d.shortconv.conv.weight` ([H,3] BF16 depthwise),
   `model.layers.%d.shortconv.out_proj.weight` (H→H). Add `TensorDesc` fields to
   `ModelWeights` beside the q/k/v/o/gu/d ones, plus `find_tensor` calls next to the `mlp.*`
   block (~line 708).
3. **Layer type** — a layer is a conv layer iff `shortconv.*` resolves, else attention. The
   attention lookups already match LFM2 (`self_attn.{q,k,v,o}_proj.weight` and
   `self_attn.{q,k}_norm.weight`), so the existing q/k-norm path applies unchanged. The layer
   count derivation (~line 622) already keys on `.input_layernorm.weight`, which every LFM2
   layer has.

**Packer — `npu_pack_layer_bo` / `npu_layer_tile_offsets`** (`npu-infer/src/runtime_layer.cpp`):
the BO layout is Qwen3-shaped (`offs[6] = {q,k,v,o,gu,d}`). Conv layers need their own offsets
for `in_proj` (H→3H), `conv` (BF16 [H,3]) and `out_proj` (H→H); attention layers can reuse the
q/k/v/o layout with HD=64.

**Engine — `engine/npu/src/npu_engine_universal.cpp`**: the per-layer body is one Qwen3
sequence. Dispatch on layer type (precedent: `is_gdn_layer`, ~line 629).

> **Block math CORRECTED 2026-09-13.** The order is counter-intuitive and an earlier
> revision of this section had it wrong. Taken from the installed transformers
> implementation (via the dsh agent's HF-identical CPU forward,
> `origin/feat/npu-lfm2-decoder:engine/npu/tools/lfm2_cpu_runner.cpp`):
>
> ```
> BCx = in_proj(h);  B, C, x = BCx.chunk(3, -1)
> t   = B * x                                   # elementwise, chunks 0 and 2
> t   = causal_conv1d(t, shortconv.conv.weight) # depthwise, k=3, 2-tap cache
> y   = C * t                                   # gate, chunk 1
> out = out_proj(y)
> ```
>
> So the conv is applied to `B*x` and the gate is `C*(...)`, NOT "conv on B then C⊙X".
> The old wording ("depthwise k=3 ... → C⊙X gate") is wrong; do not implement from it.
> The `conv_L_cache=3` state persists across the whole prefill (causal).
>
> Then the shared `post_attention_layernorm → mlp gate/up → SiLU → down → residual`.
> The Zaya CCA path (`conv_state`, 2-tap) is the closest precedent for the stateful
> depthwise conv.

**Config**: `head_dim` already comes from the parsed config (commit `5a9d1d6c9`), which LFM2
needs (HD=64). Still open: `norm_eps` 1e-5 vs the engine's `EPS=1e-6f` constant.

## 8. VERIFIED — the int4 convention is detected from the data (2026-09-13)

Implemented in `npu_engine_universal.cpp` (commit `c53a80d33`): at startup the engine reads
one 512-byte zero-point block (`model.layers.0.mlp.down_proj.weight`, zeros at `+512` in a
5120-byte int4 row) and selects the decoder. The intent4 convention is NOT in the header —
it is a flat tensor-name -> offset dict with no quant metadata (checked LFM2 149 keys, Qwen3
311, Nanbeige 291) — but the DATA carries it, because the two encodings are two
parameterisations of one affine map (`q*s+zp` over q in [0,15] vs `v*s+zp` over v in
[-8,7]); a SIGNED bundle therefore stores `zp == 0` for every group and an UNSIGNED one
stores a centred zp.

Measured `zp/scale` across all 16 installed `*-NPU2` bundles: **-7.255 to -7.734 for every
non-LFM2 bundle, and exactly 0.000 for both LFM2-1.2B and LFM2-2.6B.** Nothing lies in
between, so one read decides it.

NPU verification (this run, `NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=1024`):

| model | probe line | boot @1024 |
|---|---|---|
| Qwen3-0.6B | `511/512 non-zero -> UNSIGNED` | 25 (unchanged) |
| Qwen3-4B | `511/512 non-zero -> UNSIGNED` | 220 (unchanged) |
| Llama-3.1-8B | `511/512 non-zero -> UNSIGNED` | 220 (unchanged) |
| LFM2-1.2B (@256) | `0/512 non-zero -> SIGNED (two's complement)` | dims only; no native path yet |

So the probe is correct on both classes and the change is a no-op for the four working
families. Ported decoder: commit `69ea8e744` (the dsh agent's
`dequant_i8_group_signed_to_float_ex`, +62/-0, file-wise from their `3bb836eb0`).

### Status after this section (2026-09-13)

LFM2 now loads and runs end to end but produces a WRONG token, announced loudly:

```
[HYBRID] 10/16 layers are gated short-conv layers and the conv block
         is NOT implemented -- the token below WILL BE WRONG.
         Gate for a correct LFM2 forward: boot=5242.
  [0] boot=63260 (7ms)
```

Landed: decoder ported `69ea8e744`; convention auto-detected `c53a80d33`; tensors loaded
`d1bb27f23`; layer-BO sizing fixed (was a SIGSEGV) `e2e65ede4`; short-conv weights packed
`5ea5b5a6f`; block order corrected `a123bb911`. The detector reports 10/16 conv layers,
matching the bundle's layer map exactly.

## 10. The route: FLM ships the kernels (2026-09-13)

LFM2's GEMM shapes do not exist in the engine's own xclbin set. Needed vs present
(`engine/npu/xclbins/`):

| projection | K | N | present in engine |
|---|---|---|---|
| shortconv.in_proj | 2048 | 6144 | no |
| shortconv.out_proj | 2048 | 2048 | no |
| attn q / o | 2048 | 2048 | no |
| attn k / v | 2048 | 512 | `final_i8_G_K2048_N512` / `final_i8_U_K2048_N512` |
| mlp gate / up | 2048 | 8192 | `final_i8_QKV_K2048_N8192` |
| mlp down | 8192 | 2048 | no |

BUT FLM ships a complete per-model xclbin set for LFM2 at
`~/.local/flm-v0946/xclbins/LFM2-1.2B-NPU2/` (and `LFM2-2.6B-NPU2`, plus `.5-Thinking`
and `Transcript` variants):

| file | bytes | role |
|---|---|---|
| `mm.xclbin` | 512220 | the GEMMs, in LFM2's own shapes |
| `attn.xclbin` | 397308 | the hd64 / nh32 attention kernel the engine has no ELF for |
| **`conv.xclbin`** | 118476 | **the gated short-conv block** — no analogue anywhere in the engine |
| `layer.xclbin` | 357756 | possibly a fused per-layer pipeline |
| `dequant.xclbin` | 112524 | int4 dequant |

This is the route, and it is why `NPU_FLM_PREFILL=1` already reproduces boot=5242.
Consequences:

1. A **host-GEMM fallback is NOT needed** for correctness — the device kernels exist.
2. The engine must be pointed at FLM's xclbin ROOT by model name: the model dir under
   `~/.config/flm/models/LFM2-1.2B-NPU2/` holds only config.json, model.q4nx and the
   tokenizer, so the `fmd`/`fxd` derivation from `argv[1]`'s parent (commit `77874d5a7`)
   finds nothing there.
3. Inspect `layer.xclbin` FIRST — if it is a fused per-layer kernel, the dispatch is far
   less work than composing mm + attn + conv by hand.

### 10.1 What the files actually are (checked 2026-09-13)

`layer.xclbin` cannot be introspected by name. `xclbinutil --info` on all five reports
`Kernels: <unknown>` and only HOST/SRAM memory entries, so they are opaque self-contained
NPU instruction containers (the same shape as the engine's own `final_i8_*.xclbin`: header,
`NumOps`, `TransactionSize`, UID) rather than standard kernel xclbins. The interposer
manifest from the capture logs BO allocations and sizes, not kernel roles, so the mapping
from file to role has to come from FLM's own plumbing.

That plumbing is readable from the symbol table, and it settles the design:

```
nm -DC ~/.local/flm-v0946/lib/xrt/liblfm2_npu.so
  lfm2_npu::Impl::forward(int)
  lfm2_npu::Impl::load_weights(Q4NX&)
  lfm2_npu::Impl::load_attn_proj_weights(Q4NX&, int)
  lfm2_npu::Impl::load_conv_proj_weights(Q4NX&, int)   <- the hybrid split, explicit
  lfm2_npu::Impl::fill_kv_cache(...)  get_k_cache/get_v_cache
  lfm2_npu::Impl::get_logits(...)     checkpoint/restore/clear_context
```

So FLM's LFM2 support is a complete per-family class with its own `forward()`, and it
distinguishes conv from attention layers explicitly (`load_conv_proj_weights` vs
`load_attn_proj_weights`), matching the bundle's layer map. The native path is therefore
NOT new math: it is driving FLM's existing LFM2 kernels with the engine's own host
orchestration — exactly the relationship the engine already has with FLM's other
per-family libraries. The performance lever is host scheduling (the per-slot
double-buffering that was worth ~30% on the bf16 prefill path), not re-deriving the conv.
