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
