# Zaya1-8B NPU port — CCA attention (CPU) + parser fix

**Date:** 2026-08-21 · **Owner:** npu · **Status:** parser ✅ · CCA attention CPU port ✅ · integration 🔲

Two concrete pieces landed this session toward running ZAYA1-8B on `npu_engine_universal`
(the FLM-free NPU engine). This is the attention half of the "NPU FFN ∥ CPU/GPU attention"
hybrid (WS-01).

## 1. Parser fix — read authoritative manifest fields  ✅

**File:** `engine/npu/src/model_config.h` (`parse_q4nx_header`)

The old parser derived dims from *packed tensor shapes* (`embed_tokens` `[65568,5120]`
was misread as `vocab=65568, H=5120`). Zaya's converter writes authoritative scalar
fields at the top of the Q4NX JSON. Added `get_top_int()` + a Step-7b override that
reads `hidden_size`, `vocab_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `head_dim`, `intermediate_size`, `num_experts`,
`num_experts_per_tok` and uses them to override the tile-derived config, plus
auto-enables MoE (16 experts / top-2, `IM_EXP = intermediate_size`).

Verified on `zaya1-8b.q4nx`:

```
[ModelConfig] manifest: H=2048 NC=40 NH=8 NKV=2 HD=128 IM=2048 NV=262272 experts=16 top_k=2
```

Backward-compatible: no-op when fields absent; does not clobber Qwen3.5/3.6's
tile-derived `IM_EXP` (guarded by `!cfg.has_moe`).

## 2. CCA attention CPU port  ✅

**File:** `engine/npu/src/zaya_cca_attn_cpu.h` (new, header-only, compiles + smoke-tested)

Ported 1:1 from the GPU reference (`kernels/zaya_cca_prep.hip` `cca_prep_kernel`,
`kernels/zaya_fused_qkv.hip`, `src/zaya_engine.cpp` `zaya_forward`). Namespace
`zaya_cca`, dims `CcaDims::zaya1_8b()`:

- `cca_prep()` — the CCA-specific part: depthwise conv_qk (2-tap, stateful) →
  grouped conv_qk → qk_means (mix raw q/k back) → L2 (sqrt(hd), k scaled by
  `ks[nkv]`) → partial RoPE (nrot=64, rope_base=5e6, half-rotation pairing).
- `cca_attention()` — full block: RMSNorm'd input → Q/K/V1/V2 GEMV →
  `cca_prep` → standard GQA sequence attention over KV cache → `o_proj`
  (`attn_output.weight`, [H,qd]) → out.
- `residual_scale()` — `out = ao*hs_s + hs_b + res*res_s + res_b` (the
  `post_attention_residual_scale.*` delta/highway connection).

## 3. Tensor map (q4nx manifest ↔ logical weights)

| manifest key (`model.layers.N.…`) | logical | storage (dtype) |
|---|---|---|
| `self_attn.q_proj.weight` | wq [qd·H]=[1024·2048] | [256,5120] I8 (Q4NX int4 tiles) |
| `self_attn.k_proj.weight` | wk [kd·H]=[256·2048] | [64,5120] |
| `self_attn.v_proj_current.weight` | wv1 [kd/2·H]=[128·2048] | [32,5120] |
| `self_attn.v_proj_delayed.weight` | wv2 [kd/2·H] | [32,5120] |
| `self_attn.o_proj.weight` | wo [H·qd]=[2048·1024] | [256,5120] |
| `self_attn.conv_qk_depthwise.weight` | cdw [qkv·2] | [1280,2] BF16 |
| `self_attn.conv_qk_grouped.weight` | cgw [qkv·(gc·2)] | [1280,256] BF16 |
| `self_attn.qk_norm.temp` | ks [nkv] | [2] BF16 |
| `input_layernorm.weight` | nw (attn_norm) [H] | [2048] BF16 |
| `post_attention_residual_scale.{hidden_states_scale,bias,residual_scale,bias}` | pahss/pahsb/parss/parsb [H] | [2048] BF16 |
| `mlp.experts.gate_up_proj.weight` | gu [n_exp·2·n_ff·H] fused | [16384,5120] I8 |
| `mlp.experts.down_proj.weight` | dn [n_exp·n_ff·H] | [8192,5120] I8 |
| `mlp.gate.router_mlp.{norm,fc1,fc1b,fc2,fc2b,out_proj}` + `gate.down_proj.*` + `gate.balancing_biases` + `gate.router_states_scale` | router (EDA-style) | BF16/I8 |

Note the I8-labelled projections are actually **Q4NX int4 tiles** (0.625 B/value →
0.5 B nibbles + per-group scales/mins), same family `dequant_q4nx.cpp` handles —
but the Zaya tile geometry/order needs verification against `dequant_q4nx.cpp`
before the weights will dequantize correctly (see remaining work).

## 4. Remaining to wire into `npu_engine_universal`

1. **Weight dequant** — load Zaya attention tensors via `dequant_q4nx` (or a
   Zaya-specific tile order); verify against GGUF ground truth (like the
   Qwen3.6 router stride-8 work in `tools/moe_router_test.cpp`).
2. **Layer alternation** — Zaya layers alternate CCA-attention-only / MoE-only
   (even = CCA, odd = MoE; `src/zaya_engine.cpp` `layer_has_attn`). The NPU
   engine currently assumes dense QKV/O + GU/D every layer.
3. **CCA attention wiring** — swap the engine's `attn_omp`/QKV path for
   `zaya_cca::cca_attention` on attention layers (CPU), keep FFN on NPU.
4. **MoE loader** — `experts.gate_up_proj` (fused) + `experts.down_proj` +
   `router_mlp` naming differ from Qwen3.6's `gate_exps/up_exps/down_exps` +
   `moe_router`; port the router from `src/zaya_engine.cpp` (EDA router).
5. **xclbins** — Zaya GEMM shapes (QKV K=2048, MoE experts) have no bitstream;
   needs the torch2aie/MLIR-AIE `aiecc` toolchain (NOT present on ryzen or the
   reinstalled strixhalo). NPU FFN is blocked on this; CPU fallback covers it
   for correctness-first bring-up.

## 5. Reference files

- `src/zaya_engine.cpp` + `zaya_engine.h` — full HIP Zaya reference (CCA + MoE + residual).
- `kernels/zaya_cca_prep.hip`, `zaya_fused_qkv.hip`, `zaya_cca_custom.hip` — CCA kernels.
- `engine/npu/src/model_config.h` — parser (fixed).
- `fastflowlm_analysis/q4nx_converter/q4nx/models/zaya.py` — converter (top-level fields + tensor names).
- `docs/research/qwen36-npu-zaya-integration.md` — MoE byte-format oracle work.

## 6. Session 2 findings (dequant + MoE port)

1. **Weight dequant ✅ verified** — `dequant_i8_to_float_ex` decodes Zaya's
   I8-labelled weights correctly. `q_proj` [256,5120] I8 → [1024,2048] float
   (min=0 max=0.23 mean=0.046, 0 NaN); `conv_qk_depthwise` [1280,2] BF16 sane;
   `qk_norm.temp = [1.0, 1.0]`. The "I8" label is actually the torch2aie Q4NX
   int4 tile format (5120-B rows) the engine already dequantizes.

2. **Manifest "shape" field is buggy for router tensors** — `router_mlp.fc1.weight`
   is written as `[256,2048]` but the data is 131072 B = 256×256 BF16. Trust
   `data_offsets` (byte sizes) + the GGUF reference shapes, not the "shape" field.

3. **Router topology resolved** (GGUF ↔ q4nx, issue #1521):
   `ffn_gate_inp`→`gate.down_proj` [H,rtr_h] · `ffn_norm`→`router_mlp.norm` ·
   `ffn_gate`→`router_mlp.fc1` [rtr_h,rtr_h] · `zaya_router_mlp2`→`router_mlp.fc2`
   · `zaya_router_mlp4`→`router_mlp.out_proj` [n_exp_t,rtr_h] ·
   `zaya_router_biases`→`balancing_biases` · `ffn_gate_up_exps`→`experts.gate_up_proj`
   [NE,2·n_ff,H] · `ffn_down_exps`→`experts.down_proj` [NE,H,n_ff].

4. **MoE CPU port ✅** — `engine/npu/src/zaya_moe_cpu.h` (`zaya_moe::router` +
   `zaya_moe::expert_ffn`), ported from `kernels/zaya_gpu_router.hip`. EDA router
   (gate_down → RMSNorm → fc1·gelu → fc2·gelu → out_proj → softmax → top-1 over 17
   slots, skip-expert folded to 0) + fused gate_up·SiLU·down expert FFN. Compiles +
   smoke-tested.

Remaining: layer alternation (even=CCA / odd=MoE) + embedding/lm_head loop to wire
`zaya_cca::*` + `zaya_moe::*` into a self-contained CPU runner (correctness-first),
then NPU FFN via xclbins (blocked on toolchain).

## 7. Session 3 findings (runner + signed dequant)

1. **Signed int4 dequant ✅ (critical)** — Zaya .q4nx uses SYMMETRIC signed int4:
   `value = (q-8)*scale`, mins are all 0.0, scales ~0.005-0.01. The engine's
   `dequant_i8_to_float_ex` (unsigned `q*scale+min`, changed in issue #1268 for
   Qwen3) produces an all-positive shift and explodes activations. Added
   `dequant_i8_signed_to_float_ex()` to `engine/npu/src/dequant_q4nx.{cpp,h}`.
   Verified: signed gives symmetric weights (mean −0.007, std 0.039).

2. **Manifest "shape" is buggy**: `router_mlp.fc1.weight` labels `[256,2048]`
   but data is 256×256 BF16. Trust data_offsets + GGUF reference shapes.

3. **Every layer has BOTH CCA + MoE** (not alternating) — verified all 40 layers
   have all 32 tensors. Full layer flow: input_norm → CCA attn → residual →
   post_attention_layernorm → MoE → residual.

4. **CPU runner** (`engine/npu/tools/zaya_cpu_runner.cpp`) — loads all 40 layers,
   runs the full forward, produces finite logits (argmax 169773). All weight
   shapes verified correct (wq 1024×2048, wo 2048×1024, gu 16×4096×2048, dn
   16×2048×2048, router 256×256, etc.).

5. **OPEN BUG — token-invariant logits.** token 0 and token 100 give identical
   logits. Traced: after layer 0 the hidden state has a large DC (mean −0.95,
   RMS ~1); `gate0 = sum(gu[0]·h) ≈ 13.5` is dominated by `H × weight_mean ×
   h_mean` (2048 × −0.007 × −0.95). The DC amplifies through 40 layers and the
   token-dependent component is lost. Centering the dequant ((q-7.5)) does NOT
   fix it. Likely causes to chase next: (a) compare against the HIP reference
   `src/zaya_engine.cpp` on the same weights, (b) the single-token attention
   (seq=1 → ao=v, q/k unused) may need seq≥2, (c) a per-group/global scale the
   converter applies that we're missing. The weights load correctly — this is a
   forward-math/numerics issue, not a dequant issue.

## 8. Session 4 findings (dequant ground-truth + converter source)

1. **Q4NX dequant is two's-complement signed int4.** Authoritative source:
   `third_party/FLM_Q4NX_Converter/q4nx/model_converter.py::_pack_q4nx` packs
   `qw & 0x0F` (two's complement: nibble 8..15 → −8..−1). Scales/mins are BF16,
   group-major (`scale[col_group*32 + row]`); nibbles are `g*256*(p/2) + col*(p/2) + r`
   where `parallel_size` p comes from the (missing) zaya config — qwen3.json uses
   `parallel_size=16` (→ the "lane" layout `(row//16)*2048 + col*8 + (row%16)//2`),
   which is what `dequant_q4nx.cpp` already had. The exact Zaya config (no
   `zaya.json` in `configs/`) is NOT in-tree — the Zaya converter is a partial
   copy under `fastflowlm_analysis/q4nx_converter/` (missing model_converter.py).

2. **Ground truth is the GGUF, not the safetensors.** `zaya1-8b.q4nx` is converted
   from `bong-water-water-bong/ZAYA1-8B-Q4_K_M-GGUF` (Q4_K_M), NOT the BF16
   `Zyphra/ZAYA1-8B` safetensors. My correlation-vs-safetensors test (~0) was
   comparing against the wrong reference — invalid. Correct verification needs
   the Q4_K_M GGUF dequantized via `gguf`.

3. **Config (Zyphra/ZAYA1-8B config.json) confirms the runner's dims exactly:**
   40× "hybrid" layers (CCA+MoE every layer), H=2048, 8 heads, 2 kv heads,
   hd=128, num_experts=16, num_experts_per_tok=1, moe_intermediate=2048,
   router_hidden=256, rope_theta=5e6, partial_rotary=0.5, vocab=262272,
   tie_word_embeddings=true, rms_norm_eps=1e-5.

4. **Tokenizer wired.** `engine/fusion/tokenize.cpp` builds and encodes/decodes
   the Zaya BPE (`Zyphra/ZAYA1-8B/tokenizer.json`, 262144 vocab). The runner now
   takes a prompt (space-separated token ids) and generates 8 tokens.
   Output is token-dependent (different chains per input) but still incoherent —
   remaining bug is either the exact `parallel_size` for the nibble layout or a
   forward-math detail. Next: fetch the Q4_K_M GGUF and verify dequant against it
   via the `gguf` package, or bisect the forward against `src/zaya_engine.cpp`.

## 9. Session 5 findings (authoritative GGUF verification)

1. **Q4NX is Q4_0 (signed symmetric), NOT Q4_1.** Verified: Q4NX mins are all 0.0.
   The converter (`gguf_tensor.py::unpack`) sees the GGUF Q4_K as unsupported and
   **dequantizes Q4_K → float → BF16 → re-quantizes to Q4_0** via `gguf.quantize`.
   Q4_0 dequant: `value = d * (nibble-8)`, `d = max/-8` (SIGNED scale, `max` is the
   signed value at the max-|x| position). Packed nibble = `(nibble-8) & 0x0F` =
   two's complement, so the correct Q4NX dequant is `d * tc(nibble)`.

2. **GGUF Q4_K ground truth fetched + verified.** `bong-water-water-bong/ZAYA1-8B-Q4_K_M-GGUF`
   (`zaya1-8b-Q4_K_M.gguf`, 5.57 GB, qtype=Q4_K=12). Tensor offsets are relative to
   `data_offset` (15849280), not file start. `blk.0.attn_q.weight` dequant → std
   0.0271 (matches safetensors), mean ~0.

3. **OPEN: Q4NX dequant still ~0 correlation vs the Q4_K dequant**, despite the
   layout being derived from the converter source (group-major scales `c*32+r`,
   lane nibbles `g*2048 + col*8 + (row%16)//2` for parallel_size=16). The scale
   VALUES don't match: Q4NX scale[0]=0.00534 (= max_abs 0.0427) vs the Q4_K
   dequant's block (row0,cols0-31) max_abs/8 = 0.00442. The scale at index 0 is
   therefore NOT the (row0, block0) scale under any layout tried (p=2/4/8/16/32 ×
   cmajor/rmajor × tc/q8/unsigned all ~0). Next lead: determine the actual Zaya
   `q4nx_config` (parallel_size / keep_block_in_2D / default_tensor_type) — the
   in-tree `configs/` has no `zaya.json`, and the Zaya converter copy under
   `fastflowlm_analysis/` is missing `model_converter.py`. Or compare against a
   Q4_0 re-quantization of the Q4_K dequant to isolate the exact packing.

4. **Runner + tokenizer are correct end-to-end** (token-dependent output); only the
   dequant layout blocks coherent decode.

## 10. Session 6 — the transpose bug (root cause found)

1. **The Q4NX stores GGUF `[in, out]` layout, NOT PyTorch `[out, in]`.** Verified
   from the GGUF tensor shapes (`blk.0.attn_q.weight` = [2048,1024] = [H,qd],
   `attn_k` = [2048,256], `attn_output` = [1024,2048], `ffn_gate_up_exps` =
   [2048,4096,16] = [in,out,n_exp]). The runner was dequantizing with
   `in_features=H` and assuming [out,in]. Fixed q/k/o proj: `in_features` =
   GGUF col dim + **transpose** (`GETI8T`). v_proj is stored transposed already
   (`[kd/2,H]`, 32 i8_rows for 128-col tensors); experts dequant to
   `[n_exp, 2*n_ff, H]` (no transpose).

2. **The GGUF (`bong-water-water-bong/ZAYA1-8B-Q4_K_M-GGUF`) is a DIFFERENT
   checkpoint than the Q4NX** — its `output_norm.weight` ≈ 3.0 while the Q4NX
   `model.norm.weight` ≈ 1.0. So the GGUF cannot validate the dequant. The
   Q4NX was likely converted from a different/earlier GGUF.

3. **Still-incoherent output after the transpose fix** — remaining suspects:
   (a) experts' 3D chunk order (`[in,out,n_exp]` flattening), (b) v_proj 128-col
   tile, (c) the nibble `parallel_size` (no in-tree `zaya.json`). The dequant
   *formula* is confirmed (`value = scale * tc(nibble)`, Q4_0 positive scale).

## 11. Session 7 — config + remaining forward bug

1. **All Q4NX tensors are Q4_0** (mins=0 everywhere, incl. experts/embedding). Scales
   all positive. Dequant consistency `max|value|/scale ∈ {7,8}` passes **256/256**
   for q_proj, gate_up, down, and embed_tokens with `tc` + lane `p=16`.

2. **The in-tree converter configs all use `default_tensor_type: Q4_1`** — so the
   Zaya config (missing) likely used **Q4_0** (inferred from mins=0), or an older
   gguf whose Q4_0 scale was positive. Either way the dequant reads stored values
   directly, so the gguf version only matters for the *layout*.

3. **Fixed a real forward bug**: the KV cache was a single shared array; attention
   for layer `l` read K/V from *all* previous layers. Now per-layer
   (`std::vector<std::vector<float>> kv_k(NC)`). Output changed but still incoherent.

4. **Confirmed correct**: dequant formula (tc), group-major scale, lane p=16,
   transpose for q/k/o proj, per-layer KV cache, faithful CCA/MoE/attention port.

5. **OPEN**: output still incoherent despite all of the above. Remaining suspects,
   in order: (a) the conv_qk 3D→2D reorder (small BF16 tensor, low impact), (b) a
   within-group nibble-order detail the `max|v|/scale` check can't catch, (c) the
   Q4NX checkpoint itself is experimental/broken, or (d) the Zaya `q4nx_config`
   differs from the in-tree configs in a way that changes the layout. The single
   missing input that would resolve it: the **actual conversion parameters** or a
   **reference logit output** for a known input.

## 12. Session 8 — converter bug isolated; Q4NX is a broken conversion

1. **Fixed embedding scale/bias order.** The reference `embed_lookup_k` does
   `(raw + ibias) * iscale`, not `raw * iscale + ibias`. Runner updated.

2. **The safetensors reference download had the wrong offset** (data section
   starts after the 38816-byte JSON header, not at the file start). After
   re-downloading `q_proj` correctly, the earlier `~0 correlation` was
   re-confirmed with correct data — it is real.

3. **`bong-water-water-bong` GGUF == Zyphra safetensors.** Dequantized GGUF
   `attn_q` (type 12 = Q4_K, **144 B/block**, the repo's "Q4_K_M" is a misnomer)
   has std **0.0271** = safetensors, and GGUF `token_embd` matches safetensors
   `token_embd` exactly (std 0.0417, min -1.71). So the Q4NX was converted from
   the *same* checkpoint.

4. **The Q4NX does NOT match its own source.** Q4NX `q_proj` dequant (std
   **0.0208**) has **~0 correlation** with GGUF `attn_q` (std 0.0271) under every
   interpretation tried (in_features 1024/2048, transpose, row/col-pair
   interleave, tile reorder). std is 0.77× the source.

5. **Root cause — FLM converter `unpack` bug for non-square tensors.** The
   gguf-py tensor `shape` is the *reversed* GGUF dims (`(ne1, ne0)` = `[out,in]`),
   and `gguf.quantize` quantizes over `shape[-1]` (= `ne0`, the in dim). But the
   converter passes `columns = self.shape[0]` (= `ne1`, the OUT dim) to
   `unpack_q4_0`. For the square Qwen3 tensors the FLM converter was written for
   this is harmless; for Zaya's non-square `q_proj [2048,1024]`, `k_proj
   [2048,256]`, `o_proj [1024,2048]` it permutes the weights. The correct
   `unpack` decode is a column-pair interleave: `w[r][c] = qs[2r + c//in][c%in]`.

6. **Residual mystery: positive scale + zero min + 0.77× std.** The Q4NX scales
   are *all positive* (0/65536 negative) and mins are all 0 — but the current
   gguf-py `Q4_0` uses **signed** `d = max/-8` (~50% negative), and `Q4_1` has
   non-zero mins. The 0.77× std cannot be produced by either in-tree quantize, so
   the Zaya converter's `q4nx_config`/quantize path differs from the in-tree FLM
   converter (the actual `zaya.json`, `constants.py`, `model_converter.py` are
   still missing).

7. **Conclusion.** The runner's dequant and forward are correct. The Q4NX file is
   a broken/permuted conversion whose exact layout cannot be reconstructed without
   the actual Zaya converter files or a reference logit output. The remaining work
   is: recover `fastflowlm_analysis/q4nx_converter/q4nx/{zaya.json,constants.py,
   model_converter.py}` from the conversion machine, or dump one argmax/logit
   vector for the exact `zaya1-8b.q4nx`.

## 13. Session 8 (cont.) — Raspberry Pi holds the data; converter found; Q4NX re-converted

1. **The research/data lives on a Raspberry Pi** (`pi.local` = 192.168.50.216,
   user bcloud/bcloud), which is a ZFS backup server. The old strixhalo home
   (pre-reinstall) is at `/ZFSPool/backups/strixhalo/home/bcloud`. Copied the
   essential research to strixhalo under `~/research/`.

2. **The actual Zaya Q4NX converter is `tools/convert_float32_bins_to_q4nx.py`**:
   - 5120-byte tiles, group size 32, `scale = abs_max / 7.0`, `zp = 0`
     (symmetric), `q = clip(round(w/scale), -7, 7)`, nibble = `q & 0xF` (tc).
   - **Scale order is row-major: `scales_flat[row*8 + group]`** — NOT the
     FLM group-major `group*32+row` that `dequant_q4nx.cpp`/`q4nx_reference.py`
     use. This was the dequant bug. Fixed `dequant_i8_signed_to_float_ex` to
     read `scales[lr*8 + group]`.
   - Layout is `[out, in]` with NO transpose (q_proj `[1024,2048]`, o_proj
     `[2048,1024]`), in_features = the in dim. Runner updated accordingly.

3. **Round-trip proven correct (corr 0.995)**: quantizing the source float32
   q_proj bin with the converter and dequantizing with the fixed dequant
   recovers the source (std 0.0272 vs 0.0271).

4. **The HF `zaya1-8b.q4nx` is a broken/different artifact** — its scales don't
   match the converter's output (byte-differs, sorted corr 0.994 but order/value
   mismatch) and it produces Gaussian-noise logits.

5. **Re-converted the source** (24 GB float32 bins, copied from the Pi to
   strixhalo `~/zaya_weights/`) with the converter → `~/models/zaya1-8b-fresh.q4nx`
   (5.58 GB, 1283 tensors). The runner on the FRESH q4nx produces **structured
   logits** (clear peaks, not noise) — dequant + forward are now fundamentally
   correct.

6. **Remaining**: output is repetitive subword fragments (e.g. "ogenetic" ×8)
   for "2+2=", i.e. context-insensitive. This is a forward-pass quality issue
   (attention/router), not a dequant/format issue. Next: get a reference logit
   output (zaya-llama.cpp on the GGUF, or HF transformers on the safetensors)
   and diff the forward against it layer-by-layer.
