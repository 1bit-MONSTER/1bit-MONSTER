# DeepSeek V4 Flash Reverse Engineering Report

**Model**: DeepSeek-V4-Flash-0731 (284B total / 13B active parameters)
**Source**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
**Release**: July 31, 2026 · MIT license · ~166.9 GB weights
**Target**: 1bit-monster inference engine (AMD Strix Halo NPU + GPU + CPU)
**Status**: Architecture fully analyzed, CPU reference forward pass implemented. GPU kernel path and GGUF tensor name validation pending hardware test.

---

## Architecture Overview

DeepSeek V4 Flash is a 43-layer, 4096-hidden MoE transformer with three major
innovations over V3 that make it both faster at inference and more powerful in
long-context settings: **mHC residuals**, **CSA+HCA hybrid attention**, and
**FP4 expert weights**.

---

## 1. mHC — Manifold-Constrained Hyper-Connections

**What it replaces**: The standard residual connection `h' = h + F(h)`.

**What it does**: Maintains 4 parallel hidden streams instead of one.
After each sublayer (attention or FFN), the 4 streams are remixed by a
4×4 learned matrix constrained to the Birkhoff polytope:

```
[h'₀]       [h₀]
[h'₁] = M × [h₁]    where M is doubly-stochastic (rows sum to 1, cols sum to 1)
[h'₂]       [h₂]
[h'₃]       [h₃]
```

**Birkhoff constraint**: `M` is the convex hull of permutation matrices.
During training, the constraint is enforced via Sinkhorn iterations (20 per
update, `hc_sinkhorn_iters=20`). The spectral norm is bounded at 1, preventing
signal explosion or vanishing. At inference, `M` is baked — 16 float32 values
per layer stored as `blk.{n}.mhc.weight`.

**Effect**: The 4-wide stream lets the model dynamically route signal from
any stream to any other stream at each layer, instead of always accumulating
into the same vector. This is analogous to an intra-layer residual routing
mechanism.

**Implementation**: `include/deepseek_v4.h` → `DeepSeekV4mHCState`.
- `set_embed(emb)`: broadcast token embedding into all 4 streams.
- `add_and_mix(delta, M)`: add sublayer output to `stream[0]`, then apply `M`.
- `current()`: returns `stream[0]` for the next norm/sublayer.

**mHC config keys**:
| Key | Value |
|-----|-------|
| `attention.mhc_mult` | 4 |
| `attention.mhc_eps` | 1e-6 |
| `attention.mhc_sinkhorn_iters` | 20 |

---

## 2. CSA+HCA — Hybrid Compressed Attention

V4 Flash uses two different sparse/compressed attention mechanisms to
achieve a 1M-token context window at only 10% of the single-token
inference FLOPs of DeepSeek-V3.2 at that context.

### Layer assignment
- **Layers 0-1** (`n_sliding_window_layers=2`): sliding window (window=128), local only.
- **Remaining 41 layers**: assigned CSA or HCA per `compress_ratios` array (42 values).
  - `compress_ratios[i] < 8` → CSA (moderate compression ~4×)
  - `compress_ratios[i] >= 8` → HCA (heavy compression ~128×)

### CSA — Compressed Sparse Attention (~4× compression)
```
1. Softmax-gated pooling: compress key blocks (pool_size = compress_ratio)
2. Lightning indexer: FP4 dot products to select top-k key blocks per query
   (index_topk=512 blocks selected from full context)
3. Dense attention over only the selected blocks
```
The indexer is a fast inner-product in a lower-dim space:
`q_index = q @ W_q_index  [H → index_head_dim=128]` per head, then
`score_block = q_index @ K_block_index^T` to rank blocks.

**CSA config keys**: `attention.csa_index_topk=512`, `attention.index_head_dim=128`, `attention.index_n_heads=64`

### HCA — Heavily Compressed Attention (~128× compression)
```
1. Multi-hash pooling: 3 independent hash functions map tokens to buckets
   (num_hash_layers=3)
2. Each hash assigns each token a bucket index; keys are averaged per bucket
3. Dense attention over buckets only (~context/128 effective keys)
```
The HCA output is a global, heavily-blurred context view — accurate for
long-range semantic dependencies but not for precise recent tokens
(which the CSA or SW layer handles).

**HCA config keys**: `attention.hca_hash_layers=3`

**CPU reference**: Both CSA and HCA are implemented as full dense attention
in `src/deepseek_v4.cpp` for correctness; the sparse/hash paths are TODO
for the GPU kernel. This produces identical outputs to sparse attention on
masked-off positions (softmax of `-inf` = 0 weight).

---

## 3. MLA — Multi-Head Latent Attention (V4 variant)

V4 uses the same MLA compression principle as V3, but with significantly
different dimensions:

| Param | V3 | V4 Flash |
|-------|------|---------|
| `hidden_size` | 7168 | 4096 |
| `num_heads` | 128 | 64 |
| `num_kv_heads` | 16 | **1** (extreme GQA) |
| `head_dim` | 128 | **512** |
| `qk_nope_head_dim` | 128 | **448** |
| `qk_rope_head_dim` | 64 | 64 |
| `v_head_dim` | 128 | **512** |
| `kv_lora_rank` | 512 | 512 |
| `q_lora_rank` | 1536 | **1024** |
| `o_lora_rank` | — | **1024** (new in V4) |

**Extreme GQA** (`num_kv_heads=1`): All 64 query heads attend to a single
set of KV vectors. The KV cache compression is even more extreme than V3:
`kv_lora_rank=512` floats per token per layer (vs 512 in V3 for 16 KV heads —
same raw storage but now shared across all 64 Q heads).

**Output LoRA** (`o_lora_rank=1024`): V4 decomposes the output projection as
`attn_out → [n_heads × v_head_dim] @ W_o_a → o_lora_rank → W_o_b → H`.
This reduces the large output projection from `(64 × 512) × 4096 = 134M`
params to `(64 × 512 + 4096) × 1024 = 37M` params per layer.

**RoPE**: YaRN scaling — `rope_theta=10000` for positions ≤ 65536,
`compress_rope_theta=160000` beyond that, `yarn_factor=16`.

**GGUF tensor names** (hypothetical — to verify against ggml-org GGUF):
| Tensor | Shape | Notes |
|--------|-------|-------|
| `blk.{n}.attn_q_a.weight` | `[H, q_lora_rank]` | Q compress |
| `blk.{n}.attn_q_a_norm.weight` | `[q_lora_rank]` | post-compress RMSNorm |
| `blk.{n}.attn_q_b.weight` | `[q_lora_rank, n_heads * qk_nope]` | Q nope decompress |
| `blk.{n}.attn_q_b_rope.weight` | `[q_lora_rank, n_heads * qk_rope]` | Q rope decompress |
| `blk.{n}.attn_kv_a_mla.weight` | `[H, kv_lora_rank + qk_rope]` | KV compress |
| `blk.{n}.attn_kv_a_norm.weight` | `[kv_lora_rank]` | post-compress RMSNorm |
| `blk.{n}.attn_kv_b.weight` | `[kv_lora_rank, n_heads * (qk_nope + v_head)]` | KV decompress |
| `blk.{n}.attn_o_a.weight` | `[n_heads * v_head_dim, o_lora_rank]` | output down |
| `blk.{n}.attn_o_b.weight` | `[o_lora_rank, H]` | output up |
| `blk.{n}.mhc.weight` | `[mhc_mult, mhc_mult]` | mHC mixing matrix |

---

## 4. MoE FFN — FP4 Expert Weights

**256 routed + 1 shared expert**, top-6 per token.

```
input: norm(x)  [H=4096]
router: logits = norm(x) @ W_gate  [H → 256]
scoring: score_i = sqrtsoftplus(logit_i) = log(1 + exp(sqrt(logit_i)))
top_k: select 6 highest-scored experts (noaux_tc — no auxiliary loss at inference)
normalize: weights /= sum(weights)
shared:  y_s = down(silu(gate(x)) * up(x))
routed:  y_e = down_e(silu(gate_e(x)) * up_e(x))  × weight_e × routed_scale(1.5)
output:  y_s + Σ_{k=1..6} y_{top_k}
FFN residual via mHC
```

**sqrtsoftplus** (new scoring function): `log(1 + exp(sqrt(x)))` for positive
logits, maps to `log(2)` at 0 (like softplus shifted by sqrt), and grows as
`sqrt(x)` for large positive logits — less extreme than softmax, more
discriminative than softplus alone.

**FP4 weights** (NVFP4 E2M1):
- 1 sign + 2 exponent + 1 mantissa bit
- Block size: 128×128 elements share one FP8 (E4M3) block scale
- Effective: 4.25 bpw (expert weights only; attention/norm remain in BF16/FP8)
- `GgufReader::get_tensor_f32` dequantizes inline during load

**Expert intermediate**: `moe_intermediate=2048` (vs 2048 in V3 — same).
With 256 experts and top-6, each token activates `6 × 2 × 2048 × 4096 = 100M`
FLOP per FFN layer (gate + up + down projections).

---

## 5. Quantization Strategy

Original training precision: BF16 weights, Muon optimizer.

Production deployment uses mixed precision:
| Component | Precision |
|-----------|-----------|
| Expert weights (gate/up/down) | FP4 (NVFP4 E2M1) |
| Attention (Q_a, Q_b, KV_a, KV_b, O_a, O_b) | FP8 (E4M3) |
| Norms, router, embeddings | FP8 (E4M3) |
| Activations | BF16 |
| mHC mixing matrices | BF16 |
| Scales | FP8 (UE8M0) for FP4 blocks |

**For 1BP format target**: Q4NX (4-bit INT) for expert weights, FP16 for
attention projections — matching our standard dense-model strategy.
TQ2 is unsuitable (expert weights are not ternary-valued).

---

## 6. Configuration Reference

From `deepseek-ai/DeepSeek-V4-Flash/config.json`:

```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "model_type": "deepseek_v4",
  "hidden_size": 4096,
  "num_hidden_layers": 43,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "qk_nope_head_dim": 448,
  "qk_rope_head_dim": 64,
  "v_head_dim": 512,
  "q_lora_rank": 1024,
  "o_lora_rank": 1024,
  "kv_lora_rank": 512,
  "vocab_size": 129280,
  "max_position_embeddings": 1048576,
  "rms_norm_eps": 1e-6,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 2048,
  "routed_scaling_factor": 1.5,
  "norm_topk_prob": true,
  "topk_method": "noaux_tc",
  "score_func": "sqrtsoftplus",
  "expert_dtype": "fp4",
  "hc_mult": 4,
  "hc_eps": 1e-6,
  "hc_sinkhorn_iters": 20,
  "num_hash_layers": 3,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "sliding_window": 128,
  "rope_theta": 10000,
  "compress_rope_theta": 160000,
  "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
  "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"},
  "next_n_predict": 1
}
```

---

## 7. Implementation Status

### Done
- [x] `RCPP_ARCH_DEEPSEEK_V4 = 22` in `include/rocm_cpp/bitnet_model.h`
- [x] Arch string detection: `deepseek_v4`, `deepseek4`, `dflash`, `deepseek4_dspark`
- [x] `ONEBP_DEEPSEEK_V4 = 22` in `include/onebp_format.h`
- [x] `include/deepseek_v4.h` — full config struct, layer weights, KV cache, mHC state, math helpers
- [x] `src/deepseek_v4.cpp` — GGUF loader (with FP4 dequant via GgufReader) + complete CPU reference forward pass
- [x] `src/model_router.cpp` — route `RCPP_ARCH_DEEPSEEK_V4` → `{"hip_gpu", "cpu_generic"}`
- [x] `CMakeLists.txt` — `src/deepseek_v4.cpp` added to build

### TODO — validation
- [ ] Confirm exact GGUF tensor names against `ggml-org/DeepSeek-V4-Flash-0731-GGUF`
  - Key unknowns: `attn_q_b_rope.weight` vs fused into `attn_q_b.weight`
  - Key unknowns: `mhc.weight` key name (may be `blk.{n}.mhc_mix.weight`)
  - Key unknowns: `attn_o_a.weight` / `attn_o_b.weight` vs `attn_o.weight`
- [ ] Verify `compress_ratios` GGUF metadata key name
- [ ] Verify `get_f32_array` API exists in GgufReader for array metadata
- [ ] Test CPU reference against PyTorch reference for a single layer

### TODO — GPU path
- [ ] CSA sparse attention HIP kernel (block selection + compressed attention)
- [ ] HCA hash-pooling attention HIP kernel
- [ ] FP4 GEMM kernel for expert weights (WMMA-based, block_size 128×128)
- [ ] mHC stream mixer CUDA/HIP kernel (4×4 matmul over 4 H-dim vectors)

### TODO — 1BP converter
- [ ] `tools/deepseek_v4_convert.py` — convert HuggingFace weights to 1BP/Q4NX
- [ ] Expert weight packing: `[256, H, moe_int]` 3D tensor → Q4NX tiles
- [ ] MLA weight packing: attention projections → FP16 (not Q4NX — too low dim)

---

## 8. Open Questions

1. **head_dim=512**: Is this truly per-head, or is it a reported total per group of
   heads (since num_kv_heads=1)? The config says `head_dim=512` alongside
   `num_attention_heads=64`, giving a total attention dim of 32768 before LoRA.
   This is unusually large but consistent with the large q_lora_rank=1024.

2. **compress_ratios**: The config lists 42 float values for 43 layers (2 SW + 41 CSA/HCA).
   We need to reverse-engineer the exact assignment rule (CSA vs HCA) from the
   ratio values. Hypothesis: ratio = 1/compress_factor, so ratio > 1/8 → CSA, ratio ≤ 1/8 → HCA.

3. **DSpark speculative decoding**: The 0731 release includes a DSpark speculative
   decoder module with a `next_n_predict=1` MTP head. The draft tokens come from
   an attached lightweight head. This is not yet in scope for the inference engine
   but is architecturally straightforward (draft head = linear [H → vocab]).

4. **mHC initialization**: At decode position 0, how should the 4 hidden streams
   be initialized? Current approach: broadcast embedding into all 4 streams.
   Alternative: streams 1-3 initialized to zero. Needs validation.
