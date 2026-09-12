# ws13-arch-gap-closure — Run the architectures the census flags (DeepSeek V4/V4.1, Mamba-3)

**Status:** 🔲 scoped, not started — scope + evidence: this file and `Testing/arch-gaps.md`
**Owner:** engine (CPU/attention + SSM lanes)
**Papers:** none specific — the oracle is the vendors' own reference code (`HF modeling_deepseek_v4.py 5.14`, already the oracle for the existing V4 path) and the model configs themselves.

## Goal

Turn the census's `!! SIGNIFICANT` arrivals from *flagged* into *runnable*: complete the DeepSeek V4 attention machinery, add the V4.1 deltas, and add a Mamba-3 SSM path — each with a reference-oracle gate and a measured "runs the real checkpoint" line. Nothing here is an alias; the evidence that these are new architectures (not renames) is in `Testing/arch-gaps.md`.

## Measured state (2026-09-12, from the published configs + `model.safetensors.index.json`)

**What the engine has today** (`include/deepseek_v4.h` + `src/deepseek_v4.cpp`, 552 lines, CPU f32, HF safetensors names):

| implemented | not implemented |
|---|---|
| mHC streams + Sinkhorn-Knopp (`hc_mult`, 20 iters) + final `hc_head` | **per-layer compressors** — CSA ratio 4 / HCA ratio 128 |
| shared-KV MQA (1 KV head, K=V), `q_a/q_a_norm/q_b` (unweighted) / `kv_proj/kv_norm`, partial RoPE on the last `qk_rope_head_dim`, per-head attention sinks, grouped `o_a`/`o_b` | **Lightning Indexer** (block selection) |
| MoE: sqrtsoftplus routing, `e_score_correction_bias`, `tid2eid` hash routing (`num_hash_layers`), fused `experts.gate_up_proj` + `down_proj`, shared SwiGLU + `swiglu_limit` | **GGUF `blk.*` aliases** ("NOT handled by this loader yet") |

The existing path was validated on a **mini gate with `compress_ratios == 0`** (sliding layers only) — i.e. against a *shape*, not a checkpoint.

**What the real checkpoints actually contain** (tensor-inventory diff of the two official indexes):

| | `deepseek-ai/DeepSeek-V4-Flash` | `deepseek-ai/DeepSeek-V4.1-Flash` |
|---|---|---|
| shape | 43 layers, H=4096, 64 heads/1 kv, 256 experts, moe_int 2048, 3 hash layers | 40 layers, H=**5120**, 384 experts, moe_int **2304**, **no hash layers** |
| compress_ratios | nonzero **41/44**, values {0, 4, 128} | nonzero 38/43, values {0, 1, 2} — but only **4** layers carry compressor/`indexer.k_norm` tensors and **8** carry indexer tensors |
| tensors / size | 69,187 / **159.6 GB** | 96,085 / **510.3 GB** |
| quantization | fp8 e4m3, `ue8m0` block scales, block [128,128] | fp8 + `expert_dtype: fp4`, block **[32,32]** |
| V4.1-only module families | — | `attn.indexer.k_norm/wk`; **`engram.*`** (`embed.weight/scale`, `q_weight`, `k_weight`, `wkv.weight/scale`, 2 layers); `ffn.gate.bias_vl` (all 40); a rewritten **MTP** stack (`main_norm`/`main_proj`/`markov_head`/`confidence_head` vs V4's `e_proj`/`e_norm`/`h_proj`/`h_norm`); **vision tower** (`vision.blocks.*`, `patch_embed`, `norm`) + `aligner.wN` + image tokens |
| dropped vs V4 | — | `compressor.ape`, `ffn.gate.tidNeid` (hash routing gone), V4-style MTP projections |

**The scale is the real constraint.** The loader materializes every tensor as **f32** (`get_tensor_f32`). A real V4.1 Flash is 510 GB in *fp8* — as f32 that is ~2 TB, so "add the architecture" without a quantized/streamed weight path is dead on arrival. That is not new work for this workstream to invent: expert staging is **WS-07** (PagedWeight/DraftExpert) and the tier stream is **WS-11** (NVMe→DRAM→SRAM), with **WS-05** supplying a 1BP v2 format. This workstream supplies the *architecture*, and takes the loading constraint as an input.

## Theory

The V4 attention stack is one design with three layer flavours, and V4.1 keeps it: **sliding** (what we have), and two **compressed-sparse** branches that pool the KV stream through a learned gate (`compressor.wgate`/`wkv`(+`norm`, `ape` in V4)) and then attend over a *selected* subset chosen by the **Lightning Indexer** (an auxiliary low-dim query/key score with its own norm and `wq_b`). The compressor is a pooling operator, so its cost is dominated by the *selection* math in fp32 — cheap on CPU relative to experts.

Consequences that shape the plan:

1. **One implementation serves both families.** V4 needs compressors on 41/44 layers, V4.1 on ~4/40 — the same code with different ratios and a candidate-block variant on top. So do the compressed-attention work once and both checkpoints move.
2. **Config is already shape-agnostic** (`DeepSeekV4Config` reads everything from JSON); the risk is in *assumptions* baked into the loops (head counts, `o_groups`, expert tensors), not in the constants.
3. **V4.1 is a superset, not a variant.** Engram layers, `gate.bias_vl`, candidate blocks and the MTP rewrite are additive; hash routing is *removed*, so a V4-only path is not a correct V4.1 path even where the names match.
4. **Mamba-3 is a different sub-system** (SSM recurrence), so it is a separate lane inside this workstream, not a variant of lane A.

## Tasks

### P0 — instrument first (nothing below is claimed without it)
- [ ] **P0.1 tiny-config oracle.** Build a 2–4 layer V4.1-shaped reference (config fields from the official `text_config`, `num_hidden_layers=2`, `n_routed_experts=8`, H=256, head_dim=64) and dump per-layer weights + activations from the HF reference implementation. Gate: the **existing** modules (mHC, shared-KV MQA, MoE, sinks) reproduce it to ≤1e-6 max|Δ| per layer — i.e. the instrument is proven before it is used to prove new code. The pattern already exists in `Testing/e2e_gen_check_hf.py` (real HF model + torch fp32, no mocks); why it cannot be used as-is: that harness loads a *whole* checkpoint in fp32, which the 510 GB fp8 V4.1 makes impossible — hence a tiny config plus per-layer dumps.
- [ ] **P0.2 write the maths spec** for `compressor` + `indexer` from the reference code (no new code): pooling, gate, `ape`, block selection, how `topk`/`index_topk` interact, and the exact tensor-name map for both repositories.
- [ ] **P0.3 fix the runtime target in writing**: which format a real checkpoint is served from (fp8 direct? GGUF? Q4NX/1BP?) and the memory ceiling it implies. Inputs to WS-05/WS-07/WS-11; output is a decision, not code.
- [ ] **P0.4 Mamba-3 lane 0**: a tiny-config `Qwen3Mamba3` oracle (same instrument pattern) + the parameter list read from the config (`mamba_mimo_rank`, `mamba_bc_norm_eps`, `mamba_bc_bias_init`, `mamba_rope_fraction`, `mamba_a_floor`, `mamba_chunk_size`, `mamba_d_head/d_state`, `mamba_expand`, `mamba_n_groups/n_heads`, `mamba_dt_*`).

### P1 — the compressed-sparse attention path (unblocks real V4 *and* most of V4.1)
- [ ] **P1.1** implement the compressor (CSA ratio 4 + HCA ratio 128) and the Lightning Indexer; gate per-layer against P0.1 on a config with `compress_ratios ≠ 0`.
- [ ] **P1.2** make the loader shape-agnostic (40/43 layers, H=4096/5120, 256/384 experts, `o_groups`) and add the GGUF `blk.*` alias map.
- [ ] **P1.3** run a **real** V4-Flash checkpoint end-to-end (from whatever P0.3 chose) and measure: coherent continuation, first-N-token identity vs the independent GGUF/llama.cpp reference (`Testing/e2e_deepseek.cpp` is the existing GGUF e2e-gate pattern for the V2/MLA path), decode tok/s on strixhalo.

### P2 — V4.1 deltas, then Mamba-3
- [ ] **P2.1** `engram` layers (2), `ffn.gate.bias_vl`, candidate-block selection, and the MTP rewrite; shape 40/H5120/384/2304; **no** hash routing.
- [ ] **P2.2** vision tower + `aligner` — scope decision first (text-only serving is a legitimate, statable scope; claiming multimodal support is not).
- [ ] **P2.3** **Mamba-3**: extend the SSM path (`src/mamba2_kernels.{cpp,h,hip}`, `src/backend_mamba1.cpp`, the hybrid engines `falconh1_engine.cpp` / `nemotron_h_engine.cpp` / `granitemoehybrid_engine.cpp`) for MIMO rank, BC normalisation, the rope fraction and `a_floor`; gate against P0.4.

## Validation

- **Instrument rule**: every phase ships with the harness that could refute it (WS-00 pattern). No phase is done on an argument.
- Numbers to hit (honesty tags in the WS-00 style):
  - P0.1/P0.4: tiny-config per-layer **max|Δ| ≤ 1e-6** — `validated`, with the dump committed under `FINDINGS/`.
  - P1.1: same bound on a `compress_ratios ≠ 0` config — the new code's *own* gate.
  - P1.3: real checkpoint → **first-8-token identity** with the independent GGUF/llama.cpp reference; then tok/s on strixhalo, tagged `optimized` only after the identity line passes.
  - P2.3: Mamba-3 tiny-config per-layer bound, then a real hybrid checkpoint.
- **Not claimed** by this workstream: 100% HF coverage, multimodal (P2.2 is a scope decision), any speed claim before P1.3's identity gate.

## Notes

- Evidence for "not an alias": `Testing/arch-gaps.md` (config diffs for `deepseekv41`, `qwen3mamba3`, `molmoact2`, `cosmos3edge`), which lands with PR #2244; the census (`Testing/hf_new_models.py`, routed by `scripts/census-watch.sh`) is the detector that keeps this list honest.
- Reuse, do not duplicate: **WS-07** expert staging/PagedWeight, **WS-11** NVMe→DRAM→SRAM tiering, **WS-05** 1BP v2, **WS-08** compressed-KV machinery (different mechanism — MLA latent vs CSA/HCA pooling — but the same `codec_gauge_probe.py` style applies).
- Known traps to expect: `ue8m0` block-scale conventions; per-head attention sinks; top-k ties at 384 experts; candidate-block index arithmetic; `ape` present in V4 and absent in V4.1; MTP tensors that must **not** be loaded into the main stack.
- `molmoact2` (VLA, action expert) and `cosmos3edge` (video world model) are deliberately **out** of this workstream: neither is a text-decoder job, and both need a scope decision before code.
