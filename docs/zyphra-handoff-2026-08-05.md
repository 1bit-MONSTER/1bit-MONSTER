# Zyphra Family on Strix Halo — Handoff (2026-08-05)

Session outcome + open issues for the next session. Everything below was verified
on this box (Ryzen AI MAX+ 395, Radeon 8060S/gfx1151, 128GB UMA, RADV Vulkan, ROCm TheRock).

## 1. Validated performance (measured this session)

| Model | Runtime | tok/s | Correctness vs HF |
|---|---|---|---|
| Zamba2-2.7B-Instruct-v2 | HIP e2e (`bench_zamba2_e2e`) | **12.5** | logits corr **0.997**, 20/20 top-20 |
| Zamba2-7B-Instruct-v2 | HIP e2e | **4.5** | corr **0.992** |
| Zamba2-1.2B-Instruct-v2 | HIP e2e | ⚠️ see Issue 1 | mamba layers match CPU; e2e hangs |
| ZR1-1.5B | llama.cpp Vulkan (`/tmp/zamba-cpp`) | **187.9** tg / **1936** pp | — |
| BlackMamba-1.5B | `test_mamba1_backend` HIP | **79.3** | — |
| BlackMamba-2.8B | `test_mamba1_backend` HIP | **46.2** | — |
| ZAYA1-8B | — | see Issue 2 | converter mismatch |
| ZUNA1.1 | — | see Issue 3 | custom arch, no path |
| Zamba2-2.7B CPU ref | `run_zamba2` | 0.6 | — |

Artifacts: `models/Zamba2-{1.2B,7B}-Instruct-v2.gguf`, `models/Zyphra_ZR1-1.5B-Q4_K_M.gguf`,
`models/blackmamba-{1.5b,2.8b}.gguf`, `models/ZAYA1-8B.gguf` (broken, see Issue 2).
The 2.7B GGUF was regenerated at `/tmp/zyphra/z2-27b-fixed.gguf` (tmpfs, may be gone).

## 2. Open issues (do these next)

### Issue 1 — Zamba2-1.2B e2e hang at layer 20 (d_state=128)
- **Symptom**: `bench_zamba2_e2e models/Zamba2-1.2B-Instruct-v2.gguf 2 1` hangs in the
  first forward after L19; GPU idle, CPU spins on the stream sync. Variable in early
  runs (L8/L16), now consistent at L20. Debug: `Z2_DEBUG_HIP=1` prints L0–L19 then stalls.
- **What works**: chunked scan (committed 50b62757d) makes L0–L19 bit-close to CPU
  (`[hip-debug] L0 = 0.0192 …` == CPU). d_state=128 is handled via a 64-thread chunk loop.
- **Suspects** (in order): a race in the chunked scan state writeback vs the next-hd
  reload (only mutable per-layer state is `ssm_states`/`conv_states`); or the conv
  kernel with conv_dim=4352; or an async-only hang (the `Z2_DEBUG_HIP` hipMemcpy syncs
  per layer, so the hang is in the layer's kernels, not the queue).
- **Repro commands**: see Symptom; also `Z2_DEBUG_HIP=1` for per-layer.
- **Debug hooks already in place** (env-gated, committed): `Z2_DEBUG_HIP` (per-layer
  hidden), `Z2_DEBUG_HYBRID_GPU` (hybrid stages), `Z2_DEBUG_MAMBA` (tuned-block internals).
- **Files**: `src/mamba2_kernels.hip` (chunked scan), `src/zamba2_engine_hip.hip` (loop).

### Issue 2 — ZAYA1-8B converter source keys mismatch the real checkpoint
- `tools/convert_zaya_safetensors_to_gguf.py` targets `zaya_block.*` / `self_attn.qkv.*`
  but the real `Zyphra/ZAYA1-8B` checkpoint uses:
  - `model.layers.{i}.self_attn.qkv_proj.{q_proj,k_proj,v_proj_current,v_proj_delayed,conv_qk_depthwise,conv_qk_grouped,temp}`
  - `model.layers.{i}.self_attn.qk_norm.temp`
  - `model.layers.{i}.mlp.experts.{down_proj,gate_up_proj}` — **stacked** `[8 experts, …]` (1 tensor each, needs split)
  - `model.layers.{i}.mlp.gate.{down_proj,router_mlp.fc1,router_mlp.fc2,router_mlp.norm,router_mlp.out_proj,balancing_biases,router_states_scale}`
  - `model.layers.{i}.post_attention_layernorm.weight`, `post_attention_{residual,mlp}_residual_scale.*`
  - Global: `model.input_hidden_states_bias` / `_scale` (no `.weight` suffix), `model.final_norm.weight`
- Result: current output is 41 tensors (attn_output + embedding only) → garbage.
- Model structure: **10 layers × 8 experts**, d=1472-ish hidden, no Mamba (pure attn + MoE).
- Engine side expects the old GGUF names (`ffn_gate_up_exps`, `ffn_down_exps`, `cca_*`,
  `zaya_router_*`); check `src/backend_manager.cpp` case 19 + the zaya loader before
  rewriting the converter. Wiki claims "~64 tok/s HIP" from a synthetic Tile8 bench.
- Full shards are on disk at `/tmp/zyphra/zaya8/` (17.7GB; tmpfs — copy to models/ if needed).

### Issue 3 — ZUNA1.1: no runtime path
- Custom architecture (config: `dim:1024, n_layers:16`, sliding-window xattn, not a
  transformer, not RWKV). No GGUF converter, no llama.cpp arch, not in the 1bit engine.
- Decision needed: skip, or add a converter + engine arch (large effort).

### Issue 4 — stale hosted 1BP files on HF (bong-water-water-bong)
- The Zamba2 1BP repos (Jul 20) were converted from the **pre-fix GGUFs** → wrong weights
  (scrambled embedding, swapped norms, mamba1-style scans baked into downstream tools).
- Rebuilt + verified locally this session (per-tensor decode, 0 mismatches):
  - `Zamba2-1.2B`: `/tmp/zyphra/z2-12b-fixed.1bp` (1.19 GB)
  - `Zamba2-2.7B`: `/tmp/zyphra/z2-27b-fixed.1bp` (2.58 GB)
  - `Zamba2-7B`: `/tmp/zyphra/z2-7b-fixed.1bp` (7.25 GB)
- **Action**: re-upload to HF (`scripts/push_to_hub.sh` / `hf upload`) and update the
  wiki catalog sizes (currently 375MB–1.8GB, now 1.19–7.25GB).

### Issue 5 — BlackMamba-2.8B GGUF on HF is a 0-byte file
- `bong-water-water-bong/BlackMamba-2.8B-GGUF/blackmamba-2.8b.gguf` = 0 bytes (upload
  failed at the time). Rebuilt locally: `models/blackmamba-2.8b.gguf` (1.74 GB, via
  `scripts/blackmamba_to_gguf.py`, Q4_0, 46.2 tok/s). Re-upload.

### Issue 6 — llama.cpp (zamba-cpp fork) can't load the old BlackMamba GGUF
- Missing `mamba.`-prefixed KVs (`mamba.context_length`, `mamba.attention.layer_norm_rms_epsilon`)
  and tokenizer KVs. `models/blackmamba-1.5b-fixed.gguf` is a partial rebuild that still
  needs the tokenizer KVs (copy from the ZR1 GGUF — `add_token_list`/`add_token_merges`).
  Only worth finishing if the llama.cpp path is wanted; the engine's `test_mamba1_backend`
  already works on the originals.

## 3. Knowledge — the bug catalog (do not re-introduce)

Validation methodology that caught everything: convert → run engine → compare first-token
logits against HF transformers (same weights), corr target > 0.99. The reference loader:
`/tmp/zyphra/hf27_dump.py` + `hf27.py` (2.7B needs the adapter-parity rebuild + tie copy;
transformers 5.14 constructs layer-12 adapters at even indices while the checkpoint stores
odd — see the `rebuild_adapter_list` in those scripts).

1. **Converter — shared-block duplication**: Zamba2 ties transformer blocks (ABAB,
   `num_mem_blocks=2`). Only layers 6/12 store weights; the other 7 hybrids must duplicate
   them and fold the per-layer gate_up LoRA (`W_eff = W_shared + B@A`, adapter at hybrid
   position p lives in block p%2). `tools/convert_zamba2_safetensors_to_gguf.py`.
2. **Converter — norm mapping**: `post_attention_norm.weight` = the concat/attn input norm
   (HF `input_layernorm`), `ffn_norm.weight` = the pre-FFN norm (HF `pre_ff_layernorm`).
   Swapping them silently degrades every hybrid layer.
3. **Converter — GGUF 2D layout**: the loader's `read_tensor_transposed` treats flat data
   as `[input, output]`-major with `ne[0]=input`. Write numpy `(out,in).T` with
   `raw_shape=(out,in)`. Embedding: d_model-major data + `raw_shape=(vocab,d_model)`.
   Conv1d: k-major `(d_conv,1,conv_dim)`. Without this every 2D matrix is scrambled.
4. **Loader — conv kernel reversal**: the checkpoint stores nn.Conv1d cross-correlation
   order; the engine's recurrence needs the kernel reversed along the conv axis
   (`normalize_conv1d_reverse` in `src/gguf_zamba2_loader.cpp`).
5. **Engine — TRUE mamba2 scan**: state is `[head][d_state][head_dim]`; each head_dim
   slice evolves only from the previous token (no intra-token coupling). The old
   mamba1-style shared-state scan decays x[0] into x[1..] within a token — diverges from
   every mamba2 reference. (CPU: `src/mamba2_kernels.cpp`; HIP: 3 kernels in
   `src/mamba2_kernels.hip`, all must agree.)
6. **Engine — gate before group norm** in the mamba mixer (HF `Zamba2RMSNormGated`:
   `hidden * silu(gate)` then RMSNorm). HIP tuned block must match.
7. **Engine — dt clamp**: HF clamps to `(time_step_min, inf)` and **ignores time_step_max**
   (0.1). Clamping the upper bound diverges.
8. **Engine — GELU not SiLU** in the Zamba2 shared-transformer FFN (HF `hidden_act=gelu`;
   the numpy ref `tools/zamba2_ref.py` has the old SiLU bug — it and the engine were
   "validated" against each other with the same wrong math).
9. **Engine — hybrid residual**: `out = input + mamba(norm(input+th))` — the ssm_mix
   output is consumed inside the norm; adding `th` to the residual double-counts it.
10. **HIP — engine-loop bugs**: RMS-norm kernel args were swapped (`out,dst` vs `x,src`);
    pure-mamba path must save the pre-norm input and add it back (block overwrites
    d_hidden); the mamba block + pre-norm must be skipped for hybrid layers (double
    mamba); RoPE freq exponent needs `/head_dim` (the old kernel omitted it — the 1.2B
    is the only family member with `use_mem_rope=true`, so it was never exercised).
11. **Mamba2 state size**: `d_state*d_inner` per layer == `n_head*d_state*head_dim`, so
    the state layout change fits existing allocations.

## 4. Session logistics notes

- **The other agent(s) periodically `git reset --hard origin/<branch>` + switch branches**,
  wiping uncommitted work (happened 3× this session). Commit after every change; keep
  work on a dedicated branch (`docs/engine-reorg-full` currently).
- Commits this session (on `docs/engine-reorg-full`):
  `1fbe2321c` (cherry-picked as `e68964758`), `79dacbdb1`, `04e6bc87f`, `d8d6b009c`,
  `50b62757d`, `a031948e4`, `fc5f4874a`, `8f88c82ee`, `7c345bc4c`.
- `/tmp` is tmpfs (62G) — GGUFs/safetensors there are ephemeral. `models/` is safe.
- Build commands: CPU ref `make -f Makefile.zamba2`; HIP bench
  `cmake --build build --target bench_zamba2_e2e`; verify `build/verify_1bp <1bp> <gguf>`.
- The llama.cpp fork with the zamba2 arch (unmerged upstream PR #21412): `/tmp/zamba-cpp`
  (Vulkan build; EchoLabs GGUFs use a separate gate/up format the fork can't load —
  our converter produces fused gate/up, so use ours or patch the fork loader).
