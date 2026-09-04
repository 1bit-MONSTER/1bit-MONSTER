# Round-28 zaya Q4NX decode — bug log

Session: 2026-09-04, `~/hrx-ws/amd-hrx-graph` (branch `hrx-graph-develop-v2`, base 6319038 + round-28 zaya port).
Target: `zaya-q4nx-c43.gguf` (GGML_TYPE_Q4NX = slot 43) decodes empty instead of matching the f32 twin.

## Fixed (this session)

### BUG-A — `MUL_MAT_ID_Q4NX` computes only 1 of N tokens

- File: `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_id_q4nx`.
- Bug: `const int64_t n_tok = ne11;` — the MoE input is 3-D `[n_embd, 1, n_tokens]` (tokens in `ne[2]`), so `ne11 == 1` and only token 0 was computed.
- Fix: `n_tok = ne11 * ne12 * ne13` (flatten all batch dims, mirroring `ggml_mul_mat_id`), and `src1_col` indexed with the `(i11,i12,i13)` decomposition.
- Also fixed the output shape in `ggml/src/ggml.c`, `ggml_mul_mat_id_q4nx`: `ne[4] = { rows, ids->ne[0], b->ne[1]*b->ne[2]*b->ne[3], 1 }`.

### BUG-B — `MUL_MAT_ID_Q4NX` reads expert ids as a flat array

- File: same function.
- Bug: `ids_data[si * n_tok + j]` — `selected_experts` (= `ggml_argsort_top_k`) is a **strided view** `[n_sel, n_tokens]` with `nb[1]` = the `[n_expert, n_tokens]` argsort's row stride (64 B). Reading it flat made token 1 select `argsort[1]` (= expert 11) instead of `argsort[16]` (= expert 4).
- Fix: `e = *(const int32_t *)((const char *)ids->data + si*ids->nb[0] + j*ids->nb[1]);` — identical to the standard `ggml_mul_mat_id` (ggml-cpu.c:1797).

After A+B, the layer-0 `ffn_moe_gate_up` is **bit-exact** for both tokens (corr 0.9999999999, maxdiff ~1.7e-6 vs the f32-twin reference).

## Remaining (root-caused, not yet fixed)

### BUG-C — flash-attention NaN from uninitialised KV-cache padding

- Symptom: full decode emits empty/whitespace tokens; logits are valid for the prefill (`argmax 4764`, no NaN) but become **all-NaN** a few decode steps in.
- Root cause (verified): `FLASH_ATTN_EXT` produces NaN because the KV cache (`SET_ROWS_cache_k_l1` / `cache_v_l1`) contains **NaN at random slots** — nondeterministic across runs (rows 180 / 36 / 6+92 / 18-21 in three probes). The flash-attn inputs themselves (`ROPE_Qcur`, `ROPE_Kcur`, `Vcur`) are NaN-free, so the NaN enters through the cache read.
- Interpretation: the KV-cache padding slots are **not zeroed** (or are re-faulted). `llama-kv-cache.cpp:307` does `ggml_backend_buffer_clear(buf, 0)` with the comment "initialize the buffers to avoid NaNs in the padding", but the zaya recurrent path still hits NaN padding — either the clear is not covering the recurrent memory, or a later op re-dirties the slots.
- Fix direction (pick one):
  1. Make the zaya KV-cache padding genuinely zeroed (audit `llama_kv_cache` vs `llama_memory_recurrent` buffer init for the recurrent path).
  2. Ensure the flash-attn mask limits reads to `[0, n_tokens)` so padding is never consumed.

## Repro

```bash
cd ~/hrx-ws/amd-hrx-graph
./build/bin/llama-cli -m ~/zaya-q4nx-c43.gguf -p "The capital of France is" \
  -n 8 -ngl 0 --seed 42 --temp 0     # empty output; logits -> NaN after ~2 decode steps

# bisect dumps (GGML_DUMP_NODE writes /tmp/nodedump/*.bin with ne in the filename):
env GGML_DUMP_NODE=1 ZAYA_1LAYER=1 ./build/bin/llama-cli -m ~/zaya-q4nx-c43.gguf -p "The capital" -n 1 -ngl 0
# first NaN node = FLASH_ATTN_EXT; cache_k_l1/cache_v_l1 (fp16) hold NaN at random rows
```

## Instrumentation added (env-gated, harmless)

- `Q4NX_ID_DBG=1` — `MUL_MAT_ID_Q4NX` ne/nb/tile/final-output dumps.
- `GGML_DUMP_NODE` + `GGML_DUMP_FA` + `GGML_DUMP_CPYV` — node-value / flash-attn / kv-copy dumps (uncommitted debug edits in `ggml-cpu.c`, `ops.cpp`, `llama-kv-cache.cpp`, `zaya.cpp`).
- `gguf-py` patched to accept type 43 (`Q4NX_C43 = 43`, `GGML_QUANT_SIZES`).

## BUG-C follow-up (kv-cache investigation, 2026-09-04) — corrected

- The KV-cache buffer **is** zeroed at init (verified 0 non-zero / 0 NaN over 10 MB), and the recurrent-state buffer is a **separate** allocation (the earlier "same address" was comparing `ggml_backend_buffer_t` *struct* pointers, not the data bases — the data bases are far apart, no overlap).
- The memory module is constructed, destroyed, and reconstructed (a sizing pass with `hparams.no_alloc`, then the real pass) — normal, and the real KV cache is cleared.
- The NaN appears in `SET_ROWS_cache_k_l1` / the flash-attn K/V input (`GGML_DUMP_FA` `*_k.bin`, ne=[128,256,2] fp16) at random slots, while the flash-attn inputs (`ROPE_Qcur/Kcur`, `Vcur`) are clean.
- Suspicious lead (not yet resolved): `set_input_v_idxs` logs V-cache position indices `[18 19 20 21]` and `[1..17]` for a 2-token prompt — the KV-cache slot allocation / `v_heads` may be wrong for the recurrent model (the cache is written at the wrong slots, so the flash-attn reads slots that were never written). Next step: trace `llama_kv_cache::prepare()` / `v_heads` for the zaya hybrid path.
