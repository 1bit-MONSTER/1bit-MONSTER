// npu_runlist_bridge.h — single-launch whole-layer per-ctx ELF decode bridge.
//
// The FastFlowLM-validated whole-layer decode path (npu-infer's
// RuntimeLayerEngine) uses npu-infer's own ModelConfig (npu-infer/include/
// common.h), which CLASHES by name with the engine's ModelConfig
// (engine/npu/src/model_config.h). This bridge isolates the two TUs: the
// engine only sees this plain C-linkage function, never the npu-infer headers.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Decode `ng` greedy tokens for dense Qwen3 (0.6B/1.7B/4B) via the whole-layer
// per-ctx ELF path (RuntimeLayerEngine, one xrt::runlist submit/token).
// `ids_file` is whitespace-separated prompt token ids (or NULL/"-" for stdin).
// `H/NC/NH/NKV/IM/NV` are the model dims (drives the per-model config +
// layer.xclbin/ELF-dir selection). Emits the markers flm_parity.sh parses.
// Returns 0 on success, non-zero to fall back to the split path.
int npu_runlist_decode(const char* model_path, int ng, const char* ids_file,
                       int H, int NC, int NH, int NKV, int IM, int NV);

// Tell the runlist decode which RoPE base the model uses. It builds its own i6 cos/sin table
// (FLM host-writes it before every forward) and had a hardcoded Qwen3 theta = 1e6, so every
// non-Qwen3 family was decoded with the wrong rotation frequencies. Called before
// npu_runlist_decode(); a non-positive value is ignored.
extern "C" void npu_runlist_set_rope_theta(float th);

// Session API — the unified bf16-prefill -> runlist-decode path. The engine
// runs the fast bf16 prefill, then hands its device KV + final hidden to the
// RuntimeLayerEngine and continues greedy decode via per-ctx ELF runlists.
// Returns 0 on success (non-zero aborts the unified path).
int npu_runlist_session_init(const char* model_path, int H, int NC, int NH, int NKV, int IM, int NV);
// Write `n_tokens` of bf16 K/V (runtime 4-region layout) into layer `layer`'s
// device KV BO at token offset `token_begin`.
//
// src_region_stride_u16 is the region stride the CALLER laid the source out
// with, in u16. It is NOT always the session's stride: the bf16 attention ELF
// bakes its own, and it is per-shape (8 MB for the nh16 captures, 4 MB for the
// nh32 ones) while every runlist layer ELF is baked at MAX_L=8192 -> 8 MB. When
// they differ the source is re-packed region by region; when they match the
// copy is direct. Passing the caller's real stride is what makes the unified
// handoff work for nh32 at all — assuming the session's stride silently read
// regions 1..3 from the wrong offsets and the decode was wrong from the first
// step (see the results doc).
int npu_runlist_write_kv(int layer, int token_begin, int n_tokens,
                         const uint16_t* bf16_kv, int src_region_stride_u16);
// Write the prefill's final hidden state (bf16, H) into the act BO.
int npu_runlist_write_act(const uint16_t* bf16_hidden);

// Diagnostic: write the session's layer-0 KV BO to `path` (default 32 MB).
// Inert unless called; used to diff the bf16-prefill handoff against the host
// prefill (see RESULTS-unified-decode-penalty-2026-09-18.md).
extern "C" int npu_runlist_dump_kv(const char* path);
// Embed a token into the act BO (for the decode loop's forward step).
int npu_runlist_embed(int token);
// Run the lm_head on the act BO and write logits to `logits` (NV floats).
int npu_runlist_lmhead(float* logits, int vocab);
// One decode forward at context length `ctx_len` (1-based), then lm_head into
// `logits`.
int npu_runlist_forward(int ctx_len, float* logits, int vocab);
// Free the session (engine exit path).
void npu_runlist_session_free(void);

// bf16 prefill (mm.xclbin dequant + GEMM): load the model once, then pack a
// per-layer Q4NX weight BO + tile offsets for the dequant bridge.
int npu_bf16_prefill_init(const char* model_path, int H, int NC, int NH, int NKV, int IM, int NV, int HD);
int npu_bf16_pack_layer(int layer, uint8_t* bo /* >= layer BO bytes */, int* offs /* 6 ints */);
int npu_bf16_layer_bo_bytes(void);

#ifdef __cplusplus
}
#endif
