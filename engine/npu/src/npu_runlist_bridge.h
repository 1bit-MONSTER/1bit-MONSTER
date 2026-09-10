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

// bf16 prefill (mm.xclbin dequant + GEMM): load the model once, then pack a
// per-layer Q4NX weight BO + tile offsets for the dequant bridge.
int npu_bf16_prefill_init(const char* model_path, int H, int NC, int NH, int NKV, int IM, int NV);
int npu_bf16_pack_layer(int layer, uint8_t* bo /* >= 10 MB */, int* offs /* 6 ints */);

#ifdef __cplusplus
}
#endif
