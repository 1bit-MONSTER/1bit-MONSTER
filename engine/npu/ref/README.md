# NPU engine variants reference

| File | Architecture | GEMM format | Dependencies |
|---|---|---|---|
| `npu_engine_v2_bf16.cpp` | BF16 NpuKernel struct (from npu-infer) | ATB-shuffled BF16 for MAI AI Engine xclbin | common.h, model.h, helper.h, gemm_atb_layout.h |
| `../src/npu_engine_v2.cpp` | int8 I8Ctx (active) | Dynamic-scale int8 for i8 xclbin | dequant_q4nx.c, xrt headers, stdlib |

`npu_engine_v2_bf16.cpp` (409 lines, from npu-infer) is a fundamentally different engine targeting BF16 packed weights with an ATB (AI Engine Block) layout kernel. It uses `ModelConfig`-based dimensions (not hardcoded) and supports GQA attention expansion, SiLU, and proper error handling. It was preserved separately because the two engines share only the overall transformer pipeline, not the weight format, kernel dispatch, or math types.

Recovered from git history (commit e5f2d15, path `npu-infer/src/npu_engine_v2.cpp`). Original authored by `bong-water-water-bong`.
