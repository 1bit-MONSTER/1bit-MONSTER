// backend_fused_npu.h — NPU FFN state (pure C++, not HIP).
#pragma once
#ifdef __cplusplus

struct NpuState;

NpuState* npu_state_create(const char* xclbin_dir, int H, int IM, int NC);
void npu_state_destroy(NpuState* s);
void npu_state_pack_layer(NpuState* s, int layer,
                           const float* w1, const float* w2, const float* w3,
                           const float* ffn_norm_w);
bool npu_state_ffn(NpuState* s, int layer, float* h, int H);
// Batched multi-sequence FFN: processes `am` independent rows (h = [am, H]
// row-major) in ONE GU + ONE D launch each.  The B weight DMA is read once
// for all rows (m8 xclbin: 8 rows in ~2.05 ms vs 2.06 ms for 1).  am must be
// <= the AIE tile width (8 for the m8 family, 32 for the m32 full-grid
// family, 128 for the M=128 family).
// Row results are bit-identical to npu_state_ffn called per row.
bool npu_state_ffn_batch(NpuState* s, int layer, float* h, int H, int am);

#endif
