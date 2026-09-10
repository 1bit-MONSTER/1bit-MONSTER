#ifndef NPU_INFER_MODEL_H
#define NPU_INFER_MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tensor descriptor from Q4NX metadata
typedef struct {
    // Order by descending alignment to eliminate implicit padding (fixes #1336)
    int64_t  shape[4];        // offset 0,  8-byte aligned
    uint64_t data_offset;     // offset 32, 8-byte aligned
    uint64_t data_size;       // offset 40, 8-byte aligned
    uint64_t num_elements;    // offset 48, 8-byte aligned
    int      ndim;            // offset 56, 4-byte aligned
    char     name[128];       // offset 60
    char     dtype[16];       // offset 188
} TensorDesc;
#ifdef __cplusplus
static_assert(sizeof(TensorDesc) == 208, "TensorDesc padding mismatch");
#endif

// Per-layer weights
typedef struct {
    TensorDesc input_layernorm_weight;
    TensorDesc post_attention_layernorm_weight;
    TensorDesc q_norm_weight;
    TensorDesc k_norm_weight;
    TensorDesc q_proj_weight;
    TensorDesc k_proj_weight;
    TensorDesc v_proj_weight;
    TensorDesc o_proj_weight;
    TensorDesc gate_proj_weight;
    TensorDesc up_proj_weight;
    TensorDesc down_proj_weight;

    // ---- MoE (Qwen3.6-35B) routed + shared experts ----
    TensorDesc up_exps_weight;         // mlp.up_exps_proj.weight   [4096,8,5120] I8
    TensorDesc gate_exps_weight;       // mlp.gate_exps_proj.weight [4096,8,5120] I8
    TensorDesc down_exps_weight;       // mlp.down_exps_proj.weight [16384,2,5120] I8
    TensorDesc share_up_exps_weight;   // mlp.share_up_exps_proj.weight
    TensorDesc share_gate_exps_weight; // mlp.share_gate_exps_proj.weight
    TensorDesc share_down_exps_weight; // mlp.share_down_exps_proj.weight
    TensorDesc moe_router_weight;      // moe_router.weight [2048,256] BF16
    TensorDesc shared_expert_gate_weight; // shared_expert_gate.weight [2048] BF16

    // ---- linear-attn (GateDeltaNet) tensors, present on linear layers ----
    TensorDesc self_attn_gate_proj_weight; // self_attn.gate_proj.weight [128,8,8704] I8
    TensorDesc qkv_proj_weight;        // linear_attn.qkv_proj.weight [256,8,8704] I8
    TensorDesc ssm_out_proj_weight;    // linear_attn.ssm_out_proj.weight [64,16,8704] I8
    TensorDesc ssm_conv1d_weight;      // linear_attn.ssm_conv1d.weight [4,8192] BF16
    TensorDesc ssm_norm_weight;        // linear_attn.ssm_norm.weight [128] BF16
    TensorDesc ssm_a;                  // linear_attn.ssm_a [32] F32
    TensorDesc ssm_dt_bias;            // linear_attn.ssm_dt.bias [32] F32
    TensorDesc ssm_alpha_proj_weight;  // linear_attn.ssm_alpha_proj.weight [2048,32] BF16
    TensorDesc ssm_beta_proj_weight;   // linear_attn.ssm_beta_proj.weight [2048,32] BF16
} LayerWeights;

// Full model weights
typedef struct {
    ModelConfig config;
    
    // Embedding/LM Head (tied)
    TensorDesc embed_tokens;   // BF16 [vocab_size, hidden_size]
    
    TensorDesc lm_head_weight; // I8 tied to embed_tokens
    
    // Per-layer
    LayerWeights* layers;
    
    // Final norm
    TensorDesc norm_weight;
    
    // Raw file data (mmap)
    uint8_t* file_data;
    uint64_t file_size;
    // Offset of the first tensor's data: 8 (json length) + json_len. The
    // metadata data_offsets are relative to THIS, so tensor data lives at
    // file_data + data_base + desc->data_offset.
    uint64_t data_base;
} ModelWeights;

// ========== Model Loader ==========
ModelWeights* model_load(const char* path, ModelConfig config);
void model_free(ModelWeights* mw);
void* model_tensor_data(ModelWeights* mw, TensorDesc* desc);
void model_print_info(ModelWeights* mw);
int model_find_tensor(const char* name, ModelWeights* mw);

// ========== NPU Weight Packer ==========

// Get number of NPU blocks needed for a weight tensor
// in_features is the input dimension (column dimension that gets blocked to 1024)
int npu_weight_num_blocks(const TensorDesc* desc, const ModelConfig* config,
                          int in_features);

// Dequantize Q4NX I8 tensor (torch2aie 32x256 tiles) to BF16, one [256,1024]
// block at a time. Each I8 row (5120 B) is one tile: [512 B bf16 scales][512 B
// bf16 zeros][4096 B packed int4]; the block grid is row-major over
// (logical_rows/256) x (in_features/1024) blocks.
// out: pre-allocated buffer of block_rows * block_cols BF16 values (second half padded with zeros)
// in: Q4NX I8 weight data pointer
// block_idx: block index, row-major over (row-blocks x col-blocks)
// Returns number of BF16 values written
int npu_dequant_block(void* out, const void* in, 
                       const TensorDesc* desc, const ModelConfig* config,
                       int block_idx, int in_features);

// Pack a single 1MB BO from a tensor
// bo_buffer: pre-allocated 1MB buffer (will be filled with [block] + zeros)
// in: Q4NX weight data pointer
// block_idx: which block to pack (row-major)
// Returns 0 on success
int npu_pack_weight_bo(uint8_t* bo_buffer, const void* in,
                        const TensorDesc* desc, const ModelConfig* config,
                        int block_idx, int in_features);

// ========== 35B MoE weight-BO packing (Round 50, byte-verified) ==========
// docs/35b-forward-integration.md Round 50 + tools/verify_moe_current_layout.py.
//
// Expert pool (512 MB BO per LINEAR layer), rows 0..100959:
//   rows 0..65535: alternating 32-row up/gate blocks (1024 each), window order
//                  j = base + 8*(i%4) + i/4  (4736-B windows from file offset 0)
//   rows 65536..100959: down, all 35424 windows in 8-window groups
//                  [0,2,4,6,1,3,5,7] (+8 per group)
//   rows 100960..102623: self_attn.gate_proj (documented; not yet packed)
//   rows 102624+: zeros.
//
// 5 MB linear-attn BO per LINEAR layer:
//   head (328192 B) = [ssm_conv1d 65536][ssm_norm 256][ssm_a 128]
//                     [ssm_dt.bias 128][ssm_alpha_proj 131072][ssm_beta_proj 131072]
//   then ssm_out_proj windows (from file offset 0) in 32-row blocks,
//   order j = base + 16*(i%2) + i/2 (windows 0..1880); rest zero.
//
// Both are byte-identical to the Python reference (which was itself
// byte-verified against the moe-cap4 runtime captures, Round 50).

// Pack one linear layer's expert pool (up+gate+down) into a 512 MB buffer.
// Returns bytes written (478,146,560) or 0 on error.
int64_t npu_pack_moe_expert_pool(uint8_t* bo, ModelWeights* mw, int layer);

// Pack one linear layer's 5 MB linear-attn BO. Returns bytes written
// (5,242,880) or 0 on error.
int64_t npu_pack_moe_linear5_bo(uint8_t* bo, ModelWeights* mw, int layer);

// Pack one linear layer's region-B weight content (share_* + qkv + gate_proj,
// 8704-B tiles trimmed to 4736 + A/B interleave). Returns bytes written
// (16,367,616) or 0 on error.
int64_t npu_pack_moe_region_b(uint8_t* bo, ModelWeights* mw, int layer);

// Pack one layer's router BO (arg-2 of the MoE layer kernel):
// shared_expert_gate @0x2000, moe_router @0x3000. Returns bytes written
// (0x3000 + moe_router bytes) or 0 on error.
int64_t npu_pack_moe_router_bo(uint8_t* bo, ModelWeights* mw, int layer);

#ifdef __cplusplus
}
#endif

// Convert BF16 buffer back to float for verification
float bf16_to_float(uint16_t v);
uint16_t float_to_bf16(float v);

#endif // NPU_INFER_MODEL_H
