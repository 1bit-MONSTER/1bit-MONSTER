#ifndef NPU_INFER_ROCM_ENGINE_H
#define NPU_INFER_ROCM_ENGINE_H

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <hip/hip_runtime.h>

// ROCm GPU Inference Engine for Qwen3-0.6B
// Uses pre-allocated GPU buffers with raw HIP GEMM kernels
// Target: Radeon 8060S (gfx1151, 40 CU)
class RocmInferenceEngine {
public:
    RocmInferenceEngine();
    ~RocmInferenceEngine();

    // Load model from .q4nx file, init GPU
    bool init(const char* model_path);

    // Generate output tokens from input tokens
    // Returns number of tokens generated, or -1 on error
    int generate(const int* input_tokens, int num_input_tokens,
                 int* output_tokens, int max_output_tokens);

    // Get timing stats
    double last_token_ms() const { return last_token_ms_; }
    double avg_token_ms() const { return total_tokens_ > 0 ? total_time_ms_ / total_tokens_ : 0.0; }

private:
    // Model dimensions (Qwen3-0.6B)
    static constexpr int H = 1024;      // hidden size
    static constexpr int NC = 28;       // num layers
    static constexpr int NH = 16;       // num query heads
    static constexpr int NKV = 8;       // num key/value heads
    static constexpr int HD = 128;      // head dim
    static constexpr int IM = 3072;     // intermediate (FFN)
    static constexpr int NV = 151936;   // vocab size
    static constexpr int GQA = 2;       // NH / NKV
    static constexpr float EPS = 1e-6f;

    // Per-layer weight offsets from JSON
    struct LayerOffsets {
        uint64_t qp, kp, vp, op, gp, up, dp;
        uint64_t in_off, pa_off, qn_off, kn_off;
    };

    // Pre-packed per-layer weights (dequantized FP32, host-side)
    struct LayerWeights {
        std::vector<float> qkv;  // H × 4096 (fused Q+K+V)
        std::vector<float> o;    // 2048 × H
        std::vector<float> gu;   // H × 6144 (fused G+U)
        std::vector<float> d;    // IM × H
        std::vector<float> in_norm;   // H
        std::vector<float> pa_norm;   // H
        std::vector<float> q_norm;    // HD
        std::vector<float> k_norm;    // HD
        // GPU buffer for QKV weight
        float* d_qkv = nullptr;
        float* d_o = nullptr;
        float* d_gu = nullptr;
        float* d_down = nullptr;
    };

    // GPU state
    bool initialized_ = false;

    // Model data (loaded from .q4nx)
    std::vector<float> embed_table_;    // NV × H
    std::vector<float> final_norm_;     // H
    std::vector<float> lm_head_;        // H × NV
    std::vector<LayerWeights> layers_;
    std::vector<LayerOffsets> offsets_;

    // GPU buffers (pre-allocated at init)
    float* d_hidden_ = nullptr;     // H (GEMM input buffer, sized for max K)
    float* d_residual_ = nullptr;   // H (for GPU-side residual)
    float* d_q_out_ = nullptr;      // 4096 (QKV output)
    float* d_k_out_ = nullptr;      // NKV*HD = 1024
    float* d_v_out_ = nullptr;      // NKV*HD = 1024
    float* d_kv_cache_k_ = nullptr; // NC × MAX_POS × NKV × HD
    float* d_kv_cache_v_ = nullptr;
    float* d_attn_scores_ = nullptr; // MAX_POS = 4096
    float* d_attn_out_ = nullptr;   // NH*HD = 2048
    float* d_o_out_ = nullptr;      // H = 1024
    float* d_gate_ = nullptr;       // 6144 (GU fused)
    float* d_silu_out_ = nullptr;   // IM = 3072
    float* d_down_ = nullptr;       // H = 1024
    float* d_logits_ = nullptr;     // NV = 151936
    float* d_lm_head_ = nullptr;    // H × NV

    // GPU buffer for LM head weights
    float* d_embed_ = nullptr;      // NV × H (embedding table)
    float* d_final_norm_ = nullptr; // H
    float* d_rope_cos_ = nullptr;    // MAX_POS × HD
    float* d_rope_sin_ = nullptr;

    // Host buffers (for memcpy)
    std::vector<float> hidden_;     // H
    std::vector<float> residual_;   // H
    std::vector<float> q_out_;      // 4096
    std::vector<float> k_out_;      // NKV*HD
    std::vector<float> v_out_;      // NKV*HD
    std::vector<float> attn_scores_; // max seq len
    std::vector<float> attn_out_;   // NH*HD
    std::vector<float> o_out_;      // H
    std::vector<float> gate_;       // 6144
    std::vector<float> silu_out_;   // IM
    std::vector<float> down_;       // H
    std::vector<float> logits_;     // NV

    // KV cache host mirror
    struct KVCache {
        std::vector<float> k;  // MAX_POS * NKV * HD
        std::vector<float> v;
        int len = 0;
    };
    std::vector<KVCache> kv_cache_;
    static constexpr int MAX_POS = 4096;
    // Max K dimension in any GEMM (for d_hidden_ sizing)
    static constexpr int MAX_GEMM_K = IM;  // 3072 for down projection

    // Timing
    double last_token_ms_ = 0.0;
    double total_time_ms_ = 0.0;
    int total_tokens_ = 0;

    // ---- Internal methods ----

    // Load model from memory-mapped .q4nx
    bool load_model(const uint8_t* data, size_t size);

    // Load model from pre-exported raw float32 files (HF safetensors converted)
    bool load_model_from_raw(const char* dir);

    // Allocate GPU buffers and upload weights
    bool gpu_init();

    // Per-layer forward pass (GPU-assisted)
    void forward_layer(int l, int pos);

    // GEMM: C[M×N] = A[M×K] × B[K×N] via HIP kernel
    // A is host-side, B is device-side (pre-uploaded), C is host-side
    void gemm(const float* A_host, float* d_B, float* C_host,
              int M, int K, int N);
};

#endif // NPU_INFER_ROCM_ENGINE_H
