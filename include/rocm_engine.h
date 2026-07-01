#ifndef NPU_INFER_ROCM_ENGINE_H
#define NPU_INFER_ROCM_ENGINE_H

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

// ROCm GPU Inference Engine for Qwen3-0.6B
// Uses raw HIP kernels (not hipBLAS) for GEMM
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

    // Pre-packed per-layer weights (dequantized FP32)
    struct LayerWeights {
        std::vector<float> qkv;  // H × 4096 (fused Q+K+V)
        std::vector<float> o;    // 2048 × H
        std::vector<float> gu;   // H × 6144 (fused G+U)
        std::vector<float> d;    // IM × H
        std::vector<float> in_norm;   // H
        std::vector<float> pa_norm;   // H
        std::vector<float> q_norm;    // HD
        std::vector<float> k_norm;    // HD
    };

    // GPU state
    void* d_handle_ = nullptr;  // placeholder for GPU device handle
    bool initialized_ = false; // Is engine initialized?

    // Model data (loaded from .q4nx)
    std::vector<float> embed_table_;    // NV × H
    std::vector<float> final_norm_;     // H
    std::vector<float> lm_head_;        // H × NV
    std::vector<LayerWeights> layers_;
    std::vector<LayerOffsets> offsets_;

    // Buffer pools (host-side)
    std::vector<float> hidden_;     // H
    std::vector<float> residual_;   // H
    std::vector<float> q_out_;      // NH × HD = 2048
    std::vector<float> k_out_;      // NKV × HD = 1024
    std::vector<float> v_out_;      // NKV × HD = 1024
    std::vector<float> attn_scores_; // 4096 max
    std::vector<float> attn_out_;   // NH × HD = 2048
    std::vector<float> o_out_;      // H
    std::vector<float> gate_;       // IM
    std::vector<float> up_;         // IM
    std::vector<float> silu_out_;   // IM
    std::vector<float> down_;       // H
    std::vector<float> logits_;     // NV

    // KV cache: [layer][position][head][dim]
    struct KVCache {
        std::vector<float> k;
        std::vector<float> v;
        int len = 0;
    };
    std::vector<KVCache> kv_cache_;

    // Timing
    double last_token_ms_ = 0.0;
    double total_time_ms_ = 0.0;
    int total_tokens_ = 0;

    // ---- Internal methods ----

    // Load model from memory-mapped .q4nx
    bool load_model(const uint8_t* data, size_t size);

    // Per-layer forward pass
    void forward_layer(int layer_idx, int position);

    // Embedding lookup
    void embedding_lookup(int token_id, float* out);

    // RMS normalization (in-place)
    void rms_norm(float* x, const float* weight, int n);

    // RoPE (in-place)
    void apply_rope(float* q, float* k, int position);

    // GEMM: C[M×N] = A[M×K] × B[K×N] via hipBLAS
    void gemm(const float* A, const float* B, float* C,
              int M, int K, int N);

    // Softmax (in-place)
    void softmax(float* x, int n);

    // SiLU activation
    float silu(float x);

    // Load JSON offsets from model file
    void parse_offsets(const char* json, size_t json_len);
};

#endif // NPU_INFER_ROCM_ENGINE_H
