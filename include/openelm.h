// openelm.h — Apple OpenELM bespoke backend (per-layer heterogeneous dims).
// The generic engine assumes uniform per-layer dims; OpenELM's per-layer
// query/kv heads and per-layer FFN intermediate break it (documented-limitation
// tier before this). This backend reads the HF safetensors directly.
//
// Arch (modeling_openelm.py): per-layer q_h/k_h/v_h, fused qkv with ROW split
// [q|k|v], per-head RMSNorm on q/k (normalize_qk), HF half-split rope (chunk
// pairing, freq_constant 10000, full head_dim), GQA repeat, SDPA scale
// 1/sqrt(head_dim), swish-GLU FFN (proj_1 -> [y1|y2] -> swish(y1)*y2 -> proj_2),
// pre-norm RMSNorm, tied lm_head.
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <cstring>

namespace openelm_math {

struct OpenELMConfig {
    int model_dim = 1280;
    int num_layers = 16;
    int head_dim = 64;
    int vocab_size = 32000;
    float rms_eps = 1e-6f;
    std::vector<int> q_heads;   // [num_layers]
    std::vector<int> kv_heads;  // [num_layers]
    std::vector<int> intermediate; // [num_layers]
};

struct OpenELMLayer {
    std::vector<float> attn_norm;    // [model_dim]
    std::vector<float> qkv_w;        // [(q+k+v)*hd, model_dim] row split
    std::vector<float> q_norm;       // [head_dim]
    std::vector<float> k_norm;       // [head_dim]
    std::vector<float> out_proj;     // [model_dim, q_h*hd]
    std::vector<float> ffn_norm;     // [model_dim]
    std::vector<float> proj_1;       // [2*inter, model_dim]
    std::vector<float> proj_2;       // [model_dim, inter]
};

struct OpenELMModel {
    OpenELMConfig cfg;
    std::vector<float> token_emb;    // [vocab, model_dim]
    std::vector<float> final_norm;   // [model_dim]
    std::vector<OpenELMLayer> layers;
    bool load(const std::string& dir, const std::string& safetensors_path);
};

// math helpers
static inline void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * w[i] * inv;
}

// HF half-split rope (chunk pairing i, i+dim/2): (x*cos) + (rotate_half(x)*sin)
static inline void rope_half(float* x, int dim, int pos, float theta = 10000.0f) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = pos / powf(theta, 2.0f * i / dim);
        float c = cosf(freq), s = sinf(freq);
        float a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// Run one token; kv_cache[layer] = [cached positions][kv_heads*2, hd] interleaved k then v.
std::vector<float> openelm_forward(
    const OpenELMModel& model, int token_id,
    std::vector<std::vector<float>>& kv_cache, int& pos);

}  // namespace openelm_math
