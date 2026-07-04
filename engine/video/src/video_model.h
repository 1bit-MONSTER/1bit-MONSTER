// video_model.h — Video diffusion model configuration.
// Targets Wan2.2-1.3B DiT architecture. Reuses NPU engine's I8Ctx for INT8 GEMM.

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --- Wan2.2-1.3B DiT architecture constants ---
// Hidden size H = 1024
// Num layers   NC = 30
// Num heads    NH = 16
// Head dim     HD = 64  (1024/16)
// MLP hidden   IM = 4096
// Patch size   PS = 2   (2×2 spatial patches)
// Latent channels C = 16 (Wan uses 16-channel VAE)
// Text hidden  TH = 4096 (T5 encoder)

struct VideoModelConfig {
    // DiT architecture
    int H = 1024;        // Hidden size
    int NC = 30;         // Number of transformer layers
    int NH = 16;         // Number of attention heads
    int HD = 64;         // Head dimension (H/NH)
    int IM = 4096;       // MLP intermediate size
    int PS = 2;          // Patch size
    int C = 16;          // Latent channels
    float rope_theta = 10000.0f;

    // Text conditioning
    int TH = 4096;       // Text hidden size (T5-encoder)
    int TX = 512;        // Max text token length

    // Video dimensions
    int num_frames = 1;
    int height = 480;
    int width = 640;
    int latent_h = 0;    // derived: height / (PS * 8)
    int latent_w = 0;    // derived: width / (PS * 8)
    int num_patches = 0; // derived: latent_h * latent_w

    // Sampling
    int num_steps = 50;
    float cfg_scale = 5.0f;

    // Weight file paths
    std::string model_dir;
    std::string model_tag;

    bool valid() const { return H > 0 && NC > 0 && NH > 0 && HD > 0 && IM > 0; }

    void derive() {
        // VAE compresses 8×: input = height×width, latent = height/8 × width/8
        // Then patch embedding: patches = (latent_h/PS) × (latent_w/PS)
        latent_h = (height / 8) / PS;
        latent_w = (width / 8) / PS;
        if (latent_h < 1) latent_h = 1;
        if (latent_w < 1) latent_w = 1;
        num_patches = latent_h * latent_w;
    }
};

// --- Video weight file container ---
// Simple binary format: [header][float weights]
// Header: model_tag string + dimension info

#pragma pack(push, 1)
struct VideoWeightHeader {
    char magic[8] = {'V','I','D','W','E','I','G','H'};
    uint32_t version = 1;
    uint32_t num_layers = 0;
    uint32_t hidden_size = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t mlp_hidden = 0;
    uint32_t text_hidden = 0;
    // For each layer: 
    //   q_proj [H, NH*HD]
    //   k_proj [H, NH*HD]
    //   v_proj [H, NH*HD]
    //   o_proj [NH*HD, H]
    //   g_proj [H, IM] (gate)
    //   u_proj [H, IM] (up)
    //   d_proj [IM, H] (down)
    //   adaln_modulation [6*H, H] or [12*H, H]
    //   cross_attn_q [TH, NH*HD]
    //   cross_attn_kv [TH, 2*NH*HD]
    uint64_t weight_offset = 0; // offset to float weight data
    uint8_t padding[20];          // pad to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(VideoWeightHeader) == 64, "VideoWeightHeader must be 64 bytes");

// Load video model weights from file
inline bool load_video_weights(const char* path, VideoModelConfig& cfg,
                                std::vector<float>& weights) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[VideoModel] Cannot open %s\n", path); return false; }

    VideoWeightHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "[VideoModel] Failed to read header\n");
        fclose(f); return false;
    }

    if (memcmp(header.magic, "VIDWEIGH", 8) != 0) {
        fprintf(stderr, "[VideoModel] Invalid magic (got %.8s, expected VIDWEIGH)\n", header.magic);
        fclose(f); return false;
    }

    cfg.NC = header.num_layers;
    cfg.H = header.hidden_size;
    cfg.NH = header.num_heads;
    cfg.HD = header.head_dim;
    cfg.IM = header.mlp_hidden;
    cfg.TH = header.text_hidden;
    cfg.derive();

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long weight_bytes = file_size - (long)sizeof(VideoWeightHeader);
    
    weights.resize(weight_bytes / sizeof(float));
    fseek(f, sizeof(VideoWeightHeader), SEEK_SET);
    if (fread(weights.data(), 1, weight_bytes, f) != (size_t)weight_bytes) {
        fprintf(stderr, "[VideoModel] Failed to read weights\n");
        fclose(f); return false;
    }

    fclose(f);
    fprintf(stderr, "[VideoModel] Loaded %zu floats (%ld MB) for %s\n",
            weights.size(), weight_bytes / (1024*1024), path);
    return true;
}

// --- NPU bridge: reused from npu_engine_universal.cpp I8Ctx ---
// This is the INT8 GEMM dispatch on XDNA 2.
// For CPU fallback we provide the same interface using OpenMP BLAS.
// In production, include npu_engine_universal.cpp and use its I8Ctx directly.

#ifdef USE_NPU
  // Include NPU engine's I8Ctx
  // #include "../npu/src/npu_engine_universal.cpp"
  // (linked externally)
#else
  // CPU fallback: pure OpenMP GEMM for development/testing
  #include <omp.h>
  struct CPUGEMM {
      static void gemm(const float* A, const float* B, float* C,
                        int M, int K, int N) {
          #pragma omp parallel for
          for (int m = 0; m < M; m++) {
              for (int n = 0; n < N; n++) {
                  double sum = 0.0;
                  #pragma omp simd reduction(+:sum)
                  for (int k = 0; k < K; k++) {
                      sum += (double)A[m * K + k] * B[k * N + n];
                  }
                  C[m * N + n] = (float)sum;
              }
          }
      }
  };
#endif
