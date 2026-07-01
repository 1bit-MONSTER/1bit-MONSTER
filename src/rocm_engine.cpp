/**
 * ROCm GPU Inference Engine for Qwen3-0.6B
 * Uses raw HIP GEMM kernels (no hipBLAS/rocBLAS dependency) on Radeon 8060S (gfx1151).
 *
 * All weights pre-allocated in GPU VRAM at init time.
 * GEMM = hipMemcpy hidden->GPU, launch kernel, hipMemcpy result->host.
 * No malloc/free in hot path.
 *
 * Build with system hipcc:
 *   cd build && hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 \
 *       -I../include -I/opt/rocm/include \
 *       ../src/rocm_engine.cpp dequant_q4nx.o -lm -o rocm_engine
 *
 * Run:
 *   LD_LIBRARY_PATH=/opt/rocm/lib ./rocm_engine
 */
#include "rocm_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);

#define HIP_CHECK(cmd) do {                                    \
    hipError_t _err = cmd;                                     \
    if (_err != hipSuccess) {                                  \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n",      \
                __FILE__, __LINE__, hipGetErrorString(_err), _err); \
        return;                                                \
    }                                                          \
} while(0)

#define HIP_CHECK_RET(cmd, rv) do {                            \
    hipError_t _err = cmd;                                     \
    if (_err != hipSuccess) {                                  \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n",      \
                __FILE__, __LINE__, hipGetErrorString(_err), _err); \
        return rv;                                             \
    }                                                          \
} while(0)

#define HIP_CHECK_BOOL(cmd) do {                               \
    hipError_t _err = cmd;                                     \
    if (_err != hipSuccess) {                                  \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n",      \
                __FILE__, __LINE__, hipGetErrorString(_err), _err); \
        return false;                                          \
    }                                                          \
} while(0)

// M=1 GEMV kernel for autoregressive decode
__global__ void gemv_kernel(const float* A, const float* B, float* C,
                             int K, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[k] * B[(size_t)k * N + col];
        C[col] = sum;
    }
}

// General GEMM for M > 1 (prefill)
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = sum;
    }
}

// Host-side RoPE tables
std::vector<float> g_rope_cos;
std::vector<float> g_rope_sin;

// ============================================================
// BF16 → float32 conversion helper
// ============================================================
static float bf16_to_float(const uint16_t* src, size_t n) {
    // bf16: 1 sign, 8 exponent, 7 mantissa → pack into upper 16 bits of float32
    union { uint32_t u; float f; } conv;
    conv.u = (uint32_t)src[n] << 16;
    return conv.f;
}

// ============================================================
// JSON offset finder
// ============================================================
static uint64_t find_json_offset(const char* json, size_t json_len, const char* name) {
    size_t nl = strlen(name);
    const char* p = json;
    const char* end = json + json_len;
    while (p < end) {
        auto q = (const char*)memmem(p, end - p, name, nl);
        if (!q) return 0;
        if (q > json && *(q-1) == '"' && *(q+nl) == '"') {
            auto off = strstr(q, "\"data_offsets\"");
            if (off) {
                auto br = strchr(off, '[');
                if (br) return strtoull(br + 1, NULL, 10);
            }
        }
        p = q + 1;
    }
    return 0;
}

// ============================================================
// Ctor / Dtor
// ============================================================
RocmInferenceEngine::RocmInferenceEngine() : initialized_(false) {
    kv_cache_.resize(NC);
    for (auto& kv : kv_cache_) {
        kv.k.resize(MAX_POS * NKV * HD, 0.0f);
        kv.v.resize(MAX_POS * NKV * HD, 0.0f);
        kv.len = 0;
    }
}

RocmInferenceEngine::~RocmInferenceEngine() {
    auto fg = [](float*& p) { if (p) { hipFree(p); p = nullptr; } };
    fg(d_hidden_); fg(d_residual_); fg(d_q_out_);
    fg(d_k_out_); fg(d_v_out_);
    fg(d_kv_cache_k_); fg(d_kv_cache_v_);
    fg(d_attn_scores_); fg(d_attn_out_); fg(d_o_out_);
    fg(d_gate_); fg(d_silu_out_); fg(d_down_); fg(d_logits_);
    fg(d_lm_head_); fg(d_embed_); fg(d_final_norm_);
    fg(d_rope_cos_); fg(d_rope_sin_);
    for (auto& l : layers_) {
        fg(l.d_qkv); fg(l.d_o); fg(l.d_gu); fg(l.d_down);
    }
}

// ============================================================
// init
// ============================================================
bool RocmInferenceEngine::init(const char* model_path) {
    printf("[ROCm] Engine init: model=%s\n", model_path);

    HIP_CHECK_BOOL(hipSetDevice(0));
    int dev_count = 0;
    HIP_CHECK_BOOL(hipGetDeviceCount(&dev_count));
    if (dev_count < 1) { fprintf(stderr, "[ROCm] No GPU found\n"); return false; }

    hipDeviceProp_t props;
    HIP_CHECK_BOOL(hipGetDeviceProperties(&props, 0));
    printf("[ROCm] Device: %s (arch %d.%d, %d CUs, %.0f MB VRAM)\n",
           props.name, props.major, props.minor,
           props.multiProcessorCount,
           (double)props.totalGlobalMem / (1024.0 * 1024.0));

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[ROCm] Cannot open %s\n", model_path); return false; }
    struct stat st;
    fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    bool ok = load_model(md, st.st_size);
    munmap(md, st.st_size);
    if (!ok) return false;

    hidden_.resize(H); residual_.resize(H);
    q_out_.resize(4096); k_out_.resize(NKV * HD); v_out_.resize(NKV * HD);
    attn_scores_.resize(MAX_POS); attn_out_.resize(NH * HD);
    o_out_.resize(H); gate_.resize(6144);
    silu_out_.resize(IM); down_.resize(H); logits_.resize(NV);

    if (!gpu_init()) return false;

    printf("[ROCm] Engine ready.\n");
    initialized_ = true;
    return true;
}

// ============================================================
// gpu_init
// ============================================================
bool RocmInferenceEngine::gpu_init() {
    auto gm = [](float*& p, size_t nb) -> bool {
        if (nb == 0) { p = nullptr; return true; }
        hipError_t e = hipMalloc(&p, nb);
        if (e != hipSuccess) { fprintf(stderr, "[ROCm] hipMalloc(%zu): %s\n", nb, hipGetErrorString(e)); return false; }
        return true;
    };
    auto up = [](float* d, const float* s, size_t n) -> bool {
        if (n == 0) return true;
        hipError_t e = hipMemcpy(d, s, n * sizeof(float), hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "[ROCm] H2D: %s\n", hipGetErrorString(e)); return false; }
        return true;
    };

    for (int l = 0; l < NC; l++) {
        auto& lw = layers_[l];
        if (!gm(lw.d_qkv,  lw.qkv.size() * 4)) return false;
        if (!gm(lw.d_o,    lw.o.size()   * 4)) return false;
        if (!gm(lw.d_gu,   lw.gu.size()  * 4)) return false;
        if (!gm(lw.d_down, lw.d.size()   * 4)) return false;
        if (!up(lw.d_qkv,  lw.qkv.data(), lw.qkv.size())) return false;
        if (!up(lw.d_o,    lw.o.data(),   lw.o.size())) return false;
        if (!up(lw.d_gu,   lw.gu.data(),  lw.gu.size())) return false;
        if (!up(lw.d_down, lw.d.data(),   lw.d.size())) return false;
        lw.qkv.clear(); lw.qkv.shrink_to_fit();
        lw.o.clear();   lw.o.shrink_to_fit();
        lw.gu.clear();  lw.gu.shrink_to_fit();
        lw.d.clear();   lw.d.shrink_to_fit();
    }

    if (!gm(d_lm_head_, (size_t)H * NV * 4)) return false;
    if (!up(d_lm_head_, lm_head_.data(), (size_t)H * NV)) return false;
    lm_head_.clear(); lm_head_.shrink_to_fit();

    if (!gm(d_embed_, (size_t)NV * H * 4)) return false;
    if (!up(d_embed_, embed_table_.data(), (size_t)NV * H)) return false;

    if (!gm(d_final_norm_, H * 4)) return false;
    if (!up(d_final_norm_, final_norm_.data(), H)) return false;

    // RoPE tables
    int rp_sz = MAX_POS * HD;
    if (!gm(d_rope_cos_, rp_sz * 4)) return false;
    if (!gm(d_rope_sin_, rp_sz * 4)) return false;
    g_rope_cos.resize(rp_sz); g_rope_sin.resize(rp_sz);
    for (int p = 0; p < MAX_POS; p++) {
        for (int d = 0; d < HD; d += 2) {
            float t = 1.0f / powf(1000000.0f, (float)d / (float)HD);
            float a = p * t;
            g_rope_cos[p * HD + d] = cosf(a);
            g_rope_sin[p * HD + d] = sinf(a);
            g_rope_cos[p * HD + d + 1] = cosf(a);
            g_rope_sin[p * HD + d + 1] = sinf(a);
        }
    }
    if (!up(d_rope_cos_, g_rope_cos.data(), rp_sz)) return false;
    if (!up(d_rope_sin_, g_rope_sin.data(), rp_sz)) return false;

    size_t kv_sz = (size_t)NC * MAX_POS * NKV * HD * 4;
    if (!gm(d_kv_cache_k_, kv_sz)) return false;
    if (!gm(d_kv_cache_v_, kv_sz)) return false;
    HIP_CHECK_RET(hipMemset(d_kv_cache_k_, 0, kv_sz), false);
    HIP_CHECK_RET(hipMemset(d_kv_cache_v_, 0, kv_sz), false);

    if (!gm(d_hidden_,      MAX_GEMM_K * 4)) return false;
    if (!gm(d_residual_,    H * 4)) return false;
    if (!gm(d_q_out_,       4096 * 4)) return false;
    if (!gm(d_k_out_,       NKV * HD * 4)) return false;
    if (!gm(d_v_out_,       NKV * HD * 4)) return false;
    if (!gm(d_attn_scores_, MAX_POS * 4)) return false;
    if (!gm(d_attn_out_,    NH * HD * 4)) return false;
    if (!gm(d_o_out_,       H * 4)) return false;
    if (!gm(d_gate_,        6144 * 4)) return false;
    if (!gm(d_silu_out_,    IM * 4)) return false;
    if (!gm(d_down_,        H * 4)) return false;
    if (!gm(d_logits_,      (size_t)NV * 4)) return false;

    return true;
}

// ============================================================
// load_model
// ============================================================
bool RocmInferenceEngine::load_model(const uint8_t* data, size_t size) {
    uint64_t hdr_len;
    memcpy(&hdr_len, data, 8);
    uint64_t data_start = 8 + hdr_len;
    const char* json = (const char*)(data + 8);
    size_t json_len = hdr_len;

    // Embedding table (BF16 in file, convert to float32)
    embed_table_.resize(NV * H);
    const uint16_t* bf16_embed = (const uint16_t*)(data + data_start);
    for (size_t i = 0; i < (size_t)NV * H; i++) {
        union { uint32_t u; float f; } conv;
        conv.u = (uint32_t)bf16_embed[i] << 16;
        embed_table_[i] = conv.f;
    }

    offsets_.resize(NC);
    char key[128];
    for (int l = 0; l < NC; l++) {
        snprintf(key, 128, "model.layers.%d.self_attn.q_proj.weight", l);
        offsets_[l].qp = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.self_attn.k_proj.weight", l);
        offsets_[l].kp = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.self_attn.v_proj.weight", l);
        offsets_[l].vp = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.self_attn.o_proj.weight", l);
        offsets_[l].op = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.mlp.gate_proj.weight", l);
        offsets_[l].gp = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.mlp.up_proj.weight", l);
        offsets_[l].up = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.mlp.down_proj.weight", l);
        offsets_[l].dp = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.input_layernorm.weight", l);
        offsets_[l].in_off = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.post_attention_layernorm.weight", l);
        offsets_[l].pa_off = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.self_attn.q_norm.weight", l);
        offsets_[l].qn_off = find_json_offset(json, json_len, key);
        snprintf(key, 128, "model.layers.%d.self_attn.k_norm.weight", l);
        offsets_[l].kn_off = find_json_offset(json, json_len, key);
    }

    uint64_t fn_off = find_json_offset(json, json_len, "model.norm.weight");
    uint64_t lm_off = find_json_offset(json, json_len, "lm_head.weight");

    layers_.resize(NC);
    for (int l = 0; l < NC; l++) {
        int qr, kr, vr, or_, gr, ur, dr, unused;
        float* qw = dequant_i8_to_float(data + data_start + offsets_[l].qp, 256, &qr, &unused);
        float* kw = dequant_i8_to_float(data + data_start + offsets_[l].kp, 128, &kr, &unused);
        float* vw = dequant_i8_to_float(data + data_start + offsets_[l].vp, 128, &vr, &unused);
        int tqkv = qr + kr + vr;
        layers_[l].qkv.resize((size_t)H * tqkv);
        // dequant returns (out_features, in_features) = (qr, H=1024).
        // GEMM needs weight[in_feat=H][out_feat=tqkv], so transpose:
        // weight[k][i] = qw[i][k], kw[i-qr][k], vw[i-qr-kr][k]
        for (int k = 0; k < H; k++) {
            for (int i = 0; i < qr; i++)
                layers_[l].qkv[(size_t)k * tqkv + i] = qw[(size_t)i * H + k];
            for (int i = 0; i < kr; i++)
                layers_[l].qkv[(size_t)k * tqkv + qr + i] = kw[(size_t)i * H + k];
            for (int i = 0; i < vr; i++)
                layers_[l].qkv[(size_t)k * tqkv + qr + kr + i] = vw[(size_t)i * H + k];
        }
        free(qw); free(kw); free(vw);

        float* ow = dequant_i8_to_float(data + data_start + offsets_[l].op, 256, &or_, &unused);
        // O projection: dequant hardcodes n_tile_cols=4 (in_feat=1024) but
        // O has in_feat=2048. Need to fix tile layout.
        layers_[l].o.resize((size_t)or_ * H);
        {
            int tile_cols = or_ / 256;  // 8
            for (int k = 0; k < or_; k++) {
                for (int j = 0; j < H; j++) {
                    int i8 = (j / 32) * tile_cols + (k / 256);
                    int dq_tr = i8 / 4, dq_tc = i8 % 4;
                    layers_[l].o[(size_t)k * H + j] = ow[(size_t)(dq_tr * 32 + j % 32) * H + (dq_tc * 256 + k % 256)];
                }
            }
        }
        free(ow);

        float* gw = dequant_i8_to_float(data + data_start + offsets_[l].gp, 384, &gr, &unused);
        float* uw = dequant_i8_to_float(data + data_start + offsets_[l].up, 384, &ur, &unused);
        int tgu = gr + ur;
        layers_[l].gu.resize((size_t)H * tgu);
        // GU transpose: same as QKV
        for (int k = 0; k < H; k++) {
            for (int i = 0; i < gr; i++)
                layers_[l].gu[(size_t)k * tgu + i] = gw[(size_t)i * H + k];
            for (int i = 0; i < ur; i++)
                layers_[l].gu[(size_t)k * tgu + gr + i] = uw[(size_t)i * H + k];
        }
        free(gw); free(uw);

        float* dw = dequant_i8_to_float(data + data_start + offsets_[l].dp, 384, &dr, &unused);
        // Down projection: same tile-grid issue (in_feat=3072, not 1024)
        layers_[l].d.resize((size_t)dr * H);
        {
            int tile_cols = dr / 256;  // 12
            for (int k = 0; k < dr; k++) {
                for (int j = 0; j < H; j++) {
                    int i8 = (j / 32) * tile_cols + (k / 256);
                    int dq_tr = i8 / 4, dq_tc = i8 % 4;
                    layers_[l].d[(size_t)k * H + j] = dw[(size_t)(dq_tr * 32 + j % 32) * H + (dq_tc * 256 + k % 256)];
                }
            }
        }
        free(dw);

        // Norms are BF16 in the file
        layers_[l].in_norm.resize(H); layers_[l].pa_norm.resize(H);
        layers_[l].q_norm.resize(HD); layers_[l].k_norm.resize(HD);
        auto read_bf16 = [data, data_start](uint64_t off, float* dst, int n) {
            const uint16_t* src = (const uint16_t*)(data + data_start + off);
            for (int i = 0; i < n; i++) {
                union { uint32_t u; float f; } conv;
                conv.u = (uint32_t)src[i] << 16;
                dst[i] = conv.f;
            }
        };
        read_bf16(offsets_[l].in_off, layers_[l].in_norm.data(), H);
        read_bf16(offsets_[l].pa_off, layers_[l].pa_norm.data(), H);
        read_bf16(offsets_[l].qn_off, layers_[l].q_norm.data(), HD);
        read_bf16(offsets_[l].kn_off, layers_[l].k_norm.data(), HD);
    }

    // Final norm (BF16)
    final_norm_.resize(H);
    const uint16_t* fn_bf16 = (const uint16_t*)(data + data_start + fn_off);
    for (int i = 0; i < H; i++) {
        union { uint32_t u; float f; } conv;
        conv.u = (uint32_t)fn_bf16[i] << 16;
        final_norm_[i] = conv.f;
    }

    int lm_r, lm_c;
    float* lm = dequant_i8_to_float(data + data_start + lm_off, 18992, &lm_r, &lm_c);
    lm_head_.resize((size_t)H * NV);
    for (int k = 0; k < H; k++)
        for (int n = 0; n < NV; n++)
            lm_head_[k * NV + n] = lm[n * H + k];
    free(lm);

    printf("[ROCm] Model: %d layers, H=%d, IM=%d, V=%d\n", NC, H, IM, NV);
    return true;
}

// ============================================================
// CPU math helpers
// ============================================================
inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

void rms_norm_f(float* x, const float* w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(x[i])) x[i] = 0.0f;
        ss += (double)x[i] * x[i];
    }
    float ir = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] = x[i] * ir * w[i];
}

void softmax_f(float* x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mx;
        d = (d > 80.0f) ? 80.0f : ((d < -80.0f) ? -80.0f : d);
        x[i] = expf(d); sum += x[i];
    }
    if (sum <= 0.0) { float iv = 1.0f / n; for (int i = 0; i < n; i++) x[i] = iv; return; }
    float is = 1.0f / (float)sum;
    for (int i = 0; i < n; i++) x[i] *= is;
}

// ============================================================
// GEMM
// ============================================================
void RocmInferenceEngine::gemm(const float* A_host, float* d_B, float* C_host,
                                int M, int K, int N) {
    // Clear any prior HIP errors
    hipGetLastError();

    if (M == 1) {
        hipMemcpy(d_hidden_, A_host, (size_t)K * sizeof(float), hipMemcpyHostToDevice);
        hipError_t e1 = hipGetLastError();
        if (e1 != hipSuccess) {
            fprintf(stderr, "HIP gemm H2D(%d): %s\n", K, hipGetErrorString(e1));
            return;
        }
        int bs = 256;
        int gs = (int)(((size_t)N + bs - 1) / bs);
        hipLaunchKernelGGL(gemv_kernel, gs, bs, 0, 0, d_hidden_, d_B, d_logits_, K, N);
        hipError_t e2 = hipDeviceSynchronize();
        if (e2 != hipSuccess) {
            fprintf(stderr, "HIP gemm kernel<%d,%d>(K=%d,N=%d): %s\n", gs, bs, K, N, hipGetErrorString(e2));
            return;
        }
        hipMemcpy(C_host, d_logits_, (size_t)N * sizeof(float), hipMemcpyDeviceToHost);
        hipError_t e3 = hipGetLastError();
        if (e3 != hipSuccess) {
            fprintf(stderr, "HIP gemm D2H(%d): %s\n", N, hipGetErrorString(e3));
            return;
        }
    } else {
        hipMemcpy(d_hidden_, A_host, (size_t)M * K * sizeof(float), hipMemcpyHostToDevice);
        hipError_t e1 = hipGetLastError();
        if (e1 != hipSuccess) { fprintf(stderr, "HIP gemm H2D(M=%d,K=%d): %s\n", M, K, hipGetErrorString(e1)); return; }
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0,
                           d_hidden_, d_B, d_logits_, M, K, N);
        hipError_t e2 = hipDeviceSynchronize();
        if (e2 != hipSuccess) { fprintf(stderr, "HIP gemm kernel: %s\n", hipGetErrorString(e2)); return; }
        hipMemcpy(C_host, d_logits_, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost);
        hipError_t e3 = hipGetLastError();
        if (e3 != hipSuccess) { fprintf(stderr, "HIP gemm D2H(%d): %s\n", M*N, hipGetErrorString(e3)); return; }
    }
}

// ============================================================
// forward_layer
// ============================================================
void RocmInferenceEngine::forward_layer(int l, int pos) {
    auto& layer = layers_[l];
    auto& kv = kv_cache_[l];

    memcpy(residual_.data(), hidden_.data(), H * 4);
    rms_norm_f(hidden_.data(), layer.in_norm.data(), H);

    gemm(hidden_.data(), layers_[l].d_qkv, q_out_.data(), 1, H, 4096);

    memcpy(k_out_.data(), &q_out_[2048], NKV * HD * 4);
    memcpy(v_out_.data(), &q_out_[3072], NKV * HD * 4);

    for (int h = 0; h < NH; h++) {
        float* hq = &q_out_[h * HD];
        double sq = 0.0;
        for (int d = 0; d < HD; d++) sq += (double)hq[d] * hq[d];
        float iq = 1.0f / sqrtf((float)(sq / HD) + EPS);
        for (int d = 0; d < HD; d++) hq[d] *= iq * layer.q_norm[d];
        if (h % GQA == 0) {
            int kvh = h / GQA;
            float* hk = &k_out_[kvh * HD];
            double sk = 0.0;
            for (int d = 0; d < HD; d++) sk += (double)hk[d] * hk[d];
            float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
            for (int d = 0; d < HD; d++) hk[d] *= ik * layer.k_norm[d];
        }
    }

    for (int h = 0; h < NH; h++) {
        float* hq = &q_out_[h * HD];
        for (int d = 0; d < HD; d += 2) {
            float a = hq[d], b = hq[d+1];
            float c = g_rope_cos[pos * HD + d];
            float s = g_rope_sin[pos * HD + d];
            hq[d] = a*c - b*s; hq[d+1] = b*c + a*s;
        }
    }
    for (int h = 0; h < NKV; h++) {
        float* hk = &k_out_[h * HD];
        for (int d = 0; d < HD; d += 2) {
            float a = hk[d], b = hk[d+1];
            float c = g_rope_cos[pos * HD + d];
            float s = g_rope_sin[pos * HD + d];
            hk[d] = a*c - b*s; hk[d+1] = b*c + a*s;
        }
    }

    memcpy(&kv.k[pos * NKV * HD], k_out_.data(), NKV * HD * 4);
    memcpy(&kv.v[pos * NKV * HD], v_out_.data(), NKV * HD * 4);
    kv.len = pos + 1;
    int cl = kv.len;

    for (int h = 0; h < NH; h++) {
        int kvh = h / GQA;
        float* hq = &q_out_[h * HD];
        float* hk = &kv.k[kvh * HD];
        float* hv = &kv.v[kvh * HD];
        float* sc = attn_scores_.data();
        for (int p = 0; p < cl; p++) {
            double s = 0.0;
            for (int d = 0; d < HD; d++)
                s += (double)hq[d] * hk[p * NKV * HD + d];
            sc[p] = (float)(s / sqrtf((float)HD));
        }
        softmax_f(sc, cl);
        for (int d = 0; d < HD; d++) {
            float s = 0.0f;
            for (int p = 0; p < cl; p++) s += sc[p] * hv[p * NKV * HD + d];
            attn_out_[h * HD + d] = s;
        }
    }

    gemm(attn_out_.data(), layers_[l].d_o, o_out_.data(), 1, NH * HD, H);
    for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + o_out_[i];
    memcpy(residual_.data(), hidden_.data(), H * 4);
    rms_norm_f(hidden_.data(), layer.pa_norm.data(), H);

    gemm(hidden_.data(), layers_[l].d_gu, gate_.data(), 1, H, 6144);
    for (int i = 0; i < IM; i++) {
        float g = gate_[i];
        if (!std::isfinite(g)) g = 0.0f;
        silu_out_[i] = silu_f(g) * gate_[IM + i];
    }
    gemm(silu_out_.data(), layers_[l].d_down, down_.data(), 1, IM, H);
    for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + down_[i];
}

// ============================================================
// generate
// ============================================================
int RocmInferenceEngine::generate(const int* input_tokens, int num_input_tokens,
                                    int* output_tokens, int max_output_tokens) {
    if (!initialized_) { fprintf(stderr, "[ROCm] Not initialized\n"); return -1; }

    printf("[ROCm] Generate: %d in, %d max out\n", num_input_tokens, max_output_tokens);
    auto gen_start = std::chrono::steady_clock::now();

    int pos = 0;
    for (int pi = 0; pi < num_input_tokens; pi++) {
        memcpy(hidden_.data(), &embed_table_[input_tokens[pi] * H], H * 4);
        for (int l = 0; l < NC; l++) forward_layer(l, pos);
        pos++;
        if (pi % 10 == 0) printf("[ROCm]  Prefill %d/%d\n", pi+1, num_input_tokens);
    }
    printf("[ROCm] Prefill: %d tokens\n", pos);

    int ngen = 0;
    for (int g = 0; g < max_output_tokens; g++) {
        auto ts = std::chrono::steady_clock::now();

        rms_norm_f(hidden_.data(), final_norm_.data(), H);
        gemm(hidden_.data(), d_lm_head_, logits_.data(), 1, H, NV);

        float mx = logits_[0];
        for (int i = 1; i < NV; i++) if (logits_[i] > mx) mx = logits_[i];
        double sum = 0.0;
        for (int i = 0; i < NV; i++) {
            float d = logits_[i] - mx;
            d = (d > 80.0f) ? 80.0f : ((d < -80.0f) ? -80.0f : d);
            logits_[i] = expf(d); sum += logits_[i];
        }
        float r = (float)rand() / (float)RAND_MAX * (float)sum;
        float acc = 0.0f; int tok = 0;
        for (int i = 0; i < NV; i++) { acc += logits_[i]; if (acc >= r) { tok = i; break; } }

        output_tokens[g] = tok; ngen++;

        memcpy(hidden_.data(), &embed_table_[tok * H], H * 4);
        for (int l = 0; l < NC; l++) forward_layer(l, pos);
        pos++;

        auto te = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(te - ts).count();
        last_token_ms_ = ms;
        total_time_ms_ += ms;
        total_tokens_++;
        printf("[ROCm]  [%d] tok=%d (%.0f ms)\n", g, tok, ms);
    }

    double gen_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gen_start).count();
    printf("[ROCm] Done: %d tok, %.0f ms total, %.0f ms/tok\n",
           ngen, gen_ms, ngen > 0 ? gen_ms / ngen : 0);
    return ngen;
}

// ============================================================
// main
// ============================================================
int main() {
    printf("=== ROCm GPU Inference Engine -- Qwen3-0.6B ===\n\n");

    RocmInferenceEngine engine;
    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";

    if (!engine.init(model_path)) {
        fprintf(stderr, "Init failed\n");
        return 1;
    }

    int prompt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int prompt_len = sizeof(prompt) / sizeof(prompt[0]);
    int output[32];
    int ngen = engine.generate(prompt, prompt_len, output, 4);

    printf("\nOutput tokens:");
    for (int i = 0; i < ngen; i++) printf(" %d", output[i]);
    printf("\n");
    printf("=== Done (%.0f ms/tok) ===\n", engine.avg_token_ms());
    return 0;
}
