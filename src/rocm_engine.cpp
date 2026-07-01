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

// Set to 1 to enable per-operation dumps for layer 0
#define DEBUG_DUMP 1

// Profile: accumulate elapsed time in microseconds
struct PerfTimer {
    std::chrono::steady_clock::time_point start;
    uint64_t& accum;
    PerfTimer(uint64_t& a) : start(std::chrono::steady_clock::now()), accum(a) {}
    ~PerfTimer() {
        auto end = std::chrono::steady_clock::now();
        accum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};

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

    bool ok = false;

    // Detect if model_path is a directory (raw float32 format) or a file (.q4nx)
    struct stat st_path;
    if (stat(model_path, &st_path) != 0) {
        fprintf(stderr, "[ROCm] Cannot access %s\n", model_path);
        return false;
    }
    if (S_ISDIR(st_path.st_mode)) {
        // Raw float32 directory format (exported by tools/safetensors_to_raw)
        ok = load_model_from_raw(model_path);
    } else {
        // .q4nx file format
        int fd = open(model_path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "[ROCm] Cannot open %s\n", model_path); return false; }
        struct stat st;
        fstat(fd, &st);
        uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        ok = load_model(md, st.st_size);
        munmap(md, st.st_size);
    }
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
        // O projection: dequant outputs (or_=2048, H=1024). The GEMM B has
        // shape (K=or_=2048, N=H=1024), which matches the dequant layout.
        // Direct memcpy is correct — no tile remap needed.
        layers_[l].o.resize((size_t)or_ * H);
        memcpy(layers_[l].o.data(), ow, (size_t)or_ * H * 4);
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
        // Down projection: dequant outputs (dr=3072, H=1024). The GEMM B has
        // shape (K=dr=3072, N=H=1024), which matches the dequant layout.
        // Direct memcpy is correct — no tile remap needed.
        layers_[l].d.resize((size_t)dr * H);
        memcpy(layers_[l].d.data(), dw, (size_t)dr * H * 4);
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
        if (l == 0 && DEBUG_DUMP) {
            printf("[DBG] q_norm[0..7]:"); for (int idx=0;idx<8;idx++) printf(" %.6f", layers_[l].q_norm[idx]); printf("\n");
            printf("[DBG] k_norm[0..7]:"); for (int idx=0;idx<8;idx++) printf(" %.6f", layers_[l].k_norm[idx]); printf("\n");
        }
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
// stats helpers for debug dump
// ============================================================
static void dump_vec_stats(const char* label, const float* v, int n, int first_n = 8) {
    if (!DEBUG_DUMP) return;
    double sum = 0, ss = 0;
    float mn = v[0], mx = v[0];
    int nnan = 0, ninf = 0;
    for (int i = 0; i < n; i++) {
        float x = v[i];
        if (std::isnan(x)) { nnan++; continue; }
        if (!std::isfinite(x)) { ninf++; continue; }
        sum += x; ss += (double)x * x;
        if (x < mn) mn = x; if (x > mx) mx = x;
    }
    int nvalid = n - nnan - ninf;
    float mean = (float)(sum / (nvalid > 0 ? nvalid : 1));
    float norm = sqrtf((float)(ss / (nvalid > 0 ? nvalid : 1)));
    int pr = first_n < n ? first_n : n;
    printf("[DBG] %s: n=%d nnan=%d ninf=%d min=%.6f max=%.6f mean=%.6f rms=%.6f first%d=[",
           label, n, nnan, ninf, mn, mx, mean, norm, pr);
    for (int i = 0; i < pr; i++) printf("%s%.6f", i?",":"", v[i]);
    printf("]\n");
}

// ============================================================
// load_model_from_raw — load from pre-exported float32 files
// ============================================================
static std::vector<float> read_f32_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ROCm] Cannot open %s\n", path); return {}; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz % 4 != 0) { fprintf(stderr, "[ROCm] %s: size %ld not multiple of 4\n", path, sz); fclose(f); return {}; }
    std::vector<float> buf(sz / 4);
    if (fread(buf.data(), 4, buf.size(), f) != buf.size()) {
        fprintf(stderr, "[ROCm] Short read from %s\n", path); fclose(f); return {};
    }
    fclose(f);
    return buf;
}

bool RocmInferenceEngine::load_model_from_raw(const char* dir) {
    printf("[ROCm] Loading model from raw float32 directory: %s\n", dir);

    char path[512];

    // Embedding table (NV × H, row-major)
    snprintf(path, 512, "%s/embed.F32", dir);
    auto embed = read_f32_file(path);
    if (embed.empty()) return false;
    embed_table_ = std::move(embed);
    printf("[ROCm]   embed: %zu elements\n", embed_table_.size());

    // LM head (H × NV, already transposed for gemm_kernel layout)
    snprintf(path, 512, "%s/lm_head.F32", dir);
    auto lm_head = read_f32_file(path);
    if (lm_head.empty()) return false;
    lm_head_ = std::move(lm_head);
    printf("[ROCm]   lm_head: %zu elements\n", lm_head_.size());

    // Final norm (H)
    snprintf(path, 512, "%s/final_norm.F32", dir);
    auto fn = read_f32_file(path);
    if (fn.empty()) return false;
    final_norm_ = std::move(fn);
    printf("[ROCm]   final_norm: %zu elements\n", final_norm_.size());

    // Per-layer weights
    layers_.resize(NC);
    for (int l = 0; l < NC; l++) {
        char layer_dir[512];
        snprintf(layer_dir, 512, "%s/layer_%d", dir, l);

        auto& lw = layers_[l];

        // QKV (H × 4096)
        snprintf(path, 512, "%s/qkv.F32", layer_dir);
        auto qkv = read_f32_file(path);
        if (qkv.empty()) return false;
        lw.qkv = std::move(qkv);

        // GU (H × 6144)
        snprintf(path, 512, "%s/gu.F32", layer_dir);
        auto gu = read_f32_file(path);
        if (gu.empty()) return false;
        lw.gu = std::move(gu);

        // O (2048 × 1024)
        snprintf(path, 512, "%s/o.F32", layer_dir);
        auto o = read_f32_file(path);
        if (o.empty()) return false;
        lw.o = std::move(o);

        // Down (3072 × 1024)
        snprintf(path, 512, "%s/down.F32", layer_dir);
        auto down = read_f32_file(path);
        if (down.empty()) return false;
        lw.d = std::move(down);

        // Norms (1D)
        snprintf(path, 512, "%s/in_norm.F32", layer_dir);
        lw.in_norm = read_f32_file(path);
        if (lw.in_norm.empty()) return false;

        snprintf(path, 512, "%s/pa_norm.F32", layer_dir);
        lw.pa_norm = read_f32_file(path);
        if (lw.pa_norm.empty()) return false;

        snprintf(path, 512, "%s/q_norm.F32", layer_dir);
        lw.q_norm = read_f32_file(path);
        if (lw.q_norm.empty()) return false;

        snprintf(path, 512, "%s/k_norm.F32", layer_dir);
        lw.k_norm = read_f32_file(path);
        if (lw.k_norm.empty()) return false;

        printf("[ROCm]   layer %d: qkv=%zu gu=%zu o=%zu down=%zu norms=%zu+%zu+%zu+%zu\n",
               l, lw.qkv.size(), lw.gu.size(), lw.o.size(), lw.d.size(),
               lw.in_norm.size(), lw.pa_norm.size(), lw.q_norm.size(), lw.k_norm.size());
    }

    printf("[ROCm] Raw model loaded successfully.\n");
    return true;
}

// ============================================================
// forward_layer
// ============================================================
void RocmInferenceEngine::forward_layer(int l, int pos) {
    auto& layer = layers_[l];
    auto& kv = kv_cache_[l];
    int dbg = DEBUG_DUMP && l == 0 && pos == 0;

    memcpy(residual_.data(), hidden_.data(), H * 4);
    if (dbg) dump_vec_stats("hidden BEFORE input norm", hidden_.data(), H);

    rms_norm_f(hidden_.data(), layer.in_norm.data(), H);
    if (dbg) dump_vec_stats("hidden AFTER input RMSNorm", hidden_.data(), H);

    gemm(hidden_.data(), layers_[l].d_qkv, q_out_.data(), 1, H, 4096);
    if (dbg) {
        dump_vec_stats("Q_out AFTER QKV GEMM (first 2048)", q_out_.data(), 2048);
        dump_vec_stats("K_out AFTER QKV (2048..3071)", q_out_.data() + 2048, 1024);
        dump_vec_stats("V_out AFTER QKV (3072..4095)", q_out_.data() + 3072, 1024);
    }

    memcpy(k_out_.data(), &q_out_[2048], NKV * HD * 4);
    memcpy(v_out_.data(), &q_out_[3072], NKV * HD * 4);

    if (dbg) {
        // Dump K head 0 BEFORE QK norm
        double sk0 = 0;
        for (int d = 0; d < HD; d++) sk0 += (double)k_out_[d] * k_out_[d];
        printf("[DBG] K head 0 PRE-norm (before QK norm): sum_sq=%.4f mean_sq=%.4f\n", sk0, sk0/HD);
        printf("[DBG] K head 0 first 8 PRE-norm:");
        for (int d = 0; d < 8; d++) printf(" %.6f", k_out_[d]);
        printf("\n");
    }

    { PerfTimer pt(perf_.qk_norm_us);
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
    }
    if (dbg) {
        dump_vec_stats("Q_out AFTER QK norm", q_out_.data(), 2048);
        dump_vec_stats("K_out AFTER QK norm", k_out_.data(), NKV * HD);
        // Show k_norm values
        printf("[DBG] k_norm loaded:"); for (int idx=0;idx<8;idx++) printf(" %.6f", layer.k_norm[idx]); printf("\n");
        // Compute K norm manually for first kv head
        double sk_check = 0;
        for (int d = 0; d < HD; d++) sk_check += (double)k_out_[d] * k_out_[d];
        float ik_check = 1.0f / sqrtf((float)(sk_check / HD) + EPS);
        printf("[DBG] K head 0 pre-norm: sum_sq=%.6f mean_sq=%.6f ik=%.6f\n", sk_check, sk_check/HD, ik_check);
        printf("[DBG] K head 0 first 8 AFTER norm:");
        for (int idx=0;idx<8;idx++) printf(" %.6f", k_out_[idx]);
        printf("\n");
    }

    { PerfTimer pt(perf_.rope_us);
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
    }
    if (dbg) {
        dump_vec_stats("Q_out AFTER RoPE", q_out_.data(), 2048);
        dump_vec_stats("K_out AFTER RoPE", k_out_.data(), NKV * HD);
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
    if (dbg) {
        dump_vec_stats("attn_scores BEFORE softmax (head 0)", attn_scores_.data(), cl, 1);
        dump_vec_stats("attn_out AFTER attention mix", attn_out_.data(), NH * HD);
    }

    gemm(attn_out_.data(), layers_[l].d_o, o_out_.data(), 1, NH * HD, H);
    if (dbg) dump_vec_stats("o_out AFTER O projection GEMM", o_out_.data(), H);

    for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + o_out_[i];
    if (dbg) dump_vec_stats("hidden AFTER +residual (attn)", hidden_.data(), H);

    memcpy(residual_.data(), hidden_.data(), H * 4);
    rms_norm_f(hidden_.data(), layer.pa_norm.data(), H);
    if (dbg) dump_vec_stats("hidden AFTER post-attn RMSNorm", hidden_.data(), H);

    gemm(hidden_.data(), layers_[l].d_gu, gate_.data(), 1, H, 6144);
    if (dbg) {
        dump_vec_stats("gate[0..3071] AFTER GU GEMM", gate_.data(), IM);
        dump_vec_stats("up[3072..6143] AFTER GU GEMM", gate_.data() + IM, IM);
    }

    { PerfTimer pt(perf_.silu_us);
    for (int i = 0; i < IM; i++) {
        float g = gate_[i];
        if (!std::isfinite(g)) g = 0.0f;
        silu_out_[i] = silu_f(g) * gate_[IM + i];
    }
    }
    if (dbg) dump_vec_stats("silu_out AFTER SiLU(gate)*up", silu_out_.data(), IM);

    { PerfTimer pt(perf_.gemm_us);
    gemm(silu_out_.data(), layers_[l].d_down, down_.data(), 1, IM, H);
    }
    if (dbg) dump_vec_stats("down AFTER Down projection GEMM", down_.data(), H);

    { PerfTimer pt(perf_.elementwise_us);
    for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + down_[i];
    }
    if (dbg) dump_vec_stats("hidden FINAL after FFN+residual", hidden_.data(), H);
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

    // Print profiled per-operation breakdown
    uint64_t total_perf =
        perf_.rms_norm_us + perf_.gemm_us + perf_.qk_norm_us +
        perf_.rope_us + perf_.attention_us + perf_.silu_us + perf_.elementwise_us;
    printf("\n=== Perf Profile (%d layers x %d tokens) ===\n", NC, pos);
    printf("  RMSNorm:    %10llu us (%5.1f%%)\n", perf_.rms_norm_us, 100.0 * perf_.rms_norm_us / total_perf);
    printf("  GEMM:       %10llu us (%5.1f%%)\n", perf_.gemm_us, 100.0 * perf_.gemm_us / total_perf);
    printf("  QK Norm:    %10llu us (%5.1f%%)\n", perf_.qk_norm_us, 100.0 * perf_.qk_norm_us / total_perf);
    printf("  RoPE:       %10llu us (%5.1f%%)\n", perf_.rope_us, 100.0 * perf_.rope_us / total_perf);
    printf("  Attention:  %10llu us (%5.1f%%)\n", perf_.attention_us, 100.0 * perf_.attention_us / total_perf);
    printf("  SiLU:       %10llu us (%5.1f%%)\n", perf_.silu_us, 100.0 * perf_.silu_us / total_perf);
    printf("  Elementwise:%10llu us (%5.1f%%)\n", perf_.elementwise_us, 100.0 * perf_.elementwise_us / total_perf);
    printf("  Total:      %10llu us (%5.1f%% of %llu ms)\n",
           total_perf, 100.0 * total_perf / (1000 * (uint64_t)last_token_ms_), (uint64_t)last_token_ms_ * 1000);

    return ngen;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    printf("=== ROCm GPU Inference Engine -- Qwen3-0.6B ===\n\n");

    // Default: use raw float32 weights exported from HF safetensors
    const char* default_model = "/tmp/qwen3_raw";
    const char* model_path = argc > 1 ? argv[1] : default_model;

    RocmInferenceEngine engine;
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
