/**
 * ROCm GPU Inference Engine for Qwen3-0.6B
 * Uses hipBLAS for GEMM on Radeon 8060S (gfx1151, 40 CU).
 *
 * Part of the therock repo ecosystem (github.com/rocm/therock).
 * Compile with therock's amdclang++ for native gfx1151 codegen.
 *
 * Build (system ROCm runtime + therock device compiler):
 *   cd build
 *   CXX=/path/to/therock/gfx1151-7.13.0/llvm/bin/amdclang++
 *   $CXX -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 \
 *       -I../include -I/home/bcloud/torch2aie/examples \
 *       -I/opt/rocm/include -L/opt/rocm/lib \
 *       ../src/rocm_engine.cpp dequant_q4nx.o \
 *       -lhipblas -lm -o rocm_engine
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
#include <hip/hip_runtime.h>

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);

// ============================================================
// ROCm Error checking macros
// ============================================================
#define HIP_CHECK(cmd) do {                                    \
    hipError_t err = cmd;                                      \
    if (err != hipSuccess) {                                   \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n",     \
                __FILE__, __LINE__, hipGetErrorString(err), err); \
        return;                                                \
    }                                                          \
} while(0)

#define HIP_CHECK_BOOL(cmd) do {                               \
    hipError_t err = cmd;                                      \
    if (err != hipSuccess) {                                   \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n",     \
                __FILE__, __LINE__, hipGetErrorString(err), err); \
        return false;                                          \
    }                                                          \
} while(0)

// No hipBLAS dependency — raw HIP GEMM kernel instead

// ============================================================
// Helper: find JSON value by name (same format as npu_engine_i8.cpp)
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
// RocmInferenceEngine: Constructor / Destructor
// ============================================================
RocmInferenceEngine::RocmInferenceEngine()
    : d_handle_(nullptr), initialized_(false) {
    kv_cache_.resize(NC);
    for (auto& kv : kv_cache_) {
        kv.k.resize(4096 * NKV * HD, 0.0f);
        kv.v.resize(4096 * NKV * HD, 0.0f);
        kv.len = 0;
    }
}

RocmInferenceEngine::~RocmInferenceEngine() {
    // no hipBLAS handle to destroy
    // GPU buffers freed by hipblasDestroy / vector destructors
}

// ============================================================
// init
// ============================================================
bool RocmInferenceEngine::init(const char* model_path) {
    printf("[ROCm] Engine init: model=%s\n", model_path);

    // --- GPU discovery ---
    HIP_CHECK_BOOL(hipSetDevice(0));
    int dev_count = 0;
    HIP_CHECK_BOOL(hipGetDeviceCount(&dev_count));
    if (dev_count < 1) {
        fprintf(stderr, "[ROCm] No GPU found\n");
        return false;
    }

    hipDeviceProp_t props;
    HIP_CHECK_BOOL(hipGetDeviceProperties(&props, 0));
    printf("[ROCm] Device: %s (arch %d.%d, %d CUs, %.0f MB VRAM)\n",
           props.name, props.major, props.minor,
           props.multiProcessorCount,
           (double)props.totalGlobalMem / (1024.0 * 1024.0));

    // --- hipBLAS handle ---
    // No hipBLAS — using raw HIP kernels instead

    // --- Load model ---
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[ROCm] Cannot open %s\n", model_path); return false; }
    struct stat st;
    fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    bool ok = load_model(md, st.st_size);
    munmap(md, st.st_size);
    if (!ok) return false;

    // --- Host buffers ---
    hidden_.resize(H);
    residual_.resize(H);
    q_out_.resize(4096);            // QKV fused: Q(2048) + K(1024) + V(1024) = 4096
    k_out_.resize(NKV * HD);      // 1024
    v_out_.resize(NKV * HD);      // 1024
    attn_scores_.resize(4096);
    attn_out_.resize(NH * HD);    // 2048
    o_out_.resize(H);             // 1024
    gate_.resize(6144);             // GU fused: Gate(3072) + Up(3072) = 6144
    up_.resize(IM);               // 3072
    silu_out_.resize(IM);         // 3072
    down_.resize(H);              // 1024
    logits_.resize(NV);           // 151936

    printf("[ROCm] Engine ready.\n");
    initialized_ = true;
    return true;
}

// ============================================================
// load_model — parse .q4nx, dequant weights with dequant_i8_to_float
// ============================================================
bool RocmInferenceEngine::load_model(const uint8_t* data, size_t size) {
    uint64_t hdr_len;
    memcpy(&hdr_len, data, 8);
    uint64_t data_start = 8 + hdr_len;
    const char* json = (const char*)(data + 8);
    size_t json_len = hdr_len;

    // Embedding table (float32 after header)
    const float* raw = (const float*)(data + data_start);
    embed_table_.resize(NV * H);
    for (int i = 0; i < NV * H; i++) embed_table_[i] = raw[i];

    // Parse offsets
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

    // Dequant & pack per-layer weights
    layers_.resize(NC);
    for (int l = 0; l < NC; l++) {
        int qr, kr, vr, or_, gr, ur, dr, unused;
        // QKV
        float* qw = dequant_i8_to_float(data + data_start + offsets_[l].qp, 256, &qr, &unused);
        float* kw = dequant_i8_to_float(data + data_start + offsets_[l].kp, 128, &kr, &unused);
        float* vw = dequant_i8_to_float(data + data_start + offsets_[l].vp, 128, &vr, &unused);
        int tqkv = qr + kr + vr;
        layers_[l].qkv.resize((size_t)H * tqkv);
        for (int k = 0; k < H; k++) {
            memcpy(&layers_[l].qkv[k * tqkv], &qw[k * qr], qr * 4);
            memcpy(&layers_[l].qkv[k * tqkv + qr], &kw[k * kr], kr * 4);
            memcpy(&layers_[l].qkv[k * tqkv + qr + kr], &vw[k * vr], vr * 4);
        }
        free(qw); free(kw); free(vw);
        // O
        float* ow = dequant_i8_to_float(data + data_start + offsets_[l].op, 256, &or_, &unused);
        layers_[l].o.resize((size_t)or_ * H);
        memcpy(layers_[l].o.data(), ow, or_ * H * 4);
        free(ow);
        // GU
        float* gw = dequant_i8_to_float(data + data_start + offsets_[l].gp, 384, &gr, &unused);
        float* uw = dequant_i8_to_float(data + data_start + offsets_[l].up, 384, &ur, &unused);
        int tgu = gr + ur;
        layers_[l].gu.resize((size_t)H * tgu);
        for (int k = 0; k < H; k++) {
            memcpy(&layers_[l].gu[k * tgu], &gw[k * gr], gr * 4);
            memcpy(&layers_[l].gu[k * tgu + gr], &uw[k * ur], ur * 4);
        }
        free(gw); free(uw);
        // D
        float* dw = dequant_i8_to_float(data + data_start + offsets_[l].dp, 384, &dr, &unused);
        layers_[l].d.resize((size_t)dr * H);
        memcpy(layers_[l].d.data(), dw, dr * H * 4);
        free(dw);
        // Norms
        layers_[l].in_norm.assign(H, 0); layers_[l].pa_norm.assign(H, 0);
        layers_[l].q_norm.assign(HD, 0); layers_[l].k_norm.assign(HD, 0);
        memcpy(layers_[l].in_norm.data(), data + data_start + offsets_[l].in_off, H * 4);
        memcpy(layers_[l].pa_norm.data(), data + data_start + offsets_[l].pa_off, H * 4);
        memcpy(layers_[l].q_norm.data(), data + data_start + offsets_[l].qn_off, HD * 4);
        memcpy(layers_[l].k_norm.data(), data + data_start + offsets_[l].kn_off, HD * 4);
    }

    // Final norm
    final_norm_.resize(H);
    memcpy(final_norm_.data(), data + data_start + fn_off, H * 4);

    // LM head: dequant as (NV, H), transpose to (H, NV) for row-major C = A × B
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
// Math helpers (CPU-side)
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

// RoPE tables
static std::vector<float> rope_cos, rope_sin;
static bool rope_init = false;

static void init_rope_f(int max_pos, int hd) {
    rope_cos.resize(max_pos * hd);
    rope_sin.resize(max_pos * hd);
    for (int p = 0; p < max_pos; p++) {
        for (int d = 0; d < hd; d += 2) {
            float t = 1.0f / powf(1000000.0f, (float)d / (float)hd);
            float a = p * t;
            rope_cos[p * hd + d] = cosf(a);
            rope_sin[p * hd + d] = sinf(a);
            rope_cos[p * hd + d + 1] = cosf(a);
            rope_sin[p * hd + d + 1] = sinf(a);
        }
    }
    rope_init = true;
}

// ============================================================
// GEMM: C[M,N] = A[M,K] × B[K,N]  (CPU for now — GPU path needs proper
// memory pool to avoid malloc/free corruption with 1800+ calls per generation)
// ============================================================
void gemm_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void RocmInferenceEngine::gemm(const float* A, const float* B, float* C,
                                int M, int K, int N) {
    gemm_cpu(A, B, C, M, K, N);
}

// ============================================================
// forward_layer — complete per-layer forward pass
// ============================================================
void RocmInferenceEngine::forward_layer(int l, int pos) {
    auto& layer = layers_[l];
    auto& kv = kv_cache_[l];

    // 1. Save residual
    memcpy(residual_.data(), hidden_.data(), H * 4);

    // 2. RMSNorm
    rms_norm_f(hidden_.data(), layer.in_norm.data(), H);

    // 3. QKV: hidden[1×H] × W_QKV[H×4096] → q_out[1×4096]
    gemm(hidden_.data(), layer.qkv.data(), q_out_.data(), 1, H, 4096);
    for (int i = 0; i < 4096; i++) if (!std::isfinite(q_out_[i])) q_out_[i] = 0.0f;

    // 4. Split Q(2048), K(1024), V(1024)
    memcpy(k_out_.data(), &q_out_[2048], NKV * HD * 4);
    memcpy(v_out_.data(), &q_out_[3072], NKV * HD * 4);

    // 5. Q/K norms per head
    for (int h = 0; h < NH; h++) {
        float* hq = &q_out_[h * HD];
        double sq = 0.0;
        for (int d = 0; d < HD; d++) sq += (double)hq[d] * hq[d];
        float iq = 1.0f / sqrtf((float)(sq / HD) + 1e-6f);
        for (int d = 0; d < HD; d++) hq[d] *= iq * layer.q_norm[d];

        if (h % GQA == 0) {
            int kvh = h / GQA;
            float* hk = &k_out_[kvh * HD];
            double sk = 0.0;
            for (int d = 0; d < HD; d++) sk += (double)hk[d] * hk[d];
            float ik = 1.0f / sqrtf((float)(sk / HD) + 1e-6f);
            for (int d = 0; d < HD; d++) hk[d] *= ik * layer.k_norm[d];
        }
    }

    // 6. RoPE
    if (!rope_init) init_rope_f(4096, HD);
    for (int h = 0; h < NH; h++) {
        float* hq = &q_out_[h * HD];
        for (int d = 0; d < HD; d += 2) {
            float a = hq[d], b = hq[d+1];
            float c = rope_cos[pos * HD + d];
            float s = rope_sin[pos * HD + d];
            hq[d]   = a*c - b*s;
            hq[d+1] = b*c + a*s;
        }
    }
    for (int h = 0; h < NKV; h++) {
        float* hk = &k_out_[h * HD];
        for (int d = 0; d < HD; d += 2) {
            float a = hk[d], b = hk[d+1];
            float c = rope_cos[pos * HD + d];
            float s = rope_sin[pos * HD + d];
            hk[d]   = a*c - b*s;
            hk[d+1] = b*c + a*s;
        }
    }

    // 7. KV cache
    memcpy(&kv.k[pos * NKV * HD], k_out_.data(), NKV * HD * 4);
    memcpy(&kv.v[pos * NKV * HD], v_out_.data(), NKV * HD * 4);
    kv.len = pos + 1;
    int cl = kv.len;

    // 8. Attention (CPU — softmax is bandwidth-bound, not compute-bound)
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
            for (int p = 0; p < cl; p++)
                s += sc[p] * hv[p * NKV * HD + d];
            attn_out_[h * HD + d] = s;
        }
    }

    // 9. O projection: attn[1×2048] × W_O[2048×H] → o_out[1×H]
    gemm(attn_out_.data(), layer.o.data(), o_out_.data(), 1, NH * HD, H);
    for (int i = 0; i < H; i++) if (!std::isfinite(o_out_[i])) o_out_[i] = 0.0f;

    // 10. Residual
    for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + o_out_[i];
    memcpy(residual_.data(), hidden_.data(), H * 4);

    // 11. Post-attention RMSNorm
    rms_norm_f(hidden_.data(), layer.pa_norm.data(), H);

    // 12. Gate+Up: hidden[1×H] × W_GU[H×6144] → gate[1×6144]
    gemm(hidden_.data(), layer.gu.data(), gate_.data(), 1, H, 6144);
    for (int i = 0; i < 6144; i++) if (!std::isfinite(gate_[i])) gate_[i] = 0.0f;

    // 13. SwiGLU
    for (int i = 0; i < IM; i++) {
        float g = gate_[i];
        if (!std::isfinite(g)) g = 0.0f;
        silu_out_[i] = silu_f(g) * gate_[IM + i];
    }

    // 14. Down: silu[1×IM] × W_D[IM×H] → down[1×H]
    gemm(silu_out_.data(), layer.d.data(), down_.data(), 1, IM, H);
    for (int i = 0; i < H; i++) if (!std::isfinite(down_[i])) down_[i] = 0.0f;

    // 15. Residual
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

    // Prefill
    int pos = 0;
    for (int pi = 0; pi < num_input_tokens; pi++) {
        memcpy(hidden_.data(), &embed_table_[input_tokens[pi] * H], H * 4);
        for (int l = 0; l < NC; l++) forward_layer(l, pos);
        pos++;
        if (pi % 10 == 0) printf("[ROCm]  Prefill %d/%d\n", pi+1, num_input_tokens);
    }
    printf("[ROCm] Prefill: %d tokens\n", pos);

    // Decode
    int ngen = 0;
    for (int g = 0; g < max_output_tokens; g++) {
        auto ts = std::chrono::steady_clock::now();

        // Final norm
        memcpy(hidden_.data(), hidden_.data(), H * 4);
        rms_norm_f(hidden_.data(), final_norm_.data(), H);

        // LM head: hidden[1×H] × W_LM[H×NV] → logits[1×NV]
        gemm(hidden_.data(), lm_head_.data(), logits_.data(), 1, H, NV);
        for (int i = 0; i < NV; i++) if (!std::isfinite(logits_[i])) logits_[i] = -1e10f;

        // Softmax + sample
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

        // Next token embedding + layers
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
// main — standalone test run
// ============================================================
int main() {
    printf("=== ROCm (therock) Inference Engine — Qwen3-0.6B ===\n\n");

    RocmInferenceEngine engine;
    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";

    if (!engine.init(model_path)) {
        fprintf(stderr, "Init failed\n");
        return 1;
    }

    // Test prompt: <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
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
