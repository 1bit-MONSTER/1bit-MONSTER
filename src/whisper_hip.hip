// whisper_hip.hip — GPU-accelerated Whisper forward pass (HIP/ROCm).
//
// Implements the exact scalar math of src/whisper.cpp (whisper_encode /
// whisper_decode_step) as device kernels, with per-model device weight
// caching. The CPU path remains the reference; this module is an
// acceleration layer behind the extern "C" bridge in include/whisper.h.
//
// Kernel notes:
//  - All kernels are single-wave32 (32 threads/block) unless noted — the
//    shuffle reductions assume one wave32 warp (audit #22).
//  - Mel spectrogram is intentionally NOT replaced: the scalar version is
//    cheap and bit-exact (whisper.cpp keeps it).
//  - Numerics: float32, same op order as the scalar path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "whisper.h"

#define WHIP_CHECK(call)                                                        \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr, "[whisper-gpu] %s:%d %s failed: %s\n", __FILE__,    \
                    __LINE__, #call, hipGetErrorString(_e));                    \
            return -1;                                                          \
        }                                                                       \
    } while (0)

// ─── Kernels ────────────────────────────────────────────────────────

// out[C_out, L_out] = conv1d(in[C_in, L_in], w[C_out, C_in, K], stride)
__global__ void conv1d_kernel(float* __restrict__ out, const float* __restrict__ in,
                              const float* __restrict__ w, const float* __restrict__ bias,
                              int C_in, int L_in, int C_out, int K, int stride) {
    int co = blockIdx.x, lo = blockIdx.y;
    int L_out = (L_in - K) / stride + 1;
    if (co >= C_out || lo >= L_out) return;
    float sum = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < C_in; ci++)
        for (int ki = 0; ki < K; ki++)
            sum += in[(size_t)ci * L_in + lo * stride + ki] * w[(size_t)co * C_in * K + (size_t)ci * K + ki];
    out[(size_t)co * L_out + lo] = sum;
}

__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = 0.7978845608f;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

// out[m,n] = a[m,k]·b[n,k] + bias[n]  (row-major, b transposed like whisper's weights)
__global__ void gemm_kernel(float* __restrict__ out, const float* __restrict__ a,
                            const float* __restrict__ b, const float* __restrict__ bias,
                            int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float acc = bias ? bias[n] : 0.0f;
    const float* ar = a + (size_t)m * K;
    const float* br = b + (size_t)n * K;
    for (int k = 0; k < K; k++) acc += ar[k] * br[k];
    out[(size_t)m * N + n] = acc;
}

__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// x[t] := LN(x[t]); in-place safe (row loaded into shared first).
__global__ void layernorm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                 const float* __restrict__ w, const float* __restrict__ b,
                                 int N, int D, float eps) {
    int t = blockIdx.x;
    if (t >= N) return;
    extern __shared__ float s_buf[];
    float* vals = s_buf;
    for (int d = threadIdx.x; d < D; d += blockDim.x) vals[d] = x[(size_t)t * D + d];
    __syncthreads();

    float mean = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) mean += vals[d];
    for (int o = 16; o > 0; o >>= 1) mean += __shfl_xor_sync(0xffffffffULL, mean, o);
    mean /= D;
    __syncthreads();

    float var = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { float df = vals[d] - mean; var += df * df; }
    for (int o = 16; o > 0; o >>= 1) var += __shfl_xor_sync(0xffffffffULL, var, o);
    var /= D;
    __syncthreads();

    float inv = rsqrtf(var + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        out[(size_t)t * D + d] = (vals[d] - mean) * inv * w[d] + (b ? b[d] : 0.0f);
}

// Attention over Q,K,V [N, D] (single head per block y). causal=1 masks s>t.
__global__ void self_attention_kernel(float* __restrict__ out, const float* __restrict__ q,
                                      const float* __restrict__ k, const float* __restrict__ v,
                                      int N, int D, int n_heads, float scale, int causal) {
    int t = blockIdx.x, h = blockIdx.y;
    if (t >= N || h >= n_heads) return;
    int H = D / n_heads;
    extern __shared__ float s_scores[];
    float* scores = s_scores;

    float max_score = -1e30f;
    for (int s = 0; s < N; s++) {
        float acc = 0.0f;
        for (int d = threadIdx.x; d < H; d += blockDim.x)
            acc += q[(size_t)t * D + (size_t)h * H + d] * k[(size_t)s * D + (size_t)h * H + d];
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffULL, acc, o);
        float sc = (causal && s > t) ? -1e30f : acc * scale;
        if (threadIdx.x == 0) { scores[s] = sc; max_score = fmaxf(max_score, sc); }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int s = 0; s < N; s++) { scores[s] = expf(scores[s] - max_score); sum += scores[s]; }
        float inv = 1.0f / (sum + 1e-10f);
        for (int s = 0; s < N; s++) scores[s] *= inv;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < N; s++) acc += scores[s] * v[(size_t)s * D + (size_t)h * H + d];
        out[(size_t)t * D + (size_t)h * H + d] = acc;
    }
}

// Cross-attention: Q [N_dec,D], K/V [N_enc,D]. Same layout as scalar cross_attn.
__global__ void cross_attention_kernel(float* __restrict__ out, const float* __restrict__ q,
                                       const float* __restrict__ k, const float* __restrict__ v,
                                       int N_dec, int N_enc, int D, int n_heads, float scale) {
    int t = blockIdx.x, h = blockIdx.y;
    if (t >= N_dec || h >= n_heads) return;
    int H = D / n_heads;
    extern __shared__ float s_scores[];
    float* scores = s_scores;

    float max_score = -1e30f;
    for (int s = 0; s < N_enc; s++) {
        float acc = 0.0f;
        for (int d = threadIdx.x; d < H; d += blockDim.x)
            acc += q[(size_t)t * D + (size_t)h * H + d] * k[(size_t)s * D + (size_t)h * H + d];
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffULL, acc, o);
        if (threadIdx.x == 0) { scores[s] = acc * scale; max_score = fmaxf(max_score, scores[s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int s = 0; s < N_enc; s++) { scores[s] = expf(scores[s] - max_score); sum += scores[s]; }
        float inv = 1.0f / (sum + 1e-10f);
        for (int s = 0; s < N_enc; s++) scores[s] *= inv;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < N_enc; s++) acc += scores[s] * v[(size_t)s * D + (size_t)h * H + d];
        out[(size_t)t * D + (size_t)h * H + d] = acc;
    }
}

// x[t,d] = h[t,d] + pos[t,d] for t < n_ctx (encoder positional add).
__global__ void posadd_kernel(float* __restrict__ x, const float* __restrict__ h,
                              const float* __restrict__ pos, int n_ctx, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_ctx * D;
    if (i >= total) return;
    int t = i / D, d = i % D;
    x[i] = h[i] + pos[(size_t)t * D + d];
}

// ─── Device weight registry (per model) ────────────────────────────

struct GpuBuf {
    float* d = nullptr;
    size_t n = 0;
};

struct GpuModel {
    std::unordered_map<const void*, GpuBuf> bufs;  // key: host vector .data()
    float* d_enc_out = nullptr;
    int enc_ctx = 0;
};

namespace {
std::unordered_map<const void*, GpuModel*> g_models;
std::mutex g_mtx;
int g_checked = 0, g_gpu_ok = 0;
}

static float* upload_buf(GpuModel& m, const std::vector<float>& v) {
    if (v.empty()) return nullptr;
    const void* key = v.data();
    auto it = m.bufs.find(key);
    if (it != m.bufs.end()) return it->second.d;
    float* d = nullptr;
    hipError_t e = hipMalloc(&d, v.size() * sizeof(float));
    if (e != hipSuccess) {
        fprintf(stderr, "[whisper-gpu] upload hipMalloc(%zu) failed: %s\n",
                v.size() * sizeof(float), hipGetErrorString(e));
        return nullptr;
    }
    hipMemcpy(d, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice);
    m.bufs[key] = {d, v.size()};
    return d;
}

static GpuModel* model_ctx(const WhisperModel* model) {
    std::lock_guard<std::mutex> lock(g_mtx);
    auto it = g_models.find(model);
    if (it != g_models.end()) return it->second;
    auto* m = new GpuModel;
    g_models[model] = m;
    return m;
}

static int alloc_buf(float** d, size_t nbytes) {
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        fprintf(stderr, "[whisper-gpu] pending error before hipMalloc(%zu): %s\n", nbytes, hipGetErrorString(e));
    }
    e = hipMalloc(d, nbytes);
    if (e != hipSuccess) {
        fprintf(stderr, "[whisper-gpu] hipMalloc(%zu) failed: %s\n", nbytes, hipGetErrorString(e));
        return -1;
    }
    return 0;
}

// ─── Encoder ───────────────────────────────────────────────────────

static int gpu_encode(GpuModel& gm, const WhisperModel& model, const float* mel, int n_frames,
                      float* out_x, int* out_ctx) {
    const auto& cfg = model.cfg;
    const int DA = cfg.n_audio_state;
    const int n_mels = cfg.n_mels;
    const int c1_out = n_frames - 3 + 1;
    const int c2_out = (c1_out - 3) / 2 + 1;
    const int n_ctx = std::min(c2_out, cfg.n_audio_ctx);

    const float* d_conv1w = upload_buf(gm, model.enc_conv1_w);
    const float* d_conv1b = upload_buf(gm, model.enc_conv1_b);
    const float* d_conv2w = upload_buf(gm, model.enc_conv2_w);
    const float* d_conv2b = upload_buf(gm, model.enc_conv2_b);
    const float* d_pos    = upload_buf(gm, model.enc_pos_emb);
    const float* d_ln_w   = upload_buf(gm, model.enc_ln_w);
    const float* d_ln_b   = upload_buf(gm, model.enc_ln_b);
    if (!d_conv1w || !d_pos) return -1;

    float *d_mel = nullptr, *d_h1 = nullptr, *d_h2 = nullptr, *d_x = nullptr, *d_t = nullptr;
    float *d_attn = nullptr, *d_gate = nullptr, *d_up = nullptr;
    if (alloc_buf(&d_mel, (size_t)n_mels * n_frames * 4)) return -1;
    if (alloc_buf(&d_h1, (size_t)DA * c1_out * 4)) return -1;
    if (alloc_buf(&d_h2, (size_t)DA * c2_out * 4)) return -1;
    if (alloc_buf(&d_x, (size_t)n_ctx * DA * 4)) return -1;
    if (alloc_buf(&d_t, (size_t)n_ctx * DA * 4)) return -1;
    if (alloc_buf(&d_attn, (size_t)n_ctx * DA * 4)) return -1;
    if (alloc_buf(&d_gate, (size_t)n_ctx * DA * 4 * 4)) return -1;
    if (alloc_buf(&d_up, (size_t)n_ctx * DA * 4)) return -1;

    WHIP_CHECK(hipMemcpy(d_mel, mel, (size_t)n_mels * n_frames * 4, hipMemcpyHostToDevice));

    conv1d_kernel<<<dim3(DA, c1_out), 1>>>(d_h1, d_mel, d_conv1w, d_conv1b, n_mels, n_frames, DA, 3, 1);
    gelu_kernel<<<(DA * c1_out + 255) / 256, 256>>>(d_h1, DA * c1_out);
    conv1d_kernel<<<dim3(DA, c2_out), 1>>>(d_h2, d_h1, d_conv2w, d_conv2b, DA, c1_out, DA, 3, 2);
    gelu_kernel<<<(DA * c2_out + 255) / 256, 256>>>(d_h2, DA * c2_out);
    posadd_kernel<<<(n_ctx * DA + 255) / 256, 256>>>(d_x, d_h2, d_pos, n_ctx, DA);

    const int head = cfg.n_audio_head;
    const float scale = 1.0f / sqrtf((float)(DA / head));
    const dim3 attn_grid(n_ctx, head), ln_grid(n_ctx), gemm_grid((DA + 15) / 16, (DA + 15) / 16);
    const dim3 b16(16, 16);

    for (int il = 0; il < cfg.n_audio_layer; il++) {
        auto& l = model.enc_layers[il];
        const float* qw = upload_buf(gm, l.attn_q_w), *qb = upload_buf(gm, l.attn_q_b);
        const float* kw = upload_buf(gm, l.attn_k_w), *kb = upload_buf(gm, l.attn_k_b);
        const float* vw = upload_buf(gm, l.attn_v_w), *vb = upload_buf(gm, l.attn_v_b);
        const float* ow = upload_buf(gm, l.attn_o_w), *ob = upload_buf(gm, l.attn_o_b);
        const float* gw = upload_buf(gm, l.ffn_gate_w), *gb = upload_buf(gm, l.ffn_gate_b);
        const float* dw = upload_buf(gm, l.ffn_down_w), *db = upload_buf(gm, l.ffn_down_b);
        const float* lw = upload_buf(gm, l.attn_ln_w), *lb = upload_buf(gm, l.attn_ln_b);
        const float* fw = upload_buf(gm, l.ffn_ln_w), *fb = upload_buf(gm, l.ffn_ln_b);
        if (!qw || !gw || !dw) return -1;

        layernorm_kernel<<<ln_grid, 32, (size_t)DA * 4>>>(d_x, d_x, lw, lb, n_ctx, DA, 1e-5f);
        gemm_kernel<<<gemm_grid, b16>>>(d_t, d_x, qw, qb, n_ctx, DA, DA);
        gemm_kernel<<<gemm_grid, b16>>>(d_attn, d_x, kw, kb, n_ctx, DA, DA);  // reused as K
        gemm_kernel<<<gemm_grid, b16>>>(d_gate, d_x, vw, vb, n_ctx, DA, DA);  // reused as V
        self_attention_kernel<<<attn_grid, 32, (size_t)n_ctx * 4>>>(d_t, d_t, d_attn, d_gate,
                                                                    n_ctx, DA, head, scale, 0);
        gemm_kernel<<<gemm_grid, b16>>>(d_attn, d_t, ow, ob, n_ctx, DA, DA);
        add_kernel<<<(n_ctx * DA + 255) / 256, 256>>>(d_x, d_attn, n_ctx * DA);

        layernorm_kernel<<<ln_grid, 32, (size_t)DA * 4>>>(d_x, d_x, fw, fb, n_ctx, DA, 1e-5f);
        gemm_kernel<<<gemm_grid, b16>>>(d_gate, d_x, gw, gb, n_ctx, DA * 4, DA);
        gelu_kernel<<<(n_ctx * DA * 4 + 255) / 256, 256>>>(d_gate, n_ctx * DA * 4);
        gemm_kernel<<<dim3((DA + 15) / 16, (DA * 4 + 15) / 16), b16>>>(d_up, d_gate, dw, db,
                                                                       n_ctx, DA, DA * 4);
        add_kernel<<<(n_ctx * DA + 255) / 256, 256>>>(d_x, d_up, n_ctx * DA);
        hipError_t _err = hipGetLastError();
        if (_err != hipSuccess) {
            fprintf(stderr, "[whisper-gpu] encoder layer %d kernel error: %s\n",
                    il, hipGetErrorString(_err));
            return -1;
        }
    }
    layernorm_kernel<<<ln_grid, 32, (size_t)DA * 4>>>(d_x, d_x, d_ln_w, d_ln_b, n_ctx, DA, 1e-5f);

    WHIP_CHECK(hipMemcpy(out_x, d_x, (size_t)n_ctx * DA * 4, hipMemcpyDeviceToHost));
    *out_ctx = n_ctx;

    // Keep encoder output resident for decode steps.
    if (!gm.d_enc_out) {
        if (alloc_buf(&gm.d_enc_out, (size_t)n_ctx * DA * 4)) return -1;
    }
    WHIP_CHECK(hipMemcpy(gm.d_enc_out, d_x, (size_t)n_ctx * DA * 4, hipMemcpyDeviceToDevice));
    gm.enc_ctx = n_ctx;

    (void)hipFree(d_mel); (void)hipFree(d_h1); (void)hipFree(d_h2); (void)hipFree(d_x);
    (void)hipFree(d_t); (void)hipFree(d_attn); (void)hipFree(d_gate); (void)hipFree(d_up);
    return 0;
}

// ─── Decoder step ──────────────────────────────────────────────────

static int gpu_decode_step(GpuModel& gm, const WhisperModel& model, const int* tokens, int n_tokens,
                           float* out_logits) {
    const auto& cfg = model.cfg;
    const int DT = cfg.n_text_state;
    const int N = n_tokens;
    const int vocab = cfg.n_vocab;

    // Embedding gather (CPU: N is small, weights live in host vectors).
    std::vector<float> x_h((size_t)N * DT);
    for (int t = 0; t < N; t++) {
        int tok = tokens[t];
        if (tok >= 0 && tok < vocab) {
            for (int d = 0; d < DT; d++)
                x_h[(size_t)t * DT + d] = model.dec_token_emb[(size_t)tok * DT + d] +
                                          model.dec_pos_emb[(size_t)t * DT + d];
        }
    }

    const float* d_emb = upload_buf(gm, model.dec_token_emb);
    const float* d_outw = upload_buf(gm, model.output_w);
    const float* d_ln_w = upload_buf(gm, model.dec_ln_w);
    const float* d_ln_b = upload_buf(gm, model.dec_ln_b);
    if (!d_emb || !d_ln_w) return -1;
    if (!gm.d_enc_out) {
        fprintf(stderr, "[whisper-gpu] decode before encode\n");
        return -1;
    }

    float *d_x = nullptr, *d_t = nullptr, *d_qk = nullptr, *d_attn = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_logits = nullptr;
    if (alloc_buf(&d_x, (size_t)N * DT * 4)) return -1;
    if (alloc_buf(&d_t, (size_t)N * DT * 4)) return -1;
    if (alloc_buf(&d_qk, (size_t)std::max(N, gm.enc_ctx) * DT * 4)) return -1;
    if (alloc_buf(&d_attn, (size_t)N * DT * 4)) return -1;
    if (alloc_buf(&d_gate, (size_t)N * DT * 4 * 4)) return -1;
    if (alloc_buf(&d_up, (size_t)N * DT * 4)) return -1;
    if (alloc_buf(&d_logits, (size_t)vocab * 4)) return -1;

    WHIP_CHECK(hipMemcpy(d_x, x_h.data(), (size_t)N * DT * 4, hipMemcpyHostToDevice));

    const int head = cfg.n_text_head;
    const float scale = 1.0f / sqrtf((float)(DT / head));
    const dim3 ln_grid(N), gemm_grid((DT + 15) / 16, (DT + 15) / 16);
    const dim3 b16(16, 16);
    const dim3 attn_grid(N, head), cross_grid(N, head);

    for (int il = 0; il < cfg.n_text_layer; il++) {
        auto& l = model.dec_layers[il];
        const float* qw = upload_buf(gm, l.attn_q_w), *qb = upload_buf(gm, l.attn_q_b);
        const float* kw = upload_buf(gm, l.attn_k_w), *kb = upload_buf(gm, l.attn_k_b);
        const float* vw = upload_buf(gm, l.attn_v_w), *vb = upload_buf(gm, l.attn_v_b);
        const float* ow = upload_buf(gm, l.attn_o_w), *ob = upload_buf(gm, l.attn_o_b);
        const float* cqw = upload_buf(gm, l.cross_q_w), *cqb = upload_buf(gm, l.cross_q_b);
        const float* ckw = upload_buf(gm, l.cross_k_w), *ckb = upload_buf(gm, l.cross_k_b);
        const float* cvw = upload_buf(gm, l.cross_v_w), *cvb = upload_buf(gm, l.cross_v_b);
        const float* cow = upload_buf(gm, l.cross_o_w), *cob = upload_buf(gm, l.cross_o_b);
        const float* gw = upload_buf(gm, l.ffn_gate_w), *gb = upload_buf(gm, l.ffn_gate_b);
        const float* dw = upload_buf(gm, l.ffn_down_w), *db = upload_buf(gm, l.ffn_down_b);
        const float* lw = upload_buf(gm, l.attn_ln_w), *lb = upload_buf(gm, l.attn_ln_b);
        const float* clw = upload_buf(gm, l.cross_ln_w), *clb = upload_buf(gm, l.cross_ln_b);
        const float* fw = upload_buf(gm, l.ffn_ln_w), *fb = upload_buf(gm, l.ffn_ln_b);
        if (!qw || !cqw || !gw || !dw) return -1;

        // Self-attention (causal)
        layernorm_kernel<<<ln_grid, 32, (size_t)DT * 4>>>(d_x, d_x, lw, lb, N, DT, 1e-5f);
        gemm_kernel<<<gemm_grid, b16>>>(d_t, d_x, qw, qb, N, DT, DT);
        gemm_kernel<<<gemm_grid, b16>>>(d_qk, d_x, kw, kb, N, DT, DT);
        gemm_kernel<<<gemm_grid, b16>>>(d_attn, d_x, vw, vb, N, DT, DT);
        self_attention_kernel<<<attn_grid, 32, (size_t)N * 4>>>(d_t, d_t, d_qk, d_attn, N, DT, head, scale, 1);
        gemm_kernel<<<gemm_grid, b16>>>(d_attn, d_t, ow, ob, N, DT, DT);
        add_kernel<<<(N * DT + 255) / 256, 256>>>(d_x, d_attn, N * DT);

        // Cross-attention
        layernorm_kernel<<<ln_grid, 32, (size_t)DT * 4>>>(d_x, d_x, clw, clb, N, DT, 1e-5f);
        gemm_kernel<<<gemm_grid, b16>>>(d_t, d_x, cqw, cqb, N, DT, DT);
        gemm_kernel<<<dim3((DT + 15) / 16, (DT + 15) / 16), b16>>>(d_qk, gm.d_enc_out, ckw, ckb,
                                                                   gm.enc_ctx, DT, DT);
        gemm_kernel<<<dim3((DT + 15) / 16, (DT + 15) / 16), b16>>>(d_attn, gm.d_enc_out, cvw, cvb,
                                                                   gm.enc_ctx, DT, DT);
        cross_attention_kernel<<<cross_grid, 32, (size_t)gm.enc_ctx * 4>>>(d_t, d_t, d_qk, d_attn,
                                                                           N, gm.enc_ctx, DT, head, scale);
        gemm_kernel<<<gemm_grid, b16>>>(d_attn, d_t, cow, cob, N, DT, DT);
        add_kernel<<<(N * DT + 255) / 256, 256>>>(d_x, d_attn, N * DT);

        // FFN
        layernorm_kernel<<<ln_grid, 32, (size_t)DT * 4>>>(d_x, d_x, fw, fb, N, DT, 1e-5f);
        gemm_kernel<<<dim3((DT + 15) / 16, (DT * 4 + 15) / 16), b16>>>(d_gate, d_x, gw, gb, N, DT * 4, DT);
        gelu_kernel<<<(N * DT * 4 + 255) / 256, 256>>>(d_gate, N * DT * 4);
        gemm_kernel<<<dim3((DT + 15) / 16, (DT * 4 + 15) / 16), b16>>>(d_up, d_gate, dw, db, N, DT, DT * 4);
        add_kernel<<<(N * DT + 255) / 256, 256>>>(d_x, d_up, N * DT);
    }

    // Final LN on last token + output projection
    layernorm_kernel<<<dim3(1), 32, (size_t)DT * 4>>>(d_t, d_x + (size_t)(N - 1) * DT, d_ln_w, d_ln_b, 1, DT, 1e-5f);
    const float* logits_w = d_outw ? d_outw : d_emb;  // tied embeddings fallback
    gemm_kernel<<<dim3((vocab + 15) / 16, 1), b16>>>(d_logits, d_t, logits_w, nullptr, 1, vocab, DT);
    WHIP_CHECK(hipMemcpy(out_logits, d_logits, (size_t)vocab * 4, hipMemcpyDeviceToHost));

    (void)hipFree(d_x); (void)hipFree(d_t); (void)hipFree(d_qk); (void)hipFree(d_attn);
    (void)hipFree(d_gate); (void)hipFree(d_up); (void)hipFree(d_logits);
    return 0;
}

// ─── extern "C" bridge ─────────────────────────────────────────────

extern "C" {

int whisper_gpu_available(void) {
    if (!g_checked) {
        g_checked = 1;
        int n = 0;
        g_gpu_ok = (hipGetDeviceCount(&n) == hipSuccess && n > 0);
        if (g_gpu_ok) fprintf(stderr, "[whisper-gpu] %d device(s) — GPU path enabled (WHISPER_GPU=0 to disable)\n", n);
    }
    return g_gpu_ok;
}

int whisper_gpu_encode(const void* model, const float* mel, int n_frames, float* out_x, int* out_ctx) {
    auto& gm = *model_ctx((const WhisperModel*)model);
    return gpu_encode(gm, *(const WhisperModel*)model, mel, n_frames, out_x, out_ctx);
}

int whisper_gpu_decode_step(const void* model, const int* tokens, int n_tokens, float* out_logits) {
    auto& gm = *model_ctx((const WhisperModel*)model);
    return gpu_decode_step(gm, *(const WhisperModel*)model, tokens, n_tokens, out_logits);
}

void whisper_gpu_free(const void* model) {
    std::lock_guard<std::mutex> lock(g_mtx);
    auto it = g_models.find(model);
    if (it == g_models.end()) return;
    GpuModel* m = it->second;
    for (auto& [k, b] : m->bufs) hipFree(b.d);
    hipFree(m->d_enc_out);
    delete m;
    g_models.erase(it);
}

}  // extern "C"
