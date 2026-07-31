// tq2_gemv_real.cpp — WS-04 P1: packed TQ2 GEMV vs dequant+fp32-matmul
// on the REAL 1BP model, using the repo's own OnebpModel loader for offsets.
//
// TQ2 1BP tile (32x256, from onebp_loader.cpp — NOTE: onebp_format.h's
// interleaved doc is stale; actual layout is block-separated):
//   [0..511]:   256 BF16 scales (32 rows x 8 groups)
//   [512..2559]: 2048 bytes packed 2-bit codes (4/byte, LSB-first)
//   codes: 0=-scale, 1=0, 2=+scale, 3=0
//   tiles ordered row-tile-major then col-tile
//
// Build:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mbmi2 \
//       -fopenmp -I include -I engine/npu/src \
//       tq2_gemv_real.cpp -o tq2_gemv_real
// Run: ./tq2_gemv_real [threads] [model.1bp]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>
#include <immintrin.h>
#include "../../engine/npu/src/onebp_loader.cpp"   // OnebpModel (main behind ONEBP_LOADER_MAIN)

static float bf16_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float f; memcpy(&f, &b, 4); return f;
}

// Current engine path: tile-dequant to f32, then fp32 matmul (M=1).
static void matmul(float* y, const float* W, const float* x, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double acc = 0;
        const float* row = W + (size_t)i * K;
        for (int j = 0; j < K; j++) acc += (double)row[j] * x[j];
        y[i] = (float)acc;
    }
}

// FairyFuse-style packed GEMV with per-group scales. TQ2 mapping:
//   pos = (code==2) = hi & ~lo ; neg = (code==0) = ~(hi|lo)
// `get_tile` returns the 2560-byte tile (scales block then codes block).
static void gemv_tq2_packed(const OnebpModel& model, const char* tname,
                            const float* x, float* y, int N, int K,
                            int tr, int tc, int gs) {
    int ntr = (N + tr - 1) / tr, ntc = (K + tc - 1) / tc;
    int groups = tc / gs;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float acc_row = 0;
        int trr = i / tr, rr = i % tr;
        for (int tcc = 0; tcc < ntc; tcc++) {
            const uint8_t* tile = model.get_tile_ptr(tname, trr, tcc);
            if (!tile) { y[i] = 0.f; continue; }
            const uint16_t* sc = (const uint16_t*)tile;              // [32][8] bf16
            const uint8_t* qd = tile + (size_t)tr * groups * 2;      // codes
            int c0 = tcc * tc;
            for (int g = 0; g < groups; g++) {
                float s = bf16_f32(sc[rr * groups + g]);
                const uint8_t* q = qd + (size_t)(rr * tc + g * gs) / 4;
                __m512 acc = _mm512_setzero_ps();
                for (int hh = 0; hh < gs / 16; hh++) {
                    uint32_t v;
                    memcpy(&v, q + 4 * hh, 4);
                    uint32_t lo = _pext_u32(v, 0x55555555u);
                    uint32_t hi = _pext_u32(v, 0xAAAAAAAAu);
                    uint32_t pos = hi & ~lo;
                    uint32_t neg = ~(hi | lo);
                    __m512 acts = _mm512_loadu_ps(x + c0 + g * gs + 16 * hh);
                    acc = _mm512_add_ps(acc, _mm512_maskz_mov_ps((__mmask16)pos, acts));
                    acc = _mm512_sub_ps(acc, _mm512_maskz_mov_ps((__mmask16)neg, acts));
                }
                acc_row += _mm512_reduce_add_ps(acc) * s;
            }
        }
        y[i] = acc_row;
    }
}

int main(int argc, char** argv) {
    int threads = (argc > 1) ? atoi(argv[1]) : 16;
    omp_set_num_threads(threads);
    const char* path = (argc > 2) ? argv[2] : "models/Qwen3-0.6B.1bp";

    OnebpModel model;
    if (!model.open(path)) { fprintf(stderr, "open failed\n"); return 1; }
    auto& h = model.header();
    printf("model: %s  quant=%u H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d  tile=%ux%u gs=%u\n",
           path, h.quant, h.hidden_size, h.num_layers, h.num_attention_heads,
           h.num_kv_heads, h.head_dim, h.intermediate_size, h.vocab_size,
           h.tile_rows, h.tile_cols, h.group_size);
    if (h.quant != ONEBP_TQ2) { fprintf(stderr, "not TQ2\n"); return 1; }
    int tr = h.tile_rows, tc = h.tile_cols, gs = h.group_size;

    struct Op { std::string name; int N, K; };
    std::vector<Op> ops;
    int H = h.hidden_size, NH = h.num_attention_heads, NKV = h.num_kv_heads;
    int HD = h.head_dim, IM = h.intermediate_size;
    std::vector<float> probe;
    auto try_add = [&](const char* fmt, int N, int K) {
        char buf[128];
        for (int l = 0; l < h.num_layers && l < 2; l++) {
            snprintf(buf, sizeof(buf), fmt, l);
            if (model.get_tensor_f32(buf, probe) && (int)probe.size() == N * K)
                ops.push_back({buf, N, K});
        }
    };
    try_add("blk.%d.attn_q.weight", NH * HD, H);
    try_add("blk.%d.attn_k.weight", NKV * HD, H);
    try_add("blk.%d.attn_v.weight", NKV * HD, H);
    try_add("blk.%d.attn_output.weight", H, NH * HD);
    try_add("blk.%d.ffn_gate.weight", IM, H);
    try_add("blk.%d.ffn_up.weight", IM, H);
    try_add("blk.%d.ffn_down.weight", H, IM);
    if (ops.empty()) { fprintf(stderr, "no layer weights found\n"); return 1; }

    printf("\n%-8s %8s %8s | %12s %12s %12s | %8s | %10s\n",
           "op", "N", "K", "dequant+mm", "fp32-mm", "packed", "fp32/pk", "maxdiff");
    double t_ref_total = 0, t_mm_total = 0, t_pk_total = 0;
    int maxK = IM > H ? IM : H;
    std::vector<float> x(maxK);
    for (int j = 0; j < maxK; j++) x[j] = (float)((j * 2654435761u % 2000) - 1000) / 100.0f;

    for (auto& op : ops) {
        int N = op.N, K = op.K;
        std::vector<float> Wf, y_ref(N), y_pk(N);
        if (!model.get_tensor_f32(op.name.c_str(), Wf)) { fprintf(stderr, "load %s failed\n", op.name); continue; }

        auto t0 = std::chrono::steady_clock::now();
        matmul(y_ref.data(), Wf.data(), x.data(), N, K);
        double t_ref = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        // fp32-resident reference (real engine: dequant once at load, matmul per token)
        double t_mm = 0;
        {
            auto tA = std::chrono::steady_clock::now();
            for (int it = 0; it < 10; it++) matmul(y_ref.data(), Wf.data(), x.data(), N, K);
            t_mm = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tA).count() / 10;
        }

        auto t1 = std::chrono::steady_clock::now();
        gemv_tq2_packed(model, op.name.c_str(), x.data(), y_pk.data(), N, K, tr, tc, gs);
        double t_pk = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t1).count();

        float md = 0;
        for (int i = 0; i < N; i++) md = std::max(md, fabsf(y_ref[i] - y_pk[i]));
        printf("%-8s %8d %8d | %10.3fms %10.3fms %10.3fms | %6.1fx | %10.6f\n",
               op.name.c_str(), N, K, t_ref, t_mm, t_pk, t_mm / t_pk, md);
        // debug: worst rows
        float wm = 0; int wi = 0;
        for (int i = 0; i < N; i++) if (fabsf(y_ref[i] - y_pk[i]) > wm) { wm = fabsf(y_ref[i]-y_pk[i]); wi = i; }
        if (wm > 0.01f) printf("    worst row %d: ref=%f pk=%f\n", wi, y_ref[wi], y_pk[wi]);
        t_ref_total += t_ref;
        t_mm_total += t_mm;
        t_pk_total += t_pk;
    }
    printf("\nper-token layer GEMMs (2-layer avg x%d layers): dequant+mm %.1fms  fp32-resident %.1fms  packed %.1fms\n",
           h.num_layers, t_ref_total / 2 * h.num_layers, t_mm_total / 2 * h.num_layers, t_pk_total / 2 * h.num_layers);
    printf("decode ceiling (kernel-bound, excl. attn/norms): fp32-resident %.0f tok/s  packed %.0f tok/s\n",
           1000.0 / (t_mm_total / 2 * h.num_layers), 1000.0 / (t_pk_total / 2 * h.num_layers));
    return 0;
}
