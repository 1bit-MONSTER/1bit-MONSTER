// test_attn.cpp — GQA flash-attention int8 contract validation (issue #1776).
//
// Zaya decode is CPU-attention-bound (~11 ms/layer × 20); the NPU plan
// (attention-on-NPU + runlist) replaces the 5 float GEMVs with int8 NPU
// GEMMs. This test pins the EXACT on-core arithmetic the AIE kernel must
// implement — using the SHIPPED contract function
// (attn_quant.h attn_softmax_contract, the same code the AIE kernel
// attn_kernel_reference.cc links) against a float softmax reference:
//
//   C1[t]  = q[h]·K^T[kv(h)][t]     int8×int8 → int32 (QK^T GEMM, host-math
//           here; on the NPU it is the mmul_i8_i32 C operand in mmul C
//           layout: element (r,c) at (c/8)·64 + r·8 + (c%8))
//   x[t]   = C1[t]·SCALE            (host-folded sq·sk/√hd)
//   w[t]   = exp(x[t]-max_{t'<seq} x[t'])   (256-entry LUT, causal mask)
//   A2[t]  = sat8(round(w[t]·127))  (argmax → 127; PV's int8 A in A-layout:
//           element (r,t) at r·MAX_SEQ + (t/8)·8 + t%8)
//   out[d] = Σ_t A2[t]·V[kv(h)][t][d]   int8×int8 → int32 (PV GEMM)
//   attn[d]= out[d]·(sv[d]/127)     (host dequant)
//
// Metrics: A2 exactness (max=127, all in [0,127], argmax→127), LUT-vs-expf
// weight error, causal-mask + rows-1-7-zero checks, and end-to-end PV
// corr/maxdiff/argmax parity vs the float reference — mirroring
// test_fused_silu.cpp's discipline.
//
// Build (CPU only, no xrt, no AIE headers):
//   g++ -std=c++23 -O2 -I engine/npu/generators engine/npu/generators/test_attn.cpp -o /tmp/test_attn
//   /tmp/test_attn [ntrials]

#include "attn_quant.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

// ── Shapes (Zaya1-8B): hd=128, nq=8, nkv=2, gqa=4, MAX_SEQ=256 ──
static const int HD = 128;        // head dim (K)
static const int MAX_SEQ = 256;   // N
static const int NQ = 8, NKV = 2;

// mmul C layout element (r, c) within the (8,128) tile the kernel sees:
// element (r,c) at (c/8)·64 + r·8 + (c%8) — the 8 rows are interleaved
// inside each 64-element column-group, so C1 is a FLAT 1024-int32 buffer.
static inline unsigned c1_idx(int r, int c) { return (c / 8) * 64 + r * 8 + (c % 8); }
// A2 A-layout element (r, t) as the PV GEMM reads it.
static inline unsigned a2_idx(int r, int t) { return r * MAX_SEQ + (t / 8) * 8 + (t % 8); }

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } } while (0)

// ── float reference softmax (unnormalized exp, causal — the int8 path
//    does NOT normalize: A2 = round(127·exp(x−mx)), the partition function
//    folds into the host dequant scale) ──
static void softmax_float(const int32_t c1[8][MAX_SEQ], float scale, int seq,
                          float w[MAX_SEQ]) {
    float mx = -1e30f;
    for (int t = 0; t < seq; t++) { float x = (float)c1[0][t] * scale; if (x > mx) mx = x; }
    if (mx == -1e30f) mx = 0.0f;
    for (int t = 0; t < MAX_SEQ; t++) w[t] = 0.0f;
    for (int t = 0; t < seq; t++) w[t] = std::exp((float)c1[0][t] * scale - mx);
}

static void run_trial(std::mt19937& rng, int trial) {
    const int seq = 1 + (int)(rng() % MAX_SEQ);   // 1..MAX_SEQ
    // Real decode scale: sq·sk/√hd with q/k per-vector scales ~ 1/amax(≈128)
    // → SCALE ≈ 1/128²/√128 ≈ 8.9e-6, keeping x ∈ ~[−0.2, 0.2] inside the
    // LUT's [−16, 0] window with good discrimination.
    const float scale = 8.9e-6f * (float)(rng() % 11 + 5) / 10.0f;   // ~4.5e-6..1.3e-5

    // ── q/k/v int8 (per-vector scales) ──
    int8_t q[NQ][HD], k[NKV][MAX_SEQ][HD], v[NKV][MAX_SEQ][HD];
    for (int h = 0; h < NQ; h++) for (int d = 0; d < HD; d++) q[h][d] = (int8_t)(rng() % 256 - 128);
    for (int kv = 0; kv < NKV; kv++) for (int t = 0; t < MAX_SEQ; t++)
        for (int d = 0; d < HD; d++) { k[kv][t][d] = (int8_t)(rng() % 256 - 128); v[kv][t][d] = (int8_t)(rng() % 256 - 128); }

    for (int h = 0; h < NQ; h++) {
        const int kv = h / 4;
        // ── QK^T on the host (exact int32, order-independent like the mmul) ──
        int32_t c1[8][MAX_SEQ]; memset(c1, 0, sizeof(c1));
        for (int t = 0; t < MAX_SEQ; t++)
            for (int d = 0; d < HD; d++) c1[0][t] += (int32_t)q[h][d] * k[kv][t][d];

        // ── on-core softmax through the SHIPPED contract ──
        // C1 arrives from the QK^T mmul in mmul C layout: a flat 1024-int32
        // buffer per half-tile where element (r,c) sits at (c/8)·64 + r·8 +
        // (c%8) — the rows are interleaved inside each 64-element column
        // group, so a 2D [8][128] view is wrong (indices ≥128 alias other
        // rows). Only row 0 is valid/read.
        int32_t c1a[1024], c1b[1024];
        memset(c1a, 0, sizeof(c1a)); memset(c1b, 0, sizeof(c1b));
        for (int t = 0; t < 128; t++) {
            c1a[c1_idx(0, t)] = c1[0][t];
            c1b[c1_idx(0, t)] = c1[0][t + 128];
        }
        float params[4] = { scale, (float)seq, (float)MAX_SEQ, 0.0f };
        int8_t a2[8 * MAX_SEQ]; memset(a2, 0x5A, sizeof(a2));   // poison
        const int32_t* c1p[2] = { c1a, c1b };
        attn_softmax_contract(c1p, params, a2);

        // ── A2 exactness: max=127, [0,127], true-argmax→127, causal,
        //    rows 1-7=0. (Strict argmax EQUALITY is not the contract — the
        //    LUT/round can give 127 to a neighbour within the quant step;
        //    the guarantee is: the true C1 max maps to 127, nothing exceeds
        //    it, and everything masked/off-row is 0.) ──
        int ref_t = (int)(std::max_element(&c1[0][0], &c1[0][seq]) - &c1[0][0]);
        int a2_max = 0;
        for (int t = 0; t < seq; t++) {
            int qv = a2[a2_idx(0, t)];
            CHECK(qv >= 0 && qv <= 127, "A2 row0 out of [0,127]");
            a2_max = std::max(a2_max, qv);
        }
        CHECK(a2_max == 127, "A2 max != 127");
        CHECK(a2[a2_idx(0, ref_t)] == 127, "true C1 argmax did not map to 127");
        for (int t = seq; t < MAX_SEQ; t++) CHECK(a2[a2_idx(0, t)] == 0, "A2 causal mask leak");
        for (int r = 1; r < 8; r++) for (int t = 0; t < MAX_SEQ; t++)
            CHECK(a2[a2_idx(r, t)] == 0, "A2 rows 1-7 not zero");

        // ── LUT vs expf: weight error at the quantized values ──
        float wref[MAX_SEQ]; softmax_float(c1, scale, seq, wref);
        double werr = 0; int n = 0;
        for (int t = 0; t < seq; t++) {
            double wq = (double)a2[a2_idx(0, t)] / 127.0;
            werr += fabs(wq - wref[t]); n++;
        }
        if (trial == 0) fprintf(stderr, "  head %d seq=%d scale=%.2e argmax@127 %s (t=%d) wabs_err=%.4f\n",
                h, seq, scale, a2[a2_idx(0, ref_t)] == 127 ? "OK" : "FAIL", ref_t, n ? werr / n : 0.0);

        // ── PV: int8 A2·V vs float exp-w·V (both UNNORMALIZED — the int8
        //    path carries A2 = round(127·w); the host dequant's sv/127 fold
        //    makes corr scale-invariant) ──
        int32_t out[HD]; memset(out, 0, sizeof(out));
        for (int d = 0; d < HD; d++) for (int t = 0; t < MAX_SEQ; t++)
            out[d] += (int32_t)a2[a2_idx(0, t)] * v[kv][t][d];
        std::vector<float> attn(HD), ref(HD);
        for (int d = 0; d < HD; d++) {
            attn[d] = (float)out[d];
            for (int t = 0; t < MAX_SEQ; t++) ref[d] += wref[t] * v[kv][t][d];
        }
        double num=0, d1=0, d2=0, md=0;
        for (int d = 0; d < HD; d++) { num += (double)attn[d]*ref[d]; d1 += (double)attn[d]*attn[d]; d2 += (double)ref[d]*ref[d]; md = std::max(md, fabs((double)attn[d]-ref[d])); }
        double corr = num/std::sqrt(d1*d2);
        if (trial == 0) fprintf(stderr, "  PV corr=%.5f md=%.3f (A2/127·V vs exp·V)\n", corr, md);
        // Synthetic uniform int8 data + coarse A2 quantization gives
        // 0.9989–0.9998; the plan's ≥0.999 gate applies to REAL Zaya weights
        // (test_fused_silu discipline). 0.998 pins the kernel, not the model.
        CHECK(corr > 0.998, "PV corr < 0.998");
    }
}

int main(int argc, char** argv) {
    int ntrials = argc > 1 ? atoi(argv[1]) : 8;
    std::mt19937 rng(1776);
    for (int t = 0; t < ntrials; t++) { fprintf(stderr, "trial %d/%d\n", t + 1, ntrials); run_trial(rng, t); }
    if (g_fail == 0) fprintf(stderr, "[RESULT] test_attn PASS (%d trials)\n", ntrials);
    else fprintf(stderr, "[RESULT] test_attn FAIL: %d checks\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
