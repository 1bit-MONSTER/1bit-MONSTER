// cascade_real_weight_probe.cpp — real-weight calibration of the Qwen3
// single-launch GU→SiLU→D cascade on silicon.  Packs the REAL blk.0 FFN
// weights into the AB/B_d layout (interleaved gate/up tiles), runs the
// xclbin, and compares C2 against a CPU mirror of the kernel's EXACT integer
// math (int8 dots → int32 C1, Q22-LUT sigmoid silu, sat8, int8 D dot → C2).
// A match pins the host quant convention; a mismatch shows the delta to
// calibrate (the in-kernel scale fold).
//
// Two fill modes (argv[5]):
//   "pad" (default) and "rep": both pack the SAME A-tile (all 8 rows carry the
//   identical h2 slice — the batch-replicated reading; "pad"'s old "A rows
//   1..7 = 0" assumption was WRONG and is removed).  The D phase sums ALL 8
//   k-slices (the worker's `for ks in range(8)` loop), so the mirror's D
//   contraction always uses ks_max = 8.
// The full-ks D model is SILICON-VERIFIED EXACT (2026-08-31, kernel 7.2.0:
//   bad=0/8192, maxrel=0.0000) — the calibration is CLOSED.
// Both are compared against the same mirror of the literal kernel loop
// (D: for cg: for ks: a8s[kstep][c_] = h2b[ks][cg*64 + kstep*8 + c_];
//  mmul a8s[8,8] @ b8[8,N_D_row] → acc[kstep]).
//
// Usage: cascade_real_weight_probe <model.1bp> <xclbin> <insts.txt> [layer] [pad|rep]
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <functional>
#include "../src/onebp_loader.cpp"   // NpuOnebpModel
#include "../generators/silu_quant.h"  // silu_sigmoid_q22 (compiled into the AIE kernel too)

// Qwen3 fused-cascade shapes (K_GU=1024, N_GU=6144, K_D=3072, N_D=1024)
static constexpr int M = 8, H = 1024, IM = 3072;
static constexpr int n_k = 16, n_cg = 6, n_cols = 8, m = 8, k = 64, n = 128;
static constexpr int AB_tile = m * k + k * n;          // 8704
static constexpr long AB_BYTES = (long)n_cols * n_cg * n_k * AB_tile;
static constexpr int K_D = 3072, N_D = 1024;

static inline int q127(float v, float is) {
    int q = (int)roundf(v * is);
    if (q > 127) q = 127; else if (q < -127) q = -127;
    return q;
}
// Exact mirror of silu_quant_i8_fused_q22 (mm_kernel_reference.cc):
//   gc = clamp(c1g, -4, 4);  idx = round((gc+4)*255/8)  (integer)
//   sig = silu_sigmoid_q22[idx];  silu = (c1g * sig) >> 22   (RAW gate!)
//   h = sat8(silu * c1u)                                    (RAW up!)
static inline int silu_q22(int g, int u) {
    int gc = g < -4 ? -4 : (g > 4 ? 4 : g);
    int idx = ((gc + 4) * 255 + 4) / 8;
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    int sig = silu_sigmoid_q22[idx];
    long long silu = ((long long)g * sig) >> 22;
    long long h = silu * u;
    if (h > 127) h = 127; else if (h < -127) h = -127;
    return (int)h;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.1bp xclbin insts.txt [layer] [pad|rep]\n", argv[0]); return 2; }
    const int layer = argc > 4 ? atoi(argv[4]) : 0;
    const char* mode = argc > 5 ? argv[5] : "pad";
    const bool rep = strcmp(mode, "rep") == 0 || strcmp(mode, "bd1") == 0 || strcmp(mode, "bd1p") == 0 || strcmp(mode, "h2r") == 0;
    const bool bd1 = strcmp(mode, "bd1") == 0 || strcmp(mode, "bd1p") == 0;
    const bool lay = strcmp(mode, "lay") == 0;   // one-hot layout probe
    const bool h2r = strcmp(mode, "h2r") == 0;   // per-pair h2 read: argv[6] = h2s index
    const int h2r_idx = h2r ? atoi(argv[6]) : 0;
    const bool guread = strcmp(mode, "guread") == 0;   // one-hot-A B-tile read probe
    const int gu_k0 = guread ? atoi(argv[6]) : 0;      // the one-hot A element
    const int gu_hidx = guread ? atoi(argv[7]) : 0;    // the h2r idx for the bd
    const bool bread = strcmp(mode, "bread") == 0;     // one-hot-B A-value read probe
    const int br_p = bread ? atoi(argv[6]) : 0;        // the one-hot B col pair
    fprintf(stderr, "mode: %s (%s fill, B_d=%s)\n", mode, rep ? "rep" : "pad", bd1 ? "ONES" : "real");

    // Load the layer's FFN weights (f32, [out,in])
    NpuOnebpModel mdl;
    if (!mdl.open(argv[1])) { fprintf(stderr, "open %s failed\n", argv[1]); return 2; }
    char buf[128];
    snprintf(buf, sizeof(buf), "blk.%d.ffn_gate.weight", layer);
    std::vector<float> w1; if (!mdl.get_tensor_f32(buf, w1)) return 2;
    snprintf(buf, sizeof(buf), "blk.%d.ffn_up.weight", layer);
    std::vector<float> w2; if (!mdl.get_tensor_f32(buf, w2)) return 2;
    snprintf(buf, sizeof(buf), "blk.%d.ffn_down.weight", layer);
    std::vector<float> w3; if (!mdl.get_tensor_f32(buf, w3)) return 2;
    fprintf(stderr, "layer %d: w1=%zu w2=%zu w3=%zu\n", layer, w1.size(), w2.size(), w3.size());

    // Deterministic h2 (row 0) + per-token ascale (standard convention)
    std::vector<float> h2v(H);
    for (int i = 0; i < H; i++) h2v[i] = 0.02f * (float)((i * 2654435761u >> 16) % 2001 - 1000) / 1000.0f;
    float ascale = 0;
    for (int i = 0; i < H; i++) { float a = fabsf(h2v[i]); if (a > ascale) ascale = a; }
    ascale = (ascale < 1e-12f) ? 1.0f : ascale / 127.0f;

    // Weight scales (standard per-tensor amax/127)
    float gu_amax = 0, d_amax = 0;
    for (size_t i = 0; i < w1.size(); i++) { float a = fabsf(w1[i]); if (a > gu_amax) gu_amax = a; }
    for (size_t i = 0; i < w2.size(); i++) { float a = fabsf(w2[i]); if (a > gu_amax) gu_amax = a; }
    for (size_t i = 0; i < w3.size(); i++) { float a = fabsf(w3[i]); if (a > d_amax) d_amax = a; }
    float gu_scale = (gu_amax < 1e-12f) ? 1.0f : gu_amax / 127.0f;
    float d_scale  = (d_amax  < 1e-12f) ? 1.0f : d_amax  / 127.0f;
    float gu_is = 127.0f / (gu_amax < 1e-12f ? 1.0f : gu_amax);
    float d_is  = 127.0f / (d_amax  < 1e-12f ? 1.0f : d_amax);
    float a_is = 1.0f / ascale;

    // Build the AB BO (B_gu interleaved: col 2j = gate j0+j, 2j+1 = up j0+j)
    const int j0_hot = (argc > 6 ? atoi(argv[6]) : 0);   // layout-probe pair within (col=0, cg=0)
    std::vector<int8_t> ab(AB_BYTES);
    for (int col = 0; col < n_cols; col++)
        for (int ki = 0; ki < n_k; ki++)
            for (int cg = 0; cg < n_cg; cg++) {
                long base = ((long)col * n_cg * n_k + cg * n_k + ki) * AB_tile;
                int8_t* A = ab.data() + base;
                int8_t* B = A + m * k;
                int j0 = (cg * n_cols + col) * 64;
                memset(A, 0, (size_t)m * k);
                if (lay) {
                    // one-hot A: kk=0 = 127, everything else 0 (all rows)
                    for (int r = 0; r < m; r++) A[r * k] = (int8_t)127;
                } else if (guread) {
                    // one-hot A at (row 0, col k0), rows 1..7 zero: the C1[0][n]
                    // = 127·B(k0, n) — the kernel's exact B-tile read position
                    // for the K=k0 becomes observable via the h2r read.
                    if (col == 0 && ki == 0 && cg == 0 && gu_k0 < 512) A[gu_k0] = (int8_t)127;
                } else {
                    // REINDEXED A-tile (silicon-pinned): the mmul reads
                    // A(row, K = i*8+k') = A_tile[i*64 + row*8 + k'] — the
                    // A-tile's ROW i = the K-slice; row i holds the h2's
                    // 8-element slice i replicated 8× (cols row*8+k').
                    for (int i = 0; i < 8; i++)
                        for (int c = 0; c < 64; c++)
                            A[i * 64 + c] = (int8_t)q127(h2v[ki * 64 + i * 8 + (c % 8)], a_is);
                }
                memset(B, 0, (size_t)k * n);
                if (lay && col == 0 && ki == 0 && cg == 0) {
                    // B_gu one-hot: (k=0, cols 2*j0_hot, 2*j0_hot+1) = 1 →
                    // C1[2p0]=127, C1[2p0+1]=127 → h2[p0]=127, others 0.
                    B[0 * 128 + 2 * j0_hot] = 1;
                    B[0 * 128 + 2 * j0_hot + 1] = 1;
                } else if (bread && col == 0 && ki == 0 && cg == 0 && br_p < 64) {
                    // one-hot B at (row 0, cols 2*br_p, 2*br_p+1): the C1[0][n]
                    // = the A-tile value at the K whose read hits (0, 2*br_p) —
                    // the A's VALUE becomes observable via the h2r read.
                    B[0 * 128 + 2 * br_p] = 1;
                    B[0 * 128 + 2 * br_p + 1] = 1;
                } else if (!lay && !bread) {
                    // DERIV-INVERSE B_gu packing (the silicon read formula,
                    // confirmed by the guread one-hot-A probes 64/64):
                    //   B(K, n) = B_tile[n/16 + 8·(K/8),
                    //                64·((n/8)%2) + (K%8)·8 + n%8]
                    // So for output pair j (gate n=2j, up n=2j+1) and K within
                    // the ki slice, the weight w1[j0+j][ki*64+K] must sit at
                    //   row = j/8 + 8·(K/8)
                    //   col = 64·((j/4)%2) + (K%8)·8 + 2·(j%4)   [gate]
                    //   col = 64·((j/4)%2) + (K%8)·8 + 2·(j%4)+1 [up]
                    // (j, K) → (row, col) is a bijection onto the even/odd
                    // cols, so every B-tile position is filled exactly once.
                    // The old DIRECT pack (B[r][2j] = w1[j0+j][ki*64+r]) only
                    // coincides at j<8 ∧ K%8==0 and was the GU open item.
                    for (int j = 0; j < 64; j++)
                        for (int K = 0; K < k; K++) {
                            int row = j / 8 + 8 * (K / 8);
                            int cgc = 64 * ((j / 4) % 2) + (K % 8) * 8 + 2 * (j % 4);
                            B[row * 128 + cgc]     = (int8_t)q127(w1[(size_t)(j0 + j) * H + ki * 64 + K], gu_is);
                            B[row * 128 + cgc + 1] = (int8_t)q127(w2[(size_t)(j0 + j) * H + ki * 64 + K], gu_is);
                        }
                }
            }
    std::vector<int8_t> bd((size_t)K_D * N_D);
    if (h2r || guread || bread) {
        // Per-pair h2 read: make C2[nn] = exactly the h2 pair at the h2s index
        // (h2r_idx for h2r, gu_hidx for guread). (col, cg, j) = (idx/384,
        // (idx%384)/64, idx%64); the D reads h2b[ks][cg*64 + ks*8 + c_] at bd
        // ROW (cg*8+col)*64 + ks*8 + n/64 and COL rh*512 + 64*((n/8)%8) +
        // c_*8 + n%8 — so setting those elements to 1 (for every n/64 = q)
        // isolates the single pair's h2.
        int hridx = h2r ? h2r_idx : (guread ? gu_hidx : 0);
        int hc = hridx / 384, hg = (hridx % 384) / 64, hj = hridx % 64;
        int hks = hj / 8, hc_ = hj % 8;
        int hkk0 = (hg * 8 + hc) * 64 + hks * 8;
        for (int q = 0; q < 8; q++)
            for (int rh = 0; rh < 2; rh++)
                for (int n = 0; n < 512; n++)
                    bd[(size_t)(hkk0 + q) * N_D + rh * 512 + 64 * ((n / 8) % 8) + hc_ * 8 + n % 8] = 1;
        fprintf(stderr, "%s: idx=%d (col=%d cg=%d j=%d ks=%d c_=%d kk0=%d)\n",
                h2r ? "h2r" : "guread", hridx, hc, hg, hj, hks, hc_, hkk0);
    } else {
        for (int kk = 0; kk < K_D; kk++)
            for (int nn = 0; nn < N_D; nn++)
                bd[(size_t)kk * N_D + nn] = bd1 ? (int8_t)1
                                                : lay ? (int8_t)((kk + nn) & 0x7F)
                                                      : (int8_t)q127(w3[(size_t)nn * IM + kk], d_is);  // w3 [IM,H]=[K,N]
    }

    // CPU mirror: exact integer math. GU: h2s[col][cg*64+j] = silu(C1[2j], C1[2j+1])
    const bool gusolve = strcmp(mode, "gusolve") == 0;
    std::vector<int> h2s(n_cg * 64 * n_cols);   // the silu'd pairs (per column 384)
    if (gusolve) {
        // Enumerate the GU gate-formula variants against the h2r NPU truth for
        // the col-0 64 pairs. The gate for the pair p of (col, ki, cg):
        //   C1[2p] = Σ_{i,kp} A_el(i,kp) · w1q[pair(p,i,kp)][IN(p,i,kp)]
        static const int npu_col0[64] = {
           -127, -127, -127, 127, -127, -127, 127, 127,
           -127, -127, 127, 127, 127, 127, -127, 127,
           -127, 127, -127, -127, -127, -127, 127, 127,
           127, -127, 127, -127, -127, -127, -127, -127,
           127, 127, 127, -127, -127, -127, -127, -127,
           127, -127, 127, 127, 127, -127, -127, 127,
           -127, 127, -127, 127, 127, -127, -127, 127,
           -127, 127, -127, -127, 127, 127, 127, -127 };


        // quantized weights + A for the col 0
        std::vector<int8_t> aq(H), w1q_all((size_t)3072 * H), w2q_all((size_t)3072 * H);
        for (int i = 0; i < H; i++) aq[i] = (int8_t)q127(h2v[i], a_is);
        for (size_t i = 0; i < w1.size(); i++) w1q_all[i] = (int8_t)q127(w1[i], gu_is);
        for (size_t i = 0; i < w2.size(); i++) w2q_all[i] = (int8_t)q127(w2[i], gu_is);
        struct Var { const char* name; int (*f)(int, const int8_t*, const int8_t*, const int8_t*); };
        // variant helpers: gate(p) from aq (the A), w1q (the weights)
        auto vA_direct = [&](int p) {
            long g = 0;
            for (int ki = 0; ki < n_k; ki++) {
                const int8_t* w = w1q_all.data() + (size_t)p * H;
                for (int a = 0; a < 64; a++) g += (long)aq[ki * 64 + a] * w[ki * 64 + a];
            }
            return (int)g;
        };
        auto vB_reidx = [&](int p) {   // current V-a: A[8i+kp], pair j0+32*((2p/8)%2)+4kp+p%4, IN 8i+p/8
            long g = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4;
                        if (pair >= 3072) continue;
                        g += (long)aq[ki * 64 + 8 * i + kp] * w1q_all[(size_t)pair * H + ki * 64 + 8 * i + p / 8];
                    }
            return (int)g;
        };
        auto vC = [&](int p) {   // A[kp] (first 8), pair + 4kp + p%4, IN 8i + p/8
            long g = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4;
                        if (pair >= 3072) continue;
                        g += (long)aq[ki * 64 + kp] * w1q_all[(size_t)pair * H + ki * 64 + 8 * i + p / 8];
                    }
            return (int)g;
        };
        auto vD = [&](int p) {   // A[8i+kp], pair + 4kp + p%4, IN i + p/8
            long g = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4;
                        if (pair >= 3072) continue;
                        g += (long)aq[ki * 64 + 8 * i + kp] * w1q_all[(size_t)pair * H + ki * 64 + i + p / 8];
                    }
            return (int)g;
        };
        auto vE = [&](int p) {   // A[8i+kp], pair + 4kp + p%4, IN 8i + 8*(p/8)... keep p%8? no — IN 8i
            long g = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4;
                        if (pair >= 3072) continue;
                        g += (long)aq[ki * 64 + 8 * i + kp] * w1q_all[(size_t)pair * H + ki * 64 + 8 * i];
                    }
            return (int)g;
        };
        auto vF = [&](int p) {   // A[8i+kp], pair + 4kp + (p%4), IN 8i + (p%8)/... use p/8 but A first-8-of-ki
            long g = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4;
                        if (pair >= 3072) continue;
                        g += (long)aq[ki * 64 + kp] * w1q_all[(size_t)pair * H + ki * 64 + 8 * i];
                    }
            return (int)g;
        };
        const char* names[12] = { "A direct dot", "B reidx A[8i+kp]", "C A[kp]", "D IN=i", "E IN=8i", "F A[kp] IN=8i",
                                  "G A[kp] pair-no32", "H A[kp] IN=i", "I A[kp] pair8kp", "J A[8i+kp] pair-no32", "K A[8i+kp] IN=i", "L A[8i+kp] pair8kp" };
        // per-variant up: mirror the gate's A-style and pair/IN mapping
        auto up_of = [&](int p, int style) {   // 0=direct, 1=A[8i+kp] reidx, 2=A[kp] reidx, 3=no32, 4=pair8kp
            long u = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = p, a = 0, inr = 0;
                        if (style == 0) { pair = p; a = 8 * i + kp; inr = 8 * i + kp; }
                        if (style == 1) { pair = p + 32 * (((2 * p + 1) / 8) % 2) + 4 * kp + (2 * p + 1) % 8 / 2; a = 8 * i + kp; inr = 8 * i + p / 8; }
                        if (style == 2) { pair = p + 32 * (((2 * p + 1) / 8) % 2) + 4 * kp + (2 * p + 1) % 8 / 2; a = kp; inr = 8 * i + p / 8; }
                        if (style == 3) { pair = p + 4 * kp + (2 * p + 1) % 8 / 2; a = kp; inr = 8 * i + p / 8; }
                        if (style == 4) { pair = p + 8 * kp + (2 * p + 1) % 8 / 2; a = kp; inr = 8 * i + p / 8; }
                        if (pair >= 3072) continue;
                        u += (long)aq[ki * 64 + a] * w2q_all[(size_t)pair * H + ki * 64 + inr];
                    }
            return (int)u;
        };
        auto gate_of = [&](int p, int style) {
            long g = 0;
            for (int ki = 0; ki < n_k; ki++)
                for (int i = 0; i < 8; i++)
                    for (int kp = 0; kp < 8; kp++) {
                        int pair = p, a = 0, inr = 0;
                        if (style == 0) { pair = p; a = 8 * i + kp; inr = 8 * i + kp; }
                        if (style == 1) { pair = p + 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4; a = 8 * i + kp; inr = 8 * i + p / 8; }
                        if (style == 2) { pair = p + 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4; a = kp; inr = 8 * i + p / 8; }
                        if (style == 3) { pair = p + 4 * kp + p % 4; a = kp; inr = 8 * i + p / 8; }
                        if (style == 4) { pair = p + 8 * kp + p % 4; a = kp; inr = 8 * i + p / 8; }
                        if (pair >= 3072) continue;
                        g += (long)aq[ki * 64 + a] * w1q_all[(size_t)pair * H + ki * 64 + inr];
                    }
            return (int)g;
        };
        // style per variant: 0=direct, 1=B(reidx,A8i), 2=C(Akp), 3=D(IN=i)... map:
        // A→0, B→1, C→2, D→1-with-inr-i, E→1-with-inr-8i, F→2-with-inr-8i,
        // G→3, H→2-with-inr-i, I→4, J→3-with-A8i, K→1-with-inr-i, L→4-with-A8i
        int styles[12][3] = {
            {0,0,0}, {1,1,1}, {2,2,2}, {1,1,1}, {1,1,1}, {2,2,2},
            {3,3,3}, {2,2,2}, {4,4,4}, {3,3,3}, {1,1,1}, {4,4,4} };
        // custom overrides: D: inr=i (style 1 + inr override); E: inr=8i; F: inr=8i; H: inr=i; J: A8i+no32; K: A8i+inr-i; L: A8i+8kp
        int vi = 0;
        for (int fi = 0; fi < 12; fi++) {
            int good = 0;
            for (int p = 0; p < 64; p++) {
                int g = gate_of(p, styles[fi][0]);                int u = up_of(p, styles[fi][1]);
                int h = silu_q22(g, u);
                if (h == npu_col0[p]) good++;
            }
            fprintf(stderr, "variant %d (%-22s): h2 match %d/64\n", vi, names[vi], good);
            if (fi == 2 || fi == 0) {   // side-by-side for C (A[kp] reidx) and A (direct)
                fprintf(stderr, "  pairs 0..31 %s: ", names[fi]);
                for (int p = 0; p < 32; p++) {
                    int h = silu_q22(gate_of(p, styles[fi][0]), up_of(p, styles[fi][1]));
                    fprintf(stderr, "%s%d", h == npu_col0[p] ? " " : "!", h);
                }
                fprintf(stderr, "\n  NPU truth    : ");
                for (int p = 0; p < 32; p++) fprintf(stderr, "  %d", npu_col0[p]);
                fprintf(stderr, "\n");
            }
            vi++;
        }
        // ── guread solver: the one-hot-A B-tile read observations. The acc
        // rows t (pair t*8, hidx=0) for the one-hot A at (0, k0) =
        // h2s[0][t*8] = sat8( silu(127*w1q[pos]) * 127*w2q[pos'] ) — the sign
        // = sign(w1q[pos])*sign(w2q[pos']) (0 if either weight is 0). Solve
        // the read mapping (k0, p) → (w1 pos, w2 pos').
        static const int gur_obs[4][8] = {   // k0=0..3, the pairs t*8 (t=0..7)
            { 127, -127, 127, -127, -127, -127, 127, -127 },
            { 127, -127, -127, 127, 127, 127, 127, 127 },
            { 127, 0, -127, 0, 127, 127, 127, 127 },
            { 127, 127, -127, 127, -127, 127, -127, 127 } };
        auto gur_pred = [&](int k0, int p, int style, int& w1v, int& w2v) {
            // style 0 = direct, 1 = reidx (n/16 + 8*(k0/8), ...), 2 = reidx-no-8k0
            int pos1 = 0, pos2 = 0;
            if (style == 0) { pos1 = p * H + k0; pos2 = p * H + k0; }
            if (style == 1) { pos1 = (p + 32 * ((2 * p / 8) % 2) + 4 * (k0 % 8) + p % 4) * H + p / 8 + 8 * (k0 / 8);
                              pos2 = (p + 32 * (((2 * p + 1) / 8) % 2) + 4 * (k0 % 8) + ((2 * p + 1) % 8) / 2) * H + (2 * p + 1) / 16 + 8 * (k0 / 8); }
            if (style == 2) { pos1 = (p + 32 * ((2 * p / 8) % 2) + 4 * (k0 % 8) + p % 4) * H + p / 8;
                              pos2 = (p + 32 * (((2 * p + 1) / 8) % 2) + 4 * (k0 % 8) + ((2 * p + 1) % 8) / 2) * H + (2 * p + 1) / 16; }
            w1v = pos1 < 3072 * H ? w1q_all[pos1] : 0;
            w2v = pos2 < 3072 * H ? w2q_all[pos2] : 0;
        };
        const char* gnames[3] = { "direct", "reidx+8k0/8", "reidx" };
        for (int st = 0; st < 3; st++) {
            int good = 0, tot = 0;
            for (int k0 = 0; k0 < 4; k0++)
                for (int t = 0; t < 8; t++) {
                    int p = t * 8, w1v, w2v;
                    gur_pred(k0, p, st, w1v, w2v);
                    int pred = (w1v == 0 || w2v == 0) ? 0 : ((w1v < 0) != (w2v < 0) ? -127 : 127);
                    if (pred == gur_obs[k0][t]) good++;
                    tot++;
                }
            fprintf(stderr, "guread solver (%s): %d/32\n", gnames[st], good);
        }
        // ── systematic B-tile read search: the read (K, n) → B_tile (row, col)
        // with the direct fill B_tile[r][c] = w1q[(j0+c/2)][ki*64+r]. Enumerate
        // row/col formulas (the up uses n = 2p+1) and count the observation fit.
        struct RC { const char* nm; int (*row)(int K, int n); int (*col)(int K, int n); };
        auto r_deriv = [](int K, int n) { return n / 16 + 8 * (K / 8); };
        auto r_K     = [](int K, int n) { (void)n; return K; };
        auto r_n2K16 = [](int K, int n) { return n / 2 + K / 16; };
        auto r_n16   = [](int K, int n) { return n / 16; };
        auto r_K8    = [](int K, int n) { return 8 * (K / 8) + n / 16; };
        auto c_deriv = [](int K, int n) { return 64 * ((n / 8) % 2) + (K % 8) * 8 + n % 8; };
        auto c_n     = [](int K, int n) { (void)K; return n; };
        auto c_K     = [](int K, int n) { (void)n; return K; };
        auto c_nK8   = [](int K, int n) { return 64 * ((n / 8) % 2) + n % 8 + (K % 8) * 8; };
        auto c_n8K   = [](int K, int n) { return (n % 8) * 8 + K % 8; };
        auto c_n64K  = [](int K, int n) { return 64 * ((n / 8) % 2) + n % 8; };
        RC rcs[10] = {
            { "deriv/deriv", r_deriv, c_deriv }, { "deriv/n", r_deriv, c_n }, { "deriv/K", r_deriv, c_K },
            { "K/deriv", r_K, c_deriv }, { "K/n", r_K, c_n }, { "K/K", r_K, c_K },
            { "n2K16/deriv", r_n2K16, c_deriv }, { "n16/deriv", r_n16, c_deriv }, { "K8/deriv", r_K8, c_deriv },
            { "deriv/nK8", r_deriv, c_nK8 } };
        int best = -1; const char* bestnm = "";
        // row-1 one-hot observations (the A-tile[64+k0'] = 127 → C1[0] =
        // 127·B(8+k0', n) IF the A-tile rows are the K-slices)
        static const int gur_obs_r1[4][8] = {
            { -127, 127, -127, -127, 127, 127, -127, 127 },
            { 127, 127, -127, -127, 127, 127, 127, -127 },
            { -127, 127, -127, 127, 127, 0, 127, 0 },
            { 127, 127, 127, 127, -127, -127, 127, -127 } };
        for (int ri = 0; ri < 10; ri++) {
            int good = 0, tot = 0;
            for (int k0 = 0; k0 < 4; k0++)   // row 0
                for (int t = 0; t < 8; t++) {
                    int p = t * 8;
                    int row1 = rcs[ri].row(k0, 2 * p), col1 = rcs[ri].col(k0, 2 * p);
                    int row2 = rcs[ri].row(k0, 2 * p + 1), col2 = rcs[ri].col(k0, 2 * p + 1);
                    if (row1 < 0 || row1 >= 64 || col1 < 0 || col1 >= 128) continue;
                    if (row2 < 0 || row2 >= 64 || col2 < 0 || col2 >= 128) continue;
                    if (col1 % 2 != 0 || col2 % 2 != 1) continue;
                    int w1v = col1 / 2 < 3072 ? w1q_all[(size_t)(col1 / 2) * H + row1] : 0;
                    int w2v = col2 / 2 < 3072 ? w2q_all[(size_t)(col2 / 2) * H + row2] : 0;
                    int pred = (w1v == 0 || w2v == 0) ? 0 : ((w1v < 0) != (w2v < 0) ? -127 : 127);
                    if (pred == gur_obs[k0][t]) good++;
                    tot++;
                }
            for (int k1 = 0; k1 < 4; k1++)   // row 1 (the A-tile[64+k1] → K=8+k1)
                for (int t = 0; t < 8; t++) {
                    int p = t * 8, K = 8 + k1;
                    int row1 = rcs[ri].row(K, 2 * p), col1 = rcs[ri].col(K, 2 * p);
                    int row2 = rcs[ri].row(K, 2 * p + 1), col2 = rcs[ri].col(K, 2 * p + 1);
                    if (row1 < 0 || row1 >= 64 || col1 < 0 || col1 >= 128) continue;
                    if (row2 < 0 || row2 >= 64 || col2 < 0 || col2 >= 128) continue;
                    if (col1 % 2 != 0 || col2 % 2 != 1) continue;
                    int w1v = col1 / 2 < 3072 ? w1q_all[(size_t)(col1 / 2) * H + row1] : 0;
                    int w2v = col2 / 2 < 3072 ? w2q_all[(size_t)(col2 / 2) * H + row2] : 0;
                    int pred = (w1v == 0 || w2v == 0) ? 0 : ((w1v < 0) != (w2v < 0) ? -127 : 127);
                    if (pred == gur_obs_r1[k1][t]) good++;
                    tot++;
                }
            if (good > best) { best = good; bestnm = rcs[ri].nm; }
            fprintf(stderr, "guread search (%s): %d/%d\n", rcs[ri].nm, good, tot);
        }
        fprintf(stderr, "guread BEST: %s %d/64\n", bestnm, best);
        // ── A-layout search: with the CONFIRMED B-read, score the A-value
        // mapping variants against the full col-0 64 truth.
        // gate(p) = Σ_{ki,i,kp} aq[fA(i,kp,ki)] · w1q[32·((2p/8)%2) + 4kp +
        // p%4][ki·64 + 8i + p/8]; up analog with w2.
        auto scoreA = [&](int fA, const char* nm) {
            int good = 0;
            for (int p = 0; p < 64; p++) {
                long g = 0, u = 0;
                for (int ki = 0; ki < n_k; ki++)
                    for (int i = 0; i < 8; i++)
                        for (int kp = 0; kp < 8; kp++) {
                            int a = 0;
                            if (fA == 0) a = ki * 64 + i * 8 + kp;         // full K (current)
                            if (fA == 1) a = ki * 64 + kp * 8 + i;         // transposed slice
                            if (fA == 2) a = ki * 64 + kp;                 // first-8
                            if (fA == 3) a = ki * 64 + i * 8;              // slice start
                            if (fA == 4) a = ki * 64 + i * 8 + (kp % 8) == 0 ? 0 : a; (void)a; // placeholder
                            int pair = 32 * ((2 * p / 8) % 2) + 4 * kp + p % 4;
                            if (pair >= 3072) continue;
                            if (fA == 4) { a = ki * 64 + kp * 8 + (i % 8); }
                            g += (long)aq[a] * w1q_all[(size_t)pair * H + ki * 64 + 8 * i + p / 8];
                            int pairu = 32 * (((2 * p + 1) / 8) % 2) + 4 * kp + ((2 * p + 1) % 8) / 2;
                            if (pairu >= 3072) continue;
                            u += (long)aq[a] * w2q_all[(size_t)pairu * H + ki * 64 + 8 * i + p / 8];
                        }
                if (silu_q22((int)g, (int)u) == npu_col0[p]) good++;
            }
            fprintf(stderr, "A-layout %d (%s): %d/64\n", fA, nm, good);
        };
        scoreA(0, "full K");
        scoreA(1, "transposed slice");
        scoreA(2, "first-8");
        scoreA(3, "slice start");
        scoreA(4, "kp-major");
        return 2;   // gusolve is a diagnostic; don't run the NPU
    }
    for (int col = 0; col < n_cols; col++)
        for (int cg = 0; cg < n_cg; cg++) {
            int j0 = (cg * n_cols + col) * 64;   // MUST shadow the C-lib ::j0 (Bessel)
            for (int j = 0; j < 64; j++) {
                long g = 0, u = 0;
                for (int ki = 0; ki < n_k; ki++) {
                    // A must be the ki-th element's A-tile (h2 slice ki*64);
                    // the previous version pinned A at ki=0 and reused it for
                    // every ki — a mirror bug that decoupled the mirror from
                    // the kernel's per-ki A delivery.
                    const int8_t* A  = ab.data() + ((long)col * n_cg * n_k + cg * n_k + ki) * AB_tile;
                    const int8_t* B  = A + m * k;
                    // GU mirror — SILICON-CONFIRMED read (guread one-hot-A
                    // probes: 32/32): B(K, n) = B_tile[n/16 + 8·(K/8),
                    // 64·((n/8)%2) + (K%8)·8 + n%8], and the A-tile is
                    // [K-slices][M·8+k'] so A(0, i·8+k') = A_tile[i·64 + k']
                    // = the row i's element k' = h2[k'] (the rep). So:
                    //   gate(p) = Σ_{i<8} Σ_{k'<8} h2[k']·w1q[j0 + 32·((2p/8)%2)
                    //     + 4k' + p%4][ki·64 + 8i + p/8]; up analog with w2.
                    for (int i = 0; i < 8; i++)
                        for (int kp = 0; kp < 8; kp++) {
                            // the packed B-tile at the kernel's confirmed read
                            int rg2 = 8 * i + j / 8;
                            g += (long)A[i * 64 + kp] * B[rg2 * 128 + 64 * ((2 * j / 8) % 2) + kp * 8 + (2 * j) % 8];
                            u += (long)A[i * 64 + kp] * B[rg2 * 128 + 64 * (((2 * j + 1) / 8) % 2) + kp * 8 + ((2 * j + 1) % 8)];
                        }
                }
                h2s[col * (n_cg * 64) + cg * 64 + j] = silu_q22((int)g, (int)u);
            }
        }
    // h2s stats (the silu output; sat8 → mostly ±127 at this scale convention)
    { long sat = 0, tot = (long)h2s.size(); int mn = 999, mx = -999; long full = 0;
      for (int v : h2s) { if (v >= 127 || v <= -127) sat++; if (v < mn) mn = v; if (v > mx) mx = v; full += v; }
      fprintf(stderr, "h2s: %ld pairs, sat8=%ld (%.1f%%), range [%d,%d], FULLSUM=%ld, first8:",
              tot, sat, 100.0 * sat / tot, mn, mx, full);
      for (int i = 0; i < 8; i++) fprintf(stderr, " %d", h2s[i]); fprintf(stderr, "\n");
      // the mirror's col-0 64 pairs vs the NPU h2r truth (extracted 2026-08-30)
      static const int npu_col0[64] = {
           -127, -127, -127, 127, -127, -127, 127, 127,
           -127, -127, 127, 127, 127, 127, -127, 127,
           -127, 127, -127, -127, -127, -127, 127, 127,
           127, -127, 127, -127, -127, -127, -127, -127,
           127, 127, 127, -127, -127, -127, -127, -127,
           127, -127, 127, 127, 127, -127, -127, 127,
           -127, 127, -127, 127, 127, -127, -127, 127,
           -127, 127, -127, -127, 127, 127, 127, -127 };

      fprintf(stderr, "col0 pairs  mirror vs NPU:");
      for (int j = 0; j < 64; j++) {
          int m = h2s[j], n = npu_col0[j];
          fprintf(stderr, " %d%s", m, m == n ? "" : (m == 0 ? "=0" : "!"));
          if (j % 16 == 15) fprintf(stderr, "\n              ");
      }
      fprintf(stderr, "\n");
      // the mirror's gates/ups for the col-0 pairs 0..15 (debug the silu)
      {
          const int8_t* A0 = ab.data() + ((long)0 * n_cg * n_k + 0 * n_cg + 0) * AB_tile;
          for (int jj = 0; jj < 8; jj++) { int j = jj * 8;
              long g = 0, u = 0;
              for (int ki = 0; ki < n_k; ki++) {
                  const int8_t* B = ab.data() + ((long)0 * n_cg * n_k + 0 * n_k + ki) * AB_tile + m * k;
                  for (int i = 0; i < 8; i++)
                      for (int kp = 0; kp < 8; kp++) {
                          int rg2 = 8 * i + j / 8;
                          g += (long)A0[i * 64 + kp] * B[rg2 * 128 + 64 * ((2 * j / 8) % 2) + kp * 8 + (2 * j) % 8];
                          u += (long)A0[i * 64 + kp] * B[rg2 * 128 + 64 * (((2 * j + 1) / 8) % 2) + kp * 8 + ((2 * j + 1) % 8)];
                      }
              }
              int gc = g < -4 ? -4 : (g > 4 ? 4 : g);
              int idx = ((gc + 4) * 255 + 4) / 8; if (idx < 0) idx = 0; if (idx > 255) idx = 255;
              fprintf(stderr, "pair %d: g=%ld u=%ld sig=%d silu=%lld h=%lld mir=%d npu=%d%s\n",
                      j, g, u, silu_sigmoid_q22[idx], ((long long)g * silu_sigmoid_q22[idx]) >> 22,
                      (((long long)g * silu_sigmoid_q22[idx]) >> 22) * u, h2s[j], npu_col0[j],
                      h2s[j] == npu_col0[j] ? "" : "  <<<");
          }
      }
      // non-saturated pairs (the candidates for the one-pair ±127 delta)
      int shown = 0;
      for (int i = 0; i < tot && shown < 20; i++)
          if (h2s[i] > -127 && h2s[i] < 127) {
              fprintf(stderr, "  non-sat h2s[%d] = %d\n", i, h2s[i]);
              shown++;
          }
    }
    // D: SILICON-PINNED contraction (layout probes j0=0/1/2 + the h2r per-pair
    // reads — CORRECTED: the slice is the ACC ROW (kstep), per the source):
    //  a8s[kstep][c_] = h2b[ks][cg*64 + kstep*8 + c_]   ← the source's literal
    //  → the acc row t reads the h2 at (cg*64 + t*8 + c_) — the h2r per-pair
    //    probes show the rows DIFFER (row t = the pair t*8+c_*), so the D is
    //    the kstep-slice, NOT the ks-slice.
    //  bd read: the mmul's B tile (k, n) = b8[(n/64)][64*((n/8)%8) + k*8 + n%8]
    //    → bd ROW = ki*64 + ks*8 + n/64, bd COL = rh*512 + 64*((n/8)%8) +
    //    c_*8 + n%8  (n = the half-col nn%512)
    //  C2_bo[r*512 + kstep*1024 + n_row] = the FULL microtile dump: the acc
    //    row (p/8)%8 at the col (p/64)*8 + p%8, p = kstep*512 + n_row.
    std::vector<int> c2_ref((size_t)M * N_D);
    for (int t = 0; t < M; t++)
        for (int nn = 0; nn < N_D; nn++) {
            int rh = nn / 512, n = nn % 512;
            long acc = 0;
            for (int col = 0; col < n_cols; col++)
                for (int cg = 0; cg < n_cg; cg++) {
                    int ki = cg * n_cols + col;
                    // The D phase sums ALL 8 k-slices (worker loop `for ks in
                    // range(8)`); the old pad-mode ks_max=1 was the calibration
                    // bug — the full-ks model is the silicon-verified one.
                    const int ks_max = 8;
                    for (int ks = 0; ks < ks_max; ks++)
                        for (int c_ = 0; c_ < 8; c_++) {
                            int h2 = h2s[col * (n_cg * 64) + cg * 64 + t * 8 + c_];
                            int kk = ki * 64 + ks * 8 + n / 64;
                            int bcol = rh * 512 + 64 * ((n / 8) % 8) + c_ * 8 + n % 8;
                            acc += (long)h2 * bd[(size_t)kk * N_D + bcol];
                        }
                }
            c2_ref[(size_t)t * N_D + nn] = (int)acc;
        }
    std::vector<int> scr((size_t)M * N_D, 0);
    for (int kstep = 0; kstep < M; kstep++)
        for (int r = 0; r < 2; r++)
            for (int n_row = 0; n_row < 512; n_row++) {
                int p = kstep * 512 + n_row;
                scr[(size_t)r * 512 + kstep * 1024 + n_row] =
                    c2_ref[(size_t)((p / 8) % 8) * N_D + r * 512 + (p / 64) * 8 + p % 8];
            }
    fprintf(stderr, "CPU mirror C2[0..7] ="); for (int i = 0; i < 8; i++) fprintf(stderr, " %d", c2_ref[i]); fprintf(stderr, "\n");

    // Launch on the NPU
    FILE* f = fopen(argv[3], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> ins(sz / 4); fread(ins.data(), 4, ins.size(), f); fclose(f);
    FILE* xf = fopen(argv[2], "rb"); fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
    std::vector<char> xbuf(xsz); fread(xbuf.data(), 1, xsz, xf); fclose(xf);
    xrt::device dev(0); xrt::xclbin x{xbuf}; dev.register_xclbin(x);
    xrt::hw_context hw(dev, x.get_uuid()); xrt::kernel kk(hw, "MLIR_AIE");
    auto bI = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, kk.group_id(1));
    auto bA = xrt::bo(dev, AB_BYTES, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(3));
    auto bB = xrt::bo(dev, (size_t)M * N_D * 4, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(4));
    auto bC = xrt::bo(dev, (size_t)K_D * N_D, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(5));
    memcpy(bI.map(), ins.data(), ins.size() * 4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bA.map(), ab.data(), AB_BYTES);
    memcpy(bC.map(), bd.data(), (size_t)K_D * N_D);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto r = kk((unsigned)3, bI, (unsigned)ins[2], bA, bB, bC);  // ninstr = ncmds
    r.wait();
    bB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const int32_t* C2 = (const int32_t*)bB.map();
    if (lay) {
        // Layout probe: with the one-hot h2, C2[t][nn] = 127 * (nn & 0x7F), so
        // C2_bo[i]/127 = the OUTPUT COLUMN (mod 128) of the value at position i.
        fprintf(stderr, "LAYOUT PROBE (nn = C2_bo/127):\n");
        fprintf(stderr, "bd[0][0..15] ="); for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)bd[i]); fprintf(stderr, "\n");
        fprintf(stderr, "bd[0][64..71] ="); for (int i = 64; i < 72; i++) fprintf(stderr, " %d", (int)bd[i]); fprintf(stderr, "\n");
        fprintf(stderr, "bd[0][8]=%d bd[0][9]=%d bd[0][72]=%d bd[1][8]=%d bd[8][8]=%d\n",
                (int)bd[8], (int)bd[9], (int)bd[72], (int)bd[N_D + 8], (int)bd[8 * N_D + 8]);
        fprintf(stderr, "C2 row0[0..127] =");
        for (int i = 0; i < 128; i++) { fprintf(stderr, " %d", C2[i]); if (i % 16 == 15) fprintf(stderr, "\n"); }
        // full comparison against the pinned model (V3 + microtile scr)
        // ALSO the +c_ row variant (bd row = ki*64 + ks*8 + c_)
        std::vector<int> c2f(N_D, 0);
        for (int nn = 0; nn < N_D; nn++) {
            int rh = nn / 512, nr = nn % 512;
            long acc = 0;
            for (int col = 0; col < n_cols; col++)
                for (int cg = 0; cg < n_cg; cg++) {
                    int ki = cg * n_cols + col;
                    for (int ks = 0; ks < 8; ks++)
                        for (int c_ = 0; c_ < 8; c_++) {
                            int h2 = h2s[col * (n_cg * 64) + cg * 64 + ks * 8 + c_];
                            int kk = ki * 64 + ks * 8 + c_;
                            int bcol = (nr / 8) * 64 + c_ * 8 + nr % 8;
                            acc += (long)h2 * bd[(size_t)kk * N_D + rh * 512 + bcol];
                        }
                }
            c2f[nn] = (int)acc;
        }
        std::vector<int> scrf((size_t)M * N_D, 0);
        for (int kstep = 0; kstep < M; kstep++)
            for (int r = 0; r < 2; r++)
                for (int n_row = 0; n_row < 512; n_row++)
                    if (n_row % 64 < 8)
                        scrf[(size_t)r * 512 + kstep * 1024 + n_row] =
                            c2f[(size_t)r * 512 + (8 * kstep + n_row / 64) * 8 + n_row % 8];
        long bad = 0, badf = 0;
        for (int i = 0; i < M * N_D; i++) {
            if (C2[i] != scr[i]) bad++;
            if (C2[i] != scrf[i]) badf++;
        }
        printf("LAYOUT probe j0=%d: row0-only=%s(bad=%ld/8192)  full-row=%s(bad=%ld/8192)\n",
               j0_hot, bad == 0 ? "EXACT" : "no", bad, badf == 0 ? "EXACT" : "no", badf);
        for (int i = 0, shown = 0; i < M * N_D && shown < 24; i++)
            if (C2[i] != scr[i]) {
                fprintf(stderr, "i=%d(k%d,r%d,n%d) NPU=%d row0=%d full=%d\n",
                        i, i / 1024, (i % 1024) / 512, i % 512, C2[i], scr[i], scrf[i]);
                shown++;
            }
        return 0;
    }
    fprintf(stderr, "NPU   C2[0..7] ="); for (int i = 0; i < 8; i++) fprintf(stderr, " %d", C2[i]); fprintf(stderr, "\n");
    if (guread) {
        // One-hot A at (0, k0): the acc row t (t*8+hc_ pair) = the h2 of the
        // pair whose gate = 127·B(k0 - 8t, 2p) — the C2_bo[8t+c] shows the
        // h2s; print rows 0..7 at col 0 (the pair t*8+hc_).
        fprintf(stderr, "guread k0=%d hidx=%d: NPU rows(pairs t*8+%d) =", gu_k0, gu_hidx, gu_hidx % 8);
        for (int t = 0; t < 8; t++) fprintf(stderr, " %d", C2[t * 8]);
        fprintf(stderr, "\n");
        // prediction under the CURRENT pack (cg-major + deriv-inverse):
        // the one-hot A at A_tile[k0] feeds A(0, K=k0), and the deriv-inverse
        // pack places w1q[j0+j][ki*64+k0] at the mmul's read position for
        // pair j — so C1[0][2j] = 127·w1q[j0+j][k0], h2 sign = sign(w1q).
        // w1q = q127(w1, gu_is); ki=0, col=0, cg=0 -> j0=0, K_abs=k0.
        fprintf(stderr, "guread PREDICT (sign(g*u), ki=0 col0 cg0):");
        for (int t = 0; t < 8; t++) {
            int j = t * 8;
            int g = j < (int)w1.size() ? (int)q127(w1[(size_t)j * H + gu_k0], gu_is) : -999;
            int u = j < (int)w2.size() ? (int)q127(w2[(size_t)j * H + gu_k0], gu_is) : -999;
            int p = (g == 0 || u == 0) ? 0 : ((g < 0) != (u < 0) ? -127 : 127);
            fprintf(stderr, " %d", p);
        }
        fprintf(stderr, "\n");
        // also the full rows 0..7 × cols 0..7 for the pattern
        fprintf(stderr, "guread C2[0..63] =");
        for (int i = 0; i < 64; i++) { fprintf(stderr, " %d", C2[i]); if (i % 16 == 15) fprintf(stderr, "\n              "); }
        fprintf(stderr, "\n");
        return 0;
    }
    if (bread) {
        // One-hot B at (0, 2*br_p): the acc row 0 (the pair br_p%8 via the
        // hidx) = h2 = silu(A[K])·(A[K']) where the K = the A index whose
        // read (K, 2p) hits the one-hot — the A's VALUE directly observable.
        // Print the acc rows + the mirror's A-tile values at the read indices.
        fprintf(stderr, "bread br_p=%d: NPU rows(pairs t*8+0) =", br_p);
        for (int t = 0; t < 8; t++) fprintf(stderr, " %d", C2[t * 8]);
        fprintf(stderr, "\n");
        // prediction: the pair p (< 8, p ≡ br_p mod 4) gate reads A[K] with
        // K = (2br_p - 2p%8)/8; the up reads A[K'] analog. Show the A values.
        for (int p = br_p % 4; p < 8; p += 4) {
            int K = (2 * br_p - 2 * (p % 4)) / 8;
            int Ku = (2 * br_p + 1 - (2 * p + 1) % 8) / 8;
            const int8_t* A0 = ab.data() + ((long)0 * n_cg * n_k + 0 * n_cg + 0) * AB_tile;
            fprintf(stderr, "  pair %d: K=%d A=%d Ku=%d A'=%d (mirror A values)\n",
                    p, K, K >= 0 && K < 64 ? A0[K] : -999, Ku, Ku >= 0 && Ku < 64 ? A0[Ku] : -999);
        }
        return 0;
    }
    if (h2r) {
        // C2_bo[8t+c] = acc row t at col c (constant per row for the h2r bd) —
        // so C2_bo[8t..8t+7] = the pair (t*8+hc_)'s NPU h2, readable directly.
        fprintf(stderr, "NPU rows(pairs t*8+%d) =", h2r_idx % 8);
        for (int t = 0; t < 8; t++) fprintf(stderr, " %d", C2[t * 8]);
        fprintf(stderr, "\n");
    }
    if (h2r) {
        // The h2r bd isolates the pairs t*8+hc_ into the acc rows; compare the
        // FULL microtile expectation (scr) against the NPU per position — a
        // match pins the GU h2s for the 8 pairs (t*8+hc_, t=0..7).
        long bad = 0; long nz = 0; double maxrel = 0;
        for (int i = 0; i < M * N_D; i++) {
            if (C2[i] != 0) nz++;
            if (C2[i] != scr[i]) bad++;
            double rel = std::fabs((double)C2[i] - scr[i]) / (std::fabs((double)scr[i]) + 1.0);
            if (rel > maxrel) maxrel = rel;
        }
        printf("h2r[%d]: mirror pairs t*8+%d =", h2r_idx, h2r_idx % 8);
        for (int t = 0; t < 8; t++) {
            int pidx = (h2r_idx / 384) * 384 + (h2r_idx % 384 / 64) * 64 + t * 8 + h2r_idx % 8;
            fprintf(stderr, " %d", pidx < 3072 ? h2s[pidx] : -999);
        }
        fprintf(stderr, "  %s (bad=%ld/8192 nz=%ld maxrel=%.2f)\n",
                bad == 0 ? "EXACT" : "MISMATCH", bad, nz, maxrel);
        return bad == 0 ? 0 : 1;
    }
    fprintf(stderr, "mirror C2[0..7] ="); for (int i = 0; i < 8; i++) fprintf(stderr, " %d", scr[i]); fprintf(stderr, " (untiled c2[0]=%d)\n", c2_ref[0]);
    long bad = 0; double maxrel = 0; long bad_r0 = 0;
    for (int i = 0; i < M * N_D; i++) {
        if (C2[i] != scr[i]) bad++;
        double rel = std::fabs((double)C2[i] - scr[i]) / (std::fabs((double)scr[i]) + 1.0);
        if (rel > maxrel) maxrel = rel;
    }
    for (int i = 0; i < N_D; i++)
        if (C2[i] != scr[i]) bad_r0++;
    printf("cascade real-weight [%s]: %s (bad=%ld/%d row0_bad=%ld/%d maxrel=%.4f)  gu_scale=%.6f d_scale=%.6f ascale=%.6f\n",
           rep ? "rep" : "pad", bad == 0 ? "EXACT MATCH" : "MISMATCH", bad, M * N_D, bad_r0, N_D,
           maxrel, gu_scale, d_scale, ascale);
    return bad == 0 ? 0 : 1;
}
