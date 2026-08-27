// test_i4_silu_q22.cpp — CPU gate for the fused int4 silu fixed-point
// contract (issue #1769, blocker #1844).
//
// The on-core int4 silu (silu_quant_i8_fused_i4 in mm_kernel_reference.cc,
// per-pair arithmetic silu_pair_q22 in silu_quant.h) is PURE int32 because
// the aie2p backend mis-compiles the float loop (#1836) and int64 (#1843).
// The v50/v51 Q22 formulation was broken on the real weights:
//   (a) gQ22 = c1*fold and uQ22 = c1*fold overflowed int32 for |g|>512 /
//       |u|>512 — wrapped garbage and the reported "host h2=12 -> NPU 0"
//       zero pairs (corr ~ -0.02 on strixhalo);
//   (b) the fixed Q22 fold rounded the small per-column scales to zero.
// This test emulates the FIXED (v59) kernel arithmetic bit-exactly on x86
// and compares it against the float reference (silu_quant.h silu_lut),
// on realistic synthetic (c1, S') pairs — the same magnitude structure as
// the measured zaya data (gate g = c1*S' ~ O(1), up u ~ O(100), S'
// log-uniform over ~5 decades, c1 = g/S').
//
// Usage:
//   g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src \
//       engine/npu/tests/test_i4_silu_q22.cpp -o /tmp/t && /tmp/t
//   (real-data mode, mirroring test_i4_grouped_fused.cpp:
//    /tmp/t zaya1-8b.q4nx [layer] [expert] [activation.bin])
//
// Gates (must hold on the synthetic set AND on the real weights):
//   corr(kernel, float) >= 0.999
//   >= 98% of pairs within |dH2| <= 1; max |dH2| <= 8

#include "silu_quant.h"
#include "gu_i4_pack.h"   // write_silu_pad_meta — the REAL host pad writer
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ── float reference: h2 = sat8(round(silu_lut(g)·u)), g = c1g·sg, u = c1u·su
static int ref_pair_float(int32_t c1g, int32_t c1u, float sg, float su) {
    float g = (float)c1g * sg;
    float u = (float)c1u * su;
    float h = silu_lut(g) * u;
    return silu_sat8(silu_roundf(h));
}

// ── host fold math (the REAL host code: write_silu_pad_meta) ──
// Per 64-pair tile the host writes, into each tile's pad:
// Q (22 - s, s from the tile MIN |S'|), foldG = round(S'·2^Q),
// boundG = (2^31-1)/|foldG|, boundU = 4·((2^31-1)/|foldG|)+3. The test calls
// the shared gu_i4_pack.h writer on a dummy tile and reads the int32s back,
// so it gates the ACTUAL host math, not a copy.
struct TileMeta {
    int Q;
    int shg, shu;   // v60: Q-11, Q-7 (host-precomputed; aie2p miscompiles register shifts)
    std::vector<int32_t> foldg, foldu, boundg, boundu;
};
static TileMeta host_tile_meta(const std::vector<float>& sg,
                               const std::vector<float>& su, int t0, int n) {
    // interleave gate/up scales into the N-col scol layout the host sees:
    // col 2p = gate[p], col 2p+1 = up[p] (write_silu_pad_meta folds ag/qn_s
    // itself; pass ag=1, qn_s=1 and pre-fold here).
    static thread_local uint8_t tile[4][8192];
    static thread_local std::vector<float> scol;
    static thread_local int scolN = 0;
    int N = 2 * n;
    if (scolN < N) { scol.resize(N); scolN = N; }
    for (int i = 0; i < n; i++) {
        scol[2 * i]     = sg[t0 + i];
        scol[2 * i + 1] = su[t0 + i];
    }
    // v65: the silu metadata is CHUNKED across the col_group's k-tiles at
    // [META_BASE..META_BASE+512): ki%4==0 foldG, 1 boundG, 2 boundU,
    // 3 Q/shG/shU — one 512-B chunk per k-tile, assembled by the kernel
    // into C1 rows 1-4 (the silu reads foldG/boundG/boundU/Q/shG/shU from
    // there). Write the four chunks into separate dummy tiles (each chunk
    // lives in a DIFFERENT k-tile in the real BO).
    for (int kchunk = 0; kchunk < 4; kchunk++)
        write_silu_pad_meta(tile[kchunk], scol.data(), 0, kchunk, 1.0f, 1.0f, N);
    TileMeta m;
    const int32_t* qm = (const int32_t*)(tile[3] + GuI4Pack::META_BASE);
    m.Q = qm[0];     // ki%4==3 chunk: Q at [0]
    m.shg = qm[1];   // shG
    m.shu = qm[2];   // shU
    const int32_t* fq = (const int32_t*)(tile[0] + GuI4Pack::META_BASE);   // foldG
    const int32_t* bg = (const int32_t*)(tile[1] + GuI4Pack::META_BASE);   // boundG
    const int32_t* bu = (const int32_t*)(tile[2] + GuI4Pack::META_BASE);   // boundU
    // the pad layout is per GU COLUMN (interleaved): col 2i = gate of pair i,
    // col 2i+1 = up of pair i.
    m.foldg.resize(n); m.foldu.resize(n);
    m.boundg.resize(n); m.boundu.resize(n);
    for (int i = 0; i < n; i++) {
        m.foldg[i] = fq[2 * i];
        m.foldu[i] = fq[2 * i + 1];
        m.boundg[i] = bg[2 * i];
        m.boundu[i] = bu[2 * i + 1];
    }
    return m;
}

static double corr(const std::vector<int>& a, const std::vector<int>& b) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); i++) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= a.size();
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) * (a[i] - ma);
        db += (b[i] - mb) * (b[i] - mb);
    }
    return num / std::sqrt(da * db);
}

// Run the gate on a set of (c1g, c1u, sg, su) pairs, 64-pair tiles.
// Returns the number of failing gates (0 = pass).
static int run_gate(const char* name, const std::vector<int32_t>& c1g,
                    const std::vector<int32_t>& c1u,
                    const std::vector<float>& sg,
                    const std::vector<float>& su) {
    const size_t N = c1g.size();
    const int T = 64;
    std::vector<int> hr(N), hk(N);
    int nzero = 0, nbad = 0, worst = 0;
    double rms_r = 0, rms_k = 0;
    for (size_t t0 = 0; t0 < N; t0 += T) {
        int n = (int)std::min<size_t>(T, N - t0);
        TileMeta m = host_tile_meta(sg, su, (int)t0, n);
        for (int i = 0; i < n; i++) {
            size_t idx = t0 + i;
            hr[idx] = ref_pair_float(c1g[idx], c1u[idx], sg[idx], su[idx]);
            hk[idx] = silu_pair_q22(c1g[idx], c1u[idx], m.foldg[i], m.foldu[i],
                                    m.boundg[i], m.boundu[i], m.Q, m.shg, m.shu);
            if (hr[idx] != 0 && hk[idx] == 0) nzero++;
            int d = std::abs(hr[idx] - hk[idx]);
            if (d > 1) nbad++;
            if (d > worst) worst = d;
            rms_r += (double)hr[idx] * hr[idx];
            rms_k += (double)hk[idx] * hk[idx];
        }
    }
    rms_r = std::sqrt(rms_r / N);
    rms_k = std::sqrt(rms_k / N);
    double c = corr(hr, hk);
    double within = 100.0 * (double)(N - nbad) / (double)N;
    std::printf("[%s] pairs=%zu corr=%.6f within+/-1=%.2f%% zero-pairs=%d "
                "worst|dH2|=%d rms(ref)=%.2f rms(kern)=%.2f\n",
                name, N, c, within, nzero, worst, rms_r, rms_k);
    int fails = 0;
    if (!(c >= 0.999)) { std::fprintf(stderr, "FAIL: corr %.4f < 0.999\n", c); fails++; }
    if (!(within >= 98.0)) { std::fprintf(stderr, "FAIL: %.2f%% within +-1 < 98%%\n", within); fails++; }
    if (!(worst <= 8)) { std::fprintf(stderr, "FAIL: worst |dH2| %d > 8\n", worst); fails++; }
    if (fails == 0) std::printf("[%s] PASS\n", name);
    return fails;
}

// ── kernel-indexing gate (v66) ──
// run_gate validates the silu_pair_q22 ARITHMETIC with semantic values. This
// gate additionally emulates the KERNEL's exact C1-row metadata assembly and
// extraction so the INDEXING contract (where each value lives in the
// microtiled C1buf) is pinned on x86, not discovered on strixhalo:
//   - the four chunk tiles (ki%4 = foldG / boundG / boundU / Q+shG+shU) come
//     from the REAL host writer (write_silu_pad_meta), exactly as the
//     col_group's k-tile stream carries them;
//   - the kernel assembles them into C1buf rows 1-4 at the row-0 microtile
//     positions + r*8 (row r col c sits at (c/8)*64 + r*8 + c%8);
//   - the silu extracts per pair p at gos[p]: folds at +8/+9 (row 1),
//     boundG at +16 (row 2), boundU at +25 (row 3, the UP column — the v65
//     +24 off-by-one used the GATE col's boundU and FAILS here: the up-clamp
//     threshold then belongs to foldg, not foldu), Q/shG/shU at C1[32..34].
// A drift in ANY of these indices (or in the assembly offsets) fails this
// gate before the kernel is ever flashed.
static const int gos_table[64] = {
    0, 2, 4, 6, 64, 66, 68, 70, 128, 130, 132, 134, 192, 194, 196, 198,
    256, 258, 260, 262, 320, 322, 324, 326, 384, 386, 388, 390, 448, 450, 452, 454,
    512, 514, 516, 518, 576, 578, 580, 582, 640, 642, 644, 646, 704, 706, 708, 710,
    768, 770, 772, 774, 832, 834, 836, 838, 896, 898, 900, 902, 960, 962, 964, 966 };
static int run_kernel_index_gate(const char* name,
                                 const std::vector<int32_t>& c1g,
                                 const std::vector<int32_t>& c1u,
                                 const std::vector<float>& sg,
                                 const std::vector<float>& su) {
    const size_t N = c1g.size();
    const int T = 64;
    std::vector<int> hr(N), hk(N);
    int nzero = 0, nbad = 0, worst = 0;
    double rms_r = 0, rms_k = 0;
    static thread_local uint8_t tile[4][8192];
    static thread_local std::vector<float> scol;
    static thread_local int scolN = 0;
    for (size_t t0 = 0; t0 < N; t0 += T) {
        int n = (int)std::min<size_t>(T, N - t0);
        int Ncol = 2 * n;
        if (scolN < Ncol) { scol.resize(Ncol); scolN = Ncol; }
        for (int i = 0; i < n; i++) {
            scol[2 * i]     = sg[t0 + i];
            scol[2 * i + 1] = su[t0 + i];
        }
        // 1) the four chunk tiles, exactly as the k-tile stream carries them
        for (int kchunk = 0; kchunk < 4; kchunk++)
            write_silu_pad_meta(tile[kchunk], scol.data(), 0, kchunk, 1.0f, 1.0f, Ncol);
        // 2) kernel assembly into a (8,128) int32 microtiled C1buf
        int32_t C1[8 * 128];
        std::memset(C1, 0, sizeof C1);
        for (int kchunk = 0; kchunk < 4; kchunk++) {
            const int32_t* mq = (const int32_t*)(tile[kchunk] + GuI4Pack::META_BASE);
            if (kchunk == 3) {
                C1[32] = mq[0];   // Q   (row 4 col 0)
                C1[33] = mq[1];   // shG (row 4 col 1)
                C1[34] = mq[2];   // shU (row 4 col 2)
            } else {
                const unsigned rowoff = 8 + (unsigned)kchunk * 8;   // row 1/2/3
                for (int j = 0; j < 128; j++) {
                    unsigned p0 = (j / 8) * 64 + (j % 8);
                    C1[p0 + rowoff] = mq[j];
                }
            }
        }
        // 3) row-0 C1 dots (the real GU GEMM values for this tile's cols;
        // pairs >= n of the partial last tile have no reference and are never
        // read by the extraction below, so leave their dots zero)
        for (int j = 0; j < 128; j++) {
            unsigned p0 = (j / 8) * 64 + (j % 8);
            int pair = j / 2;
            if (pair < n)
                C1[p0] = (j % 2 == 0) ? c1g[t0 + pair] : c1u[t0 + pair];
        }
        const int Q = C1[32], shG = C1[33], shU = C1[34];
        // 4) kernel extraction per pair (the EXACT indices the kernel uses)
        for (int i = 0; i < n; i++) {
            size_t idx = t0 + i;
            int go = gos_table[i];
            hr[idx] = ref_pair_float(c1g[idx], c1u[idx], sg[idx], su[idx]);
            hk[idx] = silu_pair_q22(C1[go], C1[go + 1],
                                    C1[go + 8], C1[go + 9],     // foldg, foldu (row 1)
                                    C1[go + 16], C1[go + 25],   // boundg (row 2), boundu (row 3, UP col)
                                    Q, shG, shU);
            if (hr[idx] != 0 && hk[idx] == 0) nzero++;
            int d = std::abs(hr[idx] - hk[idx]);
            if (d > 1) nbad++;
            if (d > worst) worst = d;
            rms_r += (double)hr[idx] * hr[idx];
            rms_k += (double)hk[idx] * hk[idx];
        }
    }
    rms_r = std::sqrt(rms_r / N);
    rms_k = std::sqrt(rms_k / N);
    double c = corr(hr, hk);
    double within = 100.0 * (double)(N - nbad) / (double)N;
    std::printf("[%s] pairs=%zu corr=%.6f within+/-1=%.2f%% zero-pairs=%d "
                "worst|dH2|=%d rms(ref)=%.2f rms(kern)=%.2f\n",
                name, N, c, within, nzero, worst, rms_r, rms_k);
    int fails = 0;
    if (!(c >= 0.999)) { std::fprintf(stderr, "FAIL: corr %.4f < 0.999\n", c); fails++; }
    if (!(within >= 98.0)) { std::fprintf(stderr, "FAIL: %.2f%% within +-1 < 98%%\n", within); fails++; }
    if (!(worst <= 8)) { std::fprintf(stderr, "FAIL: worst |dH2| %d > 8\n", worst); fails++; }
    if (fails == 0) std::printf("[%s] PASS\n", name);
    return fails;
}


// ── real-data mode (mirrors test_i4_grouped_fused.cpp) ──
static int get_top_int(const char* js, size_t jl, const char* key) {
    char pat[128]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* q = strstr(js, pat);
    if (!q) return 0;
    q = strchr(q, ':');
    if (!q) return 0;
    return atoi(q + 1);
}
static bool get_offsets(const char* js, size_t jl, const char* key,
                        uint64_t* off, uint64_t* size) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) {
                auto b = strchr(o, '[');
                if (b) {
                    *off  = (uint64_t)strtoull(b + 1, nullptr, 10);
                    auto c = strchr(b + 1, ',');
                    if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *size > 0;
                }
            }
        }
        p = q + kl;
    }
    return false;
}
static int run_real_gate(const char* q4nx_path, int L, int E, const char* actfile) {
    int fd = open(q4nx_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { perror("mmap"); return 1; }
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    const uint8_t* D = md + 8 + hsz;
    int H = get_top_int(js, hsz, "hidden_size");
    int NC = get_top_int(js, hsz, "num_hidden_layers");
    int n_ff = get_top_int(js, hsz, "intermediate_size");
    int n_exp = get_top_int(js, hsz, "num_experts");
    if (L % 2 == 0 || L >= NC) { fprintf(stderr, "layer %d is not MoE (NC=%d)\n", L, NC); return 1; }
    char key[256];
    snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", L);
    uint64_t off, size;
    if (!get_offsets(js, hsz, key, &off, &size)) { fprintf(stderr, "no GU tensor\n"); return 1; }
    int gu_i8_rows = (n_exp * 2 * n_ff / 32) * (H / 256);
    auto raw_all = read_q4nx_raw(D, off, gu_i8_rows, H);
    // slice to expert E (rows [E*2n_ff, (E+1)*2n_ff): gate [0,n_ff), up [n_ff,2n_ff))
    RawQ4Tensor raw_gu;
    raw_gu.rows = 2 * n_ff; raw_gu.cols = H;
    raw_gu.q4.assign((size_t)raw_gu.rows * H, 0);
    raw_gu.scl.assign((size_t)raw_gu.rows * (H / 32), 0.0f);
    raw_gu.zp.assign((size_t)raw_gu.rows * (H / 32), 0.0f);   // Zaya symmetric: zeros
    const size_t gbase = (size_t)E * 2 * n_ff;
    for (int r = 0; r < 2 * n_ff; r++) {
        memcpy(&raw_gu.q4[(size_t)r * H], &raw_all.q4[(gbase + r) * H], sizeof(int8_t) * H);
        memcpy(&raw_gu.scl[(size_t)r * (H / 32)], &raw_all.scl[(gbase + r) * (H / 32)],
               sizeof(float) * (H / 32));
    }
    auto pack = pack_gu_fused_i4(raw_gu, 0, H, n_ff);   // kernel-exact B_shadow + S_col
    // A activation (raw floats, unit-RMS like the grouped test)
    std::vector<float> A(H);
    if (actfile) {
        FILE* f = fopen(actfile, "rb");
        if (!f || fread(A.data(), 4, H, f) != (size_t)H) {
            fprintf(stderr, "cannot read activation %s\n", actfile ? actfile : ""); return 1;
        }
        if (f) fclose(f);
    } else {
        std::mt19937 rng(1); std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < H; i++) A[i] = nd(rng);
    }
    double ss = 0; for (int i = 0; i < H; i++) ss += A[i] * A[i];
    float rms = std::sqrt(ss / H); if (rms < 1e-9f) rms = 1;
    for (int i = 0; i < H; i++) A[i] /= rms;
    float ag = 0; for (int i = 0; i < H; i++) ag = std::max(ag, std::fabs(A[i]));
    if (ag < 1e-9f) ag = 1;
    // c1 = Aq^T * B_shadow (the kernel's int8 GU GEMM; grouped-test Aq = round(A/ag))
    const size_t N = 2 * (size_t)n_ff;
    std::vector<int32_t> c1(N, 0);
    for (int i = 0; i < H; i++) {
        int aq = (int)std::roundf(A[i] / ag);
        const int8_t* b = pack.B_shadow.data() + (size_t)i * N;
        for (size_t j = 0; j < N; j++) c1[j] += (int32_t)aq * b[j];
    }
    // float h2 (pre-qn_s) and the per-token qn_s (the host's fold factor)
    std::vector<float> h2f(n_ff);
    for (int p = 0; p < n_ff; p++) {
        float g = (float)c1[2 * p] * ag * pack.scol[2 * p];
        float u = (float)c1[2 * p + 1] * ag * pack.scol[2 * p + 1];
        h2f[p] = silu_lut(g) * u;
    }
    float mx = 0; for (int p = 0; p < n_ff; p++) mx = std::max(mx, std::fabs(h2f[p]));
    float qn_s = mx > 0 ? 127.0f / mx : 1.0f;
    fprintf(stderr, "real: H=%d n_ff=%d n_exp=%d expert=%d ag=%.4f qn_s=%.3f max|h2f|=%g\n",
            H, n_ff, n_exp, E, ag, qn_s, mx);
    std::vector<int32_t> c1g(n_ff), c1u(n_ff);
    std::vector<float> sg(n_ff), su(n_ff);
    for (int p = 0; p < n_ff; p++) {
        c1g[p] = c1[2 * p];
        c1u[p] = c1[2 * p + 1];
        sg[p] = ag * pack.scol[2 * p];            // S' gate
        su[p] = ag * qn_s * pack.scol[2 * p + 1]; // S' up (qn_s folded)
    }
    munmap(md, st.st_size);
    // kernel-indexing gate too: the real-data set must satisfy BOTH the
    // arithmetic contract and the kernel's exact C1-row extraction indices.
    return run_gate("real-zaya", c1g, c1u, sg, su)
         + run_kernel_index_gate("kernidx-real-zaya", c1g, c1u, sg, su);
}

int main(int argc, char** argv) {
    int fails = 0;
    if (argc > 1) {
        // Real-data mode: load the Q4NX weights + a (real or synthetic) MoE
        // input, compute the int4 GU GEMM c1 from the packer's kernel-exact
        // B_shadow and the per-column S' fold, then gate kernel-vs-float silu
        // on the REAL weight/scale distribution (the strixhalo round's gate).
        const int L = argc > 2 ? atoi(argv[2]) : 1;   // odd = MoE layer
        const int E = argc > 3 ? atoi(argv[3]) : 0;   // expert
        const char* actfile = argc > 4 ? argv[4] : nullptr;
        return run_real_gate(argv[1], L, E, actfile);
    }

    // ── Synthetic realistic envelope ──
    // gate: g ~ N(0, 1.2) clamped +-8 (measured gate range ~[-3.4, 3.4]);
    // up: u ~ +-10^U(-0.5, 2.8) (measured up ~ +-74..250, tails to ~600);
    // S': log-uniform over [1e-5.5, 1e-1.5] (the "small weights majority"
    // spans ~5 decades); c1 = g/S' (the c1 and S' anti-correlate so the
    // pre-activations stay O(1), exactly like the real GU GEMM).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sdist(-5.5f, -1.5f);
    std::uniform_real_distribution<float> udist(-0.5f, 2.8f);
    std::normal_distribution<float> gdist(0.0f, 1.2f);
    std::bernoulli_distribution sign(0.5f);
    const int N = 60000;
    std::vector<int32_t> c1g(N), c1u(N);
    std::vector<float> sg(N), su(N);
    auto clamp_c1 = [](double v) -> int32_t {
        if (v > 33000000.0) v = 33000000.0;
        if (v < -33000000.0) v = -33000000.0;
        return (int32_t)(v >= 0 ? std::floor(v + 0.5) : std::ceil(v - 0.5));
    };
    for (int i = 0; i < N; i++) {
        sg[i] = std::pow(10.0f, sdist(rng));
        su[i] = std::pow(10.0f, sdist(rng));
        double g = gdist(rng);
        if (g > 8.0) g = 8.0;
        if (g < -8.0) g = -8.0;
        double u = std::pow(10.0, udist(rng));
        if (sign(rng)) u = -u;
        c1g[i] = clamp_c1(g / sg[i]);
        c1u[i] = clamp_c1(u / su[i]);
    }
    fails += run_gate("synthetic-v59", c1g, c1u, sg, su);
    fails += run_kernel_index_gate("kernidx-synthetic", c1g, c1u, sg, su);

    // ── adversarial corners ──
    // the reported failure class: small gate (silu ~ 0.15..0.02) with up
    // around the old Q22 overflow boundary |u| ~ 550..650 (the old uQ22 =
    // c1*fold wrapped there -> "host h2=12 -> NPU 0"), plus tiny-gate/huge-up
    // corners (g ~ 0.001..0.05, u ~ 500..2000) that the v50 truncations
    // zeroed.
    std::vector<int32_t> a1g, a1u;
    std::vector<float> a1sg, a1su;
    std::uniform_real_distribution<float> gcorner(-0.7f, 0.7f);
    std::uniform_real_distribution<float> ucorner(550.0f, 650.0f);
    std::uniform_real_distribution<float> gtiny(0.001f, 0.05f);
    std::uniform_real_distribution<float> utiny(500.0f, 2000.0f);
    std::normal_distribution<float> scat(0.0f, 1.0f);
    for (int i = 0; i < 2000; i++) {
        float s_g = std::pow(10.0f, sdist(rng));
        float s_u = std::pow(10.0f, sdist(rng));
        a1sg.push_back(s_g); a1su.push_back(s_u);
        if (i % 2 == 0) {
            double g = gcorner(rng);
            double u = ucorner(rng);
            if (sign(rng)) u = -u;
            a1g.push_back(clamp_c1(g / s_g));
            a1u.push_back(clamp_c1(u / s_u));
        } else {
            double g = gtiny(rng);
            double u = utiny(rng);
            if (sign(rng)) u = -u;
            a1g.push_back(clamp_c1(g / s_g));
            a1u.push_back(clamp_c1(u / s_u));
        }
    }
    fails += run_gate("adversarial-u600", a1g, a1u, a1sg, a1su);
    fails += run_kernel_index_gate("kernidx-adversarial", a1g, a1u, a1sg, a1su);

    if (fails == 0) {
        std::printf("ALL GATES PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d GATE(S) FAILED\n", fails);
    return 1;
}
