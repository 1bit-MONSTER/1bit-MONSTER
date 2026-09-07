// zaya_merge_q4nx.cpp — merge trained LoRA adapters into a Zaya Q4NX model.
//
// Reads:  base model.q4nx (mmap) + an adapters.bin checkpoint produced by the
//         M2 trainer (zaya_train_main.cpp save_adapters)
// Writes: merged model.q4nx — a copy of the base with ONLY the adapter-
//         affected tensors re-encoded in place:
//           CCA layers (even):  q/k/v_cur/v_del/o projections
//           MoE  layers (odd):  experts.gate_up_proj + experts.down_proj
//         Tensor byte lengths are fixed by the tile geometry, so the manifest
//         and all offsets stay valid; untouched tensors are byte-identical.
//
// Tile format (Zaya signed INT4, see dequant_q4nx.cpp):
//   per tile row (5120 B): [512 B bf16 scales row-major (lr*8+g)]
//                          [512 B bf16 zeros (0)]
//                          [4096 B nibble data: lane*2048 + col*8 + (lr%16)/2,
//                           low nibble = even tile row, int4 two's-complement]
//
// Build (engine/npu/src): g++ -O2 -std=c++20 -I. -o /tmp/zaya_merge \
//       ../../../tools/zaya_merge_q4nx.cpp dequant_q4nx.cpp -fopenmp
// Run: /tmp/zaya_merge base.q4nx adapters.bin merged.q4nx
//
// (Optional) env ADAPTER_CFG=cca_first|none unused; all adapter files are in
// the checkpoint layout: per-CCA [Bq,Aq,Bk,Ak,Bv1,Av1,Bv2,Av2,Bo,Ao], per-MoE
// [Bg,Ag,Bd,Ad], dims[3]={H,L,r} trailer.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "dequant_q4nx.h"

// ── manifest helpers (same as dump_zaya_f32.cpp) ─────────────────────────
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
static int get_top_int(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto c = strchr(q + kl, ':');
            if (c) return (int)strtoll(c + 1, nullptr, 10);
        }
        p = q + kl;
    }
    return 0;
}

// ── adapter checkpoint reader (mirror of trainer save_adapters) ─────────
struct Adapters {
    struct CCA { std::vector<double> Bq,Aq,Bk,Ak,Bv1,Av1,Bv2,Av2,Bo,Ao; };
    struct MOE { std::vector<double> Bg,Ag,Bd,Ad; };
    std::vector<CCA> cca;
    std::vector<MOE> moe;
    int H = 0, L = 0, r = 0;
};
static bool load_adapters(const char* path, Adapters& A, int H, int L, int r) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open adapters %s\n", path); return false; }
    auto rd = [&](std::vector<double>& v) {
        size_t n = 0; if (fread(&n, sizeof n, 1, f) != 1) return false;
        v.resize(n); return fread(v.data(), sizeof(double), n, f) == n;
    };
    int ncca = 0; for (int l = 0; l < L; l++) if (l % 2 == 0) ncca++;
    int nmoe = L / 2;
    A.cca.resize(ncca); A.moe.resize(nmoe);
    for (int ci = 0; ci < ncca; ci++) {
        Adapters::CCA& c = A.cca[ci];
        if (!rd(c.Bq) || !rd(c.Aq) || !rd(c.Bk) || !rd(c.Ak) || !rd(c.Bv1) || !rd(c.Av1) ||
            !rd(c.Bv2) || !rd(c.Av2) || !rd(c.Bo) || !rd(c.Ao)) { fclose(f); return false; }
    }
    for (int mi = 0; mi < nmoe; mi++) {
        Adapters::MOE& m = A.moe[mi];
        if (!rd(m.Bg) || !rd(m.Ag) || !rd(m.Bd) || !rd(m.Ad)) { fclose(f); return false; }
    }
    int dims[3] = {0,0,0};
    if (fread(dims, sizeof dims, 1, f) == 1) { A.H = dims[0]; A.L = dims[1]; A.r = dims[2]; }
    fclose(f);
    fprintf(stderr, "adapters: cca=%d moe=%d dims(H=%d L=%d r=%d)\n", ncca, nmoe, A.H, A.L, A.r);
    if (A.H != H || A.L != L || A.r != r) {
        fprintf(stderr, "adapter dims mismatch vs model (H=%d L=%d r=%d)\n", H, L, r);
        return false;
    }
    return true;
}

// ── Q4NX asymmetric-int4 re-encode (preserves the original per-(row,group)
//    scale + zero-point; only the val nibbles are re-derived) ────────────
static uint16_t f32_to_bf16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return 0;
    return (uint16_t)(u >> 16);
}
static inline float bf16_bits_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, 4); return f;
}
static void encode_tile(const float* w, int tile_row, int tile_col,
                        int out_cols, const uint8_t* orig_row, uint8_t* row) {
    constexpr int TR = 32, TC = 256;
    // carry the original scales + zero-points verbatim
    memcpy(row, orig_row, 1024);
    uint8_t* packed = row + 1024;
    // precompute per (lr, g) scale/zp from the ORIGINAL row
    float sc[TR][8], zp[TR][8];
    for (int lr = 0; lr < TR; lr++)
        for (int g = 0; g < 8; g++) {
            const uint8_t* ps = orig_row + (lr * 8 + g) * 2;
            const uint8_t* pz = orig_row + 512 + (lr * 8 + g) * 2;
            uint16_t sraw = (uint16_t)(ps[0] | (ps[1] << 8));
            uint16_t zraw = (uint16_t)(pz[0] | (pz[1] << 8));
            sc[lr][g] = bf16_bits_to_f32(sraw);
            zp[lr][g] = bf16_bits_to_f32(zraw);
        }
    for (int lr = 0; lr < TR; lr++) {
        int lane = lr / 16;
        int byte_idx = (lr % 16) / 2;
        bool lo = (lr % 2) == 0;
        for (int col = 0; col < TC; col++) {
            int g = col / 32;
            float s = sc[lr][g], z = zp[lr][g];
            float v = w[((size_t)tile_row * TR + lr) * out_cols +
                        ((size_t)tile_col * TC + col)];
            if (!(s > 0) || !(s < 1e30f)) { continue; }   // degenerate group: leave bytes as-is
            double qd = std::floor((double)(v - z) / s + 0.5);
            if (qd > 7.0) qd = 7.0;
            if (qd < -8.0) qd = -8.0;
            uint8_t nib = (uint8_t)(((int)qd) & 0x0F);
            size_t off = (size_t)lane * 2048 + (size_t)col * 8 + byte_idx;
            uint8_t& b = packed[off];
            if (lo) b = (uint8_t)((b & 0xF0) | nib);
            else    b = (uint8_t)((b & 0x0F) | (nib << 4));
        }
    }
}
static bool put_tensor(uint8_t* D, uint64_t rel_off, const uint8_t* orig,
                       const float* w, int out_rows, int in_features) {
    constexpr int TR = 32, TC = 256;
    int n_tile_cols = in_features / TC;
    int n_tile_rows = out_rows / TR;
    if (out_rows % TR || in_features % TC) return false;
    for (int tr = 0; tr < n_tile_rows; tr++)
        for (int tc = 0; tc < n_tile_cols; tc++) {
            size_t tidx = (size_t)(tr * n_tile_cols + tc);
            uint8_t* dst = D + rel_off + tidx * 5120;
            const uint8_t* src = orig + tidx * 5120;
            encode_tile(w, tr, tc, n_tile_cols * TC, src, dst);
        }
    return true;
}
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s base.q4nx adapters.bin out.q4nx\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open base"); return 1; }
    struct stat st; fstat(fd, &st);
    const uint8_t* md = (const uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hsz;

    int H  = get_top_int(js, jl, "hidden_size");
    int NC = get_top_int(js, jl, "num_hidden_layers");
    int NV = get_top_int(js, jl, "vocab_size");
    int nq = get_top_int(js, jl, "num_attention_heads");
    int nkv = get_top_int(js, jl, "num_key_value_heads");
    int hd = get_top_int(js, jl, "head_dim");
    int nff = get_top_int(js, jl, "intermediate_size");
    int n_exp = get_top_int(js, jl, "num_experts");
    int qd = nq * hd, kd = nkv * hd, hv2 = kd / 2;
    int rtr = 256;
    fprintf(stderr, "model H=%d L=%d nq=%d nkv=%d hd=%d qd=%d kd=%d nff=%d n_exp=%d\n",
            H, NC, nq, nkv, hd, qd, kd, nff, n_exp);

    Adapters A;
    if (!load_adapters(argv[2], A, H, NC, argc > 5 ? atoi(argv[5]) : 2)) return 1;
    const int r = A.r;

    // output = copy of base
    std::string outtmp = std::string(argv[3]) + ".tmp";
    FILE* fo = fopen(outtmp.c_str(), "wb");
    if (!fo) { perror("fopen out"); return 1; }
    fwrite(md, 1, st.st_size, fo);
    fclose(fo);
    int ofd = open(outtmp.c_str(), O_RDWR);
    if (ofd < 0) { perror("open outtmp"); return 1; }
    struct stat ost; fstat(ofd, &ost);
    uint8_t* D = (uint8_t*)mmap(nullptr, ost.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, ofd, 0);
    close(ofd);
    uint64_t data0 = 8 + hsz;   // data section start (offsets in manifest are relative)

    uint64_t o_, s_;
    char key[256];
    size_t changed = 0;

    // ── CCA-layer projections (even l): q,k,v_cur,v_del (in=H), o (in=qd) ──
    for (int l = 0; l < NC; l += 2) {
        int ci = l / 2;
        const Adapters::CCA& ad = A.cca[ci];
        struct PJ { const char* name; int out; int in; const double* B; const double* AA; };
        std::vector<PJ> pjs = {
            {"model.layers.%d.self_attn.q_proj.weight",         qd,  H,   ad.Bq.data(),  ad.Aq.data()},
            {"model.layers.%d.self_attn.k_proj.weight",         kd,  H,   ad.Bk.data(),  ad.Ak.data()},
            {"model.layers.%d.self_attn.v_proj_current.weight", hv2, H,   ad.Bv1.data(), ad.Av1.data()},
            {"model.layers.%d.self_attn.v_proj_delayed.weight", hv2, H,   ad.Bv2.data(), ad.Av2.data()},
            {"model.layers.%d.self_attn.o_proj.weight",         H,   qd,  ad.Bo.data(),  ad.Ao.data()},
        };
        for (auto& pj : pjs) {
            snprintf(key, sizeof key, pj.name, l);
            if (!get_offsets(js, jl, key, &o_, &s_)) { fprintf(stderr, "missing %s\n", key); continue; }
            int orows = 0, ocols = 0;
            int tiles = (int)(s_ / 5120);
            float* base = dequant_i8_signed_to_float_ex(md + 8 + hsz + o_, tiles, pj.in, &orows, &ocols);
            if (!base) { fprintf(stderr, "dequant fail %s\n", key); continue; }
            if (orows != pj.out || ocols != pj.in) {
                fprintf(stderr, "%s: dequant %dx%d != %dx%d\n", key, orows, ocols, pj.out, pj.in);
                free(base); continue;
            }
            // merged = base + B @ A
            std::vector<float> m((size_t)pj.out * pj.in);
            for (int i = 0; i < pj.out; i++) {
                const double* Bi = pj.B + (size_t)i * r;
                double dacc[4] = {0,0,0,0};
                for (int k = 0; k < r; k++) dacc[k] = Bi[k];
                for (int j = 0; j < pj.in; j++) {
                    double d = 0;
                    for (int k = 0; k < r; k++) d += dacc[k] * pj.AA[(size_t)k * pj.in + j];
                    m[(size_t)i * pj.in + j] = base[(size_t)i * pj.in + j] + (float)d;
                }
            }
            put_tensor(D + data0, o_, md + 8 + hsz + o_, m.data(), pj.out, pj.in);
            free(base);
            fprintf(stderr, "  CCA l=%d %s (%d x %d) rewritten @%llu\n", l, key, pj.out, pj.in,
                    (unsigned long long)o_);
            changed++;
        }
    }
    // ── MoE experts (odd l): gu = [n_exp*2*nff, H], dn = [n_exp*H, nff] ──
    for (int l = 1; l < NC; l += 2) {
        int mi = l / 2;
        const Adapters::MOE& ad = A.moe[mi];
        // gate_up_proj: rows n_exp*2nff, in H; delta per expert e: Bg_e (2nff x r) @ Ag_e (r x H)
        snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", l);
        if (!get_offsets(js, jl, key, &o_, &s_)) { fprintf(stderr, "missing %s\n", key); continue; }
        {
            int tiles = (int)(s_ / 5120);
            int tcols = H / 256;
            int orows = 0, ocols = 0;
            float* base = dequant_i8_signed_to_float_ex(md + 8 + hsz + o_, tiles, H, &orows, &ocols);
            size_t rowE = (size_t)2 * nff;       // rows per expert
            std::vector<float> m((size_t)n_exp * rowE * H);
            if (base && orows == (int)(n_exp * rowE) && ocols == H) {
                for (int e = 0; e < n_exp; e++) {
                    const double* Bg = ad.Bg.data() + (size_t)e * rowE * r;
                    const double* Ag = ad.Ag.data() + (size_t)e * r * H;
                    for (int i = 0; i < (int)rowE; i++) {
                        for (int j = 0; j < H; j++) {
                            double d = 0;
                            for (int k = 0; k < r; k++) d += Bg[(size_t)i * r + k] * Ag[(size_t)k * H + j];
                            m[((size_t)e * rowE + i) * H + j] = base[((size_t)e * rowE + i) * H + j] + (float)d;
                        }
                    }
                }
                put_tensor(D + data0, o_, md + 8 + hsz + o_, m.data(), n_exp * (int)rowE, H);
                fprintf(stderr, "  MoE l=%d gate_up rewritten @%llu\n", l, (unsigned long long)o_);
                changed++;
            } else fprintf(stderr, "gu dequant mismatch l=%d (%d x %d)\n", l, orows, ocols);
            if (base) free(base);
        }
        // down_proj: rows n_exp*H, in nff
        snprintf(key, sizeof key, "model.layers.%d.mlp.experts.down_proj.weight", l);
        if (!get_offsets(js, jl, key, &o_, &s_)) { fprintf(stderr, "missing %s\n", key); continue; }
        {
            int tiles = (int)(s_ / 5120);
            int tcols = nff / 256;
            int orows = 0, ocols = 0;
            float* base = dequant_i8_signed_to_float_ex(md + 8 + hsz + o_, tiles, nff, &orows, &ocols);
            std::vector<float> m((size_t)n_exp * H * nff);
            if (base && orows == n_exp * H && ocols == nff) {
                for (int e = 0; e < n_exp; e++) {
                    const double* Bd = ad.Bd.data() + (size_t)e * H * r;
                    const double* Ad2 = ad.Ad.data() + (size_t)e * r * nff;
                    for (int i = 0; i < H; i++) {
                        for (int j = 0; j < nff; j++) {
                            double d = 0;
                            for (int k = 0; k < r; k++) d += Bd[(size_t)i * r + k] * Ad2[(size_t)k * nff + j];
                            m[((size_t)e * H + i) * nff + j] = base[((size_t)e * H + i) * nff + j] + (float)d;
                        }
                    }
                }
                put_tensor(D + data0, o_, md + 8 + hsz + o_, m.data(), n_exp * H, nff);
                fprintf(stderr, "  MoE l=%d down rewritten @%llu\n", l, (unsigned long long)o_);
                changed++;
            } else fprintf(stderr, "dn dequant mismatch l=%d (%d x %d)\n", l, orows, ocols);
            if (base) free(base);
        }
    }
    msync(D, ost.st_size, MS_SYNC);
    munmap(D, ost.st_size);
    munmap((void*)md, st.st_size);
    rename(outtmp.c_str(), argv[3]);
    fprintf(stderr, "done: %zu tensors rewritten -> %s\n", changed, argv[3]);
    return 0;
}
