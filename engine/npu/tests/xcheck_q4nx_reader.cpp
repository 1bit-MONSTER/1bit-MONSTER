// xcheck_q4nx_reader.cpp — does read_q4nx_raw (the fused int4 pack path) agree
// with dequant_i8_to_float_ex (the engine's ground-truth Q4NX dequant) on the
// SAME gate/up tensor? If they disagree, the fused pack reads the model with
// the wrong (symmetric/zaya) layout and corrupts the int4 weights (#1934).
//
// Build: g++ -std=c++23 -O2 -I engine/npu/src -I engine/npu/generators \
//        engine/npu/tests/xcheck_q4nx_reader.cpp engine/npu/src/dequant_q4nx.cpp \
//        -o /tmp/xcheck
// Run:   /tmp/xcheck models/FastFlowLM-Qwen3-0.6B-NPU2/model.q4nx [layer]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "q4nx_raw.h"

extern "C" float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows,
                                         int in_features, int* out_rows, int* out_cols);

static uint64_t get_off(const char* js, size_t jl, const std::string& name) {
    std::string k = "\"" + name + "\":";
    auto q = strstr(js, k.c_str());
    if (!q || q - js + k.size() >= (long)jl) return ~0ull;
    auto o = strstr(q + k.size(), "data_offsets");
    if (!o) return ~0ull;
    auto b = strchr(o, '['); if (!b) return ~0ull;
    return (uint64_t)strtoull(b + 1, nullptr, 10);
}

static double pear(const std::vector<double>& x, const std::vector<double>& y) {
    int m = (int)x.size(); if (!m) return 0;
    double sx=0,sy=0,sx2=0,sy2=0,sxy=0;
    for (int k=0;k<m;k++){sx+=x[k];sy+=y[k];sx2+=x[k]*x[k];sy2+=y[k]*y[k];sxy+=x[k]*y[k];}
    double mx=sx/m,my=sy/m,cxx=sx2/m-mx*mx,cyy=sy2/m-my*my,cxy=sxy/m-mx*my;
    return (cxx>0&&cyy>0)?cxy/std::sqrt(cxx*cyy):(cxx==0&&cyy==0)?1.0:0.0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.q4nx> [layer]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 0;
    const int H = 1024, IM = 3072;   // qwen3-0.6b

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { perror("mmap"); return 2; }
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;                  // data region base (matches engine i8p)
    const char* js = (const char*)(md + 8);

    const char* tensors[2] = {"gate_proj", "up_proj"};
    for (int ti = 0; ti < 2; ti++) {
        char key[256];
        snprintf(key, sizeof key, "model.layers.%d.mlp.%s.weight", L, tensors[ti]);
        uint64_t off = get_off(js, hsz, key);
        if (off != ~0ull) off += df;        // data_offsets are relative to data start
        fprintf(stderr, "\n=== %s (L%d) @ %llu ===\n", tensors[ti], L, (unsigned long long)off);
        if (off == ~0ull) { fprintf(stderr, "  no offset\n"); continue; }

        // The on-disk tensor is [i8_rows, 5120] tiles. i8_rows = (out/32)*(in/256).
        int i8_rows = (IM / 32) * (H / 256);   // 96*4 = 384
        // 1) fused pack path: read_q4nx_raw (signed nibbles, row-major scales)
        auto r = read_q4nx_raw(md, off, i8_rows, H);
        // 2) ground truth: dequant_i8_to_float_ex (asymmetric, group-major scales)
        int or_=0, oc=0;
        float* dq = dequant_i8_to_float_ex(md + off, i8_rows, H, &or_, &oc);
        fprintf(stderr, "  read_q4nx_raw rows=%d cols=%d ; dequant out_rows=%d out_cols=%d\n",
                r.rows, r.cols, or_, oc);

        // Compare over the populated region (dequant rows = (i8_rows/(H/256))*32... careful)
        int DQrows = or_, DQcols = oc;
        const int RC = H / 32;
        // read_q4nx_raw populates rows = (i8_rows/(H/256))*32 = 3072
        std::vector<double> xq, yq;  // reconstructed weights
        long long bad = 0, tot = 0; double mae = 0;
        int sample = 0;
        for (int row = 0; row < DQrows && row < r.rows; row++) {
            for (int col = 0; col < DQcols && col < H; col++) {
                // read_q4nx_raw reconstruction (signed nibble q, row-major scale)
                float sr = r.scl[(size_t)row * RC + col / 32];
                float zr = r.zp[(size_t)row * RC + col / 32];
                float wq = (float)r.q4[(size_t)row * H + col] * sr + zr;
                // dequant reconstruction
                float wd = dq[(size_t)row * DQcols + col];
                if (!std::isfinite(wq) || !std::isfinite(wd)) continue;
                double e = std::fabs(wq - wd);
                double scale = std::fabs(wd) > 1e-9 ? std::fabs(wd) : 1e-9;
                tot++; mae += e;
                if (e / scale > 0.01) bad++;
                if (sample < 6) {
                    fprintf(stderr, "    (r=%d c=%d) raw=%.6g deq=%.6g q4=%d scl=%.6g zp=%.6g dq=%.6g\n",
                            row, col, wq, wd, r.q4[(size_t)row*H+col], sr, zr, wd);
                    sample++;
                }
            }
        }
        fprintf(stderr, "  compared %lld (mae=%.6g, rel>1%% bad=%lld/%lld)\n", tot, mae/(tot?tot:1), bad, tot);
        if (tot > 0 && sample == 0) {
            // compute global pearson on a subsample
            std::vector<double> A, B;
            for (int row = 0; row < DQrows && row < r.rows; row += 7)
                for (int col = 0; col < DQcols && col < H; col += 7) {
                    float sr = r.scl[(size_t)row * RC + col / 32];
                    float zr = r.zp[(size_t)row * RC + col / 32];
                    float wq = (float)r.q4[(size_t)row * H + col] * sr + zr;
                    float wd = dq[(size_t)row * DQcols + col];
                    if (std::isfinite(wq) && std::isfinite(wd)) { A.push_back(wq); B.push_back(wd); }
                }
            fprintf(stderr, "  GLOBAL corr(read_q4nx_raw, dequant)=%.6f over %zu\n", pear(A, B), A.size());
        }
        free(dq);
    }
    munmap(md, st.st_size);
    return 0;
}
