// test_i4_asym_reader.cpp — CPU gate for the #1934 asymmetric Qwen3 raw reader.
//
// `read_q4nx_raw` (symmetric/zaya layout) MISreads the asymmetric Qwen3 FLM
// model: it signs the unsigned nibbles and transposes the row-major scales vs
// the group-major layout dequant_i8_to_float_ex() reads. This gate proves the
// NEW read_q4nx_raw_asym() reconstructs the EXACT float weights that the
// engine's ground-truth dequant produces, and quantifies the old reader's
// corruption (the #1934 fused-pack weight bug being fixed).
//
// Build: g++ -std=c++23 -O2 -I engine/npu/src -I engine/npu/generators \
//        engine/npu/tests/test_i4_asym_reader.cpp engine/npu/src/dequant_q4nx.cpp \
//        -o /tmp/test_i4_asym_reader
// Run:   /tmp/test_i4_asym_reader models/FastFlowLM-Qwen3-0.6B-NPU2/model.q4nx [layer]
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

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.q4nx> [layer]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 0;
    const int H = 1024, IM = 3072;
    int rc = 0;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { perror("mmap"); return 2; }
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    const char* js = (const char*)(md + 8);

    const int i8_rows = (IM / 32) * (H / 256);   // 384
    const int RC = H / 32;

    const char* tensors[2] = {"gate_proj", "up_proj"};
    for (int ti = 0; ti < 2; ti++) {
        char key[256];
        snprintf(key, sizeof key, "model.layers.%d.mlp.%s.weight", L, tensors[ti]);
        uint64_t off = get_off(js, hsz, key) + df;
        fprintf(stderr, "\n=== %s (L%d) @ %llu ===\n", tensors[ti], L, (unsigned long long)off);
        auto as = read_q4nx_raw_asym(md, off, i8_rows, H);
        int or_=0, oc=0;
        float* dq = dequant_i8_to_float_ex(md + off, i8_rows, H, &or_, &oc);
        int DQcols = oc;

        long long bad = 0, tot = 0, badsign = 0; double mae = 0, maxe = 0;
        // also quantify the OLD (signed/row-major) reader's corruption
        auto oldr = read_q4nx_raw(md, off, i8_rows, H);
        long long oldbad = 0;
        for (int row = 0; row < or_ && row < as.rows; row++) {
            for (int col = 0; col < DQcols && col < H; col++) {
                float w_asym = (float)as.q4[(size_t)row * H + col] * as.scl[(size_t)row * RC + col / 32]
                               + as.zp[(size_t)row * RC + col / 32];
                float w_old = (float)oldr.q4[(size_t)row * H + col] * oldr.scl[(size_t)row * RC + col / 32]
                              + oldr.zp[(size_t)row * RC + col / 32];
                float w_dq = dq[(size_t)row * DQcols + col];
                if (!std::isfinite(w_asym) || !std::isfinite(w_dq)) continue;
                double e = std::fabs(w_asym - w_dq);
                double scale = std::fabs(w_dq) > 1e-6 ? std::fabs(w_dq) : 1e-6;
                tot++; mae += e; if (e > maxe) maxe = e;
                if (e / scale > 1e-3) bad++;
                if (std::fabs(w_old - w_dq) / scale > 0.01) oldbad++;
            }
        }
        fprintf(stderr, "  asym reader vs dequant: %lld/%lld exact-1e-3 (mae=%.6g max=%.6g)\n",
                tot - bad, tot, mae / (tot ? tot : 1), maxe);
        fprintf(stderr, "  OLD read_q4nx_raw vs dequant: %lld/%lld rel>1%% bad (the bug being fixed)\n",
                oldbad, tot);
        // Print a couple of mismatched samples for debugging
        if (bad) {
            int shown = 0;
            for (int row = 0; row < or_ && row < as.rows && shown < 3; row++)
                for (int col = 0; col < DQcols && col < H && shown < 3; col++) {
                    float w_asym = (float)as.q4[(size_t)row * H + col] * as.scl[(size_t)row * RC + col / 32]
                                   + as.zp[(size_t)row * RC + col / 32];
                    float w_dq = dq[(size_t)row * DQcols + col];
                    double e = std::fabs(w_asym - w_dq), scale = std::fabs(w_dq) > 1e-6 ? std::fabs(w_dq) : 1e-6;
                    if (e / scale > 1e-3) { fprintf(stderr, "    mismatch r=%d c=%d asym=%.6g dq=%.6g q4=%d scl=%.6g zp=%.6g\n", row, col, w_asym, w_dq, as.q4[(size_t)row*H+col], as.scl[(size_t)row*RC+col/32], as.zp[(size_t)row*RC+col/32]); shown++; }
                }
        }
        fprintf(stderr, "  %s\n", bad == 0 ? "ASYMMETRIC READER GATE PASSED (byte-exact vs ground-truth dequant)"
                                          : "ASYMMETRIC READER GATE HAD MISMATCHES");
        if (bad) rc = 1;
        free(dq);
    }
    munmap(md, st.st_size);
    fprintf(stderr, "\n%s\n", rc ? "=== I4 ASYM READER GATE FAILED ===" : "=== I4 ASYM READER GATE PASSED ===");
    return rc;
}
