// test_i4_dequant.cpp — kernel-round gate for the ws09 int4 GU dequant
// (issue #1769). Consumes the packed regions A/B/C exactly as the kernel
// will and verifies the dequant reproduces B_shadow byte-identically.
//
// Build (CPU only):
//   g++ -std=c++23 -O2 -I engine/npu/src -I engine/npu/generators \
//       engine/npu/tests/test_i4_dequant.cpp -o /tmp/test_i4_dequant
//   /tmp/test_i4_dequant /home/bcloud/ZAYA1-8B-Q4NX/zaya1-8b.q4nx [layer] [expert]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "q4nx_raw.h"
#include "gu_i4_pack.h"
#include "i4_dequant.h"

// ── manifest parsing (same pattern as zaya_decode.cpp) ──
static int get_top_int(const char* js, size_t jl, const char* field) {
    size_t fl = strlen(field);
    const char* p = js;
    while (p < js + jl) {
        const char* q = (const char*)memmem(p, jl - (p - js), field, fl);
        if (!q) return 0;
        if ((q == js || *(q - 1) == '"') && *(q + fl) == '"') {
            const char* colon = strchr(q + fl, ':');
            if (colon) { colon++; while (*colon == ' ') colon++; return atoi(colon); }
        }
        p = q + fl;
    }
    return 0;
}
static int get_offsets(const char* js, size_t jl, const char* key,
                       uint64_t* off, uint64_t* sz) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        const char* q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q - 1) == '"') && *(q + kl) == '"') {
            const char* o = strstr(q, "\"data_offsets\"");
            if (o) {
                const char* b = strchr(o, '[');
                if (b) {
                    *off = (uint64_t)strtoull(b + 1, nullptr, 10);
                    const char* c = strchr(b + 1, ',');
                    if (c) *sz = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *sz > 0;
                }
            }
        }
        p = q + kl;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s zaya1-8b.q4nx [layer] [expert]\n", argv[0]); return 1; }
    int L = argc > 2 ? atoi(argv[2]) : 1;
    const int E = argc > 3 ? atoi(argv[3]) : 0;
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t* D = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, D, 8);
    const char* js = (const char*)(D + 8);
    size_t jl = (size_t)hsz;
    const uint8_t* M = D + 8 + hsz;
    int H = get_top_int(js, jl, "hidden_size");
    int n_ff = get_top_int(js, jl, "intermediate_size");
    int n_exp = get_top_int(js, jl, "num_experts");
    fprintf(stderr, "H=%d n_ff=%d n_exp=%d L=%d E=%d\n", H, n_ff, n_exp, L, E);
    if (L % 2 == 0) L++;   // MoE layer
    char key[256];
    snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", L);
    uint64_t gu_off, gu_size;
    get_offsets(js, jl, key, &gu_off, &gu_size);
    int gu_i8_rows = (int)(gu_size / 5120);
    auto raw_all = read_q4nx_raw(M, gu_off, gu_i8_rows, H);
    auto pack = pack_gu_fused_i4(raw_all, E, H, n_ff);

    // Kernel consumption: for each tile, extract the nibbles + the 2
    // colgroup scale rows + S_col, run the dequant, compare with B_shadow.
    const size_t N = 2 * (size_t)n_ff;
    const int n_tiles_k = H / 64, n_tiles_n = (int)(N / 128);
    int neq = 0, ntot = H * (int)N;
    for (int ki = 0; ki < n_tiles_k; ki++)
        for (int nt = 0; nt < n_tiles_n; nt++) {
            // unpack the tile's nibbles (the kernel's vldb.unpack output)
            int8_t q4[64 * 128];
            size_t tbase = ((size_t)ki * n_tiles_n + nt) * GuI4Pack::TILE_BYTES;
            for (int s4 = 0; s4 < 64 * 64; s4++) {
                uint8_t b = pack.nibbles[tbase + s4];
                int lo = b & 0x0F, hi = (b >> 4) & 0x0F;
                // byte s4 = i0*512+i1*32+i2*4+i3/2; element (i2, i3): 2*s4 = ...
                // the flat unpacked position 2*s4 = the (i0,i1,i2,i3) row-major
                q4[2 * s4] = (int8_t)(lo >= 8 ? lo - 16 : lo);
                q4[2 * s4 + 1] = (int8_t)(hi >= 8 ? hi - 16 : hi);
            }
            // the tile's scale rows (region B): (i/32)*N + j for the tile
            float row_scl[2 * 128], scol_inv[128];
            for (int cg = 0; cg < 2; cg++)
                for (int j = 0; j < 128; j++) {
                    int i = ki * 64 + cg * 32;   // any row of the colgroup
                    uint16_t sb = pack.row_scales[(size_t)(i / 32) * N + nt * 128 + j];
                    uint32_t sbits = (uint32_t)sb << 16; float srow; memcpy(&srow, &sbits, 4);
                    row_scl[cg * 128 + j] = srow;
                }
            for (int j = 0; j < 128; j++)
                scol_inv[j] = pack.scol[nt * 128 + j];
            // dequant (the kernel's function)
            int8_t bpp[64 * 128];
            dequant_i4_b_ref(bpp, q4, row_scl, scol_inv);
            // compare with B_shadow (bpp is the microtiled mmul layout)
            for (int i = 0; i < 64; i++)
                for (int j = 0; j < 128; j++) {
                    int i0 = i / 8, i2 = i % 8, i1 = j / 8, i3 = j % 8;
                    int8_t bv = bpp[i0 * 1024 + i1 * 64 + i2 * 8 + i3];
                    if (bv == pack.B_shadow[(size_t)(ki * 64 + i) * N + nt * 128 + j])
                        neq++;
                }
        }
    fprintf(stderr, "  [kernel-dequant] B'' byte-identity vs B_shadow: %d/%d exact\n", neq, ntot);
    if (neq != ntot) { fprintf(stderr, "FAIL\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
