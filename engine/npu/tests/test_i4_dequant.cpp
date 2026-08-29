// test_i4_dequant.cpp — kernel-round gate for the ws09 int4 GU dequant
// (issue #1769). Consumes the packed v2 per-tile 4864-B chunks (gu_i4_pack.h:
// [nibbles 4096][s 512 (2 groups x 128 cols bf16)][S_col 256 (128 cols bf16)])
// exactly as matmul_i8_i32_i4 will and verifies the dequant reproduces
// B_shadow byte-identically.
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

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

// The kernel's per-(8,8)-chunk dequant (matmul_i8_i32_i4, byte-pinned): within
// a chunk all 8 rows share one K-group, so the ratio collapses to 8 per-column
// values read as bf16 FROM THE TILE:
//   ratio[c] = (bf16_to_f32(s[group][col]) * 0.0625f) / bf16_to_f32(S_col[col])
//   B''[r][c] = sat8(round((q4<<4)[r][c] * ratio[c]))
static inline int8_t i4d_sat8(int x) { return (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x); }

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

    // Kernel consumption: per (64,128) tile, read the ONE linear 8192-B tile
    // exactly as matmul_i8_i32_i4 does — nibbles [0,4096), ratioQ22 int32
    // [4096,5120) — run the dequant, compare with B_shadow.
    //
    // v65/v66 contract (gu_i4_pack.h): the on-chip dequant reads ONLY the
    // ratioQ22 int32 at [4096 + group*512 + col*4] (group = k/32, col within
    // the 128-wide tile). The old bf16 s/S_col bytes at [4096..4864) were
    // REMOVED in v65 (they overlapped this region), and B_shadow is the
    // kernel's EXACT B'': sat8(round-half-away(q4*rq / 2^18)) — NOT the old
    // float ratio path. This test was stale against the v65 pack (read the
    // old bf16 layout -> ~12% matches); updated to the v66 contract.
    const size_t N = 2 * (size_t)n_ff;
    const int n_tiles_k = H / 64, n_tiles_n = (int)(N / 128);
    int neq = 0, ntot = H * (int)N;
    for (int ki = 0; ki < n_tiles_k; ki++)
        for (int nt = 0; nt < n_tiles_n; nt++) {
            const size_t tbase = ((size_t)ki * n_tiles_n + nt) * GuI4Pack::TILE_TOTAL;
            const uint8_t* tile = pack.tiles.data() + tbase;
            int8_t bpp[64 * 128];
            // per (8,8) chunk at (k-step i0, col-tile i1): nibbles at
            // i0*512+i1*32 (32 B), ratioQ22 at 4096+(i0*8+i2)/32*512+(i1*8+i3)*4
            // — one int32 per (col-in-tile), group = (k)/32 within the tile.
            for (int i0 = 0; i0 < 8; i0++)
                for (int i1 = 0; i1 < 16; i1++) {
                    const uint8_t* nib = tile + i0 * 512 + i1 * 32;
                    // ratioQ22 row for this k-step: group = (i0*8+i2)/32, but
                    // within an (8,8) chunk all 8 rows share one k-group
                    // (i0*8+i2 in [i0*8, i0*8+8) -> group = i0/4).
                    const int32_t* rq = (const int32_t*)(tile + 4096 + (i0 / 4) * 512 + i1 * 32);
                    // unpack 32 nibble bytes -> 64 q4 (sign-extended), then
                    // ONE int32 multiply by the ratioQ22 (kernel arithmetic:
                    // sat8(round-half-away(q4*rq / 2^18)) — identical to
                    // matmul_i8_i32_i4 and the v66 B_shadow)
                    for (int i2 = 0; i2 < 8; i2++)
                        for (int i3 = 0; i3 < 8; i3++) {
                            uint8_t b = nib[i2 * 4 + i3 / 2];
                            int q4 = (i3 % 2 == 0) ? (int)(b & 0x0F) : (int)((b >> 4) & 0x0F);
                            if (q4 >= 8) q4 -= 16;
                            int x = q4 * rq[i3];
                            int ax = x < 0 ? -x : x;
                            int rr = (ax + (1 << 17)) >> 18;   // round-half-away
                            rr = x < 0 ? -rr : rr;
                            bpp[i0 * 1024 + i1 * 64 + i2 * 8 + i3] =
                                i4d_sat8(rr);
                        }
                }
            // compare with B_shadow (bpp is the microtiled mmul layout)
            for (int i = 0; i < 64; i++)
                for (int j = 0; j < 128; j++) {
                    int i0 = i / 8, i2 = i % 8, i1 = j / 8, i3 = j % 8;
                    int8_t bv = bpp[i0 * 1024 + i1 * 64 + i2 * 8 + i3];
                    if (bv == pack.B_shadow[(size_t)(ki * 64 + i) * N + nt * 128 + j])
                        neq++;
                }
        }
    fprintf(stderr, "  [kernel-dequant] B'' byte-identity vs B_shadow (v66 ratioQ22 contract): %d/%d exact\n", neq, ntot);
    if (neq != ntot) { fprintf(stderr, "FAIL\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
