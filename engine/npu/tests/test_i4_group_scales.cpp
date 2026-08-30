// test_i4_group_scales.cpp — CPU gate for the #1934 per-group-scale pack.
//
// Verifies pack_gu_fused_i4_group_scales() emits the EXACT per-(row,
// 32-col-group) bf16 scales from a real Q4NX GU tensor — the host-side
// contract the restructured int4 kernel (C1 carrying per-group scales) will
// consume. The current production path collapses these into a per-column
// ratioQ22 (K-uniform, corr cap ~0.972); this grid is the finer data.
//
// Usage: /tmp/test_i4_group_scales zaya1-8b.q4nx [layer] [expert]
//   (layer defaults to 1 = first MoE layer; expert defaults to 0)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "q4nx_raw.h"
#include "gu_i4_pack.h"

static int rc = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); rc = 1; } else { fprintf(stderr, "ok:   "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static int get_top_int(const char* js, size_t jl, const char* key) {
    std::string k = std::string("\"") + key + "\":";
    auto p = strstr(js, k.c_str());
    if (!p || p - js + k.size() >= (long)jl) return 0;
    return atoi(p + k.size());
}
static bool get_offsets(const char* js, size_t jl, const char* key,
                        uint64_t* off, uint64_t* size) {
    std::string k = std::string("\"") + key + "\":";
    auto q = strstr(js, k.c_str());
    if (!q) return false;
    auto b = strchr(q + k.size(), '[');
    if (!b) return false;
    auto c = strchr(b, ',');
    *off  = (uint64_t)strtoull(b + 1, nullptr, 10);
    if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.q4nx> [layer] [expert]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 1;
    const int E = argc > 3 ? atoi(argv[3]) : 0;
    const int H = 2048, n_ff = 2048;   // zaya1-8b

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { perror("mmap"); return 2; }
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);

    char key[256];
    snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", L);
    uint64_t off, size;
    if (!get_offsets(js, hsz, key, &off, &size)) { fprintf(stderr, "no GU tensor at layer %d\n", L); return 2; }
    int NC = get_top_int(js, hsz, "num_hidden_layers");
    int n_exp = get_top_int(js, hsz, "num_experts");
    int gu_i8_rows = (n_exp * 2 * n_ff / 32) * (H / 256);
    auto raw_all = read_q4nx_raw(md, off, gu_i8_rows, H);
    munmap(md, st.st_size);

    // slice to expert E (rows [E*2n_ff, (E+1)*2n_ff): gate [0,n_ff), up [n_ff,2n_ff))
    RawQ4Tensor raw_gu;
    raw_gu.rows = 2 * n_ff; raw_gu.cols = H;
    raw_gu.q4.assign((size_t)raw_gu.rows * H, 0);
    raw_gu.scl.assign((size_t)raw_gu.rows * (H / 32), 0.0f);
    raw_gu.zp.assign((size_t)raw_gu.rows * (H / 32), 0.0f);
    const size_t gbase = (size_t)E * 2 * n_ff;
    for (int r = 0; r < 2 * n_ff; r++) {
        memcpy(&raw_gu.q4[(size_t)r * H], &raw_all.q4[(gbase + r) * H], sizeof(int8_t) * H);
        memcpy(&raw_gu.scl[(size_t)r * (H / 32)], &raw_all.scl[(gbase + r) * (H / 32)],
               sizeof(float) * (H / 32));
    }
    fprintf(stderr, "GU tensor: expert %d, rows=%d cols=%d (NC=%d n_exp=%d)\n",
            E, raw_gu.rows, raw_gu.cols, NC, n_exp);

    GuI4Pack p;
    pack_gu_fused_i4_group_scales(raw_gu, 0, H, n_ff, p);

    const int RC = H / 32;
    CHECK((int)p.scl_g_bf16.size() == 2 * n_ff * RC,
          "grid size %zu == 2*n_ff*RC %d", p.scl_g_bf16.size(), 2 * n_ff * RC);

    // The emitted bf16 must equal the raw tensor's scale bits exactly.
    // Note: the grid is indexed by the SLICED expert's local row (r-gbase),
    // matching the gate/up interleaved layout the kernel will consume.
    size_t bad = 0, checked = 0;
    for (size_t r = 0; r < 2 * (size_t)n_ff; r++) {
        for (int g = 0; g < RC; g++) {
            uint16_t want = f32_to_bf16_impl(raw_gu.scl[r * RC + g]);
            uint16_t got  = p.scl_g_bf16[r * RC + g];
            checked++;
            if (got != want) { if (bad < 5) fprintf(stderr, "  row %zu grp %d: got %04x want %04x\n", r, g, got, want); bad++; }
        }
    }
    CHECK(bad == 0, "per-group scale grid byte-exact vs raw Q4NX (%zu checked, %zu bad)", checked, bad);

    // Cross-check: reconstruct a sampled weight from the grid's bf16 scale
    // bits and confirm it matches the raw float-scale reconstruction exactly.
    // (Some zaya Q4NX rows carry NaN scales — padding/unused experts; those
    // are identical in both reconstructions, so NaN==NaN counts as equal.)
    size_t wbad = 0, wchecked = 0, wnan = 0;
    for (size_t r = 0; r < 2 * (size_t)n_ff; r += 11) {
        for (int i = 0; i < H; i += 17) {
            uint32_t sbits = (uint32_t)p.scl_g_bf16[r * RC + i / 32] << 16;
            float srow; memcpy(&srow, &sbits, 4);
            float w_grid = (float)raw_gu.q4[r * H + i] * srow + raw_gu.zp[r * RC + i / 32];
            uint16_t want_bits = f32_to_bf16_impl(raw_gu.scl[r * RC + i / 32]);
            uint32_t wbits = (uint32_t)want_bits << 16;
            float sraw; memcpy(&sraw, &wbits, 4);
            float w_raw = (float)raw_gu.q4[r * H + i] * sraw + raw_gu.zp[r * RC + i / 32];
            bool gnan = (w_grid != w_grid), rnan = (w_raw != w_raw);
            if (gnan && rnan) { wnan++; continue; }          // matching NaN padding
            if (w_grid != w_raw) { if (wbad < 3) fprintf(stderr, "  row %zu i %d: grid %.8g raw %.8g\n", r, i, w_grid, w_raw); wbad++; }
            wchecked++;
        }
    }
    CHECK(wbad == 0, "grid-derived weights == raw bf16-derived weights (%zu finite + %zu NaN-pad matched)", wchecked, wnan);

    fprintf(stderr, "\n%s\n", rc ? "=== I4 GROUP-SCALES GATE FAILED ===" : "=== I4 GROUP-SCALES GATE PASSED (per-group scale grid byte-exact) ===");
    return rc;
}
