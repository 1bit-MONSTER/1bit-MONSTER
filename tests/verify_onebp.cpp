/** verify_onebp.cpp — round-trip verification for gguf_to_onebp output.
 *
 * Usage: verify_onebp <source.gguf> <converted.1bp> [--tolerance-scale <s>]
 *
 * 1. Loads the .1bp with OnebpModel (the production loader's parse: header,
 *    index, v2 per-tensor quant, v4 dedup aliases, offset bounds).
 * 2. Dequantizes every tensor back to f32 using the exact tile math the
 *    converter writes (F16 / Q4NX / TQ2 / TQ2NZ / TQ2NZ_E4M3 / TQ2BS / TQ1 /
 *    Q4_ROCMFP4 / Q4_ROCMFP4_FAST).
 * 3. Compares against gguf_reader's f32 view of the source GGUF.
 * 4. Prints per-tensor max abs error and a global verdict.
 *
 * Exit code 0 = every tensor round-tripped within tolerance.
 */
#include "onebp_format.h"
#include "onebp_loader.h"
#include "gguf_reader.h"
#include "block_scaled_ternary.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cinttypes>

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t out;
    if (e == 0) {
        if (m == 0) out = s;
        else { // subnormal
            int exp = -14;
            uint32_t mm = m;
            while (!(mm & 0x400)) { mm <<= 1; exp--; }
            mm &= 0x3ff;
            out = s | ((uint32_t)(exp + 127) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        out = s | 0x7f800000 | (m << 13);
    } else {
        out = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; memcpy(&f, &out, 4); return f;
}
static inline float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float f; memcpy(&f, &b, 4); return f;
}

// ── Codebook10 (ROCmFP4) nibble decode — mirror of gguf_reader.cpp ──
static inline int8_t cb10(uint8_t q) {
    uint8_t mag3 = q & 0x07;
    int mag = mag3 <= 4 ? mag3 : 2 * mag3 - 4;
    return (q & 0x08) ? (int8_t)-mag : (int8_t)mag;
}
static inline float ue4m3(uint8_t e) {
    if (e > 0x7e) return 0.0f;
    uint32_t exp = e >> 3, mant = e & 7;
    return exp == 0 ? (float)mant * 0.0009765625f
                    : (8.0f + mant) * ldexpf(1.0f, (int)exp - 11);
}

// Dequantize one tiled tensor region [R rows x C cols] starting at `base`
// into out[] (out must be R*C). Mirrors the converter's write layout.
static void dequant_tiled(const uint8_t* base, int R, int C, OnebpQuant q,
                          std::vector<float>& out) {
    const int tr = 32, tc = 256, gs = 32;
    out.assign((size_t)R * C, 0.0f);
    int ntr = (R + tr - 1) / tr, ntc = (C + tc - 1) / tc;
    size_t base_idx = 0;
    for (int r = 0; r < ntr; r++) {
        for (int c = 0; c < ntc; c++) {
            const int r0 = r * tr, c0 = c * tc;
            const int rh = (R - r0) < tr ? (R - r0) : tr;
            const int cw = (C - c0) < tc ? (C - c0) : tc;
            size_t tile_bytes = onebp_tiled_size(tr, tc, tr, tc, gs, q);
            const uint8_t* td = base + base_idx;
            base_idx += tile_bytes;
            float* tile_out = out.data() + (size_t)r0 * C + c0;

            if (q == ONEBP_F16) {
                const uint16_t* src = (const uint16_t*)td;
                for (int rr = 0; rr < rh; rr++)
                    for (int cc = 0; cc < cw; cc++)
                        tile_out[(size_t)rr * C + cc] = f16_to_f32(src[(size_t)rr * tc + cc]);
            } else if (q == ONEBP_Q4NX) {
                const uint16_t* sc = (const uint16_t*)td;
                const uint16_t* zp = (const uint16_t*)(td + (size_t)tr * (tc / gs) * 2);
                const uint8_t*  qd = td + (size_t)tr * (tc / gs) * 4;
                for (int rr = 0; rr < rh; rr++) {
                    for (int g = 0; g < tc / gs; g++) {
                        float s = bf16_to_f32(sc[(size_t)rr * (tc / gs) + g]);
                        float mn = bf16_to_f32(zp[(size_t)rr * (tc / gs) + g]);
                        for (int i = 0; i < gs; i++) {
                            int cc = g * gs + i;
                            if (cc >= cw) break;
                            int local = rr * tc + cc;
                            uint8_t byte = qd[local / 2];
                            uint8_t code = (local & 1) ? (byte >> 4) : (byte & 0x0f);
                            tile_out[(size_t)rr * C + cc] = mn + (float)code * s;
                        }
                    }
                }
            } else if (q == ONEBP_TQ2) {
                size_t sb = (size_t)tr * (tc / gs) * 2;
                const uint16_t* sc = (const uint16_t*)td;
                const uint8_t* qd = td + sb;
                for (int rr = 0; rr < rh; rr++) {
                    for (int g = 0; g < tc / gs; g++) {
                        float s = bf16_to_f32(sc[(size_t)rr * (tc / gs) + g]);
                        for (int i = 0; i < gs; i++) {
                            int cc = g * gs + i;
                            if (cc >= cw) break;
                            size_t pos = (size_t)rr * tc + cc;
                            uint8_t code = (qd[pos / 4] >> ((pos & 3) * 2)) & 3;
                            tile_out[(size_t)rr * C + cc] = (float)((int)code - 1) * s;
                        }
                    }
                }
            } else if (q == ONEBP_TQ2NZ || q == ONEBP_TQ2NZ_E4M3) {
                bool e4m3 = (q == ONEBP_TQ2NZ_E4M3);
                size_t sb = (size_t)tr * (tc / gs) * (e4m3 ? 1 : 2);
                const uint8_t* qd = td + sb;
                for (int rr = 0; rr < rh; rr++) {
                    for (int g = 0; g < tc / gs; g++) {
                        float s;
                        if (e4m3) s = ue4m3(td[(size_t)rr * (tc / gs) + g]);
                        else s = bf16_to_f32(((const uint16_t*)td)[(size_t)rr * (tc / gs) + g]);
                        for (int i = 0; i < gs; i++) {
                            int cc = g * gs + i;
                            if (cc >= cw) break;
                            size_t pos = (size_t)rr * tc + cc;
                            uint8_t code = (qd[pos / 4] >> ((pos & 3) * 2)) & 3;
                            float cv = code == 0 ? -4.0f : code == 1 ? -1.0f : code == 2 ? 1.0f : 4.0f;
                            tile_out[(size_t)rr * C + cc] = cv * s;
                        }
                    }
                }
            } else if (q == ONEBP_TQ1) {
                const int tq1_grps = (tc + 4) / 5;
                size_t sb = (size_t)tr * tq1_grps * 2;
                const uint16_t* sc = (const uint16_t*)td;
                const uint8_t* qd = td + sb;
                static const int pow3[5] = {1, 3, 9, 27, 81};
                for (int rr = 0; rr < rh; rr++) {
                    for (int g = 0; g < tq1_grps; g++) {
                        float s = bf16_to_f32(sc[(size_t)rr * tq1_grps + g]);
                        uint8_t packed = qd[(size_t)rr * tq1_grps + g];
                        for (int i = 0; i < 5; i++) {
                            int cc = g * 5 + i;
                            if (cc >= cw) break;
                            uint8_t code = (packed / pow3[i]) % 3;
                            tile_out[(size_t)rr * C + cc] = (float)((int)code - 1) * s;
                        }
                    }
                }
            } else if (q == ONEBP_TQ2BS) {
                for (int rr = 0; rr < rh; rr++) {
                    const uint8_t* blocks = td + (size_t)rr * (tc / 16) * BST_BLOCK_BYTES;
                    for (int b = 0; b < tc / 16; b++) {
                        for (int i = 0; i < 16; i++) {
                            int cc = b * 16 + i;
                            if (cc >= cw) break;
                            tile_out[(size_t)rr * C + cc] =
                                block_scaled_ternary_dequant(blocks + b * BST_BLOCK_BYTES, i);
                        }
                    }
                }
            } else if (q == ONEBP_Q4_ROCMFP4 || q == ONEBP_Q4_ROCMFP4_FAST) {
                const bool fast = (q == ONEBP_Q4_ROCMFP4_FAST);
                const int block_bytes = fast ? 17 : 18;
                const int nb = (tc + 31) / 32;
                for (int rr = 0; rr < rh; rr++) {
                    const uint8_t* row = td + (size_t)rr * nb * block_bytes;
                    for (int b = 0; b < nb; b++) {
                        const uint8_t* blk = row + (size_t)b * block_bytes;
                        float d0 = ue4m3(blk[16]);
                        float d1 = fast ? d0 : ue4m3(blk[17]);
                        for (int i = 0; i < 32; i++) {
                            int cc = b * 32 + i;
                            if (cc >= cw) break;
                            int j = i & 15;
                            uint8_t nib = (i < 16) ? (blk[j] & 0x0f) : (blk[j] >> 4);
                            tile_out[(size_t)rr * C + cc] = (float)cb10(nib) * ((i < 16) ? d0 : d1);
                        }
                    }
                }
            } else {
                fprintf(stderr, "  [verify] unsupported quant %u — skipping tensor\n", (unsigned)q);
                return;
            }
        }
    }
}

static void maybe_reorder_zaya_cca(GgufReader& reader, const std::string& name, std::vector<float>& fw) {
    if (name.find("cca_conv_grp.weight") == std::string::npos) return;
    const GgufTensorInfo* inf = reader.tensor_info(name);
    if (!inf || inf->shape.size() != 3 || inf->shape[0] != 2) return;
    const int gc = (int)inf->shape[1], qkv = (int)inf->shape[2];
    std::vector<float> re(fw.size());
    for (int oc = 0; oc < qkv; oc++)
        for (int j = 0; j < gc; j++)
            for (int t = 0; t < 2; t++)
                re[(size_t)t * gc * qkv + (size_t)j * qkv + oc] =
                    fw[(size_t)oc * gc * 2 + (size_t)j * 2 + t];
    fw.swap(re);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <source.gguf> <converted.1bp>\n", argv[0]);
        return 2;
    }

    GgufReader reader;
    if (!reader.open(argv[1])) { fprintf(stderr, "cannot open GGUF %s\n", argv[1]); return 2; }

    OnebpModel mdl;
    if (!mdl.load(argv[2])) { fprintf(stderr, "OnebpModel::load FAILED for %s\n", argv[2]); return 1; }

    printf("1BP header: magic=0x%08X version=%u arch=%u quant=%u H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d tensors=%u\n",
           mdl.header.magic, mdl.header.version, mdl.header.arch, mdl.header.quant,
           mdl.header.hidden_size, mdl.header.num_layers, mdl.header.num_attention_heads,
           mdl.header.num_kv_heads, mdl.header.head_dim, mdl.header.intermediate_size,
           mdl.header.vocab_size, mdl.header.tensor_count);
    printf("file_size=%zu data_section_offset=%llu\n", mdl.file_size,
           (unsigned long long)mdl.data_section_offset);

    double global_max_abs = 0.0, global_max_rel = 0.0;
    int bad_tensors = 0, checked = 0, aliased = 0;

    for (auto& t : mdl.tensors) {
        const GgufTensorInfo* inf = reader.tensor_info(t.name);
        if (!inf) {
            // Tensors the converter skips (unknown ndim / over the element
            // cap) are absent from the 1BP index entirely — skip quietly.
            printf("  [skip] %s (absent from GGUF — converter-filtered)\n", t.name.c_str());
            continue;
        }
        // Per-tensor quant is authoritative for v2+ files even when it is 0
        // (ONEBP_Q4NX == 0 — a routed Q4NX tensor must NOT fall back to the
        // header quant; the engine loader reads it the same way).
        uint32_t q = mdl.header.version >= 2 ? t.quant : mdl.header.quant;
        std::vector<float> src;
        if (!reader.get_tensor_f32(t.name, src)) {
            fprintf(stderr, "  [FAIL] get_tensor_f32 %s\n", t.name.c_str());
            bad_tensors++; continue;
        }
        maybe_reorder_zaya_cca(reader, t.name, src);

        if (t.ndim == 1) {
            // raw f32
            std::vector<float> got((size_t)t.bytes / 4);
            memcpy(got.data(), mdl.tensor_data(t), t.bytes);
            double mx = 0; size_t n = std::min(got.size(), src.size());
            for (size_t i = 0; i < n; i++) {
                double e = fabs((double)got[i] - (double)src[i]);
                if (e > mx) mx = e;
            }
            checked++;
            printf("  %-46s [1D] n=%zu maxabs=%.3e %s\n", t.name.c_str(), n, mx,
                   mx < 1e-4 ? "OK" : "** LARGE **");
            if (mx > 1e-3) bad_tensors++;
            continue;
        }

        // ndim 2 or 3: tiled
        int R, C, NE;
        if (t.ndim == 2) { R = (int)t.dims[0]; C = (int)t.dims[1]; NE = 1; }
        else             { NE = (int)t.dims[0]; R = (int)t.dims[1]; C = (int)t.dims[2]; }

        size_t expected = (size_t)R * C * NE;
        if (src.size() != expected) {
            // non-expert 3D stored flat as 2D: dims were flattened by converter
            fprintf(stderr, "  [note] %s: src=%zu vs R*C*NE=%zu — flat 3D; re-read as flat\n",
                    t.name.c_str(), src.size(), expected);
        }

        uint64_t per_expert = onebp_tiled_size(R, C, 32, 256, 32, (OnebpQuant)q);
        double tmax_abs = 0, tmax_rel = 0, src_maxabs = 0;
        size_t ncmp = 0;
        for (size_t i = 0; i < src.size(); i++) {
            double a = fabs((double)src[i]);
            if (a > src_maxabs) src_maxabs = a;
        }
        for (int ei = 0; ei < NE; ei++) {
            const uint8_t* base = mdl.tensor_data(t) + (size_t)ei * per_expert;
            std::vector<float> got;
            dequant_tiled(base, R, C, (OnebpQuant)q, got);
            for (size_t i = 0; i < got.size(); i++) {
                size_t si = (size_t)ei * (size_t)R * C + i;
                if (si >= src.size()) break;
                double a = (double)src[si], b = (double)got[i];
                double e = fabs(a - b);
                if (e > tmax_abs) tmax_abs = e;
                double rel = e / (fabs(a) > 1e-30 ? fabs(a) : 1e-30);
                if (rel > tmax_rel) tmax_rel = rel;
                ncmp++;
            }
        }
        // Lossless F16: error is pure f32->f16 rounding (<= 2^-10 * |v|).
        // Lossy quants: worst case is bounded by the group quantization
        // step — symmetric-ternary/codebook modes (TQ2/TQ2NZ/TQ1/TQ2BS) have
        // group scale = group max, so error <= max|v|/2; Q4NX/ROCmFP4 use
        // finer steps (<= ~range/30). Layout corruption (issue #1522) sits at
        // ~100%+ of the weight magnitudes, so these bounds are all tight
        // enough to catch it.
        double bound;
        if (q == ONEBP_F16) bound = 2e-3;
        else if (q == ONEBP_TQ2 || q == ONEBP_TQ2NZ || q == ONEBP_TQ2NZ_E4M3 ||
                 q == ONEBP_TQ1 || q == ONEBP_TQ2BS) bound = 0.6 * src_maxabs + 1e-6;
        else bound = 0.35 * src_maxabs + 1e-6;   // Q4NX / ROCmFP4 family
        bool pass = tmax_abs <= bound;
        if (tmax_abs > global_max_abs) global_max_abs = tmax_abs;
        if (tmax_rel > global_max_rel) global_max_rel = tmax_rel;
        checked++;
        if (t.bytes == 0) aliased++;
        const char* qn = q == ONEBP_F16 ? "F16" : q == ONEBP_Q4NX ? "Q4NX" :
                         q == ONEBP_TQ2 ? "TQ2" : q == ONEBP_TQ2NZ ? "TQ2NZ" :
                         q == ONEBP_TQ2NZ_E4M3 ? "TQ2NZ-E4M3" : q == ONEBP_TQ2BS ? "TQ2BS" :
                         q == ONEBP_TQ1 ? "TQ1" : q == ONEBP_Q4_ROCMFP4 ? "ROCmFP4" :
                         q == ONEBP_Q4_ROCMFP4_FAST ? "ROCmFP4-FAST" : "?";
        printf("  %-46s [%s] %dx%dx%d maxabs=%.3e (src max|v|=%.3f, bound=%.3e) %s\n", t.name.c_str(), qn,
               NE, R, C, tmax_abs, src_maxabs, bound, pass ? "OK" : "** CHECK **");
        if (!pass) bad_tensors++;
    }

    printf("\n=== verified %d tensors (%d aliased) — global maxabs=%.3e maxrel=%.3e ===\n",
           checked, aliased, global_max_abs, global_max_rel);
    if (bad_tensors) { printf("RESULT: FAIL (%d suspicious tensors)\n", bad_tensors); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
