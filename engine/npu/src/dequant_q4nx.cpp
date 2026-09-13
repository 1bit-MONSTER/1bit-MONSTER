/**
 * Q4NX INT4 dequantization — torch2aie chunk format.
 *
 * Each I8 row (5120 bytes) = ONE tile of [32 BF16 rows × 256 BF16 cols].
 * Tiles are arranged row-major in a grid covering the full weight matrix.
 *
 * Per I8 row (5120 bytes):
 *   [0..511]:   256 BF16 scales. For group g=0..7, row r=0..31: scales[g*32+r]
 *   [512..1023]: 256 BF16 zero_points. Same layout.
 *   [1024..5119]: 4096 bytes packed INT4:
 *     Lane 0 (rows 0-15): bytes 1024-3071
 *     Lane 1 (rows 16-31): bytes 3072-5119
 *     Within lane: for col 0..255, byte_idx 0..7: lane_base + col*8 + byte_idx
 *     nibbles: lo = row(byte_idx*2), hi = row(byte_idx*2+1)
 *
 * Tile grid: I8 rows row-major.
 *   n_tile_cols = in_features / 256 (usually 4 for hidden=1024)
 *   tile_row = I8_row / n_tile_cols
 *   tile_col = I8_row % n_tile_cols
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>

constexpr int TILE_ROWS = 32;
constexpr int TILE_COLS = 256;

static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

// The .q4nx tile buffer is not guaranteed 2-byte aligned (tensor data can
// start at an odd byte in the mapped file), so reading the bf16 scales/zps
// through a (const uint16_t*) cast is a misaligned load — UB flagged by
// UBSan and a hard fault on ARM/AIE targets (issue #1775 sanitizer run).
// Read the two bytes explicitly instead.
static inline uint16_t load_bf16_bytes(const uint8_t* p) {
    return (uint16_t)(p[0]) | ((uint16_t)(p[1]) << 8);
}

/**
 * Dequantize an I8 tensor (torch2aie Q4NX chunk format) to float.
 * Output: [out_rows, out_cols] row-major float array (caller must free).
 * out_rows = n_tile_rows * 32, out_cols = n_tile_cols * 256
 */

// ── The one int4 tile decoder, parameterised by the two axes it varies on ────
// Every Q4NX int4 bundle we have decodes with the SAME tile geometry (32x256,
// 5120-byte rows: 256 bf16 scales, 256 bf16 zero-points, 4096 packed nibbles)
// and differs on exactly two things:
//
//   scale_group_major : scale index is (group*32 + row) rather than (row*8 + group)
//   signed_nibbles    : nibble is two's complement (q<8 ? q : q-16) rather than the
//                       raw unsigned q with the zero-point carrying the offset
//
// Four combinations exist; three are in use. Keeping them as separate functions
// whose NAMES encode neither axis is what made "there is a third combination"
// invisible for as long as it was: LFM2-1.2B/2.6B need group-major scales AND
// signed nibbles, and with two named entry points the possibility did not present
// itself. A wrong pairing is silent — it still yields a plausible weight
// distribution — so the convention has to be measured, never inferred from a
// family name. engine/npu/tests/check_bundle_decoders.py gates on that
// measurement: group+unsigned for Qwen3 / Gemma3-4B / Llama-3.2 / Phi4-mini /
// Qwen3-VL, group+signed for LFM2.
extern "C" float* dequant_q4nx_i8_ex(const uint8_t* data, int i8_rows, int in_features,
                                     int scale_group_major, int signed_nibbles,
                                     int* out_rows, int* out_cols) {
    int n_tile_cols = in_features / TILE_COLS;
    int n_tile_rows = i8_rows / n_tile_cols;
    *out_rows = n_tile_rows * TILE_ROWS;
    *out_cols = n_tile_cols * TILE_COLS;

    float* out = static_cast<float*>(std::calloc((size_t)(*out_rows) * (*out_cols), sizeof(float)));
    if (!out) return nullptr;

    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = data + ir * 5120;
        int tile_row = ir / n_tile_cols;
        int tile_col = ir % n_tile_cols;
        const uint8_t* scales = rd;
        const uint8_t* zeros  = rd + 512;
        const uint8_t* packed = rd + 1024;
        for (int lr = 0; lr < TILE_ROWS; lr++) {
            int lane = lr / 16;
            int lane_row = lr % 16;
            int byte_idx = lane_row / 2;
            int nibble_sel = lr % 2;
            const uint8_t* lane_data = packed + lane * (TILE_COLS * 8);
            for (int col = 0; col < TILE_COLS; col++) {
                int group = col / 32;
                int si = scale_group_major ? (group * 32 + lr) : (lr * 8 + group);
                float scale = bf16_to_float(load_bf16_bytes(scales + si * 2));
                float zp = bf16_to_float(load_bf16_bytes(zeros + si * 2));
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zp) || std::fabs(zp) > 100.0f) zp = 0.0f;
                uint8_t byte_val = lane_data[col * 8 + byte_idx];
                int q = (nibble_sel == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
                int v = signed_nibbles ? (int)(int8_t)(q < 8 ? q : q - 16) : (int)q;
                out[(tile_row * TILE_ROWS + lr) * (*out_cols) +
                    (tile_col * TILE_COLS + col)] = (float)v * scale + zp;
            }
        }
    }
    return out;
}

// Which nibble convention a model's bundles use, by tag or directory name.
// Everything measured so far is unsigned except LFM2. Prefer probing where a
// probe is possible (check_bundle_decoders.py); this exists so the engine can
// select without a probe on every load.
extern "C" int q4nx_i8_signed_nibbles_for_tag(const char* tag) {
    if (!tag) return 0;
    for (const char* p = tag; *p; p++) {
        if (std::tolower((unsigned char)p[0]) == 'l' &&
            std::tolower((unsigned char)p[1]) == 'f' &&
            std::tolower((unsigned char)p[2]) == 'm' &&
            std::tolower((unsigned char)p[3]) == '2') return 1;
        if (!p[1] || !p[2] || !p[3]) break;
    }
    return 0;
}

// Forward declaration for the wrapper
extern "C" float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols);

extern "C" float* dequant_i8_to_float(const uint8_t* data, int i8_rows,
                           int* out_rows, int* out_cols) {
    return dequant_i8_to_float_ex(data, i8_rows, 1024, out_rows, out_cols);
}

/**
 * Extended version with explicit in_features (hidden_dim).
 * For Q4NX format: n_tile_cols = in_features / TILE_COLS.
 */
extern "C" float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols) {
    // group-major scales, unsigned nibbles
    return dequant_q4nx_i8_ex(data, i8_rows, in_features, /*scale_group_major=*/1,
                              /*signed_nibbles=*/0, out_rows, out_cols);
}

// ── Signed Q4NX int4 dequant (Zaya): value = (q - 8) * scale + min ──
// The Zaya converter (zaya.py) packs int4 SYMMETRICALLY: nibble q in [0,15]
// maps to [-8,7], mins are all 0.0, scales ~0.005-0.01. The unsigned
// dequant_i8_to_float_ex (issue #1268, Qwen3 asymmetric) would produce an
// all-positive weight shift and explode activations. Verified on zaya1-8b.q4nx:
// signed -> mean ~ -0.007, range [-0.086, 0.076] (symmetric, correct).
extern "C" float* dequant_i8_signed_to_float_ex(const uint8_t* data, int i8_rows,
                              int in_features, int* out_rows, int* out_cols) {
    // row-major scales, signed nibbles  (zaya1-8b.q4nx, our own converter)
    return dequant_q4nx_i8_ex(data, i8_rows, in_features, /*scale_group_major=*/0,
                              /*signed_nibbles=*/1, out_rows, out_cols);
}

// ── Q8_0 dequant (8704 bytes/row, used by Qwen3.6 attention projections) ──
// Row layout: [0..511]: 256 BF16 scales, [512..8703]: 8192 signed INT8 values.
// No zero-points. Values are signed: byte 0x00→0, 0x01→1, ..., 0x80→-128.
extern "C" float* dequant_q8_0_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols) {
    constexpr int Q8_0_ROW_BYTES = 8704;
    int n_tile_cols = in_features / TILE_COLS;
    int n_tile_rows = i8_rows / n_tile_cols;
    *out_rows = n_tile_rows * TILE_ROWS;
    *out_cols = n_tile_cols * TILE_COLS;

    float* out = static_cast<float*>(std::calloc((*out_rows) * (*out_cols), sizeof(float)));
    if (!out) return nullptr;

    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = data + ir * Q8_0_ROW_BYTES;
        int tile_row = ir / n_tile_cols;
        int tile_col = ir % n_tile_cols;

        const uint8_t* scales = rd;
        const int8_t*   values = (const int8_t*)(rd + 512);  // signed INT8

        for (int lr = 0; lr < TILE_ROWS; lr++) {
            for (int col = 0; col < TILE_COLS; col++) {
                int group = col / 32;
                float scale = bf16_to_float(load_bf16_bytes(scales + (group * 32 + lr) * 2));
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;

                // Signed INT8: row-major layout
                int8_t val = values[lr * TILE_COLS + col];

                out[(tile_row * TILE_ROWS + lr) * (*out_cols) +
                    (tile_col * TILE_COLS + col)] = (float)val * scale;
            }
        }
    }
    return out;
}

// ── Group-major scales + SIGNED int4 (LFM2 NPU2 bundles) ────────────────────
// The two decoders above each get exactly one of the two axes right:
//   dequant_i8_to_float_ex        : scales group-major (group*32+row), UNSIGNED
//   dequant_i8_signed_to_float_ex : scales row-major   (row*8+group),   SIGNED
// LFM2-1.2B-NPU2 is a third combination: group-major scales AND signed nibbles.
// Neither existing function decodes it, and the failure is silent — the wrong
// pairings still yield a plausible weight distribution.
//
// Measured against ground truth (both models have tie_word_embeddings=true, so
// lm_head must equal embed_tokens; correlation of the decoded lm_head with the
// BF16 embedding rows, first 64 rows, fresh conversion of the raw tiles):
//
//   convention              Qwen3-0.6B   LFM2-1.2B
//   scales=group unsigned      +0.9973      -0.4112
//   scales=group signed        -0.4442      +0.9915   <- this function
//   scales=row   unsigned      +0.9016      -0.0215
//   scales=row   signed        -0.4656      +0.0300
//
// So: Qwen3 needs the unsigned function, LFM2 needs this one. Callers that do
// not know which family they hold must probe (the tie test above) rather than
// assume — see engine/npu/tools/lfm2_cpu_runner.cpp.
extern "C" float* dequant_i8_group_signed_to_float_ex(const uint8_t* data, int i8_rows,
                              int in_features, int* out_rows, int* out_cols) {
    // group-major scales, signed nibbles  (LFM2's NPU2 bundles)
    return dequant_q4nx_i8_ex(data, i8_rows, in_features, /*scale_group_major=*/1,
                              /*signed_nibbles=*/1, out_rows, out_cols);
}
