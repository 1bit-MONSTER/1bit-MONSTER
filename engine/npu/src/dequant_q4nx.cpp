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
#include <cmath>

constexpr int TILE_ROWS = 32;
constexpr int TILE_COLS = 256;

static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

/**
 * Dequantize an I8 tensor (torch2aie Q4NX chunk format) to float.
 * Output: [out_rows, out_cols] row-major float array (caller must free).
 * out_rows = n_tile_rows * 32, out_cols = n_tile_cols * 256
 */
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
    // Determine tile grid: rows first, columns second
    int n_tile_cols, n_tile_rows;

    n_tile_cols = in_features / TILE_COLS;
    n_tile_rows = i8_rows / n_tile_cols;

    *out_rows = n_tile_rows * TILE_ROWS;
    *out_cols = n_tile_cols * TILE_COLS;

    float* out = static_cast<float*>(std::calloc((*out_rows) * (*out_cols), sizeof(float)));
    if (!out) return nullptr;

    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = data + ir * 5120;
        int tile_row = ir / n_tile_cols;
        int tile_col = ir % n_tile_cols;

        const uint16_t* scales = (const uint16_t*)(rd);
        const uint16_t* zeros  = (const uint16_t*)(rd + 512);
        const uint8_t* packed  = rd + 1024;

        for (int lr = 0; lr < TILE_ROWS; lr++) {
            int lane = lr / 16;
            int lane_row = lr % 16;
            int byte_idx = lane_row / 2;
            int nibble_sel = lr % 2;

            const uint8_t* lane_data = packed + lane * (TILE_COLS * 8);

            for (int col = 0; col < TILE_COLS; col++) {
                int group = col / 32;
                float scale = bf16_to_float(scales[group * 32 + lr]);
                float zp = bf16_to_float(zeros[group * 32 + lr]);
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zp) || std::fabs(zp) > 100.0f) zp = 0.0f;

                uint8_t byte_val = lane_data[col * 8 + byte_idx];
                // UNSIGNED asymmetric Q4NX — the zero-point is carried in the
                // per-group `zeros` array; the old `val >= 8 -> val -= 16`
                // signed reinterpretation decoded the same bytes differently
                // from onebp_loader's round-trip-verified decoder and the
                // cpu_q4nx_loader (issue #1268).
                uint8_t val;
                if (nibble_sel == 0) val = (byte_val & 0x0F);
                else                 val = ((byte_val >> 4) & 0x0F);

                out[(tile_row * TILE_ROWS + lr) * (*out_cols) +
                    (tile_col * TILE_COLS + col)] = (float)val * scale + zp;
            }
        }
    }
    return out;
}

// ── Signed Q4NX int4 dequant (Zaya): value = (q - 8) * scale + min ──
// The Zaya converter (zaya.py) packs int4 SYMMETRICALLY: nibble q in [0,15]
// maps to [-8,7], mins are all 0.0, scales ~0.005-0.01. The unsigned
// dequant_i8_to_float_ex (issue #1268, Qwen3 asymmetric) would produce an
// all-positive weight shift and explode activations. Verified on zaya1-8b.q4nx:
// signed -> mean ~ -0.007, range [-0.086, 0.076] (symmetric, correct).
extern "C" float* dequant_i8_signed_to_float_ex(const uint8_t* data, int i8_rows,
                              int in_features, int* out_rows, int* out_cols) {
    int n_tile_cols = in_features / TILE_COLS;
    int n_tile_rows = i8_rows / n_tile_cols;
    *out_rows = n_tile_rows * TILE_ROWS;
    *out_cols = n_tile_cols * TILE_COLS;

    float* out = static_cast<float*>(std::calloc((*out_rows) * (*out_cols), sizeof(float)));
    if (!out) return nullptr;

    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = data + ir * 5120;
        int tile_row = ir / n_tile_cols;
        int tile_col = ir % n_tile_cols;
        const uint16_t* scales = (const uint16_t*)(rd);
        const uint16_t* zeros  = (const uint16_t*)(rd + 512);
        const uint8_t* packed  = rd + 1024;
        for (int lr = 0; lr < TILE_ROWS; lr++) {
            int lane = lr / 16;
            int lane_row = lr % 16;
            int byte_idx = lane_row / 2;
            int nibble_sel = lr % 2;
            const uint8_t* lane_data = packed + lane * (TILE_COLS * 8);
            for (int col = 0; col < TILE_COLS; col++) {
                int group = col / 32;
                // Zaya converter (convert_float32_bins_to_q4nx.py) stores scales
                // row-major: scales_flat[row*8 + group] (NOT group-major g*32+r).
                float scale = bf16_to_float(scales[lr * 8 + group]);
                float zp = bf16_to_float(zeros[lr * 8 + group]);
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zp) || std::fabs(zp) > 100.0f) zp = 0.0f;
                // nibble layout (parallel_size=16): byte = lane*2048 + col*8 + (row%16)/2, low nibble = even row
                uint8_t byte_val = lane_data[col * 8 + byte_idx];
                int q = (nibble_sel == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
                int8_t val = (int8_t)(q < 8 ? q : q - 16);  // two's-complement signed int4 (0..7, -8..-1)
                out[(tile_row * TILE_ROWS + lr) * (*out_cols) +
                    (tile_col * TILE_COLS + col)] = (float)val * scale + zp;
            }
        }
    }
    return out;
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

        const uint16_t* scales = (const uint16_t*)(rd);
        const int8_t*   values = (const int8_t*)(rd + 512);  // signed INT8

        for (int lr = 0; lr < TILE_ROWS; lr++) {
            for (int col = 0; col < TILE_COLS; col++) {
                int group = col / 32;
                float scale = bf16_to_float(scales[group * 32 + lr]);
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
