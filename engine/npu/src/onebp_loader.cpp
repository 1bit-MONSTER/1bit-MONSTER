/** onebp_loader.cpp — Load 1BP format models for NPU inference.
 *
 *  Reads the unified 1BP format (256-byte header + tensor index + Q4NX tiles)
 *  and provides dequantized float32 weights or packed I8 buffers for the NPU.
 *
 *  Layout on disk:
 *    [OnebpHeader: 256 bytes]
 *    [Tensor Index: variable length]
 *    [Weight Data: Q4NX tiled arrays (32×256 tiles with bf16 scales)]
 *
 *  Tile layout per 32×256 block:
 *    [0..511]:   256 BF16 scales   (8 groups × 32 rows)
 *    [512..1023]: 256 BF16 zero_points
 *    [1024..5119]: 4096 bytes packed INT4 (2 per byte, low nibble first)
 *    Total: 5120 bytes per tile
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "onebp_format.h"

// ─── Helper: convert bf16 to float32 ───────────────────────────────
static inline float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

// ─── 1BP Model Loader ──────────────────────────────────────────────
//
// Memory-maps the entire 1BP file for zero-copy access.
// Provides methods to read tensor metadata and dequantize weights.
//
// NOTE: named NpuOnebpModel, not OnebpModel — include/onebp_loader.h has a
// DIFFERENT OnebpModel (CPU-side, load()/tensor_data()) compiled into
// libonebp_model.a. Same class name = same dtor symbol = the linker keeps one
// implementation (the strong out-of-line one) and every TU destroys its
// object with the wrong layout → SIGSEGV in ~OnebpModel (ODR collision).
class NpuOnebpModel {
    int         fd_ = -1;
    uint8_t*    map_ = nullptr;
    size_t      map_size_ = 0;
    OnebpHeader hdr_;

    // Tensor index: parsed from file after header
    struct TensorEntry {
        std::string name;
        int         ndim;         // 1 = raw vector, 2 = tiled matrix, 3 = MoE expert stack
        int         rows, cols;   // per-expert dims for ndim==3; length is in `cols` for ndim==1
        int         num_experts;  // 1 unless ndim==3
        uint64_t    file_offset;  // start of this tensor's data (raw floats, or first expert's tiles)
        uint64_t    total_bytes;  // whole tensor, all experts included
        OnebpQuant  quant;        // v2: per-tensor quant (mixed-quant files)
    };
    std::vector<TensorEntry> tensors_;

public:
    NpuOnebpModel() = default;
    ~NpuOnebpModel() { close(); }

    bool open(const char* path) {
        // Memory-map the file
#ifdef _WIN32
        // File/mapping handles can be closed right after MapViewOfFile
        // succeeds — the view keeps its own reference (same pattern as
        // Q4nxReader::open in src/q4nx_reader.cpp) — so fd_ stays unused.
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "open: %s\n", path); return false; }
        LARGE_INTEGER li;
        if (!GetFileSizeEx(hFile, &li)) { CloseHandle(hFile); return false; }
        map_size_ = (size_t)li.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { CloseHandle(hFile); return false; }
        map_ = (uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMap);
        CloseHandle(hFile);
        if (!map_) { return false; }
#else
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) { perror("open"); return false; }

        struct stat st;
        if (fstat(fd_, &st) < 0) { close(); return false; }
        map_size_ = (size_t)st.st_size;

        map_ = (uint8_t*)mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) { close(); return false; }
#endif

        // Read header
        if (map_size_ < sizeof(OnebpHeader)) { fprintf(stderr, "File too small\n"); close(); return false; }
        memcpy(&hdr_, map_, sizeof(OnebpHeader));

        if (!hdr_.valid()) {
            fprintf(stderr, "Invalid 1BP header (magic=0x%x ver=%u)\n", hdr_.magic, hdr_.version);
            close(); return false;
        }

        // Parse tensor index
        const uint8_t* p = map_ + sizeof(OnebpHeader);
        for (uint32_t i = 0; i < hdr_.tensor_count; i++) {
            if ((size_t)(p - map_) + 4 > map_size_) break;

            uint32_t name_len;
            memcpy(&name_len, p, 4); p += 4;
            if (name_len > 128) break;  // sanity

            std::string name((const char*)p, name_len);
            p += name_len;
            if (*p == 0) p++;  // skip null terminator if present

            uint32_t ndim;
            memcpy(&ndim, p, 4); p += 4;

            // dims on disk: ndim==1 -> [length], ndim==2 -> [rows,cols],
            // ndim==3 -> [num_experts,rows,cols] (see onebp_format.h)
            uint32_t dims[3] = {1, 1, 1};
            for (uint32_t d = 0; d < ndim && d < 3; d++) {
                memcpy(&dims[d], p, 4); p += 4;
            }

            uint64_t offset, bytes;
            memcpy(&offset, p, 8); p += 8;
            memcpy(&bytes, p, 8); p += 8;
            OnebpQuant tquant = (OnebpQuant)hdr_.quant;
            if (hdr_.version >= 2) {
                uint32_t tq; memcpy(&tq, p, 4); p += 4;
                tquant = (OnebpQuant)tq;
            }

            TensorEntry te;
            te.name = name;
            te.quant = tquant;
            te.ndim = (int)ndim;
            if (ndim == 1) {
                te.rows = 1; te.cols = (int)dims[0]; te.num_experts = 1;
            } else if (ndim == 2) {
                te.rows = (int)dims[0]; te.cols = (int)dims[1]; te.num_experts = 1;
            } else {
                te.num_experts = (int)dims[0]; te.rows = (int)dims[1]; te.cols = (int)dims[2];
            }
            te.file_offset = offset;
            te.total_bytes = bytes;
            tensors_.push_back(te);
        }

        // Compute data section start = position after all index entries
        uint64_t data_start = (uint64_t)(p - map_);

        // Fix offsets: they are relative to data_start
        for (auto& t : tensors_) {
            t.file_offset += data_start;
        }

        return true;
    }

    void close() {
#ifdef _WIN32
        if (map_) UnmapViewOfFile(map_);
#else
        if (map_ && map_ != MAP_FAILED) munmap(map_, map_size_);
        if (fd_ >= 0) ::close(fd_);
#endif
        map_ = nullptr; fd_ = -1; map_size_ = 0;
        tensors_.clear();
    }

    bool is_open() const { return map_ != nullptr; }
    const OnebpHeader& header() const { return hdr_; }

    int tensor_count() const { return (int)tensors_.size(); }
    const TensorEntry* tensor(int i) const {
        return (i >= 0 && i < (int)tensors_.size()) ? &tensors_[i] : nullptr;
    }
    const TensorEntry* find_tensor(const char* name) const {
        for (auto& t : tensors_)
            if (t.name == name) return &t;
        return nullptr;
    }

    // ── Dequantize a single tile (32×256) to float32 ──
    // `tile_data` points to the 5120-byte tile in the mmap'ed file
    static void dequant_tile(const uint8_t* tile_data, float* output,
                             int out_rows, int out_cols,
                             int tile_rows = 32, int tile_cols = 256, int group_size = 32,
                             OnebpQuant q = ONEBP_Q4NX) {
        (void)q;  // Q4NX-only; TQ2-family handled by dequant_tile_tq2
        int groups = tile_cols / group_size;
        const uint16_t* scales = (const uint16_t*)tile_data;
        const uint16_t* zps    = (const uint16_t*)(tile_data + (size_t)tile_rows * groups * 2);
        const uint8_t*  qdata  = tile_data + (size_t)tile_rows * groups * 4;

        for (int r = 0; r < tile_rows && r < out_rows; r++) {
            for (int g = 0; g < groups; g++) {
                float scale = bf16_to_f32(scales[r * groups + g]);
                float zp    = bf16_to_f32(zps[r * groups + g]);
                if (scale < 1e-10f) scale = 1.0f;

                for (int i = 0; i < group_size && g * group_size + i < out_cols; i += 2) {
                    int col = g * group_size + i;
                    int byte_idx = (r * tile_cols + col) / 2;
                    uint8_t packed = qdata[byte_idx];
                    uint8_t v0 = packed & 0xF;
                    uint8_t v1 = packed >> 4;

                    output[r * out_cols + col] = (float)v0 * scale + zp;
                    if (col + 1 < out_cols)
                        output[r * out_cols + col + 1] = (float)v1 * scale + zp;
                }
            }
        }
    }

    // ── Dequantize a single TQ2/TQ2NZ tile (32×256) to float32 ──
    // TQ2 symmetric ternary: code 0=-scale, 1=0, 2=+scale, 3=unused->0.
    // TQ2NZ no-zero S40: code 0=-4s, 1=-1s, 2=+1s, 3=+4s.
    // Packed 2 bits/value, 4 per byte LSB-first (see onebp_format.h).
    static void dequant_tile_tq2(const uint8_t* tile_data, float* output,
                                 int out_rows, int out_cols,
                                 int tile_rows = 32, int tile_cols = 256, int group_size = 32,
                                 bool no_zero = false, bool e4m3_scales = false) {
        int groups = tile_cols / group_size;
        const uint8_t* qdata  = tile_data + (size_t)tile_rows * groups * (e4m3_scales ? 1 : 2);

        for (int r = 0; r < tile_rows && r < out_rows; r++) {
            for (int g = 0; g < groups; g++) {
                float scale;
                if (e4m3_scales) {
                    scale = onebp_ue4m3_to_f32(tile_data[r * groups + g]);
                } else {
                    const uint16_t* scales = (const uint16_t*)tile_data;
                    scale = bf16_to_f32(scales[r * groups + g]);
                }

                for (int i = 0; i < group_size && g * group_size + i < out_cols; i += 4) {
                    int col = g * group_size + i;
                    int byte_idx = (r * tile_cols + col) / 4;
                    uint8_t packed = qdata[byte_idx];
                    for (int k = 0; k < 4 && col + k < out_cols; k++) {
                        uint8_t code = (packed >> (2 * k)) & 0x3;
                        float v;
                        if (no_zero) {
                            static const float cb[4] = { -4.0f, -1.0f, 1.0f, 4.0f };
                            v = cb[code] * scale;
                        } else {
                            v = (code == 0) ? -scale : (code == 2) ? scale : 0.0f;
                        }
                        output[r * out_cols + col + k] = v;
                    }
                }
            }
        }
    }

    // ── Dequantize a tiled 2D matrix starting at `base` into `out` ──
    // Dispatches on hdr_.quant — Q4NX (4-bit, default) or TQ2 (2-bit ternary).
    void dequant_matrix(const uint8_t* base, int R, int C, std::vector<float>& out, OnebpQuant q = (OnebpQuant)0xFFFFFFFFu) const {
        int tr = hdr_.tile_rows, tc = hdr_.tile_cols, gs = hdr_.group_size;
        int ntr = (R + tr - 1) / tr;
        int ntc = (C + tc - 1) / tc;
        bool is_tq2 = q == ONEBP_TQ2 || q == ONEBP_TQ2NZ || q == ONEBP_TQ2NZ_E4M3;
        bool tq2nz = q == ONEBP_TQ2NZ || q == ONEBP_TQ2NZ_E4M3;
        bool e4m3 = q == ONEBP_TQ2NZ_E4M3;
        size_t tile_bytes = is_tq2
            ? (size_t)tr * (tc / gs) * (e4m3 ? 1 : 2) + (size_t)tr * tc / 4
            : (size_t)tr * (tc / gs) * 4 + (size_t)tr * tc / 2;

        out.resize((size_t)R * C);
        memset(out.data(), 0, out.size() * sizeof(float));

        for (int trr = 0; trr < ntr; trr++) {
            for (int tcc = 0; tcc < ntc; tcc++) {
                int r0 = trr * tr, c0 = tcc * tc;
                int rh = (R - r0) < tr ? (R - r0) : tr;
                int cw = (C - c0) < tc ? (C - c0) : tc;

                float tile_buf[32 * 256];  // max tile size
                // Pass the FULL tile width (tc) as out_cols, not the actual
                // tile width (cw): the dequant helpers write tile_buf with
                // out_cols stride while the copy below reads it back with the
                // full tile stride (r*tc + c). For a partial last tile column
                // (C % 256 != 0, e.g. Gemma-3-1B hidden=1152) the compact
                // stride misaligned every row — reading row 2r's data for
                // row r and corrupting the last 128 cols of every matrix
                // (caught by the #1243 ppl gate: Gemma-3-1B ppl 5e9).
                if (is_tq2) dequant_tile_tq2(base, tile_buf, rh, tc, tr, tc, gs, tq2nz, e4m3);
                else        dequant_tile(base, tile_buf, rh, tc, tr, tc, gs, q);

                for (int r = 0; r < rh; r++)
                    for (int c = 0; c < cw; c++)
                        out[(size_t)(r0 + r) * C + (c0 + c)] = tile_buf[r * tc + c];

                base += tile_bytes;
            }
        }
    }

    // ── Get a tensor as float32: raw vector (ndim==1) or dequantized
    // matrix (ndim==2). For ndim==3 (MoE expert stack) use
    // get_tensor_f32_expert() instead — flattening experts together
    // would silently mix independent weight matrices. ──
    bool get_tensor_f32(const char* name, std::vector<float>& out) const {
        auto* te = find_tensor(name);
        if (!te) return false;

        if (te->ndim == 1) {
            out.resize((size_t)te->cols);
            memcpy(out.data(), map_ + te->file_offset, out.size() * sizeof(float));
            return true;
        }
        if (te->ndim == 3) {
            fprintf(stderr, "'%s' is a %d-expert MoE tensor — use get_tensor_f32_expert()\n",
                    name, te->num_experts);
            return false;
        }
        // Truncated-file guard (issue #1243): a converter that bailed mid-
        // write leaves planned offsets past EOF — mmap'ed reads there SIGSEGV
        // instead of failing the gate cleanly. total_bytes is the conservative
        // bound (>= what dequant_matrix actually reads).
        if (te->file_offset + te->total_bytes > map_size_) {
            fprintf(stderr, "'%s' extends past EOF (off=%llu+%llu > map=%zu) — truncated file\n",
                    name, (unsigned long long)te->file_offset,
                    (unsigned long long)te->total_bytes, map_size_);
            return false;
        }
        dequant_matrix(map_ + te->file_offset, te->rows, te->cols, out, te->quant);
        return true;
    }

    // ── Get one expert's slice of a 3D MoE tensor as float32 ──
    bool get_tensor_f32_expert(const char* name, int expert_idx, std::vector<float>& out) const {
        auto* te = find_tensor(name);
        if (!te || te->ndim != 3) return false;
        if (expert_idx < 0 || expert_idx >= te->num_experts) return false;

        uint64_t per_expert_bytes = te->total_bytes / (uint64_t)te->num_experts;
        if (te->file_offset + (uint64_t)expert_idx * per_expert_bytes + per_expert_bytes > map_size_) {
            fprintf(stderr, "'%s' expert %d extends past EOF — truncated file\n", name, expert_idx);
            return false;
        }
        dequant_matrix(map_ + te->file_offset + expert_idx * per_expert_bytes,
                        te->rows, te->cols, out, te->quant);
        return true;
    }

    // ── Get raw tile pointer for direct NPU DMA (ndim==2 tensors only) ──
    // Returns pointer to the tile data in the mmap'ed file
    const uint8_t* get_tile_ptr(const char* name, int tile_row, int tile_col) const {
        auto* te = find_tensor(name);
        if (!te || te->ndim != 2) return nullptr;

        int tr = hdr_.tile_rows, tc = hdr_.tile_cols, gs = hdr_.group_size;
        int ntc = (te->cols + tc - 1) / tc;
        size_t tile_bytes = (te->quant == ONEBP_TQ2 || te->quant == ONEBP_TQ2NZ ||
                              te->quant == ONEBP_TQ2NZ_E4M3)
            ? (size_t)tr * (tc / gs) * (te->quant == ONEBP_TQ2NZ_E4M3 ? 1 : 2) + (size_t)tr * tc / 4
            : (size_t)tr * (tc / gs) * 4 + (size_t)tr * tc / 2;

        uint64_t off = te->file_offset + (uint64_t)(tile_row * ntc + tile_col) * tile_bytes;
        if (off + tile_bytes > map_size_) return nullptr;
        return map_ + off;
    }

    // ── Get raw tile pointer within one expert's slice of a 3D tensor ──
    const uint8_t* get_tile_ptr_expert(const char* name, int expert_idx,
                                        int tile_row, int tile_col) const {
        auto* te = find_tensor(name);
        if (!te || te->ndim != 3) return nullptr;
        if (expert_idx < 0 || expert_idx >= te->num_experts) return nullptr;

        int tr = hdr_.tile_rows, tc = hdr_.tile_cols, gs = hdr_.group_size;
        int ntc = (te->cols + tc - 1) / tc;
        size_t tile_bytes = (te->quant == ONEBP_TQ2 || te->quant == ONEBP_TQ2NZ ||
                              te->quant == ONEBP_TQ2NZ_E4M3)
            ? (size_t)tr * (tc / gs) * (te->quant == ONEBP_TQ2NZ_E4M3 ? 1 : 2) + (size_t)tr * tc / 4
            : (size_t)tr * (tc / gs) * 4 + (size_t)tr * tc / 2;
        uint64_t per_expert_bytes = te->total_bytes / (uint64_t)te->num_experts;

        uint64_t off = te->file_offset + expert_idx * per_expert_bytes
                     + (uint64_t)(tile_row * ntc + tile_col) * tile_bytes;
        if (off + tile_bytes > map_size_) return nullptr;
        return map_ + off;
    }
};

// ─── Example usage ─────────────────────────────────────────────────
#ifdef ONEBP_LOADER_MAIN
#ifdef ONEBP_LOADER_MAIN
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.1bp [tensor_name]\n", argv[0]);
        return 1;
    }

    NpuOnebpModel model;
    if (!model.open(argv[1])) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }

    auto& h = model.header();
    printf("1BP Model: %s\n", argv[1]);
    printf("  Architecture: %s\n", h.model_tag);
    printf("  H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",
           h.hidden_size, h.num_layers, h.num_attention_heads,
           h.num_kv_heads, h.head_dim, h.intermediate_size, h.vocab_size);
    printf("  Quant: %s  Tiles: %dx%d  Group: %d\n",
           h.quant == 0 ? "Q4NX" : "other",
           h.tile_rows, h.tile_cols, h.group_size);
    printf("  Tensors: %d\n", h.tensor_count);

    if (argc > 2) {
        std::vector<float> data;
        if (model.get_tensor_f32(argv[2], data)) {
            printf("  Tensor '%s': %zu elements\n", argv[2], data.size());
            printf("  First 8 values: ");
            for (int i = 0; i < 8 && i < (int)data.size(); i++)
                printf("%.4f ", data[i]);
            printf("\n");
        } else {
            printf("  Tensor '%s' not found\n", argv[2]);
        }
    }

    // List all tensors
    printf("\n  Tensor list:\n");
    for (int i = 0; i < model.tensor_count() && i < 10; i++) {
        auto* t = model.tensor(i);
        printf("    %-50s ndim=%d %4dx%-4d x%d experts  %llu bytes\n",
               t->name.c_str(), t->ndim, t->rows, t->cols, t->num_experts,
               (unsigned long long)t->total_bytes);
    }
    if (model.tensor_count() > 10)
        printf("    ... and %d more\n", model.tensor_count() - 10);

    return 0;
#endif
}
#endif
