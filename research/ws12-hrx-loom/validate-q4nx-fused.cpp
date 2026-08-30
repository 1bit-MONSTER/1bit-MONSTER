// validate_q4nx_fused.cpp
//
// FUSED Q4NX-on-HRX demo (option C): two proven ggml-hrx2 routes pipelined on
// the Strix Halo (gfx1151) GPU with NO CPU round-trip between them.
//
//   1. q4nx_dequant_f32: 8 real Q4NX tiles from zaya1-8b-fresh.q4nx
//      (layer-0 q_proj, file offset 335961069; each 5120 B = [scales 512 B |
//      zeros 512 B | packed 4096 B]) -> dequant on GPU to a f32 weight
//      [32 rows x 2048 cols]. The output buffer stays on the device.
//   2. mul_mat_f32_f32: that SAME device buffer is src0 (rows=32, k=2048);
//      an activation vector x [2048] is src1 (cols=1) -> y [32].
//
// Reference on host: dequant with the engine's verified raw semantics
// (q4nx_raw.h: q = nibble two's-complement int4; W = q*scale + zp), then
// y_ref[row] = sum_k W[row][k]*x[k] in double. The GPU pipeline is correct
// iff it matches the CPU reference that starts from the same Q4NX bytes.
//
// Build:
//   g++ -std=c++17 -O2 validate_q4nx_fused.cpp \
//     -I/tmp/hrx-v2-src/ggml/src/ggml-hrx2 \
//     -I/tmp/hrx-new-install/include -I/tmp/hrx-new-install/include/hrx \
//     -L/tmp/hrx-v2-src/build/bin -lggml-hrx2 \
//     -L/tmp/hrx-new-install/lib -lhrx \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin \
//     -Wl,-rpath,/tmp/hrx-new-install/lib \
//     -o validate_q4nx_fused

#include "hrx/hrx_runtime.h"
#include "ggml-hrx2-catalog.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static bool check(hrx_status_t status, const char * what) {
    if (hrx_status_is_ok(status)) return true;
    char * message = nullptr;
    size_t length = 0;
    if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) && message) {
        std::fprintf(stderr, "FAIL %s: %.*s\n", what, (int) length, message);
        hrx_status_free_message(message);
    } else {
        std::fprintf(stderr, "FAIL %s (status not ok)\n", what);
    }
    hrx_status_ignore(status);
    return false;
}

// BF16 -> f32 (engine q4nx_raw.h rdbf16)
static inline float bf16_to_f32(const uint8_t * p) {
    uint16_t v = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    uint32_t bits = (uint32_t) v << 16;
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char ** argv) {
    const char * tile_path = argc > 1 ? argv[1] : "/tmp/q4nx_w8.bin";
    const uint32_t n_tiles = 8;                 // 8 consecutive I8 rows = tile_row 0, cols 0..2047
    const uint32_t ncols = n_tiles * 256;       // 2048
    const uint32_t nrows = 32;                  // one tile row
    const uint32_t wg_size = 256;

    std::ifstream tf(tile_path, std::ios::binary);
    std::vector<char> tiles((std::istreambuf_iterator<char>(tf)), {});
    if (tiles.size() != (size_t) n_tiles * 5120) {
        std::fprintf(stderr, "expected %u bytes, got %zu\n", n_tiles * 5120, tiles.size());
        return 1;
    }

    // --- CPU reference: raw dequant (q4nx_raw.h semantics) + double matmul ----
    // W[rows=32][cols=2048]; per I8 row ir (tile): tile_col = ir, logical cols
    // [tile_col*256, (tile_col+1)*256)
    const uint32_t cols = ncols;
    std::vector<float> W_cpu((size_t) nrows * cols, 0.0f);
    for (uint32_t ir = 0; ir < n_tiles; ++ir) {
        const uint8_t * scales = (const uint8_t *) tiles.data() + (size_t) ir * 5120;
        const uint8_t * zeros  = scales + 512;
        const uint8_t * packed = scales + 1024;
        for (uint32_t lr = 0; lr < nrows; ++lr) {
            uint32_t lane = lr / 16, lane_row = lr % 16;
            uint32_t byte_idx = lane_row / 2, nib = lr % 2;
            const uint8_t * lane_data = packed + lane * 2048;
            uint32_t row = lr;
            for (uint32_t c = 0; c < 256; ++c) {
                uint32_t col = ir * 256 + c;
                uint8_t b = lane_data[c * 8 + byte_idx];
                int q = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
                if (q >= 8) q -= 16;                       // two's-complement int4
                float scale = bf16_to_f32(scales + (lr * 8 + c / 32) * 2);
                float zp    = bf16_to_f32(zeros  + (lr * 8 + c / 32) * 2);
                // engine dequant_q4nx.cpp clamp: non-finite/outlier -> 0
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zp) || std::fabs(zp) > 100.0f) zp = 0.0f;
                W_cpu[(size_t) row * cols + col] = (float) q * scale + zp;
            }
        }
    }
    std::vector<float> x(cols);
    for (uint32_t i = 0; i < cols; ++i) x[i] = std::sin(0.01f * (float) i + 0.25f);
    std::vector<double> y_ref(nrows, 0.0);
    for (uint32_t r = 0; r < nrows; ++r) {
        double acc = 0.0;
        const float * wrow = W_cpu.data() + (size_t) r * cols;
        for (uint32_t i = 0; i < cols; ++i) acc += (double) wrow[i] * (double) x[i];
        y_ref[r] = acc;
    }

    // --- device + streams ------------------------------------------------------
    if (!check(hrx_gpu_initialize(0), "hrx_gpu_initialize")) return 2;
    hrx_device_t device = nullptr;
    if (!check(hrx_gpu_device_get(0, &device), "hrx_gpu_device_get(0)")) return 2;
    std::string arch(64, '\0');
    size_t arch_size = arch.size();
    if (!check(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch_size), "get architecture")) return 2;
    arch.resize(std::strlen(arch.c_str()));
    hrx_stream_t stream = nullptr;
    if (!check(hrx_stream_create(device, 0, &stream), "hrx_stream_create")) return 2;

    // --- catalog + providers (both routes) ---------------------------------------
    ggml_backend_hrx2_catalog_ptr catalog = ggml_backend_hrx2_load_catalog();
    if (!catalog) { std::fprintf(stderr, "FAIL: catalog load\n"); return 2; }

    const ggml_backend_hrx2_kernel_route * deq_route =
        ggml_backend_hrx2_catalog_find_route(*catalog, "q4nx_dequant_f32");
    const ggml_backend_hrx2_kernel_route * mm_route =
        ggml_backend_hrx2_catalog_find_route(*catalog, "mul_mat_f32_f32_moe_logits_k2048_r128_c1_512_wg256");
    if (!deq_route || !mm_route) return 2;

    ggml_backend_hrx2_device_info device_info{ device, arch.c_str() };
    std::vector<ggml_backend_hrx2_config_binding> deq_cfg;
    deq_cfg.push_back({"@hrx2.shape.ncols", std::to_string(ncols)});
    deq_cfg.push_back({"@hrx2.shape.nrows", std::to_string(nrows)});
    deq_cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
    std::unique_ptr<ggml_backend_hrx2_provider> deq =
        ggml_backend_hrx2_load_provider(device_info, *catalog, *deq_route, deq_cfg, "fused-deq");
    if (!deq || !deq->executable) { std::fprintf(stderr, "FAIL: dequant provider\n"); return 3; }

    std::vector<ggml_backend_hrx2_config_binding> mm_cfg;
    mm_cfg.push_back({"@hrx2.shape.k", std::to_string(ncols)});
    mm_cfg.push_back({"@hrx2.shape.rows", std::to_string(nrows)});
    mm_cfg.push_back({"@hrx2.shape.cols", "1"});
    mm_cfg.push_back({"@hrx2.tuning.workgroup_size", std::to_string(wg_size)});
    std::unique_ptr<ggml_backend_hrx2_provider> mm =
        ggml_backend_hrx2_load_provider(device_info, *catalog, *mm_route, mm_cfg, "fused-mm");
    if (!mm || !mm->executable) { std::fprintf(stderr, "FAIL: matmul provider\n"); return 3; }

    // --- buffers ----------------------------------------------------------------
    const size_t packed_bytes = (size_t) n_tiles * 4096;
    const size_t scl_bytes    = (size_t) n_tiles * 512;
    const size_t W_bytes      = (size_t) nrows * ncols * 4;
    const size_t x_bytes      = (size_t) ncols * 4;
    const size_t y_bytes      = (size_t) nrows * 4;
    hrx_allocator_t alloc = hrx_device_allocator(device);
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        0,
    };
    hrx_buffer_t b_packed = nullptr, b_scl = nullptr, b_zp = nullptr,
                 b_W = nullptr, b_x = nullptr, b_y = nullptr;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, packed_bytes, &b_packed), "alloc packed")) return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, scl_bytes,    &b_scl),    "alloc scales")) return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, scl_bytes,    &b_zp),     "alloc zeros"))  return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, W_bytes,      &b_W),      "alloc W"))      return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, x_bytes,      &b_x),      "alloc x"))      return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, y_bytes,      &b_y),      "alloc y"))      return 3;

    // tile sections: [scales 512][zeros 512][packed 4096] per tile
    std::vector<char> scales(scl_bytes), zeros(scl_bytes), packed(packed_bytes);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        const char * base = tiles.data() + (size_t) t * 5120;
        std::memcpy(scales.data() + (size_t) t * 512, base, 512);
        std::memcpy(zeros.data()  + (size_t) t * 512, base + 512, 512);
        std::memcpy(packed.data() + (size_t) t * 4096, base + 1024, 4096);
    }
    if (!check(hrx_synchronous_h2d(device, packed.data(), b_packed, 0, packed_bytes), "h2d packed")) return 3;
    if (!check(hrx_synchronous_h2d(device, scales.data(), b_scl, 0, scl_bytes), "h2d scales")) return 3;
    if (!check(hrx_synchronous_h2d(device, zeros.data(),  b_zp,  0, scl_bytes), "h2d zeros"))  return 3;
    if (!check(hrx_synchronous_h2d(device, x.data(),      b_x,   0, x_bytes),   "h2d x"))      return 3;

    // --- FUSED dispatch: dequant -> matmul, W stays on device ----------------------
    hrx_buffer_ref_t deq_bindings[4] = {
        { b_packed, 0, packed_bytes },
        { b_scl,    0, scl_bytes },
        { b_zp,     0, scl_bytes },
        { b_W,      0, W_bytes },
    };
    hrx_dispatch_config_t deq_cfg_d = {
        { (nrows * ncols + wg_size - 1) / wg_size, 1, 1 },
        { wg_size, 1, 1 },
        0,
    };
    hrx_buffer_ref_t mm_bindings[3] = {
        { b_W, 0, W_bytes },   // src0 = dequant output, same device buffer
        { b_x, 0, x_bytes },   // src1 = activations
        { b_y, 0, y_bytes },
    };
    hrx_dispatch_config_t mm_cfg_d = {
        { nrows, 1, 1 },
        { wg_size, 1, 1 },
        0,
    };

    auto t0 = std::chrono::steady_clock::now();
    if (!check(hrx_stream_dispatch(stream, deq->executable, deq->export_ordinal,
                                   &deq_cfg_d, nullptr, 0, deq_bindings, 4, 0), "dispatch dequant")) return 3;
    if (!check(hrx_stream_dispatch(stream, mm->executable, mm->export_ordinal,
                                   &mm_cfg_d, nullptr, 0, mm_bindings, 3, 0), "dispatch matmul")) return 3;
    if (!check(hrx_stream_synchronize(stream), "hrx_stream_synchronize")) return 3;
    auto t1 = std::chrono::steady_clock::now();
    const double fused_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // --- read back + compare --------------------------------------------------------
    std::vector<float> y(nrows);
    if (!check(hrx_synchronous_d2h(device, b_y, 0, y.data(), y_bytes), "hrx_synchronous_d2h")) return 3;

    double max_abs = 0.0, max_rel = 0.0;
    for (uint32_t r = 0; r < nrows; ++r) {
        const double d = std::fabs((double) y[r] - y_ref[r]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(std::fabs(y_ref[r]), 1e-12));
    }
    std::printf("FUSED Q4NX pipeline (dequant [32x2048] + matmul [32x2048]x[2048]) on %s\n", arch.c_str());
    std::printf("  fused dispatch time: %.0f us (dequant + matmul, no CPU round-trip)\n", fused_us);
    std::printf("  y_ref range: [%.6f, %.6f]\n",
                *std::min_element(y_ref.begin(), y_ref.end()),
                *std::max_element(y_ref.begin(), y_ref.end()));
    std::printf("  y_out range: [%.6f, %.6f]\n",
                *std::min_element(y.begin(), y.end()),
                *std::max_element(y.begin(), y.end()));
    std::printf("  max abs diff : %.6e\n", max_abs);
    std::printf("  max rel diff : %.6e\n", max_rel);
    for (uint32_t r = 0; r < nrows && r < 8; ++r) {
        std::printf("  y[%2u] ref=%.6f out=%.6f\n", r, y_ref[r], y[r]);
    }
    std::printf("PASS %s\n", (max_rel < 1e-3) ? "YES" : "NO");

    hrx_buffer_release(b_packed);
    hrx_buffer_release(b_scl);
    hrx_buffer_release(b_zp);
    hrx_buffer_release(b_W);
    hrx_buffer_release(b_x);
    hrx_buffer_release(b_y);
    hrx_stream_release(stream);
    hrx_device_release(device);
    return 0;
}
