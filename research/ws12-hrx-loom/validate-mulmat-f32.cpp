// validate_mulmat_f32.cpp
//
// Standalone validation of the ggml-hrx2 mul_mat_f32_f32 route on the
// Strix Halo (gfx1151) GPU — the first REAL matmul through the ported
// loom-jit + restored full catalog pipeline.
//
// Kernel contract (kernels/mul_mat_f32_f32.loom):
//   dst[col*rows + row] = sum_k src0[row*k + k] * src1[col*k + k]
//   grid: workgroups(rows, cols, 1) x workgroup_size(256,1,1)
//
// Reference computed in double on the host, compared with f32 tolerance
// (the kernel accumulates with reassoc in f32, so expect ~1e-6..1e-7
// relative differences, not bit-exactness).
//
// Build:
//   g++ -std=c++17 -O2 validate_mulmat_f32.cpp \
//     -I/tmp/hrx-v2-src/ggml/src/ggml-hrx2 \
//     -I/tmp/hrx-new-install/include -I/tmp/hrx-new-install/include/hrx \
//     -L/tmp/hrx-v2-src/build/bin -lggml-hrx2 \
//     -L/tmp/hrx-new-install/lib -lhrx \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin \
//     -Wl,-rpath,/tmp/hrx-new-install/lib \
//     -o validate_mulmat_f32

#include "hrx/hrx_runtime.h"
#include "ggml-hrx2-catalog.h"

#include <algorithm>
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

int main(int argc, char ** argv) {
    const uint32_t k = 2048, rows = 128, cols = 64;
    const uint32_t wg_size = 256;

    if (!check(hrx_gpu_initialize(0), "hrx_gpu_initialize")) return 2;
    hrx_device_t device = nullptr;
    if (!check(hrx_gpu_device_get(0, &device), "hrx_gpu_device_get(0)")) return 2;

    std::string arch(64, '\0');
    size_t arch_size = arch.size();
    if (!check(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch_size), "get architecture")) return 2;
    arch.resize(std::strlen(arch.c_str()));
    std::printf("device architecture: %s\n", arch.c_str());

    hrx_stream_t stream = nullptr;
    if (!check(hrx_stream_create(device, 0, &stream), "hrx_stream_create")) return 2;

    ggml_backend_hrx2_catalog_ptr catalog = ggml_backend_hrx2_load_catalog();
    if (!catalog) { std::fprintf(stderr, "FAIL: catalog load\n"); return 2; }
    const ggml_backend_hrx2_kernel_route * route =
        ggml_backend_hrx2_catalog_find_route(*catalog, "mul_mat_f32_f32_moe_logits_k2048_r128_c1_512_wg256");
    if (!route) return 2;
    std::printf("route: id=%s root=%s export=%s bindings=%u\n",
                route->id.c_str(), route->root_symbol.c_str(),
                route->export_name.c_str(), route->binding_count);

    std::vector<ggml_backend_hrx2_config_binding> cfg;
    cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
    cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
    cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
    cfg.push_back({"@hrx2.tuning.workgroup_size", std::to_string(wg_size)});

    ggml_backend_hrx2_device_info device_info{ device, arch.c_str() };
    std::unique_ptr<ggml_backend_hrx2_provider> provider =
        ggml_backend_hrx2_load_provider(device_info, *catalog, *route, cfg, "validate-mulmat-f32");
    if (!provider || !provider->executable) {
        std::fprintf(stderr, "FAIL: provider load (loom-jit compile of mul_mat_f32 route)\n");
        return 3;
    }
    std::printf("provider loaded: export=%s ordinal=%u\n",
                route->export_name.c_str(), provider->export_ordinal);

    // --- host data: deterministic -------------------------------------------------
    const size_t src0_count = (size_t) rows * k;
    const size_t src1_count = (size_t) cols * k;
    const size_t dst_count  = (size_t) cols * rows;
    std::vector<float> src0(src0_count), src1(src1_count);
    for (size_t i = 0; i < src0_count; ++i) src0[i] = std::sin(0.001f * (float) i);
    for (size_t i = 0; i < src1_count; ++i) src1[i] = std::cos(0.001f * (float) i + 0.5f);

    // double-precision reference: dst[col*rows + row] = sum_k src0[row*k+k]*src1[col*k+k]
    std::vector<double> ref(dst_count, 0.0);
    for (uint32_t c = 0; c < cols; ++c) {
        for (uint32_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            const float * a = src0.data() + (size_t) r * k;
            const float * b = src1.data() + (size_t) c * k;
            for (uint32_t i = 0; i < k; ++i) acc += (double) a[i] * (double) b[i];
            ref[(size_t) c * rows + r] = acc;
        }
    }

    // --- buffers + upload ---------------------------------------------------------
    hrx_allocator_t alloc = hrx_device_allocator(device);
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        0,
    };
    hrx_buffer_t b_src0 = nullptr, b_src1 = nullptr, b_dst = nullptr;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, src0_count * 4, &b_src0), "alloc src0")) return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, src1_count * 4, &b_src1), "alloc src1")) return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, dst_count * 4,  &b_dst),  "alloc dst"))  return 3;
    if (!check(hrx_synchronous_h2d(device, src0.data(), b_src0, 0, src0_count * 4), "h2d src0")) return 3;
    if (!check(hrx_synchronous_h2d(device, src1.data(), b_src1, 0, src1_count * 4), "h2d src1")) return 3;

    // --- dispatch ------------------------------------------------------------------
    hrx_buffer_ref_t bindings[3] = {
        { b_src0, 0, src0_count * 4 },
        { b_src1, 0, src1_count * 4 },
        { b_dst,  0, dst_count * 4 },
    };
    hrx_dispatch_config_t config = {
        { rows, cols, 1 },
        { wg_size, 1, 1 },
        0,
    };
    std::printf("dispatch: workgroups=[%u,%u,1] workgroup_size=[%u,1,1]\n",
                config.workgroup_count[0], config.workgroup_count[1], config.workgroup_size[0]);
    if (!check(hrx_stream_dispatch(stream, provider->executable, provider->export_ordinal,
                                   &config, nullptr, 0, bindings, 3, 0), "hrx_stream_dispatch")) return 3;
    if (!check(hrx_stream_synchronize(stream), "hrx_stream_synchronize")) return 3;

    // --- read back + compare ----------------------------------------------------------
    std::vector<float> out(dst_count);
    if (!check(hrx_synchronous_d2h(device, b_dst, 0, out.data(), out.size() * 4), "hrx_synchronous_d2h")) return 3;

    double max_abs = 0.0, max_rel = 0.0, sum_abs = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < dst_count; ++i) {
        const double d = std::fabs((double) out[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        const double denom = std::max(std::fabs(ref[i]), 1e-12);
        max_rel = std::max(max_rel, d / denom);
        if (d > 1e-3) ++mismatches;
    }
    std::printf("RESULTS total=%zu k=%u rows=%u cols=%u\n", dst_count, k, rows, cols);
    std::printf("  ref   range: [%.4f, %.4f]\n", *std::min_element(ref.begin(), ref.end()),
                *std::max_element(ref.begin(), ref.end()));
    std::printf("  out   range: [%.4f, %.4f]\n", *std::min_element(out.begin(), out.end()),
                *std::max_element(out.begin(), out.end()));
    std::printf("  max abs diff : %.6e\n", max_abs);
    std::printf("  max rel diff : %.6e\n", max_rel);
    std::printf("  mean abs diff: %.6e\n", sum_abs / dst_count);
    std::printf("  mismatches >1e-3: %zu / %zu\n", mismatches, dst_count);
    for (size_t i = 0; i < 6; ++i) {
        std::printf("  [%3zu] ref=%.6f out=%.6f\n", i, ref[i], out[i]);
    }
    std::printf("PASS %s\n", (max_rel < 1e-4 && mismatches == 0) ? "YES" : "NO");

    hrx_buffer_release(b_src0);
    hrx_buffer_release(b_src1);
    hrx_buffer_release(b_dst);
    hrx_stream_release(stream);
    hrx_device_release(device);
    return 0;
}
