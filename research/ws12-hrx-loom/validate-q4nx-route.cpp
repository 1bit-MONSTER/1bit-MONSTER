// validate_q4nx_route.cpp
//
// Standalone validation of the ggml-hrx2 q4nx_dequant_f32 route on the
// Strix Halo (gfx1151) GPU:
//   1. Open the HRX GPU device + stream.
//   2. Load the embedded ggml-hrx2 catalog, find the q4nx_dequant_f32 route.
//   3. Load the provider — this exercises the *ported* loom-jit compile path
//      (loomc_target_specialization_*, loomc_amdgpu_target_identity_*) to
//      turn the embedded .loombc artifact into a gfx1151 HSACO executable.
//   4. Split the 5120-byte Q4NX tile into packed/scales/zeros buffers,
//      dispatch the kernel over 32x256 output elements, read back f32.
//   5. Compare against the CPU reference (/tmp/q4nx_ref.f32) and report
//      max abs diff + correlation + PASS/FAIL.
//
// Build:
//   g++ -std=c++17 -O2 validate_q4nx_route.cpp \
//     -I/tmp/hrx-v2-src/ggml/src/ggml-hrx2 \
//     -I/tmp/hrx-new-install/include \
//     -L/tmp/hrx-v2-src/build/bin -lggml-hrx2 \
//     -L/tmp/hrx-new-install/lib -lhrx \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin \
//     -Wl,-rpath,/tmp/hrx-new-install/lib \
//     -o validate_q4nx

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
    if (hrx_status_is_ok(status)) {
        return true;
    }
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

static std::vector<char> read_file(const char * path, size_t expect_size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return {};
    }
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (expect_size != 0 && data.size() != expect_size) {
        std::fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, expect_size, data.size());
        return {};
    }
    return data;
}

int main(int argc, char ** argv) {
    const char * tile_path = argc > 1 ? argv[1] : "/tmp/q4nx_tile.bin";
    const char * ref_path  = argc > 2 ? argv[2] : "/tmp/q4nx_ref.f32";

    const uint32_t ncols = 256;   // Q4NX tile geometry: 32 BF16 rows x 256 cols
    const uint32_t nrows = 32;
    const uint32_t wg_size = 256;
    const uint32_t total = ncols * nrows;

    std::vector<char> tile = read_file(tile_path, 5120);
    std::vector<char> refb = read_file(ref_path, total * 4);
    if (tile.empty() || refb.empty()) {
        return 1;
    }
    const float * ref = reinterpret_cast<const float *>(refb.data());

    // --- device + stream -------------------------------------------------
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

    // --- catalog + route -------------------------------------------------
    ggml_backend_hrx2_catalog_ptr catalog = ggml_backend_hrx2_load_catalog();
    if (!catalog) { std::fprintf(stderr, "FAIL: catalog load\n"); return 2; }
    const ggml_backend_hrx2_kernel_route * route =
        ggml_backend_hrx2_catalog_find_route(*catalog, "q4nx_dequant_f32");
    if (!route) return 2;
    std::printf("route: id=%s family=%s op=%s root=%s export=%s bindings=%u\n",
                route->id.c_str(), route->family.c_str(), route->op.c_str(),
                route->root_symbol.c_str(), route->export_name.c_str(),
                route->binding_count);

    std::vector<ggml_backend_hrx2_config_binding> cfg;
    cfg.push_back({"@hrx2.shape.ncols", std::to_string(ncols)});
    cfg.push_back({"@hrx2.shape.nrows", std::to_string(nrows)});
    cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});

    ggml_backend_hrx2_device_info device_info{ device, arch.c_str() };
    std::unique_ptr<ggml_backend_hrx2_provider> provider =
        ggml_backend_hrx2_load_provider(device_info, *catalog, *route, cfg, "validate-q4nx");
    if (!provider || !provider->executable) {
        std::fprintf(stderr, "FAIL: provider load (loom-jit compile of q4nx route)\n");
        return 3;
    }
    std::printf("provider loaded: export=%s ordinal=%u workgroup_size=%u\n",
                route->export_name.c_str(), provider->export_ordinal,
                provider->export_info.workgroup_size[0]);

    // --- buffers ----------------------------------------------------------
    hrx_allocator_t alloc = hrx_device_allocator(device);
    hrx_buffer_params_t params = {
        /* .type           = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        /* .access         = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT |
                               HRX_BUFFER_USAGE_MAPPING_SCOPED |
                               HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    hrx_buffer_t packed = nullptr, scales = nullptr, zeros = nullptr, dst = nullptr;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, 4096, &packed), "alloc packed")) return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, 512,  &scales), "alloc scales")) return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, 512,  &zeros),  "alloc zeros"))  return 3;
    if (!check(hrx_allocator_allocate_buffer(alloc, params, total * 4, &dst), "alloc dst")) return 3;

    // tile layout: [0..511] scales BF16, [512..1023] zeros BF16, [1024..5119] packed int4
    if (!check(hrx_synchronous_h2d(device, tile.data() + 1024, packed, 0, 4096), "h2d packed")) return 3;
    if (!check(hrx_synchronous_h2d(device, tile.data() + 0,    scales, 0, 512),  "h2d scales")) return 3;
    if (!check(hrx_synchronous_h2d(device, tile.data() + 512,  zeros,  0, 512),  "h2d zeros"))  return 3;

    // --- dispatch ----------------------------------------------------------
    hrx_buffer_ref_t bindings[4] = {
        { packed, 0, 4096 },
        { scales, 0, 512 },
        { zeros,  0, 512 },
        { dst,    0, total * 4 },
    };
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { (total + wg_size - 1) / wg_size, 1, 1 },
        /* .workgroup_size = */  { wg_size, 1, 1 },
        /* .subgroup_size  = */  0,
    };
    std::printf("dispatch: workgroups=[%u,1,1] workgroup_size=[%u,1,1]\n",
                config.workgroup_count[0], config.workgroup_size[0]);

    if (!check(hrx_stream_dispatch(stream, provider->executable, provider->export_ordinal,
                                   &config, nullptr, 0, bindings, 4, 0), "hrx_stream_dispatch")) return 3;
    if (!check(hrx_stream_synchronize(stream), "hrx_stream_synchronize")) return 3;

    // --- read back + compare ------------------------------------------------
    std::vector<float> out(total);
    if (!check(hrx_synchronous_d2h(device, dst, 0, out.data(), out.size() * 4), "hrx_synchronous_d2h")) return 3;

    double max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_ref = 0.0, sum_out = 0.0, sum_ref2 = 0.0, sum_out2 = 0.0, sum_ro = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < total; ++i) {
        const double d = std::fabs((double) out[i] - (double) ref[i]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += ref[i];   sum_out += out[i];
        sum_ref2 += (double) ref[i] * ref[i];
        sum_out2 += (double) out[i] * out[i];
        sum_ro   += (double) ref[i] * out[i];
        if (d > 1e-3) ++mismatches;
    }
    const double mean_ref = sum_ref / total, mean_out = sum_out / total;
    const double cov = sum_ro / total - mean_ref * mean_out;
    const double var_ref = sum_ref2 / total - mean_ref * mean_ref;
    const double var_out = sum_out2 / total - mean_out * mean_out;
    const double corr = cov / std::sqrt(var_ref * var_out);

    std::printf("RESULTS total=%u\n", total);
    std::printf("  ref   range: [%.6f, %.6f] mean=%.6f\n",
                *std::min_element(ref, ref + total), *std::max_element(ref, ref + total), mean_ref);
    std::printf("  out   range: [%.6f, %.6f] mean=%.6f\n",
                *std::min_element(out.data(), out.data() + total),
                *std::max_element(out.data(), out.data() + total), mean_out);
    std::printf("  max abs diff : %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", sum_abs / total);
    std::printf("  correlation  : %.6f\n", corr);
    std::printf("  mismatches >1e-3: %zu / %u\n", mismatches, total);
    for (uint32_t i = 0; i < 8; ++i) {
        std::printf("  [%3u] ref=%.6f out=%.6f\n", i, ref[i], out[i]);
    }
    std::printf("PASS %s\n", (max_abs < 1e-3 && corr > 0.999) ? "YES" : "NO");

    hrx_buffer_release(packed);
    hrx_buffer_release(scales);
    hrx_buffer_release(zeros);
    hrx_buffer_release(dst);
    hrx_stream_release(stream);
    hrx_device_release(device);
    return 0;
}
