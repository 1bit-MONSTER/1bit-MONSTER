// cpp26_name_maps_selfcheck.cpp — issue #1956 verification harness.
//
// Proves the enum↔name tables (common.h BackendType, pilot.h SubLayer +
// PilotBackend, backend_manager.h BackendTier) are exhaustive and unchanged
// in output, and exercises the toolchain-shipped C++26 features the issue
// tracked:
//   - P2996 reflection (std::meta): with -freflection the *headers themselves*
//     static_assert table completeness (compile error if an enumerator lacks a
//     row); this TU additionally prints the mapping auto-derived from the enum
//     via enumerators_of (the "generate from enum" direction).
//   - std::inplace_vector (P0843): decode-path style fixed-capacity buffers,
//     no heap.
//
// Build (ryzen/strixhalo, g++-16/libstdc++16 installed 2026-09-06):
//   g++-16 -std=c++26 -freflection -O2 -Iinclude -Isrc \
//       Testing/cpp26_name_maps_selfcheck.cpp -o /tmp/cpp26_check
//   /tmp/cpp26_check
//
// A plain compiler (g++-15, clang/amdclang) also builds and runs this — the
// reflection/inplace_vector sections are macro-gated off, runtime name checks
// still run. That keeps CI runners (no g++-16 yet) green.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "common.h"          // BackendType / kBackendTypeNames / backend_name
#include "pilot.h"           // SubLayer + PilotBackend tables
#include "backend_manager.h" // BackendTier / kBackendTierNames / tier_name

// ---------------------------------------------------------------------------
// 1. Runtime checks: every table row still yields the exact legacy strings.
// ---------------------------------------------------------------------------
static int g_fail = 0;
#define CHECK(cond, label)                                                     \
    do {                                                                       \
        if (cond) std::printf("ok   %s\n", label);                             \
        else { std::printf("FAIL %s\n", label); ++g_fail; }                    \
    } while (0)

static void check_backend_names() {
    struct { BackendType type; const char* name; } expect[] = {
        {BackendType::NONE,        "none"},
        {BackendType::HIP_GPU,     "HIP GPU (ROCm)"},
        {BackendType::VULKAN,      "Vulkan GPU (portable)"},
        {BackendType::NPU_XRT,     "NPU XDNA (XRT)"},
        {BackendType::CPU_AVX512,  "CPU AVX-512"},
        {BackendType::CPU_SCALAR,  "CPU (scalar)"},
        {BackendType::GENERIC,     "Generic CPU (GGUF)"},
        {BackendType::ZAMBA2,      "Zamba2 (Mamba2 CPU)"},
        {BackendType::ZAMBA2_GPU,  "Zamba2 (Mamba2 GPU)"},
        {BackendType::ZINC_GPU,    "ZINC GPU (Vulkan, multi-arch)"},
        {BackendType::Q4NX_FUSION, "Q4NX Fusion (CPU)"},
        {BackendType::CUDA_GPU,    "CUDA GPU (NVIDIA)"},
        {BackendType::METAL_GPU,   "Metal GPU (Apple)"},
        {BackendType::VART,        "VART (Versal/Zynq DPU)"},
        {BackendType::ONNX_NPU,    "ONNX NPU (VitisAI EP)"},
        {BackendType::LSE_GPU,     "LSE GPU (MLX via lse-server)"},
        {BackendType::HRX_GPU,     "HRX GPU (fused GGUF via hrx llama-server)"},
    };
    bool all = true;
    for (auto& e : expect)
        if (std::strcmp(backend_name(e.type), e.name) != 0) all = false;
    CHECK(all, "backend_name matches legacy strings for every enumerator");
    CHECK(std::strcmp(backend_name(static_cast<BackendType>(250)), "none") == 0,
          "backend_name out-of-range falls back to \"none\"");
}

static void check_pilot_names() {
    CHECK(std::strcmp(sublayer_name(SubLayer::ATTENTION_Q), "attn_q") == 0 &&
              std::strcmp(sublayer_name(SubLayer::ROUTER), "router") == 0 &&
              std::strcmp(sublayer_name(SubLayer::FFN_DOWN), "ffn_down") == 0,
          "sublayer_name matches legacy strings");
    CHECK(std::strcmp(pilot_backend_name(PilotBackend::UNKNOWN), "?") == 0 &&
              std::strcmp(pilot_backend_name(PilotBackend::NPU), "NPU") == 0 &&
              std::strcmp(pilot_backend_name(PilotBackend::CPU), "CPU") == 0,
          "pilot_backend_name matches legacy strings");
}

static void check_tier_names() {
    CHECK(std::strcmp(tier_name(BackendTier::T1_ACCELERATOR), "NPU/Accelerator") == 0 &&
              std::strcmp(tier_name(BackendTier::T2_GPU), "GPU") == 0 &&
              std::strcmp(tier_name(BackendTier::T3_CPU), "CPU") == 0,
          "tier_name matches legacy strings");
    CHECK(std::strcmp(tier_name(static_cast<BackendTier>(99)), "unknown") == 0,
          "tier_name out-of-range falls back to \"unknown\"");
}

// ---------------------------------------------------------------------------
// 2. P2996: print the BackendType mapping derived straight from the enum.
//    Compiles only under g++-16 -freflection (RCPP26_HAS_REFLECTION).
// ---------------------------------------------------------------------------
#if RCPP26_HAS_REFLECTION
static void print_reflected_backend_names() {
    std::printf("-- BackendType mapping via std::meta::enumerators_of --\n");
    constexpr static auto kEnums =
        std::define_static_array(std::meta::enumerators_of(^^BackendType));
    template for (constexpr auto e : kEnums) {
        constexpr BackendType v = [:e:];
        std::printf("  %2d  %-16s -> %s\n", (int)v,
                    std::meta::identifier_of(e).data(), backend_name(v));
    }
}
#endif

// ---------------------------------------------------------------------------
// 3. std::inplace_vector (P0843, libstdc++ 16): fixed-capacity decode buffers
//    (token batch + per-layer KV header) — the issue's hot-path example.
//    Note: __cpp_lib_inplace_vector materializes when <inplace_vector> (or
//    <version>) is included — test __has_include first, then include, then the
//    macro is defined (libstdc++ per-header want-flag mechanism).
// ---------------------------------------------------------------------------
#if __has_include(<inplace_vector>)
#include <inplace_vector>
#define RCPP26_HAS_INPLACE_VECTOR 1
#else
#define RCPP26_HAS_INPLACE_VECTOR 0
#endif

static constexpr long kLibInplaceVector =
#if RCPP26_HAS_INPLACE_VECTOR
    (long)__cpp_lib_inplace_vector;
#else
    0;
#endif

#if RCPP26_HAS_INPLACE_VECTOR
static void check_inplace_vector() {
    // Token batch for one decode step (bounded by n_batch; fixed 8 here).
    std::inplace_vector<uint32_t, 8> batch;
    for (uint32_t i = 0; i < 6; ++i) batch.push_back(1000 + i);
    // KV cache header: per-layer scratch [n_layers ≤ 64]; emulate layer 0..3.
    std::inplace_vector<std::pair<uint32_t, uint32_t>, 64> kv_headers;
    for (uint32_t l = 0; l < 4; ++l) kv_headers.emplace_back(l, 128 * (l + 1));

    bool ok = batch.size() == 6 && kv_headers.size() == 4 &&
              batch[5] == 1005 && kv_headers[3].second == 512;
    CHECK(ok, "std::inplace_vector decode buffers (no heap alloc)");
    std::printf("   batch.size=%zu cap=%zu kv_headers.size=%zu cap=%zu\n",
                (size_t)batch.size(), (size_t)batch.capacity(),
                (size_t)kv_headers.size(), (size_t)kv_headers.capacity());
}
#endif

int main() {
    std::printf("== cpp26 name-maps selfcheck (compiler: %s) ==\n", __VERSION__);
    std::printf("RCPP26_HAS_REFLECTION=%d __cpp_lib_inplace_vector=%ld\n",
                (int)RCPP26_HAS_REFLECTION, kLibInplaceVector);
    check_backend_names();
    check_pilot_names();
    check_tier_names();
#if RCPP26_HAS_REFLECTION
    print_reflected_backend_names();
#endif
#if RCPP26_HAS_INPLACE_VECTOR
    check_inplace_vector();
#else
    std::printf("skip std::inplace_vector (libstdc++ < 16 / no <inplace_vector>)\n");
#endif
    if (g_fail == 0) std::printf("CPP26-NAME-MAPS: ALL PASSED\n");
    else            std::printf("CPP26-NAME-MAPS: %d FAILED\n", g_fail);
    return g_fail ? 1 : 0;
}
