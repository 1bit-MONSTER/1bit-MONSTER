// tests/test_vulkan_prism.cpp — Vulkan GEMV for the Prism flat-128-block layouts through our own
// kernels/vulkan/dmmv_prism.comp, run on the minimal src/vulkan_rt.h runtime.
//
// Why this exists (P5): the objective's third honest tok/s column is Vulkan/ZINC. The ZINC binary
// on this box loads a Prism pack and then segfaults uploading a 13.9 MB tensor to a stack address
// (zinc/src/model/loader.zig:666), and there is no zig and no source to rebuild it. This path uses
// OUR shader and OUR runtime instead, so the column does not depend on a third party.
//
// Same contract as test_vulkan_gemv.cpp: standalone, no model loading, no HIP device pointers.
// It proves the Prism block decoding in the shader matches a CPU dequant+dot reference, and gives
// a throughput figure in GB/s for the weight stream.
//
// Prism block, 128 weights each, flat row-major (the packs are NOT the 32x256 tile grid):
//   Q1_0   (18 B): [f16 d][16 B sign bits]      value = bit ? +d : -d
//   PQ2_0  (34 B): [f16 d][32 B 2-bit codes]    value = code*d - d   (code 3 never emitted)
//   PTQ1_0 (28 B): [24 B qs][2 B qh][f16 d]     base-3, non-positional order
// This tool covers the two the shader declares with a plain block size (Q1_0, PQ2_0) and reports
// PTQ1_0 as not-covered rather than pretending.
//
// Build:  g++ -O2 -std=c++17 -DVK_SHADER_DIR=\"/path/to/spv\" tests/test_vulkan_prism.cpp -lvulkan
// Run:    ./a.out            (expects dmmv_prism.spv in VK_SHADER_DIR)

#include "../src/vulkan_rt.h"
#include "../include/onebp_loader.h"   // folded-pack detection (fail closed)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct PushConstants {
    uint32_t M, K, a_offset, x_offset, y_offset, acc_mode;
};
static const uint32_t kRowsPerWg = 2;   // matches local_size_x=64, 2 rows/workgroup

#ifndef VK_SHADER_DIR
#define VK_SHADER_DIR "."
#endif

static uint16_t f32ToF16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
static float f16ToF32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) bits = sign;
    else bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ── Q1_0: 18 B per 128 weights ────────────────────────────────────────────────────────────
static const uint32_t kQ1BlockWeights = 128;
static const uint32_t kQ1BlockBytes = 18;

static void fillQ1(std::vector<uint8_t>& w, uint32_t m, uint32_t k) {
    const uint32_t nb = k / kQ1BlockWeights;
    const size_t row_bytes = (size_t)nb * kQ1BlockBytes;
    w.assign((size_t)m * row_bytes, 0);
    uint32_t s = 12345;
    for (uint32_t r = 0; r < m; r++) {
        for (uint32_t b = 0; b < nb; b++) {
            uint8_t* blk = w.data() + (size_t)r * row_bytes + (size_t)b * kQ1BlockBytes;
            const uint16_t d = f32ToF16(0.01f + (float)(r % 7) * 0.003f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; i++) { s = s * 1103515245u + 12345u; blk[2 + i] = (uint8_t)(s >> 24); }
        }
    }
}
static void refQ1(const uint8_t* w, const float* x, float* y, uint32_t m, uint32_t k) {
    const uint32_t nb = k / kQ1BlockWeights;
    for (uint32_t r = 0; r < m; r++) {
        const uint8_t* row = w + (size_t)r * nb * kQ1BlockBytes;
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * kQ1BlockBytes;
            uint16_t d16;
            std::memcpy(&d16, blk, 2);
            const float d = f16ToF32(d16);
            for (int i = 0; i < 128; i++) {
                const uint8_t byte = blk[2 + (i >> 3)];
                const int bit = (byte >> (i & 7)) & 1;
                acc += (bit ? d : -d) * x[b * kQ1BlockWeights + i];
            }
        }
        y[r] = acc;
    }
}

// ── PQ2_0: 34 B per 128 weights ───────────────────────────────────────────────────────────
static const uint32_t kQ2BlockBytes = 34;
static void fillQ2(std::vector<uint8_t>& w, uint32_t m, uint32_t k) {
    const uint32_t nb = k / 128;
    const size_t row_bytes = (size_t)nb * kQ2BlockBytes;
    w.assign((size_t)m * row_bytes, 0);
    uint32_t s = 999;
    for (uint32_t r = 0; r < m; r++)
        for (uint32_t b = 0; b < nb; b++) {
            uint8_t* blk = w.data() + (size_t)r * row_bytes + (size_t)b * kQ2BlockBytes;
            const uint16_t d = f32ToF16(0.008f + (float)(r % 5) * 0.002f);
            std::memcpy(blk, &d, 2);
            // Codes must be 0..2: code 3 is never emitted by the real encoder and the codec
            // maps it to 0, so random bytes would test a case the format cannot produce.
            for (int i = 0; i < 32; i++) {
                s = s * 1103515245u + 12345u;
                uint8_t b = 0;
                for (int c = 0; c < 4; c++) { s = s * 1103515245u + 12345u; b |= (uint8_t)(((s >> 24) % 3u) << (2 * c)); }
                blk[2 + i] = b;
            }
        }
}
static void refQ2(const uint8_t* w, const float* x, float* y, uint32_t m, uint32_t k) {
    const uint32_t nb = k / 128;
    for (uint32_t r = 0; r < m; r++) {
        const uint8_t* row = w + (size_t)r * nb * kQ2BlockBytes;
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * kQ2BlockBytes;
            uint16_t d16;
            std::memcpy(&d16, blk, 2);
            const float d = f16ToF32(d16);
            for (int i = 0; i < 128; i++) {
                const uint8_t byte = blk[2 + (i >> 2)];
                const int code = (byte >> (2 * (i & 3))) & 3;
                acc += ((code == 3) ? 0.0f : ((float)code * d - d)) * x[b * 128 + i];
            }
        }
        y[r] = acc;
    }
}


// ── PTQ1_0: 28 B per 128 weights, base-3 trits, element order NOT positional ──────────────
static const uint32_t kPtqBlockBytes = 28;
static void fillPTQ(std::vector<uint8_t>& w, uint32_t m, uint32_t k) {
    const uint32_t nb = k / 128;
    const size_t row_bytes = (size_t)nb * kPtqBlockBytes;
    w.assign((size_t)m * row_bytes, 0);
    uint32_t s = 4242;
    for (uint32_t r = 0; r < m; r++)
        for (uint32_t b = 0; b < nb; b++) {
            uint8_t* blk = w.data() + (size_t)r * row_bytes + (size_t)b * kPtqBlockBytes;
            for (int i = 0; i < 24 + 2; i++) { s = s * 1103515245u + 12345u; blk[i] = (uint8_t)(s >> 24); }
            const uint16_t d = f32ToF16(0.006f + (float)(r % 6) * 0.001f);
            std::memcpy(blk + 26, &d, 2);
        }
}
// follows include/prism_codec.h case 13 exactly (the base-3 chain and the (byte,n) mapping)
static void refPTQ(const uint8_t* w, const float* x, float* y, uint32_t m, uint32_t k) {
    const uint32_t nb = k / 128;
    for (uint32_t r = 0; r < m; r++) {
        const uint8_t* row = w + (size_t)r * nb * kPtqBlockBytes;
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * kPtqBlockBytes;
            uint16_t d16;
            std::memcpy(&d16, blk + 26, 2);
            const float d = f16ToF32(d16);
            for (int e = 0; e < 128; e++) {
                uint8_t bb;
                int n;
                if (e < 80)       { bb = blk[e & 15];              n = e >> 4; }
                else if (e < 120) { const int t = e - 80;  bb = blk[16 + (t & 7)]; n = t >> 3; }
                else              { const int t = e - 120; bb = blk[24 + (t & 1)]; n = t >> 1; }
                uint32_t v = bb;
                for (int kk = 0; kk < n; kk++) v = (v * 3u) & 0xFFu;
                const int trit = (int)((v * 3u) >> 8);
                acc += (float)(trit - 1) * d * x[b * 128 + e];
            }
        }
        y[r] = acc;
    }
}

// ── runner ────────────────────────────────────────────────────────────────────────────────
static int runFormat(vkrt::VkCtx& ctx, const char* spv, const char* name,
                     std::vector<uint8_t>& weights, uint32_t m, uint32_t k,
                     uint32_t block_bytes, uint32_t nb, uint32_t q, const std::vector<float>& x,
                     void (*ref)(const uint8_t*, const float*, float*, uint32_t, uint32_t)) {
    std::vector<float> y_ref((size_t)m), y_got((size_t)m);
    ref(weights.data(), x.data(), y_ref.data(), m, k);

    vkrt::Pipeline pipeline;
    const uint32_t spec[4] = {m, k, nb, q};   // constant_id 0..3 = SPEC_M, SPEC_K, SPEC_NB, SPEC_Q
    pipeline.create(ctx, spv, 3, sizeof(PushConstants), spec, 4);

    vkrt::GpuBuffer bufW, bufX, bufY;
    bufW.create(ctx.dev, ctx.memProps, weights.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufX.create(ctx.dev, ctx.memProps, x.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufY.create(ctx.dev, ctx.memProps, (size_t)m * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufW.upload(weights.data());
    bufX.upload(x.data());

    vkrt::GpuBuffer* bufs[3] = {&bufW, &bufX, &bufY};
    VkDescriptorSet ds = vkrt::createDescriptorSet(ctx, pipeline, bufs, 3);

    PushConstants pc{m, k, 0, 0, 0, 0};
    uint32_t wg_x = (m + kRowsPerWg - 1) / kRowsPerWg;
    vkrt::dispatchRepeatedTimed(ctx, pipeline, ds, wg_x, 1, 1, &pc, 5);
    bufY.download(y_got.data());
    if (ctx.dev) vkDeviceWaitIdle(ctx.dev);

    double max_abs = 0, max_rel = 0;
    for (uint32_t r = 0; r < m; r++) {
        const double e = std::fabs((double)y_got[r] - (double)y_ref[r]);
        max_abs = std::max(max_abs, e);
        const double den = std::fabs((double)y_ref[r]) + 1e-6;
        max_rel = std::max(max_rel, e / den);
    }
    const bool ok = max_abs < 1e-3 * (double)k / 512.0 + 1e-3;
    printf("  %s correctness %s: M=%u K=%u max_abs_err=%.6f max_rel=%.2e\n",
           name, ok ? "PASSED" : "FAILED", m, k, max_abs, max_rel);

    const uint32_t warmup = 25, iterations = 200;
    vkrt::dispatchRepeatedTimed(ctx, pipeline, ds, wg_x, 1, 1, &pc, warmup);
    const double ms_total = vkrt::dispatchRepeatedTimed(ctx, pipeline, ds, wg_x, 1, 1, &pc, iterations);
    const double ms = ms_total / iterations;
    const double bytes = (double)((size_t)m * (k / 128) * block_bytes) + (double)k * 4.0 + (double)m * 4.0;
    printf("  %s M=%u K=%u: %.4f ms/iter | %.1f GB/s (warmup=%u, iters=%u)\n",
           name, m, k, ms, bytes / (ms / 1000.0) / 1e9, warmup, iterations);

    vkFreeDescriptorSets(ctx.dev, ctx.dpool, 1, &ds);
    pipeline.destroy(ctx.dev);
    bufW.destroy(); bufX.destroy(); bufY.destroy();
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    // FAIL CLOSED on a folded pack, before touching the device. This harness applies no folded
    // basis: it never rotates the activation by the Hadamard manifest and never applies the
    // ssm_out head permutation, both of which belong to the FOLDED basis only. Running it on a
    // folded pack's weights as if they were plain would produce plausible-looking garbage, which
    // is the exact failure mode this lane exists to avoid. The HIP engine refuses the same case
    // (include/prism_engine.h: a transform manifest it cannot honour -> return false).
    // Uses the repo's own loader rather than a byte scan, so the check is the same one P1 owns.
    if (argc > 1) {
        OnebpModel pack;
        if (!pack.load(argv[1])) {
            std::fprintf(stderr, "REFUSING: cannot load %s as a 1BP container\n", argv[1]);
            return 2;
        }
        bool folded = false;
        for (const auto& t : pack.tensors)
            if (t.name == "__onebp_ext_prism_transform") folded = true;
        if (folded) {
            std::fprintf(stderr,
                "REFUSING %s: this is a FOLDED Prism pack (it carries __onebp_ext_prism_transform).\n"
                "  This Vulkan harness does not apply the folded Hadamard basis or the ssm_out head\n"
                "  permutation, so its weights are NOT interchangeable with plain ones. Serving them\n"
                "  as plain would silently produce plausible garbage. Failing closed.\n", argv[1]);
            return 3;
        }
        std::printf("pack %s is UNFOLDED (no __onebp_ext_prism_transform): safe to treat as plain\n", argv[1]);
    }

    const std::string shader_dir = VK_SHADER_DIR;
    const std::string spv = shader_dir + "/dmmv_prism.spv";

    vkrt::VkCtx ctx;
    ctx.init();
    if (!ctx.dev) { printf("no Vulkan device\n"); return 2; }

    printf("\n== Prism flat-128-block Vulkan GEMV (dmmv_prism.comp) ==\n");
    printf("device: %s\n", ctx.deviceName);

    int fails = 0;
    {   // small: matches the shader's default specialization constants (M=K=4096 is the id
        // default; the push constants carry the real M/K, so a small case is enough to check math)
        const uint32_t m = 16, k = 256;
        std::vector<float> x((size_t)k);
        for (uint32_t i = 0; i < k; i++) x[i] = std::sin((double)i * 0.017) * 0.7f;
        std::vector<uint8_t> w;
        fillQ1(w, m, k);
        fails += runFormat(ctx, spv.c_str(), "Q1_0", w, m, k, kQ1BlockBytes, 18, 11, x, refQ1);
        fillQ2(w, m, k);
        fails += runFormat(ctx, spv.c_str(), "PQ2_0", w, m, k, kQ2BlockBytes, 34, 12, x, refQ2);
        fillPTQ(w, m, k);
        fails += runFormat(ctx, spv.c_str(), "PTQ1_0", w, m, k, kPtqBlockBytes, 28, 13, x, refPTQ);
    }
    {   // throughput: the shape of the largest Prism GEMV in the model
        const uint32_t m = 17408, k = 5120;
        std::vector<float> x((size_t)k);
        for (uint32_t i = 0; i < k; i++) x[i] = std::sin((double)i * 0.013) * 0.5f;
        std::vector<uint8_t> w;
        printf("\n== throughput (fits the shader's default spec constants for block size) ==\n");
        fillQ1(w, m, k);
        fails += runFormat(ctx, spv.c_str(), "Q1_0", w, m, k, kQ1BlockBytes, 18, 11, x, refQ1);
        fillQ2(w, m, k);
        fails += runFormat(ctx, spv.c_str(), "PQ2_0", w, m, k, kQ2BlockBytes, 34, 12, x, refQ2);
        fillPTQ(w, m, k);
        fails += runFormat(ctx, spv.c_str(), "PTQ1_0", w, m, k, kPtqBlockBytes, 28, 13, x, refPTQ);
    }
    ctx.destroy();
    printf("\n%s\n", fails == 0 ? "PRISM VULKAN GEMV: ALL PASSED" : "PRISM VULKAN GEMV: FAILURES");
    return fails == 0 ? 0 : 1;
}
