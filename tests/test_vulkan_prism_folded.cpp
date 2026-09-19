// tests/test_vulkan_prism_folded.cpp — Vulkan folded-pack GEMV through our own shaders.
//
// This is the unlock that the GEMV-only harness (test_vulkan_prism.cpp) deliberately
// refused to do: it serves a FOLDED Prism pack (W' = W·H) correctly by applying the
// Hadamard activation transform on-device (kernels/vulkan/fwht.comp) before the GEMV
// (kernels/vulkan/dmmv_prism.comp). The two run through src/vulkan_rt.h, no third party.
//
// Correctness is a per-element gate against the repo's own CPU oracle
// (include/prism_codec.h): dequant the folded weights, hadamard_forward() the
// activation with the manifest signs, dot. The shader path must match that to the
// same tolerance the unfolded GEMV harness uses, on REAL packed weight bytes from a
// real folded pack — not synthetic weights.
//
// Build:
//   glslc -O -o /tmp/fwht.spv kernels/vulkan/fwht.comp
//   glslc -O -o /tmp/dmmv_prism.spv kernels/vulkan/dmmv_prism.comp
//   g++ -O2 -std=c++17 -I include -I src -DVK_SHADER_DIR=\"/tmp\" \
//       tests/test_vulkan_prism_folded.cpp src/onebp_model.cpp -lvulkan
// Run:  ./a.out <folded.1bp>

#include "../src/vulkan_rt.h"
#include "../include/onebp_loader.h"
#include "../include/prism_codec.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef VK_SHADER_DIR
#define VK_SHADER_DIR "."
#endif

// ── push constant layouts (must match the shaders) ─────────────────────────
struct GmvPC { uint32_t M, K, a_offset, x_offset, y_offset, acc_mode; };
struct FwhtPC { uint32_t width, x_off, s_off, has_signs; };

static const uint32_t kRowsPerWg = 2;   // dmmv_prism.comp: 2 rows/workgroup

static int fail(const char* msg) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <folded.1bp>\n", argv[0]); return 2; }

    // ── load the folded pack + manifest ────────────────────────────────────
    OnebpModel pack;
    if (!pack.load(argv[1])) return fail("cannot load 1BP");
    if (!pack.has_prism_transform()) return fail("not a folded pack (no __onebp_ext_prism_transform)");

    const OnebpTensor* tform = nullptr;
    const OnebpTensor* tsigns = nullptr;
    for (const auto& t : pack.tensors) {
        if (t.name == "__onebp_ext_prism_transform") tform = &t;
        if (t.name == "__onebp_ext_prism_signs")     tsigns = &t;
    }
    if (!tform || !tsigns) return fail("folded pack missing transform or signs entry");

    OnebpPrismTransformView view;
    if (!onebp_prism_transform_parse(pack.tensor_data(*tform), tform->bytes,
                                     (const int8_t*)pack.tensor_data(*tsigns), tsigns->bytes, view))
        return fail("transform manifest parse failed");
    const uint32_t block = view.hdr->block_size;
    std::printf("folded pack: block=%u sign_count=%u n_folded=%u n_inverse=%u\n",
                block, view.hdr->sign_count, view.hdr->n_folded, view.hdr->n_inverse);

    // ── pick a real folded weight tensor: first quant-13 tensor whose input
    //    width has a manifest sign vector, in folded-index order ────────────
    const OnebpTensor* wt = nullptr;
    for (uint32_t fi = 0; fi < view.hdr->n_folded && !wt; fi++) {
        uint32_t idx = view.folded[fi];
        if (idx >= pack.tensors.size()) continue;
        const auto& t = pack.tensors[idx];
        if (t.ndim != 2 || t.dims.size() < 2) continue;
        if (t.quant != 13) continue;                    // PTQ1_0 in the folded pack
        uint32_t K = t.dims[1];
        if (view.width_index(K) < 0) continue;          // needs a sign vector
        if ((K % block) != 0 || (K % 128) != 0) continue;
        wt = &t;
    }
    if (!wt) return fail("no folded quant-13 weight tensor with a manifest width");

    const uint32_t K = wt->dims[1];
    const int wi = view.width_index(K);
    const int8_t* signs_i8 = view.signs_for_width(K);
    const uint32_t nb = onebp_prism_block_bytes(wt->quant);   // 28
    const uint32_t blk_per_row = K / 128;
    // Cap rows so the CPU dequant reference stays quick; the GEMV shader reads
    // M rows from the buffer, so uploading the first M rows is sufficient.
    const uint32_t M = wt->dims[0] < 1024 ? wt->dims[0] : 1024;

    std::printf("tensor %s: M=%u (of %u) K=%u quant=%u nb=%u block=%u signs_width=%u\n",
                wt->name.c_str(), M, (uint32_t)wt->dims[0], K, wt->quant, nb, block, view.widths[wi].width);

    // ── activation + signs (host side) ─────────────────────────────────────
    std::vector<float> x((size_t)K), x_cpu((size_t)K);
    for (uint32_t i = 0; i < K; i++) x[i] = std::sin((double)i * 0.017) * 0.7f;
    x_cpu = x;

    std::vector<float> signs((size_t)K);
    for (uint32_t i = 0; i < K; i++) signs[i] = (float)signs_i8[i];

    // ── CPU oracle: y_ref = W' · fwht_forward(x) ────────────────────────────
    if (!prism::hadamard_forward(x_cpu.data(), (int)K, (int)block, signs_i8))
        return fail("cpu hadamard_forward");
    const uint8_t* wbytes = pack.tensor_data(*wt);
    const size_t row_bytes = (size_t)blk_per_row * nb;
    std::vector<float> y_ref((size_t)M, 0.f);
    {
        float tmp[128];
        for (uint32_t r = 0; r < M; r++) {
            const uint8_t* row = wbytes + (size_t)r * row_bytes;
            float acc = 0.f;
            for (uint32_t b = 0; b < blk_per_row; b++) {
                prism::dequant_block(wt->quant, row + (size_t)b * nb, tmp);
                const float* xb = x_cpu.data() + (size_t)b * 128;
                for (int e = 0; e < 128; e++) acc += tmp[e] * xb[e];
            }
            y_ref[r] = acc;
        }
    }

    // ── device: fwht.comp (forward) then dmmv_prism.comp ────────────────────
    vkrt::VkCtx ctx;
    ctx.init();
    if (!ctx.dev) return fail("no Vulkan device");
    std::printf("device: %s\n", ctx.deviceName);

    vkrt::GpuBuffer bufX, bufS, bufW, bufY;
    bufX.create(ctx.dev, ctx.memProps, x.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufS.create(ctx.dev, ctx.memProps, signs.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufW.create(ctx.dev, ctx.memProps, (size_t)M * row_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufY.create(ctx.dev, ctx.memProps, (size_t)M * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufX.upload(x.data());
    bufS.upload(signs.data());
    bufW.upload(wbytes);
    bufY.upload(y_ref.data());   // acc_mode=0 overwrites; content irrelevant

    // FWHT forward: grid = K/block workgroups.
    {
        vkrt::Pipeline fwht;
        const uint32_t spec[2] = { block, 0 };   // SPEC_BLOCK, SPEC_MODE(forward)
        fwht.create(ctx, (std::string(VK_SHADER_DIR) + "/fwht.spv").c_str(), 2, sizeof(FwhtPC), spec, 2);
        vkrt::GpuBuffer* bufs[2] = { &bufX, &bufS };
        VkDescriptorSet ds = vkrt::createDescriptorSet(ctx, fwht, bufs, 2);
        FwhtPC pc{ K, 0, 0, 1 };
        vkrt::dispatchOnce(ctx, fwht, ds, K / block, 1, 1, &pc);

        // Verify the FWHT shader against the CPU transform in isolation.
        std::vector<float> x_got((size_t)K);
        bufX.download(x_got.data());
        double max_abs = 0;
        for (uint32_t i = 0; i < K; i++) max_abs = std::max(max_abs, std::fabs((double)x_got[i] - (double)x_cpu[i]));
        std::printf("  fwht.comp forward vs CPU: max_abs_err=%.6f %s\n",
                    max_abs, max_abs < 1e-5 ? "PASSED" : "FAILED");
        if (max_abs >= 1e-5) return fail("fwht shader mismatch");
        vkFreeDescriptorSets(ctx.dev, ctx.dpool, 1, &ds);
        fwht.destroy(ctx.dev);
    }

    // GEMV on the folded weights, x now rotated in bufX.
    {
        vkrt::Pipeline gmv;
        const uint32_t spec[4] = { M, K, nb, wt->quant };   // SPEC_M/K/NB/Q
        gmv.create(ctx, (std::string(VK_SHADER_DIR) + "/dmmv_prism.spv").c_str(), 3, sizeof(GmvPC), spec, 4);
        vkrt::GpuBuffer* bufs[3] = { &bufW, &bufX, &bufY };
        VkDescriptorSet ds = vkrt::createDescriptorSet(ctx, gmv, bufs, 3);
        GmvPC pc{ M, K, 0, 0, 0, 0 };
        vkrt::dispatchOnce(ctx, gmv, ds, (M + kRowsPerWg - 1) / kRowsPerWg, 1, 1, &pc);

        std::vector<float> y_got((size_t)M);
        bufY.download(y_got.data());
        double max_abs = 0, max_rel = 0;
        for (uint32_t r = 0; r < M; r++) {
            double e = std::fabs((double)y_got[r] - (double)y_ref[r]);
            max_abs = std::max(max_abs, e);
            max_rel = std::max(max_rel, e / (std::fabs((double)y_ref[r]) + 1e-6));
        }
        const bool ok = max_abs < 1e-3 * (double)K / 512.0 + 1e-3;
        std::printf("  folded GEMV (fwht + dmmv) vs CPU: M=%u K=%u max_abs_err=%.6f max_rel=%.2e %s\n",
                    M, K, max_abs, max_rel, ok ? "PASSED" : "FAILED");
        vkFreeDescriptorSets(ctx.dev, ctx.dpool, 1, &ds);
        gmv.destroy(ctx.dev);
        if (!ok) return fail("folded GEMV mismatch");
    }

    bufX.destroy(); bufS.destroy(); bufW.destroy(); bufY.destroy();
    ctx.destroy();
    std::printf("\nFOLDED-PACK VULKAN GEMV: PASSED\n");
    return 0;
}
