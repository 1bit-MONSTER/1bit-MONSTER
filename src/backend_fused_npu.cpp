// backend_fused_npu.cpp — NPU FFN caller (pure C++, not HIP).
// Compiled as CXX to avoid HIP compiler context conflicts with XRT.
#include "backend_fused_npu.h"
#include "npu_device_path.h"
#include "../engine/fusion/zero_copy/npu_gemm_kernel.h"
#include <cmath>
#include <vector>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

struct NpuState {
    xrt::device dev{0};
    std::unique_ptr<fusion::NpuGemmKernel> gu, d;
    // Per-layer packed int8 B buffers + scales.  The kernels expose ONE shared
    // bB BO, so each layer's weights get their own BO here, packed once at
    // load; npu_state_ffn hands the layer's BO to goB().  (Packing every layer
    // into the shared bB — the old design — left the LAST packed layer's
    // weights in bB for every FFN call, collapsing the hidden state: #1207.)
    std::vector<std::unique_ptr<xrt::bo>> gu_b, d_b;
    std::vector<float> gu_scale, d_scale;  // per-layer scales
    std::vector<std::vector<float>> ffn_norm;  // per-layer FFN RMSNorm weights (H)
    // Scratch buffers (sized to xm rows at create): the per-call std::vector
    // allocations in npu_state_ffn(_batch) cost ~ms-level malloc/mmap churn
    // per layer (the gu scratch is 768 KB at am=32 → 2 mmap syscalls/call).
    // Reused across calls — only the active am rows are touched.
    std::vector<float> hnorm, gu_s, silu_in, ffn_out;
    int H = 0, IM = 0, NC = 0;
    int xm = 0;   // AIE tile row count (32 = m32 full grid, 8 = m8, 1 = m1, 128 = default)
    bool ok = false;
};

NpuState* npu_state_create(const char* xclbin_dir, int H, int IM, int NC) {
    int npu_fd = open(npu_device_path(), O_RDONLY);
    if (npu_fd < 0) return nullptr;
    close(npu_fd);

    auto* s = new NpuState();
    s->H = H; s->IM = IM; s->NC = NC;
    try { s->dev = xrt::device(0); } catch (...) { delete s; return nullptr; }

    auto find_xclbin = [&](const char* tag) -> std::tuple<std::string,std::string,int> {
        std::string xd(xclbin_dir ? xclbin_dir : "engine/npu/xclbins");
        std::string b = xd + "/final_i8_" + tag;
        std::string i = xd + "/insts_i8_" + tag;
        // Preference order (all silicon-verified, oracle bit-identical):
        // 1. _m32 (n1_core_i8_v27.py -M 32 -r 4 -c 8, FULL 32-core grid):
        //    multi-sequence-decode family — one B DMA serves 32 rows (GU
        //    2.26 ms for 32 rows vs 1.93 ms for 8 on m8 = 3.4x cheaper per
        //    row, D 0.95 ms for 32 rows vs 0.94 ms for 8 = 4.0x).  Preferred
        //    when FUSED_BATCH > 8 (batched decode); m8 stays the default for
        //    single/small-batch because its launch is ~12% cheaper.
        // 2. _m8 (n1_core_i8_v27.py -M 8 + M8_VECTORIZED mmul): the fastest
        //    single-stream launch — 2.06 ms for the 6.3 MB GU B (3.1 GB/s)
        //    vs 4.32 ms for the m1 scalar stream (the scalar matmul throttles
        //    the DMA to ~1 GB/s) and 2.76 ms for the M=128-baked stream.
        //    am=8 also amortizes the B DMA across 8 sequences (2045 us total).
        // 3. _m1 (n1_core_i8_m1.py, DIM_M=1 scalar): correct but compute-bound.
        // 4. default (M=128-baked FLM-parity streams).
        const char* fb = getenv("FUSED_BATCH");
        bool want_m32 = fb && atoi(fb) > 8;
        std::string m32b = b + "_qwen3_0_6b_m32.xclbin", m32i = i + "_qwen3_0_6b_m32.txt";
        if (want_m32 && access(m32b.c_str(), F_OK) == 0 && access(m32i.c_str(), F_OK) == 0)
            return {m32b, m32i, 32};
        std::string m8b = b + "_qwen3_0_6b_m8.xclbin", m8i = i + "_qwen3_0_6b_m8.txt";
        if (access(m8b.c_str(), F_OK) == 0 && access(m8i.c_str(), F_OK) == 0) return {m8b, m8i, 8};
        std::string m1b = b + "_qwen3_0_6b_m1.xclbin", m1i = i + "_qwen3_0_6b_m1.txt";
        if (access(m1b.c_str(), F_OK) == 0 && access(m1i.c_str(), F_OK) == 0) return {m1b, m1i, 1};
        std::string ms = b + "_qwen3_0_6b.xclbin", mi = i + "_qwen3_0_6b.txt";
        if (access(ms.c_str(), F_OK) == 0 && access(mi.c_str(), F_OK) == 0) return {ms, mi, 128};
        std::string ts = b + "_v.xclbin", ti = i + "_v.txt";
        if (access(ts.c_str(), F_OK) == 0 && access(ti.c_str(), F_OK) == 0) return {ts, ti, 128};
        return {"", "", 0};
    };

    auto [xgu, igu, xm] = find_xclbin("GU");
    auto [xdd, idd, xmd] = find_xclbin("D");
    // Both kernels must come from the same family (both m1 or both 128-row);
    // a mixed/partial install is broken — fail and let the caller run GPU-only.
    if (xgu.empty() || xdd.empty() || xm != xmd) { delete s; return nullptr; }
    s->xm = xm;
    // Scratch buffers sized once for the full tile width (am <= xm always).
    s->hnorm.assign((size_t)xm * H, 0.0f);
    s->gu_s.assign((size_t)xm * 2 * IM, 0.0f);
    s->silu_in.assign((size_t)xm * IM, 0.0f);
    s->ffn_out.assign((size_t)xm * H, 0.0f);

    s->gu = std::make_unique<fusion::NpuGemmKernel>();
    s->d  = std::make_unique<fusion::NpuGemmKernel>();
    // The precompiled per-model xclbins (final_i8_{GU,D}_qwen3_0_6b) expect a
    // FIXED 128-row AIE tile buffer regardless of the active row count (the
    // passing reference — engine/fusion/zero_copy/test_npu_ffn_real_weights.cpp
    // and test_pipeline_real.cpp — uses XM=128).  A smaller MD reads/writes the
    // wrong tile region of bA/bC and the FFN output comes back garbage — the
    // #1207 all-zeros symptom (hidden state collapses to zeros).  The _m1 and
    // _m8 families are the exceptions: they are BUILT for those tile widths
    // (MD = xm) with bit-identical integer math.
    printf("[npu] FFN xclbins: %s (XM=%d, %s)\n", xgu.c_str(), xm,
           xm == 32 ? "full 32-core grid M=32 decode" :
           xm == 8 ? "vectorized M=8 decode" :
           xm == 1 ? "true single-row decode" : "fixed 128-row tile");
    if (!s->gu->init(s->dev, xgu.c_str(), igu.c_str(), xm, H, 2*IM) ||
        !s->d->init(s->dev, xdd.c_str(), idd.c_str(), xm, IM, H)) {
        delete s; return nullptr;
    }

    s->ok = true;
    return s;
}

void npu_state_destroy(NpuState* s) {
    delete s;
}

void npu_state_pack_layer(NpuState* s, int layer,
                           const float* w1, const float* w2, const float* w3,
                           const float* ffn_norm_w) {
    if (!s || !s->ok) return;
    if ((int)s->gu_scale.size() <= layer) {
        s->gu_scale.resize(layer + 1);
        s->d_scale.resize(layer + 1);
        s->gu_b.resize(layer + 1);
        s->d_b.resize(layer + 1);
        s->ffn_norm.resize(layer + 1);
    }
    int H = s->H, IM = s->IM;

    // Per-layer FFN RMSNorm weights (the GPU FFN normalizes its input with
    // gl.pon; without it the NPU FFN amplifies unboundedly — the #1207
    // divergence. NULL/absent norm is allowed (treat as identity).
    if (ffn_norm_w) s->ffn_norm[layer].assign(ffn_norm_w, ffn_norm_w + H);
    else            s->ffn_norm[layer].clear();

    // Transpose GGUF [out,in] → packB's [in,out]
    std::vector<float> gu((size_t)H * 2 * IM);
    for (int k = 0; k < H; k++) for (int n = 0; n < IM; n++) {
        gu[(size_t)k*(2*IM)+n]       = w1[(size_t)n*H + k];
        gu[(size_t)k*(2*IM)+IM+n]    = w2[(size_t)n*H + k];
    }
    std::vector<float> dw((size_t)IM * H);
    for (int k = 0; k < IM; k++) for (int n = 0; n < H; n++)
        dw[(size_t)k*H + n] = w3[(size_t)n*IM + k];

    // Each layer packs into its OWN B buffer (same size/group as the kernel's
    // default bB).  Allocating lazily here keeps npu_state_create unchanged.
    if (!s->gu_b[layer])
        s->gu_b[layer] = std::make_unique<xrt::bo>(s->dev, (size_t)H * 2 * IM,
                                                   XRT_BO_FLAGS_HOST_ONLY, s->gu->k->group_id(4));
    if (!s->d_b[layer])
        s->d_b[layer] = std::make_unique<xrt::bo>(s->dev, (size_t)IM * H,
                                                  XRT_BO_FLAGS_HOST_ONLY, s->d->k->group_id(4));
    // Row-major [K,N] for both families (the m1 descriptor gathers 8x8 blocks
    // from row-major; a microtiled source was measured no faster — the
    // single-launch DMA path is ~1.4 GB/s regardless of source layout).
    s->gu->packB_into(*s->gu_b[layer], gu.data(), H, 2*IM, s->gu_scale[layer]);
    s->d->packB_into(*s->d_b[layer], dw.data(), IM, H, s->d_scale[layer]);
}

bool npu_state_ffn(NpuState* s, int layer, float* h, int H) {
    if (!s || !s->ok || layer >= (int)s->gu_scale.size()) return false;
    int IM = s->IM;

    try {
        if (layer >= (int)s->gu_b.size() || !s->gu_b[layer] || !s->d_b[layer]) return false;

        // FFN RMSNorm on a COPY of the input — h (the slot) stays the raw
        // residual that ffn_out is added back to, matching the GPU FFN:
        //   dffn = h; h = rmsnorm(h, pon); ...; h = down + dffn
        // (see FusedBackend::forward FFN section).  Without the norm the
        // int8 GEMMs amplify the unnormalized input layer over layer and the
        // hidden state diverges (the #1207 explosion).
        std::vector<float> hnorm(H);
        const auto& nw = s->ffn_norm[layer];
        if (!nw.empty()) {
            double ss = 0;
            for (int i = 0; i < H; i++) ss += (double)h[i] * h[i];
            float inv = 1.0f / sqrtf((float)(ss / H) + 1e-6f);
            for (int i = 0; i < H; i++) hnorm[i] = h[i] * inv * nw[i];
        } else {
            memcpy(hnorm.data(), h, (size_t)H * sizeof(float));
        }

        float ascale = 0;
        for (int i = 0; i < H; i++) { float a = fabsf(hnorm[i]); if (a > ascale) ascale = a; }
        ascale = (ascale < 1e-12f) ? 1.0f : ascale / 127.0f;

        float* gu = s->gu_s.data();
        s->gu->goB(hnorm.data(), 1, H, ascale, s->gu_scale[layer], gu, 2*IM, *s->gu_b[layer]);
        for (int i = 0; i < IM; i++)
            gu[i] = (gu[i] / (1.0f + expf(-gu[i]))) * gu[IM+i];

        float dscale = 0;
        for (int i = 0; i < IM; i++) { float a = fabsf(gu[i]); if (a > dscale) dscale = a; }
        dscale = (dscale < 1e-12f) ? 1.0f : dscale / 127.0f;

        float* ffn_out = s->ffn_out.data();
        s->d->goB(gu, 1, IM, dscale, s->d_scale[layer], ffn_out, H, *s->d_b[layer]);
        for (int i = 0; i < H; i++) h[i] += ffn_out[i];
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[npu_ffn] l=%d exception: %s\n", layer, e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[npu_ffn] l=%d unknown exception\n", layer);
        return false;
    }
}

bool npu_state_ffn_batch(NpuState* s, int layer, float* h, int H, int am) {
    if (!s || !s->ok || layer >= (int)s->gu_scale.size() || am <= 0) return false;
    if (am > s->xm) {
        fprintf(stderr, "[npu_ffn] batch am=%d exceeds tile width %d\n", am, s->xm);
        return false;
    }
    int IM = s->IM;
    try {
        if (layer >= (int)s->gu_b.size() || !s->gu_b[layer] || !s->d_b[layer]) return false;

        // Per-row FFN RMSNorm + activation scale (each sequence has its own
        // dynamic range — the batched kernel quantizes per row).  Scratch
        // buffers come from NpuState (sized to xm rows at create).
        float* hnorm = s->hnorm.data();
        std::vector<float> ascales(am), dscales(am);
        const auto& nw = s->ffn_norm[layer];
        for (int m = 0; m < am; m++) {
            const float* hm = h + (size_t)m * H;
            float* hn = hnorm + (size_t)m * H;
            if (!nw.empty()) {
                double ss = 0;
                for (int i = 0; i < H; i++) ss += (double)hm[i] * hm[i];
                float inv = 1.0f / sqrtf((float)(ss / H) + 1e-6f);
                for (int i = 0; i < H; i++) hn[i] = hm[i] * inv * nw[i];
            } else {
                memcpy(hn, hm, (size_t)H * sizeof(float));
            }
            float a = 0;
            for (int i = 0; i < H; i++) { float f = fabsf(hn[i]); if (f > a) a = f; }
            ascales[m] = (a < 1e-12f) ? 1.0f : a / 127.0f;
        }

        float* gu = s->gu_s.data();
        s->gu->goB_rows(hnorm, am, H, ascales.data(), s->gu_scale[layer], gu, 2*IM, *s->gu_b[layer]);
        for (int m = 0; m < am; m++)
            for (int i = 0; i < IM; i++) {
                float g = gu[(size_t)m * 2 * IM + i], u = gu[(size_t)m * 2 * IM + IM + i];
                gu[(size_t)m * 2 * IM + i] = (g / (1.0f + expf(-g))) * u;
            }
        for (int m = 0; m < am; m++) {
            float a = 0;
            for (int i = 0; i < IM; i++) { float f = fabsf(gu[(size_t)m * 2 * IM + i]); if (f > a) a = f; }
            dscales[m] = (a < 1e-12f) ? 1.0f : a / 127.0f;
        }
        // The D kernel's A is [am, IM] contiguous — compact the silu'd first
        // half of each gu row (gu is [am, 2*IM]; goB_rows would read row m at
        // m*IM, i.e. the wrong stride, without this).
        float* silu_in = s->silu_in.data();
        for (int m = 0; m < am; m++)
            memcpy(silu_in + (size_t)m * IM, gu + (size_t)m * 2 * IM,
                   (size_t)IM * sizeof(float));

        float* ffn_out = s->ffn_out.data();
        s->d->goB_rows(silu_in, am, IM, dscales.data(), s->d_scale[layer], ffn_out, H, *s->d_b[layer]);
        for (int m = 0; m < am; m++)
            for (int i = 0; i < H; i++) h[(size_t)m * H + i] += ffn_out[(size_t)m * H + i];
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[npu_ffn] batch l=%d exception: %s\n", layer, e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[npu_ffn] batch l=%d unknown exception\n", layer);
        return false;
    }
}
