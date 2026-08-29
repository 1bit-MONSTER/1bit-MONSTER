// test_npu_ffn_real_weights.cpp — NPU FFN correctness against REAL model weights.
//
// test_pipeline_real.cpp proves the GPU-attn/NPU-FFN pipeline plumbing works,
// but only with dummy constant (0.01f) weights and never checked the output
// against anything. This test replaces the dummy weights with the actual
// blk.0 FFN weights from models/Qwen3-0.6B.1bp (H=1024, L=28, IM=3072 — an
// exact shape match for the shipped final_i8_{GU,D}_qwen3_0_6b.xclbin) and
// compares the NPU's int8-quantized GEMM output against a float32 CPU
// reference computed from the same real, dequantized weights.
//
// CURRENT STATUS: PASSES (cosine ~0.998 vs the float32 CPU reference) after
// #1756's xclbin/insts rebuild + the #1207 fixes in the fused NPU path
// (src/backend_fused_npu.cpp):
//   - the final_i8_{GU,D}_qwen3_0_6b xclbins need MD=128 (a fixed 128-row AIE
//     tile), not MD=16;
//   - each layer's packed B must live in its OWN buffer — the kernels share
//     one bB, so packing every layer left the LAST layer's weights in bB for
//     every FFN call (see goB/packB_into);
//   - the FFN RMSNorm (gl.pon) must be applied to the FFN input — the GPU FFN
//     normalizes; without it the int8 GEMMs amplify the raw input and the
//     hidden state diverges to ±inf (all-zero tokens).
// The notes below document the pre-#1756 investigation (cosine ~0,
// alternating exact-zero output elements) for history.
//
// Usage: ./test_npu_ffn_real_weights [path/to/Qwen3-0.6B.1bp]

#include "npu_gemm_kernel.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>

// OnebpModel (with get_tensor_f32, TQ2+Q4NX dequant) — no ONEBP_LOADER_MAIN,
// so this pulls in just the class, no main().
#include "../../npu/src/onebp_loader.cpp"

static void rmsnorm(std::vector<float>& x, const std::vector<float>& w) {
    double ss = 0;
    for (float v : x) ss += (double)v * v;
    float inv = 1.0f / sqrtf((float)(ss / x.size()) + 1e-6f);
    for (size_t i = 0; i < x.size(); i++) x[i] = x[i] * inv * w[i];
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";

    int acc = open("/dev/accel/accel0", O_RDONLY);
    if (acc < 0) { fprintf(stderr, "No NPU — skip.\n"); return 77; }
    close(acc);

    NpuOnebpModel model;
    if (!model.open(model_path)) { fprintf(stderr, "Failed to open %s\n", model_path); return 1; }
    const auto& h = model.header();
    fprintf(stderr, "Model: %s  H=%d L=%d IM=%d\n", model_path, h.hidden_size, h.num_layers, h.intermediate_size);
    const int H = h.hidden_size, IM = h.intermediate_size;

    // ── Load real blk.0 FFN weights + norm + a real embedding row ──
    std::vector<float> gate_w, up_w, down_w, ffn_norm_w, embed_row;
    if (!model.get_tensor_f32("blk.0.ffn_gate.weight", gate_w)) { fprintf(stderr, "missing ffn_gate.weight\n"); return 1; }
    if (!model.get_tensor_f32("blk.0.ffn_up.weight", up_w)) { fprintf(stderr, "missing ffn_up.weight\n"); return 1; }
    if (!model.get_tensor_f32("blk.0.ffn_down.weight", down_w)) { fprintf(stderr, "missing ffn_down.weight\n"); return 1; }
    if (!model.get_tensor_f32("blk.0.ffn_norm.weight", ffn_norm_w)) { fprintf(stderr, "missing ffn_norm.weight\n"); return 1; }
    if ((int)gate_w.size() != IM * H || (int)up_w.size() != IM * H || (int)down_w.size() != H * IM) {
        fprintf(stderr, "unexpected tensor sizes: gate=%zu up=%zu down=%zu (want IM*H=%d, H*IM=%d)\n",
                gate_w.size(), up_w.size(), down_w.size(), IM * H, H * IM);
        return 1;
    }

    // Real input activation: token_embd.weight row for a real token id, RMSNorm'd
    // with the real ffn_norm weight — a realistic FFN-input distribution.
    std::vector<float> token_embd;
    if (!model.get_tensor_f32("token_embd.weight", token_embd)) { fprintf(stderr, "missing token_embd.weight\n"); return 1; }
    const int token_id = 100;
    std::vector<float> x(token_embd.begin() + (size_t)token_id * H, token_embd.begin() + (size_t)(token_id + 1) * H);
    rmsnorm(x, ffn_norm_w);

    // ── CPU float32 reference: standard nn.Linear forward on the ORIGINAL
    // [out,in] layout, no transpose needed. ──
    std::vector<float> cpu_gate(IM), cpu_up(IM), cpu_silu(IM), cpu_final(H);
    for (int o = 0; o < IM; o++) {
        double s = 0; for (int k = 0; k < H; k++) s += (double)x[k] * gate_w[(size_t)o * H + k];
        cpu_gate[o] = (float)s;
    }
    for (int o = 0; o < IM; o++) {
        double s = 0; for (int k = 0; k < H; k++) s += (double)x[k] * up_w[(size_t)o * H + k];
        cpu_up[o] = (float)s;
    }
    for (int i = 0; i < IM; i++) {
        float g = cpu_gate[i];
        cpu_silu[i] = (g / (1.0f + expf(-g))) * cpu_up[i];
    }
    for (int o = 0; o < H; o++) {
        double s = 0; for (int k = 0; k < IM; k++) s += (double)cpu_silu[k] * down_w[(size_t)o * IM + k];
        cpu_final[o] = (float)s;
    }

    // ── NPU path: transpose real weights into packB's [K,N] convention ──
    std::vector<float> gu_kn((size_t)H * 2 * IM);   // [H, 2*IM]: gate cols then up cols
    for (int k = 0; k < H; k++) {
        for (int n = 0; n < IM; n++) {
            gu_kn[(size_t)k * (2 * IM) + n]      = gate_w[(size_t)n * H + k];
            gu_kn[(size_t)k * (2 * IM) + IM + n] = up_w[(size_t)n * H + k];
        }
    }
    std::vector<float> d_kn((size_t)IM * H);        // [IM, H]
    for (int k = 0; k < IM; k++)
        for (int n = 0; n < H; n++)
            d_kn[(size_t)k * H + n] = down_w[(size_t)n * IM + k];

    xrt::device npu(0);
    const char* xd = getenv("NPU_XCLBIN_DIR") ?: "engine/npu/xclbins";
    // xclbin selection: NPU_XCLBIN_SUFFIX picks the file family (default
    // "_qwen3_0_6b" = the fixed 128-row-tile xclbins; "_qwen3_0_6b_m1" = the
    // true M=1 single-row decode xclbins) and NPU_XCLBIN_XM overrides the
    // AIE tile row count passed to init() (128 for the M=128-baked xclbins,
    // 1 for the m1 builds).
    const char* suf_env = getenv("NPU_XCLBIN_SUFFIX");
    std::string suf = suf_env ? suf_env : "_qwen3_0_6b";
    const int XM = getenv("NPU_XCLBIN_XM") ? atoi(getenv("NPU_XCLBIN_XM")) : 128;
    auto xp = [&](const char* t) { static char b[256]; snprintf(b, 256, "%s/final_i8_%s%s.xclbin", xd, t, suf.c_str()); return b; };
    auto ip = [&](const char* t) { static char b[256]; snprintf(b, 256, "%s/insts_i8_%s%s.txt", xd, t, suf.c_str()); return b; };
    fprintf(stderr, "  xclbins: %s  (XM=%d)\n", suf.c_str(), XM);
    fusion::NpuGemmKernel cg, cd;
    if (!cg.init(npu, xp("GU"), ip("GU"), XM, H, 2 * IM)) { fprintf(stderr, "FAIL NPU GU init\n"); return 1; }
    if (!cd.init(npu, xp("D"), ip("D"), XM, IM, H)) { fprintf(stderr, "FAIL NPU D init\n"); return 1; }

    float gs, ds;
    cg.packB(gu_kn.data(), H, 2 * IM, gs);
    cd.packB(d_kn.data(), IM, H, ds);

    auto dyn_scale = [](const float* v, int n) {
        float a = 0; for (int i = 0; i < n; i++) { float f = fabsf(v[i]); if (f > a) a = f; }
        return a < 1e-12f ? 1.0f : a / 127.0f;
    };

    std::vector<float> npu_gu(2 * IM), npu_silu(IM), npu_final(H);
    float as_x = dyn_scale(x.data(), H);
    cg.go(x.data(), 1, H, as_x, gs, npu_gu.data(), 2 * IM);
    for (int i = 0; i < IM; i++) {
        float g = npu_gu[i], u = npu_gu[IM + i];
        npu_silu[i] = (g / (1.0f + expf(-g))) * u;
    }
    float as_silu = dyn_scale(npu_silu.data(), IM);
    cd.go(npu_silu.data(), 1, IM, as_silu, ds, npu_final.data(), H);

    // ── Compare ──
    double max_abs_diff = 0, sum_abs_ref = 0, dot = 0, norm_cpu = 0, norm_npu = 0;
    for (int i = 0; i < H; i++) {
        double diff = fabs(npu_final[i] - cpu_final[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_abs_ref += fabs(cpu_final[i]);
        dot += (double)npu_final[i] * cpu_final[i];
        norm_cpu += (double)cpu_final[i] * cpu_final[i];
        norm_npu += (double)npu_final[i] * npu_final[i];
    }
    double cosine = dot / (sqrt(norm_cpu) * sqrt(norm_npu) + 1e-12);
    double mean_abs_ref = sum_abs_ref / H;

    fprintf(stderr, "\n=== NPU FFN vs CPU float32 reference (real Qwen3-0.6B blk.0 weights, token %d) ===\n", token_id);
    fprintf(stderr, "  max abs diff:       %.6f\n", max_abs_diff);
    fprintf(stderr, "  mean |cpu_final|:   %.6f\n", mean_abs_ref);
    fprintf(stderr, "  cosine similarity:  %.6f\n", cosine);
    fprintf(stderr, "  first 8 CPU:  "); for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", cpu_final[i]); fprintf(stderr, "\n");
    fprintf(stderr, "  first 8 NPU:  "); for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", npu_final[i]); fprintf(stderr, "\n");

    // int8 quantization noise expected; cosine similarity is the meaningful
    // pass/fail signal (direction/magnitude preserved), not exact match.
    bool pass = cosine > 0.95;
    fprintf(stderr, "  %s (cosine %s 0.95)\n", pass ? "PASS" : "FAIL", pass ? ">" : "<=");
    return pass ? 0 : 1;
}
