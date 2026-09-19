// tests/prism/test_prism_gdn_layer_hip.hip — device composition gate for one GDN layer (P3.3).
//
// Runs the whole Bonsai-27B layer-0 GDN block ON DEVICE out of the .1bp, using only our
// own kernels, and prints the same diagnostic lines as tests/prism/prism_layer0.cpp so
// compare_prism_layer0.py gates them against the (now corrected) numpy reference:
//
//   embedding row -> inverse fwht -> RMSNorm -> fwht
//     -> qkv/z/a/b GEMVs -> conv1d+silu -> q/k l2norm -> g/β
//     -> gated-delta recurrence + gated RMSNorm -> ssmo-out perm -> fwht -> ssm_out GEMV
//     -> residual -> RMSNorm -> fwht -> gate/up GEMV -> silu*up -> fwht -> down GEMV -> residual
//
// Build/run (device required):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//       tests/prism/test_prism_gdn_layer_hip.hip kernels/prism_hadamard_fwht.hip \
//       kernels/prism_gemv.hip kernels/prism_gdn.hip kernels/prism_ops.hip \
//       src/onebp_model.cpp -o /tmp/pgdnl && /tmp/pgdnl <model.1bp> [token]

#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_hadamard_fwht_f32(const void*, const int8_t*, void*, int, int, int, void*);
extern "C" int prism_gemv_f32(const void*, const float*, float*, int, int, int, void*);
extern "C" int prism_conv1d_silu_f32(const float*, const float*, float*, float*, int, int, void*);
extern "C" int prism_l2norm_heads_f32(float*, int, int, void*);
extern "C" int prism_gbeta_f32(const float*, const float*, const float*, const float*, float*, float*, int, void*);
extern "C" int prism_gdn_recurrence_f32(const float*, const float*, const float*, const float*, const float*,
                                        const float*, const float*, float*, float*, int, int, int, int, void*);
extern "C" int prism_ssmout_perm_f32(const float*, float*, int, int, int, void*);
extern "C" int prism_rmsnorm_f32(float*, const float*, int, void*);
extern "C" int prism_add_f32(float*, const float*, int, void*);
extern "C" int prism_silu_mul_f32(float*, const float*, int, void*);
extern "C" int prism_gemv_dense_f32(const float*, const float*, float*, int, int, void*);

#define TRY(label, expr) do { int rc_ = (expr); if (rc_) { std::fprintf(stderr, "FAIL %s rc=%d\n", label, rc_); return 2; } } while (0)

static OnebpPrismTransformView g_tv;
static const OnebpTensor* ft(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}
static int8_t* up_signs(int width) {
    const int8_t* s = g_tv.signs_for_width((uint32_t)width);
    if (!s) { std::fprintf(stderr, "no signs for width %d\n", width); std::exit(3); }
    int8_t* d = nullptr; HC(hipMalloc(&d, width)); HC(hipMemcpy(d, s, width, hipMemcpyHostToDevice));
    return d;
}
static uint8_t* up_prism(OnebpModel& m, const std::string& n, int* rows, int* cols, int* nb) {
    const OnebpTensor* t = ft(m, n);
    *rows = (int)t->dims[0]; *cols = (int)t->dims[1]; *nb = (int)prism::block_bytes(t->quant);
    const size_t bytes = (size_t)(*rows) * ((*cols) / 128) * (*nb);
    uint8_t* d = nullptr; HC(hipMalloc(&d, bytes));
    HC(hipMemcpy(d, m.tensor_data(*t), bytes, hipMemcpyHostToDevice));
    return d;
}
static float* up_vec(OnebpModel& m, const std::string& n, int* len) {
    const OnebpTensor* t = ft(m, n); *len = (int)t->dims[0];
    float* d = nullptr; HC(hipMalloc(&d, (size_t)*len * 4));
    HC(hipMemcpy(d, m.tensor_data(*t), (size_t)*len * 4, hipMemcpyHostToDevice));
    return d;
}
static float* up_f16tile(OnebpModel& m, const std::string& n, int rows, int cols) {
    const OnebpTensor* t = ft(m, n);
    const uint8_t* p = m.tensor_data(*t);
    std::vector<float> out((size_t)rows * cols);
    const int ntc = (cols + 255) / 256;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            const size_t tile = (size_t)(r / 32) * ntc + (c / 256);
            const size_t off = tile * 32 * 256 * 2 + ((size_t)(r % 32) * 256 + (c % 256)) * 2;
            out[(size_t)r * cols + c] = prism::f16_to_f32((uint16_t)(p[off] | (p[off + 1] << 8)));
        }
    float* d = nullptr; HC(hipMalloc(&d, out.size() * 4));
    HC(hipMemcpy(d, out.data(), out.size() * 4, hipMemcpyHostToDevice));
    return d;
}
static void back(const float* d, std::vector<float>& h) {
    HC(hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost));
}
static double l2v(const std::vector<float>& v) {
    double s = 0; for (float x : v) s += (double)x * x; return std::sqrt(s);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.1bp> [token]\n", argv[0]); return 2; }
    const int token = argc > 2 ? std::atoi(argv[2]) : 1000;
    OnebpModel m;
    if (!m.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    const OnebpTensor* et = ft(m, "__onebp_ext_prism_transform");
    const OnebpTensor* es = ft(m, "__onebp_ext_prism_signs");
    if (!et || !es) { std::fprintf(stderr, "no transform metadata (folded pack only)\n"); return 1; }
    if (!onebp_prism_transform_parse(m.tensor_data(*et), (size_t)et->bytes,
                                     (const int8_t*)m.tensor_data(*es), (size_t)es->bytes, g_tv))
        return 1;
    const int block = (int)g_tv.hdr->block_size;
    const int H = m.header.hidden_size, NK = 16, NV = 48, HK = 128, HV = 128;
    const int KD = NK * HK, VD = NV * HV, CD = 2 * KD + VD, NF = m.header.intermediate_size;
    std::printf("token=%d H=%d NK=%d NV=%d CD=%d block=%d\n", token, H, NK, NV, CD, block);

    // embedding row (host-dequant) -> inverse fwht on device
    const OnebpTensor* emb = ft(m, "token_embd.weight");
    std::vector<float> h(H);
    {
        const uint32_t nb = prism::block_bytes(emb->quant);
        const uint8_t* p = m.tensor_data(*emb) + (size_t)token * (H / 128) * nb;
        if (!prism::dequant_flat(emb->quant, p, h.data(), (size_t)H)) return 1;
    }
    float* dh = nullptr; HC(hipMalloc(&dh, H * 4)); HC(hipMemcpy(dh, h.data(), H * 4, hipMemcpyHostToDevice));
    int8_t* sgH = up_signs(H), * sgV = up_signs(VD), * sgF = up_signs(NF);
    if (prism_hadamard_fwht_f32(dh, sgH, dh, H, block, /*inverse=*/1, nullptr) != 0) return 2;
    back(dh, h); std::printf("h_embed_l2 %.9e\n", l2v(h));

    int len = 0, r, c, nb;
    float* d_attn_norm = up_vec(m, "blk.0.attn_norm.weight", &len);
    float* d_xn = nullptr; HC(hipMalloc(&d_xn, H * 4));
    HC(hipMemcpy(d_xn, dh, H * 4, hipMemcpyDeviceToDevice));   // norm a COPY; keep h for the residual
    if (prism_rmsnorm_f32(d_xn, d_attn_norm, H, nullptr) != 0) return 2;
    back(d_xn, h); std::printf("xnorm_l2 %.9e\n", l2v(h));

    float* dxr = nullptr; HC(hipMalloc(&dxr, H * 4));
    if (prism_hadamard_fwht_f32(d_xn, sgH, dxr, H, block, 0, nullptr) != 0) return 2;

    uint8_t* Wqkv = up_prism(m, "blk.0.attn_qkv.weight", &r, &c, &nb);
    float* d_qkv = nullptr; HC(hipMalloc(&d_qkv, (size_t)CD * 4));
    if (prism_gemv_f32(Wqkv, dxr, d_qkv, CD, H, nb, nullptr) != 0) return 2;
    std::vector<float> qkv(CD); back(d_qkv, qkv); std::printf("qkv_l2 %.9e\n", l2v(qkv));

    uint8_t* Wz = up_prism(m, "blk.0.attn_gate.weight", &r, &c, &nb);
    float* d_z = nullptr; HC(hipMalloc(&d_z, (size_t)VD * 4));
    if (prism_gemv_f32(Wz, dxr, d_z, VD, H, nb, nullptr) != 0) { std::fprintf(stderr, "FAIL z r=%d c=%d nb=%d\n", VD, H, nb); return 2; }
    float* Wa = up_f16tile(m, "blk.0.ssm_alpha.weight", NV, H);   // F16 tile in the folded pack
    float* d_a = nullptr; HC(hipMalloc(&d_a, (size_t)NV * 4));
    // ssm_alpha/ssm_beta are NOT in prism.hadamard.weight_names (unfolded), so they take the
    // un-rotated normalised activation d_xn; qkv/gate are folded and take dxr.
    if (prism_gemv_dense_f32(Wa, d_xn, d_a, NV, H, nullptr) != 0) { std::fprintf(stderr, "FAIL a\n"); return 2; }
    float* Wb = up_f16tile(m, "blk.0.ssm_beta.weight", NV, H);
    float* d_b = nullptr; HC(hipMalloc(&d_b, (size_t)NV * 4));
    if (prism_gemv_dense_f32(Wb, d_xn, d_b, NV, H, nullptr) != 0) { std::fprintf(stderr, "FAIL b\n"); return 2; }

    float* d_conv_w = up_f16tile(m, "blk.0.ssm_conv1d.weight", CD, 4);
    float* d_conv_state = nullptr; HC(hipMalloc(&d_conv_state, (size_t)CD * 3 * 4));
    HC(hipMemset(d_conv_state, 0, (size_t)CD * 3 * 4));
    float* d_conv = nullptr; HC(hipMalloc(&d_conv, (size_t)CD * 4));
    if (prism_conv1d_silu_f32(d_qkv, d_conv_w, d_conv_state, d_conv, CD, 3, nullptr) != 0) { std::fprintf(stderr, "FAIL conv\n"); return 2; }
    std::vector<float> conv(CD); back(d_conv, conv); std::printf("conv_l2 %.9e\n", l2v(conv));

    if (prism_l2norm_heads_f32(d_conv, NK, HK, nullptr) != 0) return 2;
    if (prism_l2norm_heads_f32(d_conv + KD, NK, HK, nullptr) != 0) return 2;

    int nA = 0, ndt = 0, nnw = 0;
    float* d_A = up_vec(m, "blk.0.ssm_a", &nA);
    float* d_dt = up_vec(m, "blk.0.ssm_dt.bias", &ndt);
    float* d_nw = up_vec(m, "blk.0.ssm_norm.weight", &nnw);
    float* d_g = nullptr; HC(hipMalloc(&d_g, (size_t)NV * 4));
    float* d_beta = nullptr; HC(hipMalloc(&d_beta, (size_t)NV * 4));
    if (prism_gbeta_f32(d_a, d_dt, d_A, d_b, d_g, d_beta, NV, nullptr) != 0) return 2;

    float* d_state = nullptr; HC(hipMalloc(&d_state, (size_t)NV * HK * HV * 4));
    HC(hipMemset(d_state, 0, (size_t)NV * HK * HV * 4));
    float* d_core = nullptr; HC(hipMalloc(&d_core, (size_t)VD * 4));
    if (prism_gdn_recurrence_f32(d_conv, d_conv + KD, d_conv + 2 * KD, d_g, d_beta, d_nw, d_z,
                                 d_state, d_core, NK, NV, HK, HV, nullptr) != 0) return 2;
    std::vector<float> state((size_t)NV * HK * HV), core(VD);
    back(d_state, state); std::printf("state_l2 %.9e\n", l2v(state));
    back(d_core, core);  std::printf("core_l2 %.9e\n", l2v(core));

    float* d_perm = nullptr; HC(hipMalloc(&d_perm, (size_t)VD * 4));
    if (prism_ssmout_perm_f32(d_core, d_perm, NV, NK, HV, nullptr) != 0) return 2;
    float* d_cr = nullptr; HC(hipMalloc(&d_cr, (size_t)VD * 4));
    if (prism_hadamard_fwht_f32(d_perm, sgV, d_cr, VD, block, 0, nullptr) != 0) return 2;

    uint8_t* Wout = up_prism(m, "blk.0.ssm_out.weight", &r, &c, &nb);
    float* d_out = nullptr; HC(hipMalloc(&d_out, (size_t)H * 4));
    if (prism_gemv_f32(Wout, d_cr, d_out, H, VD, nb, nullptr) != 0) return 2;
    std::vector<float> out(H); back(d_out, out); std::printf("gdn_out_l2 %.9e\n", l2v(out));

    if (prism_add_f32(dh, d_out, H, nullptr) != 0) return 2;

    float* d_post_norm = up_vec(m, "blk.0.post_attention_norm.weight", &len);
    float* d_yn = nullptr; HC(hipMalloc(&d_yn, H * 4));
    HC(hipMemcpy(d_yn, dh, H * 4, hipMemcpyDeviceToDevice));   // norm a COPY; keep h for the residual
    if (prism_rmsnorm_f32(d_yn, d_post_norm, H, nullptr) != 0) return 2;
    float* d_yr = nullptr; HC(hipMalloc(&d_yr, H * 4));
    if (prism_hadamard_fwht_f32(d_yn, sgH, d_yr, H, block, 0, nullptr) != 0) return 2;
    uint8_t* Wgate = up_prism(m, "blk.0.ffn_gate.weight", &r, &c, &nb);
    float* d_gate = nullptr; HC(hipMalloc(&d_gate, (size_t)NF * 4));
    if (prism_gemv_f32(Wgate, d_yr, d_gate, NF, H, nb, nullptr) != 0) return 2;
    uint8_t* Wup = up_prism(m, "blk.0.ffn_up.weight", &r, &c, &nb);
    float* d_up = nullptr; HC(hipMalloc(&d_up, (size_t)NF * 4));
    if (prism_gemv_f32(Wup, d_yr, d_up, NF, H, nb, nullptr) != 0) return 2;
    if (prism_silu_mul_f32(d_gate, d_up, NF, nullptr) != 0) return 2;
    float* d_gr = nullptr; HC(hipMalloc(&d_gr, (size_t)NF * 4));
    if (prism_hadamard_fwht_f32(d_gate, sgF, d_gr, NF, block, 0, nullptr) != 0) return 2;
    uint8_t* Wdown = up_prism(m, "blk.0.ffn_down.weight", &r, &c, &nb);
    float* d_down = nullptr; HC(hipMalloc(&d_down, (size_t)H * 4));
    if (prism_gemv_f32(Wdown, d_gr, d_down, H, NF, nb, nullptr) != 0) return 2;
    if (prism_add_f32(dh, d_down, H, nullptr) != 0) return 2;

    back(dh, h);
    std::printf("layer_out_l2 %.9e\n", l2v(h));
    std::printf("layer_out[0..7]");
    for (int i = 0; i < 8; i++) std::printf(" %.9e", (double)h[i]);
    std::printf("\n");
    return 0;
}
