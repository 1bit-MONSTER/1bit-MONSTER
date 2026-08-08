// ONNX → WMMA_I8 loader test: builds a minimal INT8 QDQ ONNX model with a
// hand-rolled protobuf wire encoder (zero deps — mirrors the loader itself),
// loads it through rcpp_bitnet_load_onnx, and verifies:
//   1. the parse works at all (the pre-1.22 field numbers found nothing),
//   2. weight_format == WMMA_I8 with the Hadamard flag set,
//   3. device INT8 weights are bit-exact vs CPU dequant→Hadamard→requant,
//   4. rcpp_wmma_i8_gemv over the loaded weights matches an int32 dot ref.
#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/bitnet_model.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); return 1; } } while(0)

// ── Deterministic LCG (no libc rand state) ────────────────────────────────
static uint32_t lcg = 42;
static int irand(int lo, int hi) { lcg = lcg * 1103515245u + 12345u; return lo + (int)(lcg % (uint32_t)(hi - lo + 1)); }
static float frand(float lo, float hi) { return lo + (hi - lo) * ((float)irand(0, 1000000) / 1000000.0f); }

// ── Minimal protobuf wire encoder (field numbers per onnx 1.22) ──────────
static void pv(std::vector<uint8_t>& b, uint64_t v) { while (v >= 0x80) { b.push_back((uint8_t)(v | 0x80)); v >>= 7; } b.push_back((uint8_t)v); }
static void tag(std::vector<uint8_t>& b, int field, int wire) { pv(b, ((uint64_t)field << 3) | wire); }
static void bvarint(std::vector<uint8_t>& b, int field, uint64_t v) { tag(b, field, 0); pv(b, v); }
static void bbytes(std::vector<uint8_t>& b, int field, const void* data, size_t n) {
    tag(b, field, 2); pv(b, n);
    const uint8_t* p = (const uint8_t*)data;
    b.insert(b.end(), p, p + n);
}
static void bstr(std::vector<uint8_t>& b, int field, const std::string& s) { bbytes(b, field, s.data(), s.size()); }
static void bmsg(std::vector<uint8_t>& b, int field, const std::vector<uint8_t>& m) { bbytes(b, field, m.data(), m.size()); }
static void packed_i64(std::vector<uint8_t>& b, int field, const int64_t* dims, int n) {
    std::vector<uint8_t> inner;
    for (int i = 0; i < n; i++) pv(inner, (uint64_t)dims[i]);
    bbytes(b, field, inner.data(), inner.size());
}

// TensorProto with raw_data (field 9)
static std::vector<uint8_t> make_tensor(const std::string& name, int data_type,
                                        const int64_t* dims, int ndims,
                                        const void* raw, size_t raw_n) {
    std::vector<uint8_t> t;
    packed_i64(t, 1, dims, ndims);
    bvarint(t, 2, data_type);
    bstr(t, 8, name);
    bbytes(t, 9, raw, raw_n);
    return t;
}
// NodeProto: DequantizeLinear(input, scale, [zp]) → output
static std::vector<uint8_t> make_dq(const std::string& w, const std::string& sc,
                                    const std::string& zp, const std::string& out) {
    std::vector<uint8_t> n;
    bstr(n, 1, w);                 // input (field 1)
    bstr(n, 1, sc);
    if (!zp.empty()) bstr(n, 1, zp);
    bstr(n, 2, out);               // output (field 2)
    bstr(n, 3, "dq_" + w);         // name (field 3)
    bstr(n, 4, "DequantizeLinear");// op_type (field 4)
    return n;
}

// ── Reference transform (must match src/onnx_loader.cpp exactly) ─────────
constexpr int HB = 128;
static void hadamard_row(float* row, int K) {
    for (int base = 0; base < K; base += HB) {
        float* blk = row + base;
        for (int stage = 0; (1 << stage) < HB; ++stage) {
            int dist = 1 << stage;
            for (int i = 0; i < HB; ++i) if ((i & dist) == 0) {
                int j = i | dist; float a = blk[i], b = blk[j];
                blk[i] = a + b; blk[j] = a - b;
            }
        }
        constexpr float inv = 1.0f / 11.31370849898476f;
        for (int i = 0; i < HB; ++i) blk[i] *= inv;
    }
}
static void requant_row(const float* row, int K, std::vector<int8_t>& out, float& scale) {
    float m = 0; for (int i = 0; i < K; ++i) { float a = fabsf(row[i]); if (a > m) m = a; }
    scale = m / 127.0f; if (scale < 1e-10f) scale = 1e-10f;
    out.resize(K);
    for (int i = 0; i < K; ++i) {
        float v = row[i] / scale;
        v = fminf(fmaxf(roundf(v), -128.0f), 127.0f);
        out[i] = (int8_t)v;
    }
}

int main() {
    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        fprintf(stderr, "no HIP device available, skipping\n");
        return 77;
    }
    HIP_CHECK(hipSetDevice(0));

    // ── Build the model: H=128, V=64, IS=256, 1 layer ─────────────────────
    const int H = 128, V = 64, IS = 256;
    const char* lin_names[7] = {
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",    "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    };
    const int lin_rows[7] = { H, H, H, H, IS, IS, H };
    const int lin_cols[7] = { H, H, H, H, H, H, IS };
    // q_proj gets zp = -3; the rest default zp 0 (absent input)
    const int lin_zp[7] = { -3, 0, 0, 0, 0, 0, 0 };

    std::vector<std::vector<uint8_t>> inits;
    std::vector<std::vector<uint8_t>> nodes;
    std::vector<std::vector<int8_t>> lin_i8(7);
    std::vector<std::vector<float>> lin_scale(7);

    // embed F16, norms F32
    std::vector<uint16_t> emb((size_t)V * H);
    for (auto& v : emb) v = (uint16_t)irand(0, 65535);
    int64_t dims2[2] = { V, H };
    inits.push_back(make_tensor("model.embed_tokens.weight", 10, dims2, 2, emb.data(), emb.size() * 2));
    std::vector<float> normf(H), normp(H), normi(H);
    for (int i = 0; i < H; i++) { normf[i] = frand(0.5f, 1.5f); normp[i] = frand(0.5f, 1.5f); normi[i] = frand(0.5f, 1.5f); }
    int64_t dims1[1] = { H };
    inits.push_back(make_tensor("model.norm.weight", 1, dims1, 1, normf.data(), H * 4));
    inits.push_back(make_tensor("model.layers.0.input_layernorm.weight", 1, dims1, 1, normi.data(), H * 4));
    inits.push_back(make_tensor("model.layers.0.post_attention_layernorm.weight", 1, dims1, 1, normp.data(), H * 4));

    for (int l = 0; l < 7; l++) {
        int R = lin_rows[l], C = lin_cols[l];
        std::vector<float> scale(R);
        for (int r = 0; r < R; r++) scale[r] = frand(0.002f, 0.02f);
        std::vector<int8_t> w8((size_t)R * C);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                float v = frand(-1.0f, 1.0f);
                int q = (int)roundf(v / scale[r]);
                if (q > 127) q = 127; if (q < -128) q = -128;
                w8[(size_t)r * C + c] = (int8_t)q;
            }
        int64_t dims[2] = { R, C };
        int64_t dims_sc[1] = { R };
        inits.push_back(make_tensor(lin_names[l], 3, dims, 2, w8.data(), w8.size()));
        inits.push_back(make_tensor(std::string(lin_names[l]) + "_scale", 1, dims_sc, 1, scale.data(), R * 4));
        if (lin_zp[l] != 0) {
            int8_t zpv = (int8_t)lin_zp[l];
            int64_t dims1_[1] = { 1 };
            inits.push_back(make_tensor(std::string(lin_names[l]) + "_zp", 3, dims1_, 1, &zpv, 1));
        }
        nodes.push_back(make_dq(lin_names[l], std::string(lin_names[l]) + "_scale",
                                lin_zp[l] != 0 ? std::string(lin_names[l]) + "_zp" : std::string(),
                                std::string(lin_names[l]) + "_dq"));
        lin_i8[l] = w8; lin_scale[l] = scale;
    }

    // GraphProto + ModelProto
    std::vector<uint8_t> graph;
    bstr(graph, 2, "t");
    for (auto& n : nodes) bmsg(graph, 1, n);
    for (auto& i : inits) bmsg(graph, 5, i);
    std::vector<uint8_t> model;
    bvarint(model, 1, 13);           // ir_version
    bmsg(model, 7, graph);           // graph

    const char* path = "/tmp/onnx_wmma_i8_test.onnx";
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    fwrite(model.data(), 1, model.size(), fp);
    fclose(fp);
    fprintf(stderr, "wrote %s (%zu bytes)\n", path, model.size());

    // ── Load ───────────────────────────────────────────────────────────────
    rcpp_bitnet_model_t m;
    memset(&m, 0, sizeof(m));
    rcpp_status_t st = rcpp_bitnet_load_onnx(path, &m);
    if (st != RCPP_OK) { fprintf(stderr, "FAIL: load returned %d\n", (int)st); return 1; }
    if (m.weight_format != RCPP_WEIGHT_FORMAT_WMMA_I8) {
        fprintf(stderr, "FAIL: weight_format %d, expected WMMA_I8\n", (int)m.weight_format);
        return 1;
    }
    if (!(m.flags & H1B_FLAG_HADAMARD_ROTATED)) {
        fprintf(stderr, "FAIL: H1B_FLAG_HADAMARD_ROTATED not set\n");
        return 1;
    }
    if (!m.layers[0].q_i8_dev || !m.layers[0].q_i8_scales_dev) {
        fprintf(stderr, "FAIL: q_i8_dev/q_i8_scales_dev null\n");
        return 1;
    }

    // ── Check 3: bit-exact weights vs CPU reference (q_proj, zp=-3) ───────
    const int R0 = H, C0 = H;
    std::vector<int8_t> ref((size_t)R0 * C0);
    std::vector<float> ref_scale_f(R0);
    for (int r = 0; r < R0; r++) {
        std::vector<float> row(C0);
        for (int c = 0; c < C0; c++)
            row[c] = (float)((int32_t)lin_i8[0][(size_t)r * C0 + c] + 3) * lin_scale[0][r]; // zp=-3
        hadamard_row(row.data(), C0);
        std::vector<int8_t> qi8;
        requant_row(row.data(), C0, qi8, ref_scale_f[r]);
        memcpy(ref.data() + (size_t)r * C0, qi8.data(), C0);
    }
    std::vector<int8_t> got((size_t)R0 * C0);
    std::vector<float> got_scale(R0);
    HIP_CHECK(hipMemcpy(got.data(), m.layers[0].q_i8_dev, R0 * C0, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_scale.data(), m.layers[0].q_i8_scales_dev, R0 * 4, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < ref.size(); i++)
        if (got[i] != ref[i]) { fprintf(stderr, "FAIL: weight byte %zu: got %d want %d\n", i, got[i], ref[i]); return 1; }
    for (int r = 0; r < R0; r++)
        if (fabsf(got_scale[r] - ref_scale_f[r]) > 1e-7f * fabsf(ref_scale_f[r])) {
            fprintf(stderr, "FAIL: row scale %d: got %f want %f\n", r, got_scale[r], ref_scale_f[r]);
            return 1;
        }

    // ── Check 4: rcpp_wmma_i8_gemv over the loaded weights vs int32 dot ───
    hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
    std::vector<float> hx(H);
    for (auto& v : hx) v = frand(-1.0f, 1.0f);
    std::vector<_Float16> hx16(H);
    for (int i = 0; i < H; i++) hx16[i] = (_Float16)hx[i];
    _Float16* dx; int8_t* dxi; float* dscdev; __half* dy;
    HIP_CHECK(hipMalloc(&dx, H * 2));
    HIP_CHECK(hipMalloc(&dxi, H));
    HIP_CHECK(hipMalloc(&dscdev, 4));
    HIP_CHECK(hipMalloc(&dy, R0 * 2));
    HIP_CHECK(hipMemcpy(dx, hx16.data(), H * 2, hipMemcpyHostToDevice));

    rcpp_hadamard_rotate_fp16(dx, dx, H, s);
    rcpp_quantize_fp16_to_i8(dx, dxi, dscdev, H, s);
    float x_scale; HIP_CHECK(hipMemcpy(&x_scale, dscdev, 4, hipMemcpyDeviceToHost));
    rcpp_wmma_i8_gemv(m.layers[0].q_i8_dev, dxi, x_scale, m.layers[0].q_i8_scales_dev, dy, R0, C0, s);
    HIP_CHECK(hipStreamSynchronize(s));

    // CPU reference using the SAME quantized x (copied back) — exact
    std::vector<int8_t> xi8(H);
    HIP_CHECK(hipMemcpy(xi8.data(), dxi, H, hipMemcpyDeviceToHost));
    std::vector<__half> ygot(R0);
    HIP_CHECK(hipMemcpy(ygot.data(), dy, R0 * 2, hipMemcpyDeviceToHost));
    for (int r = 0; r < R0; r++) {
        int32_t acc = 0;
        for (int c = 0; c < C0; c++) acc += (int32_t)got[(size_t)r * C0 + c] * (int32_t)xi8[c];
        float want = (float)acc * x_scale * got_scale[r];
        float have = (float)ygot[r];
        if (fabsf(have - want) > 1e-3f * (fabsf(want) + 1e-6f)) {
            fprintf(stderr, "FAIL: gemv row %d: got %f want %f\n", r, have, want);
            return 1;
        }
    }

    printf("PASS: onnx load → WMMA_I8 (%d linears), bit-exact weights, gemv match\n", 7);
    rcpp_bitnet_free(&m);
    hipFree(dx); hipFree(dxi); hipFree(dscdev); hipFree(dy);
    return 0;
}
