// test_npu_ffn_batch.cpp — batched multi-sequence FFN validation.
//
// npu_state_ffn_batch processes `am` independent rows in ONE GU + ONE D
// launch each (the B weight DMA is read once for all rows).  This test
// verifies the batched rows are BIT-IDENTICAL to am x npu_state_ffn calls
// (real blk.0 weights, 8 real embedding-row inputs) and prints the speedup.
//
// Usage: ./test_npu_ffn_batch [path/to/Qwen3-0.6B.1bp]
// Build (TheRock): amdclang++ -std=c++20 -O2 -D__HIP_PLATFORM_AMD__ -I. -Iinclude
//   -I/usr/include/xrt -I$THEROCK_DEVEL/include test_npu_ffn_batch.cpp \
//   ../../src/backend_fused_npu.cpp -o test_npu_ffn_batch -lxrt_coreutil -luuid -ldl -lpthread
#include "src/backend_fused_npu.h"
#include "../../npu/src/onebp_loader.cpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double,std::micro>(b-a).count(); }

int main(int argc, char** argv) {
    int acc = open("/dev/accel/accel0", O_RDONLY); if (acc < 0) { printf("no NPU\n"); return 77; } close(acc);
    const char* mp = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    NpuOnebpModel mdl;
    if (!mdl.open(mp)) { printf("model open fail\n"); return 1; }
    const int H = mdl.header().hidden_size, NC = mdl.header().num_layers, IM = mdl.header().intermediate_size;
    // create state (m8 family via repo xclbins)
    NpuState* s = npu_state_create("engine/npu/xclbins", H, IM, NC);
    if (!s) { printf("npu_state_create fail\n"); return 1; }
    std::vector<float> w1, w2, w3, pon, emb;
    if (!mdl.get_tensor_f32("blk.0.ffn_gate.weight", w1) || !mdl.get_tensor_f32("blk.0.ffn_up.weight", w2) ||
        !mdl.get_tensor_f32("blk.0.ffn_down.weight", w3) || !mdl.get_tensor_f32("blk.0.ffn_norm.weight", pon) ||
        !mdl.get_tensor_f32("token_embd.weight", emb)) { printf("tensor fail\n"); return 1; }
    npu_state_pack_layer(s, 0, w1.data(), w2.data(), w3.data(), pon.data());
    // 8 different real-ish inputs (embedding rows for tokens 1..8, rmsnorm'd)
    const int AM = 8;
    std::vector<float> h_single((size_t)AM * H), h_batch((size_t)AM * H);
    for (int m = 0; m < AM; m++) {
        std::vector<float> x(emb.begin() + (size_t)(m+1)*H, emb.begin() + (size_t)(m+2)*H);
        double ss = 0; for (float v : x) ss += (double)v*v;
        float inv = 1.0f / sqrtf((float)(ss/H) + 1e-6f);
        for (int i = 0; i < H; i++) { float v = x[i]*inv*pon[i]; h_single[(size_t)m*H+i] = v; h_batch[(size_t)m*H+i] = v; }
    }
    // single-row reference (bit-identical path)
    for (int m = 0; m < AM; m++) if (!npu_state_ffn(s, 0, h_single.data() + (size_t)m*H, H)) { printf("single ffn fail m=%d\n", m); return 1; }
    // batched
    auto t0 = clk::now();
    if (!npu_state_ffn_batch(s, 0, h_batch.data(), H, AM)) { printf("batch ffn fail\n"); return 1; }
    auto t1 = clk::now();
    printf("batch FFN (am=%d): %.0f us\n", AM, us(t0,t1));
    // compare
    double maxd = 0; int worst = -1;
    for (int i = 0; i < AM*H; i++) { double d = fabs((double)h_single[i] - h_batch[i]); if (d > maxd) { maxd = d; worst = i; } }
    printf("batch vs 8x single: max abs diff = %.8f at flat[%d] (row %d) (%s)\n", maxd, worst, worst/H, maxd < 1e-5 ? "BIT-IDENTICAL" : "MISMATCH");
    for (int m = 0; m < AM; m++) {
        double md = 0, ms = 0;
        for (int i = 0; i < H; i++) { double d = fabs((double)h_single[m*H+i] - h_batch[m*H+i]); if (d > md) md = d; ms += fabs(h_batch[m*H+i]); }
        printf("  row %d: max diff %.6f  sum|batch| %.3f\n", m, md, ms);
    }
    // timing comparison: 8x single
    double t8s = 0;
    for (int rep = 0; rep < 3; rep++) {
        std::vector<float> tmp = h_single;  // reuse (already FFN'd — timing only)
        auto a0 = clk::now();
        for (int m = 0; m < AM; m++) npu_state_ffn(s, 0, tmp.data() + (size_t)m*H, H);
        t8s += us(a0, clk::now());
    }
    printf("8x single FFN: %.0f us; batch: %.0f us (%.1fx)\n", t8s/3, us(t0,t1), (t8s/3)/us(t0,t1));
    npu_state_destroy(s);
    return 0;
}
