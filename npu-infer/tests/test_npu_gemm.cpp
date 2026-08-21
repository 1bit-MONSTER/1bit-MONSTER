// test_npu_gemm.c — does mm.xclbin compute a real GEMM with our packed weights?
// Packs q_proj block 0 (BF16 [256,1024]), multiplies by known x[1024], compares vs CPU.
// Usage: LD_LIBRARY_PATH=/opt/fastflowlm/lib ./test_npu_gemm [model_path]
#include "engine.h"
#include "model.h"
#include "common.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static float bf16_to_f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }
static uint16_t f_to_bf16(float v) { uint32_t b; memcpy(&b, &v, 4); uint32_t r = ((b >> 16) & 1) + 0x7FFF; return (uint16_t)((b + r) >> 16); }

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    ModelConfig cfg = QWEN3_0_6B_CONFIG;

    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) return 1;
    TensorDesc* q = &mw->layers[0].q_proj_weight;

    // Pack block 0 into a host buffer
    static uint8_t block[1048576];
    const void* wdata = model_tensor_data(mw, q);
    npu_pack_weight_bo(block, wdata, q, &cfg, 0);
    printf("packed block0 (%d blocks total)\n", npu_weight_num_blocks(q, &cfg));

    // Reference: y = W[0:256,:] @ x  (BF16 weights as floats)
    const uint16_t* W = (const uint16_t*)block;
    float x[1024], ref[256];
    for (int i = 0; i < 1024; i++) x[i] = (i % 7) * 0.01f - 0.3f;
    for (int r = 0; r < 256; r++) {
        double acc = 0;
        for (int c = 0; c < 1024; c++) acc += bf16_to_f(W[r*1024 + c]) * x[c];
        ref[r] = (float)acc;
    }

    // NPU setup
    xrt::device dev(0);
    const char* xclbin_path = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin";
    FILE* f = fopen(xclbin_path, "rb"); fseek(f, 0, SEEK_END);
    long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    xrt::kernel kern(dev, xclbin->get_uuid(), "MLIR_AIE");

    xrt::bo act(dev, cfg.npu_activation_bo_size, xrt::bo::flags::host_only, 0);
    xrt::bo ws(dev, 10485760, xrt::bo::flags::host_only, 0);
    xrt::bo wt(dev, cfg.npu_weight_bo_size, xrt::bo::flags::host_only, 0);
    xrt::bo kv(dev, cfg.npu_kv_cache_bo_size, xrt::bo::flags::host_only, 0);
    memset(ws.map(), 0, 10485760);
    memset(kv.map(), 0, cfg.npu_kv_cache_bo_size);
    memcpy(wt.map(), block, 1048576);

    // Activation: x as BF16 at start of act BO
    uint16_t* actmap = (uint16_t*)act.map();
    for (int i = 0; i < 1024; i++) actmap[i] = f_to_bf16(x[i]);
    memset(act.map() + 2048, 0, cfg.npu_activation_bo_size - 2048);

    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, cfg.npu_activation_bo_size, 0);
    ws.sync(XCL_BO_SYNC_BO_TO_DEVICE, 10485760, 0);
    wt.sync(XCL_BO_SYNC_BO_TO_DEVICE, cfg.npu_weight_bo_size, 0);
    kv.sync(XCL_BO_SYNC_BO_TO_DEVICE, cfg.npu_kv_cache_bo_size, 0);

    auto run = kern((uint64_t)3, (uint64_t)0, (uint32_t)0,
                    act, ws, wt, wt, kv);
    run.wait();

    act.sync(XCL_BO_SYNC_BO_FROM_DEVICE, cfg.npu_activation_bo_size, 0);
    const uint16_t* out = (const uint16_t*)act.map();
    printf("NPU out row0 first 16 (bf16->float): ");
    for (int i = 0; i < 16; i++) printf("%.4f ", bf16_to_f(out[i]));
    printf("\nref      row0 first 16:              ");
    for (int i = 0; i < 16; i++) printf("%.4f ", ref[i]);
    printf("\n");

    double err = 0; double maxerr = 0;
    for (int i = 0; i < 256; i++) {
        double d = fabs(bf16_to_f(out[i]) - ref[i]);
        err += d; if (d > maxerr) maxerr = d;
    }
    printf("mean abs err over 256 outputs: %.6f  max err: %.6f\n", err/256, maxerr);
    printf("first output: npu=%.4f ref=%.4f\n", bf16_to_f(out[0]), ref[0]);

    model_free(mw);
    return 0;
}
