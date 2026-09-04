// runlist_layer_test.cpp — run a REAL per-ctx layer kernel through xrt::runlist
// on the live NPU (issue #1776 / HRX runlist milestone), using the runtime's
// exact ABI (3,0,0, act, w, o1, o2, kv) from test_npu_layer_elf.cpp.
//
// Build against the runlist-capable XRT 2.26.0 stack. Uses the captured layer
// ELF (elf_0001_layer.bin) + layer.xclbin + the Qwen3-0.6B model.q4nx.
#include "model.h"
#include "common.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_module.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

static float bf16_to_f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }

// npu_pack_layer_bo is defined in model.c (C) but not declared in model.h.
extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, ModelWeights* mw,
                                 const ModelConfig* config, int layer_idx);

static std::vector<char> read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    const char* elf_path  = (argc > 2) ? argv[2] : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs/elf_0001_layer.bin";
    const char* xclbin_path = (argc > 3) ? argv[3] : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
    int nlayers = (argc > 4) ? atoi(argv[4]) : 1;

    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    std::string model_file = std::string(model_dir) + "/model.q4nx";
    ModelWeights* mw = model_load(model_file.c_str(), cfg);
    if (!mw) { std::fprintf(stderr, "model_load failed\n"); return 1; }
    std::printf("[1] model loaded\n");

    // One per-layer weight BO (10 MB) via npu_pack_layer_bo; reuse for nlayers.
    std::vector<uint8_t> wbo(10485760);
    if (npu_pack_layer_bo(wbo.data(), mw, &cfg, 0) <= 0) { std::fprintf(stderr, "pack failed\n"); return 1; }
    std::printf("[2] layer weight packing OK\n");

    std::vector<char> ebuf = read_file(elf_path);
    if (ebuf.empty()) { std::fprintf(stderr, "no elf %s\n", elf_path); return 1; }
    std::vector<char> raw = read_file(xclbin_path);
    if (raw.empty()) { std::fprintf(stderr, "no xclbin %s\n", xclbin_path); return 1; }

    xrt::device dev(0);
    std::string xcp(xclbin_path);
    xrt::xclbin xc(xcp);
    dev.register_xclbin(xc);
    auto hwctx = xrt::hw_context(dev, xc.get_uuid());
    xrt::elf elf(ebuf.data(), ebuf.size());
    xrt::module mod(elf);
    xrt::ext::kernel kern(hwctx, mod, "MLIR_AIE");
    std::printf("[3] hwctx + layer kernel ready\n");

    // Runtime ABI BOs.
    xrt::ext::bo bo_act(dev, 1048576);
    xrt::ext::bo bo_w(dev, 10485760);
    xrt::ext::bo bo_o1(dev, 1048576);
    xrt::ext::bo bo_o2(dev, 1048576);
    xrt::ext::bo bo_kv(dev, cfg.npu_kv_cache_bo_size);
    memcpy(bo_w.map(), wbo.data(), wbo.size());
    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bo_act.map(), 0, 1048576);
    memset(bo_o1.map(), 0, 1048576);
    memset(bo_o2.map(), 0, 1048576);
    memset(bo_kv.map(), 0, cfg.npu_kv_cache_bo_size);
    uint16_t* am = (uint16_t*)bo_act.map();
    for (int i = 0; i < 1024; i++) am[i] = bf16_to_f((uint16_t)(i * 37));
    bo_act.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::printf("[4] BOs packed\n");

    // Build the runlist: nlayers submissions of the SAME kernel, chained.
    xrt::runlist rl(hwctx);
    uint32_t v0 = 3, v1 = 0, v2 = 0;
    for (int l = 0; l < nlayers; l++) {
        xrt::run r(kern);
        r.set_arg(0, (const void*)&v0, sizeof(v0));
        r.set_arg(1, (const void*)&v1, sizeof(v1));
        r.set_arg(2, (const void*)&v2, sizeof(v2));
        r.set_arg(3, (const xrt::bo&)bo_act);
        r.set_arg(4, (const xrt::bo&)bo_w);
        r.set_arg(5, (const xrt::bo&)bo_o1);
        r.set_arg(6, (const xrt::bo&)bo_o2);
        r.set_arg(7, (const xrt::bo&)bo_kv);
        rl.add(r);
    }
    std::printf("[5] runlist %d runs built\n", nlayers);

    auto t0 = std::chrono::steady_clock::now();
    rl.execute();
    rl.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::printf("[6] runlist execute+wait in %.2f ms\n", ms);

    bo_o1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_o2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint16_t* p1 = (const uint16_t*)bo_o1.map();
    const uint16_t* p2 = (const uint16_t*)bo_o2.map();
    size_t nz1 = 0, nz2 = 0; float s1 = 0, s2 = 0;
    for (size_t i = 0; i < 524288; i++) { if (p1[i]) nz1++; if (p2[i]) nz2++; s1 += bf16_to_f(p1[i]); s2 += bf16_to_f(p2[i]); }
    std::printf("[7] o1 nz=%zu  o2 nz=%zu (nonzero outputs => kernel executed)\n", nz1, nz2);
    return 0;
}
