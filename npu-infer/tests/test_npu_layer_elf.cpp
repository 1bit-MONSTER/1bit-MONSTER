// test_layer_elf.cpp — replicate the runtime's EXACT submission: build
// xrt::elf from the CAPTURED ELF, xrt::module, xrt::ext::kernel, and run
// with (3,0,0, act, weight, out1, out2, kv) — the runtime's real ABI.
#include "model.h"
#include "common.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_hw_context.h>
#include <xrt/experimental/xrt_module.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

static float bf16_to_f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }
static uint16_t f_to_bf16(float v) { uint32_t b; memcpy(&b, &v, 4); uint32_t r = ((b >> 16) & 1) + 0x7FFF; return (uint16_t)((b + r) >> 16); }

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    const char* elf_path = (argc > 2) ? argv[2]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs/elf_0001_layer.bin";
    int layer_idx = (argc > 3) ? atoi(argv[3]) : 0;
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    std::string model_file = std::string(model_dir) + "/model.q4nx";
    ModelWeights* mw = model_load(model_file.c_str(), cfg);
    if (!mw) return 1;

    std::vector<uint8_t> wbo(10485760);
    const char* wcap = getenv("MM_CAPTURED_W");
    if (wcap) {
        FILE* fw = fopen(wcap, "rb");
        if (!fw) { fprintf(stderr, "no cap weight %s\n", wcap); return 1; }
        size_t got = fread(wbo.data(), 1, 10485760, fw); fclose(fw);
        printf("loaded captured weight BO (%zu B) from %s\n", got, wcap);
    } else {
        int tiles = npu_pack_layer_bo(wbo.data(), mw, &cfg, layer_idx);
        if (tiles <= 0) { fprintf(stderr, "pack failed\n"); return 1; }
        printf("packed layer %d (%d tiles)\n", layer_idx, tiles);
    }

    FILE* fe = fopen(elf_path, "rb");
    if (!fe) { fprintf(stderr, "no elf %s\n", elf_path); return 1; }
    fseek(fe, 0, SEEK_END); long esz = ftell(fe); fseek(fe, 0, SEEK_SET);
    std::vector<char> ebuf(esz);
    size_t br = fread(ebuf.data(), 1, esz, fe); fclose(fe);
    printf("loaded elf (%ld B)\n", esz);

    xrt::device dev(0);
    const char* xclbin_path = getenv("LAYER_XCLBIN")
        ? getenv("LAYER_XCLBIN")
        : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
    FILE* f = fopen(xclbin_path, "rb"); fseek(f, 0, SEEK_END);
    long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    printf("creating hwctx...\n"); fflush(stdout);
    auto hwctx = xrt::hw_context(dev, xclbin->get_uuid());
    printf("creating elf...\n"); fflush(stdout);
    xrt::elf elf((const char*)ebuf.data(), esz);
    printf("creating module...\n"); fflush(stdout);
    xrt::module mod(elf);
    printf("creating ext kernel...\n"); fflush(stdout);
    xrt::ext::kernel kern(hwctx, mod, "MLIR_AIE");
    printf("kernel done\n"); fflush(stdout);

    // match the runtime: ext::bo (shared host+device memory), no explicit group
    xrt::ext::bo bo_act(dev, 1048576);
    xrt::ext::bo bo_w(dev, 10485760);
    xrt::ext::bo bo_o1(dev, 1048576);
    xrt::ext::bo bo_o2(dev, 1048576);
    xrt::ext::bo bo_kv(dev, cfg.npu_kv_cache_bo_size);

    memcpy(bo_w.map(), wbo.data(), wbo.size());
    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const char* n1 = getenv("NORM1_CAP");
    const char* n2 = getenv("NORM2_CAP");
    if (n1) { FILE* fa = fopen(n1, "rb"); if (fa) { fread(bo_o1.map(), 1, 1048576, fa); fclose(fa); printf("norm1 from %s\n", n1); } }
    if (n2) { FILE* fa = fopen(n2, "rb"); if (fa) { fread(bo_o2.map(), 1, 1048576, fa); fclose(fa); printf("norm2 from %s\n", n2); } }
    bo_o1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_o2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bo_act.map(), 0, 1048576);
    memset(bo_o1.map(), 0, 1048576);
    memset(bo_o2.map(), 0, 1048576);
    memset(bo_kv.map(), 0, cfg.npu_kv_cache_bo_size);
    uint16_t* am = (uint16_t*)bo_act.map();
    const char* actcap = getenv("ACT_CAP");
    if (actcap) {
        FILE* fa = fopen(actcap, "rb");
        if (fa) { size_t g = fread(am, 2, 1024, fa); fclose(fa); printf("act from %s (%zu)\n", actcap, g); }
    } else {
        for (int i = 0; i < 1024; i++) am[i] = f_to_bf16(sinf(i * 0.01f));
        printf("synthetic act\n");
    }
    bo_act.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // runtime ABI via create_run: set_arg(0,3) (1,0) (2,0) then (3+i, bo)
    // cast ext::bo to xrt::bo like the runtime's bytes::bo() does
    xrt::run run(kern);
    uint32_t v0 = 3, v1 = 0, v2 = 0;
    run.set_arg(0, (const void*)&v0, sizeof(v0));
    run.set_arg(1, (const void*)&v1, sizeof(v1));
    run.set_arg(2, (const void*)&v2, sizeof(v2));
    run.set_arg(3, (const xrt::bo&)bo_act);
    run.set_arg(4, (const xrt::bo&)bo_w);
    run.set_arg(5, (const xrt::bo&)bo_o1);
    run.set_arg(6, (const xrt::bo&)bo_o2);
    run.set_arg(7, (const xrt::bo&)bo_kv);
    run.start();
    auto st = run.wait();
    printf("run wait state: %d\n", (int)st);

    bo_act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_o1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_o2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_kv.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto stats = [](const uint16_t* p, size_t n, const char* name) {
        size_t nz = 0; float s = 0, s2 = 0, mn = 1e30f, mx = -1e30f;
        for (size_t i = 0; i < n; i++) { float v = bf16_to_f(p[i]);
            if (p[i]) nz++; s += v; s2 += v*v; if (v<mn) mn=v; if (v>mx) mx=v; }
        double mean = n ? s/n : 0, var = n ? s2/n - mean*mean : 0;
        printf("%-5s nz=%6zu mean=%+.4f std=%.4f min=%.4f max=%.4f\n", name, nz, mean,
               var>0?sqrt(var):0, mn, mx);
    };
    stats((const uint16_t*)bo_act.map(), 524288, "act");
    stats((const uint16_t*)bo_o1.map(), 524288, "out1");
    stats((const uint16_t*)bo_o2.map(), 524288, "out2");
    stats((const uint16_t*)bo_kv.map(), cfg.npu_kv_cache_bo_size/2, "kv");
    if (getenv("DUMP_OUT")) {
        FILE* fo = fopen(getenv("DUMP_OUT"), "wb");
        if (fo) { fwrite(bo_act.map(), 1, 1048576, fo); fclose(fo); }
    }
    return 0;
}
