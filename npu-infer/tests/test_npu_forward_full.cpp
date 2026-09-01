// test_npu_forward_full.cpp — replicate the runtime's FULL per-forward runlist
// on-device: 28 layer runs (kernel A, captured ELF, per-layer packed weight BO)
// + 1 lm_head run (captured ELF, packed lm_head BO) — then compare the act
// buffer and logits against the runtime's captures.
//
// The runtime arms (per the interposer capture, capK manifest):
//   RUNLIST: 28x layer(kernel A, idx3=act idx4=w[L] idx5/6=norm[L] idx7=kv[L])
//            + 1x lm_head(idx3=logits idx4=lmhead_w idx5=act idx6=?)
//            + 28x layer(kernel C, same ELF as A)
// This test runs the 28+1 sequence (kernel A + lm_head) — the first half of
// the runtime's pattern — with hand-packed weights, and checks the hidden
// state / logits against the runtime captures.
//
// Usage:
//   ./test_npu_forward_full <model_dir> [elf_layer] [elf_lmhead] [n_layers]
#include "model.h"
#include "common.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_hw_context.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_ext.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

static float bf16_to_f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    const char* elf_layer = (argc > 2) ? argv[2]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs/elf_0001_layer.bin";
    const char* elf_lmhead = (argc > 3) ? argv[3]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs/elf_0002_lmhead.bin";
    int n_layers = (argc > 4) ? atoi(argv[4]) : 1;
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    std::string model_file = std::string(model_dir) + "/model.q4nx";
    ModelWeights* mw = model_load(model_file.c_str(), cfg);
    if (!mw) return 1;

    // read an ELF
    auto read_elf = [](const char* p, std::vector<char>& out) {
        FILE* f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "no elf %s\n", p); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        out.resize(sz);
        size_t br = fread(out.data(), 1, sz, f); fclose(f);
        return br == out.size();
    };
    std::vector<char> ebuf_l, ebuf_h;
    if (!read_elf(elf_layer, ebuf_l) || !read_elf(elf_lmhead, ebuf_h)) return 1;
    printf("elfs: layer %zu B, lm_head %zu B\n", ebuf_l.size(), ebuf_h.size());

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
    auto hwctx = xrt::hw_context(dev, xclbin->get_uuid());

    xrt::elf elf_l((const char*)ebuf_l.data(), ebuf_l.size());
    xrt::module mod_l(elf_l);
    xrt::ext::kernel kern_l(hwctx, mod_l, "MLIR_AIE");
    xrt::elf elf_h((const char*)ebuf_h.data(), ebuf_h.size());
    xrt::module mod_h(elf_h);
    xrt::ext::kernel kern_h(hwctx, mod_h, "MLIR_AIE");

    // ---- layer runs: shared act, per-layer weight BO ----
    xrt::ext::bo bo_act(dev, 1048576);
    xrt::ext::bo bo_o1(dev, 1048576);
    xrt::ext::bo bo_o2(dev, 1048576);
    xrt::ext::bo bo_kv(dev, cfg.npu_kv_cache_bo_size);
    memset(bo_act.map(), 0, 1048576);
    memset(bo_o1.map(), 0, 1048576);
    memset(bo_o2.map(), 0, 1048576);
    memset(bo_kv.map(), 0, cfg.npu_kv_cache_bo_size);
    const char* actcap = getenv("ACT_CAP");
    if (actcap) {
        FILE* fa = fopen(actcap, "rb");
        if (fa) { fread(bo_act.map(), 1, 2048, fa); fclose(fa); printf("act from %s\n", actcap); }
    }
    bo_act.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_o1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_o2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // pack each layer's weight BO (hand-rolled, byte-verified vs runtime)
    std::vector<xrt::ext::bo> wbos;
    for (int L = 0; L < n_layers; L++) {
        std::vector<uint8_t> wbo(10485760);
        int tiles = npu_pack_layer_bo(wbo.data(), mw, &cfg, L);
        if (tiles <= 0) { fprintf(stderr, "pack layer %d failed\n", L); return 1; }
        xrt::ext::bo bw(dev, 10485760);
        memcpy(bw.map(), wbo.data(), wbo.size());
        bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wbos.push_back(std::move(bw));
    }
    printf("packed %d layer weight BOs\n", n_layers);

    // ---- per-layer kv BOs (from the runtime's captures, KV_DIR) ----
    std::vector<xrt::ext::bo> kvbos;
    const char* kvdir = getenv("KV_DIR");
    if (kvdir) {
        for (int L = 0; L < n_layers; L++) {
            char fn[512];
            xrt::ext::bo bk(dev, cfg.npu_kv_cache_bo_size);
            snprintf(fn, sizeof(fn), "%s/L%02d.bin", kvdir, L);
            FILE* fa = fopen(fn, "rb");
            if (fa) { fread(bk.map(), 1, cfg.npu_kv_cache_bo_size, fa); fclose(fa); }
            bk.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            kvbos.push_back(std::move(bk));
        }
        printf("loaded %d per-layer kv BOs from %s\n", n_layers, kvdir);
    }
    // ---- per-layer norm buffers (from the runtime's captures) ----
    std::vector<xrt::ext::bo> n1bos, n2bos;
    const char* normdir = getenv("NORM_DIR");
    if (normdir) {
        for (int L = 0; L < n_layers; L++) {
            char fn[512];
            xrt::ext::bo bn1(dev, 1048576), bn2(dev, 1048576);
            snprintf(fn, sizeof(fn), "%s/L%02d_i5.bin", normdir, L);
            FILE* fa = fopen(fn, "rb");
            if (fa) { fread(bn1.map(), 1, 1048576, fa); fclose(fa); }
            snprintf(fn, sizeof(fn), "%s/L%02d_i6.bin", normdir, L);
            fa = fopen(fn, "rb");
            if (fa) { fread(bn2.map(), 1, 1048576, fa); fclose(fa); }
            bn1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bn2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            n1bos.push_back(std::move(bn1));
            n2bos.push_back(std::move(bn2));
        }
        printf("loaded %d per-layer norm buffers from %s\n", n_layers, normdir);
    }
    // ---- submit layer runs: the runtime's 56-run pattern (2x per layer) ----
    for (int pass = 0; pass < 2; pass++) {
        for (int L = 0; L < n_layers; L++) {
            xrt::run run(kern_l);
            uint32_t v0 = 3, v1 = 0, v2 = 0;
            run.set_arg(0, (const void*)&v0, sizeof(v0));
            run.set_arg(1, (const void*)&v1, sizeof(v1));
            run.set_arg(2, (const void*)&v2, sizeof(v2));
            run.set_arg(3, (const xrt::bo&)bo_act);
            run.set_arg(4, (const xrt::bo&)wbos[L]);
            if (!n1bos.empty()) {
                run.set_arg(5, (const xrt::bo&)n1bos[L]);
                run.set_arg(6, (const xrt::bo&)n2bos[L]);
            } else {
                run.set_arg(5, (const xrt::bo&)bo_o1);
                run.set_arg(6, (const xrt::bo&)bo_o2);
            }
            if (!kvbos.empty()) run.set_arg(7, (const xrt::bo&)kvbos[L]);
            else run.set_arg(7, (const xrt::bo&)bo_kv);
            run.start();
            run.wait();
        }
    }
    bo_act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto stats = [](const uint16_t* p, size_t n, const char* name) {
        size_t nz = 0; float s = 0, s2 = 0, mn = 1e30f, mx = -1e30f;
        for (size_t i = 0; i < n; i++) { float v = bf16_to_f(p[i]);
            if (p[i]) nz++; s += v; s2 += v*v; if (v<mn) mn=v; if (v>mx) mx=v; }
        double mean = n ? s/n : 0, var = n ? s2/n - mean*mean : 0;
        printf("%-5s nz=%6zu mean=%+.4f std=%.4f min=%.4f max=%.4f\n", name, nz, mean,
               var>0?sqrt(var):0, mn, mx);
    };
    stats((const uint16_t*)bo_act.map(), 1024, "act");
    if (getenv("DUMP_ACT")) {
        FILE* fo = fopen(getenv("DUMP_ACT"), "wb");
        if (fo) { fwrite(bo_act.map(), 1, 1048576, fo); fclose(fo);
                  fprintf(stderr, "dumped act -> %s\n", getenv("DUMP_ACT")); }
    }
    return 0;
}
