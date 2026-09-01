// test_npu_layer_txn.cpp — on-device validation of the CAPTURED runtime layer
// TXN (from xrt::elf hook, npu-infer/captures/txn-elfs/layer_ctx1_ctrl.bin):
// submit the runtime's ACTUAL per-call layer instruction stream with the
// hand-rolled packed weight BO (npu_pack_layer_bo) + the runtime's act/kv
// buffers, and check the NPU executes it (output changes / finite values /
// matches the runtime's next hidden state when fed the same input).
//
// Layer kernel BO binding (from the runtime's runlist arms, cap manifest):
//   run.set_arg(0,3); set_arg(1,0); set_arg(2,0);          // opcode/instr/ninstr
//   set_arg(3, act 1MB)  set_arg(4, weight 10MB)
//   set_arg(5, out1 1MB) set_arg(6, out2 1MB) set_arg(7, kv 32MB)
// The TXN's DDR_PATCH arg indices count host BOs from 0 (amdxdna) so
//   TXN arg0 = run arg3 (act), arg1 = run arg4 (weight),
//   arg2 = run arg5, arg3 = run arg6, arg4 = run arg7 (kv).
//
// Usage:
//   LD_LIBRARY_PATH=/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//   ./test_npu_layer_txn <model_dir> [captured_txn] [layer_idx]
// Build: g++ -O2 -std=c++17 test_npu_layer_txn.cpp -o test_npu_layer_txn \
//   -I<repo>/npu-infer/include -I<repo>/src \
//   -L<fastflowlm>/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress -lxrt_coreutil -lxrt_core \
//   -Wl,-rpath,<fastflowlm>/lib/xrt   (plus npu-infer engine sources)
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
    const char* model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    const char* txn_path = (argc > 2) ? argv[2]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs/layer_ctx1_ctrl.bin";
    int layer_idx = (argc > 3) ? atoi(argv[3]) : 0;

    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    std::string model_file = std::string(model_dir) + "/model.q4nx";
    ModelWeights* mw = model_load(model_file.c_str(), cfg);
    if (!mw) return 1;

    // 1. pack the layer weight BO (the hand-rolled layout, byte-verified
    //    28/28 vs the runtime's captured 10MB per-layer BOs)
    const size_t layer_bo_bytes = 9830400;  // 1920 tiles x 5120 B (NPU_LAYER_BO_BYTES)
    std::vector<uint8_t> wbo(10485760);      // runtime BO is 10 MB (padded)
    int tiles = npu_pack_layer_bo(wbo.data(), mw, &cfg, layer_idx);
    if (tiles <= 0) {
        fprintf(stderr, "npu_pack_layer_bo failed (rc=%d)\n", tiles); return 1;
    }
    printf("packed layer %d weight BO (%zu B, %zu used)\n", layer_idx, wbo.size(), layer_bo_bytes);

    // 2. load the CAPTURED runtime layer TXN (ELF .ctrltext = raw TXN words)
    FILE* ft = fopen(txn_path, "rb");
    if (!ft) { fprintf(stderr, "no captured txn at %s\n", txn_path); return 1; }
    fseek(ft, 0, SEEK_END); long tsz = ftell(ft); fseek(ft, 0, SEEK_SET);
    std::vector<uint32_t> txn(tsz / 4);
    size_t br = fread(txn.data(), 4, txn.size(), ft); fclose(ft);
    if (br != txn.size()) { fprintf(stderr, "short txn read\n"); return 1; }
    printf("loaded captured layer TXN (%ld words, %ld B)\n", txn.size(), tsz);

    // 3. NPU setup — layer.xclbin MLIR_AIE kernel
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
    xrt::kernel kern(dev, xclbin->get_uuid(), "MLIR_AIE");

    // 4. BOs — the runtime's layer binding (arg3..arg7)
    xrt::bo bo_act(dev, 1048576, xrt::bo::flags::host_only, kern.group_id(3));
    xrt::bo bo_w(dev, 10485760, xrt::bo::flags::host_only, kern.group_id(4));
    xrt::bo bo_o1(dev, 1048576, xrt::bo::flags::host_only, kern.group_id(5));
    xrt::bo bo_o2(dev, 1048576, xrt::bo::flags::host_only, kern.group_id(6));
    xrt::bo bo_kv(dev, cfg.npu_kv_cache_bo_size, xrt::bo::flags::host_only, kern.group_id(7));
    xrt::bo bo_instr(dev, tsz, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));

    memcpy(bo_instr.map(), txn.data(), tsz);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_w.map(), wbo.data(), wbo.size());
    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 5. act input: token embedding-ish bf16 [1024] at offset 0 (the runtime's
    //    act buffer held 1024 bf16 = 2048 B). Use the packed token's embedding
    //    if available, else a deterministic vector.
    memset(bo_act.map(), 0, 1048576);
    memset(bo_o1.map(), 0, 1048576);
    memset(bo_o2.map(), 0, 1048576);
    memset(bo_kv.map(), 0, cfg.npu_kv_cache_bo_size);
    uint16_t* am = (uint16_t*)bo_act.map();
    {
        // try the runtime's captured act (npu-infer/captures/ or env ACT_CAP)
        const char* actcap = getenv("ACT_CAP");
        if (actcap) {
            FILE* fa = fopen(actcap, "rb");
            if (fa) { size_t g = fread(am, 2, 1024, fa); fclose(fa); printf("loaded act from %s (%zu bf16)\n", actcap, g); }
        } else {
            for (int i = 0; i < 1024; i++) am[i] = f_to_bf16(sinf(i * 0.01f));
            printf("using synthetic act (sin) — set ACT_CAP for the runtime's real input\n");
        }
    }
    bo_act.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 6. run the captured layer TXN (engine ABI: (3, instrs, ninstr, ...))
    xrt::run run = kern((uint64_t)3, bo_instr, (uint32_t)txn.size(),
                        bo_act, bo_w, bo_o1, bo_o2, bo_kv);
    run.wait();

    // 7. read back and report
    bo_act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_o1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_o2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_kv.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto stats = [](const uint16_t* p, size_t n, const char* name) {
        size_t nz = 0; float s = 0, s2 = 0, mn = 1e30f, mx = -1e30f;
        for (size_t i = 0; i < n; i++) {
            float v = bf16_to_f(p[i]);
            if (p[i]) nz++;
            s += v; s2 += v * v;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        double mean = n ? s / n : 0, var = n ? s2 / n - mean * mean : 0;
        printf("%-6s nonzero=%6zu mean=%+.4f std=%.4f min=%.4f max=%.4f\n",
               name, nz, mean, var > 0 ? sqrt(var) : 0, mn, mx);
    };
    stats((const uint16_t*)bo_act.map(), 524288, "act");
    stats((const uint16_t*)bo_o1.map(), 524288, "out1");
    stats((const uint16_t*)bo_o2.map(), 524288, "out2");
    stats((const uint16_t*)bo_kv.map(), cfg.npu_kv_cache_bo_size / 2, "kv");

    if (getenv("DUMP_OUT")) {
        FILE* fo = fopen(getenv("DUMP_OUT"), "wb");
        if (fo) { fwrite(bo_o1.map(), 1, 1048576, fo); fclose(fo);
                  fprintf(stderr, "dumped out1 -> %s\n", getenv("DUMP_OUT")); }
    }

    model_free(mw);
    return 0;
}
