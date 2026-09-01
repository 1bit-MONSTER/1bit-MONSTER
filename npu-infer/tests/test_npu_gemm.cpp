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

    // EXPERIMENT (round-27): feed the mm kernel the RAW Q4NX tiles (the
    // kernel dequantizes in-kernel — feeding BF16 makes its tile-dequant clamp
    // everything to zero) and build the reference by dequantizing Q4NX.
    static uint8_t block[1048576];
    const void* wdata = model_tensor_data(mw, q);
    // block 0 = logical rows 0-255 x cols 0-1023 = I8 tile rows 0-31 (grid
    // row-major: tile_row*4 + tile_col), 32 rows * 5120 B = 160 KB.
    memset(block, 0, sizeof(block));
    // EXPERIMENT (round-28): pack the REORDERED Q4NX tiles — the exact layout
    // the runtime's reorder_cpy produces (captured from the real runtime via
    // LD_PRELOAD: out_tile = 8*(ir/8) + 2*tc + (tr%2) where ir = tr*4+tc).
    // The mm kernel dequantizes in-kernel from this layout.
    {
        const uint8_t* data = (const uint8_t*)wdata;
        for (int ir = 0; ir < 32; ir++) {
            int tr = ir / 4, tc = ir % 4;
            int out_tile = 8 * (ir / 8) + 2 * tc + (tr % 2);
            memcpy(block + (size_t)out_tile * 5120, data + (size_t)ir * 5120, 5120);
        }
        printf("packed REORDERED Q4NX tiles (32 x 5120 B) to weight BO\n");
    }
    printf("(f32 pack path disabled)\n");

    // Reference: dequantize Q4NX block 0 -> F32 [256,1024] -> y = W @ x
    float Wf[256 * 1024];
    {
        const uint8_t* data = (const uint8_t*)wdata;
        for (int r = 0; r < 256; r++) {
            int tile_row = r / 32, lr = r % 32;
            for (int c = 0; c < 1024; c++) {
                int tile_col = c / 256, cc = c % 256, g = cc / 32;
                const uint8_t* row = data + (size_t)(tile_row * 4 + tile_col) * 5120;
                int so = (g * 32 + lr) * 2;
                uint32_t sb = ((uint32_t)(row[so] | (row[so+1] << 8))) << 16;
                float scale; memcpy(&scale, &sb, 4);
                uint32_t zb = ((uint32_t)(row[512+so] | (row[512+so+1] << 8))) << 16;
                float zp; memcpy(&zp, &zb, 4);
                if (!std::isfinite(scale) || fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zp) || fabs(zp) > 100.0f) zp = 0.0f;
                int lane = lr / 16, byte_idx = (lr % 16) / 2, nib = lr % 2;
                uint8_t b = row[1024 + lane * 2048 + cc * 8 + byte_idx];
                int qv = (nib == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                // FastFlowLM Qwen3 q4nx formula (verified bit-exact vs the
                // runtime's q4nx_dequantize): W = (q - zp) * scale, scales and
                // zero-points bf16 at GROUP-major index (g*32 + lr).
                Wf[r * 1024 + c] = ((float)qv - zp) * scale;
            }
        }
    }
    float x[1024], ref[256];
    for (int i = 0; i < 1024; i++) x[i] = (i % 7) * 0.01f - 0.3f;
    for (int r = 0; r < 256; r++) {
        double acc = 0;
        for (int c = 0; c < 1024; c++) acc += (double)Wf[r*1024 + c] * x[c];
        ref[r] = (float)acc;
    }

    // NPU setup
    xrt::device dev(0);
    const char* xclbin_path = getenv("MM_XCLBIN")
        ? getenv("MM_XCLBIN")
        : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/mm.xclbin";
    FILE* f = fopen(xclbin_path, "rb"); fseek(f, 0, SEEK_END);
    long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    xrt::kernel kern(dev, xclbin->get_uuid(), "MLIR_AIE");

    // amdxdna binds host BOs to kernel arg slots via group_id (opcode=0,
    // instr=1, ninstr=2, host buffers from slot 3). group-0 BOs are silently
    // ignored by this kernel. The instruction stream is REQUIRED — without it
    // the ERT command completes but the AIE never executes.
    xrt::bo act(dev, cfg.npu_activation_bo_size, xrt::bo::flags::host_only, kern.group_id(3));
    xrt::bo ws(dev, 10485760, xrt::bo::flags::host_only, kern.group_id(4));
    xrt::bo wt(dev, cfg.npu_weight_bo_size, xrt::bo::flags::host_only, kern.group_id(5));
    xrt::bo kv(dev, cfg.npu_kv_cache_bo_size, xrt::bo::flags::host_only, kern.group_id(7));

    // Instruction stream: <xclbin>.bin (generate with tools/gen_mm_insts).
    std::string insts_path(xclbin_path);
    size_t dot = insts_path.rfind('.'); if (dot != std::string::npos) insts_path = insts_path.substr(0, dot);
    insts_path += ".bin";
    FILE* fi = fopen(insts_path.c_str(), "rb");
    if (!fi) { fprintf(stderr, "no insts at %s — run tools/gen_mm_insts\n", insts_path.c_str()); return 1; }
    fseek(fi, 0, SEEK_END); long isz = ftell(fi); fseek(fi, 0, SEEK_SET);
    std::vector<uint32_t> instrs(isz / 4);
    size_t br = fread(instrs.data(), 4, instrs.size(), fi); fclose(fi);
    if (br != instrs.size()) { fprintf(stderr, "short insts read\n"); return 1; }
    xrt::bo bo_instr(dev, isz, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    memcpy(bo_instr.map(), instrs.data(), isz);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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

    auto run = kern((uint64_t)3, bo_instr, (uint32_t)instrs.size(),
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
