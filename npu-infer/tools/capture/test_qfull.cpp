// test_qfull.cpp — verify the FULL Q GEMM recipe: two invocations of the
// mm.xclbin Q GEMM (each producing 128 correct tokens in rows 0..127) with the
// second shifted by output_offset, combined = full 256-token Q.
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <xrt/xrt_device.h>
#include "npu_utils/npu_instr_utils.hpp"
#include "npu_utils/npu_utils_xrt.hpp"
#include "lm_config.hpp"
#include "modules/gemm.hpp"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x8000)>>16); }
static float b2f(uint16_t h){ uint32_t u=((uint32_t)h)<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc, char** argv) {
    std::string model_dir = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string xdir = "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2";

    xrt::device dev(0);
    LM_Config config; config.from_pretrained(model_dir);
    xrt::xclbin mm_xc(xdir + "/mm.xclbin"); dev.register_xclbin(mm_xc);
    xrt::hw_context mm_hc(dev, mm_xc.get_uuid());
    Gemm gemm(config);

    const int M=256, K=1024, N=2048;

    auto run = [&](const uint16_t* A, uint32_t ooff_elems, uint16_t* Cbuf /*2MB*/){
        npu_app app(device_npu2, &dev, &mm_hc, "MLIR_AIE");
        gemm.generate_seq(app.seq(), M, K, N, 0, false, Gemm::NO_Activation, 0, ooff_elems);
        app.update_ctrl_seq();
        auto bA = app.create_bo_buffer<uint16_t>((size_t)M*K);
        auto bW = app.create_bo_buffer<uint16_t>((size_t)2048*2048);
        auto bC = app.create_bo_buffer<uint16_t>((size_t)512*2048);   // 2 MB
        memcpy(bA.data(), A, (size_t)M*K*2);
        uint16_t* W = bW.data(); memset(W,0,(size_t)2048*2048*2); W[0] = f2b(1.0f);
        memset(bC.data(), 0, (size_t)512*2048*2);
        app.safe_run(bC, bA, bW);
        memcpy(Cbuf, bC.data(), (size_t)512*2048*2);
    };

    // A: A[k][0] = k+1  (marker). Invocation 1 = full 256 rows; inv 2 = shifted by 128.
    std::vector<uint16_t> A(M*K, 0), Ashift(M*K, 0);
    for (int k=0;k<M;k++) A[k*K+0] = f2b((float)(k+1));
    for (int k=0;k<128;k++) Ashift[k*K+0] = f2b((float)(k+1+128));  // tokens 128..255

    std::vector<uint16_t> Cbuf(512*2048, 0);
    run(A.data(), 0, Cbuf.data());                  // inv1 -> rows 0..127 correct
    run(Ashift.data(), 262144, Cbuf.data());        // inv2 -> rows 128..255 correct

    // Check rows 0..255 (of 2048) == marker k+1
    int ok=0, bad=0;
    for (int k=0;k<256;k++){
        float v = b2f(Cbuf[k*2048+0]);
        if (v == (float)(k+1)) ok++; else { if (bad<20) printf("row %d: got %.0f want %.0f\n", k, v, (float)(k+1)); bad++; }
    }
    printf("FULL Q rows correct: %d/256 (bad %d)\n", ok, bad);
    // also verify rows 256..511 (overflow region) are zero/ignored
    int nz=0; for (int k=256;k<512;k++) if (Cbuf[k*2048+0]!=0) nz++;
    printf("overflow rows 256..511 non-zero: %d\n", nz);
    return 0;
}
