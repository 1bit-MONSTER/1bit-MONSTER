// test_qrowmap.cpp — definitive Q GEMM row-mapping probe.
// Runs the mm.xclbin Q GEMM (M=256,K=1024,N=2048) with a crafted A and W and
// reports exactly which GEMM C row lands in which C-BO row.
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
#include "modules/dequant.hpp"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

// bf16 helpers
static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x8000)>>16); }
static float b2f(uint16_t h){ uint32_t u=((uint32_t)h)<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc, char** argv) {
    std::string model_dir = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string xdir = "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2";

    xrt::device dev(0);
    LM_Config config; config.from_pretrained(model_dir);
    xrt::xclbin mm_xc(xdir + "/mm.xclbin"); dev.register_xclbin(mm_xc);
    xrt::hw_context mm_hc(dev, mm_xc.get_uuid());
    npu_app app(device_npu2, &dev, &mm_hc, "MLIR_AIE");
    Gemm gemm(config);

    const int M=256, K=1024, N=2048;
    gemm.generate_seq(app.seq(), M, K, N, 0, false, Gemm::NO_Activation, 0);
    app.update_ctrl_seq();

    auto bA = app.create_bo_buffer<uint16_t>((size_t)M*K);
    auto bW = app.create_bo_buffer<uint16_t>((size_t)2048*2048);
    auto bC = app.create_bo_buffer<uint16_t>((size_t)M*2048);

    // A[k][0] = k+1 (exact bf16 integer), rest 0.
    uint16_t* A = bA.data(); memset(A,0,(size_t)M*K*2);
    for (int k=0;k<M;k++) A[k*K+0] = f2b((float)(k+1));
    // W[0][0] = 1.0, rest 0  =>  C[k] = [k+1, 0, 0, ...]
    uint16_t* W = bW.data(); memset(W,0,(size_t)2048*2048*2);
    W[0*2048+0] = f2b(1.0f);
    uint16_t* C = bC.data(); memset(C,0,(size_t)M*2048*2);

    app.safe_run(bC, bA, bW);

    // Report: for each C-BO row r (of 2048), what marker is at Q[r][0]?
    printf("== C-BO row (256x2048) -> marker (k+1) at col0 ==\n");
    int nz=0;
    for (int r=0;r<M;r++){
        float v = b2f(C[r*2048+0]);
        if (v != 0.0f) {
            printf("row %3d : marker %.0f  (=> GEMM C row %d)\n", r, v, (int)(v-1));
            nz++;
        }
    }
    printf("non-zero rows: %d / %d\n", nz, M);
    // also scan ALL rows x cols for any non-zero anywhere (to catch split-half writes)
    int totalnz=0, first_row=-1, last_row=-1;
    for (int r=0;r<M;r++) for (int c=0;c<N;c++) if (C[r*2048+c]!=0){ totalnz++; if(first_row<0)first_row=r; last_row=r; }
    printf("total non-zero elements: %d (first row %d, last row %d)\n", totalnz, first_row, last_row);
    return 0;
}
