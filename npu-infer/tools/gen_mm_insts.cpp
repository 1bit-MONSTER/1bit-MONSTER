// Generate the instruction stream (insts) for a FastFlowLM GEMM xclbin.
// Links against FastFlowLM's prebuilt libraries.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include "npu_utils/npu_instr_utils.hpp"
#include "modules/gemm.hpp"
#include "lm_config.hpp"

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: gen_mm_insts <config.json> <out.bin> M K N <weight_offset> [bias]\n");
        return 1;
    }
    uint32_t M = (uint32_t)strtoul(argv[3], nullptr, 10);
    uint32_t K = (uint32_t)strtoul(argv[4], nullptr, 10);
    uint32_t N = (uint32_t)strtoul(argv[5], nullptr, 10);
    uint32_t woff = (uint32_t)strtoul(argv[6], nullptr, 10);
    bool add_bias = (argc > 7) && (strtoul(argv[7], nullptr, 10) != 0);

    LM_Config config;
    std::ifstream f(argv[1]);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    f >> config._json_config;

    Gemm gemm(config);
    npu_sequence seq;
    gemm.generate_seq(&seq, M, K, N, woff, add_bias, Gemm::NO_Activation, 0);
    seq.cmds2seq();
    seq.write_out_sequence(argv[2]);
    fprintf(stderr, "wrote %s for GEMM %ux%ux%u woff=%u bias=%d\n",
            argv[2], M, K, N, woff, (int)add_bias);
    return 0;
}

// Build:
//   g++ -O2 -std=c++17 -include climits -mavx2 gen_mm_insts.cpp -o gen_mm_insts \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
// Run:
//   ./gen_mm_insts <model-config.json> <out.bin> M K N <weight_offset> [bias]
// Place the output next to the xclbin as <xclbin>.bin — the engine's
// XclbinManager loads it automatically (without insts the kernel is a
// silent no-op: ERT completes, AIE never executes).
