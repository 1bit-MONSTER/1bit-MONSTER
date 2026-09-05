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

    // Issue #2103: reject unsupported M geometries up front. FastFlowLM's
    // generate_seq rejects M < 256 ("GEMM M size not aligned with total npu
    // rows") deep inside the library; without this guard the tool used to
    // exit 0 with no .bin written, which looks like success.
    if (M < 256 || (M % 256) != 0) {
        fprintf(stderr,
            "gen_mm_insts: unsupported GEMM M=%u - M must be >= 256 and a "
            "multiple of 256 (256, 512, 1024, 2048, ...); no %s written\n",
            M, argv[2]);
        return 1;
    }

    LM_Config config;
    std::ifstream f(argv[1]);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    f >> config._json_config;

    Gemm gemm(config);
    npu_sequence seq;
    gemm.generate_seq(&seq, M, K, N, woff, add_bias, Gemm::NO_Activation, 0);
    seq.cmds2seq();
    seq.write_out_sequence(argv[2]);

    // Issue #2103: never exit 0 without a usable .bin. generate_seq failures
    // surface inside the prebuilt lib (error text only, no return code), so
    // confirm the instruction stream actually landed on disk.
    {
        std::ifstream chk(argv[2], std::ios::binary | std::ios::ate);
        if (!chk.is_open() || chk.tellg() <= 0) {
            fprintf(stderr,
                "gen_mm_insts: %s was not written (generate_seq failed?) - "
                "exiting non-zero\n", argv[2]);
            return 1;
        }
    }
    fprintf(stderr, "wrote %s for GEMM %ux%ux%u woff=%u bias=%d\n",
            argv[2], M, K, N, woff, (int)add_bias);
    return 0;
}

// Build prerequisites (issue #2107):
//   * -include climits is required: npu_utils/npu_cmd.hpp uses UCHAR_MAX
//     without including <climits> itself, so g++ rejects the include chain
//     unless climits is force-included (-include).
//   * -lq4_npu_eXpress is required in the link: libqwen3_npu.so leaves
//     SafeTensors::~SafeTensors() and load_weights undefined, and only
//     libq4_npu_eXpress.so resolves them.
//   * The include tree must match the FastFlowLM build that produced the
//     prebuilt libs. In this repo that is the third_party/FastFlowLM
//     submodule (src/include + src/include/npu_utils); the dev boxes also
//     carry a separate amd-oss fastflowlm checkout or a /opt/fastflowlm
//     install that works the same way. Adjust -I/-L to the tree actually
//     installed on the machine you build on.
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
