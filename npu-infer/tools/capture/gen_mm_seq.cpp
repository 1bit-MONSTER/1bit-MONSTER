// gen_mm_seq.cpp — generate the mm.xclbin GEMM instruction stream using FLM's
// OWN Gemm::generate_seq (libgemm.so), the same generator the runtime uses for
// prefill. This closes the gap: gemm_generate_sequence_i8 (open-source) emits
// the v26 single-core-row stream, but the mm.xclbin needs FLM's multi-row
// stream — which this tool obtains directly from libgemm.so.
//
// Usage: gen_mm_seq <model_dir> <outdir> <M> <K> <N> [weight_offset]
// Emits <outdir>/mm_M<M>_K<K>_N<N>.bin (raw npu_sequence words) + .txt.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "modules/gemm.hpp"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : "/tmp";
    uint32_t M = (argc > 3) ? atoi(argv[3]) : 128;
    uint32_t K = (argc > 4) ? atoi(argv[4]) : 1024;
    uint32_t N = (argc > 5) ? atoi(argv[5]) : 4096;
    uint32_t woff = (argc > 6) ? atoi(argv[6]) : 0;

    LM_Config config;
    config.from_pretrained(model_dir);
    fprintf(stderr, "config loaded\n");

    Gemm gemm(config);
    fprintf(stderr, "Gemm constructed\n");

    npu_sequence seq(device_npu2);
    gemm.generate_seq(&seq, M, K, N, woff, false, Gemm::NO_Activation, 0);
    fprintf(stderr, "generate_seq done (M=%u K=%u N=%u)\n", M, K, N);

    auto [ptr, nw] = seq.dump();
    fprintf(stderr, "sequence words = %zu\n", nw);

    char fname[256];
    snprintf(fname, sizeof(fname), "%s/mm_M%u_K%u_N%u.bin", outdir.c_str(), M, K, N);
    FILE* f = fopen(fname, "wb");
    if (f) { fwrite(ptr, 4, nw, f); fclose(f); }
    snprintf(fname, sizeof(fname), "%s/mm_M%u_K%u_N%u.txt", outdir.c_str(), M, K, N);
    f = fopen(fname, "w");
    if (f) { for (size_t i = 0; i < nw; i++) fprintf(f, "0x%08x\n", ptr[i]); fclose(f); }
    fprintf(stderr, "wrote %s\n", fname);
    return 0;
}
