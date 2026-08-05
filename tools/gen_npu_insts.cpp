// gen_npu_insts.cpp — generate insts_i8_*.txt from xclbin + dimensions.
//
// Uses gemm_generate_sequence_i8() to produce instruction sequences for
// each NPU op (QKV, O, G, GU, D, U) at the given model dimensions, and
// writes them as .txt files (binary blob_instr_transaction format) that
// npu_engine_universal's I8Ctx::init() can load directly.
//
// Build:
//   g++ -std=c++20 -O2 -o build/gen_npu_insts tools/gen_npu_insts.cpp \
//     engine/npu/src/gemm_npu_instructions.cpp \
//     -I engine/npu/src -I npu-infer/include \
//     -I third_party/FastFlowLM/src/include \
//     -I third_party/FastFlowLM/src/include/npu_utils \
//     -luuid
//
// Run:
//   ./build/gen_npu_insts <H> <NH> <NKV> <HD> <IM> <IM_EXP> <out_dir> <model_tag>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#include "npu_utils/npu_instr_utils.hpp"

// INT8 GEMM generator (defined in gemm_npu_instructions.cpp)
void gemm_generate_sequence_i8(
    npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N,
    uint32_t a_ddr_offset, uint32_t b_base_offset,
    bool add_bias, int activation, uint32_t bias_offset, uint32_t output_offset);

static void generate(const char* out_dir, const char* tag,
                     const char* op_name, int M, int K, int N) {
    std::string path = std::string(out_dir) + "/insts_i8_" + op_name + "_" + tag + ".txt";
    npu_sequence seq(device_npu2);
    gemm_generate_sequence_i8(&seq, (uint32_t)M, (uint32_t)K, (uint32_t)N,
                              0, 0, false, 0, 0, 0);
    seq.cmds2seq();
    auto [dp, sz] = seq.dump();

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "FAIL: can't write %s\n", path.c_str()); return; }
    fwrite(dp, 1, sz, f);
    fclose(f);
    printf("  %s: M=%d K=%d N=%d → %zu bytes\n", op_name, M, K, N, sz);
}

int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "Usage: gen_npu_insts H NH NKV HD IM IM_EXP out_dir model_tag\n");
        fprintf(stderr, "  H      = hidden_size (e.g. 2048)\n");
        fprintf(stderr, "  NH     = num_attention_heads (e.g. 32)\n");
        fprintf(stderr, "  NKV    = num_kv_heads (e.g. 16)\n");
        fprintf(stderr, "  HD     = head_dim (e.g. 128)\n");
        fprintf(stderr, "  IM     = intermediate_size (e.g. 4096 for dense MLP)\n");
        fprintf(stderr, "  IM_EXP = per-expert intermediate (e.g. 512 for MoE)\n");
        return 1;
    }
    int H      = atoi(argv[1]);
    int NH     = atoi(argv[2]);
    int NKV    = atoi(argv[3]);
    int HD     = atoi(argv[4]);
    int IM     = atoi(argv[5]);
    int IM_EXP = atoi(argv[6]);
    const char* out_dir = argv[7];
    const char* tag     = argv[8];

    int QOUT = NH * HD;           // Q output dim
    int KVOUT = NKV * HD;         // K/V output dim
    int qkv_n = QOUT + 2 * KVOUT; // fused QKV output
    int XM = 128;                  // batch tile (M dimension for GEMM)

    // Each GEMM: activations[M, K] × weights[K, N] → output[M, N]
    // Instructions configure the NPU for these dims.
    // a_ddr_offset=0 (activations BO), b_base_offset=0 (weights BO)

    // QKV: [XM, H] × [H, qkv_n] → [XM, qkv_n]
    generate(out_dir, tag, "QKV", XM, H, qkv_n);

    // O: [XM, NH*HD] × [NH*HD, H] → [XM, H]
    generate(out_dir, tag, "O", XM, QOUT, H);

    // G (Gate for gu_split): [XM, H] × [H, IM] → [XM, IM]
    generate(out_dir, tag, "G", XM, H, IM);

    // GU (Gate+Up fused): [XM, H] × [H, 2*IM] → [XM, 2*IM]
    generate(out_dir, tag, "GU", XM, H, 2 * IM);

    // D (Down): [XM, IM] × [IM, H] → [XM, H]
    generate(out_dir, tag, "D", XM, IM, H);

    // U (Up for gu_split): [XM, H] × [H, IM] → [XM, IM]
    generate(out_dir, tag, "U", XM, H, IM);

    // MoE per-expert G+U: [XM, H] × [H, 2*IM_EXP] → [XM, 2*IM_EXP]
    // Per-expert D: [XM, IM_EXP] × [IM_EXP, H] → [XM, H]
    // These are for the MoE FFN when running per-expert GEMMs.
    // For 8 active experts concatenated:
    int TOP_K = 8;
    generate(out_dir, tag, "MOE_GU", XM, H, TOP_K * 2 * IM_EXP);
    generate(out_dir, tag, "MOE_D",  XM, TOP_K * IM_EXP, H);
    generate(out_dir, tag, "MOE_SGU", XM, H, 2 * IM_EXP);
    generate(out_dir, tag, "MOE_SD",  XM, IM_EXP, H);

    printf("Done. %d files written to %s/\n", 10, out_dir);
    return 0;
}
