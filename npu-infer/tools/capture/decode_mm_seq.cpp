// decode_mm_seq.cpp — generate an mm.xclbin GEMM stream via FLM's libgemm.so
// and decode every shim-DMA BD (dim0/1/2/iter sizes+strides, arg_idx, arg_offset)
// so the exact A-read / W-read / C-write tiling is visible.
//
// Usage: decode_mm_seq <model_dir> <M> <K> <N> <woff> [ooff] [use7arg]
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "modules/gemm.hpp"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

static const char* ARGNAME(uint32_t a) {
    switch (a) { case 0: return "C"; case 1: return "A"; case 2: return "W"; default: return "?"; }
}

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    uint32_t M = (argc > 2) ? atoi(argv[2]) : 256;
    uint32_t K = (argc > 3) ? atoi(argv[3]) : 1024;
    uint32_t N = (argc > 4) ? atoi(argv[4]) : 2048;
    uint32_t woff = (argc > 5) ? atoi(argv[5]) : 0;
    uint32_t ooff = (argc > 6) ? atoi(argv[6]) : 0;
    bool use7 = (argc > 7) ? (atoi(argv[7]) != 0) : true;

    LM_Config config;
    config.from_pretrained(model_dir);
    Gemm gemm(config);
    npu_sequence seq(device_npu2);
    if (use7)
        gemm.generate_seq(&seq, M, K, N, woff, false, Gemm::NO_Activation, 0);
    else
        gemm.generate_seq(&seq, M, K, N, woff, false, Gemm::NO_Activation, 0, ooff);

    auto [ptr, nw] = seq.dump();
    fprintf(stderr, "M=%u K=%u N=%u woff=%u ooff=%u use7=%d -> %zu words\n",
            M, K, N, woff, ooff, use7, nw);

    // decode: 4-word header, then ops. Each op header word distinguishes type.
    // block-write (DMA BD) = 12 words; ddr patch = 12 words; queue write = 6.
    size_t i = 4;
    int opn = 0;
    while (i < nw) {
        uint32_t op = ptr[i];
        // op headers: WRITE=0 (queue, 6 words), BLOCKWRITE=1 (DMA BD, 12),
        // MASKWRITE=3 (issue token), TCT=0x80 (wait), DDR_PATCH=0x81 (12).
        if (op == 1 /* XAIE_IO_BLOCKWRITE */ && i + 12 <= nw) {
            uint32_t w2 = ptr[i+2];
            uint32_t row = (w2 >> 20) & 0x1F, col = (w2 >> 25) & 0x7F, bd = (w2 >> 5) & 0xF;
            uint32_t blen = ptr[i+4];
            uint32_t boff = ptr[i+5];
            uint32_t d0s = (ptr[i+7] >> 20) & 0x3FF, d0st = ((ptr[i+7]) & 0xFFFFF) + 1;
            uint32_t d1s = (ptr[i+8] >> 20) & 0x3FF, d1st = ((ptr[i+8]) & 0xFFFFF) + 1;
            uint32_t d2st = ((ptr[i+9]) & 0xFFFFF) + 1;
            uint32_t d2s = (d0s && d1s) ? blen / (d0s * d1s) : 0;
            uint32_t isz = ((ptr[i+10] >> 26) & 0x3FF) + 1, ist = ((ptr[i+10]) & 0xFFFFF) + 1;
            printf("BD  op%03d @%zu row=%u col=%u bd=%u bufLen=%u bufOff=%u | d0=%u/%u d1=%u/%u d2=%u/%u iter=%u/%u\n",
                   opn, i, row, col, bd, blen, boff, d0s, d0st, d1s, d1st, d2s, d2st, isz, ist);
            i += 12; opn++;
        } else if (op == 0x81 /* DDR_PATCH */ && i + 12 <= nw) {
            uint32_t w6 = ptr[i+6];
            uint32_t row = (w6 >> 20) & 0x1F, col = (w6 >> 25) & 0x7F, bd = ((w6 - 0x04) >> 5) & 0x1F;
            uint32_t argidx = ptr[i+8];
            uint32_t argoff = ptr[i+10];
            printf("PATCH op%03d @%zu row=%u col=%u bd=%u -> arg%d(%s) off=%u B\n",
                   opn, i, row, col, bd, argidx, ARGNAME(argidx), argoff);
            i += 12; opn++;
        } else if (op == 0 /* XAIE_IO_WRITE */ && i + 6 <= nw) {
            uint32_t w2 = ptr[i+2];
            uint32_t row = (w2 >> 20) & 0x1F, col = (w2 >> 25) & 0x7F;
            uint32_t reg = w2 & 0xFFFFF;
            const char* dir = (reg & 0x10) ? "MM2S" : "S2MM";
            printf("QUEUE op%03d @%zu row=%u col=%u reg=0x%05x dir=%s\n", opn, i, row, col, reg, dir);
            i += 6; opn++;
        } else {
            // unknown / other op — print raw header and skip 1 word
            printf("OTHER op%03d @%zu header=0x%08x\n", opn, i, op);
            i += 1; opn++;
        }
    }
    return 0;
}
