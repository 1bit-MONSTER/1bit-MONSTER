// moe_smoke.cpp — NPU smoke test for MoERuntimeLayerEngine: load the 35B
// layer.xclbin + task-1 ELFs, pack the MoE weight BOs, run one layer + lm_head
// in a single xrt::runlist, and report whether it executes (non-crash) and
// whether the logits are non-NaN.
//
// Build (runlist XRT 2.26.0):
//   g++ -std=c++17 -O2 -I npu-infer/include moe_smoke.cpp npu-infer/src/model.c \
//       npu-infer/src/runtime_layer_moe.cpp -o moe_smoke \
//       -L/usr/local/xrt-runlist/lib -l:libxrt_coreutil.so.2 -l:libxrt_core.so.2 \
//       -Wl,-rpath,/usr/local/xrt-runlist/lib -laiebu -luuid -lm -ldl -pthread
#include "runtime_layer_moe.h"
#include "model.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_device.h>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    const char* elf_dir = argc > 2 ? argv[2]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs-moe35b";
    const char* lm_elf = argc > 3 ? argv[3]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs-moe35b/moe_lm_head.elf";
    const char* xclbin = argc > 4 ? argv[4]
        : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3.6-35B-A3B-NPU2/layer.xclbin";

    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) { fprintf(stderr, "model_load failed\n"); return 1; }
    fprintf(stderr, "model: %d layers, hidden %d, vocab %d\n",
            mw->config.num_layers, mw->config.hidden_size, mw->config.vocab_size);

    xrt::device dev(0);
    fprintf(stderr, "device opened\n");

    MoERuntimeLayerEngine eng;
    if (!eng.init(dev, mw, mw->config, elf_dir, lm_elf, xclbin)) {
        fprintf(stderr, "MoERuntimeLayerEngine::init FAILED\n");
        return 1;
    }
    fprintf(stderr, "engine init OK\n");

    // token 151644 (the reference prompt's first token)
    if (!eng.embed(151644)) { fprintf(stderr, "embed failed\n"); return 1; }
    fprintf(stderr, "embed done; running forward(1)...\n");

    bool ok = eng.forward(1);
    fprintf(stderr, "forward(1): %s\n", ok ? "EXECUTED" : "FAILED");
    if (!ok) return 1;

    int vocab = mw->config.vocab_size;
    std::vector<float> logits(vocab);
    eng.get_logits(logits.data(), vocab);
    int nan = 0; int argmax = 0; float mx = logits[0];
    for (int i = 0; i < vocab; i++) {
        if (!std::isfinite(logits[i])) nan++;
        if (logits[i] > mx) { mx = logits[i]; argmax = i; }
    }
    fprintf(stderr, "logits: argmax=%d max=%.4f NaN=%d (of %d)\n",
            argmax, mx, nan, vocab);
    fprintf(stderr, "reference: greedy next token = 76740\n");
    fprintf(stderr, "DONE\n");
    model_free(mw);
    return 0;
}
