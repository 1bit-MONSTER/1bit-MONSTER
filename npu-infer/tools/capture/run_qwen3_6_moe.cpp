// run_qwen3_6_moe.cpp — minimal harness that drives the REAL FastFlowLM
// runtime for the hybrid MoE model (qwen3_6_moe_npu in libqwen3_6_moe_npu.so)
// through load_weights + forward/prefill, without AutoModel/tokenizer chat
// machinery. Purpose: reproduce the runtime's load-time dequant/weight-prep
// (reorder) exactly and capture the layer TXNs via the LD_PRELOAD interposer
// (cap_interposer.so) — the 35B-A3B hybrid (GateDeltaNet linear-attn +
// full-attn + 256-expert MoE) lane of #2006/#2015.
//
// Build:
//   g++ -O2 -std=c++20 -include climits run_qwen3_6_moe.cpp -o run_qwen3_6_moe \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//     -lqwen3_6_moe_npu -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
// Run:
//   ./run_qwen3_6_moe <model_dir> [n_tokens]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "npu_utils/npu_utils_xrt.hpp"
#include <xrt/xrt_device.h>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen3_6_moe/qwen3_6_moe_npu.hpp"
#include "lm_config.hpp"

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2";
    int n_tokens = (argc > 2) ? atoi(argv[2]) : 4;

    // 1. config
    LM_Config config;
    config.from_pretrained(model_dir);
    fprintf(stderr, "config: %s\n", config._str().c_str());

    // 2. NPU manager (XRT backend)
    xrt::device dev(0);
    npu_xclbin_manager npu(device_npu2, &dev);
    fprintf(stderr, "npu_xclbin_manager created\n");

    // 3. Q4NX loader — this reads model.q4nx
    std::string model_file = model_dir;  // Q4NX ctor expects the dir
    Q4NX q4nx(model_file);
    fprintf(stderr, "Q4NX loaded: %s\n", model_file.c_str());

    // 4. the runtime model — load_weights runs the reorder + dequant path.
    //    KNOWN ISSUE (round 9c495d85): qwen3_6_reorder_cpy segfaults with a
    //    garbage size in the precompiled lib — keep this harness as the
    //    reproducer.
    qwen3_6_moe_npu model(config, &npu, 4096);
    fprintf(stderr, "qwen3_6_moe_npu constructed; calling load_weights...\n");
    model.load_weights(q4nx);
    fprintf(stderr, "load_weights done\n");

    // 5. decode/prefill steps. NPU_PROMPT_IDS (comma-separated) feeds a real
    //    prompt stream; greedy sampling reports the next token.
    const char* pids = getenv("NPU_PROMPT_IDS");
    std::vector<int> prompt;
    if (pids && *pids) {
        std::string s(pids);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t c = s.find(',', pos);
            prompt.push_back(atoi(s.substr(pos, c - pos).c_str()));
            if (c == std::string::npos) break;
            pos = c + 1;
        }
    } else {
        for (int i = 0; i < n_tokens; i++) prompt.push_back(1000 + i);
    }

    // prefill-style batch (the runtime's forward path for a prompt stream)
    auto out = model.prefill(prompt);
    fprintf(stderr, "prefill(%zu tokens) done, out size %zu\n",
            prompt.size(), out.size());
    if (out.size()) {
        char fname[64];
        snprintf(fname, sizeof(fname), "/tmp/txn_decode/moe_logits.bin");
        FILE* f = fopen(fname, "wb");
        if (f) {
            fwrite(out.data(), sizeof(bf16), out.size(), f);
            fclose(f);
            fprintf(stderr, "saved logits to %s (%zu bf16)\n", fname, out.size());
        }
        int best = 0;
        for (size_t j = 1; j < out.size(); j++) if (out[j] > out[best]) best = (int)j;
        fprintf(stderr, "GREEDY_NEXT: %d\n", best);
    }

    fprintf(stderr, "DONE\n");
    return 0;
}
