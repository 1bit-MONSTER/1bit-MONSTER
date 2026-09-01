// run_qwen3_npu.cpp — minimal harness that drives the REAL FastFlowLM runtime
// (qwen3_npu in libqwen3_npu.so) through load_weights + forward, without the
// AutoModel/tokenizer chat machinery. Purpose: reproduce the runtime's
// load-time dequant/weight-prep exactly, so an LD_PRELOAD interposer on
// xrt::bo::sync / the kernel submit can capture the actual dequant TXNs and
// weight BO contents (the runtime layer-TXN weight-BD decode, #2006/#2015).
//
// Build:
//   g++ -O2 -std=c++20 -include climits run_qwen3_npu.cpp -o run_qwen3_npu \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//     -lqwen3_npu -lq4_npu_eXpress -lgemm -ldequant -lmha -llm_head \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
// Run:
//   ./run_qwen3_npu <model_dir> [n_tokens]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "npu_utils/npu_utils_xrt.hpp"
#include <xrt/xrt_device.h>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen3/qwen3_npu.hpp"
#include "lm_config.hpp"

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
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

    // 4. the runtime model — load_weights runs the reorder + dequant path
    qwen3_npu model(config, &npu, 4096);
    fprintf(stderr, "qwen3_npu constructed; calling load_weights...\n");
    model.load_weights(q4nx);
    fprintf(stderr, "load_weights done\n");

    // 5. one forward step (decode) — submits the layer TXNs
    for (int i = 0; i < n_tokens; i++) {
        int tok = 1000 + i;
        auto out = model.forward(tok);
        fprintf(stderr, "forward(%d) done, out size %zu\n", tok, out.size());
        char fname[64];
        snprintf(fname, sizeof(fname), "/tmp/txn_decode/logits_%d.bin", tok);
        FILE* f = fopen(fname, "wb");
        if (f && out.size()) {
            fwrite(out.data(), sizeof(bf16), out.size(), f);
            fclose(f);
            fprintf(stderr, "saved logits to %s (%zu bf16)\n", fname, out.size());
        }
    }

    fprintf(stderr, "DONE\n");
    return 0;
}
