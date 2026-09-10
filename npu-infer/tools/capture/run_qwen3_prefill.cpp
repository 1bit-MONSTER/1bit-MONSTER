// run_qwen3_prefill.cpp — drive the REAL FastFlowLM runtime's batched prefill()
// (mm.xclbin + attn.xclbin path) so an LD_PRELOAD interposer can capture the
// mm.xclbin GEMM instruction streams FLM uploads during load_weights + prefill.
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
    std::string model_dir = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    const char* tokfile = getenv("RT_TOKENS");

    LM_Config config;
    config.from_pretrained(model_dir);
    fprintf(stderr, "config loaded\n");

    xrt::device dev(0);
    npu_xclbin_manager npu(device_npu2, &dev);
    Q4NX q4nx(model_dir);
    qwen3_npu model(config, &npu, 4096);
    fprintf(stderr, "calling load_weights...\n");
    model.load_weights(q4nx);
    fprintf(stderr, "load_weights done\n");

    std::vector<int> prompt;
    if (tokfile) {
        FILE* ft = fopen(tokfile, "r");
        if (ft) { int t; while (fscanf(ft, "%d", &t) == 1) prompt.push_back(t); fclose(ft); }
    } else {
        for (int i = 0; i < 8; i++) prompt.push_back(1000 + i);
    }
    fprintf(stderr, "calling prefill(%zu tokens)...\n", prompt.size());
    auto out = model.prefill(prompt, nullptr);
    fprintf(stderr, "prefill done, out size %zu\n", out.size());
    int best = 0;
    for (size_t j = 1; j < out.size(); j++) if (out[j] > out[best]) best = (int)j;
    fprintf(stderr, "GREEDY_NEXT: %d\n", best);
    return 0;
}
