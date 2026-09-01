// verify_packer.cpp — byte-verify npu_pack_layer_bo against the captured
// runtime weight BO (bo_to_0057_10485760.bin = layer 0).
#include "model.h"
#include <cstdio>
#include <cstring>
#include <vector>
int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    const char* cap_path = (argc > 2) ? argv[2]
        : "/tmp/cap2/bo_to_0057_10485760.bin";
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) return 1;
    std::vector<uint8_t> packed(10485760, 0);
    int layer = (argc > 3) ? atoi(argv[3]) : 0;
    int tiles = npu_pack_layer_bo(packed.data(), mw, &cfg, layer);
    printf("packed %d tiles\n", tiles);
    FILE* f = fopen(cap_path, "rb");
    std::vector<uint8_t> cap(10485760);
    size_t br = fread(cap.data(), 1, 10485760, f); fclose(f);
    printf("read %zu cap bytes\n", br);
    size_t n_diff = 0; size_t first_diff = SIZE_MAX;
    for (size_t i = 0; i < 10485760; i++)
        if (packed[i] != cap[i]) { n_diff++; if (first_diff == SIZE_MAX) first_diff = i; }
    printf("byte-different: %zu / 10485760 (%.4f%%)\n", n_diff, 100.0*n_diff/10485760);
    if (n_diff) printf("first diff at byte %zu\n", first_diff);
    printf("IDENTICAL: %s\n", n_diff == 0 ? "YES" : "NO");
    model_free(mw);
    return 0;
}
