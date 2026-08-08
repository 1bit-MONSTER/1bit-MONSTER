// Issue #1527 runtime check: a dense non-Zaya .1bp claimed by the HIP backend
// must now FAIL LOUDLY in load_layer_onebp (partial attention group, no MoE
// sublayer) instead of zero-filling every weight and "succeeding" with
// garbage logits. Config is built from the file's real header dims so the
// check exercises the loader, not the header gate.
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include "zaya_engine.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "SKIP: %s not present\n", path);
        return 0;
    }
    // models/Qwen3-0.6B.1bp header: H=1024 L=28 NH=16 NKV=8 HD=128 IM=3072
    // V=151936, dense (num_experts=0).
    ZayaConfig cfg = ZayaConfig::from_model(1024, 28, 16, 8, 128, 151936,
                                            /*n_exp=*/0, /*n_ff=*/3072, /*rtr_h=*/256, /*max_seq=*/4096);
    ZayaState* s = zaya_init_onebp(path, &cfg);
    if (s) {
        fprintf(stderr, "FAIL: zaya_init_onebp succeeded on a non-Zaya dense model (must abort loudly)\n");
        zaya_destroy(s);
        return 1;
    }
    fprintf(stderr, "PASS: zaya_init_onebp rejected the non-Zaya model with a loud abort\n");
    return 0;
}
