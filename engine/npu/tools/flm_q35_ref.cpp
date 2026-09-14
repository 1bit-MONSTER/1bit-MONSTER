// flm_q35_ref.cpp — tiny harness: load Qwen3.5-4B via FLM's qwen3_5vl_npu and
// run a prefill to get the reference boot token + timing. This proves FLM's own
// dequant handles the 4736-byte "I8" tiles and gives a reference for the native
// dequant (goal mttxt22c-a6rv75).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

// Provided by flm_prefill_bridge.cpp
extern "C" int flm_prefill_init(const char* model_dir, const char* family);
extern "C" int flm_prefill_run(const int* ids, int n, int* boot_token, double* prefill_ms);

namespace utils { std::string find_xclbin_path() {
    if (const char* p = getenv("FLM_ROOT")) return p;
    return "/home/bcloud/.local/flm-v0946";
} }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <ids_file> [family]\n", argv[0]);
        return 2;
    }
    const char* model_dir = argv[1];
    const char* ids_file = argv[2];
    const char* family = argc > 3 ? argv[3] : "qwen3_5vl";

    std::vector<int> ids;
    std::ifstream f(ids_file);
    int t;
    while (f >> t) ids.push_back(t);
    if (ids.empty()) { fprintf(stderr, "no ids\n"); return 2; }
    if ((int)ids.size() > 256) ids.resize(256);
    fprintf(stderr, "[ref] family=%s model=%s n=%zu\n", family, model_dir, ids.size());

    if (flm_prefill_init(model_dir, family) != 0) {
        fprintf(stderr, "[ref] flm_prefill_init FAILED\n");
        return 1;
    }
    int boot = -1;
    double ms = 0;
    int r = flm_prefill_run(ids.data(), (int)ids.size(), &boot, &ms);
    fprintf(stderr, "[ref] prefill_run r=%d boot=%d ms=%.1f\n", r, boot, ms);
    return r;
}
