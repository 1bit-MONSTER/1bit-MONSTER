// zc_bench_contract.cpp — contract-scale unified bench through the real
// router: prompt from file (>=2048 tokens), policy stock-q4k (prefill on
// the pinned prefill device, decode on the policy leg), timed end-to-end
// single generate() call. Load excluded (warm call first). Mirrors the
// fork unified-bench row (HIP pp -> memfd -> Vulkan tg) but through the
// engine PhaseRouter one-call API.
#include "router/hrx_prefill_engine.h"
#include "router/hrx_decode_engine.h"
#include "hrx_inprocess.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: zc_bench_contract <model.gguf> <prompt.txt> <n_cont> "
                "<prefill_dev> <decode_dev> [n_warm]\n");
        return 2;
    }
    const char* model = argv[1];
    std::ifstream f(argv[2]);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string prompt = ss.str();
    int n_cont = atoi(argv[3]);
    std::string pdev = argv[4];  // HRX0 | Vulkan0
    std::string ddev = argv[5];  // HRX0 | Vulkan0
    int n_warm = argc > 6 ? atoi(argv[6]) : 16;

    auto prod = std::make_shared<hrx::Inprocess>();
    auto pe = std::make_shared<engine::HrxPrefillEngine>(prod, std::string(model), -1, 4096u, pdev);
    auto dec_inprocess = std::make_shared<hrx::Inprocess>();
    auto dec = std::make_shared<engine::HrxDecodeEngine>(dec_inprocess, std::string(model), -1, 4096u, ddev);

    engine::PhaseRouter router;
    router.add_prefill_engine("hrx", pe);
    router.add_decode_engine("hrx", dec);
    engine::PhasePolicy pol;
    pol.model_class = "contract";
    pol.prefill_engine = "hrx";
    pol.decode_engine = "hrx";
    pol.state_handoff = true;
    router.set_policy(pol);

    auto t0 = clk::now();
    if (!pe->init() || !dec->init()) { fprintf(stderr, "[cb] init FAILED\n"); return 1; }
    fprintf(stderr, "[cb] load+init %.2f s; prompt %zu chars; legs prefill=%s decode=%s\n",
            std::chrono::duration<double>(clk::now() - t0).count(),
            prompt.size(), pdev.c_str(), ddev.c_str());

    std::string w = router.generate(prompt, n_warm);
    fprintf(stderr, "[cb] warm ok (%zu chars)\n", w.size());

    auto t1 = clk::now();
    std::string out = router.generate(prompt, n_cont);
    double secs = std::chrono::duration<double>(clk::now() - t1).count();
    fprintf(stderr, "[cb] integrated generate: %d cont tok in %.3f s = %.2f t/s (decode leg)\n",
            n_cont, secs, secs > 0 ? n_cont / secs : 0);
    fprintf(stderr, "[cb] out head: %.120s\n", out.c_str());
    return out.empty() ? 1 : 0;
}
