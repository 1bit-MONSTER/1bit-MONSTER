// zc_router_duallep.cpp — PhaseRouter with BOTH legs per policy:
//   model class "stock-q4k": prefill=hrx (HRX0)  decode=vulkan (Vulkan0)
//   model class "moat-q4nx": prefill=hrx (HRX0)  decode=hrx (HRX0)
// One process, two Inprocess instances (different device pins), one memfd
// handoff. Demonstrates the single-API router dispatching decode to the
// policy-winning leg for the contract 30B Q4_K_M (Vulkan0: the decode-gate
// leg at 93.65 t/s tg256; HRX0 stays for moat Q4NX/zaya).
#include "router/hrx_prefill_engine.h"
#include "router/hrx_decode_engine.h"
#include "hrx_inprocess.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: zc_router_duallep <model.gguf> <prompt> <n_cont> <class> [n_warm]\n");
        return 2;
    }
    const char* model = argv[1];
    const char* prompt = argv[2];
    int n_cont = atoi(argv[3]);
    std::string klass = argv[4];  // stock-q4k | moat-q4nx
    int n_warm = argc > 5 ? atoi(argv[5]) : 4;

    // Prefill leg: HRX0 (real tokenize + prefill + export memfd).
    auto prod = std::make_shared<hrx::Inprocess>();
    auto pe = std::make_shared<engine::HrxPrefillEngine>(prod, std::string(model), -1, 4096u, std::string("HRX0"));

    // Decode leg: policy-chosen device.
    std::string decdev = (klass == "moat-q4nx") ? "HRX0" : "Vulkan0";
    auto dec_inprocess = std::make_shared<hrx::Inprocess>();
    auto dec = std::make_shared<engine::HrxDecodeEngine>(dec_inprocess, std::string(model), -1, 4096u, decdev);

    engine::PhaseRouter router;
    router.add_prefill_engine("hrx", pe);
    router.add_decode_engine("hrx", dec);
    engine::PhasePolicy pol;
    pol.model_class = klass;
    pol.prefill_engine = "hrx";
    pol.decode_engine = "hrx";
    pol.state_handoff = true;
    router.set_policy(pol);

    auto t0 = clk::now();
    if (!pe->init() || !dec->init()) { fprintf(stderr, "[dual] init FAILED\n"); return 1; }
    fprintf(stderr, "[dual] load+init: %.2f s (policy %s decode -> %s)\n",
            std::chrono::duration<double>(clk::now() - t0).count(),
            klass.c_str(), decdev.c_str());

    std::string w = router.generate(prompt, n_warm);
    fprintf(stderr, "[dual] warm(%s): %s\n", decdev.c_str(), w.empty() ? "(empty)" : w.c_str());

    auto t1 = clk::now();
    std::string out = router.generate(prompt, n_cont);
    double secs = std::chrono::duration<double>(clk::now() - t1).count();
    fprintf(stderr, "[dual] %s integrated %d tok: %.3f s => %.2f t/s\n",
            decdev.c_str(), n_cont, secs, secs > 0 ? n_cont / secs : 0);
    fprintf(stderr, "[dual] out: %s\n", out.empty() ? "(empty)" : out.c_str());
    return out.empty() ? 1 : 0;
}
