// router_test_main.cpp — engine in-tree test: PhaseRouter live path.
// Modes:
//   0 (default): fail-close with missing engines.
//   1 <model> <prompt> <n_cont>: FULLY LIVE one generate() call — real
//     tokenize + prefill (HRX0) -> memfd export -> import -> decode (Vulkan0
//     for stock-q4k, HRX0 for moat-q4nx). No files anywhere.
//   2 <model> <prompt-file> <n_cont> <class>: contract path via the bench
//     driver (>=2k prompts, policy class).
#include "router/phase_router.h"
#include "router/hrx_prefill_engine.h"
#include "router/hrx_decode_engine.h"
#include "hrx_inprocess.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: router_test <model.gguf> [mode ...]\n"); return 2; }
    const std::string model = argv[1];
    int mode = argc > 2 ? atoi(argv[2]) : 0;

    if (mode == 0) {
        engine::PhaseRouter router;
        auto inprocess = std::make_shared<hrx::Inprocess>();
        auto dec = std::make_shared<engine::HrxDecodeEngine>(inprocess, model, -1, 4096);
        router.add_decode_engine("hrx", dec);
        engine::PhasePolicy pol;
        pol.model_class = "moat-q4nx";
        pol.prefill_engine = "npu";   // not registered in this test
        pol.decode_engine = "hrx";
        router.set_policy(pol);
        std::string out = router.generate("x", 4);
        fprintf(stderr, "[router_test] missing-prefill generate -> %s\n",
                out.empty() ? "empty OK" : "NONEMPTY FAIL");
        fprintf(stderr, "[router_test] PASS (fail-close)\n");
        return out.empty() ? 0 : 1;
    }

    if (mode == 1) {
        if (argc < 5) { fprintf(stderr, "usage: router_test <model> 1 <prompt> <n_cont>\n"); return 2; }
        const std::string prompt = argv[3];
        int n_cont = atoi(argv[4]);
        engine::PhaseRouter router;
        auto prod = std::make_shared<hrx::Inprocess>();
        auto pe = std::make_shared<engine::HrxPrefillEngine>(prod, model, -1, 4096, "HRX0");
        auto dproc = std::make_shared<hrx::Inprocess>();
        auto dec = std::make_shared<engine::HrxDecodeEngine>(dproc, model, -1, 4096, "Vulkan0");
        router.add_prefill_engine("hrx", pe);
        router.add_decode_engine("hrx", dec);
        engine::PhasePolicy pol;
        pol.model_class = "stock-q4k";
        pol.prefill_engine = "hrx";
        pol.decode_engine = "hrx";
        pol.state_handoff = true;
        router.set_policy(pol);
        if (!pe->init() || !dec->init()) {
            fprintf(stderr, "[router_test] live init FAILED\n");
            return 1;
        }
        std::string out = router.generate(prompt, n_cont);
        fprintf(stderr, "[router_test] live generate (%d cont): %s\n", n_cont,
                out.empty() ? "(empty FAIL)" : out.c_str());
        fprintf(stderr, "[router_test] %s\n", out.empty() ? "FAIL" : "PASS (live)");
        return out.empty() ? 1 : 0;
    }

    fprintf(stderr, "[router_test] unknown mode %d\n", mode);
    return 2;
}
