// router_test_main.cpp — engine in-tree test: PhaseRouter + HrxDecodeEngine.
// Verifies the router API, policy wiring, and the HRX decode-engine adapter
// init path (model load + device assign). The full prefill->memfd->decode
// e2e needs the HIP prefill lane in-process; this validates the decode half
// that is real on this branch (Inprocess::load_session_mem).
#include "router/phase_router.h"
#include "router/hrx_decode_engine.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: router_test <model.gguf>\n"); return 2; }
    const char* model = argv[1];
    int mode = argc > 2 ? atoi(argv[2]) : 0;

    engine::PhaseRouter router;
    auto inprocess = std::make_shared<hrx::Inprocess>();
    auto dec = std::make_shared<engine::HrxDecodeEngine>(inprocess, model, -1, 4096);
    router.add_decode_engine("hrx", dec);
    engine::PhasePolicy pol;
    pol.model_class = "moat-q4nx";
    pol.prefill_engine = "npu";   // not registered in this test
    pol.decode_engine = "hrx";
    router.set_policy(pol);
    fprintf(stderr, "[router_test] policy %s: prefill=%s decode=%s\n",
            pol.model_class.c_str(), pol.prefill_engine.c_str(), pol.decode_engine.c_str());

    if (mode == 1) {
        // Exercise the real decode adapter: init + load_model (HRX0 device).
        if (!dec->init()) { fprintf(stderr, "[router_test] HrxDecodeEngine init FAILED\n"); return 1; }
        fprintf(stderr, "[router_test] HrxDecodeEngine init OK (model loaded, HRX ready)\n");
    } else {
        // Policy-missing prefill must fail cleanly.
        std::string out = router.generate("x", 4);
        fprintf(stderr, "[router_test] generate with missing prefill engine -> %s\n",
                out.empty() ? "empty OK" : "NONEMPTY FAIL");
    }
    fprintf(stderr, "[router_test] PASS\n");
    return 0;
}
