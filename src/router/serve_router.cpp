#include "router/serve_router.h"
#include "router/hrx_prefill_engine.h"
#include "router/hrx_decode_engine.h"
#include "hrx_inprocess.h"
#include <chrono>
#include <cstdio>
#include <memory>

namespace engine {

PhasePolicy phase_policy_for_class(const std::string& klass) {
    PhasePolicy p;
    p.model_class = klass;
    if (klass == "moat-q4nx") {
        p.prefill_engine = "hrx";
        p.decode_engine  = "hrx";
        p.state_handoff  = false;  // same engine both legs
    } else {  // stock-q4k (default): the dual-leg 30B gate route
        p.prefill_engine = "hrx";
        p.decode_engine  = "hrx";
        p.state_handoff  = true;
    }
    return p;
}

ServedRouter make_served_router(const std::string& model_path,
                                const std::string& klass,
                                int n_gpu_layers, uint32_t ctx_size) {
    ServedRouter sr;
    const std::string decode_pin = (klass == "moat-q4nx") ? "HRX0" : "Vulkan0";
    sr.decode_pin = decode_pin;
    auto router = std::make_shared<PhaseRouter>();
    auto prod = std::make_shared<hrx::Inprocess>();
    auto pe = std::make_shared<HrxPrefillEngine>(prod, model_path,
                                                 n_gpu_layers, ctx_size, "HRX0");
    auto dec_inprocess = std::make_shared<hrx::Inprocess>();
    auto de = std::make_shared<HrxDecodeEngine>(dec_inprocess, model_path,
                                                n_gpu_layers, ctx_size, decode_pin);
    if (!pe->init()) { fprintf(stderr, "[serve] prefill engine init FAILED\n"); return sr; }
    if (!de->init()) { fprintf(stderr, "[serve] decode engine init FAILED\n"); return sr; }
    router->add_prefill_engine("hrx", pe);
    router->add_decode_engine("hrx", de);
    PhasePolicy pol = phase_policy_for_class(klass);
    router->set_policy(pol);
    sr.router = router;
    sr.prefill_engine = pe;
    sr.decode_engine = de;
    sr.ready = true;
    return sr;
}

std::string served_generate(ServedRouter& sr, const std::string& prompt,
                            int max_tokens) {
    if (!sr.router) return "";
    auto t0 = std::chrono::steady_clock::now();
    std::string out = sr.router->generate(prompt, max_tokens);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[phaseroute] served one-call: %zu chars in %.3f s (%s decode leg %s)\n",
            out.size(), secs, sr.decode_pin.c_str(),
            secs > 0 && !out.empty() ? "OK" : "EMPTY");
    return out;
}

}  // namespace engine
