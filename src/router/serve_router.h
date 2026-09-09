#pragma once
// sage-1: served single-API router — phase policy + engine factory for the
// served inference path (tools/unified_server.cpp hook target).
#include "router/phase_router.h"
#include <memory>
#include <string>

namespace engine {

// Policy rows (corrected-decode basis, mirror of the 841552005 table as
// exercised by the router mains): stock-q4k = HRX0 prefill -> memfd ->
// Vulkan0 decode (the 30B gate route); moat-q4nx = HRX0 both legs (no split).
PhasePolicy phase_policy_for_class(const std::string& klass);

struct ServedRouter {
    std::shared_ptr<PhaseRouter> router;
    std::shared_ptr<void> prefill_engine;  // owned (HrxPrefillEngine)
    std::shared_ptr<void> decode_engine;   // owned (HrxDecodeEngine)
    std::string decode_pin;  // which device the decode leg runs on
    bool ready = false;
};

// Builds the two device-pinned Inprocess engines + PhaseRouter for one model.
ServedRouter make_served_router(const std::string& model_path,
                                const std::string& klass,
                                int n_gpu_layers, uint32_t ctx_size);

// One-call served generate with a consolidated phase trace on stderr.
std::string served_generate(ServedRouter& sr, const std::string& prompt,
                            int max_tokens);

}  // namespace engine
