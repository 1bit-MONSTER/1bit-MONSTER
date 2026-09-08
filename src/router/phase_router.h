// phase_router.h — PhaseRouter: single-call prefill/decode routing across
// two engines with the zero-copy state handoff between phases.
//
// Goal mtsy05dx task single-api-router (audit follow-up). The engine's
// DynamicRouter is per-token GPU<->NPU; this composes a PREFILL engine and a
// DECODE engine behind ONE call, with the llama_state shared-memory handoff
// (Inprocess::load_session_mem, 0a54070c) between phases. The policy table
// (model class -> prefill/decode engine) is data-driven: stock Q4_K class
// prefill=HIP|Vulkan decode=Vulkan/HRX; moat Q4NX/zaya decode=HRX only.
//
// Backends implement:
//   prefill(prompt, out_state_fd)  - tokenize+decode prompt, export llama_state
//                                    to a memfd, return the fd + resume token
//   decode(state_fd, n_tokens)     - import state, continue greedy/argmax
// The caller sees ONE generate(prompt, max_tokens) call.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hrx {
class Inprocess;  // HRX in-process engine (owns load_session_mem)
}

namespace engine {

// A prefill engine: given a prompt, produce a shared-memory llama_state blob
// (memfd fd) positioned at n_tokens, ready for decode to continue from.
struct PrefillResult {
    int  state_fd = -1;    // memfd/dma-buf fd with the session-v9 state
    long n_tokens = 0;     // tokens consumed by prefill (resume position)
    bool ok       = false;
};

class PrefillEngine {
public:
    virtual ~PrefillEngine() = default;
    virtual PrefillResult prefill(const std::string& prompt) = 0;
    virtual const char* id() const = 0;
};

// A decode engine: import a shared-memory state and continue.
class DecodeEngine {
public:
    virtual ~DecodeEngine() = default;
    // Returns generated text (greedy) or empty on failure.
    virtual std::string decode(int state_fd, long n_tokens, int max_tokens) = 0;
    virtual const char* id() const = 0;
};

// Data-driven policy: which (prefill, decode) engine pair serves a model
// class. Populated from the unified-bench policy table
// (research/single-api-router/2026-09-08-policy-and-design.md 841552005).
struct PhasePolicy {
    std::string model_class;      // "stock-q4k" | "moat-q4nx" | ...
    std::string prefill_engine;   // "hip" | "vulkan" | "npu"
    std::string decode_engine;    // "vulkan" | "hrx" | "hip"
    bool state_handoff = true;    // false = same engine both phases
};

class PhaseRouter {
public:
    PhaseRouter() = default;

    void add_prefill_engine(const std::string& id, std::shared_ptr<PrefillEngine> e);
    void add_decode_engine(const std::string& id, std::shared_ptr<DecodeEngine> e);
    void set_policy(const PhasePolicy& policy);

    /// ONE call: prefill on the policy prefill engine, zero-copy handoff,
    /// decode on the policy decode engine. Caller sees no split.
    std::string generate(const std::string& prompt, int max_tokens);

    const PhasePolicy& policy() const { return policy_; }
    bool has_engines() const { return !prefill_.empty() && !decode_.empty(); }

private:
    std::vector<std::pair<std::string, std::shared_ptr<PrefillEngine>>> prefill_;
    std::vector<std::pair<std::string, std::shared_ptr<DecodeEngine>>>  decode_;
    PhasePolicy policy_;
};

}  // namespace engine
