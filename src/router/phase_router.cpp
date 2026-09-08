// phase_router.cpp — PhaseRouter implementation.
#include "phase_router.h"

#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

namespace engine {

void PhaseRouter::add_prefill_engine(const std::string& id, std::shared_ptr<PrefillEngine> e) {
    if (e) prefill_.emplace_back(id, std::move(e));
}

void PhaseRouter::add_decode_engine(const std::string& id, std::shared_ptr<DecodeEngine> e) {
    if (e) decode_.emplace_back(id, std::move(e));
}

void PhaseRouter::set_policy(const PhasePolicy& policy) { policy_ = policy; }

std::string PhaseRouter::generate(const std::string& prompt, int max_tokens) {
    // Locate the policy engines.
    std::shared_ptr<PrefillEngine> pe;
    for (auto& [id, e] : prefill_) {
        if (id == policy_.prefill_engine) { pe = e; break; }
    }
    std::shared_ptr<DecodeEngine> de;
    for (auto& [id, e] : decode_) {
        if (id == policy_.decode_engine) { de = e; break; }
    }
    if (!pe || !de) {
        fprintf(stderr, "[phaserouter] policy engines missing (prefill=%s decode=%s)\n",
                policy_.prefill_engine.c_str(), policy_.decode_engine.c_str());
        return "";
    }

    // Phase 1: prefill -> shared-memory state.
    PrefillResult r = pe->prefill(prompt);
    if (!r.ok || r.state_fd < 0) {
        fprintf(stderr, "[phaserouter] prefill failed on %s\n", pe->id());
        return "";
    }
    fprintf(stderr, "[phaserouter] prefill %s -> state fd %d (%ld tokens)\n",
            pe->id(), r.state_fd, r.n_tokens);

    // Phase 2: zero-copy handoff (the fd IS the shared memory; no file, no copy).
    // Phase 3: decode on the policy decode engine.
    std::string out = de->decode(r.state_fd, r.n_tokens, max_tokens);
    close(r.state_fd);  // decode engine dup()s or mmaps before returning
    if (out.empty()) {
        fprintf(stderr, "[phaserouter] decode failed on %s\n", de->id());
        return "";
    }
    fprintf(stderr, "[phaserouter] decode %s: %zu chars\n", de->id(), out.size());
    return out;
}

}  // namespace engine
