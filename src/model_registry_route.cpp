// model_registry_route.cpp — the capability -> backend translation (see header).
#include "model_registry_route.h"

namespace onebit {

bool backend_for(Capability c, BackendType& out_type, std::string& out_id,
                 std::string& out_constraint) {
    out_constraint.clear();
    switch (c) {
        case Capability::NPU_Q4NX:
        case Capability::NPU_1BP:
            // Type level is unambiguous: npu_flm registers as BackendType::NPU_XRT
            // (backend_manager.cpp:162) and the NPU_XRT factory prefers it
            // (:1768-1774). The ID is npu_flm, not npu_xrt — npu_xrt is registered
            // but never selected by model_router.cpp or dynamic_router.cpp.
            out_type = BackendType::NPU_XRT;
            out_id = "npu_flm";
            // THE TRAP, from @agent-ca60cf: npu_flm is Q4NX-only. For GGUF/H1B its
            // token-level forward()/generate() are text-level stubs that return
            // false, and FLM init "succeeds" on ANY model tag and then loads FLM's
            // own q4nx model — the requested file is never served. That is
            // reachability-without-correctness: it looks available and answers with
            // a different model. The registry already prevents the hand-off
            // structurally (NPU_Q4NX is assigned only to ONEBP/RAW_BIN containers,
            // never to GGUF — see derive_capabilities), and this string states it so
            // a future change to that derivation cannot silently undo it.
            out_constraint =
                "native (ONEBP/RAW_BIN) containers only: npu_flm is Q4NX-only and would "
                "accept the init then serve its own q4nx model rather than the requested file";
            return true;
        case Capability::HIP_1BP:
            out_type = BackendType::HIP_GPU;
            out_id = "hip_1bp_gpu";
            return true;
        case Capability::HIP_GGUF:
            out_type = BackendType::HIP_GPU;
            out_id = "hip_gpu";
            return true;
        case Capability::RADV_GGUF:
            // The router's GGUF/Vulkan lane is ggml_vulkan, not zinc_gpu: ZINC is a
            // separate Vulkan-IR runtime that the router only reaches for specific
            // architectures. Mapping RADV-GGUF onto zinc_gpu would be wrong.
            out_type = BackendType::VULKAN;
            out_id = "ggml_vulkan";
            return true;
        case Capability::HRX_GGUF:
            out_type = BackendType::HRX_GPU;
            out_id = "hrx_gpu";
            // D5a: the context limit belongs to the configured BUNDLE, not the
            // artifact, and is enforced only on the engine server + the in-process
            // shim — NOT under --lemonade.
            out_constraint = "context limit is bundle-scoped (see capability_limit); "
                             "enforced by the engine server and the shim, NOT under --lemonade";
            return true;
        case Capability::HRX2_GGUF_Q4NX:
            // D5c: REFUSED, deliberately. dtype 42 means three different things to
            // three readers (F5), and no registered backend advertises the type-42
            // reader. Mapping this onto HRX_GPU would hand a *-q4nx.gguf to the
            // mainline path, where the mis-read produces GARBAGE rather than an
            // error. Refusing is strictly better than guessing.
            return false;
        case Capability::MLX_GPU:
            out_type = BackendType::LSE_GPU;
            out_id = "lse";
            return true;
        case Capability::CPU:
            out_type = BackendType::CPU_SCALAR;
            out_id = "cpu_generic";
            return true;
        case Capability::UNKNOWN:
        default:
            return false;
    }
}

RoutePlan plan_route(const ModelArtifact& a, uint32_t context_tokens,
                     const std::vector<Capability>& prefer) {
    RoutePlan plan;
    plan.artifact = &a;

    const std::vector<Capability>& order = prefer.empty() ? a.capabilities : prefer;
    for (Capability c : order) {
        if (!a.has(c)) continue;   // not this artifact's business; not a refusal
        if (context_tokens && !a.supports(c, context_tokens)) {
            uint32_t lim = a.max_context_for(c);
            std::string why = "context " + std::to_string(context_tokens) +
                              " exceeds limit " + std::to_string(lim);
            const CapabilityLimit* l = capability_limit(c);
            if (l && l->bundle) why += std::string(" (bundle ") + l->bundle + ")";
            if (l && l->not_enforced_in)
                why += std::string(" — and that limit is not enforced in ") + l->not_enforced_in;
            plan.skipped_by_context.emplace_back(c, why);
            continue;
        }
        BackendTarget t;
        t.capability = c;
        if (!backend_for(c, t.type, t.engine_id, t.constraint)) {
            plan.refused.emplace_back(c, "no registered backend advertises this capability");
            continue;
        }
        plan.targets.push_back(std::move(t));
    }

    // D5b: stated, never inferred from reachability.
    if (a.has(Capability::NPU_Q4NX)) {
        plan.quality_gate = QualityGate::KNOWN_DEGRADED;
        plan.quality_note = "NPU-Q4NX is reachable; the zaya cascade decode on it is not "
                           "numerically usable in-flow (#2114) — a separate decision from "
                           "whether it can serve";
    }
    return plan;
}

}  // namespace onebit
