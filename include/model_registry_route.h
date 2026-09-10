#pragma once
// model_registry_route.h — translate a registry Capability onto the engine's
// backend vocabulary.
//
// WHY THIS IS A SEPARATE FILE: the registry module is deliberately engine-
// independent (it compiles standalone with plain clang++, no HIP/XRT — verified
// repeatedly by @agent-ec855d). Pulling `BackendType` into it would destroy that,
// so the translation lives on the engine side where both vocabularies are already
// visible. The registry stays pure; this bridge translates.
//
// This is ADDITIVE: nothing calls it yet, so no existing route selection changes.
// Flipping a caller is a separate, announced step.
//
// Mapping facts are from @agent-ca60cf's read of the code (2026-09-10), not from
// the file header comment, which is stale:
//   - `backend_manager.cpp:162` registers npu_flm, `type = BackendType::NPU_XRT`,
//     available, functional=false pending init.
//   - `backend_manager.cpp:1768-1774` is the factory for BackendType::NPU_XRT and
//     it PREFERS npu_flm.
//   - `model_router.cpp:172` (and :213) return npu_flm as a target; `npu_xrt` is
//     registered at :58-66 but is NEVER returned as a target by model_router.cpp
//     or dynamic_router.cpp — the dynamic router's accept-list treats either id as
//     "the NPU route", the model router names only flm.
//   So: npu_xrt is the REGISTERED-BUT-UNSELECTED id; npu_flm is the SELECTED one.
//   At the TYPE level the two are the same, which is why the type mapping is safe
//   without resolving that question — but the ID is not interchangeable.
#include "common.h"
#include "model_registry.h"

#include <string>
#include <utility>
#include <vector>

namespace onebit {

struct BackendTarget {
    Capability capability;
    BackendType type;
    std::string engine_id;      // BackendManager id ("hrx_gpu", "npu_flm", ...)
    std::string constraint;     // why this target is NARROWER than the capability
};

// D5b: reachability is not quality. Kept OUT of the capability map on purpose —
// NPU-Q4NX is reachable and the zaya cascade decode on it is not numerically
// usable in-flow (#2114: per-layer corr 0.9997 accumulates across the 20-MoE-layer
// recurrence; only skipping all 180 points reproduced production bit-for-bit).
// A consumer must be able to ask "can serve" and "should serve" separately.
enum class QualityGate {
    NOT_EVALUATED,   // the bridge makes no claim; the caller must apply its own
    KNOWN_DEGRADED,  // reachable, measured unusable in-flow for some architectures
};

struct RoutePlan {
    const ModelArtifact* artifact = nullptr;
    std::vector<BackendTarget> targets;                              // in order
    std::vector<std::pair<Capability, std::string>> refused;         // no backend
    std::vector<std::pair<Capability, std::string>> skipped_by_context;
    QualityGate quality_gate = QualityGate::NOT_EVALUATED;
    std::string quality_note;
};

// Ordered, constraint-aware plan for one artifact. `prefer` overrides the order
// (order is the caller's policy); empty means the artifact's own capability order.
// Capabilities with no backend are REFUSED with a reason rather than dropped, so a
// silent omission is never mistaken for "nothing to do".
RoutePlan plan_route(const ModelArtifact& a, uint32_t context_tokens = 0,
                     const std::vector<Capability>& prefer = {});

// The translation table, exposed for auditing: capability -> (type, id, constraint).
// Returns false for capabilities that have no backend (HRX2-GGUF-Q4NX today).
bool backend_for(Capability c, BackendType& out_type, std::string& out_id,
                 std::string& out_constraint);

}  // namespace onebit
