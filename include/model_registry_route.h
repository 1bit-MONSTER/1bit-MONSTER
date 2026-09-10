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
#include "model_router.h"   // BackendRoute — the engine's route currency

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

// ── Hardware availability: "what can serve this HERE" ─────────────────────
// Raised by @agent-ec855d from the box, and it is the objection that matters most
// for this step: the registry is HARDWARE-BLIND by design (that is what makes it
// engine-independent), so translating capability -> backend name is not enough.
// On ryzen the registry says zaya1-74b-preview.1bp -> NPU-Q4NX and the resolver
// answers YES, while the engine prints "NPU hardware: No" and /dev/accel* does not
// exist; HIP is likewise unproven for gfx1201. A capability table that is right in
// the abstract and wrong on the box is the exact failure this bridge exists to
// prevent, so the bridge must INTERSECT with the engine's own probe rather than
// assume it.
//
// The probe is by BackendType because that is what the engine knows (has_npu(),
// has_vulkan(), ...). UNKNOWN is the default and does NOT filter: the bridge must
// not invent hardware facts it was not given, and an unfiltered plan is the honest
// answer when nobody told it otherwise.
enum class Availability { UNKNOWN, PRESENT, ABSENT };

// Injected by the engine once at startup (set-once, like the limit overrides).
using AvailabilityProbe = Availability (*)(BackendType);
void set_backend_availability_probe(AvailabilityProbe probe);
Availability backend_availability(BackendType t);

struct RoutePlan {
    const ModelArtifact* artifact = nullptr;
    std::vector<BackendTarget> targets;                              // in order
    // THREE DISTINCT REASONS, kept apart on purpose (@agent-ec855d's second ask):
    // "no backend advertises it", "this box has no such hardware", and "the
    // constraint is violated" need different fixes, so they must not read the same.
    std::vector<std::pair<Capability, std::string>> refused;           // no backend exists
    std::vector<std::pair<Capability, std::string>> unavailable_here;  // hardware absent
    std::vector<std::pair<Capability, std::string>> skipped_by_context; // constraint violated
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

// Express a plan in the ENGINE's own currency. `BackendRoute` is what
// select_backend_route() returns and what BackendManager::init's preferred_ids
// overload consumes, so this is the last translation: after it, a caller swap is
// one line in model_router.cpp (a new function, not an edit to the existing
// select_backend_route). Refusals and context skips are folded into `reason`
// rather than dropped, because a route list that silently omits its exclusions
// cannot be audited.
BackendRoute to_backend_route(const RoutePlan& plan);

}  // namespace onebit
