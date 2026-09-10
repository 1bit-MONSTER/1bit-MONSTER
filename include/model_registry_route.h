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

// Availability must be declared before BackendTarget, which carries it.
enum class Availability {
    UNKNOWN,         // nobody asked
    PRESENT,         // the probe confirmed it works
    ABSENT,          // the probe confirmed the hardware is not here
    // "Looks runnable while being dry" (@agent-ca60cf): registered with
    // available=true but functional=false until init. They are right that the
    // intersection must test FUNCTIONAL, not available — npu_flm sits here before
    // init, and a lane reporting available while being dry is the same
    // blank-reads-as-confident problem as everywhere else in this module.
    REGISTERED_DRY,
};

struct BackendTarget {
    Capability capability;
    BackendType type;
    std::string engine_id;      // BackendManager id ("hrx_gpu", "npu_flm", ...)
    std::string constraint;     // why this target is NARROWER than the capability
    // Whether this target was VERIFIED PRESENT on this box, or merely not proven
    // absent. ABSENT never appears here (it becomes unavailable_here). Found by
    // @agent-ec855d: plan_route branched only on == ABSENT, so PRESENT and UNKNOWN
    // took the same path and produced byte-identical output — meaning an unverified
    // plan read exactly like a verified one. The plan SPOKE when it knew something
    // was absent and was SILENT when it did not know, which inverts silence into
    // reassurance. Same disease as dtype_space=empty, the unwritten-header zeros,
    // and arch_suspect:false.
    Availability availability = Availability::UNKNOWN;
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
// THE PROBE IS KEYED BY ENGINE ID, NOT BY BackendType, and that is not a style
// choice. Verified by @agent-ec855d in backend_manager.cpp on ryzen: `npu_xrt`
// (line 66) sets `available = has_npu()` -> FALSE here, while `npu_flm` (line 162)
// sets `available = true` HARDCODED with `functional = false` — and BOTH carry
// `BackendType::NPU_XRT`. Same type, OPPOSITE availability. A type-keyed probe is
// therefore AMBIGUOUS BY CONSTRUCTION on this table: it would resolve NPU_XRT to
// whichever entry it happened to find, and the one that answers `true` is the
// hardcoded one, so NPU_XRT would read PRESENT on a machine with no NPU — the exact
// failure the probe exists to prevent.
//
// AND `available` IS NOT A HARDWARE SIGNAL. It is hardcoded `true` in at least nine
// places in that file (162, 288, 374, 397, 425, 447, 470, 1481, 1519). An
// implementation must build the probe on the real predicates the OTHER entries use
// (`has_npu()`, `has_hip_gpu()`, `has_vulkan()`), and must treat `available &&
// functional` as the MINIMUM, not the target: npu_flm is simultaneously
// hardcoded-available and dry, which is the exact combination an availability-only
// check emits happily.
//
// LAYER NOTE, also from @agent-ec855d: "NPU hardware: No" is printed by the VENDORED
// Lemonade core (third_party/lemonade/src/cpp/server/model_manager.cpp:3499), NOT by
// backend_manager.cpp:58-66. This tree has TWO independent availability probes in
// different layers, both live. This bridge is engine-side, so it must read
// backend_manager's predicates — otherwise a caller inspecting the other layer can
// disagree with the route it was handed.
using AvailabilityProbe = Availability (*)(const std::string& engine_id);
void set_backend_availability_probe(AvailabilityProbe probe);
Availability backend_availability(const std::string& engine_id);

// ── THE PROBE CONTRACT, from @agent-ec855d's audit of all 19 registrations ────
// They found the general form, so an implementer needs NO per-id exemption list:
//
//   Return PRESENT only for ids whose `available` is a `has_*()` PREDICATE.
//   Return UNKNOWN for ids whose `available` is a LITERAL.
//
// Because a literal `available = true` is a DECLARATION, not a measurement, and
// there are SIX of them in backend_manager.cpp: npu_flm, nemotron_h_cpu, cpu_scalar,
// cpu_generic, lse, hrx_gpu. (Literal true is defensible for the CPU entries. For
// npu_flm / lse / hrx_gpu it is not: on a box with no NPU, no lse-server and no HRX
// bundle, all three claim available.)
//
// WHY A TYPE-KEYED PROBE WAS NOT MERELY IMPRECISE: BackendType is a FACTORY DISPATCH
// KEY, not a hardware category, and the source says so — both Vulkan rows carry
// `// factory dispatches from HIP_GPU case`. HIP_GPU therefore spans SEVEN ids
// (hip_1bp_gpu, fused_gpu_npu, vulkan_hpp_gpu, ggml_vulkan, hip_gpu, mamba1_gpu,
// zamba2_gpu), two of which are VULKAN backends gated on has_vulkan(), while Vulkan
// itself appears under three different types (HIP_GPU, VULKAN, ZINC_GPU). A type-keyed
// probe asks a dispatch question and reads the answer as hardware.
//
// ALSO: the only hardcoded `functional = true` in that file is a PLUGIN LOAD
// (load_plugins, "presume functional"). Every other functional=true is earned after
// init. So a plugin-registered id is fully trusted on presumption and can never be
// excluded — a caller mapping anything to a plugin id should know that.
//
// CONSEQUENCE FOR HRX, which is a row this table uses: hrx_gpu is available=true,
// functional=false — the same shape as npu_flm — so it survives an availability-only
// test. With no probe installed, HRX leads the GGUF route on a box that may have no
// HRX bundle at all. The probe is therefore MANDATORY for that row, not optional.
Availability availability_for(bool available_is_a_predicate, bool available, bool functional);

struct RoutePlan {
    const ModelArtifact* artifact = nullptr;
    std::vector<BackendTarget> targets;                              // in order
    // THREE DISTINCT REASONS, kept apart on purpose (@agent-ec855d's second ask):
    // "no backend advertises it", "this box has no such hardware", and "the
    // constraint is violated" need different fixes, so they must not read the same.
    std::vector<std::pair<Capability, std::string>> refused;           // no backend exists
    std::vector<std::pair<Capability, std::string>> unavailable_here;  // hardware absent
    // A FOURTH bucket, on @agent-ca60cf's review: targets that exist but are not
    // unconditional — architecture-mismatched (hip_gpu vs a Qwen GGUF) or registered
    // but dry. Kept apart from `refused` because the capability IS real here; what is
    // conditional is the target.
    std::vector<std::pair<Capability, std::string>> conditional;
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

// ── Per-row EVIDENCE STATUS ───────────────────────────────────────────────────
// @agent-ca60cf corrected their own earlier answer on this table, and the
// correction is the point: the measurement path and the engine id are not the same
// thing. Their box evidence covers TWO rows. The Vulkan tok/s figures they had
// quoted came from llama-bench INSIDE the fork build — that build's own Vulkan
// backend, outside the engine's backend-id system entirely — so they prove a
// Vulkan/RADV lane exists and is fast on this silicon without proving that the
// engine's `ggml_vulkan` id reaches it.
//
// AND ID AVAILABILITY IS BUILD-DEPENDENT, not just box-dependent: the string
// "not found or not functional" appears in exactly two of their logs, both from HRX
// worktree builds where the standard backend .so set is absent, so the manager
// rejected `ggml_vulkan`, `vulkan_hpp_gpu`, `zinc_gpu`, `hip_gpu` and `hip_1bp_gpu`
// — while the q35 build on the same box and model store loaded `hip_1bp_gpu` and
// never probed the Vulkan ids at all. So a row is only as good as the build that
// reports it, and the arbiter must be the engine's runtime available/functional
// state. Nothing here may be upgraded to "verified on hardware" on their account.
const char* backend_evidence(Capability c);

// Express a plan in the ENGINE's own currency. `BackendRoute` is what
// select_backend_route() returns and what BackendManager::init's preferred_ids
// overload consumes, so this is the last translation: after it, a caller swap is
// one line in model_router.cpp (a new function, not an edit to the existing
// select_backend_route). Refusals and context skips are folded into `reason`
// rather than dropped, because a route list that silently omits its exclusions
// cannot be audited.
BackendRoute to_backend_route(const RoutePlan& plan);

}  // namespace onebit
