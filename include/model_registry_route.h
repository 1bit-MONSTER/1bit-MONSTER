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
// BRANCH-SCOPE, resolved by @agent-44437c by reading the code rather than picking a
// side: the q35 GPU/HIP-1BP lane is ENV-GATED on `origin/main` (07d4e2c03, which
// includes their merged #2175): `if (ok && getenv("H1BP_Q35_LOAD") && getenv("H1BP_Q35_TRY"))`.
// ca60cf's un-gating commit 56e31251d exists only as an ancestor of three UNMERGED
// branches (fix/2139-q35-ungated-default, fix/2139-q35-forward-lmhead,
// fix/2139-engine-greedy-fastpath). So two lane owners appeared to contradict each other
// and neither was wrong: it is a BRANCH-STATE difference, and the capability must carry
// the gate for main and a note for those branches. Same lesson as the build-scope one —
// an id/behaviour is only as good as the branch that reports it.
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

// ── A CAPABILITY THAT NAMES A TENSOR MUST NAME **WHICH** TENSOR ───────────────
// The general form of a correction @agent-ca60cf made to their own wording, and it
// matters here because the HRX predicate names `token_embd.weight`:
//
//   The HRX constraint is on the EMBEDDING LOOKUP, not on the head. The failing node is
//   `unsupported HRX node 0: GET_ROWS ... inputs=[0:q6_K[1024,151936,1,1], ...]` — i.e.
//   GET_ROWS over `token_embd.weight`, which HRX claims and cannot run unless the
//   embedding is the Q4_K row-gather case (#1945: "Q4_K fuses; q5_0/q8_0/Q4_K_S/IQ2XXS
//   fail-closed"). `lm_head_fused` is an ORTHOGONAL flag about the OUTPUT head (tied vs
//   a separate `output.weight`) — and the one measured-PASSING file carries a separate
//   `output.weight` while still passing, so "Q4_K AND fused" would have excluded the only
//   artifact that works. Read "fused" in that sentence as "the row-gather implementation
//   GET_ROWS supports", never as tied embeddings.
//
// The same two-axis split exists in the 1BP/q35 lane and is already handled separately
// there: the embedding is read by `h1bp_embed_copy_kernel` (Q4NX 1BP) or
// `h1bp_q8embed_kernel` (Q8_0 GGUF), while the head is a DIFFERENT object with its own
// selection (`q35_out_q4nx` / `q35_out_i8` / `q35_out_f16` / `q35_out_f32` / raw
// `q35_out`). An embedding rule and a head rule must never share a predicate.
//
// ── WHICH TYPES ARE DISPATCH-KEYED: a mechanical test, not comment reading ────
// Asked and answered by @agent-ec855d, and it replaces the "I cannot see it from my
// side" I had been saying:
//
//   A TYPE IS A DISPATCH KEY IFF THE IDS UNDER IT DO NOT ALL SHARE ONE `available`
//   EXPRESSION.
//
// Grouping the 19 registrations by that test gives FOUR dispatch-keyed types:
//   NPU_XRT    has_npu() + literal `true`                      -> 2 kinds
//   HIP_GPU    7 ids: 5x has_hip_gpu(), 2x has_vulkan()        -> 2 kinds
//   ZINC_GPU   (has_vulkan()||has_hip_gpu()) + has_vulkan()    -> 2 kinds
//   GENERIC    literal true + has_hip_gpu() + literal true     -> 2 kinds
// and five consistent ones (VULKAN, CPU_AVX512, CPU_SCALAR, HRX_GPU, LSE_GPU — one
// id each). The unpredictable entry is GENERIC, which holds `laguna_gpu` (a GPU
// backend, has_hip_gpu()) beside `cpu_generic` / `nemotron_h_cpu` (literal true).
//
// THE STRONGEST FORM OF THE (type,id) RULE, from @agent-ca60cf reading the factory:
// **Vulkan-capable ids span THREE different BackendTypes** — `ggml_vulkan` and
// `vulkan_hpp_gpu` under HIP_GPU, `zamba2_vulkan` under ZINC_GPU, `vulkan_gpu` under
// VULKAN. So "Vulkan" is not expressible as a BackendType AT ALL; for this capability
// the (type,id) pair list is not a safety rule, it is the only possible representation.
//
// AND THE COST OF GETTING THE TYPE WRONG IS A DEAD LANE, not an untidy table:
// `case BackendType::VULKAN` (backend_manager.cpp:1759-1765) calls
// `create_vulkan_backend()` — the portable lane with score 0 and functional=false —
// while `ggml_vulkan` and `vulkan_hpp_gpu` have ID-SPECIFIC branches inside
// `case BackendType::HIP_GPU`:
//     fused_gpu_npu   -> create_fused_backend        (GPU attention + NPU FFN)
//     vulkan_hpp_gpu  -> create_vulkan_hpp_backend   (Vulkan-Hpp / ZINC shaders)
//     ggml_vulkan     -> create_ggml_vulkan_backend  (llama.cpp Vulkan, "MIT, 357 tok/s")
//     fall-through    -> create_hip_backend
// A type-based dispatcher seeing VULKAN for `ggml_vulkan` would call the score-0
// dead lane instead of the creator the source rates at 357 tok/s — the second instance
// of today's pattern behind npu_flm/npu_xrt (67.5 vs 0.06 tok/s).
//
// `vulkan_gpu` (line 329-338) is therefore carried as REGISTERED_DRY or omitted, NEVER
// exposed as a target: type=VULKAN, available=has_vulkan(), functional=false, score=0,
// and ZERO router mentions in model_router.cpp and dynamic_router.cpp — a
// registered-but-unrouted legacy lane, exactly like `npu_xrt`. This table names it
// nowhere, which is now correct by construction rather than an oversight.
//
// This is derivable and it GENERALISES FORWARD: any id later added under those four
// types inherits the confusion, which a per-id list cannot express. It is also why
// the probe signature below is keyed by engine_id and why `availability_for` takes
// `available_is_a_predicate` as an explicit argument rather than inferring anything
// from the type.
//
// Two rows were briefly uncertain in ec855d's audit and are now read directly:
//   zinc_gpu (line 170):      `available = false;` then `available = has_vulkan() || has_hip_gpu();`
//   zamba2_vulkan (line 257): `available = false;` then `available = has_vulkan();`
// i.e. an init-to-false followed by the real predicate — not a competing definition.

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
    // A FIFTH bucket, from @agent-44437c's measurements: targets that must not be
    // attempted AT ALL because the failure is worse than a refusal. The measured case
    // is Qwen3.6-35B-A3B-Q8_0 against b66, which does not fail closed — it **ABORTS**
    // (`rc=134`, SIGABRT via set_abort_callback). An engine that aborts on an artifact
    // takes the process with it, so "conditional" understates it: a caller cannot
    // retry past an abort the way it can past a refused init.
    std::vector<std::pair<Capability, std::string>> blocked;
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

// ── IS THE ROW'S TYPE A DISPATCH KEY? (census data, not a guess) ──────────────
// @agent-ca60cf's census (`~/issue-triage/type_ambiguity_census.py`, re-runnable per
// build — which matters given the build-scope lesson) found this is not one ambiguous
// pair. Against current main, FOUR types are collapsed:
//     HIP_GPU  7 ids  fused_gpu_npu, ggml_vulkan, hip_1bp_gpu, hip_gpu,
//                     mamba1_gpu, vulkan_hpp_gpu, zamba2_gpu
//     GENERIC  3 ids  cpu_generic, laguna_gpu, nemotron_h_cpu
//     NPU_XRT  2 ids  npu_flm, npu_xrt
//     ZINC_GPU 2 ids  zamba2_vulkan, zinc_gpu
// and five are unambiguous: HRX_GPU (hrx_gpu), LSE_GPU (lse), VULKAN (vulkan_gpu),
// CPU_SCALAR (cpu_scalar), CPU_AVX512 (cpu_avx512).
//
// The code proves the type is insufficient by branching on the id INSIDE the factory:
// backend_manager.cpp:1678 (mamba1_gpu), :1691 (zamba2_gpu), :1705 (nemotron_h_cpu),
// :1714 (hip_1bp_gpu), :1728 (fused_gpu_npu), :1735 (vulkan_hpp_gpu), :1742
// (ggml_vulkan), :1768 (npu_flm, inside `case NPU_XRT`), :1820 (laguna_gpu), :1859
// (zamba2_vulkan); plus dynamic_router.cpp:75/:139 (hip_gpu, zinc_gpu) and :80/:143
// (npu_flm, npu_xrt).
//
// THE SHARPEST ILLUSTRATION, and the reason this is not pedantry: `case
// BackendType::NPU_XRT` at backend_manager.cpp:1766-1799 has `if (info.id ==
// "npu_flm") { … return b; }` and then a FALL-THROUGH that creates the legacy worker
// subprocess backend, which the code's own comment rates at 0.06 tok/s against
// npu_flm's 67.5. Same type, ~1000x apart. A type-only mapping that lands on npu_xrt
// is not merely dry — it is three orders of magnitude slower while looking like the
// same capability.
//
// HARD RULE for this table: **no row may ever be resolved from the type alone.** Only
// (type, id) pairs are routable. Returns a non-null note for a collapsed type, and
// nullptr when the type maps to exactly one id.
const char* type_collapse_note(BackendType t);

// Express a plan in the ENGINE's own currency. `BackendRoute` is what
// select_backend_route() returns and what BackendManager::init's preferred_ids
// overload consumes, so this is the last translation: after it, a caller swap is
// one line in model_router.cpp (a new function, not an edit to the existing
// select_backend_route). Refusals and context skips are folded into `reason`
// rather than dropped, because a route list that silently omits its exclusions
// cannot be audited.
BackendRoute to_backend_route(const RoutePlan& plan);

}  // namespace onebit
