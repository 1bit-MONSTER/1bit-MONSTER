#pragma once
// onebit — the 1bit engine ('1bit unified') as a Lemonade executor.
//
// WHY THIS EXISTS (goal mtvd3pmx, R8): Lemonade's own executors reach GGUFs
// (llamacpp/llamacpp-hrx) and FLM-tagged models only. A native Q4NX/1BP container
// has no Lemonade recipe, so on the --lemonade face such an artifact could be
// listed (R7) but never executed. This backend makes the engine itself the
// executor: load() spawns `<this binary> unified -m <native path> -p <port>`, and
// the inherited LlamaCppServer forwarding path speaks /v1/chat/completions to it.
//
// The models are INJECTED from the engine's registry of record (descriptor
// dynamic_models = true; see onebit::set_onebit_models). The vendored Lemonade
// tree must not depend on engine headers, so the engine-side caller supplies
// ModelInfo entries rather than this backend including model_registry.h.
#include "lemon/backends/backend_descriptor.h"

#include <nlohmann/json.hpp>

namespace lemon {
namespace backends {
namespace onebit {

inline const BackendDescriptor descriptor = {
    /*recipe*/          "onebit",
    /*display_name*/    "1bit engine (native Q4NX/1BP)",
    // No downloadable binary: the executor is the running binary itself
    // (/proc/self/exe), resolved at load(). spec() therefore returns nullptr.
    /*binary*/          "",
    /*config_section*/  "onebit",
    /*default_device*/  DEVICE_GPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,  // the engine's unified server has no --ctx-size flag
    /*dynamic_models*/  true,   // injected from the engine registry at --lemonade startup
    /*options*/ {
        {"onebit_args", "--onebit-args", "", "ARGS",
         "Extra arguments appended to `1bit unified` for native models",
         "1bit Engine Options"},
    },
    /*support*/ {},             // the engine itself decides what this host can run
    /*supported_modes*/ {"chat"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*experimental*/    true,
    /*web_display_name*/ "",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::AtLeast,
    /*self_manages_downloads*/ true,  // the native file is already on disk; never query a registry
    /*takes_args*/      true,
    /*arg_variants*/    {},
    /*bin_variants*/    {},
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace onebit
}  // namespace backends
}  // namespace lemon
