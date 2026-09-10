// tools/registry_route_map.cpp — dump and exercise the capability -> backend map.
//
// Two jobs, both aimed at making the bridge REVIEWABLE rather than readable:
//   1. print the translation table as data, so a lane owner can check it against
//      the box without opening the source;
//   2. run plan_route() over real artifacts, which executes the mapping path for
//      real instead of leaving it at "compiles clean".
//
// This is engine-side on purpose (it needs BackendType from common.h), which is
// why it is a separate tool and not a flag on the stdlib-only registry_scan.
//
// Build (no engine link needed — the mapping path uses enums and strings only):
//   amdclang++ -std=c++23 -O2 -Iinclude tools/registry_route_map.cpp
//       src/model_registry.cpp src/model_registry_route.cpp -o route_map
//
// Usage:
//   registry_route_map --table
//   registry_route_map [--at-context N] [--prefer CAP,CAP] <root>...
#include "model_registry.h"
#include "model_registry_route.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace onebit;

static const Capability kAll[] = {
    Capability::FUSED_GPU_NPU, Capability::VULKAN_1BP,
    Capability::NPU_Q4NX, Capability::NPU_1BP, Capability::HIP_1BP, Capability::HIP_GGUF,
    Capability::RADV_GGUF, Capability::HRX2_GGUF_Q4NX, Capability::HRX_GGUF,
    Capability::MLX_GPU, Capability::CPU, Capability::UNKNOWN};

int main(int argc, char** argv) {
    bool table_only = false;
    uint32_t at_context = 0;
    std::string prefer_arg, absent_arg, present_arg, dry_arg;
    std::vector<std::string> roots;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--table") table_only = true;
        else if (a == "--at-context" && i + 1 < argc) at_context = (uint32_t)atoi(argv[++i]);
        else if (a == "--prefer" && i + 1 < argc) prefer_arg = argv[++i];
        else if (a == "--absent-cap" && i + 1 < argc) { absent_arg = (absent_arg.empty() ? "" : absent_arg + ","); absent_arg += argv[++i]; }
        else if (a == "--present-cap" && i + 1 < argc) { present_arg = (present_arg.empty() ? "" : present_arg + ","); present_arg += argv[++i]; }
        else if (a == "--dry-cap" && i + 1 < argc) { dry_arg = (dry_arg.empty() ? "" : dry_arg + ","); dry_arg += argv[++i]; }
        else roots.push_back(a);
    }

    printf("%-18s %-14s %-18s %s\n", "capability", "BackendType", "engine_id", "constraint / refusal");
    printf("%s\n", std::string(18 + 1 + 14 + 1 + 18 + 1 + 40, '-').c_str());
    std::vector<Capability> refused;
    for (Capability c : kAll) {
        if (c == Capability::UNKNOWN) continue;
        BackendType t{};
        std::string id, cons;
        if (backend_for(c, t, id, cons)) {
            printf("%-18s %-14s %-18s %s\n", to_string(c), backend_name(t), id.c_str(),
                   cons.empty() ? "-" : cons.c_str());
        } else {
            refused.push_back(c);
            printf("%-18s %-14s %-18s %s\n", to_string(c), "-", "-",
                   "REFUSED — no registered backend advertises this capability");
        }
    }
    // Demonstrate the probe contract on the ids @agent-ec855d audited, so an
    // implementer sees the rule rather than reading it.
    printf("\nPROBE CONTRACT (availability_for) - a literal `available` is a DECLARATION, not a measurement:\n");
    struct { const char* id; bool predicate; bool available; bool functional; } cases[] = {
        {"npu_xrt",     true,  false, false},
        {"npu_flm",     false, true,  false},
        {"hrx_gpu",     false, true,  false},
        {"lse",         false, true,  true},
        {"cpu_generic", false, true,  true},
        {"ggml_vulkan", true,  true,  true},
        // The state the demo was MISSING, found by @agent-ec855d running it: a
        // predicate-backed id that is available but NOT yet functional. That is the
        // NORMAL STARTUP STATE for every GPU predicate row, because functional only
        // flips true after init (backend_manager.cpp:654, 1096, 1167, 1313, 1333), so
        // the contract's most common early answer was the one outcome its table never
        // showed. With four outcomes and five rows the demo had three.
        {"hip_1bp_gpu", true,  true,  false},
    };
    for (const auto& c : cases) {
        Availability a = availability_for(c.predicate, c.available, c.functional);
        const char* n = a == Availability::PRESENT ? "PRESENT"
                      : a == Availability::ABSENT ? "ABSENT"
                      : a == Availability::REGISTERED_DRY ? "REGISTERED_DRY" : "UNKNOWN";
        printf("  %-14s predicate=%-3s available=%-3s functional=%-3s -> %s\n",
               c.id, c.predicate ? "yes" : "NO", c.available ? "yes" : "no",
               c.functional ? "yes" : "no", n);
    }
    // Per-row: is the TYPE enough? (it never is for the collapsed four)
    printf("\nTYPE-UNIQUENESS per row (a type-only mapping is NEVER sufficient):\n");
    for (Capability c : kAll) {
        if (c == Capability::UNKNOWN) continue;
        BackendType t{};
        std::string id, cons;
        if (!backend_for(c, t, id, cons)) {
            printf("  %-18s %-16s (refused; no backend)\n", to_string(c), "-");
            continue;
        }
        const char* note = type_collapse_note(t);
        printf("  %-18s %-16s %s\n", to_string(c), backend_name(t),
               note ? note : "type maps to exactly one id - (type,id) still required by rule");
    }
    printf("\nEVIDENCE STATUS per row — read this before trusting any row:\n");
    for (Capability c : kAll) {
        if (c == Capability::UNKNOWN) continue;
        printf("  %-18s %s\n", to_string(c), backend_evidence(c));
    }
    printf("\nrefused by design: %zu of %zu\n", refused.size(), sizeof(kAll) / sizeof(kAll[0]) - 1);

    if (table_only) return 0;
    if (roots.empty()) {
        fprintf(stderr, "registry_route_map: need --table or at least one root\n");
        return 2;
    }

    // Stand in for the engine's probe (has_npu(), has_vulkan(), ...) so the
    // hardware-absent path can be exercised without the engine.
    static std::vector<std::string> absent_ids, present_ids, dry_ids;
    if (!absent_arg.empty()) {
        size_t p = 0;
        while (p <= absent_arg.size()) {
            size_t c = absent_arg.find(',', p);
            std::string tok = absent_arg.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!tok.empty()) {
                auto cap = capability_from_string(tok);
                if (!cap) { fprintf(stderr, "unknown capability '%s'\n", tok.c_str()); return 2; }
                BackendType t{};
                std::string id, cons;
                if (backend_for(*cap, t, id, cons)) absent_ids.push_back(id);
                else { fprintf(stderr, "capability '%s' has no backend to be absent\n", tok.c_str()); return 2; }
            }
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }
    if (!dry_arg.empty()) {
        size_t p = 0;
        while (p <= dry_arg.size()) {
            size_t c = dry_arg.find(',', p);
            std::string tok = dry_arg.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!tok.empty()) {
                auto cap = capability_from_string(tok);
                if (!cap) { fprintf(stderr, "unknown capability '%s'\n", tok.c_str()); return 2; }
                BackendType t{}; std::string id, cons;
                if (backend_for(*cap, t, id, cons)) dry_ids.push_back(id);
            }
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }
    if (!absent_arg.empty() || !present_arg.empty() || !dry_arg.empty()) {
        static std::vector<std::string> present_ids_unused;
        if (!present_arg.empty()) {
            size_t p = 0;
            while (p <= present_arg.size()) {
                size_t c = present_arg.find(',', p);
                std::string tok = present_arg.substr(p, c == std::string::npos ? std::string::npos : c - p);
                if (!tok.empty()) {
                    auto cap = capability_from_string(tok);
                    if (!cap) { fprintf(stderr, "unknown capability '%s'\n", tok.c_str()); return 2; }
                    BackendType t{}; std::string id, cons;
                    if (backend_for(*cap, t, id, cons)) present_ids.push_back(id);
                }
                if (c == std::string::npos) break;
                p = c + 1;
            }
        }
        // A probe IS installed now, so capabilities it was not told about are
        // UNKNOWN rather than silently PRESENT — that is the distinction this flag
        // exists to demonstrate.
        // Id-keyed, because npu_xrt and npu_flm share a BackendType and disagree.
        set_backend_availability_probe([](const std::string& id) {
            for (const auto& d : dry_ids) if (d == id) return Availability::REGISTERED_DRY;
            for (const auto& a : absent_ids) if (a == id) return Availability::ABSENT;
            for (const auto& p : present_ids) if (p == id) return Availability::PRESENT;
            return Availability::UNKNOWN;
        });
    }

    ScanOptions opt;
    ModelRegistry reg = ModelRegistry::scan(roots, opt);

    std::vector<Capability> prefer;
    if (!prefer_arg.empty()) {
        size_t p = 0;
        while (p <= prefer_arg.size()) {
            size_t c = prefer_arg.find(',', p);
            std::string tok = prefer_arg.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!tok.empty()) {
                auto cap = capability_from_string(tok);
                if (!cap) { fprintf(stderr, "unknown capability '%s'\n", tok.c_str()); return 2; }
                prefer.push_back(*cap);
            }
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }

    printf("\n=== plans over %zu artifact(s)%s\n", reg.artifacts().size(),
           at_context ? (" at " + std::to_string(at_context) + " ctx").c_str() : "");
    for (const auto& a : reg.artifacts()) {
        RoutePlan plan = plan_route(a, at_context, prefer);
        printf("\n%s\n", a.id.c_str());
        if (plan.targets.empty()) printf("  no target\n");
        for (const auto& t : plan.targets)
            printf("  -> %-12s %-14s %s%s\n", to_string(t.capability), t.engine_id.c_str(),
                   t.availability == Availability::PRESENT ? "" : "[UNVERIFIED] ",
                   t.constraint.empty() ? "" : t.constraint.c_str());
        for (const auto& r : plan.refused)
            printf("  !! %-12s refused: %s\n", to_string(r.first), r.second.c_str());
        for (const auto& r : plan.conditional)
            printf("  ~~ %-12s conditional: %s\n", to_string(r.first), r.second.c_str());
        for (const auto& r : plan.unavailable_here)
            printf("  ?? %-12s %s\n", to_string(r.first), r.second.c_str());
        for (const auto& r : plan.skipped_by_context)
            printf("  -- %-12s skipped: %s\n", to_string(r.first), r.second.c_str());
        if (plan.quality_gate != QualityGate::NOT_EVALUATED)
            printf("  ~~ quality_gate=%s\n", plan.quality_note.c_str());
        // The engine's own currency, i.e. what a caller would hand to
        // BackendManager::init's preferred_ids overload.
        BackendRoute br = to_backend_route(plan);
        printf("  engine BackendRoute: [");
        for (size_t i = 0; i < br.backend_ids_in_order.size(); i++)
            printf("%s%s", i ? ", " : "", br.backend_ids_in_order[i].c_str());
        printf("]\n    reason: %s\n", br.reason.c_str());
    }
    return 0;
}
