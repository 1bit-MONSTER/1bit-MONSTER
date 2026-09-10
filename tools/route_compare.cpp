// tools/route_compare.cpp — the decision table for the step-2c-2 caller flip.
//
// The registry resolver exists, has been verified, and nothing consumes it. Before a
// caller is flipped, the honest question is not "is the resolver correct" but "where
// does it DISAGREE with the router that ships today". This prints both for each file,
// side by side, so the flip becomes a table of differences rather than a leap.
//
// It changes no behaviour: it calls select_backend_route() and plan_route() and reports.
//
// Build (engine-side: needs ModelConfig + BackendRoute):
//   cmake --build b --target route_compare
// Usage:
//   route_compare [--at-context N] <model.gguf|.q4nx|.1bp>...
#include "model_discovery.h"
#include "model_registry.h"
#include "model_registry_route.h"
#include "model_router.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace onebit;

static std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) { if (i) s += " > "; s += v[i]; }
    return s.empty() ? "(none)" : s;
}

int route_compare_main(int argc, char** argv) {
    uint32_t at_context = 0;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--at-context" && i + 1 < argc) at_context = (uint32_t)atoi(argv[++i]);
        else files.push_back(a);
    }
    if (files.empty()) { fprintf(stderr, "usage: route_compare [--at-context N] <file>...\n"); return 2; }

    int agree = 0, differ = 0, unresolved = 0;
    for (const auto& f : files) {
        // (a) what ships today: the architecture-keyed static router.
        ModelConfig cfg{};
        bool got_cfg = read_model_file_metadata(f, cfg);
        BackendRoute shipped = got_cfg ? select_backend_route(cfg)
                                       : BackendRoute{{}, "ModelConfig unavailable"};

        // (b) what the registry would choose for the same file.
        std::string dir = f.substr(0, f.find_last_of('/') == std::string::npos ? 0 : f.find_last_of('/'));
        if (dir.empty()) dir = ".";
        ModelRegistry reg = ModelRegistry::scan({dir});
        const ModelArtifact* art = reg.resolve_path(f);
        BackendRoute reg_route;
        if (art) {
            reg_route = to_backend_route(plan_route(*art, at_context));
        } else {
            unresolved++;
        }

        bool same = art && shipped.backend_ids_in_order == reg_route.backend_ids_in_order;
        if (!art) { } else if (same) agree++; else differ++;

        printf("\n%s\n", f.c_str());
        printf("  router  : %s\n", join(shipped.backend_ids_in_order).c_str());
        if (!art) {
            printf("  registry: (no artifact — the registry does not know this file)\n");
        } else {
            // The FLIP as it would ship: registry wins the head, router tail preserved.
            BackendRoute merged = merge_router_and_registry(shipped, plan_route(*art, at_context));
            printf("  merged  : %s   <- what a flip would do\n",
                   join(merged.backend_ids_in_order).c_str());
            printf("  registry: %s   %s\n", join(reg_route.backend_ids_in_order).c_str(),
                   same ? "[SAME]" : "[DIFFERS]");
            if (!same) printf("            reason: %s\n", reg_route.reason.c_str());
        }
    }
    printf("\n== same=%d differs=%d registry-unknown=%d ==\n", agree, differ, unresolved);
    return 0;
}
