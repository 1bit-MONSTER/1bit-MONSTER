// tools/registry_merge_invariants.cpp — the merge's two invariants, ASSERTED not claimed.
//
// Why this exists: my first merge put the registry's order first UNCONDITIONALLY, which
// demoted the router's head wherever the registry merely named a different lane — on
// Qwen3.6-35B-A3B-Q8_0 it moved the head off `cpu_qwen3_5` (a lane my vocabulary cannot
// express) for no defensible reason. I found that by looking at the merged output against
// the router. This turns the two properties I then claimed into something that fails:
//
//   INV-1  never remove a lane: every router id must survive in the merged list.
//   INV-2  move the head only for a STATED exclusion: if the merged head differs from the
//          router's head, the router's head id must appear in one of the registry's
//          exclusion lists (refused / unavailable_here / blocked / skipped_by_context /
//          conditional). "The registry named a different lane" is NOT a reason.
//   INV-3  the FLIP is the primitive: select_route_with_registry() must equal
//          merge_router_and_registry(select_backend_route(cfg), plan_route(*a)) for the
//          same artifact, and a null registry must return the router route unchanged.
//          Without this the gate would check the primitive while the serving path called
//          something else — the exact gap the flip closes.
//
// Validated against the known-bad version: run against the pre-defer merge and INV-2 must
// fail on the 35B-A3B. A check that has not failed on a bad input is not a check.
//
// Usage: 1bit registry-merge-invariants <dir>
#include "model_discovery.h"
#include "model_registry.h"
#include "model_registry_route.h"
#include "model_router.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace onebit;

static bool violations_ok(size_t a, size_t b) { return a == 0 && b == 0; }

int registry_merge_invariants_main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: registry-merge-invariants <dir>\n"); return 2; }
    std::string dir = argv[1];

    std::vector<ModelConfig> legacy = discover_models(dir);
    ModelRegistry reg = ModelRegistry::scan({dir});

    size_t checked = 0, inv1 = 0, inv2 = 0, inv3 = 0;
    for (const auto& m : legacy) {
        const ModelArtifact* a = reg.resolve_path(m.model_path);
        if (!a) continue;
        BackendRoute router = select_backend_route(m);
        RoutePlan plan = plan_route(*a, 0);
        BackendRoute merged = merge_router_and_registry(router, plan);
        checked++;

        // INV-1: no lane removed.
        for (const auto& id : router.backend_ids_in_order) {
            bool present = false;
            for (const auto& got : merged.backend_ids_in_order) if (got == id) { present = true; break; }
            if (!present) {
                inv1++;
                printf("INV-1 VIOLATED  %s: router lane '%s' dropped\n", a->id.c_str(), id.c_str());
            }
        }

        // INV-2: a head move needs a stated exclusion.
        if (!router.backend_ids_in_order.empty() && !merged.backend_ids_in_order.empty() &&
            router.backend_ids_in_order[0] != merged.backend_ids_in_order[0]) {
            const std::string moved = router.backend_ids_in_order[0];
            auto names = [&](const std::vector<std::pair<Capability, std::string>>& v) {
                for (const auto& kv : v) {
                    BackendType t{}; std::string i, c;
                    if (backend_for(kv.first, t, i, c) && i == moved) return true;
                }
                return false;
            };
            bool stated = names(plan.refused) || names(plan.unavailable_here) ||
                          names(plan.blocked) || names(plan.skipped_by_context) ||
                          names(plan.conditional);
            if (!stated) {
                inv2++;
                printf("INV-2 VIOLATED  %s: head moved %s -> %s with NO stated exclusion\n",
                       a->id.c_str(), moved.c_str(), merged.backend_ids_in_order[0].c_str());
            }
        }

        // INV-3 (the FLIP): the packaged call must be exactly the primitive for the same
        // artifact, and a missing registry must be the router route — so the one-line
        // swap in unified_server cannot change what this tool just verified.
        BackendRoute via_flip = select_route_with_registry(m, m.model_path, &reg, 0);
        BackendRoute no_registry = select_route_with_registry(m, m.model_path, nullptr, 0);
        if (via_flip.backend_ids_in_order != merged.backend_ids_in_order) {
            inv3++;
            printf("INV-3 VIOLATED  %s: select_route_with_registry != merge_router_and_registry\n",
                   a->id.c_str());
        }
        if (no_registry.backend_ids_in_order != router.backend_ids_in_order) {
            inv3++;
            printf("INV-3 VIOLATED  %s: null registry changed the router route\n", a->id.c_str());
        }
    }
    printf("\nchecked=%zu  INV-1 violations=%zu  INV-2 violations=%zu  INV-3 violations=%zu\n",
           checked, inv1, inv2, inv3);
    if (!violations_ok(inv1, inv2) || inv3 != 0) return 1;
    printf("all invariants hold (INV-1 no lane dropped, INV-2 head needs a stated exclusion, "
           "INV-3 the flip == the primitive)\n");
    return 0;
}
