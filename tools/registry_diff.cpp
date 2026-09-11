// tools/registry_diff.cpp — where the LEGACY scan and the REGISTRY disagree.
//
// Step 2 of goal mtvd3pmx says "extend src/model_discovery.cpp into the registry of
// record". That is a behaviour change (the server selects models from the legacy list),
// so it is the operator's call — but the EVIDENCE for it should not be ad hoc. This
// prints the two views side by side over one directory:
//
//   legacy  : discover_models(dir) — flat, non-recursive, ids from GGUF general.name
//   registry: ModelRegistry::scan  — recursive, canonical ids, containers, capabilities
//
// and the delta: which paths only the legacy scan sees, which only the registry sees
// (nested dirs, native containers), and which are the SAME FILE UNDER TWO IDS. That last
// one is the user-visible symptom (a model listed twice with different names) and the
// whole argument for replacing one with the other.
//
// Read-only. It changes no behaviour and is not wired into any caller.
//
// Usage: 1bit registry-diff <dir>
#include "model_discovery.h"
#include "model_registry.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace onebit;

static std::string base_of(const std::string& p) {
    size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

int registry_diff_main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: registry-diff <dir>\n");
        return 2;
    }
    std::string dir = argv[1];

    std::vector<ModelConfig> legacy = discover_models(dir);   // the flat, legacy view
    ModelRegistry reg = ModelRegistry::scan({dir});           // the registry view

    printf("\n== %s ==\n", dir.c_str());
    printf("legacy  : %zu entries (flat, non-recursive, ids from GGUF general.name)\n", legacy.size());
    printf("registry: %zu artifacts (recursive, canonical ids)\n", reg.artifacts().size());

    size_t same_path = 0, id_divergent = 0;
    std::vector<bool> matched(reg.artifacts().size(), false);

    printf("\n-- legacy entries, and what the registry calls the same file --\n");
    for (const auto& m : legacy) {
        const ModelArtifact* a = reg.resolve_path(m.model_path);
        printf("  %-34s %s\n", m.model_name.c_str(), base_of(m.model_path).c_str());
        if (a) {
            same_path++;
            if (a->id != m.model_name) {
                id_divergent++;
                printf("      registry id: %-34s  <- SAME FILE, DIFFERENT ID\n", a->id.c_str());
            } else {
                printf("      registry id: %s\n", a->id.c_str());
            }
            for (size_t i = 0; i < reg.artifacts().size(); ++i)
                if (&reg.artifacts()[i] == a) matched[i] = true;
        } else {
            printf("      registry: NOT FOUND (the registry does not know this path)\n");
        }
    }

    printf("\n-- artifacts the legacy scan CANNOT SEE --\n");
    size_t invisible = 0;
    for (size_t i = 0; i < reg.artifacts().size(); ++i) {
        if (matched[i]) continue;
        const ModelArtifact& a = reg.artifacts()[i];
        invisible++;
        printf("  %-40s %s", a.id.c_str(), to_string(a.container));
        if (!a.files.empty()) printf("   %s", a.files.front().path.c_str());
        printf("\n");
    }
    if (!invisible) printf("  (none)\n");

    printf("\n== same-file=%zu  id-divergent=%zu  legacy-invisible=%zu ==\n",
           same_path, id_divergent, invisible);
    printf("A non-zero id-divergent count is a model the server lists twice under two names;\n");
    printf("legacy-invisible is what a recursive registry sees that a flat scan cannot.\n");
    return 0;
}

#ifdef REGISTRY_DIFF_STANDALONE
// This tool was reachable only through `1bit registry-diff` (onebin), which made the claim it
// carries look like it needed the engine link before it could be checked. It does not: the six TUs
// below build with plain clang++ and no HIP/XRT. Added so §7's headline number is one line for any
// reader, which is the whole point of an evidence tool.
//
//   clang++ -std=c++23 -O2 -Iinclude -Isrc -DREGISTRY_DIFF_STANDALONE \
//     tools/registry_diff.cpp src/model_registry.cpp src/model_discovery.cpp \
//     src/safetensors_reader.cpp src/q4nx_reader.cpp src/gguf_reader.cpp -o registry-diff
int main(int argc, char** argv) { return registry_diff_main(argc, argv); }
#endif
