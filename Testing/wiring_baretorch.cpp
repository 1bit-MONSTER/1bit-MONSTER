// wiring_baretorch.cpp — end-to-end wiring check for #1907: discover_models
// finds the baretorch dir → select_backend_route picks cpu_baretorch →
// backend_manager's create path instantiates the engine → engine init loads
// the real weights. Mirrors how the unified server picks a backend.
// usage: wiring_baretorch <model_dir> <ids.txt>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "common.h"
#include "model_discovery.h"
#include "model_router.h"
#include "backend.h"

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model_dir> <ids.txt>\n", argv[0]); return 2; }
    auto models = discover_models(argv[1]);
    printf("discovered %zu model(s)\n", models.size());
    ModelConfig* found = nullptr;
    for (auto& m : models) {
        printf("  model: %s fmt=%d arch=%d archstr='%s' path=%s\n",
               m.model_name.c_str(), (int)m.format, (int)m.arch,
               m.architecture.c_str(), m.model_path.c_str());
        if (m.arch == RCPP_ARCH_BARETORCH) found = &m;
    }
    if (!found) { printf("FAIL: baretorch model not discovered\n"); return 1; }
    printf("baretorch discovered OK\n");

    BackendRoute r = select_backend_route(*found);
    printf("route:");
    for (auto& id : r.backend_ids_in_order) printf(" %s", id.c_str());
    printf("\n");
    bool has_engine = false;
    for (auto& id : r.backend_ids_in_order) if (id == "cpu_baretorch") has_engine = true;
    if (!has_engine) { printf("FAIL: cpu_baretorch not in route\n"); return 1; }
    printf("route picks cpu_baretorch OK\n");

    // engine init (the same create fn backend_manager dispatches for cpu_baretorch)
    extern Backend* create_baretorch_backend();
    Backend* b = create_baretorch_backend();
    if (!b->init(*found, argv[1])) { printf("FAIL: engine init\n"); return 1; }
    printf("engine init OK\n");
    b->reset();
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    int pred = -1;
    for (size_t i = 0; i < ids.size(); i++) pred = b->generate(ids[i]);
    printf("engine-next-token: %d\n", pred);
    printf("WIRING OK\n");
    return 0;
}
