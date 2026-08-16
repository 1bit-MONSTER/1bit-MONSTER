// onebit_c.cpp — C ABI seam over BackendManager (libonebit.so).
//
// The pattern is proven in npu-infer/ffi_bridge.cpp: opaque handle +
// flat extern "C" functions. Every entry point is exception-safe —
// C++ exceptions must never cross the C boundary (Mojo has no
// exception table for them).

#include "onebit_c.h"
#include "backend_manager.h"
#include "model_discovery.h"
#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// The opaque handle the C header's typedef refers to.
struct OneBitHandle {
    BackendManager mgr;
    ModelConfig cfg;
    std::string error;
    bool inited = false;
    bool scanned = false;

    void set_error(const std::string& msg) { error = msg; }
};

namespace {

const char* kOneBitVersion = "2026.08.04";  // mirrors VERSION

// Exceptions must never cross the C boundary; returns -1 on failure.
template <typename Fn>
int guarded(OneBitHandle* h, const char* what, Fn&& fn) {
    try {
        return fn();
    } catch (const std::exception& e) {
        if (h) h->set_error(std::string(what) + ": " + e.what());
        return -1;
    } catch (...) {
        if (h) h->set_error(std::string(what) + ": unknown exception");
        return -1;
    }
}

}  // namespace

extern "C" {

const char* onebit_version(void) { return kOneBitVersion; }

OneBitHandle* onebit_create(void) {
    try {
        return new OneBitHandle();
    } catch (...) {
        return nullptr;
    }
}

void onebit_destroy(OneBitHandle* h) { delete h; }

const char* onebit_last_error(const OneBitHandle* h) {
    return h ? h->error.c_str() : "null handle";
}

int onebit_init(OneBitHandle* h, const char* weights_dir,
                const char* model_name) {
    if (!h || !weights_dir) {
        if (h) h->set_error("onebit_init: null argument");
        return -1;
    }
    return guarded(h, "onebit_init", [&]() -> int {
        const std::string dir(weights_dir);

        // Phase 1: discover model files → config (same path as the
        // unified server / jarvis_app).
        std::vector<ModelConfig> discovered = discover_models(dir);
        if (discovered.empty()) {
            h->set_error("onebit_init: no models found in " + dir);
            return -1;
        }
        const std::string want = model_name ? model_name : "";
        auto it = discovered.begin();
        if (!want.empty()) {
            auto found = std::find_if(
                discovered.begin(), discovered.end(),
                [&](const ModelConfig& c) { return c.model_name == want; });
            if (found == discovered.end()) {
                h->set_error("onebit_init: model '" + want +
                             "' not found in " + dir);
                return -1;
            }
            it = found;
        }
        h->cfg = *it;

        // Phase 2: hardware scan + init (mirrors jarvis_app.cpp).
        h->mgr.discover();
        h->mgr.set_strategy(SelectionStrategy::FASTEST);
        h->mgr.set_fallback_policy(FallbackPolicy::SEQUENTIAL);
        if (!h->mgr.init(h->cfg, dir, {})) {
            h->set_error("onebit_init: engine init failed for " +
                         h->cfg.model_name);
            return -1;
        }
        h->inited = true;
        h->scanned = true;
        return 0;
    });
}

int onebit_backend_count(const OneBitHandle* h) {
    if (!h || !h->scanned) return 0;
    return static_cast<int>(h->mgr.backends().size());
}

const char* onebit_backend_id(const OneBitHandle* h, int index) {
    if (!h || !h->scanned || index < 0 ||
        index >= static_cast<int>(h->mgr.backends().size()))
        return "";
    return h->mgr.backends()[index].id.c_str();
}

const char* onebit_backend_desc(const OneBitHandle* h, int index) {
    if (!h || !h->scanned || index < 0 ||
        index >= static_cast<int>(h->mgr.backends().size()))
        return "";
    return h->mgr.backends()[index].description.c_str();
}

int onebit_select_backend(OneBitHandle* h, const char* backend_id) {
    if (!h || !backend_id) {
        if (h) h->set_error("onebit_select_backend: null argument");
        return -1;
    }
    return guarded(h, "onebit_select_backend", [&]() -> int {
        if (!h->mgr.select_backend(std::string(backend_id))) {
            h->set_error("onebit_select_backend: no backend '" +
                         std::string(backend_id) + "'");
            return -1;
        }
        return 0;
    });
}

int onebit_generate(OneBitHandle* h, int token_id) {
    if (!h) return -1;
    return guarded(h, "onebit_generate", [&]() -> int {
        if (!h->inited) {
            h->set_error("onebit_generate: not initialized");
            return -1;
        }
        return h->mgr.generate(token_id);
    });
}

int onebit_reset(OneBitHandle* h) {
    if (!h) return -1;
    return guarded(h, "onebit_reset", [&]() -> int {
        if (!h->inited) {
            h->set_error("onebit_reset: not initialized");
            return -1;
        }
        return h->mgr.reset() ? 0 : -1;
    });
}

int onebit_health_check(OneBitHandle* h) {
    if (!h) return -1;
    return guarded(h, "onebit_health_check", [&]() -> int {
        if (!h->inited) return -1;
        return h->mgr.health_check() ? 0 : -1;
    });
}

}  // extern "C"
