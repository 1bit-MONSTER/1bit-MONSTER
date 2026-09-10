// onebit_server.cpp — the 1bit engine as a Lemonade executor (goal mtvd3pmx R8).
#include "lemon/backends/onebit/onebit_server.h"

#include "lemon/backends/backend_ops.h"
#include "lemon/backends/onebit/onebit.h"
#include "lemon/utils/process_manager.h"

#include <lemon/utils/aixlog.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace lemon {
namespace backends {

namespace {

std::vector<ModelInfo>& injected_models() {
    static std::vector<ModelInfo> models;
    return models;
}

// The executor is the binary that is already running: the engine's own --lemonade
// process spawns `unified` from itself. No download, no BackendSpec.
std::string self_executable() {
#ifdef _WIN32
    return "1bit.exe";
#else
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string("1bit");
#endif
}

class OnebitOps : public BackendOps {
public:
    std::vector<ModelInfo> discover_models(const BackendOpsContext&) const override {
        return onebit::onebit_models();
    }

    std::string resolve_checkpoint_path(
        const ModelInfo&,
        const CheckpointResolveContext& ctx) const override {
        // The injected checkpoint is already the absolute path of the native
        // container; the engine opens it directly.
        return ctx.checkpoint;
    }

    bool is_downloaded(const ModelInfo&, const BackendOpsContext&) const override {
        return true;  // injected models exist on this box by construction
    }
};

}  // namespace

void OnebitServer::load(const std::string& model_name,
                        const ModelInfo& model_info,
                        const RecipeOptions& options,
                        bool do_not_upgrade) {
    (void)do_not_upgrade;

    LOG(INFO, "1bit") << "Loading native model: " << model_name << std::endl;
    LOG(DEBUG, "1bit") << "Per-model settings: " << options.to_log_string() << std::endl;

    std::string model_path = model_info.resolved_path();
    if (model_path.empty()) model_path = model_info.checkpoint();
    if (model_path.empty()) {
        throw std::runtime_error("native container path missing for model '" + model_name + "'");
    }

    const std::string engine = self_executable();
    const int backend_port = choose_port();

    std::vector<std::string> argv = {
        "unified",
        "-m", model_path,
        "-p", std::to_string(backend_port),
    };

    const bool inherit_output = (log_level_ == "info") || is_debug();
    set_process_handle(
        utils::ProcessManager::start_process(engine, argv, "", inherit_output, true, {}),
        engine,
        argv);

    // The engine's unified server exposes /v1/health; wait for it before any
    // request is forwarded.
    if (!wait_for_ready("/v1/health")) {
        const ProcessHandle handle = consume_process_handle_for_cleanup();
        if (has_process_handle(handle)) {
            utils::ProcessManager::stop_process(handle);
        }
        throw std::runtime_error("1bit unified failed to start for '" + model_name + "'");
    }

    LOG(DEBUG, "1bit") << "Native model loaded on port " << get_backend_port() << std::endl;
}

namespace onebit {

void set_onebit_models(std::vector<ModelInfo> models) {
    injected_models() = std::move(models);
}

std::vector<ModelInfo> onebit_models() {
    return injected_models();
}

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<OnebitServer>(ctx);
}

// No downloadable artifact: the executor is the running binary.
const BackendSpec* spec() { return nullptr; }

const BackendOps* ops() { return single_ops<OnebitOps>(); }

}  // namespace onebit

}  // namespace backends
}  // namespace lemon
