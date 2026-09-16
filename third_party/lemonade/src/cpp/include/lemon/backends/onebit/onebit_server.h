#pragma once
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/llamacpp/llamacpp_server.h"
#include "lemon/model_manager.h"

#include <memory>
#include <string>
#include <vector>

namespace lemon {
namespace backends {

// The engine as an executor. Reuses LlamaCppServer's OpenAI forwarding
// (chat/completion/responses/streaming) and overrides only load(), which spawns
// `<this binary> unified -m <native path> -p <port>` instead of llama-server.
class OnebitServer : public LlamaCppServer {
public:
    using LlamaCppServer::LlamaCppServer;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;
};

namespace onebit {

// The engine-side --lemonade entry injects the native artifacts it discovered
// from its registry of record. Called before Server::run(); the copy is what
// OnebitOps::discover_models() returns, so these become real (routable) models
// with recipe "onebit" rather than an HTTP-only appendix.
void set_onebit_models(std::vector<ModelInfo> models);
std::vector<ModelInfo> onebit_models();

std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<OnebitServer>(); }

}  // namespace onebit

}  // namespace backends
}  // namespace lemon
