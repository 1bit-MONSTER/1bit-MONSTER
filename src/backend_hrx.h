// backend_hrx.h — HRX (Hip Runtime Extended) GPU backend via hrx llama-server.
//
// Fused GGUF lane on AMD GPU: spawns the self-contained HRX llama-server
// (hrx-b59 bundle) on 127.0.0.1 and talks the OpenAI wire format over HTTP.
// This is the "HRX all the way" path: added first in the GGUF route so HRX is
// tried before ggml_vulkan/zinc/cpu, and the router's init-failover cascades
// to those when HRX fails closed (e.g. GET_ROWS on an unsupported graph).
//
// Pure C++17, POSIX only: no HRX headers, no ROCm/HRX at build time. All HRX
// risk (the llama-server binary, libhrx/libloomc) is self-contained in the
// bundle and found at runtime via $HRX_ROOT / PATH.
//
// Env knobs:
//   HRX_ROOT           path to the unpacked HRX bundle (default: $HRX_ROOT or
//                      the default /home/... path; contains bin/llama-server).
//   HRX_MODEL_BIN      full path override to the llama-server executable.
//   HRX_SPAWN_RETRIES  spawn attempts on failure (default 3)
//   HRX_RETRY_DELAY_S  backoff between attempts (default 3)
//   HRX_INIT_TIMEOUT_S /health wait cap (default 120; HRX JIT+kernel load)
//   HRX_CTX_SIZE       context size (default 4096)
#pragma once
#include "backend.h"
#include <memory>
#include <string>

namespace hrx {
class Inprocess;
}

class HrxBackend : public Backend {
public:
    HrxBackend();
    ~HrxBackend() override;

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override;
    bool reset() override;
    bool forward(int token_id, float* hidden_out) override;
    bool lm_head(const float* hidden, float* logits, int* argmax) override;
    int generate(int token_id) override;
    std::string generate_text(const std::string& prompt, int max_tokens) override;
    void destroy() override;
    float benchmark(int tokens = 10) override;
    bool can_infer() const override { return initialized_ && (pid_ > 0 || inprocess_); }

private:
    bool spawn_server();
    bool wait_healthy(int timeout_s);
    std::string http_get(const std::string& path);
    std::string http_post(const std::string& path, const std::string& body,
                          int* status_out);
    void kill_server();

    pid_t pid_ = -1;
    bool initialized_ = false;
    bool inprocess_mode_ = false;
    std::unique_ptr<hrx::Inprocess> inprocess_;
    std::string server_bin_;
    std::string model_path_;
    std::string port_;
    std::string ctx_size_;
    int spawn_retries_ = 3;
    int retry_delay_s_ = 3;
    int init_timeout_s_ = 120;
    double last_decode_tok_s_ = 0.0;
};

// Factory — extern "C" so BackendManager can both link it directly (onebin)
// and dlsym it from plugin builds.
extern "C" Backend* create_hrx_backend();
