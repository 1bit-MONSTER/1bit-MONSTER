// backend_lse.h — LSE (Lemon Seed Engine) backend via lse-server subprocess.
//
// Text-level backend: spawns `lse-server` on 127.0.0.1 and talks the OpenAI
// wire format over HTTP. It serves the MLX group-affine lane (Qwen3.5/3.6/3.8
// family + lemonseed) on AMD GPU — the only backend in this tree that can
// read MLX checkpoints directly.
//
// Pure C++17, POSIX only: no LSE headers, no C++26, no ROCm/HRX at build
// time. All LSE risk (g++-16, P2996 reflection, comgr JIT) stays inside the
// lse-server binary, which is found at runtime via $LSE_SERVER_BIN or PATH.
//
// Env knobs:
//   LSE_SERVER_BIN     path to the lse-server executable (default: lse-server)
//   LSE_SPAWN_RETRIES  spawn attempts on failure (default 10)
//   LSE_RETRY_DELAY_S  backoff between attempts (default 5)
//   LSE_INIT_TIMEOUT_S /health wait cap (default 120; raise for cold JIT)
//   LSE_KEEP_STDIO     keep child stdio attached (debug; default detach)
//   LSE_PORT           fixed port override (default: ephemeral)
#pragma once
#include "backend.h"
#include <string>

class LseBackend : public Backend {
public:
    LseBackend();
    ~LseBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override;
    bool reset() override;
    bool forward(int token_id, float* hidden_out) override;
    bool lm_head(const float* hidden, float* logits, int* argmax) override;
    int generate(int token_id) override;
    std::string generate_text(const std::string& prompt, int max_tokens) override;
    void destroy() override;
    float benchmark(int tokens = 10) override;
    bool can_infer() const override { return initialized_ && pid_ > 0; }

private:
    bool spawn_server();
    bool wait_healthy(int timeout_s);
    std::string http_get(const std::string& path);
    // POST a JSON body; returns the response body and sets *status_out to the
    // HTTP status (0 on transport failure). Caller owns the body string.
    std::string http_post(const std::string& path, const std::string& body,
                          int* status_out);
    void kill_server();

    pid_t pid_ = -1;
    bool initialized_ = false;
    std::string server_bin_;
    std::string model_dir_;
    std::string port_;
    std::string api_key_;
    int spawn_retries_ = 10;
    int retry_delay_s_ = 5;
    int init_timeout_s_ = 120;
    // Per-request statistics from the last /v1/completions call (M3
    // benchmark): decode tok/s reported by lse-server's timings field.
    double last_decode_tok_s_ = 0.0;
};

// Factory — extern "C" so BackendManager can both link it directly (onebin)
// and dlsym it from plugin builds. Declared here (namespace scope — linkage
// specs are illegal at block scope).
extern "C" Backend* create_lse_backend();
