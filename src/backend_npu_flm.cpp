// backend_npu_flm.cpp — NPU backend via production FLM engine (MIT, official)
//
// Spawns the FLM binary as a subprocess and communicates via its text protocol.
// FLM delivers 67.5 tok/s on Strix Halo (Qwen3-0.6B) — 1000x faster than the
// old reverse-engineered npu_engine_universal worker (0.06 tok/s).
//
// Environment variables:
//   NPU_FLM_BIN     — path to FLM binary (default: /opt/rocm/bin/flm)
//   NPU_FLM_CONFIG  — path to model_list.json (default: /opt/rocm/etc/flm/model_list.json)
//   NPU_FLM_XCLBINS — path to xclbin directory (default: /opt/rocm/share/flm/xclbins)
//   NPU_MODEL_TAG   — override auto-detected FLM model tag
//
// Part of the unified zaya_server binary.

#include "backend.h"
#include "npu_device_path.h"
#include "npu_flm_delta.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <httplib.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/syscall.h>   // SYS_gettid — PDEATHSIG is per creator-thread

// ── FLM Model Tags ──
// Maps generic model dimensions to FLM's :tag naming convention.
// FLM uses colon-separated tags like "qwen3:0.6b" to identify models
// in its model_list.json catalog. The tag encodes architecture + size.
static const char* flm_tag_for_model(const ModelConfig& cfg) {
    // Check for explicit tag first
    static const char* env_tag = getenv("NPU_MODEL_TAG");
    if (env_tag && env_tag[0]) return env_tag;

    // Architecture + size mapping — covers the 20+ FLM-supported architectures
    int H = cfg.hidden_size;
    const std::string& arch = cfg.architecture;

    // Llama family (official FLM registry: llama3.1:8b / llama3.2:1b / llama3.2:3b).
    // Weights come from the ROCm FLM_Q4NX_Converter (Q4NX pivot — no more custom
    // per-model loaders for llama-family models). q4nx manifests carry no arch,
    // so disambiguate by H+L (llama3.2:1b=2048/16, 3b=3072/28, 3.1:8b=4096/32).
    if (arch == "llama" ||
        (H == 2048 && cfg.num_layers == 16) ||
        (H == 3072 && cfg.num_layers == 28) ||
        (H == 4096 && cfg.num_layers == 32)) {
        if (H <= 2048) return "llama3.2:1b";
        if (H <= 3072) return "llama3.2:3b";
        return "llama3.1:8b";
    }

    // Qwen3.5 Gate-Delta family
    if (arch == "qwen35" || arch == "qwen35moe") {
        // Qwen3.6-35B-A3B MoE: same qwen3_5_moe GGUF arch, but 256 experts
        // and hidden 2048 — must NOT fall into the dense H-based mapping below
        // (which would pick qwen3.5:4b). Discriminate on expert count.
        if (cfg.num_experts >= 100) {
            return "qwen3.6-moe:35b-a3b";
        }
        if (H <= 1024) return "qwen3.5:0.8b";
        if (H <= 1536) return "qwen3.5:2b";
        if (H <= 2560) return "qwen3.5:4b";
        if (H <= 4096) return "qwen3.5:9b";
        return "qwen3.5:9b";
    }

    // Qwen3.6-MoE (GGUF arch qwen3_6_moe, and q4nx filename "Qwen3.6-...")
    if (arch == "qwen3.6" || arch == "qwen36moe") {
        return "qwen3.6-moe:35b-a3b";
    }

    // Gemma4
    if (arch == "gemma4") {
        return "gemma4-it:e4b";
    }

    // Qwen3 family (default catch-all for most models)
    if (H <= 1024) return "qwen3:0.6b";
    if (H <= 2048) return "qwen3:1.7b";
    if (H <= 2560) return "qwen3:4b";
    if (H <= 4096) return "qwen3:8b";
    if (H <= 5120) return "qwen3:14b";
    return "qwen3:32b";
}

// ── NPU FLM Backend ──
class NpuFlmBackend : public Backend {
    std::string flm_bin_;
    std::string flm_config_;
    std::string flm_xclbins_;
    std::string model_tag_;
    std::string last_prompt_;  // previous prompt text, for multi-turn KV reuse
    pid_t pid_ = 0;
    int stdin_fd_ = -1;   // write to child's stdin
    int stdout_fd_ = -1;  // read from child's stdout
    int spawn_retries_ = 10;      // spawn retries on NPU busy (env NPU_FLM_SPAWN_RETRIES)
    int spawn_retry_delay_s_ = 5; // backoff between retries (env NPU_FLM_RETRY_DELAY_S)
    int stderr_fd_ = -1;  // read from child's stderr
    int http_port_ = -1;  // FLM serve port (serve mode)

public:
    NpuFlmBackend() {
        type = BackendType::NPU_XRT;
        name = "NPU FLM (MIT, 67.5 tok/s)";

        // Configurable paths via env vars, with compile-time defaults from submodule
        const char* bin  = getenv("NPU_FLM_BIN");
        const char* cfg  = getenv("NPU_FLM_CONFIG");
        const char* xclb = getenv("NPU_FLM_XCLBINS");

        // Validate env-controlled paths (fixes #1329, #1349, #1351): all three
        // env vars (NPU_FLM_BIN / NPU_FLM_CONFIG / NPU_FLM_XCLBINS) feed
        // exec/access/setenv for the FLM subprocess, so all three get the same
        // whitelist. Reject any ".." segment too — a prefix like
        // /opt/../../../tmp/evil passes the prefix check but resolves outside
        // the whitelist (same rejection zaya_engine.cpp's resolve_weights_dir
        // uses for #1328).
        auto safe_bin = [](const char* p) -> bool {
            if (!p) return true;
            std::string s(p);
            if (s.find("..") != std::string::npos) return false;
            return s.find("/opt/") == 0 || s.find("/usr/") == 0 || s.find("/home/") == 0;
        };
        if (!safe_bin(bin)) {
            fprintf(stderr, "NPU: NPU_FLM_BIN=%s rejected — must be under /opt/, /usr/, or /home/ without '..'\n", bin);
            bin = nullptr;
        }
        if (!safe_bin(cfg)) {
            fprintf(stderr, "NPU: NPU_FLM_CONFIG=%s rejected — must be under /opt/, /usr/, or /home/ without '..'\n", cfg);
            cfg = nullptr;
        }
        if (!safe_bin(xclb)) {
            fprintf(stderr, "NPU: NPU_FLM_XCLBINS=%s rejected — must be under /opt/, /usr/, or /home/ without '..'\n", xclb);
            xclb = nullptr;
        }

        flm_bin_     = bin  ? bin  : FLM_BINARY_PATH;
        flm_config_  = cfg  ? cfg  : FLM_CONFIG_PATH;
        flm_xclbins_ = xclb ? xclb : FLM_XCLBIN_PATH;

        // Fallback: check if binary exists at the configured path, try /opt/rocm next
        if (access(flm_bin_.c_str(), X_OK) != 0) {
            std::string rocm_bin = "/opt/rocm/bin/flm";
            if (access(rocm_bin.c_str(), X_OK) == 0) {
                flm_bin_ = rocm_bin;
                flm_config_ = "/opt/rocm/etc/flm/model_list.json";
                flm_xclbins_ = "/opt/rocm/share/flm/xclbins";
            }
        }

        // The NPU's column/slot budget is shared with the other zaya/FLM
        // services (each model takes 8 of the 40 columns). When another
        // service is mid-startup, CREATE_CONTEXT fails with MGMT_ERT_NOAVAIL
        // and flm-real exits after its own short retry budget. Retrying the
        // spawn here makes service starts order-independent: a column always
        // frees up (total footprints fit), so a later attempt succeeds.
        if (getenv("NPU_FLM_SPAWN_RETRIES"))
            spawn_retries_ = atoi(getenv("NPU_FLM_SPAWN_RETRIES"));
        if (getenv("NPU_FLM_RETRY_DELAY_S"))
            spawn_retry_delay_s_ = atoi(getenv("NPU_FLM_RETRY_DELAY_S"));
    }

    ~NpuFlmBackend() override { destroy(); }
    bool can_infer() const override { return initialized; }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        (void)weights_dir;

        // FLM only speaks its own Q4NX format and is text-level: forward()/
        // generate() are stubs, and init "succeeds" for any model tag but
        // then loads FLM's own q4nx model, never the requested file. Reject
        // everything else up front (same guard style as the mamba1 arch
        // check in cc77f391c) so the router/strategy engine never selects
        // it for a model it cannot run.
        if (cfg.format != ModelFormat::Q4NX) {
            fprintf(stderr, "NPU: FLM is Q4NX-only — rejecting %s (format %d)\n",
                    cfg.model_path.c_str(), (int)cfg.format);
            return false;
        }

        // Detect NPU hardware (node layout varies: /dev/accel/accelN or
        // flat /dev/accelN — issue #1517)
        if (!npu_device_present()) {
            fprintf(stderr, "NPU: no %s — XDNA 2 not available\n", npu_device_path());
            return false;
        }

        // Check FLM binary
        if (access(flm_bin_.c_str(), X_OK) != 0) {
            fprintf(stderr, "NPU: FLM binary not found at %s\n", flm_bin_.c_str());
            return false;
        }
        if (access(flm_config_.c_str(), R_OK) != 0) {
            fprintf(stderr, "NPU: FLM config not found at %s\n", flm_config_.c_str());
            return false;
        }

        // Map model to FLM tag
        model_tag_ = flm_tag_for_model(cfg);
        fprintf(stderr, "NPU: launching FLM %s...\n", model_tag_.c_str());

        // #1604: the FLM registry (model_list.json) silently overrides the
        // requested --model path — the tag resolves to whatever the registry
        // points at. Warn loudly when the registry's model for the resolved
        // tag differs from the requested file, so a redirected service is
        // discoverable without digging through the FLM child's log.
        {
            std::string tag = model_tag_, variant;
            size_t colon = tag.find(':');
            if (colon != std::string::npos) { variant = tag.substr(colon + 1); tag = tag.substr(0, colon); }
            std::ifstream f(flm_config_);
            if (f) {
                try {
                    nlohmann::json j; f >> j;
                    std::string reg_name;
                    auto models = j.value("models", nlohmann::json::object());
                    if (models.contains(tag)) {
                        auto& v = models[tag];
                        if (v.is_object()) {
                            if (!variant.empty() && v.contains(variant) && v[variant].is_object())
                                reg_name = v[variant].value("name", "");
                            else if (v.contains("name")) reg_name = v.value("name", "");
                        }
                    }
                    if (!reg_name.empty()) {
                        std::string req = cfg.model_path, req_dir = req;
                        size_t sl = req.find_last_of('/');
                        if (sl != std::string::npos) { req = req.substr(sl + 1); req_dir = req_dir.substr(0, sl); }
                        size_t dl = req_dir.find_last_of('/');
                        std::string req_dir_name = dl == std::string::npos ? req_dir : req_dir.substr(dl + 1);
                        if (req_dir_name != reg_name && req != reg_name) {
                            fprintf(stderr,
                                "WARNING #1604: requested --model %s but FLM registry %s maps tag '%s' to "
                                "registry model '%s' — serving the REGISTRY model, not the requested file. "
                                "Set NPU_MODEL_TAG or fix model_list.json to serve the requested model.\n",
                                cfg.model_path.c_str(), flm_config_.c_str(), model_tag_.c_str(), reg_name.c_str());
                        }
                    }
                } catch (const std::exception& e) {
                    fprintf(stderr, "NPU: failed to parse %s for model validation: %s\n", flm_config_.c_str(), e.what());
                }
            }
        }

        // Retry the spawn until the NPU has a free column/slot (see
        // constructor comment). Each attempt is bounded by NPU_FLM_TIMEOUT.
        for (int attempt = 1; attempt <= spawn_retries_; attempt++) {
        // Spawn FLM subprocess
        int to_child[2], from_child[2], err_child[2];
        if (pipe(to_child) < 0 || pipe(from_child) < 0 || pipe(err_child) < 0) {
            perror("NPU: pipe"); return false;
        }

        // Ignore SIGPIPE so writes to a dead child return EPIPE instead of crashing
        static bool sigpipe_ignored = []{ signal(SIGPIPE, SIG_IGN); return true; }();
        (void)sigpipe_ignored;

        // PDEATHSIG is delivered when the CREATING THREAD exits, so it may only
        // be armed if the fork happens on the main thread. Decide that HERE, in
        // the parent — see the child's comment for why the child cannot.
        const bool fork_on_main_thread =
            (::getpid() == static_cast<pid_t>(::syscall(SYS_gettid)));

        // Serve mode: FLM's OpenAI-compatible server (not the interactive REPL).
        // The serve-based executor renders/detokenizes identically to the Lemonade
        // flm backend, which is what closes the last byte-level parity gap.
        const int serve_port = pick_free_port();
        const std::string serve_port_s = std::to_string(serve_port);

        pid_ = fork();
        if (pid_ < 0) {
            perror("NPU: fork");
            close(to_child[0]); close(to_child[1]);
            close(from_child[0]); close(from_child[1]);
            close(err_child[0]); close(err_child[1]);
            return false;
        }

        if (pid_ == 0) {
            // Child: FLM process
            close(to_child[1]); close(from_child[0]); close(err_child[0]);
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            dup2(err_child[1], STDERR_FILENO);
            close(to_child[0]); close(from_child[1]); close(err_child[1]);

            // Die with the parent so a crashed service cannot leak an FLM
            // child holding an NPU context (see ensure_serve() in the zaya
            // backend — same orphan/NOAVAIL problem).
            //
            // BUT PDEATHSIG is delivered when the CREATING THREAD exits, not only
            // the process, so it may only be armed when the fork happened on the
            // MAIN thread. The test is computed in the PARENT (see
            // fork_on_main_thread): a freshly forked child is single-threaded, so
            // getpid()==gettid() is always true inside it and cannot answer the
            // question. Arming it from a short-lived worker killed FLM the instant
            // that thread ended — "FLM ready" then "write to FLM failed: child
            // killed by signal 15" at request time, i.e. 0 tokens.
            if (fork_on_main_thread) {
                prctl(PR_SET_PDEATHSIG, SIGTERM);
            }

            setenv("FLM_CONFIG_PATH", flm_config_.c_str(), 1);
            setenv("FLM_XCLBIN_PATH", flm_xclbins_.c_str(), 1);

            execl(flm_bin_.c_str(), "flm", "serve", model_tag_.c_str(),
                  "-p", serve_port_s.c_str(), nullptr);
            fprintf(stderr, "NPU: failed to exec FLM: %s\n", flm_bin_.c_str());
            _exit(1);
        }

        close(to_child[0]); close(from_child[1]); close(err_child[1]);
        stdin_fd_  = to_child[1];
        stdout_fd_ = from_child[0];
        stderr_fd_ = err_child[0];

        // Readiness = the OpenAI /v1/models endpoint answering 200. FLM serve has
        // no /health (404), and the REPL's ">>> " prompt does not exist here.
        http_port_ = serve_port;
        if (!wait_for_http_ready(serve_port,
                                 getenv("NPU_FLM_TIMEOUT") ? atoi(getenv("NPU_FLM_TIMEOUT")) : 120)) {
            fprintf(stderr, "NPU: FLM serve attempt %d/%d not ready (NPU busy?) — retrying in %ds\n",
                    attempt, spawn_retries_, spawn_retry_delay_s_);
            destroy();
            sleep(spawn_retry_delay_s_);
            continue;
        }

        initialized = true;
        last_prompt_.clear();
        fprintf(stderr, "NPU: FLM ready — %s serving on 127.0.0.1:%d (%s)\n",
                model_tag_.c_str(), serve_port, flm_bin_.c_str());
        return true;
        }  // retry loop
        fprintf(stderr, "NPU: FLM failed after %d spawn attempts\n", spawn_retries_);
        return false;
    }

    bool reset() override { return true; }

    bool forward(int token_id, float* hidden_out) override {
        (void)token_id; (void)hidden_out;
        fprintf(stderr, "NPU: forward() not supported — use generate() for text-level FLM\n");
        return false;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        fprintf(stderr, "NPU: lm_head() not supported — FLM handles this internally\n");
        return false;
    }

    int generate(int token_id) override {
        (void)token_id;
        // FLM is text-level: use /dev/stdin to pass a prompt directly
        fprintf(stderr, "NPU: use query() for text-level FLM inference\n");
        return -1;
    }

    /// Text-level generation: FLM tokenizes internally, so the whole prompt
    /// goes over the REPL pipe and the generated text comes back.
    std::string generate_text(const std::string& prompt, int max_tokens,
                              float temperature = -1.0f) override {
        if (pid_ <= 0 || http_port_ <= 0 || !initialized) return "";
        if (max_tokens <= 0) max_tokens = 16;
        // The prompt arrives already chat-templated by the caller. Send it as a RAW
        // completion (`/v1/completions`) so FLM does not apply the template twice;
        // the resulting token stream is the same one the Lemonade flm backend drives
        // through /v1/chat/completions, so the rendered text matches byte-for-byte.
        nlohmann::json req;
        req["prompt"] = prompt;
        req["max_tokens"] = max_tokens;
        if (temperature >= 0.0f) req["temperature"] = temperature;
        req["stream"] = false;
        int status = 0;
        const std::string resp = http_post_json("/v1/completions", req.dump(), &status);
        if (status != 200 || resp.empty()) {
            fprintf(stderr, "NPU: /v1/completions HTTP %d — %s\n", status, resp.substr(0, 200).c_str());
            return "";
        }
        try {
            auto j = nlohmann::json::parse(resp);
            if (j.contains("error") && !j["error"].is_null()) return "";
            if (!j.contains("choices") || j["choices"].empty()) return "";
            return j["choices"][0].value("text", "");
        } catch (const nlohmann::json::exception& e) {
            fprintf(stderr, "NPU: completions JSON parse failed: %s\n", e.what());
            return "";
        }
    }

    /// Continue the live FLM session with a delta (no <<RESET>>): multi-turn
    /// KV reuse. The caller owns session bookkeeping and supplies the delta
    /// (a suffix of the growing conversation prompt, newline-terminated).
    // No cross-request KV reuse over the serve API: the caller resends the full
    // prompt, which is correct and only costs a re-prefill.
    std::string continue_text(const std::string& delta) override {
        (void)delta;
        return "";
    }

    /// Send a text prompt to FLM and get the response.
    /// The prompt is the raw text; FLM handles tokenization internally.
    /// Returns the generated text, or error string on failure.
    std::string query(const std::string& prompt) {
        if (pid_ <= 0 || stdin_fd_ < 0 || stdout_fd_ < 0) return "[npu: not loaded]";

        // 1. Multi-turn KV reuse. FLM keeps its KV cache resident across
        // inserts, so a prompt that extends the previous one (client resends
        // full history every turn) continues the session with just the delta
        // — no <<RESET>>, no full re-prefill of the conversation per turn.
        // New conversation or diverged history → reset and re-prefill.
        std::string send;
        if (npu_flm_send_delta(last_prompt_, prompt, send))
            write(stdin_fd_, "<<RESET>>\n", strlen("<<RESET>>\n"));

        // 2. Send prompt (delta on continuation, full prompt otherwise)
        std::string req = send + "\n";
        ssize_t written = write(stdin_fd_, req.c_str(), req.size());
        if (written < 0 || (size_t)written != req.size()) {
            // The FLM child is gone (e.g. it lost the NPU to a sibling backend).
            // Its stderr was piped and never read, so the reason is still in that
            // pipe; report it plus the exit status so the failure carries a cause
            // rather than only the sentinel. This is what made "0 tokens" opaque.
            std::string child_err = drain_child_stderr();
            int status = 0;
            pid_t reaped = (pid_ > 0) ? waitpid(pid_, &status, WNOHANG) : -1;
            std::string why;
            if (reaped == pid_) {
                if (WIFEXITED(status))       why = "child exited code " + std::to_string(WEXITSTATUS(status));
                else if (WIFSIGNALED(status)) why = "child killed by signal " + std::to_string(WTERMSIG(status));
                else                          why = "child reaped";
            } else {
                why = "child not reaped (still running or already reaped)";
            }
            fprintf(stderr, "NPU: write to FLM failed (%zd/%zu): %s%s%s\n",
                    written, req.size(), why.c_str(),
                    child_err.empty() ? "" : " -- child stderr: ",
                    child_err.empty() ? "" : child_err.c_str());
            if (reaped == pid_) { pid_ = 0; initialized = false; }
            return "[npu: write error]";
        }

        last_prompt_ = prompt;

        return read_response();
    }

    // Issue one FLM REPL command and discard its echo/confirmation up to the next
    // ">>> " prompt, so the following read_response() sees only the model output.
    void send_repl_command(const std::string& cmd_in) {
        if (stdin_fd_ < 0) return;
        const std::string cmd = cmd_in + "\n";
        if (write(stdin_fd_, cmd.c_str(), cmd.size()) == static_cast<ssize_t>(cmd.size()))
            (void)read_response();
    }

    // Bind :0, read the assigned port, close. A small race window (the port could
    // be taken before FLM binds it) is acceptable: readiness then fails and the
    // spawn loop retries on a fresh port.
    static int pick_free_port() {
        int s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return 18400 + (static_cast<int>(::getpid()) % 1000);
        struct sockaddr_in a {};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        if (::bind(s, reinterpret_cast<struct sockaddr*>(&a), sizeof(a)) != 0) {
            ::close(s);
            return 18400 + (static_cast<int>(::getpid()) % 1000);
        }
        socklen_t len = sizeof(a);
        if (::getsockname(s, reinterpret_cast<struct sockaddr*>(&a), &len) != 0) {
            ::close(s);
            return 18400 + (static_cast<int>(::getpid()) % 1000);
        }
        const int port = ntohs(a.sin_port);
        ::close(s);
        return port;
    }

    std::string http_post_json(const std::string& path, const std::string& body, int* status_out) {
        httplib::Client cli("127.0.0.1", http_port_);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(300, 0);
        auto res = cli.Post(path.c_str(), body, "application/json");
        if (status_out) *status_out = res ? res->status : 0;
        return res ? res->body : std::string();
    }

    bool wait_for_http_ready(int port, int timeout_s) {
        httplib::Client cli("127.0.0.1", port);
        cli.set_connection_timeout(1, 0);
        cli.set_read_timeout(2, 0);
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count() < timeout_s) {
            auto res = cli.Get("/v1/models");
            if (res && res->status == 200) return true;
            if (pid_ > 0 && waitpid(pid_, nullptr, WNOHANG) == pid_) { pid_ = -1; return false; }
            usleep(500000);
        }
        return false;
    }

    // Non-blocking drain of the child's piped stderr. Returns "" when nothing is
    // pending (or no pipe). Used to make a dead-child failure self-explaining.
    std::string drain_child_stderr() {
        std::string out;
        if (stderr_fd_ < 0) return out;
        char buf[4096];
        while (out.size() < 16384) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stderr_fd_, &fds);
            struct timeval tv = {0, 0};
            int r = select(stderr_fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (r <= 0) break;
            ssize_t n = read(stderr_fd_, buf, sizeof(buf));
            if (n <= 0) break;
            out.append(buf, (size_t)n);
        }
        return out;
    }

    std::string read_response() {
        // Read response until ">>> " prompt
        std::string resp;
        char c;
        auto t0 = std::chrono::steady_clock::now();
        int timeout_ms = 120000;
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() < timeout_ms) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
            struct timeval tv = {0, 500000}; // 500ms
            int r = select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (r > 0) {
                if (read(stdout_fd_, &c, 1) > 0) {
                    resp += c;
                    if (resp.size() >= 4 && resp.substr(resp.size()-4) == ">>> ") {
                        resp = resp.substr(0, resp.size() - 4);
                        break;
                    }
                } else break;
            } else if (r < 0) break;
        }

        // Strip leading ">>> " if present (echo from FLM)
        if (resp.size() >= 4 && resp.substr(0, 4) == ">>> ")
            resp = resp.substr(4);

        // Clean REPL artifacts from the response:
        //   - ANSI color codes ([31m...[0m)
        //   - "<<RESET>>" echo, "... " prompt echoes, "[FLM]" log lines
        //   - the "Model RAW Output:" re-print: when present, the streamed
        //     output before it is a prefix of the re-print, so cut there.
        std::string clean;
        bool saw_raw_marker = false;
        bool in_ansi = false;
        {
            std::string s;
            for (char c : resp) {
                if (in_ansi) {
                    // ANSI CSI: \x1b [ ... <final byte a-z/A-Z>
                    if (c >= 'a' && c <= 'z') in_ansi = false;
                    else if (c >= 'A' && c <= 'Z') in_ansi = false;
                    continue;
                }
                if (c == '\x1b') { in_ansi = true; continue; }
                if (c == '\n') {
                    if (!s.empty()) {
                        if (s.rfind("[FLM]", 0) == 0) {
                            // "Model RAW Output: " marker: the streamed output
                            // before it is a prefix of the re-print — drop it
                            // and keep only the clean copy that follows.
                            if (s.find("Model RAW Output") != std::string::npos) {
                                saw_raw_marker = true;
                                clean.clear();
                            }
                            s.clear();
                        } else if (s.rfind("... ", 0) == 0 || s.rfind("<<RESET>>", 0) == 0) {
                            s.clear();  // prompt echo
                        } else {
                            clean += s; clean += '\n';
                        }
                    } else if (!clean.empty()) {
                        clean += '\n';  // keep inner blank lines
                    }
                    s.clear();
                    continue;
                }
                s += c;
            }
        }
        // Trim trailing whitespace
        while (!clean.empty() && (clean.back() == '\n' || clean.back() == ' ' || clean.back() == '\r'))
            clean.pop_back();

        return clean;
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::string result = query("Hello, what is 2+2?");
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        int gen_tokens = std::max(1, (int)result.size() / 4);
        fprintf(stderr, "NPU FLM benchmark: %d chars (%d est tok) in %.0fms = %.1f tok/s\n",
                (int)result.size(), gen_tokens, ms, gen_tokens / (ms / 1000.0f));
        return ms / std::max(tokens, gen_tokens);
    }

    void destroy() override {
        initialized = false;
        last_prompt_.clear();
        if (pid_ > 0) {
            // Send graceful exit
            const char* exit_cmd = "/exit\n";
            if (stdin_fd_ >= 0) {
                write(stdin_fd_, exit_cmd, strlen(exit_cmd));
                usleep(300000);
                close(stdin_fd_);
            }
            stdin_fd_ = stdout_fd_ = stderr_fd_ = -1;

            // Terminate progressively
            kill(pid_, SIGTERM);
            int status;
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count() < 2000) {
                pid_t r = waitpid(pid_, &status, WNOHANG);
                if (r == pid_) { pid_ = 0; return; }
                usleep(50000);
            }
            kill(pid_, SIGKILL);
            waitpid(pid_, &status, 0);
            pid_ = 0;
        }
        stdin_fd_ = stdout_fd_ = stderr_fd_ = -1;
    }
};

// ── Factory ──
extern "C" Backend* create_npu_flm_backend() {
    auto* b = new NpuFlmBackend();
    // Check availability immediately
    b->can_infer();  // triggers hardware detection side effects
    return b;
}
