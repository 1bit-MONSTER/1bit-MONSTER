// backend_npu_flm.cpp — NPU backend via production FLM engine
//
// Uses /opt/fastflowlm/bin/flm (v0.9.45, validated 94 tok/s on Strix Halo).
// FLM handles: model loading, Q4NX dequant, NPU xclbin dispatch, lm_head.
// Communication: stdin/stdout line protocol with ">>> " prompt delimiter.
//
// This replaces the custom npu_engine_universal which has intermittent
// xclbin hangs (issue #56). FLM xclbins are at:
//   /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/
//   (layer.xclbin + mm.xclbin + dequant.xclbin + attn.xclbin)
//
// Part of the unified zaya_server binary.

#include "backend.h"

// Raw prompt text for the NPU FLM backend (set by the server via a plain
// extern function — a virtual method would change the vtable layout and
// break the hipcc-compiled adapter TUs, whose vtables emit garbage slots).
static std::string g_npu_prompt_text;
extern "C" void npu_flm_set_prompt_text(const char* s) {
    g_npu_prompt_text = s ? s : "";
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <fcntl.h>
#include <signal.h>

// Wait up to timeout_ms for child to exit. Returns true if exited.
static bool wait_for_child(pid_t pid, int timeout_ms) {
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < timeout_ms) {
        int status;
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return true;
        if (r < 0) return true;
        usleep(10000); // 10ms poll interval
    }
    return false;
}

class NpuFlmTestBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    bool available_ = false;
    pid_t pid_ = 0;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    int stderr_fd_ = -1;
    std::string flm_bin_ = "/opt/fastflowlm/bin/flm";
    std::string model_tag_ = "qwen3:0.6b";
    int timeout_ms_ = 300000;
    int port_ = 8097;            // flm serve port (NPU_FLM_PORT overrides)
    int max_gen_tokens_ = 512;   // per-request generation cap (probe needs ~4)

    // Cached generation state
    std::vector<int> pending_prompt_;
    std::string generated_text_;
    size_t generated_pos_ = 0;
    bool saw_prior_call_ = false;   // for prompt-complete detection (pos reset)
    bool queried_ = false;          // one query per request

public:
    BackendType type() const override { return BackendType::NPU_XRT; }
    const char* name() const override { return "NPU FLM"; }
    float estimated_tok_s() const override { return 57.0f; }  // estimate; FLM Qwen3:0.6B
    bool is_coherent() const override { return true; }

    bool is_available() override {
        if (available_) return true;
        // Check for NPU via multiple methods
        bool hw = false;
        // Method 1: amdxdna kernel module (Strix Halo)
        std::ifstream m("/proc/modules");
        if (m.good()) {
            std::string line;
            while (std::getline(m, line))
                if (line.find("amdxdna") != std::string::npos) { hw = true; break; }
        }
        // Method 2: XRT device node
        if (!hw) hw = (access("/dev/xclmgmt", F_OK) == 0);
        // Method 3: sysfs drivers
        if (!hw) hw = (access("/sys/bus/pci/drivers/amd_npu", F_OK) == 0 ||
                       access("/sys/bus/pci/drivers/xdna", F_OK) == 0);
        // Method 4: XRT can find device
        if (!hw) {
            FILE* p = popen("xrt-smi examine 2>/dev/null | grep -q RyzenAI && echo yes", "r");
            if (p) { char buf[4]={0}; fread(buf,1,3,p); pclose(p); if(buf[0]=='y') hw=true; }
        }
        if (!hw) { fprintf(stderr, "  NPU: no XDNA 2 detected\n"); return false; }
        // Honor NPU_FLM_BIN env (e.g. v0.9.46 extracted install), else defaults
        const char* env_bin = getenv("NPU_FLM_BIN");
        if (env_bin && env_bin[0] && access(env_bin, X_OK) == 0) {
            flm_bin_ = env_bin;
        } else if (access(flm_bin_.c_str(), X_OK) != 0) {
            flm_bin_ = "/usr/bin/flm";
            if (access(flm_bin_.c_str(), X_OK) != 0) {
                fprintf(stderr, "  NPU: FLM not installed\n");
                return false;
            }
        }
        available_ = true;
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();
        if (!is_available()) return false;

        // Map model dimensions to FLM tag
        // Qwen3.6-35B-A3B MoE: 256 experts (GGUF expert_count). Must not fall
        // into the dense H-based mapping below (would pick qwen3:4b).
        // FLM's catalog has no other model with >= 100 experts.
        if (cfg.num_experts >= 100) {
            model_tag_ = "qwen3.6-moe:35b-a3b";
        } else if (cfg.architecture == "qwen3.6") {
            // Q4NX filename-derived arch ("Qwen3.6-35B-A3B-NPU2")
            model_tag_ = "qwen3.6-moe:35b-a3b";
        } else if (cfg.architecture == "qwen35moe" || cfg.architecture == "qwen36moe") {
            if (cfg.hidden_size <= 1024) model_tag_ = "qwen3.5:0.8b";
            else if (cfg.hidden_size <= 1536) model_tag_ = "qwen3.5:2b";
            else if (cfg.hidden_size <= 2560) model_tag_ = "qwen3.5:4b";
            else model_tag_ = "qwen3.5:9b";
        } else if (cfg.architecture == "llama" ||
                   (cfg.hidden_size == 2048 && cfg.num_layers == 16) ||   // llama3.2:1b
                   (cfg.hidden_size == 3072 && cfg.num_layers == 28) ||   // llama3.2:3b
                   (cfg.hidden_size == 4096 && cfg.num_layers == 32)) {   // llama3.1:8b
            // Llama family (Q4NX pivot — weights from the ROCm FLM_Q4NX_Converter)
            if (cfg.hidden_size <= 2048)      model_tag_ = "llama3.2:1b";
            else if (cfg.hidden_size <= 3072) model_tag_ = "llama3.2:3b";
            else                              model_tag_ = "llama3.1:8b";
        } else if (cfg.hidden_size <= 1024)      model_tag_ = "qwen3:0.6b";
        else if (cfg.hidden_size <= 1536) model_tag_ = "qwen3:1.7b";
        else if (cfg.hidden_size <= 2560) model_tag_ = "qwen3:4b";
        else                              model_tag_ = "qwen3:8b";

        // FLM spawn strategy: per-request `flm run` with FILE stdio.
        // Known FLM v0.9.46 issues on Strix Halo:
        //  - fork+exec children with PIPE stdio hang on the NPU prefill kernel
        //  - `flm serve` mode degenerates into repeated-token loops ("plplpl")
        // FILE stdio works correctly ("2+2 equals 4" verified), so each query
        // spawns a fresh CLI process (warm model load ~11s).
        if (const char* p = getenv("NPU_FLM_PORT")) port_ = atoi(p);
        fprintf(stderr, "  NPU: FLM ready (%s, serve :%d, lazy spawn)\n", model_tag_.c_str(), port_);
        loaded_ = true;

        loaded_ = true;
        // estimated_tok_s() is a prior, not a measurement (issue #231). Real
        // throughput is reported per-request via InferenceResult.tok_s; until
        // then this is just the selection heuristic + a labelled estimate.
        fprintf(stderr, "  NPU: FLM ready (%s, ~%.0f tok/s est. — measured per-request)\n",
                model_tag_.c_str(), estimated_tok_s());
        return true;
    }

    void unload_model() override {
        if (pid_ > 0) {
            // Send /exit and give FLM a real chance to shut down gracefully
            // before escalating — the old code slept a fixed 500ms then
            // closed the pipes and sent SIGTERM after another fixed 200ms
            // sleep regardless of whether the process had already exited,
            // then sent SIGKILL unconditionally right after, even if
            // SIGTERM had already worked. Same bug class as #3
            // (backend_npu.cpp/backend_flm.cpp's already-fixed shutdown
            // paths), just a separate, not-yet-fixed occurrence here.
            // flm serve is a plain HTTP server: no stdin protocol, just
            // SIGTERM and escalate if it lingers.
            kill(pid_, SIGTERM);
            if (!wait_for_child(pid_, 3000)) {
                kill(pid_, SIGKILL);
                wait_for_child(pid_, 1000);
            }
            waitpid(pid_, nullptr, WNOHANG);
            pid_ = 0;
        }
        stdin_fd_ = stdout_fd_ = stderr_fd_ = -1;
        pending_prompt_.clear();
        generated_text_.clear();
        generated_pos_ = 0;
        loaded_ = false;
    }

    void reset_state() override {
        pending_prompt_.clear();
        generated_text_.clear();
        generated_pos_ = 0;
        saw_prior_call_ = false;
        queried_ = false;
    }

    // Send accumulated prompt to FLM, get full response
    // FLM uses ">>> " as its prompt delimiter. This function reads until
    // it sees ">>> " at the START of a line, which distinguishes it from
    // ">>> " that may appear mid-response (the actual bug this fixes).
    // Minimal HTTP/1.1 POST helper for the flm serve OpenAI API.
    static std::string http_post_json(int port, const std::string& path,
                                      const std::string& body) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return "";
        sockaddr_in a{}; a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons((uint16_t)port);
        if (connect(s, (sockaddr*)&a, sizeof(a)) != 0) { close(s); return ""; }
        struct timeval tv = {60, 0};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        std::string req = "POST " + path + " HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + body;
        send(s, req.data(), req.size(), 0);
        std::string resp;
        char buf[16384];
        int n;
        while ((n = (int)recv(s, buf, sizeof(buf), 0)) > 0) resp.append(buf, (size_t)n);
        close(s);
        size_t hdr = resp.find("\r\n\r\n");
        return (hdr == std::string::npos) ? resp : resp.substr(hdr + 4);
    }

    // Extract and unescape a string field from a JSON response body
    // (e.g. "content" in chat completions, "text" in completions).
    static std::string extract_json_string(const std::string& json, const std::string& field) {
        std::string key = "\"" + field + "\":\"";
        size_t p = json.find(key);
        if (p == std::string::npos) return "";
        p += key.size();
        std::string out;
        for (size_t i = p; i < json.size() && json[i] != '"'; i++) {
            if (json[i] == '\\' && i + 1 < json.size()) {
                char c = json[++i];
                if (c == 'n') out += '\n';
                else if (c == 't') out += '\t';
                else if (c == 'r') out += '\r';
                else if (c == 'u' && i + 4 < json.size()) {
                    unsigned v = (unsigned)strtoul(json.c_str() + i + 1, nullptr, 16);
                    i += 4;
                    if (v < 0x80) out += (char)v;
                    else if (v < 0x800) {
                        out += (char)(0xC0 | (v >> 6));
                        out += (char)(0x80 | (v & 0x3F));
                    } else {
                        out += (char)(0xE0 | (v >> 12));
                        out += (char)(0x80 | ((v >> 6) & 0x3F));
                        out += (char)(0x80 | (v & 0x3F));
                    }
                } else out += c;
            } else out += json[i];
        }
        return out;
    }

    static std::string json_escape(const std::string& s) {
        std::string out;
        for (unsigned char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                    else out += (char)c;
            }
        }
        return out;
    }

    // Send the prompt to the FLM serve HTTP API, get the full response.
    // Per-request spawn: `flm run <tag>` with FILE stdio (pipes hang in
    // fork children; serve mode degenerates). Writes the prompt to a file,
    // waits for the session transcript to reach the final ">>> ", parses
    // the response from after "Model RAW Output: ", then kills the child.
    // ── flm serve mode ──────────────────────────────────────────────
    // One persistent `flm serve <tag> -p <port>` child, spawned lazily on the
    // first query (model load ~11-25s, amortized). Queries go over the OpenAI
    // /v1/completions API. Measured ~45 tok/s vs ~4.8 tok/s for the retired
    // per-request `flm run` spawn (official FastFlowLM q4nx weights).
    bool ensure_serve() {
        if (pid_ > 0) return true;

        // Sanitize LD_LIBRARY_PATH: the parent may have the-rock HIP libs
        // (needed for zaya's GPU backends) which corrupt FLM's NPU runtime.
        if (const char* cur = getenv("LD_LIBRARY_PATH")) {
            std::string s(cur), keep;
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t colon = s.find(':', pos);
                std::string part = s.substr(pos, colon == std::string::npos
                    ? std::string::npos : colon - pos);
                std::string low = part;
                for (auto& c : low) c = (char)tolower(c);
                if (low.find("flm") != std::string::npos)
                    keep += (keep.empty() ? "" : ":") + part;
                if (colon == std::string::npos) break;
                pos = colon + 1;
            }
            if (!keep.empty()) setenv("LD_LIBRARY_PATH", keep.c_str(), 1);
            else unsetenv("LD_LIBRARY_PATH");
        }

        // Retry the spawn: the NPU's column/slot budget is shared with the
        // other zaya/FLM services (each model takes 8 of the 40 columns).
        // When another service is mid-startup, flm-real's CREATE_CONTEXT
        // fails with MGMT_ERT_NOAVAIL and the child exits after its own
        // short retry budget. A column always frees up (total footprints
        // fit), so a later attempt succeeds — wait it out instead of
        // failing the request. NPU_FLM_SERVE_RETRIES / NPU_FLM_SERVE_RETRY_DELAY_S
        // tune the budget (defaults: 6 attempts, 5s backoff).
        int attempts = getenv("NPU_FLM_SERVE_RETRIES") ? atoi(getenv("NPU_FLM_SERVE_RETRIES")) : 6;
        int delay_s = getenv("NPU_FLM_SERVE_RETRY_DELAY_S") ? atoi(getenv("NPU_FLM_SERVE_RETRY_DELAY_S")) : 5;
        for (int attempt = 1; attempt <= attempts; attempt++) {
            if (attempt > 1) {
                fprintf(stderr, "  NPU: flm serve spawn attempt %d/%d failed (NPU busy?) — retrying in %ds\n",
                        attempt, attempts, delay_s);
                sleep(delay_s);
            }

            pid_t pid = fork();
            if (pid < 0) return false;
            if (pid == 0) {
                // Die with the parent: a crashed/restarted zaya server must not
                // leak a flm-real child holding an NPU context. Leaked contexts
                // accumulate until CREATE_CONTEXT fails with MGMT_ERT_NOAVAIL
                // for everyone else.
                prctl(PR_SET_PDEATHSIG, SIGTERM);
                std::string log = "/tmp/flm_serve_" + std::to_string(getpid()) + ".log";
                int lfd = open(log.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0644);
                if (lfd >= 0) { dup2(lfd, STDOUT_FILENO); dup2(lfd, STDERR_FILENO); }
                int devnull = open("/dev/null", O_RDONLY);
                if (devnull >= 0) dup2(devnull, STDIN_FILENO);
                for (int fd = 3; fd < 1024; fd++) close(fd);
                // FLM needs its model registry; without FLM_CONFIG_PATH it exits
                // immediately ("model_list.json not found").
                const char* cfg = getenv("NPU_FLM_CONFIG");
                setenv("FLM_CONFIG_PATH", cfg ? cfg : "/opt/fastflowlm/etc/flm/model_list.json", 1);
                const char* xclb = getenv("NPU_FLM_XCLBINS");
                if (xclb) setenv("FLM_XCLBIN_PATH", xclb, 1);
                std::string port_s = std::to_string(port_);
                execl(flm_bin_.c_str(), "flm", "serve", model_tag_.c_str(),
                      "-p", port_s.c_str(), nullptr);
                _exit(1);
            }
            pid_ = pid;

            // Wait for the HTTP port to accept (model load takes ~11-25s).
            auto t0 = std::chrono::steady_clock::now();
            bool opened = false, died = false;
            while (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - t0).count() < 90) {
                int s = socket(AF_INET, SOCK_STREAM, 0);
                if (s >= 0) {
                    sockaddr_in a{}; a.sin_family = AF_INET;
                    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                    a.sin_port = htons((uint16_t)port_);
                    if (connect(s, (sockaddr*)&a, sizeof(a)) == 0) { close(s); opened = true; break; }
                    close(s);
                }
                int st = 0;
                if (waitpid(pid_, &st, WNOHANG) == pid_) { died = true; break; }
                usleep(500000);
            }
            if (opened) return true;
            if (!died) {
                // Port never opened but the child is still alive: tear it down.
                kill(pid_, SIGTERM);
                wait_for_child(pid_, 3000);
                kill(pid_, SIGKILL);
                waitpid(pid_, nullptr, WNOHANG);
            }
            pid_ = 0;
        }
        fprintf(stderr, "  NPU: flm serve failed after %d spawn attempts\n", attempts);
        return false;
    }

    std::string query_flm(const std::string& prompt) {
        fprintf(stderr, "  NPU: query_flm(prompt=%zu B): %.120s\n", prompt.size(), prompt.c_str());
        if (!ensure_serve()) {
            fprintf(stderr, "  NPU: flm serve on :%d unavailable\n", port_);
            return "";
        }
        std::string body = "{\"prompt\":\"" + json_escape(prompt) +
                           "\",\"max_tokens\":" + std::to_string(max_gen_tokens_) + "}";
        std::string resp = http_post_json(port_, "/v1/completions", body);
        std::string content = extract_json_string(resp, "text");
        return content;
    }

    // Forward: returns chars as token IDs matching SimpleTokenizer::decode() range
    // (printable ASCII 32-126 -> 132-226; raw bytes 127-255 -> 327-455).
    int forward(int token_id, int pos) override {
        if (!loaded_) return 106;

        // Accumulate prompt tokens
        pending_prompt_.push_back(token_id);

        // Query FLM when the full prompt has arrived: the router feeds prefill
        // tokens at pos 0..P-2, then generation restarts at pos 0. A pos reset
        // to 0 after any prior call means generation start = prompt complete.
        bool prompt_complete = (pos == 0 && saw_prior_call_);
        saw_prior_call_ = true;

        if (prompt_complete && !queried_) {
            fprintf(stderr, "  NPU: trigger pos=%d saw_prior=%d queried=%d pending=%zu\n",
                    pos, (int)saw_prior_call_, (int)queried_, pending_prompt_.size());
            // Prefer the raw user text from the server; fall back to the
            // char-shifted token reconstruction.
            std::string prompt = g_npu_prompt_text;
            if (prompt.empty() && !pending_prompt_.empty()) {
                for (int t : pending_prompt_) {
                    if (t == 2 || t == 106) continue;
                    if (t >= 132 && t <= 226) prompt += (char)(t - 100);
                    else if (t >= 327 && t <= 455) prompt += (char)(t - 200);
                }
            }
            if (prompt.empty()) prompt = "Hello";

            generated_text_ = query_flm(prompt);
            generated_pos_ = 0;
            queried_ = true;
        }

        // Return characters shifted to SimpleTokenizer-compatible range.
        // Collision-free scheme: printable ASCII 32-126 -> 132-226 (+100);
        // everything else (control chars 0-31, raw bytes 127-255) -> +300
        // -> [300, 555]. (The old +200 scheme collided: 'e'-'~' -> 201-226
        // overlapped control chars 1-26 -> 201-226.)
        if (generated_pos_ < generated_text_.size()) {
            unsigned char c = (unsigned char)generated_text_[generated_pos_++];
            if (c >= 32 && c <= 126) return c + 100;  // printable ASCII -> 132-226
            return (int)c + 300;  // control/raw -> 300-555
        }
        return 106; // EOS
    }

    // ─── Higher-level generate interface (used by token_router) ────
    InferenceResult generate(const std::string& prompt, int max_tokens = 256) {
        InferenceResult r;
        if (!loaded_) { r.text = "[npu: not loaded]"; return r; }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::string text = query_flm(prompt);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        r.text = text;

        // Estimate token count (rough: ~4 chars per token)
        int est_tokens = std::max(1, (int)text.size() / 4);
        r.tokens.resize(est_tokens);
        for (int i = 0; i < est_tokens && i < (int)text.size(); i++)
            r.tokens[i] = (unsigned char)text[i];

        r.gen_ms = ms;
        r.tok_s = ms > 0 ? est_tokens / (ms / 1000.0f) : 0;
        return r;
    }
};

std::vector<InferenceBackend*> detect_backends_npu() {
    std::vector<InferenceBackend*> backends;
    static NpuFlmTestBackend npu;
    backends.push_back(&npu);
    return backends;
}

// Also export the old name for backward compat
extern std::vector<InferenceBackend*> detect_backends_npu_flm() {
    return detect_backends_npu();
}
