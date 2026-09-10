// backend_lse.cpp — LSE backend: spawn lse-server, health-check, text-level
// inference over the OpenAI wire format. M0: spawn + /health + destroy.
// M1: generate_text via POST /v1/completions (stream=false), JSON parse via
// nlohmann (already in the tree — backend_npu_flm.cpp uses it; onebin links
// nlohmann_json::nlohmann_json).
//
// Lifecycle mirrors backend_npu_flm.cpp (fork/exec subprocess, retry on
// spawn, SIGTERM→SIGKILL teardown), with an HTTP protocol instead of a REPL
// pipe: lse-server is a persistent daemon, so a 19 GB model loads once.
#include "backend_lse.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace {

std::string env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::string(v) : std::string(dflt);
}

// Random hex API key from /dev/urandom (fall back to rand for embedded runs).
std::string random_key(int bytes = 16) {
    std::vector<unsigned char> buf(static_cast<size_t>(bytes));
    FILE* f = std::fopen("/dev/urandom", "rb");
    if (f) {
        size_t got = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (got != buf.size()) buf.assign(buf.size(), 0);
    } else {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (auto& b : buf) b = static_cast<unsigned char>(std::rand() & 0xff);
    }
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(buf.size() * 2);
    for (auto b : buf) { out += hex[b >> 4]; out += hex[b & 0xf]; }
    return out;
}

// Bind 127.0.0.1:0, read back the ephemeral port, close. Small race (the
// server could grab a different port) but fine for localhost spawn; retries
// cover it.
bool pick_ephemeral_port(std::string* port_out) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (bind(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return false; }
    socklen_t len = sizeof a;
    if (getsockname(fd, (sockaddr*)&a, &len) != 0) { close(fd); return false; }
    char buf[16];
    std::snprintf(buf, sizeof buf, "%u", static_cast<unsigned>(ntohs(a.sin_port)));
    *port_out = buf;
    close(fd);
    return true;
}

}  // namespace

LseBackend::LseBackend() {
    type = BackendType::LSE_GPU;
    name = "lse";
    spawn_retries_ = std::atoi(env_or("LSE_SPAWN_RETRIES", "10").c_str());
    retry_delay_s_ = std::atoi(env_or("LSE_RETRY_DELAY_S", "5").c_str());
    init_timeout_s_ = std::atoi(env_or("LSE_INIT_TIMEOUT_S", "120").c_str());
    server_bin_ = env_or("LSE_SERVER_BIN", "lse-server");
}

bool LseBackend::init(const ModelConfig& cfg, const std::string& weights_dir) {
    // LSE serves the MLX lane only. Reject everything else up front so the
    // auto-selectable registration in BackendManager never makes every model
    // load pay the spawn-retry cost below (mirrors NpuFlmBackend's Q4NX-only
    // guard).
    if (cfg.format != ModelFormat::MLX) {
        fprintf(stderr, "LSE: MLX-format only — rejecting %s (format %d)\n",
                cfg.model_path.c_str(), (int)cfg.format);
        return false;
    }
    this->cfg = cfg;
    // MLX checkpoints are directories; the router passes the checkpoint dir in
    // cfg.model_path (discovery sets it for ModelFormat::MLX). Fall back to
    // weights_dir for direct API use / selfcheck.
    model_dir_ = !cfg.model_path.empty() ? cfg.model_path : weights_dir;
    if (model_dir_.empty()) {
        fprintf(stderr, "LSE: no model directory to serve\n");
        return false;
    }
    // Fail fast when lse-server is absent (check LSE_SERVER_BIN / PATH) —
    // otherwise every non-LSE model pays 10×5s of spawn retries.
    if (access(server_bin_.c_str(), X_OK) != 0) {
        fprintf(stderr, "LSE: lse-server not found at %s (set LSE_SERVER_BIN)\n",
                server_bin_.c_str());
        return false;
    }
    port_ = env_or("LSE_PORT", "");
    if (port_.empty() && !pick_ephemeral_port(&port_)) {
        fprintf(stderr, "LSE: could not pick an ephemeral port\n");
        return false;
    }
    api_key_ = random_key();

    for (int attempt = 1; attempt <= spawn_retries_; ++attempt) {
        if (spawn_server() && wait_healthy(init_timeout_s_)) {
            initialized_ = true;
            fprintf(stderr, "LSE: lse-server up on 127.0.0.1:%s (pid %d)\n",
                    port_.c_str(), pid_);
            return true;
        }
        fprintf(stderr, "LSE: spawn attempt %d/%d failed — retrying in %ds\n",
                attempt, spawn_retries_, retry_delay_s_);
        kill_server();
        sleep(static_cast<unsigned>(retry_delay_s_));
    }
    fprintf(stderr, "LSE: failed to start lse-server after %d attempts\n",
            spawn_retries_);
    return false;
}

bool LseBackend::spawn_server() {
    pid_t pid = fork();
    if (pid < 0) { perror("LSE: fork"); return false; }
    if (pid == 0) {
        // Child: detach stdio unless asked to keep it.
        if (env_or("LSE_KEEP_STDIO", "").empty()) {
            int devnull = open("/dev/null", O_RDWR);
            if (devnull >= 0) {
                dup2(devnull, 0); dup2(devnull, 1); dup2(devnull, 2);
                if (devnull > 2) close(devnull);
            }
        }
        execl(server_bin_.c_str(), "lse-server",
              "-m", model_dir_.c_str(),
              "--host", "127.0.0.1",
              "--port", port_.c_str(),
              "--api-key", api_key_.c_str(),
              (char*)nullptr);
        fprintf(stderr, "LSE: failed to exec %s\n", server_bin_.c_str());
        _exit(127);
    }
    pid_ = pid;
    return true;
}

bool LseBackend::wait_healthy(int timeout_s) {
    const int deadline = static_cast<int>(std::time(nullptr)) + timeout_s;
    while (std::time(nullptr) < deadline) {
        std::string resp = http_get("/health");
        if (!resp.empty() && resp.find("200") != std::string::npos &&
            resp.find("ok") != std::string::npos) {
            return true;
        }
        usleep(500000);
    }
    return false;
}

std::string LseBackend::http_get(const std::string& path) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(std::atoi(port_.c_str())));
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    timeval tv{2, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (connect(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return ""; }
    // lse-server gates EVERY route (including /health) behind the API key.
    std::string req = "GET " + path + " HTTP/1.1\r\n"
                      "Host: 127.0.0.1\r\n"
                      "Authorization: Bearer " + api_key_ + "\r\n"
                      "Connection: close\r\n\r\n";
    (void)send(fd, req.data(), req.size(), 0);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof buf, 0)) > 0) out.append(buf, static_cast<size_t>(n));
    close(fd);
    return out;
}

void LseBackend::kill_server() {
    if (pid_ <= 0) return;
    kill(pid_, SIGTERM);
    int status = 0;
    for (int i = 0; i < 20; ++i) {  // ~2s grace
        if (waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; return; }
        usleep(100000);
    }
    kill(pid_, SIGKILL);
    waitpid(pid_, &status, 0);
    pid_ = -1;
}

void LseBackend::destroy() {
    kill_server();
    initialized_ = false;
}

bool LseBackend::reset() { return true; }

bool LseBackend::forward(int, float*) {
    fprintf(stderr, "LSE: forward() not supported — use generate_text() (text-level)\n");
    return false;
}

int LseBackend::generate(int) {
    fprintf(stderr, "LSE: use generate_text() for text-level inference\n");
    return -1;
}

bool LseBackend::lm_head(const float*, float*, int*) {
    fprintf(stderr, "LSE: lm_head() not supported — lse-server handles this internally\n");
    return false;
}

std::string LseBackend::http_post(const std::string& path, const std::string& body,
                                  int* status_out) {
    *status_out = 0;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(std::atoi(port_.c_str())));
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    // Generations can take tens of seconds at long context; keep the socket
    // alive for the whole request (recv timeout applies per recv, not total).
    timeval tv{30, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (connect(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return ""; }
    std::string req = "POST " + path + " HTTP/1.1\r\n"
                      "Host: 127.0.0.1\r\n"
                      "Authorization: Bearer " + api_key_ + "\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\n"
                      "Connection: close\r\n\r\n" + body;
    (void)send(fd, req.data(), req.size(), 0);
    std::string out;
    char buf[16384];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof buf, 0)) > 0) out.append(buf, static_cast<size_t>(n));
    close(fd);
    if (out.empty()) return "";
    // Split headers/body, capture the status line.
    size_t hdr_end = out.find("\r\n\r\n");
    if (hdr_end != std::string::npos) {
        std::string head = out.substr(0, hdr_end);
        size_t sp = head.find(' ');
        if (sp != std::string::npos)
            *status_out = std::atoi(head.c_str() + sp + 1);
        return out.substr(hdr_end + 4);
    }
    return out;
}

std::string LseBackend::generate_text(const std::string& prompt, int max_tokens) {
    if (pid_ <= 0 || !initialized_) return "";
    if (max_tokens <= 0) max_tokens = 16;
    if (max_tokens > 4096) max_tokens = 4096;  // server default cap

    // JSON-escape the prompt minimally (nlohmann handles \" \\ \n \t).
    std::string esc;
    esc.reserve(prompt.size() + 8);
    for (char c : prompt) {
        switch (c) {
            case '"':  esc += "\\\""; break;
            case '\\': esc += "\\\\"; break;
            case '\n': esc += "\\n";  break;
            case '\r': esc += "\\r";  break;
            case '\t': esc += "\\t";  break;
            default:   esc += c;      break;
        }
    }
    std::string body = "{\"prompt\":\"" + esc + "\",\"max_tokens\":" +
                       std::to_string(max_tokens) + ",\"stream\":false}";
    int status = 0;
    std::string resp = http_post("/v1/completions", body, &status);
    if (status != 200 || resp.empty()) {
        fprintf(stderr, "LSE: /v1/completions HTTP %d — %s\n", status,
                resp.substr(0, 200).c_str());
        return "";
    }

    // Parse {"choices":[{"text": "...", ...}], "usage": {...}, "timings": {...}}.
    try {
        nlohmann::json j = nlohmann::json::parse(resp);
        if (j.contains("error") && !j["error"].is_null()) return "";  // error JSON
        if (!j.contains("choices") || j["choices"].empty()) return "";
        const auto& choice = j["choices"][0];
        if (!choice.contains("text") || !choice["text"].is_string()) return "";
        std::string text = choice["text"].get<std::string>();
        // Decode rate from lse-server's non-OpenAI timings field (M3).
        if (j.contains("timings") && j["timings"].contains("predicted_per_second") &&
            j["timings"]["predicted_per_second"].is_number())
            last_decode_tok_s_ = j["timings"]["predicted_per_second"].get<double>();
        return text;
    } catch (const nlohmann::json::exception& e) {
        fprintf(stderr, "LSE: JSON parse failed: %s\n", e.what());
        return "";
    }
}

float LseBackend::benchmark(int tokens) {
    // M3: measured decode tok/s from the last real generation (lse-server's
    // own timings), or -1 if nothing has been generated yet.
    (void)tokens;
    return last_decode_tok_s_ > 0.0 ? static_cast<float>(last_decode_tok_s_) : -1.0f;
}

// ── Factory (static-symbol dispatched by backend_factory.cpp) ──
extern "C" Backend* create_lse_backend() {
    return new LseBackend();
}
