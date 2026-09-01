// backend_hrx.cpp — HRX GPU backend: spawn the bundled hrx llama-server,
// health-check, and serve text-level inference over the OpenAI wire format.
//
// Lifecycle mirrors backend_lse.cpp (fork/exec subprocess, retry, SIGTERM→
// SIGKILL teardown), but spawns the HRX llama-server with AMD's fused HRX0
// device. The HRX bundle is self-contained (libhrx/libloomc/libggml-hrx ship
// next to llama-server), so NO ROCm install is needed on target.
#include "backend_hrx.h"
#include "hrx_inprocess.h"

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

// Locate the HRX llama-server executable. Precedence:
//   1. $HRX_MODEL_BIN (explicit full path)
//   2. $HRX_ROOT/bin/llama-server  (unpacked bundle root)
//   3. "llama-server" on PATH (self-relocating wrapper)
// Returns empty if not found (caller fails fast).
//
// NOTE (issue #1959): HRX_ROOT must be STABLE per process. The in-process
// bundle path (src/hrx_inprocess.cpp) is snapshotted on first use and a
// later HRX_ROOT change is ignored with a warning — the ggml build aborts on
// a fresh dlopen of a different path. Do not point a new model switch at a
// different bundle dir within one process.
std::string locate_hrx_server() {
    std::string bin = env_or("HRX_MODEL_BIN", "");
    if (!bin.empty()) return bin;

    std::string root = env_or("HRX_ROOT", "");
    if (root.empty()) {
        // Known default unpacked bundle.  b66 is the current official bundle
        // (fused Qwen3-MoE graphs).  Unfused graphs (dense models) still
        // fail-closed on GET_ROWS in b59/b66 — use the gfx1151 rebuild
        // (HRX_ROOT=/home/bcloud/hrx-gfx1151/llama-src/build) for those.
        // Ported-to-new-hrx b66 (ggml-hrx on hrx-system main ae91949) lives at
        // HRX_ROOT=/home/bcloud/hrx-gfx1151/llama-b66/build — it needs
        // HRX_LD_LIBRARY_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
        // (TheRock HSA 1.21; see spawn_server).
        const char* def = "/home/bcloud/hrx-slice/hrx-llamacpp/out/llama-hrx-b66";
        root = def;
    }
    if (!root.empty()) {
        std::string cand = root + "/bin/llama-server";
        if (access(cand.c_str(), X_OK) == 0) return cand;
    }
    // Fall back to PATH lookup (a self-relocating wrapper on PATH).
    return std::string("llama-server");
}

}  // namespace

HrxBackend::HrxBackend() {
    type = BackendType::HRX_GPU;
    name = "hrx_gpu";
    spawn_retries_ = std::atoi(env_or("HRX_SPAWN_RETRIES", "3").c_str());
    retry_delay_s_ = std::atoi(env_or("HRX_RETRY_DELAY_S", "3").c_str());
    init_timeout_s_ = std::atoi(env_or("HRX_INIT_TIMEOUT_S", "120").c_str());
    ctx_size_ = env_or("HRX_CTX_SIZE", "4096");
    server_bin_ = locate_hrx_server();
}

HrxBackend::~HrxBackend() {
    destroy();
}

bool HrxBackend::init(const ModelConfig& cfg, const std::string& weights_dir) {
    // HRX is a GGUF/H1B fused GPU lane. The discovery path may pass format 0
    // (UNKNOWN) before the model is classified, so accept both GGUF/H1B and the
    // UNKNOWN-with-path case: init() will fail fast on spawn anyway, and the
    // router cascades on failure. Only hard-reject the formats HRX cannot serve
    // (MLX, Q4NX, ONEBP, SAFETENSORS) so a non-GGUF model never pays spawn cost.
    if (cfg.format != ModelFormat::GGUF && cfg.format != ModelFormat::H1B &&
        cfg.format != ModelFormat::UNKNOWN) {
        fprintf(stderr, "HRX: GGUF/H1B only — rejecting %s (format %d)\n",
                cfg.model_path.c_str(), (int)cfg.format);
        return false;
    }
    if (cfg.model_path.empty() && weights_dir.empty()) {
        fprintf(stderr, "HRX: no model path to serve\n");
        return false;
    }
    model_path_ = !cfg.model_path.empty() ? cfg.model_path : weights_dir;
    this->cfg = cfg;

    // Fork-A in-process path (HRX_INPROCESS=1 opts in; subprocess is the
    // default): dlopen the bundle's libllama.so, offload weights to the HRX
    // device, and serve token-level generate() in-process.  On any failure
    // fall back to the subprocess llama-server spawn below.
    //
    // NOTE (2026-08-30): in-process mode segfaults inside the unified server
    // (a std::regex token-table corruption in llama.cpp's unicode_regex_split
    // when the bundle DSO is dlopen'd into the multi-backend 1bit process —
    // reproduced standalone-free, crashes only under the server; the b59
    // bundle predates the std::regex splitter and is unaffected).  The
    // subprocess path is the production default; it serves text-level chat
    // correctly (verified end-to-end: Qwen3-0.6B on HRX0, GET_ROWS-capable
    // gfx1151 build).
    if (env_or("HRX_INPROCESS", "0") != "0") {
        inprocess_ = std::make_unique<hrx::Inprocess>();
        int n_gpu_layers = std::atoi(env_or("HRX_N_GPU_LAYERS", "-1").c_str());
        uint32_t ctx = (uint32_t)std::atoi(ctx_size_.c_str());
        if (ctx == 0) ctx = 4096;
        if (inprocess_->init() && inprocess_->load_model(model_path_, n_gpu_layers, ctx)) {
            inprocess_mode_ = true;
            initialized_ = true;
            fprintf(stderr, "HRX: in-process engine active (token-level, %s)\n",
                    inprocess_->has_hrx_device() ? inprocess_->hrx_device_name()
                                                 : "default device order");
            return true;
        }
        fprintf(stderr, "HRX: in-process init failed — falling back to subprocess llama-server\n");
        inprocess_.reset();
        inprocess_mode_ = false;
    }

    // Fail fast when the HRX llama-server binary is absent — otherwise every
    // non-HRX model pays the spawn-retry cost.
    if (access(server_bin_.c_str(), X_OK) != 0) {
        fprintf(stderr, "HRX: llama-server not found at %s (set HRX_ROOT / HRX_MODEL_BIN)\n",
                server_bin_.c_str());
        return false;
    }

    // Secure an ephemeral loopback port.
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
    port_ = buf;
    close(fd);

    for (int attempt = 1; attempt <= spawn_retries_; ++attempt) {
        if (spawn_server() && wait_healthy(init_timeout_s_)) {
            initialized_ = true;
            fprintf(stderr, "HRX: llama-server up on 127.0.0.1:%s (pid %d)\n",
                    port_.c_str(), pid_);
            return true;
        }
        fprintf(stderr, "HRX: spawn attempt %d/%d failed — retrying in %ds\n",
                attempt, spawn_retries_, retry_delay_s_);
        kill_server();
        sleep(static_cast<unsigned>(retry_delay_s_));
    }
    fprintf(stderr, "HRX: failed to start llama-server after %d attempts (fails closed — router falls back)\n",
            spawn_retries_);
    return false;
}

bool HrxBackend::spawn_server() {
    pid_t pid = fork();
    if (pid < 0) { perror("HRX: fork"); return false; }
    if (pid == 0) {
        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, 0); dup2(devnull, 1); dup2(devnull, 2);
            if (devnull > 2) close(devnull);
        }
        // The new hrx-system's IREE amdgpu driver dlopens libhsa-runtime64.so.1
        // at runtime and requires a recent HSA (HSA_AMD_AGENT_INFO_PM4_EMULATION,
        // agent info 0xA119). The system ROCm HSA 1.18 rejects that query, so
        // bundles built against new hrx must run with the TheRock HSA first on
        // the library path. HRX_LD_LIBRARY_PATH is prepended to the inherited
        // LD_LIBRARY_PATH (empty by default: the official b66 bundle uses the
        // old hrx-system and works with the system HSA).
        std::string extra_ld = env_or("HRX_LD_LIBRARY_PATH", "");
        std::string ld = env_or("LD_LIBRARY_PATH", "");
        if (!extra_ld.empty()) {
            std::string combined = extra_ld;
            if (!ld.empty()) combined += ":" + ld;
            setenv("LD_LIBRARY_PATH", combined.c_str(), 1);
        }
        // HRX llama-server: fused HRX0 device + the flags AMD's recipe insists on.
        execl(server_bin_.c_str(), "llama-server",
              "-m", model_path_.c_str(),
              "--device", "HRX0",
              "--ctx-size", ctx_size_.c_str(),
              "--jinja",
              "--metrics",
              "--host", "127.0.0.1",
              "--port", port_.c_str(),
              (char*)nullptr);
        fprintf(stderr, "HRX: failed to exec %s\n", server_bin_.c_str());
        _exit(127);
    }
    pid_ = pid;
    return true;
}

bool HrxBackend::wait_healthy(int timeout_s) {
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

std::string HrxBackend::http_get(const std::string& path) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(std::atoi(port_.c_str())));
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    timeval tv{2, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (connect(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return ""; }
    std::string req = "GET " + path + " HTTP/1.1\r\n"
                      "Host: 127.0.0.1\r\n"
                      "Connection: close\r\n\r\n";
    (void)send(fd, req.data(), req.size(), 0);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof buf, 0)) > 0) out.append(buf, static_cast<size_t>(n));
    close(fd);
    return out;
}

std::string HrxBackend::http_post(const std::string& path, const std::string& body,
                                  int* status_out) {
    *status_out = 0;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(std::atoi(port_.c_str())));
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    timeval tv{60, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (connect(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return ""; }
    std::string req = "POST " + path + " HTTP/1.1\r\n"
                      "Host: 127.0.0.1\r\n"
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

void HrxBackend::kill_server() {
    if (pid_ <= 0) return;
    kill(pid_, SIGTERM);
    int status = 0;
    for (int i = 0; i < 20; ++i) {
        if (waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; return; }
        usleep(100000);
    }
    kill(pid_, SIGKILL);
    waitpid(pid_, &status, 0);
    pid_ = -1;
}

void HrxBackend::destroy() {
    if (inprocess_) inprocess_->unload();
    inprocess_.reset();
    inprocess_mode_ = false;
    kill_server();
    initialized_ = false;
}

bool HrxBackend::reset() {
    if (inprocess_mode_ && inprocess_) return inprocess_->reset();
    return true;
}

bool HrxBackend::forward(int, float*) {
    fprintf(stderr, "HRX: forward() not supported — use generate() (in-process) or generate_text() (subprocess)\n");
    return false;
}

int HrxBackend::generate(int token_id) {
    if (inprocess_mode_ && inprocess_) return inprocess_->generate(token_id);
    fprintf(stderr, "HRX: use generate_text() for text-level inference\n");
    return -1;
}

bool HrxBackend::lm_head(const float*, float*, int*) {
    fprintf(stderr, "HRX: lm_head() not supported — HRX llama-server handles this internally\n");
    return false;
}

std::string HrxBackend::generate_text(const std::string& prompt, int max_tokens) {
    if (pid_ <= 0 || !initialized_) return "";
    if (max_tokens <= 0) max_tokens = 16;
    if (max_tokens > 4096) max_tokens = 4096;

    // OpenAI chat format the HRX llama-server accepts.
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
    std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"" + esc +
                       "\"}],\"max_tokens\":" + std::to_string(max_tokens) +
                       ",\"temperature\":0.3,\"stream\":false}";
    int status = 0;
    std::string resp = http_post("/v1/chat/completions", body, &status);
    if (status != 200 || resp.empty()) {
        fprintf(stderr, "HRX: /v1/chat/completions HTTP %d — %s\n", status,
                resp.substr(0, 200).c_str());
        return "";
    }

    try {
        nlohmann::json j = nlohmann::json::parse(resp);
        if (j.contains("error") && !j["error"].is_null()) return "";
        if (!j.contains("choices") || j["choices"].empty()) return "";
        const auto& choice = j["choices"][0];
        if (!choice.contains("message")) return "";
        const auto& msg = choice["message"];
        // Reasoning models (Qwen3, DeepSeek) stream the chain-of-thought into
        // `reasoning_content` and only fill `content` once reasoning finishes;
        // with a small max_tokens the reply can live entirely in the reasoning
        // field.  Return content if present, else the reasoning — an empty
        // string here reads as a backend failure and cascades to CPU.
        std::string text = msg.value("content", "");
        if (text.empty()) text = msg.value("reasoning_content", "");
        if (j.contains("timings") && j["timings"].contains("predicted_per_second") &&
            j["timings"]["predicted_per_second"].is_number())
            last_decode_tok_s_ = j["timings"]["predicted_per_second"].get<double>();
        return text;
    } catch (const nlohmann::json::exception& e) {
        fprintf(stderr, "HRX: JSON parse failed: %s\n", e.what());
        return "";
    }
}

float HrxBackend::benchmark(int tokens) {
    (void)tokens;
    return last_decode_tok_s_ > 0.0 ? static_cast<float>(last_decode_tok_s_) : -1.0f;
}

// ── Factory (static-symbol dispatched by backend_manager.cpp) ──
extern "C" Backend* create_hrx_backend() {
    return new HrxBackend();
}
