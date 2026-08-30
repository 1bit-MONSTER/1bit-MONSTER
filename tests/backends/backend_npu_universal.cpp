// backend_npu_universal.cpp — NPU backend for zaya via npu_engine_universal
// (the open NPU worker, issue #1832)
//
// The zaya TokenRouter previously had only ONE NPU InferenceBackend — the
// FLM-based NpuFlmTestBackend (backend_npu.cpp) — and load_model was
// format-blind, so a .q4nx model fell through to the GGUF-only Universal
// loader (backend_generic.cpp) and CPU. This backend ports the validated
// NpuWorker protocol from src/backend_npu.cpp (READY handshake, op=32 fused
// embed→layers→lm_head, op=33 batched) into an InferenceBackend, gated on
// cfg.format == ModelFormat::Q4NX so the router can select it for .q4nx
// models. Spawn strategy and protocol are identical to the production
// unified_server path (src/backend_manager.cpp create_npu_backend).
//
// Availability: NPU_ENGINE_BIN (or ./npu_engine_universal) must exist and
// be executable. The engine binary is Q4NX-only by contract; this backend
// rejects every other format in load_model() so the router never routes a
// GGUF here (mirrors src/backend_npu.cpp init()).
//
// Issue #56 caveat (from the FLM comment, kept as a warning): xclbin load
// can hang on some Vitis/XRT combos — the worker handshake is bounded by a
// timeout so the server never blocks forever.

#include "backend.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <fcntl.h>

namespace {

// ── NPU worker subprocess (fork/exec + pipe protocol) ──
// Ported 1:1 from src/backend_npu.cpp NpuWorker (op codes 0/31/32/33/34).
struct NpuWorker {
    pid_t pid = -1;
    int stdin_fd = -1;
    int stdout_fd = -1;
    bool ready = false;

    bool read_with_timeout(int fd, void* buf, size_t n, int timeout_ms) {
        size_t got = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (got < n) {
            int remain = timeout_ms - (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (remain <= 0) return false;
            fd_set fds; FD_ZERO(&fds); FD_SET(fd, &fds);
            struct timeval tv; tv.tv_sec = remain / 1000; tv.tv_usec = (remain % 1000) * 1000;
            int r = select(fd + 1, &fds, nullptr, nullptr, &tv);
            if (r <= 0) return false;
            ssize_t n2 = read(fd, (char*)buf + got, n - got);
            if (n2 <= 0) return false;
            got += (size_t)n2;
        }
        return true;
    }

    bool spawn(const std::string& model_path,
               int H, int NC, int NQ, int NKV, int HD, int IM, int NV) {
        static const bool _sigpipe_ignored = []{
            signal(SIGPIPE, SIG_IGN);
            return true;
        }();
        (void)_sigpipe_ignored;

        const char* engine_bin = getenv("NPU_ENGINE_BIN");
        std::string bin = engine_bin ? engine_bin : "./npu_engine_universal";

        int to_child[2], from_child[2];
        if (pipe(to_child) < 0 || pipe(from_child) < 0) return false;

        auto set_env_int = [](const char* k, int v) {
            char buf[32]; snprintf(buf, sizeof(buf), "%d", v);
            setenv(k, buf, 1);
        };
        set_env_int("NPU_H", H);
        set_env_int("NPU_NC", NC);
        set_env_int("NPU_NH", NQ);
        set_env_int("NPU_NKV", NKV);
        set_env_int("NPU_HD", HD);
        set_env_int("NPU_IM", IM);
        set_env_int("NPU_NV", NV);

        pid = fork();
        if (pid < 0) { close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]); return false; }
        if (pid == 0) {
            close(to_child[1]); dup2(to_child[0], STDIN_FILENO); close(to_child[0]);
            close(from_child[0]); dup2(from_child[1], STDOUT_FILENO); close(from_child[1]);
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) dup2(devnull, STDERR_FILENO);
            execlp(bin.c_str(), bin.c_str(), model_path.c_str(), "--worker", (char*)nullptr);
            fprintf(stderr, "NPU: failed to exec %s\n", bin.c_str());
            _exit(1);
        }
        close(to_child[0]); close(from_child[1]);
        stdin_fd = to_child[1];
        stdout_fd = from_child[0];

        // READY handshake (issue #365), bounded at 10 s
        char ready_buf[6];
        int ready_bytes = 0;
        auto t0 = std::chrono::steady_clock::now();
        // READY timeout: model pack scales with size — the 35B-A3B q4nx takes
        // ~45-60s to dequant+pack+init before emitting READY (measured on
        // strixhalo 2026-08-28). The 10s limit (copied from src/backend_npu.cpp)
        // killed big models; NPU_WORKER_READY_TIMEOUT_S overrides (default 300s).
        int ready_timeout = 300;
        if (const char* e = getenv("NPU_WORKER_READY_TIMEOUT_S")) {
            int v = atoi(e);
            if (v > 0) ready_timeout = v;
        }
        while (ready_bytes < 6 && std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count() < ready_timeout) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
            struct timeval tv = {1, 0};
            if (select(stdout_fd + 1, &fds, nullptr, nullptr, &tv) > 0) {
                ssize_t n = read(stdout_fd, ready_buf + ready_bytes, 6 - ready_bytes);
                if (n > 0) ready_bytes += (int)n;
                else if (n <= 0) break;
            }
        }
        if (ready_bytes >= 6 && memcmp(ready_buf, "READY\n", 6) == 0) {
            ready = true;
        } else {
            fprintf(stderr, "NPU: worker handshake failed (got %d bytes)\n", ready_bytes);
            kill(pid, SIGTERM); waitpid(pid, nullptr, 0);
            close(stdin_fd); close(stdout_fd); stdin_fd = stdout_fd = -1; pid = -1;
            return false;
        }
        return true;
    }

    static constexpr int GEMM_TIMEOUT_MS = 30000;
    bool gemm(int op, int layer, int batch, int in_dim,
              const float* in_data, std::vector<float>& out_data) {
        if (!ready || stdin_fd < 0 || stdout_fd < 0) return false;
        uint32_t hdr[4] = {(uint32_t)op, (uint32_t)layer, (uint32_t)batch, (uint32_t)in_dim};
        if (write(stdin_fd, hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) return false;
        if (batch * in_dim > 0 &&
            write(stdin_fd, in_data, (size_t)batch * (size_t)in_dim * sizeof(float)) !=
            (ssize_t)((size_t)batch * (size_t)in_dim * sizeof(float))) return false;
        uint32_t resp[2];
        if (!read_with_timeout(stdout_fd, resp, sizeof(resp), GEMM_TIMEOUT_MS)) return false;
        if (resp[0] != 0) return false;
        uint32_t out_dim = resp[1];
        static constexpr uint32_t MAX_SAFE_OUT_DIM = 256 * 1024;
        if (out_dim > MAX_SAFE_OUT_DIM) return false;
        out_data.resize((size_t)batch * (size_t)out_dim);
        return read_with_timeout(stdout_fd, out_data.data(),
                                 (size_t)batch * (size_t)out_dim * sizeof(float),
                                 GEMM_TIMEOUT_MS);
    }

    void shutdown() {
        if (pid > 0) {
            uint32_t quit[4] = {0, 0, 0, 0};
            if (stdin_fd >= 0) {
                write(stdin_fd, quit, sizeof(quit));
                close(stdin_fd); stdin_fd = -1;
            }
            kill(pid, SIGTERM);
            for (int i = 0; i < 20 && waitpid(pid, nullptr, WNOHANG) == 0; i++) usleep(100000);
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, 0);
            pid = -1;
        }
        if (stdout_fd >= 0) { close(stdout_fd); stdout_fd = -1; }
        ready = false;
    }

    ~NpuWorker() { shutdown(); }
};

// ── Q4NX NPU universal backend (InferenceBackend) ──
class NpuUniversalBackend : public InferenceBackend {
public:
    NpuWorker worker;
    ModelConfig cfg_;
    std::string model_path_;
    bool loaded_ = false;
    int pos_ = 0;

    BackendType type() const override { return BackendType::NPU_XRT; }
    const char* name() const override { return "NPU XDNA (universal)"; }
    float estimated_tok_s() const override { return 40.0f; }  // estimate; corrected at runtime
    bool is_coherent() const override { return true; }

    bool is_available() override {
        const char* engine_bin = getenv("NPU_ENGINE_BIN");
        std::string bin = engine_bin ? engine_bin : "./npu_engine_universal";
        if (access(bin.c_str(), X_OK) == 0) return true;
        // Also accept the repo build location so zaya finds it out of the box
        std::string alt = "build/npu_engine_universal";
        return access(alt.c_str(), X_OK) == 0;
    }

    bool load_model(const ModelConfig& cfg) override {
        // The engine binary only speaks Q4NX — reject everything else up
        // front so the router never routes a GGUF here (#1832).
        if (cfg.format != ModelFormat::Q4NX) {
            fprintf(stderr, "NPU(univ): Q4NX-only — rejecting %s (format %d)\n",
                    cfg.model_path.c_str(), (int)cfg.format);
            return false;
        }
        cfg_ = cfg;
        const char* mp = getenv("NPU_MODEL_PATH");
        model_path_ = (mp && mp[0]) ? mp : cfg.model_path;
        if (model_path_.empty()) {
            fprintf(stderr, "NPU(univ): no model path (set NPU_MODEL_PATH or pass a .q4nx path)\n");
            return false;
        }
        // npu_engine_universal needs the model dims in env; derive them from
        // the config (hidden/n_heads/n_kv/n_layers/head_dim/intermediate/vocab).
        if (!worker.spawn(model_path_,
                          cfg.hidden_size, cfg.num_layers, cfg.num_heads,
                          cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size,
                          cfg.vocab_size)) {
            fprintf(stderr, "NPU(univ): worker spawn failed for %s\n", model_path_.c_str());
            return false;
        }
        loaded_ = true;
        pos_ = 0;
        fprintf(stderr, "NPU(univ): worker ready (%s)\n", model_path_.c_str());
        return true;
    }

    void unload_model() override {
        worker.shutdown();
        loaded_ = false;
    }

    int forward(int token_id, int pos) override {
        (void)pos;
        if (!loaded_) return -1;
        // Fused op=32: embed → layers → lm_head → next token (1 IPC call).
        std::vector<float> in_data(1, (float)token_id);
        std::vector<float> out_data;
        if (!worker.gemm(32, 0, 1, 1, in_data.data(), out_data)) {
            fprintf(stderr, "NPU(univ): fused generate failed\n");
            return -1;
        }
        pos_++;
        return out_data.empty() ? -1 : (int)out_data[0];
    }

    void reset_state() override {
        pos_ = 0;
        if (worker.ready) {
            std::vector<float> dummy;
            worker.gemm(31, 0, 0, 0, nullptr, dummy);  // reset KV cache
        }
    }
};

}  // namespace

// Registered by detect_backends() in backend_cpu.cpp — placed BEFORE the
// generic/CPU backends so a .q4nx model with a working engine binary is
// selected over GGUF-only fallbacks (#1832).
std::vector<InferenceBackend*> detect_backends_npu_universal() {
    std::vector<InferenceBackend*> backends;
    static NpuUniversalBackend npu;
    backends.push_back(&npu);
    return backends;
}
