// hrx_inprocess.h — in-process HRX inference via dlopen'd bundle libllama.so.
//
// Fork-A core (docs/research/hrx-engine-goal.md, P2): instead of spawning the
// HRX llama-server as a subprocess and talking OpenAI wire format, dlopen the
// hrx-b59 bundle's libllama.so (RTLD_NOW|LOCAL|DEEPBIND — mandatory: the
// bundle's symbols are unversioned and 1bit statically links its own llama.cpp)
// and drive token-level inference through the llama C API, with weights
// offloaded to the HRX backend device. All calls go through dlsym'd function
// pointers; no HRX headers or ROCm needed at build time.
//
// ABI note: the struct layouts below are copied verbatim from the hrx-b59
// bundle's include/llama.h (version 0.0.10320) and statically asserted against
// sizes measured by compiling against that header. They are the ABI of the
// dlopen'd library, which differs from the engine's vendored llama.cpp.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace hrx {

// Token-level in-process HRX engine. Not thread-safe; callers serialize
// (BackendManager holds compute_mtx around generate()).
class Inprocess {
public:
    Inprocess();
    ~Inprocess();

    Inprocess(const Inprocess&) = delete;
    Inprocess& operator=(const Inprocess&) = delete;

    // Load the bundle's libllama.so + initialize backends. Returns false on
    // any failure (dlopen, symbol resolution, backend init).
    bool init();

    // Load a GGUF model with layers offloaded to the HRX device.
    // model_path: path to a .gguf; n_gpu_layers: <0 = all; ctx_size: 0 = model default.
    bool load_model(const std::string& model_path, int n_gpu_layers, uint32_t ctx_size);

    // #1942 D2 hybrid: import a llama_state session blob (saved by a compatible
    // llama.cpp build, e.g. the GGML_HIP prefill lane) into the current context
    // so generate() continues from the imported KV. Call AFTER load_model(),
    // BEFORE generate(). Returns the number of imported tokens, or -1.
    long load_session_file(const std::string& session_path);

    // #1942 D2 zero-copy hybrid: import a llama_state session blob from SHARED
    // MEMORY (memfd/dma-buf fd, mmap'd MAP_SHARED) instead of a file path -
    // same session-v9 format, zero file I/O. Call AFTER load_model(), BEFORE
    // generate(). Returns the number of imported tokens, or -1.
    long load_session_mem(int fd);

    /// Pin this instance's offload device before init() ("HRX0", "Vulkan0",
    /// or "none" for default order). Router policy chooses per model class.
    void set_device_pin(const std::string& dev);

    /// Last input token of the imported session (for continuation), or -1.
    int resume_token() const;

    /// Export the current context state into a caller-provided memfd (session
    /// layout, resume-token aware). Returns raw state bytes, or -1.
    int export_session_mem(int fd_out);

    /// Tokenize text with the loaded model vocab (llama_tokenize, add_special).
    /// Returns token count, or -1 on failure (n_max too small = -needed).
    int tokenize(const std::string& text, int32_t* out_tokens, int32_t n_max) const;

    // One decode step: feed token_id, return argmax next token, or -1 on failure.
    int generate(int token_id);

    // Reset KV state (recreates the context — this fork exports no kv-clear C API).
    bool reset();

    void unload();

    bool has_model() const;
    bool has_hrx_device() const;
    const char* hrx_device_name() const;
    int vocab_size() const;
    int n_embd() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Cheap static probe: dlopen the bundle and report HRX device count.
// Returns 1 if the bundle loads and the HRX backend sees ≥1 device, else 0.
bool probe_bundle(const std::string& bundle_lib_path);

}  // namespace hrx
