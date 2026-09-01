// hrx_inprocess.cpp — in-process HRX via dlopen'd bundle libllama.so.
// See hrx_inprocess.h for the design. ABI structs below mirror the hrx-b59
// bundle's include/llama.h (0.0.10320) exactly; static_asserts pin the sizes.
#include "hrx_inprocess.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

namespace hrx {

// ── ABI mirrors (verbatim from bundle llama.h 0.0.10320) ──
struct llama_batch {
    int32_t n_tokens;
    int32_t* token;
    float* embd;
    int32_t* pos;
    int32_t* n_seq_id;
    int32_t** seq_id;
    int8_t* logits;
};
using llama_token = int32_t;
using llama_pos = int32_t;
using llama_seq_id = int32_t;

struct ggml_backend_device;
using ggml_backend_dev_t = ggml_backend_device*;
enum llama_split_mode : int { LLAMA_SPLIT_MODE_NONE = 0 };
enum llama_load_mode : int { LLAMA_LOAD_MODE_MMAP = 0 };
enum llama_context_type : int { LLAMA_CONTEXT_TYPE_SEQ = 0 };
enum llama_rope_scaling_type : int { LLAMA_ROPE_SCALING_TYPE_NONE = 0 };
enum llama_pooling_type : int { LLAMA_POOLING_TYPE_NONE = 0 };
enum llama_attention_type : int { LLAMA_ATTENTION_TYPE_CAUSAL = 0 };
enum llama_flash_attn_type : int { LLAMA_FLASH_ATTN_TYPE_AUTO = 0 };
enum ggml_type : int { GGML_TYPE_F32 = 0 };
using ggml_abort_callback = void (*)(void);
using ggml_backend_sched_eval_callback = void (*)(void);

struct llama_model_tensor_buft_override;
struct llama_model_kv_override;
struct llama_sampler_seq_config;
struct llama_model;
struct llama_context;
struct llama_vocab;

struct llama_model_params {
    ggml_backend_dev_t* devices;
    const llama_model_tensor_buft_override* tensor_buft_overrides;
    int32_t n_gpu_layers;
    enum llama_split_mode split_mode;
    enum llama_load_mode load_mode;
    int32_t main_gpu;
    const float* tensor_split;
    void (*progress_callback)(float, void*);
    void* progress_callback_user_data;
    const llama_model_kv_override* kv_overrides;
    bool vocab_only;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
    bool load_mtp;
};

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    enum llama_attention_type attention_type;
    enum llama_flash_attn_type flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval;
    void* cb_eval_user_data;
    enum ggml_type type_k;
    enum ggml_type type_v;
    ggml_abort_callback abort_callback;
    void* abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    llama_sampler_seq_config* samplers;
    size_t n_samplers;
    llama_context* ctx_other;
};

static_assert(sizeof(llama_model_params) == 72, "hrx-b59 llama_model_params ABI drift");
static_assert(sizeof(llama_context_params) == 160, "hrx-b59 llama_context_params ABI drift");
static_assert(sizeof(llama_batch) == 56, "hrx-b59 llama_batch ABI drift");

// ── Function-pointer vtable (dlsym'd) ──
using fn_llama_backend_init = void (*)(void);
using fn_llama_backend_free = void (*)(void);
using fn_llama_model_default_params = llama_model_params (*)(void);
using fn_llama_context_default_params = llama_context_params (*)(void);
using fn_llama_model_load_from_file = llama_model* (*)(const char*, llama_model_params);
using fn_llama_model_free = void (*)(llama_model*);
using fn_llama_init_from_model = llama_context* (*)(llama_model*, llama_context_params);
using fn_llama_free = void (*)(llama_context*);
using fn_llama_n_ctx = uint32_t (*)(const llama_context*);
using fn_llama_decode = int32_t (*)(llama_context*, llama_batch);
using fn_llama_get_logits_ith = float* (*)(llama_context*, int32_t);
using fn_llama_batch_get_one = llama_batch (*)(llama_token*, int32_t, llama_pos, llama_seq_id);
using fn_llama_model_get_vocab = llama_vocab* (*)(const llama_model*);
using fn_llama_vocab_n_tokens = int32_t (*)(const llama_vocab*);
using fn_llama_model_n_embd = int32_t (*)(const llama_model*);
using fn_llama_model_desc = int32_t (*)(const llama_model*, char*, size_t);
using fn_ggml_backend_dev_by_name = ggml_backend_dev_t (*)(const char*);
using fn_ggml_backend_dev_name = const char* (*)(ggml_backend_dev_t);
using fn_ggml_backend_hrx_get_device_count = int32_t (*)(void);

struct Inprocess::Impl {
    void* handle = nullptr;
    fn_llama_backend_init llama_backend_init = nullptr;
    fn_llama_backend_free llama_backend_free = nullptr;
    fn_llama_model_default_params llama_model_default_params = nullptr;
    fn_llama_context_default_params llama_context_default_params = nullptr;
    fn_llama_model_load_from_file llama_model_load_from_file = nullptr;
    fn_llama_model_free llama_model_free = nullptr;
    fn_llama_init_from_model llama_init_from_model = nullptr;
    fn_llama_free llama_free = nullptr;
    fn_llama_n_ctx llama_n_ctx = nullptr;
    fn_llama_decode llama_decode = nullptr;
    fn_llama_get_logits_ith llama_get_logits_ith = nullptr;
    fn_llama_batch_get_one llama_batch_get_one = nullptr;
    fn_llama_model_get_vocab llama_model_get_vocab = nullptr;
    fn_llama_vocab_n_tokens llama_vocab_n_tokens = nullptr;
    fn_llama_model_n_embd llama_model_n_embd = nullptr;
    fn_llama_model_desc llama_model_desc = nullptr;
    fn_ggml_backend_dev_by_name ggml_backend_dev_by_name = nullptr;
    fn_ggml_backend_dev_name ggml_backend_dev_name = nullptr;
    fn_ggml_backend_hrx_get_device_count ggml_backend_hrx_get_device_count = nullptr;

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context_params ctx_params{};
    ggml_backend_dev_t hrx_dev = nullptr;
    ggml_backend_dev_t devs[2] = {nullptr, nullptr};
    std::string hrx_dev_name;
    int vocab = 0;
    int n_embd = 0;
    llama_pos pos = 0;
};

namespace {

// Resolve the bundle .so path ONCE per process (issue #1959). This ggml
// build aborts on a second dlopen after dlclose (GGML_ASSERT in its static
// initializer), so the handle must stay mapped for the process lifetime and
// every subsequent init MUST dlopen the SAME path (a same-path dlopen just
// bumps the refcount — safe). Reading HRX_ROOT fresh on every call is a
// footgun: if an operator changes HRX_ROOT between backend instances, the
// new dlopen resolves a DIFFERENT path → the DSO is freshly mapped → ggml's
// static initializer re-runs → process abort. So we snapshot the path on
// first use and reject a different HRX_ROOT afterwards with a clear error.
std::string bundle_lib_path() {
    static const std::string cached = []() -> std::string {
        const char* root = std::getenv("HRX_ROOT");
        if (root && root[0]) {
            // gfx1151 llama-build puts the DSOs in bin/ (self-contained tree);
            // the b59 tarball layout puts them in lib/.  Try the canonical
            // lib/ first, then bin/ so both layouts resolve (GET_ROWS-capable
            // gfx1151 build + stock bundle).
            std::string lib = std::string(root) + "/lib/libllama.so";
            if (access(lib.c_str(), F_OK) == 0) return lib;
            std::string bin = std::string(root) + "/bin/libllama.so";
            if (access(bin.c_str(), F_OK) == 0) return bin;
            return lib;  // canonical path; init() will fail with a clear error
        }
        return "/home/bcloud/hrx-slice/hrx-llamacpp/out/llama-hrx-b66/lib/libllama.so";
    }();
    const char* root = std::getenv("HRX_ROOT");
    if (root && root[0]) {
        std::string lib = std::string(root) + "/lib/libllama.so";
        std::string bin = std::string(root) + "/bin/libllama.so";
        std::string cur = access(lib.c_str(), F_OK) == 0 ? lib : bin;
        if (cur != cached) {
            fprintf(stderr,
                    "[hrx] HRX_ROOT changed between instances ('%s' now vs '%s' "
                    "at first load) — the bundle DSO is pinned for the process "
                    "lifetime (this ggml build cannot re-dlopen a different path; "
                    "see issue #1959). Ignoring the new HRX_ROOT.\n",
                    cur.c_str(), cached.c_str());
        }
    }
    return cached;
}

std::string env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::string(v) : std::string(dflt);
}

void* sym(void* handle, const char* name) {
    void* p = dlsym(handle, name);
    if (!p) fprintf(stderr, "[hrx] missing bundle symbol: %s (%s)\n", name, dlerror());
    return p;
}

}  // namespace

Inprocess::Inprocess() : impl_(new Impl()) {}
Inprocess::~Inprocess() { unload(); }

bool Inprocess::has_model() const { return impl_ && impl_->model; }
bool Inprocess::has_hrx_device() const { return impl_ && impl_->hrx_dev; }
const char* Inprocess::hrx_device_name() const {
    return (impl_ && !impl_->hrx_dev_name.empty()) ? impl_->hrx_dev_name.c_str() : "";
}
int Inprocess::vocab_size() const { return impl_ ? impl_->vocab : 0; }
int Inprocess::n_embd() const { return impl_ ? impl_->n_embd : 0; }

bool Inprocess::init() {
    if (impl_->handle) return true;

    std::string lib = bundle_lib_path();
    // RTLD_DEEPBIND: the bundle's llama symbols are unversioned and 1bit
    // statically links its own llama.cpp — DEEPBIND keeps the bundle's
    // internal references resolving to its own copies, not the host's.
    // HRX_NO_DEEPBIND=1 disables it (diagnostic).
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    if (getenv("HRX_NO_DEEPBIND") == nullptr) flags |= RTLD_DEEPBIND;
#endif
    impl_->handle = dlopen(lib.c_str(), flags);
    if (!impl_->handle) {
        fprintf(stderr, "[hrx] dlopen %s failed: %s\n", lib.c_str(), dlerror());
        return false;
    }

    impl_->llama_backend_init = (fn_llama_backend_init)sym(impl_->handle, "llama_backend_init");
    impl_->llama_backend_free = (fn_llama_backend_free)sym(impl_->handle, "llama_backend_free");
    impl_->llama_model_default_params = (fn_llama_model_default_params)sym(impl_->handle, "llama_model_default_params");
    impl_->llama_context_default_params = (fn_llama_context_default_params)sym(impl_->handle, "llama_context_default_params");
    impl_->llama_model_load_from_file = (fn_llama_model_load_from_file)sym(impl_->handle, "llama_model_load_from_file");
    impl_->llama_model_free = (fn_llama_model_free)sym(impl_->handle, "llama_model_free");
    impl_->llama_init_from_model = (fn_llama_init_from_model)sym(impl_->handle, "llama_init_from_model");
    impl_->llama_free = (fn_llama_free)sym(impl_->handle, "llama_free");
    impl_->llama_n_ctx = (fn_llama_n_ctx)sym(impl_->handle, "llama_n_ctx");
    impl_->llama_decode = (fn_llama_decode)sym(impl_->handle, "llama_decode");
    impl_->llama_get_logits_ith = (fn_llama_get_logits_ith)sym(impl_->handle, "llama_get_logits_ith");
    impl_->llama_batch_get_one = (fn_llama_batch_get_one)sym(impl_->handle, "llama_batch_get_one");
    impl_->llama_model_get_vocab = (fn_llama_model_get_vocab)sym(impl_->handle, "llama_model_get_vocab");
    impl_->llama_vocab_n_tokens = (fn_llama_vocab_n_tokens)sym(impl_->handle, "llama_vocab_n_tokens");
    impl_->llama_model_n_embd = (fn_llama_model_n_embd)sym(impl_->handle, "llama_model_n_embd");
    impl_->llama_model_desc = (fn_llama_model_desc)sym(impl_->handle, "llama_model_desc");
    impl_->ggml_backend_dev_by_name = (fn_ggml_backend_dev_by_name)sym(impl_->handle, "ggml_backend_dev_by_name");
    impl_->ggml_backend_dev_name = (fn_ggml_backend_dev_name)sym(impl_->handle, "ggml_backend_dev_name");
    impl_->ggml_backend_hrx_get_device_count = (fn_ggml_backend_hrx_get_device_count)sym(impl_->handle, "ggml_backend_hrx_get_device_count");

    if (!impl_->llama_backend_init || !impl_->llama_model_load_from_file ||
        !impl_->llama_init_from_model || !impl_->llama_decode ||
        !impl_->llama_get_logits_ith || !impl_->llama_batch_get_one) {
        fprintf(stderr, "[hrx] bundle symbol resolution failed\n");
        dlclose(impl_->handle);
        impl_->handle = nullptr;
        return false;
    }

    impl_->llama_backend_init();
    fprintf(stderr, "[hrx] in-process bundle initialized\n");

    if (impl_->ggml_backend_dev_by_name && impl_->ggml_backend_dev_name &&
        env_or("HRX_PIN_DEVICES", "1") != "0") {
        ggml_backend_dev_t d = impl_->ggml_backend_dev_by_name("HRX0");  // device 0 of the HRX backend
        if (d) {
            impl_->hrx_dev = d;
            const char* n = impl_->ggml_backend_dev_name(d);
            impl_->hrx_dev_name = n ? n : "HRX";
            impl_->devs[0] = d;
            impl_->devs[1] = nullptr;
            fprintf(stderr, "[hrx] HRX device found: %s\n", impl_->hrx_dev_name.c_str());
        } else {
            fprintf(stderr, "[hrx] warning: no device named HRX0 — offload will use default device order\n");
        }
    }
    if (impl_->ggml_backend_hrx_get_device_count) {
        fprintf(stderr, "[hrx] HRX backend device count: %d\n", impl_->ggml_backend_hrx_get_device_count());
    }
    return true;
}

bool Inprocess::load_model(const std::string& model_path, int n_gpu_layers, uint32_t ctx_size) {
    if (!impl_->handle) return false;

    llama_model_params mp = impl_->llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;  // <0 = all layers to VRAM
    if (impl_->hrx_dev) {
        mp.devices = impl_->devs;  // NULL-terminated: HRX only
        mp.main_gpu = 0;
    }
    impl_->model = impl_->llama_model_load_from_file(model_path.c_str(), mp);
    if (!impl_->model) {
        fprintf(stderr, "[hrx] model load failed: %s\n", model_path.c_str());
        return false;
    }
    if (impl_->llama_model_desc) {
        char desc[128] = {0};
        impl_->llama_model_desc(impl_->model, desc, sizeof desc - 1);
        fprintf(stderr, "[hrx] model loaded: %s\n", desc);
    }
    if (impl_->llama_model_get_vocab && impl_->llama_vocab_n_tokens)
        impl_->vocab = impl_->llama_vocab_n_tokens(impl_->llama_model_get_vocab(impl_->model));
    if (impl_->llama_model_n_embd)
        impl_->n_embd = impl_->llama_model_n_embd(impl_->model);

    llama_context_params cp = impl_->llama_context_default_params();
    if (ctx_size > 0) cp.n_ctx = ctx_size;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    // Threads for the CPU-side ops (MoE routing, embeddings, argmax). The
    // default (0) may under-thread on big machines; HRX_N_THREADS tunes it.
    int n_threads = std::atoi(env_or("HRX_N_THREADS", "0").c_str());
    if (n_threads > 0) {
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
    }
    impl_->ctx = impl_->llama_init_from_model(impl_->model, cp);
    if (!impl_->ctx) {
        fprintf(stderr, "[hrx] context init failed\n");
        impl_->llama_model_free(impl_->model);
        impl_->model = nullptr;
        return false;
    }
    impl_->ctx_params = cp;
    impl_->pos = 0;
    fprintf(stderr, "[hrx] in-process context ready (n_ctx=%u, vocab=%d, n_embd=%d)\n",
            impl_->llama_n_ctx ? impl_->llama_n_ctx(impl_->ctx) : 0u,
            impl_->vocab, impl_->n_embd);
    return true;
}

int Inprocess::generate(int token_id) {
    if (!impl_->ctx || !impl_->llama_decode) return -1;
    llama_token tok = (llama_token)token_id;
    llama_batch b = impl_->llama_batch_get_one(&tok, 1, impl_->pos, 0);
    impl_->pos++;
    if (impl_->llama_decode(impl_->ctx, b) != 0) {
        fprintf(stderr, "[hrx] llama_decode failed at pos %d\n", (int)impl_->pos - 1);
        return -1;
    }
    float* logits = impl_->llama_get_logits_ith(impl_->ctx, -1);
    if (!logits || impl_->vocab <= 0) return -1;
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < impl_->vocab; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

bool Inprocess::reset() {
    // This fork exports no KV-clear C API — recreate the context (same params).
    if (!impl_->model || !impl_->llama_init_from_model) return false;
    if (impl_->ctx) impl_->llama_free(impl_->ctx);
    impl_->ctx = impl_->llama_init_from_model(impl_->model, impl_->ctx_params);
    impl_->pos = 0;
    return impl_->ctx != nullptr;
}

void Inprocess::unload() {
    if (impl_->ctx) { impl_->llama_free(impl_->ctx); impl_->ctx = nullptr; }
    if (impl_->model) { impl_->llama_model_free(impl_->model); impl_->model = nullptr; }
    if (impl_->handle) {
        if (impl_->llama_backend_free) impl_->llama_backend_free();
        // Deliberately NOT dlclose(): this ggml build's global constructor
        // asserts on a second dlopen after dlclose (GGML_ASSERT(prev !=
        // ggml_uncaught_exception)), so the bundle handle is kept for the
        // process lifetime. A later dlopen of the same path just bumps the
        // refcount and does not re-run static initializers — model switches
        // (destroy + fresh instance) stay safe.
        impl_->handle = nullptr;  // mark released; DSO stays mapped
    }
}

bool probe_bundle(const std::string& bundle_lib_path) {
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    void* h = dlopen(bundle_lib_path.c_str(), flags);
    if (!h) return false;
    auto count = (int32_t (*)(void))dlsym(h, "ggml_backend_hrx_get_device_count");
    if (!count) return false;
    // One-shot diagnostic: leave the handle open (see unload() — dlclose then
    // re-dlopen aborts in ggml's static initializer).
    return count() > 0;
}

}  // namespace hrx
