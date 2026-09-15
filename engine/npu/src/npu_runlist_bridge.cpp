// npu_runlist_bridge.cpp — single-launch whole-layer per-ctx ELF decode path.
//
// Bridges npu-infer's RuntimeLayerEngine (the FastFlowLM-validated whole-layer
// decode, byte-identical to the runtime) into npu_engine_universal WITHOUT
// dragging npu-infer's ModelConfig (npu-infer/include/common.h) into the
// engine TU — see the header for the name-clash isolation rationale.
//
// This TU is compiled against the npu-infer include tree only (runtime_layer.h
// + model.h + common.h). The engine links model.c (C) + runtime_layer.cpp
// (C++) + this bridge (C++) into npu_engine_universal.
#include "runtime_layer.h"
#include "model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>    // std::fill (KV re-pack)
#include <chrono>
#include <sys/stat.h>   // struct stat / S_ISDIR for the model-dir probe
#include <unistd.h>     // readlink(/proc/self/exe), symlink, getcwd
#include <dirent.h>     // seeding the ELF cache

#include <xrt/xrt_device.h>

extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, void* mw, const void* config, int layer_idx);
extern "C" void npu_layer_tile_offsets(void* mw, int layer_idx, int* off_q, int* off_k, int* off_v, int* off_o, int* off_gu, int* off_d);
extern "C" void npu_layer_shortconv_offsets(void* mw, int layer_idx, int* off_sp, int* off_so);
extern "C" int npu_layer_bo_bytes(void* mw, const void* config);

// Read whitespace-separated token ids from a file (or stdin for NULL/"-").
static bool read_ids(const char* ids_file, std::vector<int>& ids) {
    if (ids_file && ids_file[0] && strcmp(ids_file, "-") != 0) {
        FILE* f = fopen(ids_file, "r");
        if (!f) { fprintf(stderr, "[runlist] cannot open ids file %s\n", ids_file); return false; }
        int t; while (fscanf(f, "%d", &t) == 1) ids.push_back(t);
        fclose(f);
    } else {
        int t; while (scanf("%d", &t) == 1) ids.push_back(t);
    }
    return !ids.empty();
}

// ===== session API (unified bf16-prefill -> runlist-decode) =====
// The engine runs the fast bf16 prefill, then hands its device KV + final
// hidden to a RuntimeLayerEngine and continues greedy decode via per-ctx ELF
// runlists. One session per process.
static ModelWeights* g_sess_mw = nullptr;
static ModelConfig   g_sess_cfg;
static std::unique_ptr<xrt::device> g_sess_dev;
static std::unique_ptr<RuntimeLayerEngine> g_sess_rt;
static std::string   g_sess_elf_dir;
static int           g_sess_kv_region_u16 = 0;

static void sess_build_cfg(int H, int NC, int NH, int NKV, int IM, int NV) {
    g_sess_cfg = QWEN3_0_6B_CONFIG;
    g_sess_cfg.hidden_size = H;
    g_sess_cfg.num_layers = NC;
    g_sess_cfg.num_attention_heads = NH;
    g_sess_cfg.num_key_value_heads = NKV;
    g_sess_cfg.intermediate_size = IM;
    g_sess_cfg.head_dim = 128;
    g_sess_cfg.vocab_size = NV;
    g_sess_cfg.max_position_embeddings = 40960;
    g_sess_cfg.max_seq_len = 4096;
    // KV region stride matches the layer ELFs' MAX_L=8192 bake: 8MB per
    // region = 8192 tokens x 1024 B. (The 128MB BO holds 4x that headroom.)
    // UNIT: REGION STRIDE, 8 MB = 4,194,304 u16. Four of these are written at stride 8 MB,
    // so the layout occupies [0,8,16,24] MB = exactly [0, 32 MB) — which is exactly the sync
    // length in runtime_layer.cpp's write_kv (33554432). The 128 MB BO allocation is a
    // capacity ceiling on top of that, NOT what the write covers. Three quantities, three
    // units, and two of them mention "32 MB"/"128 MB" in the same file; keep them distinct.
    g_sess_kv_region_u16 = (int)(8u << 20) / 2;
}

static const char* sess_model_dir(int H) {
    return H == 2048 ? "Qwen3-1.7B-NPU2"
         : H == 2560 ? "Qwen3-4B-NPU2"
         : H == 4096 ? "Qwen3-8B-NPU2"
                     : "Qwen3-0.6B-NPU2";
}

static const char* sess_elf_default(int H) {
    return H == 2048 ? "npu-infer/captures/txn-elfs-1p7b"
         : H == 2560 ? "npu-infer/captures/txn-elfs-4b"
         : H == 4096 ? "npu-infer/captures/txn-elfs-8b"
                     : "npu-infer/captures/txn-elfs";
}

static bool npu_dbg_elf() { const char* e = getenv("NPU_ELF_DEBUG"); return e && e[0] && e[0] != '0'; }

static void ensure_elf_gen_env(const char* model_path, int H);   // defined below

extern "C" int npu_runlist_session_init(const char* model_path, int H, int NC, int NH, int NKV, int IM, int NV) {
    ensure_elf_gen_env(model_path, H);
    sess_build_cfg(H, NC, NH, NKV, IM, NV);
    if (!getenv("LAYER_XCLBIN")) {
        std::string xb = std::string("/home/bcloud/amd-oss/fastflowlm/src/xclbins/") + sess_model_dir(H) + "/layer.xclbin";
        setenv("LAYER_XCLBIN", xb.c_str(), 0);
    }
    const char* env_elf = getenv("NPU_LAYER_ELF_DIR");
    g_sess_elf_dir = (env_elf && env_elf[0]) ? env_elf : sess_elf_default(H);
    std::string lmhead_elf = g_sess_elf_dir + "/elf_0002_lmhead.bin";

    g_sess_mw = model_load(model_path, g_sess_cfg);
    if (!g_sess_mw) { fprintf(stderr, "[runlist] model_load failed: %s\n", model_path); return 1; }
    g_sess_dev = std::make_unique<xrt::device>(0);
    g_sess_rt = std::make_unique<RuntimeLayerEngine>();
    if (!g_sess_rt->init(*g_sess_dev, g_sess_mw, g_sess_cfg, g_sess_elf_dir.c_str(), lmhead_elf.c_str())) {
        fprintf(stderr, "[runlist] RuntimeLayerEngine init failed\n");
        return 1;
    }
    return 0;
}

extern "C" int npu_runlist_write_kv(int layer, int token_begin, int n_tokens,
                                    const uint16_t* bf16_kv, int src_region_stride_u16) {
    if (!g_sess_rt) return 1;
    if (!bf16_kv) return 1;
    // The caller's region stride is the bf16 attention ELF's, which is per-shape
    // (8 MB nh16 / 4 MB nh32); this session's and every layer ELF's is 8 MB.
    // Re-pack when they differ instead of reading regions 1..3 at the wrong
    // offsets — that mismatch is what broke nh32 (4B/8B/VL-4B) decode, whose
    // first token after the boot was wrong while nh16 was correct.
    if (src_region_stride_u16 <= 0 || src_region_stride_u16 == g_sess_kv_region_u16)
        return g_sess_rt->write_kv(layer, token_begin, n_tokens, bf16_kv,
                                   g_sess_kv_region_u16) ? 0 : 1;
    const int token_u16 = (g_sess_cfg.num_key_value_heads / 2) * g_sess_cfg.head_dim;
    if (token_u16 <= 0) return 1;
    static std::vector<uint16_t> repack;
    const size_t need = (size_t)g_sess_kv_region_u16 * 4;
    if (repack.size() < need) repack.assign(need, 0);
    else std::fill(repack.begin(), repack.begin() + need, 0);
    for (int r = 0; r < 4; r++)
        memcpy(repack.data() + (size_t)r * g_sess_kv_region_u16 + (size_t)token_begin * token_u16,
               bf16_kv + (size_t)r * src_region_stride_u16 + (size_t)token_begin * token_u16,
               (size_t)n_tokens * token_u16 * sizeof(uint16_t));
    static bool warned = false;
    if (!warned) {
        warned = true;
        fprintf(stderr, "[runlist] KV re-pack: source region stride %d u16 (%d MB) -> session %d u16 (%d MB), token_u16=%d\n",
                src_region_stride_u16, src_region_stride_u16 * 2 >> 20,
                g_sess_kv_region_u16, g_sess_kv_region_u16 * 2 >> 20, token_u16);
    }
    return g_sess_rt->write_kv(layer, token_begin, n_tokens, repack.data(),
                               g_sess_kv_region_u16) ? 0 : 1;
}

extern "C" int npu_runlist_write_act(const uint16_t* bf16_hidden) {
    if (!g_sess_rt) return 1;
    return g_sess_rt->write_act(bf16_hidden) ? 0 : 1;
}

extern "C" int npu_runlist_embed(int token) {
    if (!g_sess_rt) return 1;
    return g_sess_rt->embed(token) ? 0 : 1;
}

extern "C" int npu_runlist_lmhead(float* logits, int vocab) {
    if (!g_sess_rt) return 1;
    if (!g_sess_rt->run_lmhead()) return 1;
    return g_sess_rt->get_logits(logits, vocab) ? 0 : 1;
}

extern "C" int npu_runlist_forward(int ctx_len, float* logits, int vocab) {
    if (!g_sess_rt) return 1;
    if (!g_sess_rt->forward(ctx_len)) return 1;
    return g_sess_rt->get_logits(logits, vocab) ? 0 : 1;
}

extern "C" void npu_runlist_session_free(void) {
    g_sess_rt.reset();
    g_sess_dev.reset();
    if (g_sess_mw) { model_free(g_sess_mw); g_sess_mw = nullptr; }
}

// Point RT_ELF_GEN / RT_ELF_MODEL at a generator and the model dir when the
// caller has not, so a per-context ELF the shipped set does not cover is BUILT
// instead of failing. Without this the shipped sets (ctx 1..2200) make every
// prompt longer than ~2200 tokens abandon the fast paths and land on the
// 112-launch split path at ~2 tok/s — measured, and the single largest cliff in
// the engine: a 2500-token prompt ran at 2 tok/s while the same prompt inside
// the shipped range runs at 57.
//
// The generator is looked up next to the engine binary first (build_npu.sh puts
// it there), then in the source tree, so a normal build needs no environment at
// all. Both stay overridable by the environment, which is checked first.
// Where the SHIPPED per-context ELF sets live, by shape. Duplicated from the
// callers' own default because the cache redirect below has to seed from it.
static const char* shipped_elf_dir(int H) {
    return H == 2048 ? "npu-infer/captures/txn-elfs-1p7b"
         : H == 2560 ? "npu-infer/captures/txn-elfs-4b"
         : H == 4096 ? "npu-infer/captures/txn-elfs-8b"
                     : "npu-infer/captures/txn-elfs";
}

static void mkdir_p(const std::string& d) {
    std::string cur;
    for (size_t i = 0; i <= d.size(); i++) {
        if (i == d.size() || d[i] == '/') {
            if (!cur.empty() && cur != "/") mkdir(cur.c_str(), 0755);
            if (i < d.size()) cur += '/';
        } else cur += d[i];
    }
}

// Seed `dst` with symlinks to every file in `src` (once). The shipped set is the
// baseline; contexts past its end are generated into `dst` as real files, so the
// checkout stays clean and the cache is self-contained.
static void seed_symlinks(const std::string& src, const std::string& dst) {
    DIR* d = opendir(src.c_str());
    if (!d) return;
    char cwd[4096];
    const std::string abs_src = (getcwd(cwd, sizeof cwd) ? std::string(cwd) : std::string()) + "/" + src;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        const std::string to = dst + "/" + n;
        struct stat st;
        if (lstat(to.c_str(), &st) == 0) continue;    // already seeded or generated
        if (symlink((abs_src + "/" + n).c_str(), to.c_str()) != 0) { /* best effort */ }
    }
    closedir(d);
}

static void ensure_elf_gen_env(const char* model_path, int H) {
    if (!model_path || !model_path[0]) return;
    const std::string mp(model_path);
    const size_t slash = mp.rfind('/');
    if (slash != std::string::npos && !getenv("RT_ELF_MODEL"))
        setenv("RT_ELF_MODEL", mp.substr(0, slash).c_str(), 0);
    if (getenv("RT_ELF_GEN")) return;   // explicit wins
    std::vector<std::string> cands;
    {
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (n > 0) {
            exe[n] = 0;
            std::string d(exe);
            const size_t s = d.rfind('/');
            if (s != std::string::npos) cands.push_back(d.substr(0, s) + "/gen_layer_elfs");
        }
    }
    cands.push_back("engine/npu/build/gen_layer_elfs");
    cands.push_back("npu-infer/tools/gen_layer_elfs");
    for (const std::string& c : cands) {
        struct stat st;
        if (stat(c.c_str(), &st) == 0 && S_ISREG(st.st_mode) && (st.st_mode & S_IXUSR)) {
            setenv("RT_ELF_GEN", c.c_str(), 0);
            break;
        }
    }

    // Send GENERATED contexts to a cache directory rather than into the ELF dir
    // the caller would otherwise inherit, which for the shipped sets is
    // npu-infer/captures/txn-elfs* INSIDE THE CHECKOUT. Since the default prompt
    // cap now reaches 8191, a long prompt generates a 256-context window as a
    // matter of course: one session's verification left 1540 untracked .elf/.txn
    // pairs in the working tree. The shipped set is symlinked in as the baseline
    // so a cache hit costs nothing and a miss still generates.
    //
    // Only when the caller has not chosen a directory: NPU_LAYER_ELF_DIR is an
    // explicit instruction and is never second-guessed.
    if (!getenv("NPU_LAYER_ELF_DIR")) {
        std::string root;
        if (const char* x = getenv("XDG_CACHE_HOME"); x && x[0])
            root = std::string(x) + "/1bit-monster/elfs";
        else if (const char* h = getenv("HOME"); h && h[0])
            root = std::string(h) + "/.cache/1bit-monster/elfs";
        if (!root.empty()) {
            const std::string mdir = getenv("RT_ELF_MODEL") ? getenv("RT_ELF_MODEL") : "";
            const size_t sl = mdir.rfind('/');
            const std::string base = sl == std::string::npos ? mdir : mdir.substr(sl + 1);
            if (!base.empty()) {
                const std::string dir = root + "/" + base;
                mkdir_p(dir);
                const char* ship = shipped_elf_dir(H);
                struct stat st;
                if (stat(ship, &st) == 0 && S_ISDIR(st.st_mode)) seed_symlinks(ship, dir);
                setenv("NPU_LAYER_ELF_DIR", dir.c_str(), 0);
                if (npu_dbg_elf()) fprintf(stderr, "[runlist] generated ELFs go to %s (shipped set symlinked in)\n", dir.c_str());
            }
        }
    }
}

extern "C" int npu_runlist_decode(const char* model_path, int ng, const char* ids_file,
                               int H, int NC, int NH, int NKV, int IM, int NV) {
    ensure_elf_gen_env(model_path, H);
    // 1) prompt token ids (the engine feeds pre-tokenized ids; no tokenizer here)
    std::vector<int> ids;
    if (!read_ids(ids_file, ids)) { fprintf(stderr, "[runlist] no prompt tokens\n"); return 1; }

    // 2) per-model config (dims from the caller; whole-layer packing is dim-driven)
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    cfg.hidden_size = H;
    cfg.num_layers = NC;
    cfg.num_attention_heads = NH;
    cfg.num_key_value_heads = NKV;
    cfg.intermediate_size = IM;
    cfg.head_dim = 128;
    cfg.vocab_size = NV;
    cfg.max_position_embeddings = 40960;
    cfg.max_seq_len = 4096;

    // 3) per-model layer.xclbin + per-ctx ELF dir (env overrides first)
    //
    // Derive the model dir from the MODEL PATH's basename first, as the bf16 prefill path does
    // (commit 77874d5a7). The H table below is only a fallback, and it maps H=2560 to
    // Qwen3-4B-NPU2 and H=4096 to Qwen3-8B-NPU2 -- so any non-Qwen3 model at those sizes
    // (Nanbeige H=2560, Llama-3.1-8B H=4096) silently ran on Qwen3's layer.xclbin and Qwen3's
    // per-context ELFs. That is the same bug class 77874d5a7 fixed for the prefill, and it is
    // exactly what would have made a Nanbeige reference measurement meaningless.
    static std::string mdir_own;
    const char* mdir = H == 2048 ? "Qwen3-1.7B-NPU2"
                     : H == 2560 ? "Qwen3-4B-NPU2"
                     : H == 4096 ? "Qwen3-8B-NPU2"
                                 : "Qwen3-0.6B-NPU2";
    {
        const std::string mpath(model_path);
        const size_t slash = mpath.rfind('/');
        const std::string mdir_s = (slash != std::string::npos) ? mpath.substr(0, slash) : std::string();
        const size_t slash2 = mdir_s.rfind('/');
        const std::string base = (slash2 != std::string::npos) ? mdir_s.substr(slash2 + 1) : mdir_s;
        struct stat st;
        const std::string probe = std::string("/home/bcloud/amd-oss/fastflowlm/src/xclbins/") + base;
        if (!base.empty() && stat(probe.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            mdir_own = base;
            mdir = mdir_own.c_str();
        }
    }
    const char* elf_default = H == 2048 ? "npu-infer/captures/txn-elfs-1p7b"
                            : H == 2560 ? "npu-infer/captures/txn-elfs-4b"
                            : H == 4096 ? "npu-infer/captures/txn-elfs-8b"
                                        : "npu-infer/captures/txn-elfs";
    if (!getenv("LAYER_XCLBIN")) {
        std::string xb = std::string("/home/bcloud/amd-oss/fastflowlm/src/xclbins/") + mdir + "/layer.xclbin";
        setenv("LAYER_XCLBIN", xb.c_str(), 0);
    }
    const char* env_elf = getenv("NPU_LAYER_ELF_DIR");
    std::string elf_dir = env_elf && env_elf[0] ? env_elf : elf_default;
    std::string lmhead_elf = elf_dir + "/elf_0002_lmhead.bin";

    // 4) model weights via npu-infer's model.c loader
    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) { fprintf(stderr, "[runlist] model_load failed: %s\n", model_path); return 1; }

    // 5) device + RuntimeLayerEngine (packs per-layer weight BOs + lm_head BO)
    xrt::device dev(0);
    RuntimeLayerEngine rt;
    if (!rt.init(dev, mw, cfg, elf_dir.c_str(), lmhead_elf.c_str())) {
        fprintf(stderr, "[runlist] RuntimeLayerEngine init failed\n");
        model_free(mw);
        return 1;
    }

    // 5) prefill — one whole-layer forward per prompt token (KV accumulates on
    //    device; the per-ctx ELF is regenerated/reused per context length).
    int npt = (int)ids.size();
    printf("=== Prefill %d [runlist] ===\n", npt); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    int ctx = 0;
    for (int t : ids) {
        if (!rt.embed(t) || !rt.forward(++ctx)) {
            fprintf(stderr, "[runlist] prefill forward ctx=%d failed\n", ctx);
            model_free(mw);
            return 1;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n", prefill_ms, prefill_ms / npt);

    // 6) greedy decode — bf16 argmax -> emit -> advance. The runlist build for
    //    the NEXT step is overlapped against the current device exec using the
    //    double-buffered runlist slots (see RuntimeLayerEngine::build_runlist).
    auto tgs = std::chrono::steady_clock::now();
    int total = 0;
    int sa = 0, sb = 1;
    // Prime: emit token 1 (logits already ready from the prefill's last forward)
    // and launch the first decode forward (ctx = npt+1) on slot sb.
    {
        // Parity instrumentation: the runlist path's logits come from the device
        // (bf16 -> fp32 via RuntimeLayerEngine::get_logits) and are NOT the logits
        // the dense arm's NPU_DUMP_LOGITS hook writes (that hook lives in lm_topk_omp,
        // which only the dense path reaches). Dump them here so the two arms' logits
        // can actually be compared instead of comparing a stale dense file twice.
        if (getenv("NPU_DUMP_LOGITS")) {
            std::vector<float> lg((size_t)cfg.vocab_size);
            if (rt.get_logits(lg.data(), cfg.vocab_size)) {
                FILE* fl = fopen("/tmp/runlist_logits.txt", "wb");
                if (fl) {
                    for (int n = 0; n < cfg.vocab_size && n < 4096; n++)
                        fprintf(fl, "%d %.6g\n", n, lg[(size_t)n]);
                    fclose(fl);
                }
            }
        }
        int best = rt.argmax_logits(cfg.vocab_size);
        printf("  [%d] %d\n", 1, best);
        total++;
        if (ng > 1) {
            int c1 = ctx + 1;
            rt.apply_rope(c1);
            if (!rt.build_runlist(sb, c1) || !rt.embed(best) || !rt.execute_runlist(sb)) {
                fprintf(stderr, "[runlist] decode forward ctx=%d failed\n", c1);
                model_free(mw);
                return 1;
            }
            ctx = c1;
        }
    }
    for (int i = 1; i < ng; i++) {
        int next_ctx = ctx + 1;
        // Build the NEXT runlist (pure host) while slot sb executes on-device.
        if (!rt.build_runlist(sa, next_ctx)) {
            fprintf(stderr, "[runlist] build ctx=%d failed\n", next_ctx);
            model_free(mw);
            return 1;
        }
        if (!rt.wait_runlist(sb)) {
            fprintf(stderr, "[runlist] wait ctx=%d failed\n", ctx);
            model_free(mw);
            return 1;
        }
        int best = rt.argmax_logits(cfg.vocab_size);
        printf("  [%d] %d\n", i + 1, best);
        total++;
        if (i + 1 < ng) {
            rt.apply_rope(next_ctx);
            if (!rt.embed(best) || !rt.execute_runlist(sa)) {
                fprintf(stderr, "[runlist] decode forward ctx=%d failed\n", next_ctx);
                model_free(mw);
                return 1;
            }
            ctx = next_ctx;
        }
        std::swap(sa, sb);
    }
    auto tge = std::chrono::steady_clock::now();
    double tts = std::chrono::duration<double>(tge - tgs).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) | tokens=%d ===\n",
           tts * 1000.0 / ng, ng / tts, total);

    model_free(mw);
    return 0;
}

// ===== bf16 prefill (mm.xclbin dequant + GEMM) support =====
// The engine drives the dequant/GEMM bridge (npu_engine_bf16_mm_bridge) for the
// prefill mm path; this TU packs the per-layer Q4NX weight BOs + tile offsets.
static ModelWeights* g_bf16_mw = nullptr;
static ModelConfig  g_bf16_cfg;

extern "C" int npu_bf16_prefill_init(const char* model_path, int H, int NC, int NH, int NKV, int IM, int NV, int HD) {
    g_bf16_cfg = QWEN3_0_6B_CONFIG;
    g_bf16_cfg.hidden_size = H;
    g_bf16_cfg.num_layers = NC;
    g_bf16_cfg.num_attention_heads = NH;
    g_bf16_cfg.num_key_value_heads = NKV;
    g_bf16_cfg.intermediate_size = IM;
    // head_dim is NOT always 128 (LFM2 and Llama-3.2 use 64, Gemma3/Qwen3.5 use 256);
    // taking it from the caller keeps qout = NH*HD correct for those families.
    g_bf16_cfg.head_dim = (HD > 0) ? HD : 128;
    g_bf16_cfg.vocab_size = NV;
    g_bf16_cfg.max_position_embeddings = 40960;
    g_bf16_cfg.max_seq_len = 4096;
    g_bf16_mw = model_load(model_path, g_bf16_cfg);
    if (g_bf16_mw) {
        fprintf(stderr, "[bf16prefill] loaded %s emb=%lldx%lld layers=%d\n", model_path,
                (long long)g_bf16_mw->embed_tokens.shape[0], (long long)g_bf16_mw->embed_tokens.shape[1],
                g_bf16_mw->config.num_layers);
    }
    return g_bf16_mw ? 0 : -1;
}

// Pack layer `layer`'s weight BO into bo (>= npu_bf16_layer_bo_bytes() bytes).
// Returns tiles; fills offs[6] = {q,k,v,o,gu,d} tile offsets (for bf16mm_dequant woff = tile*5120).
extern "C" int npu_bf16_pack_layer(int layer, uint8_t* bo, int* offs) {
    if (!g_bf16_mw) return 0;
    int tiles = npu_pack_layer_bo(bo, g_bf16_mw, &g_bf16_cfg, layer);
    npu_layer_tile_offsets(g_bf16_mw, layer, &offs[0], &offs[1], &offs[2], &offs[3], &offs[4], &offs[5]);
    return tiles;
}

extern "C" int npu_bf16_layer_bo_bytes(void) {
    if (!g_bf16_mw) return 0;
    return npu_layer_bo_bytes(g_bf16_mw, &g_bf16_cfg);
}
