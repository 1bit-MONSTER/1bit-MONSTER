// cap_attn.cpp — focused interposer: dump the attention kernel's act (Q/K/V)
// and kv-cache BO contents at SETARG time, for reverse-engineering the
// attn.xclbin input layout. Build:
//   g++ -O2 -fPIC -shared cap_attn.cpp -o cap_attn.so -ldl -lxrt_coreutil
// Run (lean):
//   LD_PRELOAD=./cap_attn.so CAP_DIR=/tmp/attncap ./run_qwen3_prefill
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "xrt/xrt_bo.h"

static const char* CAP_DIR = getenv("CAP_DIR") ? getenv("CAP_DIR") : "/tmp/attncap";
static FILE* g_log = nullptr;
static long g_seq = 0;

static void ensure_log() {
    if (!g_log) {
        mkdir(CAP_DIR, 0755);
        std::string p = std::string(CAP_DIR) + "/cap_attn.log";
        g_log = fopen(p.c_str(), "w");
        setvbuf(g_log, nullptr, _IONBF, 0);
    }
}

// track per-run (xrt::run) arg BOs: idx -> {ptr, size}
static std::map<unsigned long, std::map<int, std::pair<const void*, size_t>>> g_run_args;

// xrt::run::set_arg_at_index(int, const xrt::bo&)
typedef void (*set_arg_fn)(void*, int, const void*);
static set_arg_fn real_set_arg = nullptr;
extern "C" void _ZN3xrt3run16set_arg_at_indexEiRKNS_2boE(void* self, int idx, const void* bo) {
    if (!real_set_arg) real_set_arg = (set_arg_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE");
    if (real_set_arg) real_set_arg(self, idx, bo);
    try {
        const xrt::bo* b = reinterpret_cast<const xrt::bo*>(bo);
        g_run_args[(unsigned long)self][idx] = {bo, b->size()};
        // Dump act (idx4, 1MB) whenever idx5 is a 32MB kv BO (attention run).
        // The act BO holds Q/K/V written by the QKV GEMM (host-visible after
        // its sync_from_device).
        auto& m = g_run_args[(unsigned long)self];
        if (idx == 5 && b->size() == 33554432 && m.count(4) && m[4].second == 1048576) {
            ensure_log();
            const uint8_t* p = (const uint8_t*)reinterpret_cast<const xrt::bo*>(m[4].first)->map();
            if (p) {
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/attn_act_%04ld.bin", CAP_DIR, g_seq);
                FILE* f = fopen(fn, "wb");
                if (f) { fwrite(p, 1, m[4].second, f); fclose(f); }
                fprintf(g_log, "ATTN_ACT %04ld run=%p act=%p size=%zu -> %s\n", g_seq, self, m[4].first, m[4].second, fn);
            }
            // dump first 1MB of the kv BO
            const uint8_t* kv = (const uint8_t*)b->map();
            if (kv) {
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/attn_kv_%04ld.bin", CAP_DIR, g_seq);
                FILE* f = fopen(fn, "wb");
                size_t n = b->size() < 2097152 ? b->size() : 2097152;
                if (f) { fwrite(kv, 1, n, f); fclose(f); }
                fprintf(g_log, "ATTN_KV  %04ld run=%p kv=%p size=%zu (dumped %zu) -> %s\n", g_seq, self, (void*)bo, b->size(), n, fn);
            }
            g_seq++;
        }
    } catch (...) {}
}

// also dump the output BO (idx3, 1MB) AFTER the attention run: hook run::start
typedef void (*start_fn)(void*);
static start_fn real_start = nullptr;
static std::map<unsigned long, bool> g_attn_runs;
extern "C" void _ZN3xrt3run5startEv(void* self) {
    // pre-start: if this run is an attention run, remember it
    auto it = g_run_args.find((unsigned long)self);
    bool is_attn = false;
    if (it != g_run_args.end()) {
        auto& m = it->second;
        if (m.count(5) && m[5].second == 33554432 && m.count(3) && m[3].second == 1048576)
            is_attn = true;
    }
    if (is_attn) g_attn_runs[(unsigned long)self] = true;
    if (real_start) real_start(self);
    // post-start dump of the output BO (idx3) — the attention result
    if (is_attn && it != g_run_args.end()) {
        auto& m = it->second;
        try {
            const uint8_t* p = (const uint8_t*)reinterpret_cast<const xrt::bo*>(m[3].first)->map();
            if (p) {
                ensure_log();
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/attn_out_%04ld.bin", CAP_DIR, g_seq);
                FILE* f = fopen(fn, "wb");
                if (f) { fwrite(p, 1, m[3].second, f); fclose(f); }
                fprintf(g_log, "ATTN_OUT %04ld run=%p out=%p size=%zu -> %s\n", g_seq, self, m[3].first, m[3].second, fn);
            }
        } catch (...) {}
    }
}
