// cap_interposer.cpp — LD_PRELOAD interposer on xrtBOSync to capture the real
// FastFlowLM runtime's BO traffic: the instruction TXNs uploaded for each
// kernel (small BO_TO syncs) and the weight/activation BOs read back
// (BO_FROM syncs). This is the capture the runtime layer-TXN weight-BD decode
// (#2006/#2015) needs: the runtime's ACTUAL dequant TXNs + weight layout.
//
// Build:
//   g++ -O2 -fPIC -shared cap_interposer.cpp -o cap_interposer.so -ldl -lxrt_coreutil
// Run:
//   LD_PRELOAD=/tmp/txn_decode/cap_interposer.so ./run_qwen3_npu ...
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "xrt/xrt_bo.h"
extern "C" {
#include <xrt.h>
}

static const char* CAP_DIR = getenv("CAP_DIR") ? getenv("CAP_DIR") : "/tmp/cap2";
static FILE* g_log = nullptr;
#include <set>
#include <vector>
static std::set<std::pair<unsigned long, size_t>> g_bo_sizes;
static long g_seq = 0;
static std::map<unsigned long, std::string> g_bo_labels;

static void ensure_log() {
    if (!g_log) {
        mkdir(CAP_DIR, 0755);
        std::string p = std::string(CAP_DIR) + "/capture_manifest.log";
        g_log = fopen(p.c_str(), "w");
        setvbuf(g_log, nullptr, _IONBF, 0);
    }
}

// map BO memory
static void* bo_map_cached(xrtBufferHandle bhdl) {
    static std::map<unsigned long, void*> cache;
    unsigned long key = (unsigned long)bhdl;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    void* p = xrtBOMap(bhdl);
    if (p) cache[key] = p;
    return p;
}

static void dump_bo(xrtBufferHandle bhdl, size_t size, size_t offset, int dir, size_t claimed) {
    ensure_log();
    void* p = bo_map_cached(bhdl);
    size_t bosz = xrtBOSize(bhdl);
    if (!p) { fprintf(g_log, "CAP %04ld: size=%zu dir=%d (map failed)\n", g_seq, bosz, dir); return; }
    char fname[256];
    const char* dn = (dir == XCL_BO_SYNC_BO_TO_DEVICE) ? "to" : "from";
    snprintf(fname, sizeof(fname), "%s/bo_%s_%04ld_%zu.bin", CAP_DIR, dn, g_seq, bosz);
    FILE* f = fopen(fname, "wb");
    if (f) {
        fwrite(p, 1, bosz, f);
        fclose(f);
    }
    fprintf(g_log, "CAP %04ld: %s size=%zu offset=%zu synced=%zu -> %s\n",
            g_seq, dn, bosz, offset, claimed, fname);
    g_seq++;
}

// The runtime calls the C++ method xrt::bo::sync (defined in libxrt_coreutil).
// Interpose on its mangled symbol: _ZN3xrt2bo4syncE18xclBOSyncDirectionmm
typedef void (*xrt_bo_sync_fn)(void*, enum xclBOSyncDirection, size_t, size_t);
static xrt_bo_sync_fn real_sync = nullptr;

extern "C" void _ZN3xrt2bo4syncE18xclBOSyncDirectionmm(void* self, int dir,
                                                       size_t size, size_t offset) {
    if (!real_sync)
        real_sync = (xrt_bo_sync_fn)dlsym(RTLD_NEXT,
            "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
    if (real_sync) real_sync(self, (enum xclBOSyncDirection)dir, size, offset);
    // capture: the buffer handle is xrt::bo::get() at vtable+0x? — use the
    // xrt::bo public API through a reinterpreted object.
    try {
        xrt::bo* bo = reinterpret_cast<xrt::bo*>(self);
        size_t bosz = bo->size();
        g_bo_sizes.insert({(unsigned long)self, bosz});
        bool capture = true;  // capture ALL BO syncs (TXN insts + weight + act + kv)
        if (capture) {
            const uint8_t* p = (const uint8_t*)bo->map();
            ensure_log();
            char fname[256];
            const char* dn = (dir == XCL_BO_SYNC_BO_TO_DEVICE) ? "to" : "from";
            snprintf(fname, sizeof(fname), "%s/bo_%s_%04ld_%zu.bin", CAP_DIR, dn, g_seq, bosz);
            FILE* f = fopen(fname, "wb");
            if (f) { fwrite(p, 1, bosz, f); fclose(f); }
            fprintf(g_log, "CAP %04ld: %s size=%zu offset=%zu synced=%zu -> %s\n",
                    g_seq, dn, bosz, offset, size, fname);
            g_seq++;
        }
    } catch (...) {}
}

// ===== kernel-call capture: xrt::run::set_arg_at_index + start =====
#include <map>
#include <vector>
static std::map<unsigned long, std::vector<std::pair<int, size_t>>> g_run_args; // run -> (arg_idx, bo size)
static std::map<unsigned long, int> g_run_count;

// void xrt::run::set_arg_at_index(int idx, const xrt::bo&)
typedef void (*set_arg_fn)(void*, int, const void*);
static set_arg_fn real_set_arg = nullptr;
extern "C" void _ZN3xrt3run16set_arg_at_indexEiRKNS_2boE(void* self, int idx, const void* bo) {
    if (!real_set_arg) real_set_arg = (set_arg_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE");
    if (real_set_arg) real_set_arg(self, idx, bo);
    try {
        const xrt::bo* b = reinterpret_cast<const xrt::bo*>(bo);
        g_run_args[(unsigned long)self].push_back({idx, b->size()});
        ensure_log();
        fprintf(g_log, "SETARG %p idx=%d size=%zu bo=%p\n", self, idx, b->size(), (void*)bo);
    } catch (...) {}
}
// void xrt::run::run(const xrt::kernel&)
typedef void (*run_ctor_fn)(void*, const void*);
static run_ctor_fn real_run_ctor = nullptr;
extern "C" void _ZN3xrt3runC1ERKNS_6kernelE(void* self, const void* kern) {
    if (!real_run_ctor) real_run_ctor = (run_ctor_fn)dlsym(RTLD_NEXT, "_ZN3xrt3runC1ERKNS_6kernelE");
    if (real_run_ctor) real_run_ctor(self, kern);
    ensure_log();
    fprintf(g_log, "RUN_CTOR %p\n", self);
}

// void xrt::run::start()
typedef void (*start_fn)(void*);
static start_fn real_start = nullptr;
extern "C" void _ZN3xrt3run5startEv(void* self) {
    if (!real_start) real_start = (start_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run5startEv");
    if (real_start) real_start(self);
    ensure_log();
    int n = ++g_run_count[(unsigned long)self];
    fprintf(g_log, "RUN %03d: args=[", n);
    for (auto& kv : g_run_args[(unsigned long)self])
        fprintf(g_log, "%d:%zu ", kv.first, kv.second);
    fprintf(g_log, "]\n");
}

// ===== runlist::execute hook (per-forward TXN submissions) + post-exec BO dump =====
static long g_runlist_n = 0;
typedef void (*rl_exec_fn)(void*);
static rl_exec_fn real_rl_exec = nullptr;
extern "C" void _ZN3xrt7runlist7executeEv(void* self) {
    if (!real_rl_exec)
        real_rl_exec = (rl_exec_fn)dlsym(RTLD_NEXT, "_ZN3xrt7runlist7executeEv");
    if (real_rl_exec) real_rl_exec(self);
    ensure_log();
    g_runlist_n++;
    fprintf(g_log, "RUNLIST %ld: execute\n", g_runlist_n);
    int n = 0;
    for (auto& kv : g_bo_sizes) {
        if (kv.second < 1000000) continue;
        try {
            xrt::bo* bo = reinterpret_cast<xrt::bo*>(kv.first);
            size_t bosz = bo->size();
            const uint8_t* p = (const uint8_t*)bo->map();
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/post_%03ld_%02d_%zx_%zu.bin", CAP_DIR, g_runlist_n, n, (size_t)kv.first, bosz);
            FILE* f = fopen(fname, "wb");
            if (f) { fwrite(p, 1, bosz, f); fclose(f); }
            n++;
        } catch (...) {}
    }
    fprintf(g_log, "RUNLIST %ld: dumped %d big BOs\n", g_runlist_n, n);
}
