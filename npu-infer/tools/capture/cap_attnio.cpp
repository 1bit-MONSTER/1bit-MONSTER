// cap_attnio.cpp — capture the attention kernel's INPUT (act=Q) and OUTPUT
// (out) 1MB BOs in one FLM prefill run, so the attention I/O layout can be
// pinned empirically. Dumps act (idx4) at SETARG time and out (idx3) when the
// out BO is later synced FROM device.
// Build: g++ -O2 -fPIC -shared cap_attnio.cpp -o cap_attnio.so -ldl -lxrt_coreutil
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <set>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "xrt/xrt_bo.h"

static const char* CAP_DIR = getenv("CAP_DIR") ? getenv("CAP_DIR") : "/tmp/attnio";
static FILE* g_log = nullptr;
static long g_act_n = 0, g_out_n = 0;

static void ensure_log() {
    if (!g_log) {
        mkdir(CAP_DIR, 0755);
        std::string p = std::string(CAP_DIR) + "/cap_attnio.log";
        g_log = fopen(p.c_str(), "w");
        setvbuf(g_log, nullptr, _IONBF, 0);
    }
}

// per-run args: idx -> {ptr, size}
static std::map<unsigned long, std::map<int, std::pair<const void*, size_t>>> g_args;
// set of attention out-BO pointers (idx3 of attention runs)
static std::set<const void*> g_attn_out_ptrs;
static std::set<const void*> g_attn_act_ptrs;

static void dump_bo(const void* bop, size_t sz, const char* tag) {
    try {
        const xrt::bo* b = reinterpret_cast<const xrt::bo*>(bop);
        const uint8_t* p = (const uint8_t*)b->map();
        if (!p) return;
        ensure_log();
        char fn[256];
        long* n = (tag[0]=='a') ? &g_act_n : &g_out_n;
        snprintf(fn, sizeof(fn), "%s/%s_%04ld.bin", CAP_DIR, tag, *n);
        FILE* f = fopen(fn, "wb");
        if (f) { fwrite(p, 1, sz, f); fclose(f); }
        fprintf(g_log, "%s %ld bo=%p size=%zu -> %s\n", tag, *n, bop, sz, fn);
        (*n)++;
    } catch (...) {}
}

typedef void (*set_arg_fn)(void*, int, const void*);
static set_arg_fn real_set_arg = nullptr;
extern "C" void _ZN3xrt3run16set_arg_at_indexEiRKNS_2boE(void* self, int idx, const void* bo) {
    if (!real_set_arg) real_set_arg = (set_arg_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE");
    if (real_set_arg) real_set_arg(self, idx, bo);
    try {
        const xrt::bo* b = reinterpret_cast<const xrt::bo*>(bo);
        auto& m = g_args[(unsigned long)self];
        m[idx] = {bo, b->size()};
        // attention run: idx5 = 32MB kv
        if (idx == 5 && b->size() == 33554432 && m.count(4) && m[4].second == 1048576) {
            g_attn_act_ptrs.insert(m[4].first);
            if (m.count(3)) g_attn_out_ptrs.insert(m[3].first);
            // dump the act (Q) BO content NOW (it was written by the QKV GEMM,
            // host-visible after its sync_from_device)
            dump_bo(m[4].first, m[4].second, "attn_act");
            // dump the first 256KB of the kv BO (region 0 head data for 256 tokens)
            static long g_kv_n = 0;
            try {
                const uint8_t* p = (const uint8_t*)b->map();
                if (p) {
                    ensure_log();
                    char fn[256];
                    snprintf(fn, sizeof(fn), "%s/attn_kv_%04ld.bin", CAP_DIR, g_kv_n);
                    FILE* f = fopen(fn, "wb");
                    if (f) { fwrite(p, 1, (g_kv_n==0) ? 33554432 : 262144, f); fclose(f); }
                    fprintf(g_log, "attn_kv %ld bo=%p -> %s\n", g_kv_n, (void*)bo, fn);
                    g_kv_n++;
                }
            } catch (...) {}
        }
    } catch (...) {}
}

// hook xrt::bo::sync: when an attention OUT BO is synced FROM device, dump it
typedef void (*sync_fn)(void*, enum xclBOSyncDirection, size_t, size_t);
static sync_fn real_sync = nullptr;
extern "C" void _ZN3xrt2bo4syncE18xclBOSyncDirectionmm(void* self, int dir, size_t size, size_t offset) {
    if (!real_sync)
        real_sync = (sync_fn)dlsym(RTLD_NEXT, "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
    if (real_sync) real_sync(self, (enum xclBOSyncDirection)dir, size, offset);
    if (dir == XCL_BO_SYNC_BO_FROM_DEVICE) {
        try {
            xrt::bo* b = reinterpret_cast<xrt::bo*>(self);
            if (g_attn_out_ptrs.count(self) && b->size() == 1048576) {
                dump_bo(self, b->size(), "attn_out");
            }
        } catch (...) {}
    }
}
