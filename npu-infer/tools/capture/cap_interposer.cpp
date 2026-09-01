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
