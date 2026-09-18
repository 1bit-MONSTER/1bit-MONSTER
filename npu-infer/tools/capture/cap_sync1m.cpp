// cap_sync1m.cpp — dump every 1MB BO sync (TO and FROM device) so we can see
// the Q/K/V GEMM outputs that feed the attention kernel. Lean: only 1MB BOs.
// Build: g++ -O2 -fPIC -shared cap_sync1m.cpp -o cap_sync1m.so -ldl -lxrt_coreutil
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "xrt/xrt_bo.h"

static const char* CAP_DIR = getenv("CAP_DIR") ? getenv("CAP_DIR") : "/tmp/sync1m";
static FILE* g_log = nullptr;
static long g_seq = 0;
static long g_to = 0, g_from = 0;

static void ensure_log() {
    if (!g_log) {
        mkdir(CAP_DIR, 0755);
        std::string p = std::string(CAP_DIR) + "/cap_sync1m.log";
        g_log = fopen(p.c_str(), "w");
        setvbuf(g_log, nullptr, _IONBF, 0);
    }
}

typedef void (*sync_fn)(void*, enum xclBOSyncDirection, size_t, size_t);
static sync_fn real_sync = nullptr;
extern "C" void _ZN3xrt2bo4syncE18xclBOSyncDirectionmm(void* self, int dir,
                                                       size_t size, size_t offset) {
    if (!real_sync)
        real_sync = (sync_fn)dlsym(RTLD_NEXT, "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
    if (real_sync) real_sync(self, (enum xclBOSyncDirection)dir, size, offset);
    try {
        xrt::bo* bo = reinterpret_cast<xrt::bo*>(self);
        size_t bosz = bo->size();
        if (bosz == 1048576) {
            const uint8_t* p = (const uint8_t*)bo->map();
            ensure_log();
            char fn[256];
            const char* dn = (dir == XCL_BO_SYNC_BO_TO_DEVICE) ? "to" : "from";
            snprintf(fn, sizeof(fn), "%s/bo_%s_%04ld.bin", CAP_DIR, dn, g_seq);
            FILE* f = fopen(fn, "wb");
            if (f) { fwrite(p, 1, bosz, f); fclose(f); }
            long n = (dir == XCL_BO_SYNC_BO_TO_DEVICE) ? ++g_to : ++g_from;
            fprintf(g_log, "%s %ld bo=%p off=%zu synced=%zu -> %s\n", dn, n, self, offset, size, fn);
            g_seq++;
        }
    } catch (...) {}
}
