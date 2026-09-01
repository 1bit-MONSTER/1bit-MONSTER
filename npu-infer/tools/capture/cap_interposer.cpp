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
static std::set<std::pair<unsigned long, size_t>> g_extbo_sizes;
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
static std::map<unsigned long, std::map<int, const void*>> g_run_bo_ptrs;  // run -> arg_idx -> bo ptr

// void xrt::run::set_arg_at_index(int idx, const xrt::bo&)
typedef void (*set_arg_fn)(void*, int, const void*);
static set_arg_fn real_set_arg = nullptr;
extern "C" void _ZN3xrt3run16set_arg_at_indexEiRKNS_2boE(void* self, int idx, const void* bo) {
    if (!real_set_arg) real_set_arg = (set_arg_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE");
    if (real_set_arg) real_set_arg(self, idx, bo);
    try {
        const xrt::bo* b = reinterpret_cast<const xrt::bo*>(bo);
        g_run_args[(unsigned long)self].push_back({idx, b->size()});
        g_run_bo_ptrs[(unsigned long)self][idx] = bo;
        ensure_log();
        fprintf(g_log, "SETARG %p idx=%d size=%zu bo=%p\n", self, idx, b->size(), (void*)bo);
        // dump the idx3 BO (the runtime's insts BO per create_run: (3,0,0,insts,weight))
        if (idx == 3 && b->size() <= 2000000) {
            const uint8_t* pm = (const uint8_t*)b->map();
            char fn[256];
            snprintf(fn, sizeof(fn), "%s/insts_%04ld_%zu.bin", CAP_DIR, g_seq, b->size());
            FILE* ff = fopen(fn, "wb");
            if (ff) { fwrite(pm, 1, b->size(), ff); fclose(ff); }
            fprintf(g_log, "INSTS_DUMP -> %s\n", fn);
        }
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
    // post-run dump of the act BO (idx3) — the layer's output written in-place
    if (getenv("CAP_POSTRUN_ACT")) {
        auto it = g_run_bo_ptrs.find((unsigned long)self);
        if (it != g_run_bo_ptrs.end()) {
            auto a3 = it->second.find(3);
            if (a3 != it->second.end()) {
                try {
                    xrt::bo* bo = reinterpret_cast<xrt::bo*>(const_cast<void*>(a3->second));
                    const uint8_t* p = (const uint8_t*)bo->map();
                    if (p) {
                        char fname[256];
                        snprintf(fname, sizeof(fname), "%s/postrun_act_%03d_%zx.bin", CAP_DIR, n, (size_t)a3->second);
                        FILE* f = fopen(fname, "wb");
                        if (f) { fwrite(p, 1, bo->size(), f); fclose(f); }
                        fprintf(g_log, "POSTRUN_ACT -> %s\n", fname);
                    }
                } catch (...) {}
            }
        }
    }
}

// ===== runlist::execute hook (per-forward TXN submissions) + post-exec BO dump =====
static long g_runlist_n = 0;
typedef void (*rl_exec_fn)(void*);
static rl_exec_fn real_rl_exec = nullptr;
extern "C" void _ZN3xrt7runlist7executeEv(void* self) {
    if (!real_rl_exec)
        real_rl_exec = (rl_exec_fn)dlsym(RTLD_NEXT, "_ZN3xrt7runlist7executeEv");
    // PRE-exec dump: the per-call TXNs are written into the bo0/idx3 buffer
    // AFTER set_arg and BEFORE execute (coherent, no sync) — this is the only
    // moment the runtime's actual per-call insts are observable.
    {
        ensure_log();
        g_runlist_n++;
        fprintf(g_log, "RUNLIST %ld: execute (pre-dump)\n", g_runlist_n);
        int n = 0;
        std::set<unsigned long> done;
        for (auto& kv : g_run_bo_ptrs) {
            unsigned long runkey = kv.first;
            if (done.count(runkey)) continue;
            done.insert(runkey);
            for (auto& ab : kv.second) {
                int aidx = ab.first;
                const void* bop = ab.second;
                if (bop == nullptr) continue;
                try {
                    xrt::bo* bo = reinterpret_cast<xrt::bo*>(const_cast<void*>(bop));
                    size_t bosz = bo->size();
                    if (bosz > 3000000) continue;   // skip weight/kv BOs
                    const uint8_t* p = (const uint8_t*)bo->map();
                    if (p) {
                        char fname[256];
                        snprintf(fname, sizeof(fname), "%s/preinsts_%03ld_%02d_i%d_%zx_%zu.bin", CAP_DIR, g_runlist_n, n, aidx, (size_t)bop, bosz);
                        FILE* f = fopen(fname, "wb");
                        if (f) { fwrite(p, 1, bosz, f); fclose(f); }
                        fprintf(g_log, "PREINSTS run=%p arg=%d bo=%p size=%zu -> %s\n", (void*)runkey, aidx, bop, bosz, fname);
                        n++;
                    }
                } catch (...) {}
            }
        }
        fprintf(g_log, "RUNLIST %ld: pre-dumped %d insts BOs\n", g_runlist_n, n);
    }
    if (real_rl_exec) real_rl_exec(self);
    ensure_log();
    g_runlist_n++;
    fprintf(g_log, "RUNLIST %ld: execute\n", g_runlist_n);
    int n = 0;
    // ext::bo objects (the runtime's data/insts BOs) — dump the small ones
    for (auto& kv : g_extbo_sizes) {
        if (kv.second > 2000000) continue;
        try {
            const uint8_t* pm = (const uint8_t*)reinterpret_cast<xrt::bo*>(kv.first)->map();
            if (pm) {
                char fname[256];
                snprintf(fname, sizeof(fname), "%s/extsmall_%03ld_%02d_%zx_%zu.bin", CAP_DIR, g_runlist_n, n, (size_t)kv.first, kv.second);
                FILE* f = fopen(fname, "wb");
                if (f) { fwrite(pm, 1, kv.second, f); fclose(f); }
                n++;
            }
        } catch (...) {}
    }
    for (auto& kv : g_bo_sizes) {
        // dump the small BOs too (the per-call instr TXNs are written via
        // coherent map with no sync — their BOs are small)
        if (kv.second < 1000000 && kv.second > 512) {
            try {
                xrt::bo* bo = reinterpret_cast<xrt::bo*>(kv.first);
                size_t bosz = bo->size();
                const uint8_t* p = (const uint8_t*)bo->map();
                char fname[256];
                snprintf(fname, sizeof(fname), "%s/small_%03ld_%02d_%zx_%zu.bin", CAP_DIR, g_runlist_n, n, (size_t)kv.first, bosz);
                FILE* f = fopen(fname, "wb");
                if (f) { fwrite(p, 1, bosz, f); fclose(f); }
                n++;
            } catch (...) {}
            continue;
        }
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

// void xrt::run::set_arg_at_index(int, const void*) — scalar args (opcode/ninstr)
typedef void (*set_arg_v_fn)(void*, int, const void*);
static set_arg_v_fn real_set_arg_v = nullptr;
extern "C" void _ZN3xrt3run16set_arg_at_indexEiPKv(void* self, int idx, const void* val) {
    if (!real_set_arg_v) real_set_arg_v = (set_arg_v_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run16set_arg_at_indexEiPKv");
    if (real_set_arg_v) real_set_arg_v(self, idx, val);
    ensure_log();
    fprintf(g_log, "SETARGV %p idx=%d val=%p\n", self, idx, val);
}

// xrt::ext::bo::bo(const xrt::device&, size_t) — the runtime creates ALL its
// BOs through this (including the per-call instr TXN BOs, never synced).
typedef void (*extbo_fn)(void*, const void*, size_t);
static extbo_fn real_extbo = nullptr;
extern "C" void _ZN3xrt3ext2boC1ERKNS_6deviceEm(void* self, const void* dev, size_t size) {
    if (!real_extbo) real_extbo = (extbo_fn)dlsym(RTLD_NEXT, "_ZN3xrt3ext2boC1ERKNS_6deviceEm");
    if (real_extbo) real_extbo(self, dev, size);
    ensure_log();
    g_extbo_sizes.insert({(unsigned long)self, size});
    fprintf(g_log, "EXTBO %p size=%zu\n", self, size);
    if (size <= 2000000) {
        // ext::bo has its own map/size: use the C API on its handle
        // xrt::ext::bo -> handle via get()? use xrtBOAddress on the first member
        try {
            const uint8_t* pm = (const uint8_t*)xrtBOMap((xrtBufferHandle)self);
            if (pm) {
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/extbo_%04ld_%zu.bin", CAP_DIR, g_seq, size);
                FILE* ff = fopen(fn, "wb");
                if (ff) { fwrite(pm, 1, size, ff); fclose(ff); }
                fprintf(g_log, "EXTBO_DUMP size=%zu -> %s\n", size, fn);
            }
        } catch (...) {}
    }
}

// void xrt::run::set_arg_at_index(int, const void*, size_t) — scalars and raw pointers
typedef void (*set_arg3_fn)(void*, int, const void*, size_t);
static set_arg3_fn real_set_arg3 = nullptr;
extern "C" void _ZN3xrt3run16set_arg_at_indexEiPKvm(void* self, int idx, const void* val, size_t bytes) {
    if (!real_set_arg3) real_set_arg3 = (set_arg3_fn)dlsym(RTLD_NEXT, "_ZN3xrt3run16set_arg_at_indexEiPKvm");
    if (real_set_arg3) real_set_arg3(self, idx, val, bytes);
    ensure_log();
    uint64_t v = 0;
    if (bytes >= 1 && bytes <= 8) memcpy(&v, val, bytes);
    fprintf(g_log, "SETARG3 %p idx=%d bytes=%zu val=0x%llx\n", self, idx, bytes, (unsigned long long)v);
}

// xrt::ext::kernel ctor (hw_context, module, name) — the runtime creates its
// kernels here; the module may carry the instruction control code.
typedef void (*extk_fn)(void*, const void*, const void*, const void*);
static extk_fn real_extk = nullptr;
extern "C" void _ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNS_6moduleERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void* self, const void* hw, const void* mod, const void* name) {
    if (!real_extk) real_extk = (extk_fn)dlsym(RTLD_NEXT, "_ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNS_6moduleERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    if (real_extk) real_extk(self, hw, mod, name);
    ensure_log();
    fprintf(g_log, "EXTKERNEL %p\n", self);
}

// ===== THE KEY HOOK: xrt::elf ctor =====
// The runtime embeds the per-call TXNs in an ELF (from ctrl_seq->dump())
// and creates xrt::elf -> xrt::module -> xrt::ext::kernel. The ELF buffer
// IS the runtime's actual per-call instruction stream (TXN + aiebu header).
// void xrt::elf::elf(const char* buf, size_t size)
typedef void (*elf_fn)(void*, const void*, size_t);
static elf_fn real_elf = nullptr;
static long g_elf_n = 0;
extern "C" void _ZN3xrt3elfC1EPKvm(void* self, const void* buf, size_t size) {
    if (!real_elf) real_elf = (elf_fn)dlsym(RTLD_NEXT, "_ZN3xrt3elfC1EPKvm");
    if (real_elf) real_elf(self, buf, size);
    ensure_log();
    g_elf_n++;
    char fname[256];
    snprintf(fname, sizeof(fname), "%s/elf_%04ld_%zu.bin", CAP_DIR, g_elf_n, size);
    FILE* f = fopen(fname, "wb");
    if (f) { fwrite(buf, 1, size, f); fclose(f); }
    fprintf(g_log, "ELF %04ld: size=%zu -> %s\n", g_elf_n, size, fname);
    // also try the 2-arg form symbol in case it's used instead
}
extern "C" void _ZN3xrt3elfC2EPKvm(void* self, const void* buf, size_t size) {
    if (!real_elf) real_elf = (elf_fn)dlsym(RTLD_NEXT, "_ZN3xrt3elfC1EPKvm");
    if (real_elf) real_elf(self, buf, size);
    ensure_log();
    g_elf_n++;
    char fname[256];
    snprintf(fname, sizeof(fname), "%s/elf_%04ld_%zu.bin", CAP_DIR, g_elf_n, size);
    FILE* f = fopen(fname, "wb");
    if (f) { fwrite(buf, 1, size, f); fclose(f); }
    fprintf(g_log, "ELF %04ld: size=%zu -> %s\n", g_elf_n, size, fname);
}
// void xrt::elf::elf(const std::string& path) — load_elf from a FILE (the
// mm/dequant/mha kernels are loaded this way from the xclbin's stored ELFs).
typedef void (*elf_str_fn)(void*, const void*);
static elf_str_fn real_elf_str = nullptr;
extern "C" void _ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void* self, const void* path) {
    if (!real_elf_str) real_elf_str = (elf_str_fn)dlsym(RTLD_NEXT, "_ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    if (real_elf_str) real_elf_str(self, path);
    ensure_log();
    const std::string* s = reinterpret_cast<const std::string*>(path);
    fprintf(g_log, "ELF_FROM_FILE %p \"%s\"\n", self, s ? s->c_str() : "?");
}
extern "C" void _ZN3xrt3elfC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void* self, const void* path) {
    if (!real_elf_str) real_elf_str = (elf_str_fn)dlsym(RTLD_NEXT, "_ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    if (real_elf_str) real_elf_str(self, path);
    ensure_log();
    const std::string* s = reinterpret_cast<const std::string*>(path);
    fprintf(g_log, "ELF_FROM_FILE %p \"%s\"\n", self, s ? s->c_str() : "?");
}
