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
//
// GATES: the documented gates this build READS are CAP_DIR and CAP_POSTRUN_ACT.
// The marker below lists every CAP_* the file getenv()s, which is a larger set: it
// also carries the value-style knobs no operator doc gates on (CAP_BIG_MAX,
// CAP_MM_W, CAP_RUNLIST_KV, CAP_POSTRUN_KV). Testing/capture_gates_selfcheck.py
// requires that marker to equal the getenv() set, so adding a variable here means
// updating the marker in the same commit. The three dump gates -
// CAP_NO_SYNC, CAP_SKIP_BIG, CAP_DUMP_BIG - are implemented on
// goal/runlist-decode-wire, NOT here, and they are REFUSED at load time rather than
// ignored (see refuse_unimplemented_gates below). That refusal is not cosmetic:
// docs/AGENT-COORDINATION.md tells the next capture to set CAP_NO_SYNC=1 to keep a 1k
// bench at 3.3 GB, and without it a verification run wrote 181 GB in five minutes. On a
// build that ignores the gate, setting it is worse than not setting it: the operator
// believes they are protected.
//
// capture-gates-refused: CAP_NO_SYNC CAP_SKIP_BIG CAP_DUMP_BIG
// capture-gates-read: CAP_BIG_MAX CAP_DIR CAP_DUMP_BIG CAP_MM_W CAP_NO_SYNC CAP_POSTRUN_ACT CAP_POSTRUN_KV CAP_RUNLIST_KV CAP_SKIP_BIG
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

// Refused, not ignored. The names below are the gates this tool's own documentation and
// the published post tell an operator to set; the implementation lives on
// goal/runlist-decode-wire (302 lines that rework the capture paths, not a rename).
// Ignoring them silently is the trap: dump_bo() below writes every BO on every sync,
// which is the measured 181 GB case, and an operator who set CAP_NO_SYNC=1 would read a
// completed capture as one that stayed small. Fail at load instead - before a byte is
// dumped.
void refuse_unimplemented_gates(void) {
    static const char* refused[] = {"CAP_NO_SYNC", "CAP_SKIP_BIG", "CAP_DUMP_BIG"};
    for (const char* name : refused) {
        const char* v = getenv(name);
        if (v && *v && strcmp(v, "0") != 0) {
            fprintf(stderr,
                    "cap_interposer: %s=%s is NOT implemented on this build (#2528).\n"
                    "  This interposer reads CAP_DIR and CAP_POSTRUN_ACT only; the dump gates\n"
                    "  live on goal/runlist-decode-wire. Refusing to capture: ignoring the gate\n"
                    "  would dump every BO per sync (the measured 181 GB case), and a capture\n"
                    "  that silently ignores a gate reads as one that honoured it.\n"
                    "  Either unset %s, or build the interposer from that branch.\n",
                    name, v, name);
            _exit(2);
        }
    }
}

// Runs when the .so is loaded, i.e. before the traced program's main().
__attribute__((constructor)) static void cap_interposer_gate_check(void) {
    refuse_unimplemented_gates();
}
static FILE* g_log = nullptr;
#include <set>
#include <vector>
static std::set<std::pair<unsigned long, size_t>> g_bo_sizes;
static std::set<std::pair<unsigned long, size_t>> g_extbo_sizes;
static long g_seq = 0;
static std::map<unsigned long, std::string> g_bo_labels;
static std::set<size_t> g_seen_big;      // keyed by POINTER (not size) — see the dump site
static std::map<size_t, int> g_big_per_size;  // pointer-dedup alone can still flood; cap per size

// ---------------------------------------------------------------------------
// BO lifetime registry — why this exists
//
// Every capture container in this file keys by the ADDRESS of an xrt::bo that
// belongs to the RUNTIME, not to us: g_run_bo_ptrs (filled from
// run::set_arg_at_index), g_bo_sizes (from xrt::bo::sync), g_extbo_sizes (from
// xrt::ext::bo::bo) and g_act_bo / g_kv_bo. Those addresses are dereferenced
// much later, inside runlist::execute() / runlist::wait() — but xrt::bo is a
// HANDLE (detail::pimpl<bo_impl>, i.e. a shared_ptr) whose address says nothing
// about its lifetime, and FLM binds temporaries for some arguments
// (`run.set_arg(7, xrt::bo{...})`). By the time execute() walks the maps such an
// object is gone, so `reinterpret_cast<xrt::bo*>(addr)->map()` reads freed
// memory and dies at `xrt::bo::map()+163` (`mov (%rdi),%rax`).
//
// Reproduced twice on 2026-09-15: `flm bench nanbeige4.1:3b` at 00:29:49
// (CAP_DIR=capnb_L1024) and again at 00:30:37 (capnb_L2048). Both runs stop
// mid-loop at `RUNLIST 65: execute (pre-dump)` — after three PREINSTS dumps and
// before the loop's `pre-dumped N insts BOs` terminator — i.e. on the next
// address the loop dereferences. Those runs saw 114 distinct BO addresses and 11
// run addresses, so owning them is bounded.
//
// Fix: whenever we hold a pointer to an object we KNOW is alive (we are inside
// one of its own methods, or it was just handed to set_arg_at_index), keep an
// owning COPY here under the same address the capture maps already use. A copy
// shares the same bo_impl, so the impl and its buffer stay alive for as long as
// we do; every later dereference goes through bo_from_addr() and uses the copy,
// never the stale address. Containers and log/filename formats are unchanged.
//
// NOTE (2026-09-15): xrt::ext::bo derives from xrt::bo, so ownership of an
// ext::bo address also keeps the base subobject valid. The copy is a slice of
// the derived object, which is fine — size()/map() are base methods and the
// shared bo_impl is what has to survive.
// ---------------------------------------------------------------------------
static std::map<size_t, xrt::bo> g_bo_owner;

static void own_bo(const void* addr) {
    if (!addr) return;
    const xrt::bo* b = reinterpret_cast<const xrt::bo*>(addr);
    auto it = g_bo_owner.find((size_t)addr);
    if (it == g_bo_owner.end()) g_bo_owner.emplace((size_t)addr, *b);
    else it->second = *b;   // same address reused by a newer BO: keep the newer one
}

// Owning copy for an address recorded earlier, or nullptr if it was never
// registered while alive — callers must skip on nullptr rather than deref.
static xrt::bo* bo_from_addr(const void* addr) {
    if (!addr) return nullptr;
    auto it = g_bo_owner.find((size_t)addr);
    return it == g_bo_owner.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------------------
// The run-keyed path owns its BOs DIRECTLY, keyed by the very (run, arg) slots
// g_run_bo_ptrs already uses — this is the path the kernel-identification work
// actually reads (the preinsts / arg3 / postrun dumps).
//
// The address registry above removes the crash but still resolves BY ADDRESS,
// and on this path that is not enough: the runtime allocates fresh BOs per call
// (the capture log shows a4 and a7 changing on every runlist), so a freed 1 MiB
// slot is handed straight back for the next 1 MiB BO. An address recorded for an
// older run key — and execute() walks EVERY run key, not just the current one —
// can therefore resolve to a NEWER BO and dump the wrong buffer under the old
// run's name. That is a silent wrong capture, which is the failure mode this
// lane has already been burned by, so it is worth more than the crash fix.
//
// Owning the copy in the (run, arg) slot cannot alias: the slot holds exactly the
// BO currently bound to that run and argument, and a stale different run key
// keeps its own pinned copy.
// ---------------------------------------------------------------------------
static std::map<unsigned long, std::map<int, xrt::bo>> g_run_bos;

// Owning BO for one (run, arg) slot, or nullptr if that slot was never bound.
static xrt::bo* run_bo(unsigned long runkey, int arg) {
    auto it = g_run_bos.find(runkey);
    if (it == g_run_bos.end()) return nullptr;
    auto a = it->second.find(arg);
    return a == it->second.end() ? nullptr : &a->second;
}

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
        // self is this xrt::bo, so it is provably alive right now: take ownership
        // BEFORE recording the address — runlist::execute() dereferences it later.
        own_bo(self);
        xrt::bo* bo = bo_from_addr(self);
        if (!bo) return;
        size_t bosz = bo->size();
        g_bo_sizes.insert({(unsigned long)self, bosz});
        bool capture = !getenv("CAP_NO_SYNC");  // gate: CAP_NO_SYNC keeps only runlist preinsts (i6) dumps — the per-sync 32MB kv writes fill /tmp on long runs
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
        // The caller's object is alive for the duration of this call, but NOT
        // necessarily until runlist::execute() reads the address back out of
        // g_run_bo_ptrs. The (run, arg) slot below owns a copy from here on and
        // every later dereference on this path goes through run_bo(), so this
        // path does not touch the address registry at all — which also stops
        // g_bo_owner accumulating one owner per BO FLM ever binds.
        const xrt::bo* b = reinterpret_cast<const xrt::bo*>(bo);
        g_run_bos[(unsigned long)self][idx] = *b;   // exact owner for this (run, arg)
        g_run_args[(unsigned long)self].push_back({idx, b->size()});
        g_run_bo_ptrs[(unsigned long)self][idx] = bo;
        ensure_log();
        fprintf(g_log, "SETARG %p idx=%d size=%zu bo=%p\n", self, idx, b->size(), (void*)bo);
        // Dump idx3. CORRECTED 2026-09-13: idx3 is NOT an instruction BO. The vendor's
        // create_run (npu_utils_xrt.hpp:262-274) binds BOs at 3+i in CALLER order with no
        // intrinsic meaning, and FLM's first BO is its ACTIVATION buffer -- 1 MB of float/
        // bf16 data. A capture that labelled it "insts" sent a reader into a byte analysis
        // of a data buffer looking for an instruction transaction header, so the file is
        // now named arg3_* and the log says ARG3_DUMP. A real transaction starts with a
        // structured header (layer_ctx1.txn begins 0001 0406 0801 ...); this file begins
        // 14c3 1e41 4840 873f, which is bf16 data.
        if (idx == 3 && b->size() <= 2000000) {
            const uint8_t* pm = (const uint8_t*)b->map();
            char fn[256];
            snprintf(fn, sizeof(fn), "%s/arg3_%04ld_%zu.bin", CAP_DIR, g_seq, b->size());
            FILE* ff = fopen(fn, "wb");
            if (ff) { fwrite(pm, 1, b->size(), ff); fclose(ff); }
            fprintf(g_log, "ARG3_DUMP -> %s (activation/data BO, NOT instructions)\n", fn);
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
    // post-run dump of the kv BO (idx7, 32MB) — the runtime's KV write
    if (getenv("CAP_POSTRUN_KV")) {
        auto it = g_run_bo_ptrs.find((unsigned long)self);
        if (it != g_run_bo_ptrs.end()) {
            auto a7 = it->second.find(7);
            if (a7 != it->second.end()) {
                try {
                    xrt::bo* bo = run_bo((unsigned long)self, 7);
                    const uint8_t* p = bo ? (const uint8_t*)bo->map() : nullptr;
                    if (p) {
                        char fname[256];
                        snprintf(fname, sizeof(fname), "%s/postrun_kv_%03d.bin", CAP_DIR, n);
                        FILE* f = fopen(fname, "wb");
                        if (f) { fwrite(p, 1, bo->size(), f); fclose(f); }
                        fprintf(g_log, "POSTRUN_KV -> %s (%zu B)\n", fname, bo->size());
                    }
                } catch (...) {}
            }
        }
    }
    // post-run dump of the act BO (idx3) — the layer's output written in-place
    if (getenv("CAP_POSTRUN_ACT")) {
        auto it = g_run_bo_ptrs.find((unsigned long)self);
        if (it != g_run_bo_ptrs.end()) {
            auto a3 = it->second.find(3);
            if (a3 != it->second.end()) {
                try {
                    xrt::bo* bo = run_bo((unsigned long)self, 3);
                    const uint8_t* p = bo ? (const uint8_t*)bo->map() : nullptr;
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
                    xrt::bo* bo = run_bo(runkey, aidx);   // owned by the (run, arg) slot — cannot alias
                    if (!bo) continue;   // slot never bound: skip instead of dereferencing a dead address
                    size_t bosz = bo->size();
                    if (bosz > 3000000) {           // weight/kv BOs: dump once if CAP_DUMP_BIG
                        if (!getenv("CAP_DUMP_BIG")) continue;
                        // Dedup by POINTER, not by size. Keying by size kept only the FIRST BO of
                        // each size, which is NOT the one the manifest names as arg4 -- so a
                        // byte-level comparison against it silently diffed a third, same-sized
                        // object (found in 25.1, which is why that diff is void). The manifest
                        // reports the bound pointer, so the dumped file must carry it.
                        // A per-size cap keeps this bounded: CAP_BIG_MAX (default 4), because a
                        // 32-layer model otherwise writes 32 x 59 MB of same-sized weight BOs.
                        if (g_seen_big.count((size_t)bop)) continue;
                        int cap = 4;
                        if (const char* e = getenv("CAP_BIG_MAX")) { cap = atoi(e); if (cap < 1) cap = 4; }
                        if (g_big_per_size[bosz] >= cap) continue;
                        g_seen_big.insert((size_t)bop);
                        g_big_per_size[bosz]++;
                    }
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
    // plus the 32 MB kv BO (the runtime's per-layer KV cache)
    for (auto& kv : g_extbo_sizes) {
        if (kv.second == 8388608 && getenv("CAP_MM_W")) {
            try {
                xrt::bo* mwb = bo_from_addr((const void*)kv.first);
                const uint8_t* pm = mwb ? (const uint8_t*)mwb->map() : nullptr;
                if (pm) {
                    char fname[256];
                    snprintf(fname, sizeof(fname), "%s/mmw_%03ld_%zx_%zu.bin", CAP_DIR, g_runlist_n, (size_t)kv.first, kv.second);
                    FILE* f = fopen(fname, "wb");
                    if (f) { fwrite(pm, 1, kv.second, f); fclose(f); }
                    fprintf(g_log, "MMW -> %s\n", fname);
                }
            } catch (...) {}
        }
        if (kv.second == 33554432 && getenv("CAP_RUNLIST_KV")) {
            try {
                xrt::bo* kvb = bo_from_addr((const void*)kv.first);
                const uint8_t* pm = kvb ? (const uint8_t*)kvb->map() : nullptr;
                if (pm) {
                    char fname[256];
                    snprintf(fname, sizeof(fname), "%s/runlist_kv_%03ld_%zx.bin",
                             CAP_DIR, g_runlist_n, (size_t)kv.first);
                    FILE* f = fopen(fname, "wb");
                    if (f) { fwrite(pm, 1, kv.second, f); fclose(f); }
                    fprintf(g_log, "RUNLIST_KV -> %s\n", fname);
                }
            } catch (...) {}
        }
        if (kv.second > 2000000) continue;
        try {
            xrt::bo* xsb = bo_from_addr((const void*)kv.first);
            const uint8_t* pm = xsb ? (const uint8_t*)xsb->map() : nullptr;
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
                xrt::bo* bo = bo_from_addr((const void*)kv.first);
                if (!bo) continue;
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
        if (getenv("CAP_SKIP_BIG")) continue;
        if (getenv("CAP_NO_SYNC")) continue;   // lean: preinsts (i6) only
        try {
            xrt::bo* bo = bo_from_addr((const void*)kv.first);
            if (!bo) continue;
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

// ===== runlist::add hook — capture the EXACT per-forward run order =====
static long g_add_n = 0;
// Owning pointers into g_run_bos nodes (std::map nodes are stable), so the wait
// hook holds the BO itself rather than an address that may have been reused.
// The original address is kept alongside purely for filenames and log lines.
static xrt::bo* g_act_bo = nullptr;         // last-seen arg idx=3 (act) BO
static const void* g_act_bo_addr = nullptr;
static void record_act_bo(const void* run) {
    xrt::bo* b = run_bo((unsigned long)run, 3);
    if (!b) return;
    g_act_bo = b;
    auto it = g_run_bo_ptrs.find((unsigned long)run);
    if (it != g_run_bo_ptrs.end()) {
        auto a3 = it->second.find(3);
        if (a3 != it->second.end()) g_act_bo_addr = a3->second;
    }
}

static xrt::bo* g_kv_bo = nullptr;          // last-seen arg idx=7 (kv) BO
static const void* g_kv_bo_addr = nullptr;
static void record_kv_bo(const void* run) {
    xrt::bo* b = run_bo((unsigned long)run, 7);
    if (!b) return;
    g_kv_bo = b;
    auto it = g_run_bo_ptrs.find((unsigned long)run);
    if (it != g_run_bo_ptrs.end()) {
        auto a7 = it->second.find(7);
        if (a7 != it->second.end()) g_kv_bo_addr = a7->second;
    }
}
typedef void (*rl_add_fn)(void*, const void*);
static rl_add_fn real_rl_add = nullptr;
extern "C" void _ZN3xrt7runlist3addERKNS_3runE(void* self, const void* run) {
    if (!real_rl_add) real_rl_add = (rl_add_fn)dlsym(RTLD_NEXT, "_ZN3xrt7runlist3addERKNS_3runE");
    if (real_rl_add) real_rl_add(self, run);
    ensure_log();
    record_act_bo(run);
    record_kv_bo(run);
    // resolve this run's args from the map
    auto it = g_run_bo_ptrs.find((unsigned long)run);
    fprintf(g_log, "RUNLIST_ADD run=%p rl=%p", run, self);
    if (it != g_run_bo_ptrs.end()) {
        for (auto& ab : it->second)
            fprintf(g_log, " a%d=%p", ab.first, ab.second);
    }
    fprintf(g_log, "\n");
}
extern "C" void _ZN3xrt7runlist3addEONS_3runE(void* self, void* run) {
    if (!real_rl_add) real_rl_add = (rl_add_fn)dlsym(RTLD_NEXT, "_ZN3xrt7runlist3addEONS_3runE");
    if (real_rl_add) real_rl_add(self, run);
    ensure_log();
    record_act_bo(run);
    record_kv_bo(run);
    auto it = g_run_bo_ptrs.find((unsigned long)run);
    fprintf(g_log, "RUNLIST_ADD(rv) run=%p rl=%p", run, self);
    if (it != g_run_bo_ptrs.end()) {
        for (auto& ab : it->second)
            fprintf(g_log, " a%d=%p", ab.first, ab.second);
    }
    fprintf(g_log, "\n");
}

// ===== runlist::wait hook — dump the act BO AFTER the device completes =====
typedef void (*rl_wait_fn)(void*, const void*);
static rl_wait_fn real_rl_wait = nullptr;
extern "C" void _ZNK3xrt7runlist4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE(void* self, const void* dur) {
    if (!real_rl_wait) real_rl_wait = (rl_wait_fn)dlsym(RTLD_NEXT, "_ZNK3xrt7runlist4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE");
    if (real_rl_wait) real_rl_wait(self, dur);
    ensure_log();
    fprintf(g_log, "RUNLIST wait done\n");
    if (g_act_bo && !getenv("CAP_NO_SYNC")) {
        try {
            xrt::bo* bo = g_act_bo;
            size_t bosz = bo->size();
            const uint8_t* p = (const uint8_t*)bo->map();
            if (p) {
                char fname[256];
                snprintf(fname, sizeof(fname), "%s/actpost_%03ld_%zx_%zu.bin", CAP_DIR, g_runlist_n, (size_t)g_act_bo_addr, bosz);
                FILE* f = fopen(fname, "wb");
                if (f) { fwrite(p, 1, bosz, f); fclose(f); }
                fprintf(g_log, "ACTPOST runlist=%ld act=%p size=%zu -> %s\n", g_runlist_n, g_act_bo_addr, bosz, fname);
            }
        } catch (...) {}
    }
    if (g_kv_bo && !getenv("CAP_NO_SYNC")) {
        try {
            xrt::bo* bo = g_kv_bo;
            size_t bosz = bo->size();
            const uint8_t* p = (const uint8_t*)bo->map();
            if (p) {
                char fname[256];
                snprintf(fname, sizeof(fname), "%s/kvpost_%03ld_%zx_%zu.bin", CAP_DIR, g_runlist_n, (size_t)g_kv_bo_addr, bosz);
                FILE* f = fopen(fname, "wb");
                if (f) { fwrite(p, 1, bosz, f); fclose(f); }
                fprintf(g_log, "KVPOST runlist=%ld kv=%p size=%zu -> %s\n", g_runlist_n, g_kv_bo_addr, bosz, fname);
            }
        } catch (...) {}
    }
    // post-wait dump of ALL big ext::bo (complete per-layer kv/weight state)
    if (!getenv("CAP_SKIP_BIG") && !getenv("CAP_NO_SYNC")) {
        int n = 0;
        for (auto& kv : g_extbo_sizes) {
            if (kv.second <= 1000000) continue;
            try {
                xrt::bo* wpb = bo_from_addr((const void*)kv.first);
                const uint8_t* pm = wpb ? (const uint8_t*)wpb->map() : nullptr;
                if (pm) {
                    char fname[256];
                    snprintf(fname, sizeof(fname), "%s/waitpost_%03ld_%02d_%zx_%zu.bin", CAP_DIR, g_runlist_n, n, (size_t)kv.first, kv.second);
                    FILE* f = fopen(fname, "wb");
                    if (f) { fwrite(pm, 1, kv.second, f); fclose(f); }
                    n++;
                }
            } catch (...) {}
        }
        fprintf(g_log, "WAITPOST runlist=%ld dumped %d big ext BOs\n", g_runlist_n, n);
    }
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
    // self is the ext::bo just constructed (ext::bo derives from xrt::bo), so the
    // base subobject is alive now — own it before recording the address.
    own_bo(self);
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
