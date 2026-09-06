// pool_split.cpp — TWO mailboxes open on one NPU (2-hwctx split probe).
//
// Question: can a single process hold TWO amdxdna hardware contexts (each =
// its own /dev/accel fd, DEV_HEAP, mailbox channel, syncobj) on the SAME
// NPU at the same time — and do both actually execute chains concurrently?
//
// The driver's per-context mailbox model (one cq_pair per hwctx, MAX_CQ_PAIRS=2
// in the protocol) means: 1 hwctx = 1 mailbox channel. Two pools = two channels.
// If both hwctxs claim the full array (num_tiles=32), exec may serialize at the
// column level; mode --half gives each pool num_tiles=16 (split the array).
//
// Flow per pool (reuses pool_kernel's real path): open → DEV_HEAP(64MB) →
// CREATE_HWCTX → load the real QKV PDI → CONFIG_CU → tensors from the one heap
// → generate real DPU insts (gemm_generate_sequence_i8 + aiebu ELF) → one
// EXEC_CMD chain → syncobj wait → verify the DPU wrote C.
//
// Build (on the NPU box, from engine/npu/pool):
//   g++ -std=c++17 -O2 -I. -I../include -I/usr/include/drm \
//       -o pool_split pool_split.cpp ../src/gemm_npu_instructions.cpp -laiebu
// Run:  sudo ./pool_split <xclbin_dir> [iters] [--half]
//   e.g. sudo ./pool_split ../xclbins 10          # full array, both pools
//        sudo ./pool_split ../xclbins 10 --half   # 16 tiles each

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <atomic>
#include <unistd.h>
#include "npu_pool.h"
#include "npu_utils/npu_instr_utils.hpp"
#include <aiebu/aiebu.h>

static uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// xclbin2 → AIE partition (kind 32) section bytes (same as pool_kernel)
static std::vector<uint8_t> extract_pdi(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("open xclbin: ") + path);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> d(sz);
    if (fread(d.data(), 1, sz, f) != (size_t)sz) { fclose(f); throw std::runtime_error("read xclbin"); }
    fclose(f);
    if (memcmp(d.data(), "xclbin2", 8) != 0) throw std::runtime_error("not xclbin2");
    size_t off = 448;   // m_numSections (verified empirically on these xclbins)
    uint32_t num = rd32(d.data() + off); off += 8;
    for (uint32_t i = 0; i < num; i++) {
        uint32_t kind = rd32(d.data() + off);
        uint64_t sofs = rd64(d.data() + off + 24);
        uint64_t size = rd64(d.data() + off + 32);
        if (kind == 32) {
            std::vector<uint8_t> pdi(d.begin() + sofs, d.begin() + sofs + size);
            printf("  pdi: %s → %llu bytes\n", path, (unsigned long long)size);
            return pdi;
        }
        off += 40;
    }
    throw std::runtime_error("no AIE partition section");
}

// One mailbox channel = one pool object. Self-contained kernel load + run.
struct engine {
    npu::pool p;
    npu::span bA, bW, bC, ib;
    uint32_t ninstr;
    uint32_t a_off, b_off, c_off;
    std::string name;

    engine(const char* xd, const char* name_, uint32_t tiles)
        : p("/dev/accel/accel0", 64, tiles), name(name_) {
        const uint32_t M = 1024, K = 1024, N = 4096;

        // load ONE kernel (QKV) into this hwctx via CONFIG_CU
        struct cu_cfg_t {
            uint16_t num_cus; uint16_t pad[3];
            struct { uint32_t cu_bo; uint8_t cu_func; uint8_t pad2[3]; } cus[1];
        } cfg{};
        cfg.num_cus = 1;
        std::string xcl = std::string(xd) + "/final_i8_QKV_qwen3_0_6b.xclbin";
        auto pdi = extract_pdi(xcl.c_str());
        npu::span bo = p.alloc(pdi.size());
        memcpy(bo.host, pdi.data(), pdi.size());
        if (p.sync(bo.handle, SYNC_DIRECT_TO_DEVICE, 0, bo.size) != 0)
            throw std::runtime_error("sync pdi");
        cfg.cus[0].cu_bo = bo.handle;
        cfg.cus[0].cu_func = 0;
        struct amdxdna_drm_config_hwctx ch{};
        ch.handle = p.hwctx();
        ch.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
        ch.param_val = reinterpret_cast<uint64_t>(&cfg);
        ch.param_val_size = sizeof(cfg);
        if (::ioctl(p.fd(), DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &ch) != 0)
            throw std::runtime_error(std::string(name + " CONFIG_HWCTX(CU): ") + strerror(errno));

        // tensors from this pool's one heap
        bA = p.alloc((size_t)M * K);
        bW = p.alloc((size_t)K * N);
        bC = p.alloc((size_t)M * N * 4);
        for (int i = 0; i < (int)((size_t)M * K); i++) bA.host[i] = (int8_t)(i % 7);
        for (int i = 0; i < (int)((size_t)K * N); i++) bW.host[i] = (int8_t)(i % 5);
        memset(bC.host, 0, bC.size);
        if (p.sync(bA.handle, SYNC_DIRECT_TO_DEVICE, 0, bA.size) != 0) throw std::runtime_error("sync bA");
        if (p.sync(bW.handle, SYNC_DIRECT_TO_DEVICE, 0, bW.size) != 0) throw std::runtime_error("sync bW");

        // generate real DPU instructions for THIS pool's addresses
        extern void gemm_generate_sequence_i8(npu_sequence*, uint32_t, uint32_t, uint32_t,
                                              uint32_t, uint32_t, bool, int, uint32_t, uint32_t);
        npu_sequence nseq(device_npu2, false);
        a_off = (uint32_t)(bA.dev_addr - 0x4000000ull);
        b_off = (uint32_t)(bW.dev_addr - 0x4000000ull);
        c_off = (uint32_t)(bC.dev_addr - 0x4000000ull);
        gemm_generate_sequence_i8(&nseq, M, K, N, a_off, b_off, false, 0, 0, c_off);
        nseq.cmds2seq();
        auto dump = nseq.dump();
        void* elf_buf = nullptr;
        int elf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (const char*)dump.first, dump.second * sizeof(uint32_t),
            NULL, 0, &elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_size <= 0) throw std::runtime_error("aiebu elf generation failed");
        ib = p.alloc_cmd(elf_size + 4096);
        memcpy(ib.host, elf_buf, elf_size);
        if (p.sync(ib.handle, SYNC_DIRECT_TO_DEVICE, 0, ib.size) != 0) throw std::runtime_error("sync inst");
        ninstr = (uint32_t)dump.second;
        printf("  [%s] hwctx=%u fd=%d heap_dev=0x%llx tiles=%u insts=%u\n",
               name.c_str(), p.hwctx(), p.fd(),
               (unsigned long long)p.heap_handle(), tiles, ninstr);
    }

    // one full chain exec + wait; returns wall µs
    long run_once() {
        auto c = p.make_cu_cmd(0xFF, {3, (uint32_t)ib.dev_addr, ninstr,
                                      (uint32_t)bA.dev_addr, (uint32_t)bW.dev_addr,
                                      (uint32_t)bC.dev_addr});
        auto t0 = std::chrono::steady_clock::now();
        uint64_t seq = p.submit_chain({c.handle});
        int wr = p.wait(seq, 30000);
        auto t1 = std::chrono::steady_clock::now();
        if (wr != 0) throw std::runtime_error(name + " wait failed");
        return (long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    // DPU actually computed?
    bool verify() {
        int32_t* C = (int32_t*)bC.host;
        return C[0] != 0;
    }
};

static double avg_us(const std::vector<long>& v) {
    if (v.empty()) return 0;
    long s = 0;
    for (auto x : v) s += x;
    return (double)s / v.size();
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <xclbin_dir> [iters] [--half]\n", argv[0]); return 1; }
    std::string xd = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 8;
    bool half = argc > 3 && std::string(argv[3]) == "--half";
    uint32_t tiles = half ? 16 : 32;
    if (iters < 1) iters = 1;

    printf("== pool_split: TWO mailboxes on one NPU ==\n");
    printf("tiles per hwctx: %u (%s array split)\n\n", tiles,
           half ? "HALF" : "FULL (both claim everything)");

    // Phase 1: can we even OPEN two hwctx (two mailbox channels)?
    printf("phase 1: open pool A ...\n");
    engine* ea = nullptr;
    engine* eb = nullptr;

    // --solo mode: ONE process, ONE pool, loop N chains, print per-iter µs.
    // Used to run two single-pool processes concurrently (two mailboxes via
    // two processes instead of two contexts in one process).
    if (argc > 3 && std::string(argv[3]) == "--solo") {
        printf("[solo pid=%d] one pool, %d iters, tiles=%u\n", getpid(), iters, tiles);
        ea = new engine(xd.c_str(), "solo", tiles);
        fflush(stdout);
        for (int i = 0; i < iters; i++) {
            long us = ea->run_once();
            printf("[solo pid=%d] iter %d: %ld µs  computed=%s\n",
                   getpid(), i, us, ea->verify() ? "YES" : "NO");
            fflush(stdout);
        }
        delete ea;
        return 0;
    }
    try {
        ea = new engine(xd.c_str(), "A", tiles);
        printf("phase 1: A open OK (hwctx %u). open pool B ...\n", ea->p.hwctx());
        eb = new engine(xd.c_str(), "B", tiles);
        printf("phase 1: B open OK (hwctx %u) — TWO MAILBOXES ARE OPEN\n\n", eb->p.hwctx());
    } catch (const std::exception& e) {
        printf("phase 1 FAIL: %s\n", e.what());
        printf("=> second hwctx rejected. If tiles conflict, retry with --half.\n");
        delete ea; delete eb;
        return 2;
    }

    // Phase 2: single-shot sanity on both, verify both computed
    printf("phase 2: single chain on A and B ...\n");
    ea->run_once(); eb->run_once();
    printf("  A computed: %s  |  B computed: %s\n",
           ea->verify() ? "YES" : "NO", eb->verify() ? "YES" : "NO");

    // Phase 3: solo timing
    printf("\nphase 3: solo timing (%d iters each)\n", iters);
    std::vector<long> ta, tb;
    for (int i = 0; i < iters; i++) ta.push_back(ea->run_once());
    for (int i = 0; i < iters; i++) tb.push_back(eb->run_once());
    printf("  A solo: %8.0f µs/chain\n", avg_us(ta));
    printf("  B solo: %8.0f µs/chain\n", avg_us(tb));

    // Phase 4: concurrent — one thread per mailbox, both hammering
    printf("\nphase 4: CONCURRENT A ∥ B (%d iters each, 2 threads)\n", iters);
    std::atomic<bool> go{false};
    std::vector<long> ca, cb;
    std::thread thA([&] {
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        for (int i = 0; i < iters; i++) ca.push_back(ea->run_once());
    });
    std::thread thB([&] {
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        for (int i = 0; i < iters; i++) cb.push_back(eb->run_once());
    });
    go.store(true, std::memory_order_release);
    thA.join(); thB.join();
    printf("  A conc: %8.0f µs/chain  (solo %8.0f)  %s\n", avg_us(ca), avg_us(ta),
           avg_us(ca) < avg_us(ta) * 2.05 ? "<= 2x solo: REAL OVERLAP or interleave" :
                                            "~2x solo: serialized on the array");
    printf("  B conc: %8.0f µs/chain  (solo %8.0f)\n", avg_us(cb), avg_us(tb));
    double agg_us = (avg_us(ca) + avg_us(cb)) / 2.0;
    double solo_us = (avg_us(ta) + avg_us(tb)) / 2.0;
    printf("  mean per-chain: solo %.0f µs | concurrent %.0f µs | overlap factor %.2f\n",
           solo_us, agg_us, solo_us / (agg_us > 0 ? agg_us : 1));

    delete ea; delete eb;
    return 0;
}
