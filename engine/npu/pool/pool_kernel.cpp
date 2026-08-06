// pool_kernel.cpp — FULL native single-memory/single-call inference kernel
// load: extracts the AIE partition (PDI) from the engine's real xclbins and
// loads all 4 kernels (QKV/O/GU/D) into the pool's hwctx via CONFIG_CU —
// the same mailbox CONFIG_CU (0x11) the XRT path uses, but through the raw
// pool fd. Then runs a real GEMM with the real kernel + real insts + real
// weights: ONE EXEC_CMD chain, and VERIFIES the DPU actually computed.
//
// Build: g++ -std=c++17 -O2 -I. -I../include -I/usr/include/drm -o pool_kernel pool_kernel.cpp ../src/gemm_npu_instructions.cpp -laiebu
// Run:   sudo ./pool_kernel <xclbin_dir> [M] [K] [N]

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include "npu_pool.h"
#include "npu_utils/npu_instr_utils.hpp"
#include <aiebu/aiebu.h>

static uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// xclbin2 → AIE partition (kind 32) section bytes
static std::vector<uint8_t> extract_pdi(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("open xclbin: ") + path);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> d(sz);
    if (fread(d.data(), 1, sz, f) != (size_t)sz) { fclose(f); throw std::runtime_error("read xclbin"); }
    fclose(f);
    if (memcmp(d.data(), "xclbin2", 8) != 0) throw std::runtime_error("not xclbin2");
    size_t off = 448;   // m_numSections (verified empirically on these xclbins)
    uint32_t num = rd32(d.data() + off); off += 8;  // +4 pad before section headers
    fprintf(stderr, "  xclbin %s: %u sections\n", path, num);
    for (uint32_t i = 0; i < num; i++) {
        uint32_t kind = rd32(d.data() + off);
        uint64_t sofs = rd64(d.data() + off + 24);
        uint64_t size = rd64(d.data() + off + 32);
        if (kind == 32) {  // AIE_PARTITION
            std::vector<uint8_t> pdi(d.begin() + sofs, d.begin() + sofs + size);
            printf("  pdi: %s → %llu bytes\n", path, (unsigned long long)size);
            return pdi;
        }
        off += 40;
    }
    throw std::runtime_error("no AIE partition section");
}

static std::vector<uint32_t> load_insts(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("open insts: ") + path);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); throw std::runtime_error("read insts"); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string xd = argv[1];
    uint32_t M = argc > 2 ? (uint32_t)atoi(argv[2]) : 1024;
    uint32_t K = argc > 3 ? (uint32_t)atoi(argv[3]) : 1024;
    uint32_t N = argc > 4 ? (uint32_t)atoi(argv[4]) : 4096;
    const char* ops[4] = {"QKV", "O", "GU", "D"};

    try {
        npu::pool p("/dev/accel/accel0", 64);
        fprintf(stdout, "heap: 1 x 64MB BO — the SINGLE memory pool\n");

        // ── load all 4 kernels (PDIs) into the pool hwctx via CONFIG_CU ──
        struct cu_cfg_t {
            uint16_t num_cus; uint16_t pad[3];
            struct { uint32_t cu_bo; uint8_t cu_func; uint8_t pad2[3]; } cus[4];
        } cfg{};
        cfg.num_cus = 4;
        std::vector<npu::span> pdi_bo;
        for (int i = 0; i < 4; i++) {
            auto pdi = extract_pdi((xd + "/final_i8_" + ops[i] + "_qwen3_0_6b.xclbin").c_str());
            npu::span bo = p.alloc(pdi.size());
            memcpy(bo.host, pdi.data(), pdi.size());
            if (p.sync(bo.handle, SYNC_DIRECT_TO_DEVICE, 0, bo.size) != 0)
                throw std::runtime_error("sync pdi");
            pdi_bo.push_back(bo);
            cfg.cus[i].cu_bo = bo.handle;
            cfg.cus[i].cu_func = 0;
        }
        struct amdxdna_drm_config_hwctx ch{};
        ch.handle = p.hwctx();
        ch.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
        ch.param_val = reinterpret_cast<uint64_t>(&cfg);
        ch.param_val_size = sizeof(cfg);
        if (::ioctl(p.fd(), DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &ch) != 0)
            throw std::runtime_error(std::string("CONFIG_HWCTX(CU): ") + strerror(errno));
        fprintf(stdout, "kernels: 4 PDIs loaded (CONFIG_CU 0x11) — QKV/O/GU/D in the AIE array\n");

        // ── tensors from the ONE heap ──
        npu::span bA = p.alloc(M * K);
        npu::span bW = p.alloc(K * N);
        npu::span bC = p.alloc(M * N * 4);
        for (int i = 0; i < (int)(M * K); i++) bA.host[i] = (int8_t)(i % 7);
        for (int i = 0; i < (int)(K * N); i++) bW.host[i] = (int8_t)(i % 5);
        memset(bC.host, 0, bC.size);
        if (p.sync(bA.handle, SYNC_DIRECT_TO_DEVICE, 0, bA.size) != 0)
            throw std::runtime_error("sync bA");
        if (p.sync(bW.handle, SYNC_DIRECT_TO_DEVICE, 0, bW.size) != 0)
            throw std::runtime_error("sync bW");

        // ── real kernel instructions: generated for THIS run's addresses ──
        // (same generator the engine's hybrid path uses; DDR offsets patched
        //  at generation → absolute addresses inside the one heap window)
        extern void gemm_generate_sequence_i8(npu_sequence*, uint32_t, uint32_t, uint32_t,
                                              uint32_t, uint32_t, bool, int, uint32_t, uint32_t);
        npu_sequence nseq(device_npu2, false);
        uint32_t a_off = (uint32_t)(bA.dev_addr - 0x4000000ull);
        uint32_t b_off = (uint32_t)(bW.dev_addr - 0x4000000ull);
        uint32_t c_off = (uint32_t)(bC.dev_addr - 0x4000000ull);
        gemm_generate_sequence_i8(&nseq, M, K, N, a_off, b_off, false, 0, 0, c_off);
        nseq.cmds2seq();
        auto dump = nseq.dump();
        // assemble the instruction stream into the ELF the DPU executes
        // (the same aiebu step XRT/FLM do before the kernel runs)
        void* elf_buf = nullptr;
        int elf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (const char*)dump.first, dump.second * sizeof(uint32_t),
            NULL, 0, &elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_size <= 0) throw std::runtime_error("aiebu elf generation failed");
        npu::span ib = p.alloc_cmd(elf_size + 4096);
        memcpy(ib.host, elf_buf, elf_size);
        if (p.sync(ib.handle, SYNC_DIRECT_TO_DEVICE, 0, ib.size) != 0)
            throw std::runtime_error("sync inst");
        uint32_t ninstr = (uint32_t)dump.second;
        fprintf(stdout, "insts: GENERATED (%u words) + aiebu ELF (%d bytes), offsets a=0x%x b=0x%x c=0x%x\n",
                ninstr, elf_size, a_off, b_off, c_off);

        // ── ONE API call: CU exec of the loaded QKV kernel ──
        // kernel args = xrt::kernel signature: (opcode, instr_bo, ninstr, bA, bW, bC)
        auto c = p.make_cu_cmd(0xFF, {3, (uint32_t)ib.dev_addr, ninstr,
                                      (uint32_t)bA.dev_addr, (uint32_t)bW.dev_addr,
                                      (uint32_t)bC.dev_addr});
        uint64_t seq = p.submit_chain({c.handle});
        int wr = p.wait(seq, 10000);
        if (wr != 0) throw std::runtime_error("chain wait failed");
        fprintf(stdout, "exec: ONE ioctl (1 CU cmd, %u insts) → completed\n", ninstr);

        // ── verify the DPU actually computed ──
        int32_t* C = (int32_t*)bC.host;
        int64_t nonzero = 0; int32_t first = C[0];
        for (int i = 0; i < (int)(M * N) && i < 65536; i++) if (C[i] != 0) nonzero++;
        double expect = (double)K * 3.0 * 2.0;  // E[A]=3 (i%7), E[B]=2 (i%5)
        printf("verify: C[0]=%d nonzero=%lld/%d (expect ~%.0f per element)\n",
               first, (long long)nonzero, M * N, expect);
        int ok = (first != 0) ? 0 : 1;
        printf(ok == 0 ? "DEMO OK: kernel loaded via pool, computed in ONE call\n"
                       : "DEMO: chain completed but C=0 (kernel ran but no writeback)\n");
        return ok;
    } catch (const std::exception& e) {
        fprintf(stderr, "DEMO FAIL: %s\n", e.what());
        return 1;
    }
}
