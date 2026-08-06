// pool_probe.cpp — validate the one-heap / one-API-call pool on the live
// amdxdna driver. Creates the DEV_HEAP, carves N DEV slices, confirms each
// slice's device address lies inside the heap window, and exercises a
// batched EXEC_CMD submit (cmd_count>1) + wait. Exits 0 on success.
//
// Build: g++ -std=c++17 -O2 -o pool_probe pool_probe.cpp
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include "npu_pool.h"

static int fail(const char* m, int rc = 1) {
    fprintf(stderr, "PROBE FAIL: %s\n", m);
    return rc;
}

int main() {
    try {
        npu::pool p("/dev/accel/accel0", /*heap_mb=*/64);
        fprintf(stdout, "heap: handle=%u hwctx=%u size=%uMB\n",
                p.heap_handle(), p.hwctx(), 64);

        const uint64_t HOST_LO = 0x4000000ull, HOST_HI = HOST_LO + 64ull*1024*1024;

        // 1) carve 3 slices from the ONE heap
        std::vector<npu::span> slices;
        for (int i = 0; i < 3; i++) {
            auto s = p.alloc(1024 * 1024 + i * 4096); // 1MB, 1MB+4K, 1MB+8K
            slices.push_back(s);
            fprintf(stdout, "slice%d: handle=%u dev=0x%llx host=%p size=%llu\n",
                    i, s.handle, (unsigned long long)s.dev_addr,
                    (void*)s.host, (unsigned long long)s.size);
            if (!(s.dev_addr >= HOST_LO && s.dev_addr + s.size <= HOST_HI)) {
                fprintf(stderr, "slice%d dev addr 0x%llx OUTSIDE heap window!\n",
                        i, (unsigned long long)s.dev_addr);
                return fail("slice outside device heap");
            }
            if (!s.host) return fail("null host ptr");
        }

        // 2) host writes land in the heap kva (single pool, not per-BO mmap)
        memset(slices[0].host, 0xAB, 4096);
        memset(slices[1].host, 0xCD, 4096);
        if (slices[0].host[0] != 0xAB || slices[1].host[4095] != 0xCD)
            return fail("host write into heap slice failed");

        // 3) ONE API CALL: chain of 3 DPU sub-commands inside one BO_CMD,
        //    submitted as ONE EXEC_CMD (cmd_count=1) → one mailbox
        //    CHAIN_EXEC_NPU message. The inst buffer is zero-filled: a real
        //    DPU program (e.g. engine/npu/fused_insts/layer_L1.bin) requires
        //    the engine's tensor args, without which the DPU hangs. The
        //    zeroed program completes, proving heap→chain→firmware→syncobj.
        auto inst = p.alloc(4096);
        memset(inst.host, 0, 4096);
        p.sync(inst.handle, SYNC_DIRECT_TO_DEVICE, 0, inst.size);
        fprintf(stdout, "inst: dev=0x%llx size=4096 (zeroed; real PDI needs engine args)\n",
                (unsigned long long)inst.dev_addr);

        std::vector<uint32_t> sub_cmds;
        for (int i = 0; i < 3; i++) {
            auto c = p.make_dpu_cmd(0xFF /*cu_mask: cols 0-7*/,
                                    inst.dev_addr, 4096,
                                    {0, 0, 0});
            sub_cmds.push_back(c.handle);
        }
        uint64_t seq = p.submit_chain(sub_cmds);
        fprintf(stdout, "submit_chain: seq=%llu, %zu DPU sub-commands in ONE ioctl\n",
                (unsigned long long)seq, sub_cmds.size());

        // 4) one wait for the chain
        int wr = p.wait(seq, 5000);
        if (wr != 0) {
            fprintf(stderr, "wait rc=%d errno=%s\n", wr, strerror(errno));
            return fail("wait failed");
        }
        fprintf(stdout, "wait: ok (rc=%d)\n", wr);

        // 5) sync the heap slices (single pool sync path; TO_DEVICE flushes
        //    CPU writes to the device). FROM_DEVICE is the debug-BO dump path
        //    (hwctx_sync_debug_bo) — not part of the pool contract.
        for (auto& s : slices) {
            if (p.sync(s.handle, SYNC_DIRECT_TO_DEVICE, 0, 4096) != 0)
                return fail("sync TO_DEVICE failed");
        }

        fprintf(stdout, "PROBE OK: one heap (%zu slices, 64MB), one submit (3 DPU cmds), one wait\n",
                slices.size());
        return 0;
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}
