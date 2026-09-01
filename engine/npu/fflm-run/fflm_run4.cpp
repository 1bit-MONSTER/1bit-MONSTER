// Launch the REAL FastFlowLM Qwen3-0.6B layer kernel the way the real runtime does.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>
#include <aiebu/aiebu.h>
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <xclbin> <layerN.seq>\n", argv[0]); return 2; }
    xrt::device dev(0);
    std::string xp = argv[1], sp = argv[2];
    xrt::xclbin x{xp};
    dev.register_xclbin(x);
    xrt::hw_context hw(dev, x.get_uuid());
    std::string kn;
    for (auto& k : x.get_kernels()) { if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kn = k.get_name(); break; } }
    fprintf(stderr, "xclbin kernel name: %s\n", kn.c_str());
    FILE* f = fopen(sp.c_str(), "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> ins(sz/4); fread(ins.data(), 4, ins.size(), f); fclose(f);
    fprintf(stderr, "seq: %zu words (%zu B), header=%08x %08x %08x %08x\n",
            ins.size(), ins.size()*4, ins[0], ins[1], ins[2], ins[3]);
    void* elf_buf = nullptr;
    int elf_sz = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        (const char*)ins.data(), ins.size() * sizeof(uint32_t),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (elf_sz <= 0) { fprintf(stderr, "aiebu_assembler_get_elf failed: %d\n", elf_sz); return 1; }
    fprintf(stderr, "aiebu ELF: %d bytes\n", elf_sz);
    xrt::elf elf(elf_buf, (size_t)elf_sz);
    free(elf_buf);
    xrt::module mod(elf);
    xrt::ext::kernel kk(hw, mod, kn);
    // BOs per the real patch table: arg1=weights ~7745536, arg4=acts ~200704,
    // arg0/2/3 = 4096 each -> bo0..bo4 (kernel args 3..7)
    std::vector<size_t> sizes = {4096, 7745536, 4096, 4096, 200704};
    std::vector<xrt::ext::bo> bos;
    for (int a = 0; a < 5; a++) {
        bos.emplace_back(dev, sizes[a]);
        memset(bos[a].map(), (a == 4) ? 0x3f : 0x01, sizes[a]);
        bos[a].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    // NOTE: cast to xrt::bo& -- the run's variadic set_arg must hit the BO
    // overload, not the scalar template (sizeof(xrt::bo)=16 would throw).
    fprintf(stderr, "launching real kernel (instr from module, args 3,0,0)...\n");
    auto t0 = std::chrono::steady_clock::now();
    auto r = kk((unsigned)3, 0u, 0u,
                static_cast<xrt::bo&>(bos[0]), static_cast<xrt::bo&>(bos[1]),
                static_cast<xrt::bo&>(bos[2]), static_cast<xrt::bo&>(bos[3]),
                static_cast<xrt::bo&>(bos[4]));
    r.wait();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();
    fprintf(stderr, "state=%d elapsed=%ldms\n", (int)r.state(), (long)ms);
    bos[4].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float* out = (const float*)bos[4].map();
    fprintf(stderr, "arg4[0..7] = %f %f %f %f %f %f %f %f\n", out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    return 0;
}
