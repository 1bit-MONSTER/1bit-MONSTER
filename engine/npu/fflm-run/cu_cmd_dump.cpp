#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
int main(int argc, char** argv) {
    xrt::device dev(0);
    xrt::xclbin x{std::string(argv[1])};
    dev.register_xclbin(x);
    xrt::hw_context hw(dev, x.get_uuid());
    xrt::kernel kk(hw, "MLIR_AIE");
    FILE* f = fopen(argv[2], "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> ins(sz/4); fread(ins.data(), 4, ins.size(), f); fclose(f);
    auto bI = xrt::bo(dev, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, kk.group_id(1));
    memcpy(bI.map(), ins.data(), ins.size()*4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<xrt::bo> bos;
    for (int a = 0; a < 5; a++) {
        bos.emplace_back(dev, 4096, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(3+a));
        memset(bos[a].map(), 0x01, 4096); bos[a].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    xrt::run r(kk);
    r.set_arg(0, (uint64_t)3);
    r.set_arg(1, bI);
    r.set_arg(2, (uint32_t)ins[2]);
    for (int a = 0; a < 5; a++) r.set_arg(3+a, bos[a]);
    auto* pkt = r.get_ert_packet();
    printf("opcode=%u count=%u\n", pkt->opcode, pkt->count);
    uint32_t* d = (uint32_t*)pkt->data;
    for (int i = 0; i < pkt->count && i < 20; i++) printf("  d[%d]=%08x\n", i, d[i]);
    return 0;
}
