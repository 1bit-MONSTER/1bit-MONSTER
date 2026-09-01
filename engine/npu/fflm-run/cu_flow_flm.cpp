// FLM sequence via the CASCADE-style CU flow: xrt::kernel from xclbin,
// args (3, bI, ninstr, bo0..bo4) — the flow that provably works.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
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
    unsigned ninstr = ins[2];
    fprintf(stderr, "seq: %zu words, ninstr=%u header=%08x %08x %08x %08x\n", ins.size(), ninstr, ins[0],ins[1],ins[2],ins[3]);
    auto bI = xrt::bo(dev, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, kk.group_id(1));
    memcpy(bI.map(), ins.data(), ins.size()*4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<size_t> sizes = {4096, 7745536, 4096, 4096, 200704};
    std::vector<xrt::bo> bos;
    for (int a = 0; a < 5; a++) {
        bos.emplace_back(dev, sizes[a], XRT_BO_FLAGS_HOST_ONLY, kk.group_id(3+a));
        memset(bos[a].map(), (a == 4) ? 0x3f : 0x01, sizes[a]);
        bos[a].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto t0 = std::chrono::steady_clock::now();
    auto r = kk((unsigned)3, bI, ninstr, bos[0], bos[1], bos[2], bos[3], bos[4]);
    r.wait();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();
    fprintf(stderr, "CU-flow FLM: state=%d elapsed=%ldms\n", (int)r.state(), (long)ms);
    bos[4].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float* out = (const float*)bos[4].map();
    fprintf(stderr, "arg4[0..7] = %f %f %f %f %f %f %f %f\n", out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    return 0;
}
