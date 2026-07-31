// dequant_oracle_ctrl.cpp — controlled-input variant of the dequant oracle:
// feed a synthetic window (scales=1.0, q8 ramp) so the output mapping is readable.
// Build: g++ -std=c++20 -O2 -o /tmp/dequant_ctrl dequant_oracle_ctrl.cpp \
//   -I third_party/FastFlowLM/src/include \
//   -L /home/bcloud/flm-0.9.46/opt/fastflowlm/lib -ldequant \
//   -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "npu_utils/npu_instr_utils.hpp"
#include "modules/dequant.hpp"
#include "lm_config.hpp"

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main(int argc, char** argv) {
    const char* xclbin_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/dequant.xclbin";
    const char* synth_path = argc > 2 ? argv[2] : "/tmp/synth_win0.bin";

    // load synthetic input (repeated over a big BO so windows 0..N see data)
    std::vector<uint8_t> synth;
    {
        FILE* f = fopen(synth_path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", synth_path); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        synth.resize(sz);
        fread(synth.data(), 1, sz, f);
        fclose(f);
    }
    fprintf(stderr, "synth input: %zu B\n", synth.size());

    LM_Config cfg;
    cfg.exec_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    Dequant deq(cfg);
    npu_sequence seq(device_npu2);
    deq.generate_dequant_q80_packed_in_q4nx_seq(&seq, 2048, 8192, 0, 2);
    auto [dp, dsz_words] = seq.dump();
    size_t dsz = dsz_words * sizeof(uint32_t);
    fprintf(stderr, "sequence: %zu words (%zu B)\n", dsz_words, dsz);
    if (dsz == 0) return 1;
    {
        FILE* sf = fopen("/tmp/deq_seq.bin", "wb");
        fwrite(dp, 1, dsz, sf);
        fclose(sf);
    }

    try {
        xrt::device dev(0);
        xrt::xclbin xc{std::string(xclbin_path)};
        dev.register_xclbin(xc);
        xrt::hw_context hc(dev, xc.get_uuid());
        xrt::kernel k(hc, "MLIR_AIE");

        int grp_ins = k.group_id(1);
        int grp_0 = k.group_id(3);
        int grp_1 = k.group_id(4);
        int grp_2 = k.group_id(5);
        int grp_3 = k.group_id(6);
        int grp_4 = k.group_id(7);

        // BO sizes: input read span = 20.7MB; output write span = 33.2MB
        size_t in_bytes = 21 * 1024 * 1024;
        size_t out_bytes = 34 * 1024 * 1024;

        xrt::bo bIns(dev, dsz, XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(bIns.map(), dp, dsz);
        bIns.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::bo b0(dev, out_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_0);
        xrt::bo b1(dev, in_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_1);
        xrt::bo b2(dev, out_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_2);
        xrt::bo b3(dev, out_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_3);
        xrt::bo b4(dev, out_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_4);

        memset(b0.map(), 0, out_bytes);
        b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // fill bo1 with the synthetic window repeated
        uint8_t* m1 = (uint8_t*)b1.map();
        for (size_t off = 0; off < in_bytes; off += synth.size())
            memcpy(m1 + off, synth.data(), std::min(synth.size(), in_bytes - off));
        b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = k(3, bIns, (uint32_t)(dsz), b0, b1, b2, b3, b4);
        run.wait();

        b0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        b1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        {
            FILE* fo = fopen("/tmp/ctrl_bo0.bin", "wb");
            fwrite(b0.map(), 1, out_bytes, fo);
            fclose(fo);
            FILE* fi = fopen("/tmp/ctrl_bo1.bin", "wb");
            fwrite(b1.map(), 1, in_bytes, fi);
            fclose(fi);
        }
        fprintf(stderr, "saved /tmp/ctrl_bo0.bin /tmp/ctrl_bo1.bin\n");
    } catch (std::exception& ex) {
        fprintf(stderr, "XRT error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
