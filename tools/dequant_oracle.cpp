// dequant_oracle.cpp — run FLM's dequant.xclbin kernel on raw Q4NX rows and
// read back the dequantized values. The kernel's output reveals the exact
// value<->position mapping (the "torch2aie chunk format").
//
// Build:
//   g++ -std=c++20 -O2 -o dequant_oracle dequant_oracle.cpp \
//     -I third_party/FastFlowLM/src/include \
//     -L /home/bcloud/flm-0.9.46/opt/fastflowlm/lib -ldequant \
//     -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl
//   LD_LIBRARY_PATH=/home/bcloud/flm-0.9.46/opt/fastflowlm/lib ./dequant_oracle
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
#include <cmath>
#include <algorithm>

#include "npu_utils/npu_instr_utils.hpp"
#include "modules/dequant.hpp"
#include "lm_config.hpp"

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ── load raw bytes from a Q4NX file at data_offsets[0] ──
static bool load_q4nx_raw(const char* path, const char* key, std::vector<uint8_t>& out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = hsz;
    std::string k(key);
    size_t p = 0;
    const char* found = nullptr;
    while (p < jl) {
        const char* q = (const char*)memmem(js + p, jl - p, k.data(), k.size());
        if (!q) break;
        if ((q == js || *(q-1) == '"') && *(q + k.size()) == '"') { found = q; break; }
        p = (q - js) + k.size();
    }
    if (!found) { munmap(md, st.st_size); return false; }
    const char* offp = strstr(found, "\"data_offsets\"");
    long off = strtol(strchr(offp, '[') + 1, nullptr, 10);
    // size: product of shape
    const char* sp = strstr(found, "\"shape\"");
    long total = 1;
    if (sp) {
        const char* br = strchr(sp, '[');
        const char* cur = br + 1;
        while (*cur && *cur != ']') {
            while (*cur == ' ' || *cur == ',' ) cur++;
            if (*cur == ']' || !*cur) break;
            total *= strtol(cur, (char**)&cur, 10);
        }
    }
    out.assign(md + 8 + hsz + off, md + 8 + hsz + off + total);
    munmap(md, st.st_size);
    return true;
}

int main(int argc, char** argv) {
    const char* q4nx_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    const char* key = argc > 2 ? argv[2] : "model.layer.0.linear_attn.qkv_proj.weight";
    const char* xclbin_path = argc > 3 ? argv[3]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/dequant.xclbin";

    std::vector<uint8_t> raw;
    if (!load_q4nx_raw(q4nx_path, key, raw)) {
        fprintf(stderr, "cannot load %s from %s\n", key, q4nx_path);
        return 1;
    }
    fprintf(stderr, "tensor %s: %zu bytes raw\n", key, raw.size());

    // ── construct the FLM Dequant (constructor lives in libdequant.so) ──
    LM_Config cfg;  // default-constructed (std::string/nlohmann members)
    cfg.exec_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    fprintf(stderr, "constructing Dequant...\n"); fflush(stderr);
    Dequant deq(cfg);
    fprintf(stderr, "Dequant constructed\n"); fflush(stderr);

    // ── generate the Q8_0 dequant sequence for one 32x256 tile ──
    fprintf(stderr, "constructing npu_sequence...\n"); fflush(stderr);
    npu_sequence seq(device_npu2);
    fprintf(stderr, "npu_sequence ok\n"); fflush(stderr);
    // D_in = 256 (cols), D_out = 32 (rows), weight_offset = 0, mode = 0
    deq.generate_dequant_q80_packed_in_q4nx_seq(&seq, 2048, 8192, 0, 0);
    fprintf(stderr, "sequence generated\n"); fflush(stderr);
    auto [dp, dsz] = seq.dump();
    fprintf(stderr, "dequant sequence: %zu instructions (%zu B)\n", dsz / 4, dsz);
    if (dsz == 0) { fprintf(stderr, "empty sequence!\n"); return 1; }

    // ── XRT: load dequant.xclbin ──
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

        // the full qkv tensor as input; output candidates in bo1..bo4
        size_t row_bytes = raw.size();
        size_t out_bytes = 8192 * 2048 * 4;  // FP32 dequantized

        xrt::bo bIns(dev, dsz, XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(bIns.map(), dp, dsz);
        bIns.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::bo b0(dev, row_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_0);
        xrt::bo b1(dev, out_bytes * 2, XRT_BO_FLAGS_HOST_ONLY, grp_1);
        xrt::bo b2(dev, out_bytes * 2, XRT_BO_FLAGS_HOST_ONLY, grp_2);
        xrt::bo b3(dev, out_bytes * 2, XRT_BO_FLAGS_HOST_ONLY, grp_3);
        xrt::bo b4(dev, out_bytes * 2, XRT_BO_FLAGS_HOST_ONLY, grp_4);

        // fill bo0 with the first 8704-byte row of the tensor
        memcpy(b0.map(), raw.data(), row_bytes);
        b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // run: opcode 3 = run-with-instructions
        auto run = k(3, bIns, (uint32_t)(dsz), b0, b1, b2, b3, b4);
        run.wait();

        // read back and dump each BO's first values
        b1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        b2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        b3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        b4.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for (int i = 1; i <= 4; i++) {
            auto* bo = i == 1 ? &b1 : i == 2 ? &b2 : i == 3 ? &b3 : &b4;
            const float* f = (const float*)bo->map();
            const uint16_t* h = (const uint16_t*)bo->map();
            fprintf(stderr, "bo%d: f32[0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    i, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
            fprintf(stderr, "bo%d: bf16[0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    i,
                    ((uint32_t)h[0] << 16 ? 0 : 0), 0, 0, 0, 0, 0, 0, 0);
            // dump raw first 64 bytes hex
            const uint8_t* b = (const uint8_t*)bo->map();
            fprintf(stderr, "bo%d raw: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                    i, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                    b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
        }
        // save bo1..bo4 for offline correlation
        for (int i = 1; i <= 4; i++) {
            auto* bo = i == 1 ? &b1 : i == 2 ? &b2 : i == 3 ? &b3 : &b4;
            char fn[64]; snprintf(fn, sizeof(fn), "/tmp/deq_bo%d.bin", i);
            FILE* fo = fopen(fn, "wb");
            fwrite(bo->map(), 1, out_bytes * 2, fo);
            fclose(fo);
        }
        fprintf(stderr, "saved /tmp/deq_bo{1..4}.bin\n");
    } catch (std::exception& ex) {
        fprintf(stderr, "XRT error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
