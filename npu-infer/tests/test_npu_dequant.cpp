// test_npu_dequant.cpp — validate the decoded dequant TXN on silicon (#2006/#2015).
//
// The runtime's load-time weight path: reorder_cpy (host tile reorder) ->
// dequant.xclbin (Q4NX -> f32 weight BO, the captured B0). This test feeds
// the SAME inputs the runtime used (the captured reordered tiles
// npu-infer/captures/rc_dst_0_1310720.bin) through dequant.xclbin with the
// DECODED dequant TXN (tools/decode_txn.cpp, Dequant::generate_dequant_*
// output for q_proj block 0) and compares the result against the captured
// weight BO (bo_from_000_1048576.bin).
//
// If the decoded TXN + dequant.xclbin reproduce B0, the weight-BD decode is
// proven on silicon and the packer can emit the layout directly.
//
// Build:
//   g++ -O2 -std=c++17 tests/test_npu_dequant.cpp -o test_npu_dequant \
//     -I include -I /usr/include/drm -lxrt_coreutil -lxrt_core
// Run:
//   ./test_npu_dequant [insts.bin] [tiles_in.bin] [expected_out.bin]
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static std::vector<uint32_t> load_words(const char* path, const char* what) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (%s)\n", path, what); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4) { fprintf(stderr, "bad size %ld for %s\n", sz, path); exit(1); }
    std::vector<uint32_t> v(sz / 4);
    size_t br = fread(v.data(), 4, v.size(), f); fclose(f);
    if (br != v.size()) { fprintf(stderr, "short read %s\n", path); exit(1); }
    return v;
}

static std::vector<uint8_t> load_bytes(const char* path, const char* what) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (%s)\n", path, what); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz);
    size_t br = fread(v.data(), 1, sz, f); fclose(f);
    if (br != (size_t)sz) { fprintf(stderr, "short read %s\n", path); exit(1); }
    return v;
}

int main(int argc, char** argv) {
    const char* insts_path = (argc > 1) ? argv[1]
        : "/tmp/txn_decode/dequant_q80_q_proj_b0_m0.bin";
    const char* in_path = (argc > 2) ? argv[2]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/rc_dst_0_1310720.bin";
    const char* exp_path = (argc > 3) ? argv[3]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/bo_from_000_1048576.bin";
    uint64_t opcode = (argc > 4) ? strtoull(argv[4], nullptr, 0) : 2;
    const char* xclbin_path = "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/dequant.xclbin";

    auto instrs = load_words(insts_path, "insts");
    auto in_data = load_bytes(in_path, "input tiles");
    auto exp = load_bytes(exp_path, "expected output");
    fprintf(stderr, "opcode=%llu insts: %zu words, input: %zu B, expected: %zu B\n",
            (unsigned long long)opcode, instrs.size(), in_data.size(), exp.size());

    xrt::device dev(0);
    FILE* f = fopen(xclbin_path, "rb"); fseek(f, 0, SEEK_END);
    long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    xrt::kernel kern(dev, xclbin->get_uuid(), "MLIR_AIE");

    // dequant ABI: (opcode=2, insts, ninstr, arg0=out, arg1=in-tiles)
    xrt::bo bo_instr(dev, instrs.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    memcpy(bo_instr.map(), instrs.data(), instrs.size() * 4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo bo_out(dev, exp.size(), xrt::bo::flags::host_only, kern.group_id(3));
    xrt::bo bo_in(dev, in_data.size(), xrt::bo::flags::host_only, kern.group_id(4));
    memset(bo_out.map(), 0, exp.size());
    memcpy(bo_in.map(), in_data.data(), in_data.size());
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // arg binding: (arg0, arg1) = (group3, group4) normally; --swap makes
    // (arg0, arg1) = (group4, group3) to test the amdxdna arg ordering.
    bool swap = false;
    for (int a = 5; a < argc; a++) if (!strcmp(argv[a], "--swap")) swap = true;
    xrt::bo* arg0 = swap ? &bo_in : &bo_out;
    xrt::bo* arg1 = swap ? &bo_out : &bo_in;
    auto run = kern(opcode, bo_instr, (uint32_t)instrs.size(), *arg0, *arg1);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_in.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint8_t* got = (const uint8_t*)bo_out.map();
    const uint8_t* got_in = (const uint8_t*)bo_in.map();
    if (argc > 5) {
        char p1[512], p2[512];
        snprintf(p1, sizeof(p1), "%s.out.bin", argv[5]);
        snprintf(p2, sizeof(p2), "%s.in.bin", argv[5]);
        FILE* fo = fopen(p1, "wb");
        if (fo) { fwrite(got, 1, exp.size(), fo); fclose(fo); }
        fo = fopen(p2, "wb");
        if (fo) { fwrite(got_in, 1, in_data.size(), fo); fclose(fo); }
        fprintf(stderr, "saved %s and %s\n", p1, p2);
    }
    // structure of both BOs: nonzero spans by 64KB block
    printf("bo_out (1MB) nonzero by 64KB block:");
    for (int b = 0; b < 16; b++) {
        size_t nz = 0;
        for (size_t i = b*16384; i < (b+1)*16384 && i < exp.size()/4; i++)
            if (((const float*)got)[i] != 0.0f) nz++;
        printf(" %zu", nz);
    }
    printf("\n");
    // nonzero span of the output (did the kernel write anything?)
    size_t first_nz = SIZE_MAX, last_nz = 0, nz = 0;
    for (size_t i = 0; i < exp.size() / 4; i++)
        if (((const float*)got)[i] != 0.0f) { if (first_nz == SIZE_MAX) first_nz = i; last_nz = i; nz++; }
    if (nz) printf("OUTPUT: %zu nonzero floats, span [%zu, %zu] (bytes [%zu, %zu])\n",
                   nz, first_nz, last_nz, first_nz * 4, (last_nz + 1) * 4);
    else    printf("OUTPUT: ALL ZEROS (kernel did not write — opcode/insts/groups?)\n");

    // compare
    size_t n_diff = 0; double max_abs = 0; size_t max_at = 0;
    const float* gf = (const float*)got;
    const float* ef = (const float*)exp.data();
    size_t nf = exp.size() / 4;
    for (size_t i = 0; i < nf; i++) {
        double d = fabs((double)gf[i] - (double)ef[i]);
        if (d > max_abs) { max_abs = d; max_at = i; }
        if (memcmp(&gf[i], &ef[i], 4) != 0) n_diff++;
    }
    double sum = 0;
    for (size_t i = 0; i < nf; i++) sum += fabs((double)gf[i] - (double)ef[i]);
    printf("output floats: %zu\n", nf);
    printf("byte-different: %zu / %zu (%.2f%%)\n", n_diff, nf, 100.0*n_diff/nf);
    printf("mean abs diff: %.8f   max abs diff: %.8f @ %zu\n", sum/nf, max_abs, max_at);
    printf("expected[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.6f", ef[i]);
    printf("\ngot     [0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.6f", gf[i]);
    printf("\n");
    return 0;
}
