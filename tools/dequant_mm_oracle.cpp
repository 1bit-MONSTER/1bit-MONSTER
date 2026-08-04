// dequant_mm_oracle.cpp — run FLM's REAL MoE dequant path on the NPU and read
// back the output layout.
//
// The standalone dequant.xclbin + generate_dequant_q80_packed_in_q4nx_seq was
// proven to be a data-mover red herring (see docs/research/qwen36-npu-zaya-
// integration.md §"Session 2026-07-31"). The runtime actually calls
//   qwen3_6_moe_npu_sequence::gen_dequant_mm(seq, M, K, N, off, m, dtype)
// against dequant_mm.xclbin (libqwen3_6_moe_npu.so, MIT). The kernel's
// output IS the value<->position mapping for Q8_0 projections and INT4
// experts — the blocker for native MoE in npu_engine_universal.
//
// gen_dequant_mm never dereferences `this` (verified by disassembly), so we
// dlopen the .so, hand it a dummy this, and drive the sequence generator
// directly. Static init of the .so (mvm_tiles etc.) runs at dlopen.
//
// Build:
//   g++ -std=c++20 -O2 -o build/dequant_mm_oracle tools/dequant_mm_oracle.cpp \
//     -I third_party/FastFlowLM/src/include \
//     -L /opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl \
//     -Wl,-rpath,/opt/xilinx/xrt/lib
// Run:
//   LD_LIBRARY_PATH=/opt/fastflowlm/lib ./build/dequant_mm_oracle \
//     [model.q4nx] [tensor_key] [M] [K] [N] [off] [m] [dtype] [xclbin]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>

#include "npu_utils/npu_instr_utils.hpp"

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ── load raw tensor bytes from a Q4NX file at data_offsets[0] ──
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

// ── synthetic Q8_0 rows: [512 B bf16 scales][8192 B q8 ramp] ──
// mode 0: q8[i] = i & 0xFF (low byte); mode 1: q8[i] = (i >> 8) & 0xFF (high byte).
// Two runs with both modes resolve the exact input->output position mapping.
static void synth_q8_rows(std::vector<uint8_t>& out, int rows, int mode = 0) {
    out.resize((size_t)rows * 8704);
    for (int r = 0; r < rows; r++) {
        uint8_t* row = out.data() + (size_t)r * 8704;
        uint16_t* sc = (uint16_t*)row;              // bf16 scales, 256 per row
        for (int i = 0; i < 256; i++) sc[i] = 0x3F80; // 1.0f
        for (int i = 0; i < 8192; i++) row[512 + i] = mode == 1 ? (uint8_t)((i >> 8) & 0xFF) : (uint8_t)(i & 0xFF);
    }
}

// ── synthetic INT4 rows: [512 B bf16 scales][512 B mins][4096 B nibbles] ──
static void synth_int4_rows(std::vector<uint8_t>& out, int rows) {
    out.resize((size_t)rows * 5120);
    for (int r = 0; r < rows; r++) {
        uint8_t* row = out.data() + (size_t)r * 5120;
        uint16_t* sc = (uint16_t*)row;
        uint16_t* mn = (uint16_t*)(row + 512);
        for (int i = 0; i < 256; i++) { sc[i] = 0x3F80; mn[i] = 0; }
        for (int i = 0; i < 4096; i++) row[1024 + i] = (uint8_t)((i * 3) & 0xFF); // both nibbles ramp
    }
}

// ── decode DDR_PATCH commands (0x81, 12 words): arg_idx @ bd[8], offset @ bd[10] ──
static void decode_ddr_patches(const std::vector<uint32_t>& seq) {
    int n = 0;
    for (size_t i = 0; i + 11 < seq.size(); ) {
        if (seq[i] == 0x81) {
            uint32_t bd6 = seq[i + 6];
            int col = (bd6 >> 25) & 0x7F, row = (bd6 >> 20) & 0x1F, bd_id = ((bd6 - 4) >> 5) & 0x1F;
            printf("DDR_PATCH[%d]: BD(col=%d,row=%d,id=%d) arg_idx=%u off=%u B\n",
                   n++, col, row, bd_id, seq[i + 8], seq[i + 10]);
            i += 12;
        } else i++;
    }
    printf("total DDR_PATCH: %d\n", n);
}

int main(int argc, char** argv) {
    const char* q4nx_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    const char* key = argc > 2 ? argv[2] : "model.layers.0.self_attn.q_proj.weight";
    uint32_t M = argc > 3 ? (uint32_t)strtoul(argv[3], nullptr, 10) : 2048;
    uint32_t K = argc > 4 ? (uint32_t)strtoul(argv[4], nullptr, 10) : 2048;
    uint32_t N = argc > 5 ? (uint32_t)strtoul(argv[5], nullptr, 10) : 8192;
    int off    = argc > 6 ? atoi(argv[6]) : 0;
    uint32_t m = argc > 7 ? (uint32_t)strtoul(argv[7], nullptr, 10) : 0;
    int dtype  = argc > 8 ? atoi(argv[8]) : 1;   // 1 = Q8_0, 2 = INT4
    const char* xclbin_path = argc > 9 ? argv[9]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/dequant_mm.xclbin";

    std::vector<uint8_t> raw;
    bool synth = strncmp(key, "synth:", 6) == 0;
    if (synth) {
        int rows = (int)M;
        int mode = (dtype >> 8) & 1;   // high byte in the unused top bits of dtype
        if (dtype == 1 || dtype == 0x101) synth_q8_rows(raw, rows, dtype == 0x101 ? 1 : 0);
        else                             synth_int4_rows(raw, rows);
        printf("synthetic %s input: %zu B (%d rows)%s\n", dtype == 1 ? "Q8_0" : dtype == 0x101 ? "Q8_0-hi" : "INT4", raw.size(), rows, mode ? " (hi ramp)" : "");
    } else {
        if (!load_q4nx_raw(q4nx_path, key, raw)) {
            fprintf(stderr, "cannot load %s from %s\n", key, q4nx_path);
            return 1;
        }
        printf("tensor %s: %zu B raw\n", key, raw.size());
    }

    // ── dlopen FLM plugin (static init runs here) ──
    // libq4_npu_eXpress.so must load first: it defines SafeTensors, which
    // libqwen3_6_moe_npu.so imports (undefined symbol otherwise).
    void* hq = dlopen("/opt/fastflowlm/lib/libq4_npu_eXpress.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hq) { fprintf(stderr, "dlopen eXpress: %s\n", dlerror()); return 1; }
    void* h = dlopen("/opt/fastflowlm/lib/libqwen3_6_moe_npu.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    typedef void (*gen_fn)(void*, npu_sequence*, uint32_t, uint32_t, uint32_t, int, uint32_t, int);
    gen_fn gen = (gen_fn)dlsym(h, "_ZN24qwen3_6_moe_npu_sequence14gen_dequant_mmEP12npu_sequencejjjmi11flm_dtype_t");
    if (!gen) { fprintf(stderr, "dlsym gen_dequant_mm: %s\n", dlerror()); return 1; }

    // ── generate the sequence (dummy this: never dereferenced) ──
    alignas(64) uint8_t dummy_this[256] = {0};
    npu_sequence seq(device_npu2);
    printf("calling gen_dequant_mm(M=%u K=%u N=%u off=%d m=%u dtype=%d)...\n", M, K, N, off, m, dtype);
    fflush(stdout);
    gen(dummy_this, &seq, M, K, N, off, m, dtype);
    auto [dp, dsz_words] = seq.dump();
    size_t dsz = dsz_words * sizeof(uint32_t);
    printf("sequence: %zu words (%zu B)\n", dsz_words, dsz);
    if (dsz == 0) { fprintf(stderr, "empty sequence!\n"); return 1; }
    {
        FILE* sf = fopen("/tmp/dm_seq.bin", "wb");
        fwrite(dp, 1, dsz, sf);
        fclose(sf);
        std::vector<uint32_t> w(dp, dp + dsz_words);
        decode_ddr_patches(w);
    }

    // ── XRT: run dequant_mm.xclbin ──
    size_t bo_bytes = 40 * 1024 * 1024;
    try {
        xrt::device dev(0);
        xrt::xclbin xc{std::string(xclbin_path)};
        dev.register_xclbin(xc);
        xrt::hw_context hc(dev, xc.get_uuid());
        xrt::kernel k(hc, "MLIR_AIE");

        int grp_ins = k.group_id(1);
        int grp[5] = { k.group_id(3), k.group_id(4), k.group_id(5), k.group_id(6), k.group_id(7) };

        xrt::bo bIns(dev, dsz, XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(bIns.map(), dp, dsz);
        bIns.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::bo bo[5];
        for (int i = 0; i < 5; i++) {
            bo[i] = xrt::bo(dev, bo_bytes, XRT_BO_FLAGS_HOST_ONLY, grp[i]);
            uint8_t* p = (uint8_t*)bo[i].map();
            for (size_t o = 0; o < bo_bytes; o += raw.size())
                memcpy(p + o, raw.data(), std::min(raw.size(), bo_bytes - o));
            bo[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        printf("running kernel (opcode=3, ninstr=%zu)...\n", dsz);
        fflush(stdout);
        auto run = k(3, bIns, (uint32_t)dsz, bo[0], bo[1], bo[2], bo[3], bo[4]);
        run.wait();
        printf("kernel done\n");

        for (int i = 0; i < 5; i++) {
            bo[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            char fn[64]; snprintf(fn, sizeof(fn), "/tmp/dm_bo%d.bin", i);
            FILE* fo = fopen(fn, "wb");
            fwrite(bo[i].map(), 1, bo_bytes, fo);
            fclose(fo);
            // report which regions differ from the repeated input pattern
            const uint8_t* p = (const uint8_t*)bo[i].map();
            size_t first_diff = bo_bytes, last_diff = 0, ndiff = 0;
            for (size_t o = 0; o + raw.size() <= bo_bytes; o += raw.size()) {
                for (size_t j = 0; j < raw.size(); j++) {
                    if (p[o + j] != raw[j]) {
                        if (first_diff == bo_bytes) first_diff = o + j;
                        last_diff = o + j;
                        ndiff++;
                    }
                }
            }
            printf("bo%d: %zu diff bytes vs input, span [%zu, %zu)\n", i, ndiff, first_diff, last_diff + 1);
            printf("  first 32 B: ");
            for (int j = 0; j < 32; j++) printf("%02x ", p[j]);
            printf("\n");
        }
    } catch (std::exception& ex) {
        fprintf(stderr, "XRT error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
