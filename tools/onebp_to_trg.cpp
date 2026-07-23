// tools/onebp_to_trg.cpp — Convert 1BP models to .trg format
// Bridge between 1BP (the project's native format) and TRG (TheRock GPU format)
//
// Build: g++ -std=c++17 -O3 -march=native -fopenmp \
//        -I engine/fusion -I include -I src \
//        tools/onebp_to_trg.cpp src/gguf_reader.cpp engine/fusion/cpu_layer.cpp \
//        -o build/onebp_to_trg -lm -lpthread
//
// Run: ./build/onebp_to_trg model.1bp output.trg

#include "onebp_format.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

// Tiled size calculation (same as onebp_format.h)
static uint64_t tiled_size(uint32_t rows, uint32_t cols, uint32_t tr, uint32_t tc, uint32_t gs) {
    uint32_t ntr = (rows + tr - 1) / tr;
    uint32_t ntc = (cols + tc - 1) / tc;
    uint32_t gpr = tc / gs;
    // Q4NX: scales(bf16×gpr×tr) + zero_points(bf16×gpr×tr) + packed(tr×tc/2)
    uint64_t tile = (uint64_t)tr * gpr * 2  // scales
                  + (uint64_t)tr * gpr * 2  // zero_points
                  + (uint64_t)tr * tc / 2;  // 4-bit packed
    return ntr * ntc * tile;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.1bp output.trg\n", argv[0]);
        return 1;
    }

    // Memory-map the 1BP file
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    size_t fsz = lseek(fd, 0, SEEK_END);
    auto data = (const uint8_t*)mmap(0, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }

    // Parse 1BP header
    OnebpHeader hdr;
    memcpy(&hdr, data, sizeof(hdr));
    if (!hdr.valid()) {
        fprintf(stderr, "Invalid 1BP header (magic=0x%08x)\n", hdr.magic);
        munmap((void*)data, fsz);
        return 1;
    }

    int H = hdr.hidden_size;
    int L = hdr.num_layers;
    int NH = hdr.num_attention_heads;
    int NKV = hdr.num_kv_heads ? hdr.num_kv_heads : NH;
    int HD = hdr.head_dim ? hdr.head_dim : (NH ? H / NH : 64);
    int V = hdr.vocab_size;
    int IM = hdr.intermediate_size;

    printf("=== 1BP → TRG: %s\n", argv[1]);
    printf("  H=%d L=%d NH=%d NKV=%d HD=%d V=%d IM=%d\n", H, L, NH, NKV, HD, V, IM);
    printf("  arch=%u quant=%u tensors=%u\n", hdr.arch, hdr.quant, hdr.tensor_count);

    // Walk the tensor index to find weight offsets
    // Tensor index: [name_len:u32][name:str][ndim:u32][shape:u32[]][offset:u64][bytes:u64]
    uint64_t idx_off = sizeof(OnebpHeader);
    uint64_t weight_data_off = fsz; // will be set to first weight data offset

    struct TensorInfo {
        std::string name;
        uint64_t offset;
        uint64_t bytes;
        int ndim;
        uint32_t shape[3];
    };
    std::vector<TensorInfo> tensors;
    
    for (uint32_t i = 0; i < hdr.tensor_count; i++) {
        TensorInfo ti;
        uint32_t nlen;
        memcpy(&nlen, data + idx_off, 4); idx_off += 4;
        ti.name.assign((const char*)(data + idx_off), nlen); idx_off += nlen + 1; // skip null terminator
        memcpy(&ti.ndim, data + idx_off, 4); idx_off += 4;
        for (int d = 0; d < ti.ndim; d++) {
            memcpy(&ti.shape[d], data + idx_off, 4); idx_off += 4;
        }
        memcpy(&ti.offset, data + idx_off, 8); idx_off += 8;
        memcpy(&ti.bytes, data + idx_off, 8); idx_off += 8;
        
        if (weight_data_off == fsz || ti.offset < weight_data_off)
            weight_data_off = ti.offset;
        
        tensors.push_back(ti);
    }

    printf("  Tensor index: %zu entries, weight data at offset %lu\n", 
           tensors.size(), weight_data_off);
    printf("  1BP header: %lu bytes, index: %lu bytes\n", 
           (unsigned long)sizeof(OnebpHeader), (unsigned long)(weight_data_off - sizeof(OnebpHeader)));

    // For now, report that the Q4NX tiled data is ready for TRG packing
    // The actual ternary packing would need the Q4NX dequant→requant cycle
    printf("\n  Q4NX data is at offset %lu, size %lu bytes\n", 
           (unsigned long)weight_data_off, (unsigned long)(fsz - weight_data_off));
    printf("  Use trg_save with a reconstructed Q4NX file for the full pipeline.\n");

    munmap((void*)data, fsz);
    return 0;
}
