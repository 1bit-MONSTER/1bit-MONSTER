#pragma once
#include "onebp_format.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

struct OnebpTensor {
    std::string name;
    int ndim;
    std::vector<uint32_t> dims;
    uint64_t offset;
    uint64_t bytes;
    uint32_t quant = 0;   // v2: per-tensor quant (0 = use header quant)
};

struct OnebpModel {
    OnebpHeader header;
    std::vector<OnebpTensor> tensors;
    uint8_t* data = nullptr;
    size_t file_size = 0;
    uint64_t data_section_offset = 0;  // byte offset from data start to weight data
    int fd = -1;

    ~OnebpModel();
    bool load(const char* path);
    uint8_t* tensor_data(const OnebpTensor& t);
    // __onebp_ext_* metadata entries are not weights; a folded pack's transform lives
    // there. Backends that cannot apply it must fail closed (never serve folded as plain).
    bool has_prism_transform() const {
        for (const auto& t : tensors) if (t.name == "__onebp_ext_prism_transform") return true;
        return false;
    }
};
