#include "onebp_loader.h"

#ifdef _WIN32
#include <windows.h>

OnebpModel::~OnebpModel() {
    if (data) { UnmapViewOfFile(data); data = nullptr; }
}

// Windows equivalent of the POSIX open+fstat+mmap below. The file and mapping
// handles can be closed immediately after MapViewOfFile succeeds — the view
// keeps its own reference (same pattern as Q4nxReader::open in q4nx_reader.cpp).
static uint8_t* map_file(const char* path, size_t& out_size) {
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "Cannot open %s\n", path); return nullptr; }
    LARGE_INTEGER li;
    if (!GetFileSizeEx(hFile, &li)) { CloseHandle(hFile); fprintf(stderr, "Cannot stat %s\n", path); return nullptr; }
    out_size = (size_t)li.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); fprintf(stderr, "mmap failed\n"); return nullptr; }
    uint8_t* mapped = (uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!mapped) fprintf(stderr, "mmap failed\n");
    return mapped;
}
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

OnebpModel::~OnebpModel() {
    if (data && data != MAP_FAILED) munmap(data, file_size);
    if (fd >= 0) close(fd);
}

static uint8_t* map_file(const char* path, size_t& out_size, int& out_fd) {
    out_fd = open(path, O_RDONLY);
    if (out_fd < 0) { fprintf(stderr, "Cannot open %s\n", path); return nullptr; }
    struct stat st;
    fstat(out_fd, &st);
    out_size = st.st_size;
    uint8_t* mapped = (uint8_t*)mmap(nullptr, out_size, PROT_READ, MAP_PRIVATE, out_fd, 0);
    if (mapped == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return nullptr; }
    return mapped;
}
#endif

bool OnebpModel::load(const char* path) {
#ifdef _WIN32
    data = map_file(path, file_size);
#else
    data = map_file(path, file_size, fd);
#endif
    if (!data) return false;

    // File must be at least as large as the header (issue #1157).
    if (file_size < sizeof(OnebpHeader)) {
        fprintf(stderr, "1BP: file too small (%llu bytes, need %zu)\n",
                (unsigned long long)file_size, sizeof(OnebpHeader));
        return false;
    }

    memcpy(&header, data, sizeof(OnebpHeader));
    if (!header.valid()) {
        fprintf(stderr, "Invalid 1BP header (magic=0x%08X)\n", header.magic);
        return false;
    }

    static constexpr uint32_t MAX_NAME_LEN = 1024;     // 1 KiB
    static constexpr uint32_t MAX_NDIM = 8;            // reasonable max dims
    static constexpr uint32_t MAX_TENSOR_COUNT = 16384; // prevents OOM loops

    if (header.tensor_count > MAX_TENSOR_COUNT) {
        fprintf(stderr, "1BP: tensor_count %u exceeds max %u\n", header.tensor_count, MAX_TENSOR_COUNT);
        return false;
    }

    uint64_t pos = sizeof(OnebpHeader);
    for (uint32_t i = 0; i < header.tensor_count; i++) {
        // Bounds check: each entry needs at least 4(name_len) + 1(min name) + 4(ndim) + 8(offset) + 8(bytes) = 25 bytes
        if (pos + 25 > file_size) {
            fprintf(stderr, "1BP: truncated tensor index at entry %u/%u\n", i, header.tensor_count);
            return false;
        }
        OnebpTensor t;
        uint32_t name_len;
        memcpy(&name_len, data + pos, 4); pos += 4;
        if (name_len > MAX_NAME_LEN || pos + name_len + 1 > file_size) {
            fprintf(stderr, "1BP: invalid name_len=%u at entry %u\n", name_len, i);
            return false;
        }
        t.name = std::string((char*)(data + pos), name_len);
        pos += name_len + 1;  // +1 for null terminator written by converter
        if (pos + 4 > file_size) { fprintf(stderr, "1BP: truncated at ndim\n"); return false; }
        memcpy(&t.ndim, data + pos, 4); pos += 4;
        if (t.ndim > MAX_NDIM) {
            fprintf(stderr, "1BP: ndim=%d exceeds max %u at '%s'\n", t.ndim, MAX_NDIM, t.name.c_str());
            return false;
        }
        if (pos + (uint64_t)t.ndim * 4 + 16 > file_size) {
            fprintf(stderr, "1BP: truncated at dims for '%s'\n", t.name.c_str());
            return false;
        }
        t.dims.resize(t.ndim);
        for (int d = 0; d < t.ndim; d++) {
            memcpy(&t.dims[d], data + pos, 4); pos += 4;
        }
        memcpy(&t.offset, data + pos, 8); pos += 8;
        memcpy(&t.bytes, data + pos, 8); pos += 8;
        // v2 (ONEBP_VERSION 2): per-entry quant field (mixed-quant files).
        // v1 readers must skip it or every entry after the first desyncs.
        if (header.version >= 2) {
            if (pos + 4 > file_size) {
                fprintf(stderr, "1BP: truncated at per-tensor quant at entry %u/%u\n", i, header.tensor_count);
                return false;
            }
            memcpy(&t.quant, data + pos, 4); pos += 4;
        }
        tensors.push_back(t);
    }

    // The data section starts right after the tensor index.
    // The index position (pos) now points to the start of data.
    data_section_offset = pos;

    // v4 dedup aliases: an entry with bytes==0 is an alias whose offset
    // field holds the INDEX of an earlier tensor it shares data with.
    // Resolve before validation so bounds checks see real sizes.
    for (size_t i = 0; i < tensors.size(); i++) {
        auto& t = tensors[i];
        if (t.bytes != 0) continue;
        size_t ali = (size_t)t.offset;
        if (ali >= i || ali >= tensors.size()) {
            fprintf(stderr, "1BP: bad alias index %zu at '%s'\n", ali, t.name.c_str());
            return false;
        }
        t.offset = tensors[ali].offset;
        t.bytes  = tensors[ali].bytes;
        t.quant  = tensors[ali].quant;
        t.dims   = tensors[ali].dims;
        t.ndim   = tensors[ali].ndim;
    }

    // Validate all tensor offsets against file size (issue #1145).
    for (auto& t : tensors) {
        uint64_t abs_off = data_section_offset + t.offset;
        if (abs_off + t.bytes > file_size || abs_off < data_section_offset) {
            fprintf(stderr, "1BP: tensor '%s' offset=%llu bytes=%llu exceeds file size %llu\n",
                    t.name.c_str(), (unsigned long long)t.offset, (unsigned long long)t.bytes,
                    (unsigned long long)file_size);
            return false;
        }
    }

    return true;
}

uint8_t* OnebpModel::tensor_data(const OnebpTensor& t) {
    // t.offset is relative to the start of the data section.
    // Guard against out-of-bounds access from a crafted 1BP file (issue #1145).
    uint64_t abs_off = data_section_offset + t.offset;
    if (abs_off + t.bytes > file_size || abs_off < data_section_offset) {
        fprintf(stderr, "1BP: OOB tensor '%s' offset=%llu bytes=%llu file=%llu\n",
                t.name.c_str(), (unsigned long long)t.offset, (unsigned long long)t.bytes,
                (unsigned long long)file_size);
        return nullptr;
    }
    return data + abs_off;
}
