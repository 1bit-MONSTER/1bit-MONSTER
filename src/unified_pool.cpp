// unified_pool.cpp — Unified model memory pool.
// One control plane keeps every model resident (mmap) so the server can
// switch/serve any of them without touching disk twice. .1bp models get
// header-parsed metadata; every other format (gguf/q4nx/h1b/…) is pooled
// generically (path + size + resident mapping).
// NOTE: the device-heap single-BO carve (docs/plans/one-heap-pivot.md) is
// the NPU-side follow-up; this pool is the control-plane residency layer.

#include "unified_pool.h"
#include "../engine/npu/src/onebp_loader.cpp"  // NpuOnebpModel (header-only loader)
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

UnifiedModelPool::~UnifiedModelPool() {
    for (int i = (int)slots_.size() - 1; i >= 0; i--)
        unload(i);
}

int UnifiedModelPool::load(const std::string& path) {
    for (auto& s : slots_)
        if (s.path == path) { s.refcount++; return (int)(&s - &slots_[0]); }

    printf("[pool] Loading: %s\n", path.c_str());

    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "[pool] not a regular file: %s\n", path.c_str());
        return -1;
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    void* data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { fprintf(stderr, "[pool] mmap failed: %s\n", path.c_str()); return -1; }

    ModelSlot slot;
    slot.path = path;
    slot.name = path.substr(path.find_last_of('/') + 1);
    auto dot = slot.name.find_last_of('.');
    if (dot != std::string::npos) slot.name = slot.name.substr(0, dot);
    slot.mmap_data = data;
    slot.mmap_size = (size_t)st.st_size;
    slot.refcount = 1;
    slot.kind = "generic";

    // .1bp: parse header for real metadata (optional — never fatal).
    if (path.size() > 4 && path.substr(path.size() - 4) == ".1bp") {
        NpuOnebpModel model;
        if (model.open(path.c_str())) {
            auto& hdr = model.header();
            slot.H = hdr.hidden_size; slot.NC = hdr.num_layers;
            slot.NH = hdr.num_attention_heads; slot.NKV = hdr.num_kv_heads;
            slot.HD = hdr.head_dim; slot.IM = hdr.intermediate_size;
            slot.V = hdr.vocab_size; slot.quant = hdr.quant;
            slot.kind = "1bp";
        }
    }

    slots_.push_back(std::move(slot));
    auto& s = slots_.back();
    printf("[pool]   %s: %s %.0fMB%s\n", s.name.c_str(), s.kind.c_str(),
           (double)s.mmap_size / 1024 / 1024,
           s.kind == "1bp" ? "" : " (generic mmap)");
    return (int)slots_.size() - 1;
}

ModelSlot* UnifiedModelPool::get(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return nullptr;
    return &slots_[slot];
}

ModelSlot* UnifiedModelPool::find(const std::string& name) {
    for (auto& s : slots_)
        if (s.name == name) return &s;
    return nullptr;
}

bool UnifiedModelPool::has_path(const std::string& path) const {
    for (auto& s : slots_)
        if (s.path == path) return true;
    return false;
}

bool UnifiedModelPool::unload(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return false;
    auto& s = slots_[slot];
    if (--s.refcount > 0) return true;
    printf("[pool] Unloading: %s\n", s.name.c_str());
    s.gpu.reset();
    if (s.mmap_data) munmap(s.mmap_data, s.mmap_size);
    s.mmap_data = nullptr;
    return true;
}

int UnifiedModelPool::count() const { return (int)slots_.size(); }

void UnifiedModelPool::report() const {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║     Unified Model Pool                   ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  %d model(s) resident\n\n", (int)slots_.size());
    for (auto& s : slots_) {
        printf("  %-28s %-8s %6.0fMB  %s\n", s.name.c_str(), s.kind.c_str(),
               (double)s.mmap_size / 1024 / 1024, s.gpu ? "GPU" : "mmap");
    }
    printf("\n");
}
