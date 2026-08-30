// npu_embedded.h — C++26 #embed (P1967) resources for the NPU engine.
//
// Checked-in NPU artifacts are baked into the binary at compile time. When
// the runtime files are absent, the engine falls back to these embedded
// copies — "one binary, zero runtime files" for the NPU kernels.
//
// Fallback design: the on-disk files still win when present (env overrides
// like NPU_ATTN_XCLBIN / NPU_ATTN_INSTS point at custom builds); the embed
// only engages when the file is missing. If the compiler lacks #embed or the
// file was absent at build time, the macros are undefined and callers keep
// their existing file-based path untouched.
//
// Staleness: an embedded copy is a build-time snapshot. If an on-disk file
// was regenerated AFTER the engine was built, the two diverge — callers
// should use npu_embedded_stale() to warn about that at startup.
//
// Resource table:
//   attn.xclbin / attn_insts.txt       — NPU flash-attention kernel (AttnCtx)
//   final_cascade_fused.xclbin /
//   insts_cascade_fused.txt            — zero-DMA fused GU→SiLU→D cascade
//                                        (tests/fused_ab_probe.cpp)

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

// ── Flash-attention artifacts ───────────────────────────────────
#if defined(__has_embed)
#  if __has_embed("../xclbins/attn.xclbin")
#    define NPU_EMBED_ATTN_XCLBIN 1
inline constexpr unsigned char kAttnXclbin[] = {
#      embed "../xclbins/attn.xclbin"
};
#    define NPU_EMBED_ATTN_XCLBIN_SIZE (sizeof(kAttnXclbin))
#  endif
#  if __has_embed("../xclbins/attn_insts.txt")
#    define NPU_EMBED_ATTN_INSTS 1
inline constexpr unsigned char kAttnInsts[] = {
#      embed "../xclbins/attn_insts.txt"
};
#    define NPU_EMBED_ATTN_INSTS_SIZE (sizeof(kAttnInsts))
#  endif
#endif

// ── Zero-DMA fused cascade artifacts ────────────────────────────
#if defined(__has_embed)
#  if __has_embed("../xclbins/final_cascade_fused.xclbin")
#    define NPU_EMBED_CASCADE_XCLBIN 1
inline constexpr unsigned char kCascadeXclbin[] = {
#      embed "../xclbins/final_cascade_fused.xclbin"
};
#    define NPU_EMBED_CASCADE_XCLBIN_SIZE (sizeof(kCascadeXclbin))
#  endif
#  if __has_embed("../xclbins/insts_cascade_fused.txt")
#    define NPU_EMBED_CASCADE_INSTS 1
inline constexpr unsigned char kCascadeInsts[] = {
#      embed "../xclbins/insts_cascade_fused.txt"
};
#    define NPU_EMBED_CASCADE_INSTS_SIZE (sizeof(kCascadeInsts))
#  endif
#endif

// ── Staleness check ─────────────────────────────────────────────
// Returns true when BOTH an on-disk copy and an embedded copy exist and
// differ — i.e. the artifact was regenerated after this binary was built.
// Callers log a warning at startup so a stale embed is never silent.
inline bool npu_embedded_stale(const char* disk_path,
                               const unsigned char* embedded, std::size_t embedded_size) {
    FILE* f = std::fopen(disk_path, "rb");
    if (!f) return false;  // no on-disk copy → embed is authoritative, not stale
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz != (long)embedded_size) { std::fclose(f); return true; }
    std::vector<unsigned char> disk((std::size_t)sz);
    if (std::fread(disk.data(), 1, disk.size(), f) != disk.size()) {
        std::fclose(f);
        return true;
    }
    std::fclose(f);
    return std::memcmp(disk.data(), embedded, embedded_size) != 0;
}
