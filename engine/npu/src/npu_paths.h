// npu_paths.h — Centralized NPU path resolution with env var overrides.
//
// All NPU engine/tool sources should use these functions instead of
// hardcoding /home/bcloud paths. Each function checks an env var first,
// then falls back to a $HOME-based default.
//
// Usage:
//   #include "npu_paths.h"
//   const char* model = npu_model_path();  // $NPU_MODEL_PATH or ~/.../model.q4nx
//
// To override without editing code:
//   export NPU_MODEL_PATH=/my/models/Qwen3-0.6B/model.q4nx
//   export NPU_XCLBIN_DIR=/my/xclbins
//   export NPU_INSTS_DIR=/my/insts

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

inline const char* env_or(const char* name, const char* fallback) {
    const char* v = getenv(name);
    return v ? v : fallback;
}

// A directory we can actually read. Used to reject an override that names a
// path this machine does not have: a stale NPU_XCLBIN_DIR in the shell
// environment (e.g. a clone that has since been deleted) otherwise redirects
// every NPU run to nothing, and the only symptom is a downstream "fopen
// failed" far from the cause.
inline bool npu_dir_exists(const char* p) {
    if (!p || !p[0]) return false;
    struct stat st;
    return ::stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

// Say it once per process: a bad override is worth exactly one line, not one
// per file that resolves a path.
inline void npu_warn_ignored(const char* var, const char* val, const std::string& fallback) {
    static bool warned_xclbin = false, warned_insts = false;
    bool& warned = (std::string(var) == "NPU_INSTS_DIR") ? warned_insts : warned_xclbin;
    if (warned) return;
    warned = true;
    std::fprintf(stderr, "NPU: %s='%s' is not a directory on this machine — ignoring it "
                         "and using '%s' instead (unset or fix the variable)\n",
                 var, val, fallback.c_str());
}

// ── Model paths ────────────────────────────────────────────────
// $NPU_MODEL_PATH — full path to model.q4nx
// $NPU_MODEL_DIR  — directory containing model files (used with model name)
// $NPU_TOKENIZER_PATH — full path to tokenizer.json

inline const char* npu_model_path() {
    return env_or("NPU_MODEL_PATH", nullptr);  // no default — must be set or passed via CLI
}

inline std::string npu_model_dir() {
    const char* e = getenv("NPU_MODEL_DIR");
    if (e) return e;
    const char* home = getenv("HOME");
    // fixes #1347: guard against HOME set-but-empty ("" + path would resolve
    // off the filesystem root instead of the intended per-user dir)
    return std::string(home && home[0] ? home : ".") + "/.config/flm/models";
}

inline const char* npu_tokenizer_path() {
    return env_or("NPU_TOKENIZER_PATH", nullptr);
}

// ── XCLBIN paths ───────────────────────────────────────────────
// $NPU_XCLBIN_DIR — directory containing compiled .xclbin files
//
// Order: the override if this machine has it, then the repo layout, then the
// installed layout. Measured 2026-09-14: ~/.bashrc exported NPU_XCLBIN_DIR at
// /home/bcloud/1bit-MONSTER-pi/engine/npu/xclbins — a deleted tree — and every
// NPU run on this box inherited it, so the Zaya worker could not find its insts
// files and the engine face served nothing while the logs named a path that
// looked deliberate. An override that does not exist is not an override.

inline std::string npu_xclbin_dir() {
    const char* e = getenv("NPU_XCLBIN_DIR");
    if (e && e[0]) {
        if (npu_dir_exists(e)) return e;
        npu_warn_ignored("NPU_XCLBIN_DIR", e, "engine/npu/xclbins");
    }
    // Repo/dev layout — what the engine's own call sites fell back to before
    // they read the variable themselves (kept first so running from a checkout
    // keeps working).
    if (npu_dir_exists("engine/npu/xclbins")) return "engine/npu/xclbins";
    // Installed layout (fixes #1338: an install path rather than a relative
    // ./ path; fixes #1347: HOME set-but-empty must not resolve off /).
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.local/share/1bit-monster/xclbins";
    return "/usr/local/share/1bit-monster/xclbins";
}

// ── Instruction file paths ─────────────────────────────────────
// $NPU_INSTS_DIR — directory containing compiled .insts / .txt instruction files

inline std::string npu_insts_dir() {
    const char* e = getenv("NPU_INSTS_DIR");
    if (e && e[0]) {
        if (npu_dir_exists(e)) return e;
        npu_warn_ignored("NPU_INSTS_DIR", e, npu_xclbin_dir());
    }
    return npu_xclbin_dir();  // default: same as xclbin dir
}

// ── Engine binary path ─────────────────────────────────────────
// $NPU_ENGINE_BIN — path to the NPU engine executable (for subprocess spawn)

inline std::string npu_engine_bin() {
    const char* e = getenv("NPU_ENGINE_BIN");
    if (e) return e;
    // fixes #1338: resolve against install path, not cwd
    // fixes #1347: guard against HOME set-but-empty
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.local/bin/npu_engine_universal";
    return "/usr/local/bin/npu_engine_universal";
}

// ── AIE toolchain paths (torch2aie) ────────────────────────────
// $AIE_TOOLS_DIR — root of the torch2aie installation

inline std::string aie_tools_dir() {
    const char* e = getenv("AIE_TOOLS_DIR");
    if (e) return e;
    const char* home = getenv("HOME");
    return std::string(home ? home : ".") + "/torch2aie";
}
